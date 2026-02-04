#!/usr/bin/env python3
"""
Compare captions from fine-tuned model vs base Qwen model using BERTScore/BERT-BLEU.
Supports both single image and batch evaluation modes.
bbscore.py
"""

import argparse
import json
import os
from typing import List, Dict
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader
from unsloth import FastVisionModel
import random
from bert_bleu import bert_bleu
import re

OBJECT = "ships"


def parse_bboxes(text):
    """Parse bounding boxes from model output.
    Expected format: <ref>label</ref><box>(x1,y1),(x2,y2)</box>
    Returns list of (label, x1, y1, x2, y2) tuples
    """
    bboxes = []
    # Pattern to match <ref>...</ref><box>(x1,y1),(x2,y2)</box>
    pattern = r'<ref>(.*?)</ref><box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
    matches = re.findall(pattern, text)
    
    for match in matches:
        label = match[0]
        x1, y1, x2, y2 = map(int, match[1:5])
        bboxes.append((label, x1, y1, x2, y2))
    
    return bboxes


def draw_bboxes(image_path, bboxes, output_path, color=(255, 0, 0), title=""):
    """Draw bounding boxes on image and save.
    bboxes: list of (label, x1, y1, x2, y2) tuples with coordinates in [0, 1000]
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    
    # Try to use a decent font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for label, x1, y1, x2, y2 in bboxes:
        # Convert from [0, 1000] to actual image coordinates
        x1_img = int(x1 * width / 1000)
        y1_img = int(y1 * height / 1000)
        x2_img = int(x2 * width / 1000)
        y2_img = int(y2 * height / 1000)
        
        # Draw rectangle
        draw.rectangle([x1_img, y1_img, x2_img, y2_img], outline=color, width=3)
        
        # Draw label background and text
        text_bbox = draw.textbbox((x1_img, y1_img - 20), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1_img, y1_img - 20), label, fill=(255, 255, 255), font=font)
    
    # Add title if provided
    if title:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24) if font != ImageFont.load_default() else font
        draw.text((10, 10), title, fill=color, font=title_font)
    
    image.save(output_path)
    return image


# ------------------------------
# Dataset for batched inference
# ------------------------------
class QwenBatchDataset(Dataset):
    def __init__(self, entries, image_dir):
        self.entries = entries
        self.image_dir = image_dir

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # img_name = entry["file_name"]                 # <-- NEW
        img_name = entry.get("image_id")
        if img_name.endswith(".json"):
            img_name = img_name.replace(".json", ".png")
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        question = entry["question"]                 # <-- NEW
        reference = entry["ground_truth"]            # <-- NEW

        return {
            "image": image,
            "question": question,
            "reference": reference,
            "path": img_path,
        }

                
def load_finetuned_model(checkpoint_path: str, device="cuda"):
    print(f"Loading finetuned model via Unsloth batching: {checkpoint_path}")

    base_model_name = "/home/spandan/scratch/interiit/qwen/models--unsloth--Qwen3-VL-8B-Instruct/snapshots/11d38e30f7b6dec7545b704d119dc6fbecbbd639"  # Match your training script
    
    print(f"   Loading base model: {base_model_name}")
    model, tokenizer = FastVisionModel.from_pretrained(
        base_model_name,
        load_in_4bit=False,  # Set to True if you trained with 4-bit
        device_map=device,
        use_safetensors=True
    )
    processor = tokenizer

    # Then load the LoRA adapter weights
    print(f"   Loading LoRA adapters from: {checkpoint_path}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, checkpoint_path)
    
    # Merge adapters for faster inference (optional but recommended)
    print("   Merging LoRA adapters into base model...")
    model = model.merge_and_unload()
    return model, processor


def load_base_model(device="cuda"):
    print("Loading base Qwen3-VL via Unsloth (batched)...")

    model, processor = FastVisionModel.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct",
        load_in_4bit=False,
        device_map=device
    )
    return model, processor

# ------------------------------
# --- Caption generation fns ---
# ------------------------------
def generate_caption_finetuned(model, processor, image_path: str, question: str = "Please describe this image in detail.") -> Dict:
    """Generate caption using fine-tuned model (matches training format)."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"caption": None, "error": str(e), "time": 0}

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]

    convo = [
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": f"""Describe the locations of the {OBJECT} using bounding box coordinates (x1,y1,x2,y2) normalized to 1000.

MISSION CRITICAL FORMATTING INSTRUCTIONS:
Your response MUST follow this exact format for each object:
<ref>label</ref><box>(x1,y1),(x2,y2)</box>

Where:
- label: descriptive name for the object
- x1,y1,x2,y2: integer coordinates in range [0,1000]
- x1,y1 is top-left corner
- x2,y2 is bottom-right corner

Example: <ref>yellow bus</ref><box>(120,340),(450,680)</box>

Output ONLY the bounding box annotations, one per line. No explanations."""}
        ]}
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    start = time.time()
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)

    generation_time = time.time() - start

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )

    caption = output_text[0] if output_text else ""

    return {"caption": caption, "error": None, "time": generation_time}


def generate_caption_base(model, processor, image_path: str, question: str = "Please describe this image in detail.") -> Dict:
    """Generate caption using base Qwen model (matches qwen_test.py format)."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"caption": None, "error": str(e), "time": 0}

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]

    convo = [
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": f"""Describe the locations of the {OBJECT} using bounding box coordinates (x1,y1,x2,y2) normalized to 1000.

MISSION CRITICAL FORMATTING INSTRUCTIONS:
Your response MUST follow this exact format for each object:
<ref>label</ref><box>(x1,y1),(x2,y2)</box>

Where:
- label: descriptive name for the object
- x1,y1,x2,y2: integer coordinates in range [0,1000]
- x1,y1 is top-left corner
- x2,y2 is bottom-right corner

Example: <ref>yellow bus</ref><box>(120,340),(450,680)</box>

Output ONLY the bounding box annotations, one per line. No explanations."""}
        ]}
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    start = time.time()
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)

    generation_time = time.time() - start

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )

    caption = output_text[0] if output_text else ""

    return {"caption": caption, "error": None, "time": generation_time}


# ------------------------------
# --- Scoring: BERTScore + BERT-BLEU ---
# ------------------------------
def compute_bertscore(candidates: List[str], references: List[str], device: str = None) -> Dict:
    """
    Compute (a) standard BERTScore (if bert-score available) AND
            (b) BERT-BLEU5 using the bert_bleu module.
    Returns a dict containing:
      - bert-precision, bert-recall, bert-f1  (if bert-score is installed, else None)
      - per_sample_precision/recall/f1 lists (if available)
      - bert-bleu (mean across samples)
      - per_sample_bert_bleu (list)
    """
    assert len(candidates) == len(references), "Candidates and references lengths must match"

    # 1) Try to compute standard BERTScore using the library (optional)
    bert_prec = None
    bert_rec = None
    bert_f1 = None
    per_prec = []
    per_rec = []
    per_f1 = []
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(candidates, references, lang='en', verbose=False, rescale_with_baseline=True)
        bert_prec = P.mean().item()
        bert_rec = R.mean().item()
        bert_f1 = F1.mean().item()
        per_prec = P.tolist()
        per_rec = R.tolist()
        per_f1 = F1.tolist()
    except Exception as e:
        # bert-score not installed or failed — keep going, we compute BERT-BLEU
        print(f"ℹ️ bert-score library not available or failed ({e}). Only computing BERT-BLEU.")

    # 2) Compute BERT-BLEU5 using the bert_bleu module
    per_sample_bert_bleu = []
    
    for cand, ref in zip(candidates, references):
        score = bert_bleu(cand, ref, N=1)
        per_sample_bert_bleu.append(score)

    # mean BERT-BLEU across samples
    mean_bert_bleu = float(sum(per_sample_bert_bleu) / len(per_sample_bert_bleu)) if per_sample_bert_bleu else 0.0

    metrics = {
        'bert-precision': bert_prec,
        'bert-recall': bert_rec,
        'bert-f1': bert_f1,
        'per_sample_precision': per_prec,
        'per_sample_recall': per_rec,
        'per_sample_f1': per_f1,
        'bert-bleu': mean_bert_bleu,
        'per_sample_bert_bleu': per_sample_bert_bleu,
    }

    return metrics


# ------------------------------
# --- Rest of evaluation fns ---
# ------------------------------
def evaluate_single_image(args):
    """Evaluate both models on a single image against a reference caption."""
    
    # Load both models (unchanged)
    finetuned_model, finetuned_processor = load_finetuned_model(args.checkpoint)
    base_model, base_processor = load_base_model()
    
    # Check image exists
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        return
    
    print(f"\n{'='*80}")
    print(f"Evaluating image: {args.image}")
    print(f"{'='*80}\n")
    
    # Generate from fine-tuned model
    print("🔹 Generating caption from fine-tuned model...")
    result1 = generate_caption_finetuned(finetuned_model, finetuned_processor, args.image, args.question)
    
    # Generate from base model
    print("🔸 Generating caption from base model...")
    result2 = generate_caption_base(base_model, base_processor, args.image, args.question)
    
    if not result1['caption'] or not result2['caption']:
        print("❌ Caption generation failed")
        return
    
    # Print generated captions
    print(f"\n{'='*80}")
    print("Generated Captions")
    print(f"{'='*80}\n")
    
    print("❓ Question:")
    print(f"   {args.question}\n")
    
    print("🔹 Fine-tuned Model (Candidate 1):")
    print(f"   {result1['caption']}")
    print(f"   ⏱️  Time: {result1['time']:.2f}s\n")
    
    print("🔸 Base Model (Candidate 2):")
    print(f"   {result2['caption']}")
    print(f"   ⏱️  Time: {result2['time']:.2f}s\n")
    
    print("📖 Reference Answer:")
    print(f"   {args.reference}\n")
    
    # Parse and visualize bounding boxes
    print(f"{'='*80}")
    print("Bounding Box Visualization")
    print(f"{'='*80}\n")
    
    # Parse bboxes from both outputs
    finetuned_bboxes = parse_bboxes(result1['caption'])
    base_bboxes = parse_bboxes(result2['caption'])
    
    print(f"Fine-tuned model detected {len(finetuned_bboxes)} objects")
    print(f"Base model detected {len(base_bboxes)} objects\n")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file) if args.output_file else "."
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate output filenames
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    finetuned_output = os.path.join(output_dir, f"{base_name}_finetuned.png")
    base_output = os.path.join(output_dir, f"{base_name}_base.png")
    
    # Draw and save visualizations
    print("Drawing bounding boxes...")
    draw_bboxes(args.image, finetuned_bboxes, finetuned_output, color=(0, 255, 0), title="Fine-tuned Model")
    draw_bboxes(args.image, base_bboxes, base_output, color=(255, 0, 0), title="Base Model")
    
    print(f"✅ Saved fine-tuned visualization to: {finetuned_output}")
    print(f"✅ Saved base model visualization to: {base_output}\n")
    
    # Compute BERTScore/BERT-BLEU against reference
    print(f"{'='*80}")
    print("Scoring Results (compared to reference)")
    print(f"{'='*80}\n")
    
    # Score for fine-tuned model
    metrics1 = compute_bertscore([result1['caption']], [args.reference])
    
    # Score for base model
    metrics2 = compute_bertscore([result2['caption']], [args.reference])
    
    if metrics1 and metrics2:
        # If bert-score is available, display it
        if metrics1['bert-precision'] is not None:
            print("🔹 Fine-tuned Model vs Reference (BERTScore):")
            print(f"   Precision: {metrics1['bert-precision']:.4f}")
            print(f"   Recall:    {metrics1['bert-recall']:.4f}")
            print(f"   F1:        {metrics1['bert-f1']:.4f}\n")
            
            print("🔸 Base Model vs Reference (BERTScore):")
            print(f"   Precision: {metrics2['bert-precision']:.4f}")
            print(f"   Recall:    {metrics2['bert-recall']:.4f}")
            print(f"   F1:        {metrics2['bert-f1']:.4f}\n")
        else:
            print("ℹ️ bert-score (library) not available; showing only BERT-BLEU below.\n")
        
        # Always display BERT-BLEU (the PDF metric)
        print("🔹 Fine-tuned Model vs Reference (BERT-BLEU4):")
        print(f"   BERT-BLEU (sample): {metrics1['per_sample_bert_bleu'][0]:.6f}")
        print(f"   BERT-BLEU (mean):   {metrics1['bert-bleu']:.6f}\n")
        
        print("🔸 Base Model vs Reference (BERT-BLEU4):")
        print(f"   BERT-BLEU (sample): {metrics2['per_sample_bert_bleu'][0]:.6f}")
        print(f"   BERT-BLEU (mean):   {metrics2['bert-bleu']:.6f}\n")
        
        # Compare the two using BERT-BLEU mean (or F1 if you prefer)
        diff_bleu = metrics1['bert-bleu'] - metrics2['bert-bleu']
        if diff_bleu > 0:
            print(f"📊 Fine-tuned model is better by {diff_bleu:.6f} BERT-BLEU points")
        elif diff_bleu < 0:
            print(f"📊 Base model is better by {abs(diff_bleu):.6f} BERT-BLEU points")
        else:
            print(f"📊 Both models have equal BERT-BLEU scores")
    
    # Save results if output file specified
    if args.output_file:
        results = {
            'image_path': args.image,
            'question': args.question,
            'reference_answer': args.reference,
            'finetuned_caption': result1['caption'],
            'base_caption': result2['caption'],
            'finetuned_bboxes': finetuned_bboxes,
            'base_bboxes': base_bboxes,
            'finetuned_visualization': finetuned_output,
            'base_visualization': base_output,
            'finetuned_generation_time': result1['time'],
            'base_generation_time': result2['time'],
            'finetuned_scores': metrics1,
            'base_scores': metrics2,
        }
        
        print(f"\n💾 Saving results to: {args.output_file}")
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved!")
    
    print(f"\n{'='*80}\n")


def compare_models_batch(args):
    """Fast batched evaluation using Unsloth + DataLoader for QA JSON format."""

    base_model_name = "/home/spandan/scratch/interiit/qwen/models--unsloth--Qwen3-VL-8B-Instruct/snapshots/11d38e30f7b6dec7545b704d119dc6fbecbbd639"  # Match your training script
    
    print(f"   Loading finetuned model with 4-bit quantization: {base_model_name}")
    finetuned_model, finetuned_processor = FastVisionModel.from_pretrained(
        base_model_name,
        load_in_4bit=False,  # Enable 4-bit to reduce memory by ~75%
        device_map="auto",  # Use auto device mapping
        use_safetensors=True
    )

    # Then load the LoRA adapter weights
    print(f"   Loading LoRA adapters from: {args.checkpoint}")
    from peft import PeftModel
    finetuned_model = PeftModel.from_pretrained(finetuned_model, args.checkpoint)
    

    # Merge adapters for faster inference (optional but recommended)
    print("   Merging LoRA adapters into base model...")
    finetuned_model = finetuned_model.merge_and_unload()

    print(f"   Loading base model with 4-bit quantization")
    base_model, base_processor = FastVisionModel.from_pretrained(
        "/home/spandan/scratch/interiit/qwen/models--unsloth--Qwen3-VL-8B-Instruct/snapshots/11d38e30f7b6dec7545b704d119dc6fbecbbd639",
        device_map="auto",  # Use auto device mapping
        load_in_4bit=False,  # Enable 4-bit to reduce memory
    )

    # -------------------
    # Load dataset
    # -------------------
    print(f"\n📄 Loading test data: {args.test_json}")
    with open(args.test_json, "r") as f:
        test_data = json.load(f)

    if args.max_samples:
        test_data = test_data[:args.max_samples]

    print(f"✅ Total samples: {len(test_data)}")

    dataset = QwenBatchDataset(test_data, args.image_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: x,
    )

    # Storage
    cand_ft = []
    cand_base = []
    refs = []
    results = []

    # ---------------------------
    # Batched caption generation
    # ---------------------------
    print("\n🎨 Generating answers in batches...\n")

    for batch in tqdm(loader):
        
        if random.random() > 0.1:
            continue

        images   = [item["image"]     for item in batch]
        questions = [item["question"] for item in batch]
        references = [item["reference"] for item in batch]
        paths    = [item["path"]      for item in batch]

        # -----------------------------
        # Build prompts for batch
        # -----------------------------
        def build_prompt(img, q):
            return finetuned_processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": q},
                        ],
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

        prompts = [build_prompt(img, q) for img, q in zip(images, questions)]

        # -----------------------------
        # Encode batch
        # -----------------------------
        ft_inputs = finetuned_processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
        ).to(finetuned_model.device)  # Use model's actual device

        base_inputs = base_processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
        ).to(base_model.device)  # Use model's actual device

        # -----------------------------
        # Batched generate
        # -----------------------------
        with torch.no_grad():
            ft_out = finetuned_model.generate(
                **ft_inputs, max_new_tokens=250, do_sample=False)
            base_out = base_model.generate(
                **base_inputs, max_new_tokens=250, do_sample=False)

        ft_texts = finetuned_processor.batch_decode(ft_out, skip_special_tokens=True)
        base_texts = base_processor.batch_decode(base_out, skip_special_tokens=True)

        # -----------------------------
        # Save results
        # -----------------------------
        for ft_ans, base_ans, ref, path, q in zip(ft_texts, base_texts, references, paths, questions):
            cand_ft.append(ft_ans)
            cand_base.append(base_ans)
            refs.append(ref)

            results.append({
                "image_path": path,
                "question": q,
                "ground_truth": ref,
                "finetuned_answer": ft_ans,
                "base_answer": base_ans,
            })

    # ---------------------------
    # Compute metrics
    # ---------------------------
    print("\n📊 Computing metrics (BERTScore / BERT-BLEU)...\n")

    metrics_ft = compute_bertscore(cand_ft, refs)
    metrics_base = compute_bertscore(cand_base, refs)

    print("\n🔹 Fine-tuned BERT-BLEU:", metrics_ft["bert-bleu"])
    print("🔸 Base       BERT-BLEU:", metrics_base["bert-bleu"])

    results.append({
        "metrics_finetuned": metrics_ft,
        "metrics_base": metrics_base,
    })

    # ---------------------------
    # Save output JSON
    # ---------------------------
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Saved results to {args.output_file}\n")

    print("\n🎉 Done!\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare fine-tuned vs base Qwen model using BERTScore/BERT-BLEU'
    )
    
    # Model arguments
    parser.add_argument(
        '--checkpoint', 
        required=True, 
        help='Path to fine-tuned model checkpoint'
    )
    
    # Input mode selection
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--image',
        help='Path to single image for evaluation'
    )
    input_group.add_argument(
        '--test-json', 
        help='Path to test JSON file for batch evaluation'
    )
    
    parser.add_argument("--batch-size", type=int, default=16)

    # Single image mode arguments
    parser.add_argument(
        '--reference',
        help='Reference caption for single image mode (required with --image)'
    )
    parser.add_argument(
        '--question',
        default='Please describe this image in detail.',
        help='Question to ask about the image (default: "Please describe this image in detail.")'
    )
    
    # Batch mode arguments
    parser.add_argument(
        '--image-dir', 
        default='../EarthMind-Bench/img/test/sar/img',
        help='Directory containing test images (for batch mode)'
    )
    parser.add_argument(
        '--max-samples', 
        type=int, 
        default=None,
        help='Maximum number of samples to process in batch mode'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-file', 
        help='Path to save comparison results JSON'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.image and not args.reference:
        parser.error("--reference is required when using --image")
    
    # Run evaluation
    if args.image:
        evaluate_single_image(args)
    else:
        compare_models_batch(args)


if __name__ == '__main__':
    main()

# To run on specific GPUs (e.g., GPUs 4,5,6,7), use:
# CUDA_VISIBLE_DEVICES=4,5,6,7 python bbscore_all.py --test-json /home/spandan/scratch/interiit/data/VRSBench/VRSBench_EVAL_vqa.json --checkpoint /home/spandan/scratch/interiit/qwen/qwen/checkpoints_caption/final --image-dir /home/spandan/scratch/interiit/data/VRSBench/Images_val

# https://chatgpt.com/gg/v/692c37dc7e2881a3a41a7043c65f3206?token=nH4eDKxWXUlJ5TypASHxOw