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
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader
from unsloth import FastVisionModel
import random
from bert_bleu import bert_bleu


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

    base_model_name = "/home/samyak/scratch/interiit/qwen/models--unsloth--Qwen3-VL-8B-Instruct/snapshots/11d38e30f7b6dec7545b704d119dc6fbecbbd639"  # Match your training script
    
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
# --- Existing model loaders ---
# ------------------------------
def load_finetuned_model(checkpoint_path: str, device: str = "cuda"):
    """Load fine-tuned model (candidate1)."""
    print(f"Loading fine-tuned model from: {checkpoint_path}")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    
    print(f"✅ Fine-tuned model loaded")
    return model, processor


def load_base_model(device: str = "cuda"):
    """Load base Qwen model (candidate2)."""
    print(f"Loading base Qwen3-VL-8B-Instruct model...")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct",
        dtype="cuda",
        device_map=device
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    
    print(f"✅ Base model loaded")
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
            "role": "system",
            "content": [
{"type": "text", "text": "You are an assistant that answers questions about satellite and aerial imagery. Provide concise, direct answers in one word or number. There are three types of possible questions. If it is a binary question, answer with 'Yes' or 'No'. If it is a numeric question, provide the numeric answer only. If it is a one word question, answer in one word only."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
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
            "role": "system",
            "content": [
{"type": "text", "text": "You are an assistant that answers questions about satellite and aerial imagery. Provide concise, direct answers in one word or number. There are three types of possible questions. If it is a binary question, answer with 'Yes' or 'No'. If it is a numeric question, provide the numeric answer only. If it is a one word question, answer in one word only."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
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
# --- Scoring: Binary EM, Normalized MSE, BERT-BLEU ---
# ------------------------------
def compute_metrics(candidates: List[str], references: List[str], question_types: List[str] = None) -> Dict:
    """
    Compute three metrics based on answer type:
    1. Binary Exact Match (for Yes/No type answers): {0, 1}
    2. Normalized MSE (for numeric answers): (0-1]
    3. BERT-BLEU (for semantic/text answers): [0-1]
    
    Returns aggregate scores and per-sample breakdowns.
    """
    assert len(candidates) == len(references), "Candidates and references lengths must match"
    
    # Initialize storage
    binary_scores = []
    numeric_scores = []
    semantic_scores = []
    
    per_sample_results = []
    
    for i, (cand, ref) in enumerate(zip(candidates, references)):
        cand_clean = cand.strip().lower()
        ref_clean = ref.strip().lower()
        
        result = {
            'candidate': cand,
            'reference': ref,
            'score': None,
            'metric_type': None
        }
        
        # Determine answer type and compute appropriate metric
        # 1. Binary (Yes/No questions)
        if ref_clean in ['yes', 'no', 'true', 'false']:
            # Exact match for binary
            score = 1.0 if cand_clean == ref_clean else 0.0
            binary_scores.append(score)
            result['score'] = score
            result['metric_type'] = 'binary_exact_match'
        
        # 2. Numeric (integers or floats)
        elif ref_clean.replace('.', '', 1).replace('-', '', 1).isdigit():
            try:
                cand_num = float(cand_clean.replace(',', ''))
                ref_num = float(ref_clean.replace(',', ''))
                
                # Normalized MSE: 1 / (1 + MSE)
                mse = (cand_num - ref_num) ** 2
                normalized_mse = 1.0 / (1.0 + mse)
                
                numeric_scores.append(normalized_mse)
                result['score'] = normalized_mse
                result['metric_type'] = 'normalized_mse'
            except (ValueError, ZeroDivisionError):
                # If parsing fails, treat as semantic
                score = bert_bleu(cand, ref, N=1)
                semantic_scores.append(score)
                result['score'] = score
                result['metric_type'] = 'bert_bleu'
        
        # 3. Semantic (text/description answers)
        else:
            score = bert_bleu(cand, ref, N=1)
            semantic_scores.append(score)
            result['score'] = score
            result['metric_type'] = 'bert_bleu'
        
        per_sample_results.append(result)
    
    # Compute aggregate scores
    total_samples = len(candidates)
    binary_count = len(binary_scores)
    numeric_count = len(numeric_scores)
    semantic_count = len(semantic_scores)
    
    # Weighted average based on distribution (10% binary, 20% numeric, 20% semantic per spec)
    # But we compute actual counts and average them
    binary_avg = sum(binary_scores) / len(binary_scores) if binary_scores else 0.0
    numeric_avg = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0
    semantic_avg = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0.0
    
    # Overall weighted score
    # Using equal weighting across actual samples (or you can apply 10/20/20 % weights)
    all_scores = binary_scores + numeric_scores + semantic_scores
    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    metrics = {
        'overall_score': overall_avg,
        'binary_exact_match': {
            'score': binary_avg,
            'count': binary_count,
            'percentage': 100 * binary_count / total_samples if total_samples > 0 else 0
        },
        'normalized_mse': {
            'score': numeric_avg,
            'count': numeric_count,
            'percentage': 100 * numeric_count / total_samples if total_samples > 0 else 0
        },
        'bert_bleu': {
            'score': semantic_avg,
            'count': semantic_count,
            'percentage': 100 * semantic_count / total_samples if total_samples > 0 else 0
        },
        'per_sample_results': per_sample_results
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
    
    # Compute metrics against reference
    print(f"{'='*80}")
    print("Scoring Results (compared to reference)")
    print(f"{'='*80}\n")
    
    # Score for fine-tuned model
    metrics1 = compute_metrics([result1['caption']], [args.reference])
    
    # Score for base model
    metrics2 = compute_metrics([result2['caption']], [args.reference])
    
    if metrics1 and metrics2:
        result1_info = metrics1['per_sample_results'][0]
        result2_info = metrics2['per_sample_results'][0]
        
        print(f"🔹 Fine-tuned Model:")
        print(f"   Metric: {result1_info['metric_type']}")
        print(f"   Score: {result1_info['score']:.4f}\n")
        
        print(f"🔸 Base Model:")
        print(f"   Metric: {result2_info['metric_type']}")
        print(f"   Score: {result2_info['score']:.4f}\n")
        
        # Compare
        diff = result1_info['score'] - result2_info['score']
        if diff > 0:
            print(f"📊 Fine-tuned model is better by {diff:.4f} points")
        elif diff < 0:
            print(f"📊 Base model is better by {abs(diff):.4f} points")
        else:
            print(f"📊 Both models have equal scores")
    
    # Save results if output file specified
    if args.output_file:
        results = {
            'image_path': args.image,
            'question': args.question,
            'reference_answer': args.reference,
            'finetuned_caption': result1['caption'],
            'base_caption': result2['caption'],
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

    base_model_name = "/home/samyak/scratch/interiit/qwen/models--unsloth--Qwen3-VL-8B-Instruct/snapshots/11d38e30f7b6dec7545b704d119dc6fbecbbd639"  # Match your training script
    
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
        "/home/samyak/scratch/interiit/qwen/models--unsloth--Qwen3-VL-8B-Instruct/snapshots/11d38e30f7b6dec7545b704d119dc6fbecbbd639",
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
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are an assistant that answers questions about satellite and aerial imagery. Provide concise, direct answers in one word or number. There are three types of possible questions. If it is a binary question, answer with 'Yes' or 'No'. If it is a numeric question, provide the numeric answer only. If it is a one word question, answer in one word only."}
                        ]
                    },
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
    print("\n📊 Computing metrics (Binary EM / Normalized MSE / BERT-BLEU)...\n")

    metrics_ft = compute_metrics(cand_ft, refs)
    metrics_base = compute_metrics(cand_base, refs)

    print("\n🔹 Fine-tuned Model Scores:")
    print(f"   Overall: {metrics_ft['overall_score']:.4f}")
    print(f"   Binary EM: {metrics_ft['binary_exact_match']['score']:.4f} ({metrics_ft['binary_exact_match']['count']} samples, {metrics_ft['binary_exact_match']['percentage']:.1f}%)")
    print(f"   Normalized MSE: {metrics_ft['normalized_mse']['score']:.4f} ({metrics_ft['normalized_mse']['count']} samples, {metrics_ft['normalized_mse']['percentage']:.1f}%)")
    print(f"   BERT-BLEU: {metrics_ft['bert_bleu']['score']:.4f} ({metrics_ft['bert_bleu']['count']} samples, {metrics_ft['bert_bleu']['percentage']:.1f}%)")
    
    print("\n🔸 Base Model Scores:")
    print(f"   Overall: {metrics_base['overall_score']:.4f}")
    print(f"   Binary EM: {metrics_base['binary_exact_match']['score']:.4f} ({metrics_base['binary_exact_match']['count']} samples, {metrics_base['binary_exact_match']['percentage']:.1f}%)")
    print(f"   Normalized MSE: {metrics_base['normalized_mse']['score']:.4f} ({metrics_base['normalized_mse']['count']} samples, {metrics_base['normalized_mse']['percentage']:.1f}%)")
    print(f"   BERT-BLEU: {metrics_base['bert_bleu']['score']:.4f} ({metrics_base['bert_bleu']['count']} samples, {metrics_base['bert_bleu']['percentage']:.1f}%)")

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
# CUDA_VISIBLE_DEVICES=4,5,6,7 python bbscore_all.py --test-json /home/samyak/scratch/interiit/data/VRSBench/VRSBench_EVAL_vqa.json --checkpoint /home/samyak/scratch/interiit/qwen/qwen/checkpoints_caption/final --image-dir /home/samyak/scratch/interiit/data/VRSBench/Images_val

# https://chatgpt.com/gg/v/692c37dc7e2881a3a41a7043c65f3206?token=nH4eDKxWXUlJ5TypASHxOw