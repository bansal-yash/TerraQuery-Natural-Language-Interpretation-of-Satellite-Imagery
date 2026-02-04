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
from concurrent.futures import ThreadPoolExecutor
import threading


OBJECT = "digits"

BASE_MODEL_DIR = "/home/spandan/scratch/interiit/qwen/small_spandan"




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

        img_path = entry.get('image')
        if not img_path:
            fname = entry.get("file_name")
            if fname.endswith(".json"):
                fname = fname.replace(".json", ".png")
            img_path = fname

        full_path = os.path.join(self.image_dir, img_path)
        image = Image.open(full_path).convert("RGB")

        reference = entry.get("caption") or entry.get("answer")
        if isinstance(reference, list):
            reference = reference[0]

        return {
            "image": image,
            "path": full_path,
            "reference": reference
        }


# -------------------------------------------------
#      NEW: Batched compare_models_batch()
# -------------------------------------------------
def compare_models_batch(args):
    """Fast batched evaluation using Unsloth + DataLoader."""

    print("🚀 Using Unsloth for batched inference!")

    # -------------------
    # Load both models on separate GPUs
    # -------------------
    print(f"Loading fine-tuned model on {DEVICE_FINETUNED}...")
    finetuned_model, finetuned_processor = FastVisionModel.from_pretrained(
        args.checkpoint,
        device_map=DEVICE_FINETUNED,
        load_in_4bit=False,
    )

    print(f"Loading base model on {DEVICE_BASE}...")
    base_model, base_processor = FastVisionModel.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct",
        device_map=DEVICE_BASE,
        load_in_4bit=False,
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
        batch_size=args.batch_size,   # ← YOU NEED TO ADD: parser.add_argument("--batch-size", type=int, default=4)
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: x,
    )

    # Storage
    candidate1_list = []
    candidate2_list = []
    reference_list = []
    results = []

    # ---------------------------
    # Batched caption generation
    # ---------------------------
    print("\n🎨 Generating captions in batches...\n")

    for batch in tqdm(loader):

        images = [item["image"] for item in batch]
        references = [item["reference"] for item in batch]
        paths = [item["path"] for item in batch]

        # -----------------------------
        # Build prompts for batch
        # -----------------------------
        def make_prompt(img):
            return finetuned_processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": "Please describe the image in detail."},
                        ],
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

        prompts = [make_prompt(img) for img in images]

        # -----------------------------
        # Batch encode inputs
        # -----------------------------
        ft_inputs = finetuned_processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        base_inputs = base_processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        # -----------------------------
        # Generate (batched)
        # -----------------------------
        with torch.no_grad():
            ft_out = finetuned_model.generate(
                **ft_inputs,
                max_new_tokens=200,
                do_sample=False
            )

            base_out = base_model.generate(
                **base_inputs,
                max_new_tokens=200,
                do_sample=False
            )

        ft_texts = finetuned_processor.batch_decode(ft_out, skip_special_tokens=True)
        base_texts = base_processor.batch_decode(base_out, skip_special_tokens=True)

        # -----------------------------
        # Save batch outputs
        # -----------------------------
        for ft_cap, base_cap, ref, path in zip(ft_texts, base_texts, references, paths):

            candidate1_list.append(ft_cap)
            candidate2_list.append(base_cap)
            reference_list.append(ref)

            results.append({
                "image_path": path,
                "candidate1_finetuned": ft_cap,
                "candidate2_base": base_cap,
                "reference": ref
            })

    # ---------------------------
    # Compute metrics
    # ---------------------------
    print("\n📊 Computing metrics (BERTScore / BERT-BLEU)...\n")

    metrics_ft = compute_bertscore(candidate1_list, reference_list)
    metrics_base = compute_bertscore(candidate2_list, reference_list)

    print("\n🔹 Fine-tuned BERT-BLEU:", metrics_ft["bert-bleu"])
    print("🔸 Base       BERT-BLEU:", metrics_base["bert-bleu"])

    # Add metrics to result JSON
    results.append({
        "metrics_finetuned": metrics_ft,
        "metrics_base": metrics_base
    })

    # ---------------------------
    # Save JSON output
    # ---------------------------
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Saved results to {args.output_file}\n")

    print("\n🎉 Done!\n")


                
def load_finetuned_model(checkpoint_path: str, device="cuda"):
    print(f"Loading finetuned model via Unsloth batching: {checkpoint_path}")

    base_model_name = "unsloth/Qwen3-VL-8B-Instruct"  # Match your training script
    
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
        dtype="auto",
        device_map=device
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    
    print(f"✅ Base model loaded")
    return model, processor


# ------------------------------
# --- Caption generation fns ---
# ------------------------------
def generate_caption_finetuned(model, processor, image_path: str) -> Dict:
    """Generate caption using fine-tuned model (matches training format)."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"caption": None, "error": str(e), "time": 0}

    system_instruction = (
        "You are an assistant that generates detailed, accurate descriptions "
        "of satellite and aerial imagery."
    )

    convo = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
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

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Please describe this image in detail."}
        ]},
    ]

    prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        images=[image],
        text=prompt,
        return_tensors="pt"
    ).to(model.device)

    start = time.time()
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            num_beams=1,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    generation_time = time.time() - start

    full_text = processor.decode(output_ids[0], skip_special_tokens=True)

    if "assistant" in full_text:
        caption = full_text.split("assistant", 1)[-1].strip()
    else:
        caption = full_text.strip()

    return {"caption": caption, "error": None, "time": generation_time}


def generate_caption_base(model, processor, image_path: str) -> Dict:
    """Generate caption using base Qwen model (matches qwen_test.py format)."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"caption": None, "error": str(e), "time": 0}

    convo = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
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

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Please describe this image in detail."},
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
# --- Scoring: BERTScore + BERT-BLEU ---
# ------------------------------
def _tokenize_simple(text: str) -> List[str]:
    """Simple whitespace tokenizer. Keep punctuation? Minimal normalization."""
    return text.strip().lower().split()


def _generate_ngrams(tokens: List[str], n: int) -> List[str]:
    """Generate n-grams (as strings) from a token list."""
    if n <= 0:
        return []
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _embed_texts_bert(texts: List[str], bert_tokenizer, bert_model, device: torch.device, batch_size: int = 32):
    """
    Return a list of embeddings (torch.Tensor) for the given texts.
    We use mean pooling over last_hidden_state tokens (excluding special tokens).
    """
    all_embs = []
    bert_model.to(device)
    bert_model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = bert_tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            out = bert_model(**enc, return_dict=True)
            last = out.last_hidden_state  # (B, T, D)
            # mean pool excluding padding tokens (use attention mask)
            attn = enc["attention_mask"].unsqueeze(-1)  # (B, T, 1)
            summed = (last * attn).sum(dim=1)  # (B, D)
            lengths = attn.sum(dim=1).clamp(min=1e-9)  # (B, 1)
            mean_pooled = summed / lengths
            for vec in mean_pooled:
                all_embs.append(vec.cpu())
    return all_embs  # list of tensors on CPU


def compute_bertscore(candidates: List[str], references: List[str], device: str = None) -> Dict:
    """
    Compute (a) standard BERTScore (if bert-score available) AND
            (b) BERT-BLEU4 as per your PDF (semantic n-gram precision + geometric mean).
    Returns a dict containing:
      - bert-precision, bert-recall, bert-f1  (if bert-score is installed, else None)
      - per_sample_precision/recall/f1 lists (if available)
      - bert-bleu (mean across samples)
      - per_sample_bert_bleu (list)
      - per_sample_pn (list of lists per sample: P1..P4)
    """
    assert len(candidates) == len(references), "Candidates and references lengths must match"

    # device selection for embedding model
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

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

    # 2) Compute BERT-BLEU4 per PDF
    from transformers import BertTokenizer, BertModel
    bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    N = 4
    eps = 1e-8

    per_sample_bert_bleu = []
    per_sample_pn = []  # list of lists [P1,P2,P3,P4]

    for cand, ref in zip(candidates, references):
        # tokenize simply (document uses token sets; whitespace is acceptable baseline)
        cand_toks = _tokenize_simple(cand)
        ref_toks = _tokenize_simple(ref)

        Pn_list = []
        for n in range(1, N + 1):
            cand_ngrams = _generate_ngrams(cand_toks, n)
            ref_ngrams = _generate_ngrams(ref_toks, n)
            if len(cand_ngrams) == 0:
                # If no candidate n-grams: precision = 0
                Pn_list.append(0.0)
                continue
            if len(ref_ngrams) == 0:
                # if no reference n-grams but candidate has, treat similarity as 0
                Pn_list.append(0.0)
                continue

            # Compute embeddings for reference ngrams and candidate ngrams (batch)
            # We embed ref ngrams and candidate ngrams separately
            ref_embs = _embed_texts_bert(ref_ngrams, bert_tok, bert_model, torch_device, batch_size=32)
            cand_embs = _embed_texts_bert(cand_ngrams, bert_tok, bert_model, torch_device, batch_size=32)

            # Convert lists to tensors on CPU
            ref_stack = torch.stack(ref_embs, dim=0)  # (R, D)
            cand_stack = torch.stack(cand_embs, dim=0)  # (C, D)

            # Normalize for cosine similarity
            ref_norm = ref_stack / (ref_stack.norm(dim=1, keepdim=True).clamp(min=1e-9))
            cand_norm = cand_stack / (cand_stack.norm(dim=1, keepdim=True).clamp(min=1e-9))

            # For each candidate n-gram, compute cosine similarities with all reference n-grams,
            # take maximum, then average across candidate n-grams.
            sims = torch.matmul(cand_norm, ref_norm.T)  # (C, R)
            # If any numerical instability, clamp
            sims = sims.clamp(-1.0, 1.0)
            max_sims, _ = sims.max(dim=1)  # (C,)
            Pn = float(max_sims.mean().item())
            Pn_list.append(Pn)

        # compute BERT-BLEU per sample
        log_sum = 0.0
        for p in Pn_list:
            log_sum += torch.log(torch.tensor(p + eps)).item()
        bert_bleu_score = float(torch.exp(torch.tensor((1.0 / N) * log_sum)).item())
        per_sample_pn.append(Pn_list)
        per_sample_bert_bleu.append(bert_bleu_score)

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
        'per_sample_pn': per_sample_pn,
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
    result1 = generate_caption_finetuned(finetuned_model, finetuned_processor, args.image)
    
    # Generate from base model
    print("🔸 Generating caption from base model...")
    result2 = generate_caption_base(base_model, base_processor, args.image)
    
    if not result1['caption'] or not result2['caption']:
        print("❌ Caption generation failed")
        return
    
    # Print generated captions
    print(f"\n{'='*80}")
    print("Generated Captions")
    print(f"{'='*80}\n")
    
    print("🔹 Fine-tuned Model (Candidate 1):")
    print(f"   {result1['caption']}")
    print(f"   ⏱️  Time: {result1['time']:.2f}s\n")
    
    print("🔸 Base Model (Candidate 2):")
    print(f"   {result2['caption']}")
    print(f"   ⏱️  Time: {result2['time']:.2f}s\n")
    
    print("📖 Reference Caption:")
    print(f"   {args.reference}\n")
    
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
            'reference_caption': args.reference,
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


# This duplicate function is removed - only the first one above is used
    
    # Print sample comparisons
    print(f"\n{'='*80}")
    print("Sample Comparisons")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results[:3]):  # Show first 3
        if 'bertscore_finetuned' in result:
            continue
        
        print(f"Sample {i+1}: {os.path.basename(result['image_path'])}")
        print(f"\n  🔹 Fine-tuned:")
        print(f"     {result['candidate1_finetuned'][:150]}...")
        print(f"\n  🔸 Base Model:")
        print(f"     {result['candidate2_base'][:150]}...")
        
        if result.get('reference'):
            print(f"\n  📖 Reference:")
            print(f"     {result['reference'][:150]}...")
        
        print(f"\n{'-'*80}\n")
    
    print(f"{'='*80}\n")


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
    
    parser.add_argument("--batch-size", type=int, default=32)

    # Single image mode arguments
    parser.add_argument(
        '--reference',
        help='Reference caption for single image mode (required with --image)'
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

# python bbscore.py --checkpoint /home/spandan/scratch/interiit/qwen/checkpoints_tim/checkpoint-100 --image /home/spandan/scratch/interiit/sample_dataset_inter_iit_v1_2/sample1.png --reference "The scene shows an satellite view of an airport runway with two airplanes parked on the tarmac. Apart from the runway, the scene includes storage tanks,airport infrastructure, and roads leading to the runway."

# https://chatgpt.com/gg/v/692c37dc7e2881a3a41a7043c65f3206?token=nH4eDKxWXUlJ5TypASHxOw