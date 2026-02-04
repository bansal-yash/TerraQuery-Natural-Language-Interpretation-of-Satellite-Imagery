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
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BertTokenizer, BertModel
from tqdm import tqdm
import time
from threading import Thread
from peft import PeftModel, PeftConfig

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

        # 1. Resolve image path
        img_path = (
            entry.get("image")
            or entry.get("image_id")
            or entry.get("file_name")
        )

        if img_path is None:
            raise ValueError(f"No image field found in entry: {entry}")

        if isinstance(img_path, str) and img_path.endswith(".json"):
            img_path = img_path.replace(".json", ".png")

        full_path = os.path.join(self.image_dir, img_path)

        # Load image
        image = Image.open(full_path).convert("RGB")

        # 2. Resolve reference caption
        reference = (
            entry.get("caption")
            or entry.get("answer")
            or entry.get("ground_truth")
        )

        if isinstance(reference, list):
            reference = reference[0]

        if reference is None:
            raise ValueError(f"No reference caption found in entry: {entry}")

        return {
            "image": image,
            "path": full_path,
            "reference": reference,
        }

# ------------------------------
# Model loading functions
# ------------------------------
def load_finetuned_model(checkpoint_path: str, device: str = "cuda:0"):
    """Load base Qwen3-VL + LoRA adapter (fine-tuned model) on a specific GPU."""
    print(f"Loading base model from local directory: {BASE_MODEL_DIR}")
    print(f"Loading LoRA adapter from: {checkpoint_path}")
    
    # First check if the base model directory exists
    if not os.path.exists(BASE_MODEL_DIR):
        raise FileNotFoundError(f"Base model directory not found: {BASE_MODEL_DIR}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

    # 1) Load base model from local directory
    print("Loading base Qwen3-VL model...")
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=torch.bfloat16,
        local_files_only=True  # Important: only look locally
    )
    
    print(f"Moving base model to {device}...")
    base_model.to(device)

    # 2) Attach LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.to(device)
    model.eval()

    # 3) Load processor from local directory
    processor = AutoProcessor.from_pretrained(BASE_MODEL_DIR, local_files_only=True)

    # Print parameter stats
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable} / {total} = {trainable / total:.4%}")
    print(f"✅ Fine-tuned model (base + LoRA) loaded on {device}")
    
    return model, processor


def load_base_model(device: str = "cuda:1"):
    """Load the un-finetuned base Qwen3-VL model on a specific GPU."""
    print(f"Loading base Qwen3-VL model from local dir: {BASE_MODEL_DIR}")
    
    # Check if the directory exists
    if not os.path.exists(BASE_MODEL_DIR):
        raise FileNotFoundError(f"Base model directory not found: {BASE_MODEL_DIR}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=torch.bfloat16,
        local_files_only=True  # Important: only look locally
    )
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(BASE_MODEL_DIR, local_files_only=True)

    print(f"Base model loaded on {device}")
    return model, processor

# ------------------------------
# Scoring functions
# ------------------------------
def _tokenize_simple(text: str) -> List[str]:
    """Simple whitespace tokenizer."""
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
        print("✅ BERTScore computed successfully")
    except Exception as e:
        print(f"⚠️ bert-score library not available or failed ({e}). Only computing BERT-BLEU.")

    # 2) Compute BERT-BLEU4 per PDF
    print("Loading BERT model for BERT-BLEU...")
    bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    N = 4
    eps = 1e-8

    per_sample_bert_bleu = []
    per_sample_pn = []  # list of lists [P1,P2,P3,P4]

    print(f"Computing BERT-BLEU for {len(candidates)} samples...")
    for cand, ref in tqdm(zip(candidates, references), total=len(candidates), desc="BERT-BLEU"):
        # tokenize simply
        cand_toks = _tokenize_simple(cand)
        ref_toks = _tokenize_simple(ref)

        Pn_list = []
        for n in range(1, N + 1):
            cand_ngrams = _generate_ngrams(cand_toks, n)
            ref_ngrams = _generate_ngrams(ref_toks, n)
            if len(cand_ngrams) == 0:
                Pn_list.append(0.0)
                continue
            if len(ref_ngrams) == 0:
                Pn_list.append(0.0)
                continue

            # Compute embeddings
            ref_embs = _embed_texts_bert(ref_ngrams, bert_tok, bert_model, torch_device, batch_size=32)
            cand_embs = _embed_texts_bert(cand_ngrams, bert_tok, bert_model, torch_device, batch_size=32)

            # Convert lists to tensors on CPU
            ref_stack = torch.stack(ref_embs, dim=0)  # (R, D)
            cand_stack = torch.stack(cand_embs, dim=0)  # (C, D)

            # Normalize for cosine similarity
            ref_norm = ref_stack / (ref_stack.norm(dim=1, keepdim=True).clamp(min=1e-9))
            cand_norm = cand_stack / (cand_stack.norm(dim=1, keepdim=True).clamp(min=1e-9))

            # Compute similarities
            sims = torch.matmul(cand_norm, ref_norm.T)  # (C, R)
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
# Batch comparison function
# ------------------------------
def compare_models_batch(args):
    """Batched evaluation: finetuned vs base, on different GPUs, in parallel."""

    # 1) Load models on their own devices
    print(f"\n{'='*80}")
    print("Loading models...")
    print(f"{'='*80}")
    
    print(f"\nLoading fine-tuned model on {args.ft_device}...")
    finetuned_model, finetuned_processor = load_finetuned_model(
        args.checkpoint,
        device=args.ft_device,
    )
    
    print(f"\nLoading base model on {args.base_device}...")
    base_model, base_processor = load_base_model(
        device=args.base_device,
    )

    # 2) Load dataset
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

    candidate1_list = []  # finetuned captions
    candidate2_list = []  # base captions
    reference_list = []   # GT captions
    results = []

    print(f"\n{'='*80}")
    print(f"Starting batched inference with batch size {args.batch_size}")
    print(f"{'='*80}\n")

    for batch_idx, batch in enumerate(tqdm(loader, desc="Processing batches")):
        images = [item["image"] for item in batch]
        references = [item["reference"] for item in batch]
        paths = [item["path"] for item in batch]

        # 3) Build prompts for both models
        def make_prompt(proc, img):
            return proc.apply_chat_template(
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

        ft_prompts = [make_prompt(finetuned_processor, img) for img in images]
        base_prompts = [make_prompt(base_processor, img) for img in images]

        # 4) Encode inputs for each model on its own device
        ft_inputs = finetuned_processor(
            images=images,
            text=ft_prompts,
            return_tensors="pt",
            padding=True,
        ).to(finetuned_model.device)

        base_inputs = base_processor(
            images=images,
            text=base_prompts,
            return_tensors="pt",
            padding=True,
        ).to(base_model.device)

        ft_out = None
        base_out = None

        # 5) Run both generates in parallel (one GPU each)
        def run_ft():
            nonlocal ft_out
            with torch.no_grad():
                ft_out = finetuned_model.generate(
                    **ft_inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=finetuned_processor.tokenizer.pad_token_id,
                    eos_token_id=finetuned_processor.tokenizer.eos_token_id,
                )

        def run_base():
            nonlocal base_out
            with torch.no_grad():
                base_out = base_model.generate(
                    **base_inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=base_processor.tokenizer.pad_token_id,
                    eos_token_id=base_processor.tokenizer.eos_token_id,
                )

        t1 = Thread(target=run_ft)
        t2 = Thread(target=run_base)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # 6) Decode both batches
        ft_texts = finetuned_processor.batch_decode(ft_out, skip_special_tokens=True)
        base_texts = base_processor.batch_decode(base_out, skip_special_tokens=True)

        # 7) Accumulate results
        for ft_cap, base_cap, ref, path in zip(ft_texts, base_texts, references, paths):
            candidate1_list.append(ft_cap)
            candidate2_list.append(base_cap)
            reference_list.append(ref)

            results.append({
                "image_path": path,
                "candidate1_finetuned": ft_cap,
                "candidate2_base": base_cap,
                "reference": ref,
            })

    # 8) Compute metrics
    print(f"\n{'='*80}")
    print("Computing metrics (BERTScore / BERT-BLEU)...")
    print(f"{'='*80}\n")

    metrics_ft = compute_bertscore(candidate1_list, reference_list)
    metrics_base = compute_bertscore(candidate2_list, reference_list)

    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"\n🔹 Fine-tuned BERT-BLEU: {metrics_ft['bert-bleu']:.6f}")
    print(f"🔸 Base       BERT-BLEU: {metrics_base['bert-bleu']:.6f}")
    
    if metrics_ft['bert-f1'] is not None:
        print(f"\n🔹 Fine-tuned BERT-F1:  {metrics_ft['bert-f1']:.6f}")
        print(f"🔸 Base       BERT-F1:  {metrics_base['bert-f1']:.6f}")
    
    diff = metrics_ft['bert-bleu'] - metrics_base['bert-bleu']
    if diff > 0:
        print(f"\n📈 Fine-tuned model is better by {diff:.6f} BERT-BLEU points")
    elif diff < 0:
        print(f"\n📉 Base model is better by {abs(diff):.6f} BERT-BLEU points")
    else:
        print(f"\n📊 Both models have equal BERT-BLEU scores")

    # Add metrics to results
    final_results = {
        "predictions": results,
        "metrics_finetuned": metrics_ft,
        "metrics_base": metrics_base,
        "summary": {
            "finetuned_bert_bleu": metrics_ft["bert-bleu"],
            "base_bert_bleu": metrics_base["bert-bleu"],
            "difference_bert_bleu": diff,
            "finetuned_bert_f1": metrics_ft["bert-f1"],
            "base_bert_f1": metrics_base["bert-f1"],
            "num_samples": len(results),
            "batch_size": args.batch_size,
            "checkpoint": args.checkpoint,
        }
    }

    # 9) Save JSON
    if args.output_file:
        print(f"\n💾 Saving results to {args.output_file}...")
        with open(args.output_file, "w") as f:
            json.dump(final_results, f, indent=2)
        print(f"✅ Results saved successfully!")

    print(f"\n{'='*80}")
    print("🎉 Evaluation complete!")
    print(f"{'='*80}\n")
    
    return final_results

# ------------------------------
# Main function
# ------------------------------
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

    # Device arguments
    parser.add_argument(
        '--ft-device',
        default='cuda:0',
        help='CUDA device for finetuned model, e.g. cuda:0'
    )
    parser.add_argument(
        '--base-device',
        default='cuda:1',
        help='CUDA device for base model, e.g. cuda:1'
    )
    
    # Input mode selection (currently only batch mode is fully implemented)
    parser.add_argument(
        '--test-json', 
        required=True,
        help='Path to test JSON file for batch evaluation'
    )
    
    # Batch mode arguments
    parser.add_argument(
        '--image-dir', 
        required=True,
        help='Directory containing test images'
    )
    parser.add_argument(
        '--max-samples', 
        type=int, 
        default=None,
        help='Maximum number of samples to process in batch mode'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=4,
        help='Batch size for inference'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-file', 
        required=True,
        help='Path to save comparison results JSON'
    )
    
    args = parser.parse_args()
    
    # Run batch evaluation
    compare_models_batch(args)


if __name__ == "__main__":
    main()