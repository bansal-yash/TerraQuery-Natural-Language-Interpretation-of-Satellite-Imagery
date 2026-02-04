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
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from tqdm import tqdm
import time
try:
    from unsloth import FastVisionModel
    UNSLOTH_AVAILABLE = True
except Exception:
    UNSLOTH_AVAILABLE = False
    print("⚠️  Unsloth not available, will use standard transformers")


OBJECT = "digits"

# ------------------------------
# --- Existing model loaders ---
# ------------------------------
def load_model_and_processor(checkpoint_path: str, device: str = "cuda"):
    """Load fine-tuned model and processor."""
    print(f"Loading model from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Check if this is a LoRA adapter checkpoint
    is_lora_checkpoint = os.path.exists(os.path.join(checkpoint_path, "adapter_config.json"))
    
    if is_lora_checkpoint and UNSLOTH_AVAILABLE:
        print("✅ Detected LoRA adapter checkpoint, loading with Unsloth...")
        
        # First, load the base model with Unsloth
        base_model_name = "unsloth/Qwen3-VL-8B-Instruct"  # Match your training script
        
        print(f"   Loading base model: {base_model_name}")
        model, tokenizer = FastVisionModel.from_pretrained(
            base_model_name,
            load_in_4bit=False,  # Set to True if you trained with 4-bit
            device_map=device,
        )
        processor = tokenizer
        
        # Then load the LoRA adapter weights
        print(f"   Loading LoRA adapters from: {checkpoint_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint_path)
        
        # Merge adapters for faster inference (optional but recommended)
        print("   Merging LoRA adapters into base model...")
        model = model.merge_and_unload()
        
    elif is_lora_checkpoint and not UNSLOTH_AVAILABLE:
        print("⚠️  LoRA checkpoint detected but Unsloth not available!")
        print("   Attempting to load with PEFT...")
        
        from peft import PeftModel
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        
        # Load base model
        base_model_name = "Qwen/Qwen3-VL-8B-Instruct"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(base_model_name)
        
        # Load adapter
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
        
    else:
        print("✅ Loading fully fine-tuned model (not LoRA)...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(checkpoint_path)
    
    # Set to evaluation mode
    model.eval()
    
    # Convert to bfloat16 if not already
    try:
        model = model.to(dtype=torch.bfloat16)
    except:
        pass
    
    model_device = next(model.parameters()).device
    print(f"✅ Model loaded successfully")
    print(f"   Device: {model_device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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


def compare_models_batch(args):
    """Compare fine-tuned model vs base model on test images."""
    
    # Load both models (unchanged)
    finetuned_model, finetuned_processor = load_finetuned_model(args.checkpoint)
    base_model, base_processor = load_base_model()
    
    # Load test data
    print(f"\nLoading test data from: {args.test_json}")
    with open(args.test_json, 'r') as f:
        test_data = json.load(f)
    
    # Limit number of samples if specified
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    
    print(f"✅ Processing {len(test_data)} test samples\n")
    
    results = []
    candidate1_captions = []  # Fine-tuned model
    candidate2_captions = []  # Base model
    reference_captions = []
    
    print("Generating captions from both models...")
    for entry in tqdm(test_data):
        # Get image path


        img_path = entry.get('image') 
        if not img_path:
            file_name = entry.get('file_name')
            if file_name:
                img_name = file_name.replace('.json', '.png')
                img_path = os.path.join(args.image_dir, img_name)
        img_path = os.path.join(args.image_dir, img_path)
        
        if not img_path or not os.path.exists(img_path):
            print(f"⚠️  Image not found: {img_path}")
            continue
        
        # Generate from fine-tuned model (candidate1)
        result1 = generate_caption_finetuned(finetuned_model, finetuned_processor, img_path)
        
        # Generate from base model (candidate2)
        result2 = generate_caption_base(base_model, base_processor, img_path)
        
        # Get reference caption if available
        reference = entry.get('caption') or entry.get('answer')
        if isinstance(reference, list):
            reference = reference[0]
        
        result_entry = {
            'image_path': img_path,
            'candidate1_finetuned': result1['caption'],
            'candidate2_base': result2['caption'],
            'reference': reference,
            'generation_time_finetuned': result1['time'],
            'generation_time_base': result2['time'],
        }
        
        results.append(result_entry)
        
        if result1['caption'] and result2['caption'] and reference:
            candidate1_captions.append(result1['caption'])
            candidate2_captions.append(result2['caption'])
            reference_captions.append(reference)
    
    # Compute BERTScore / BERT-BLEU for both models against reference
    if candidate1_captions and reference_captions:
        print(f"\n{'='*80}")
        print("Computing scores against reference captions")
        print(f"{'='*80}\n")
        
        bertscore_ft = compute_bertscore(candidate1_captions, reference_captions)
        bertscore_base = compute_bertscore(candidate2_captions, reference_captions)
        
        if bertscore_ft and bertscore_base:
            print(f"📊 Results (n={len(candidate1_captions)}):\n")
            
            if bertscore_ft['bert-precision'] is not None:
                print("🔹 Fine-tuned Model vs Reference (BERTScore):")
                print(f"   Precision: {bertscore_ft['bert-precision']:.4f}")
                print(f"   Recall:    {bertscore_ft['bert-recall']:.4f}")
                print(f"   F1:        {bertscore_ft['bert-f1']:.4f}\n")
                
                print("🔸 Base Model vs Reference (BERTScore):")
                print(f"   Precision: {bertscore_base['bert-precision']:.4f}")
                print(f"   Recall:    {bertscore_base['bert-recall']:.4f}")
                print(f"   F1:        {bertscore_base['bert-f1']:.4f}\n")
            else:
                print("ℹ️ bert-score (library) not available; showing BERT-BLEU instead.\n")
            
            print("🔹 Fine-tuned Model vs Reference (BERT-BLEU4):")
            print(f"   Mean BERT-BLEU: {bertscore_ft['bert-bleu']:.6f}")
            print("🔸 Base Model vs Reference (BERT-BLEU4):")
            print(f"   Mean BERT-BLEU: {bertscore_base['bert-bleu']:.6f}\n")
            
            diff_f1 = bertscore_ft['bert-bleu'] - bertscore_base['bert-bleu']
            print("📊 Comparison (BERT-BLEU):")
            if diff_f1 > 0:
                print(f"   Fine-tuned model is better by {diff_f1:.6f} BERT-BLEU points")
            elif diff_f1 < 0:
                print(f"   Base model is better by {abs(diff_f1):.6f} BERT-BLEU points")
            else:
                print(f"   Both models have equal BERT-BLEU scores")
            
            avg_time_ft = sum(r['generation_time_finetuned'] for r in results) / len(results)
            avg_time_base = sum(r['generation_time_base'] for r in results) / len(results)
            print(f"\n⏱️  Average Generation Time:")
            print(f"   Fine-tuned: {avg_time_ft:.2f}s")
            print(f"   Base:       {avg_time_base:.2f}s")
            
            results.append({
                'bertscore_finetuned': bertscore_ft,
                'bertscore_base': bertscore_base
            })
    
    # Save results
    if args.output_file:
        output_path = args.output_file
        print(f"\n💾 Saving results to: {output_path}")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved!")
    
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
