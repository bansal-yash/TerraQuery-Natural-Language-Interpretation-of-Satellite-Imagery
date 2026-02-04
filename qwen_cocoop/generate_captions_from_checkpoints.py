#!/usr/bin/env python3
"""
Evaluate fine-tuned Qwen3-VL model on single images or a test set.
FIXED VERSION - Matches training generation exactly.
generate_captions_from_checkpoints.py
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
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

OBJECT="numbers, digits"


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


def generate_caption(model, processor, image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {
            "image_path": image_path,
            "caption": None,
            "error": str(e),
            "generation_time": 0,
            "image_size": None,
        }

    system_instruction = (
        "You are an assistant that generates detailed, accurate descriptions "
        "of satellite and aerial imagery."
    )

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe the image in detail."}
        ]},
    ]
    
    messages = [
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

    prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    # IMPORTANT: images FIRST
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

    # Decode whole sequence but remove prompt
    full_text = processor.decode(output_ids[0], skip_special_tokens=True)

    # Extract ASSISTANT part only
    # (Qwen always formats as "... assistant\n <model output>")
    if "assistant" in full_text:
        caption = full_text.split("assistant", 1)[-1].strip()
    else:
        caption = full_text.strip()

    return {
        "image_path": image_path,
        "caption": caption,
        "generation_time": generation_time,
        "image_size": image.size,
    }


def evaluate_single_image(args):
    """Evaluate on a single image."""
    model, processor = load_model_and_processor(args.checkpoint)
    
    print(f"\n{'='*80}")
    print(f"Generating caption for: {args.image}")
    print(f"{'='*80}\n")
    
    result = generate_caption(
        model, 
        processor, 
        args.image,
    )
    
    if result['caption']:
        print(f"📝 Generated Caption:\n{result['caption']}\n")
        print(f"⏱️  Generation time: {result['generation_time']:.2f}s")
        print(f"🖼️  Image size: {result['image_size']}")
    else:
        print(f"❌ Error: {result['error']}")
    
    print(f"\n{'='*80}\n")


def evaluate_test_set(args):
    """Evaluate on a test set from JSON."""
    model, processor = load_model_and_processor(args.checkpoint)
    
    print(f"\nLoading test data from: {args.test_json}")
    with open(args.test_json, 'r') as f:
        test_data = json.load(f)
    print(f"✅ Loaded {len(test_data)} test samples\n")
    
    results = []
    generated_captions = []
    reference_captions = []
    
    print("Generating captions...")
    for entry in tqdm(test_data):
        # Get image path
        img_path = entry.get('image')
        if not img_path:
            file_name = entry.get('file_name')
            if file_name:
                img_name = file_name.replace('.json', '.png')
                img_path = os.path.join(args.image_dir, img_name)
        
        if not img_path or not os.path.exists(img_path):
            print(f"⚠️  Image not found: {img_path}")
            continue
        
        # Generate caption
        result = generate_caption(
            model,
            processor,
            img_path,
            max_new_tokens=args.max_new_tokens,
            use_beam_search=args.use_beam_search,
            num_beams=args.num_beams if args.use_beam_search else 1,
        )
        
        # Get reference caption
        reference = entry.get('caption') or entry.get('answer')
        if isinstance(reference, list):
            reference = reference[0]
        
        result['reference'] = reference
        results.append(result)
        
        if result['caption'] and reference:
            generated_captions.append(result['caption'])
            reference_captions.append(reference)
    
    # Compute metrics if requested
    if args.compute_metrics and generated_captions:
        print(f"\n{'='*80}")
        print("Computing Evaluation Metrics")
        print(f"{'='*80}\n")
        
        metrics = compute_metrics(generated_captions, reference_captions)
        
        print(f"📊 Results on {len(generated_captions)} samples:")
        print(f"   BLEU-1: {metrics.get('bleu-1', 0):.4f}")
        print(f"   BLEU-4: {metrics.get('bleu-4', 0):.4f}")
        if 'bert-f1' in metrics:
            print(f"   BERT-F1: {metrics.get('bert-f1', 0):.4f}")
        avg_time = sum(r['generation_time'] for r in results) / len(results)
        print(f"   Avg generation time: {avg_time:.2f}s")
        
        results.append({'metrics': metrics})
    
    # Save results
    if args.output_file:
        output_path = args.output_file
        print(f"\n💾 Saving results to: {output_path}")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved!")
    
    # Print sample results
    print(f"\n{'='*80}")
    print("Sample Results")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results[:5]):  # Show first 5
        if 'metrics' in result:
            continue
        print(f"Sample {i+1}: {os.path.basename(result['image_path'])}")
        print(f"Generated: {result['caption'][:150]}...")
        if result.get('reference'):
            print(f"Reference: {result['reference'][:150]}...")
        print(f"Time: {result['generation_time']:.2f}s\n")
    
    print(f"{'='*80}\n")


def compute_metrics(generated: List[str], references: List[str]) -> Dict:
    """Compute BLEU and BERT scores."""
    metrics = {}
    
    # Compute BLEU scores
    try:
        from nltk.translate.bleu_score import corpus_bleu
        
        gen_tokens = [gen.lower().split() for gen in generated]
        ref_tokens = [[ref.lower().split()] for ref in references]
        
        bleu1 = corpus_bleu(ref_tokens, gen_tokens, weights=(1, 0, 0, 0))
        metrics['bleu-1'] = bleu1
        
        bleu4 = corpus_bleu(ref_tokens, gen_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        metrics['bleu-4'] = bleu4
        
    except ImportError:
        print("⚠️  NLTK not installed. Run: pip install nltk")
    except Exception as e:
        print(f"⚠️  BLEU computation failed: {e}")
    
    # Compute BERT Score
    try:
        from bert_score import score as bert_score_fn
        
        P, R, F1 = bert_score_fn(generated, references, lang='en', verbose=False)
        metrics['bert-f1'] = F1.mean().item()
        metrics['bert-precision'] = P.mean().item()
        metrics['bert-recall'] = R.mean().item()
        
    except ImportError:
        print("⚠️  bert-score not installed. Run: pip install bert-score")
    except Exception as e:
        print(f"⚠️  BERT Score computation failed: {e}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned Qwen3-VL model')
    
    # Model arguments
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', help='Path to single image')
    input_group.add_argument('--test-json', help='Path to test JSON file')
    
    parser.add_argument('--image-dir', default='../EarthMind-Bench/img/test/sar/img',
                       help='Directory containing test images')
    
    # Generation arguments
    parser.add_argument('--max-new-tokens', type=int, default=150,
                       help='Maximum tokens to generate')
    parser.add_argument('--use-beam-search', action='store_true',
                       help='Use beam search (default: greedy)')
    parser.add_argument('--num-beams', type=int, default=5,
                       help='Number of beams (only with --use-beam-search)')
    
    # Output arguments
    parser.add_argument('--output-file', help='Path to save results JSON')
    parser.add_argument('--compute-metrics', action='store_true',
                       help='Compute BLEU and BERT scores')
    
    args = parser.parse_args()
    
    if args.compute_metrics and not args.test_json:
        parser.error("--compute-metrics requires --test-json")
    
    # Run evaluation
    if args.image:
        evaluate_single_image(args)
    else:
        evaluate_test_set(args)


if __name__ == '__main__':
    main()