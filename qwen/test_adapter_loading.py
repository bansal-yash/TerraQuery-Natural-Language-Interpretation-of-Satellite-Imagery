#!/usr/bin/env python3
"""
Simple script to load vanilla Qwen model, then load adapters, and compare outputs.
Tests on a single image with timing information.
"""

import argparse
import time
import torch
from PIL import Image
from unsloth import FastVisionModel
from peft import PeftModel


def generate_answer(model, processor, image_path, question):
    """Generate answer for a single image-question pair."""
    image = Image.open(image_path).convert("RGB")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]
    
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = processor(
        images=[image],
        text=prompt,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    return output_text[0].strip()


def main():
    parser = argparse.ArgumentParser(description='Test vanilla model vs adapter-loaded model')
    parser.add_argument('--base-model', 
                        default='unsloth/Qwen3-VL-8B-Instruct', 
                        help='Base model name or path')
    parser.add_argument('--adapter-path', required=True, 
                        help='Path to LoRA adapter checkpoint')
    parser.add_argument('--image', required=True, 
                        help='Path to test image')
    parser.add_argument('--question', required=True, 
                        help='Question to ask about the image')
    
    args = parser.parse_args()
    
    print("Loading vanilla model...")
    start = time.time()
    vanilla_model, processor = FastVisionModel.from_pretrained(
        args.base_model,
        load_in_4bit=False,
        device_map="auto",
        use_safetensors=True
    )
    print(f"✓ Loaded in {time.time()-start:.2f}s\n")
    
    print("Generating with vanilla model...")
    start = time.time()
    vanilla_answer = generate_answer(vanilla_model, processor, args.image, args.question)
    print(f"Vanilla output: {vanilla_answer}")
    print(f"Time: {time.time()-start:.2f}s\n")
    
    print("Loading adapters...")
    start = time.time()
    finetuned_model = PeftModel.from_pretrained(vanilla_model, args.adapter_path)
    adapter_load_time = time.time() - start
    print(f"✓ Adapters loaded in {adapter_load_time:.2f}s\n")
    
    print("Generating with adapter-loaded model (before merge)...")
    start = time.time()
    adapter_answer = generate_answer(finetuned_model, processor, args.image, args.question)
    print(f"Adapter-loaded output: {adapter_answer}")
    print(f"Time: {time.time()-start:.2f}s\n")
    
    print("Merging and unloading adapters...")
    start = time.time()
    merged_model = finetuned_model.merge_and_unload()
    merge_time = time.time() - start
    print(f"✓ Merged in {merge_time:.2f}s\n")
    
    print("Generating with merged model (after unload)...")
    start = time.time()
    merged_answer = generate_answer(merged_model, processor, args.image, args.question)
    print(f"Merged output: {merged_answer}")
    print(f"Time: {time.time()-start:.2f}s\n")
    
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Adapter loading time: {adapter_load_time:.2f}s")
    print(f"Merge & unload time:  {merge_time:.2f}s")


if __name__ == '__main__':
    main()
