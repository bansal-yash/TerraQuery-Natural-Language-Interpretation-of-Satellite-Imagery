#!/usr/bin/env python3
"""
Simple script to draw bounding boxes on an image using a fine-tuned grounding model.

Usage:
    python draw_bboxes_simple.py \
        --checkpoint /path/to/checkpoint \
        --image /path/to/image.png \
        --object-class "ships" \
        --output output.png
    

"""

import argparse
import os
import re
import time
import torch
from PIL import Image, ImageDraw, ImageFont
from unsloth import FastVisionModel
from peft import PeftModel


def parse_boxes_from_text(text):
    """Parse bounding boxes from model output.
    Handles both formats:
    - <ref>label</ref><box>(x1,y1),(x2,y2)</box>
    - <ref>label</ref><box>x1,y1,x2,y2</box>
    Returns: List of (label, x1, y1, x2, y2) in coordinates [0-1000]
    """
    parsed = []
    
    # Try format 1: <box>(x1,y1),(x2,y2)</box>
    pattern1 = r'<ref>(.*?)</ref><box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
    matches1 = re.findall(pattern1, text)
    for match in matches1:
        label = match[0]
        x1, y1, x2, y2 = int(match[1]), int(match[2]), int(match[3]), int(match[4])
        parsed.append((label, x1, y1, x2, y2))
    
    # Try format 2: <box>x1,y1,x2,y2</box> (without parentheses)
    pattern2 = r'<ref>(.*?)</ref><box>(\d+),(\d+),(\d+),(\d+)</box>'
    matches2 = re.findall(pattern2, text)
    for match in matches2:
        label = match[0]
        x1, y1, x2, y2 = int(match[1]), int(match[2]), int(match[3]), int(match[4])
        parsed.append((label, x1, y1, x2, y2))
    
    return parsed


def draw_bboxes(image_path, bboxes, output_path, color=(0, 255, 0), line_width=3):
    """Draw bounding boxes on image and save.
    bboxes: list of (label, x1, y1, x2, y2) tuples with coordinates in [0, 1000]
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    
    # Try to use a decent font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
    
    print(f"\n📦 Drawing {len(bboxes)} bounding boxes...")
    
    for idx, (label, x1, y1, x2, y2) in enumerate(bboxes, 1):
        # Convert from [0, 1000] to actual image coordinates
        x1_img = int(x1 * width / 1000)
        y1_img = int(y1 * height / 1000)
        x2_img = int(x2 * width / 1000)
        y2_img = int(y2 * height / 1000)
        
        print(f"   Box {idx}: {label} at ({x1_img}, {y1_img}) → ({x2_img}, {y2_img})")
        
        # Draw rectangle
        draw.rectangle([x1_img, y1_img, x2_img, y2_img], outline=color, width=line_width)
        
        # Draw label background and text
        text_bbox = draw.textbbox((x1_img, max(0, y1_img - 25)), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1_img, max(0, y1_img - 25)), label, fill=(255, 255, 255), font=font)
    
    image.save(output_path)
    print(f"\n✅ Saved visualization to: {output_path}\n")
    return image


def load_model(checkpoint_path=None, device="cuda"):
    """Load model. If checkpoint provided, load fine-tuned model with LoRA adapters, else load base model."""
    base_model_name = "unsloth/Qwen3-VL-8B-Instruct"
    
    if checkpoint_path:
        print(f"\n🔧 Loading fine-tuned model from checkpoint: {checkpoint_path}")
        print(f"   └─ Loading base model: {base_model_name}")
        model, tokenizer = FastVisionModel.from_pretrained(
            base_model_name,
            load_in_4bit=False,
            device_map=device,
            use_safetensors=True
        )
        processor = tokenizer
        
        # Load LoRA adapter weights
        print(f"   └─ Loading LoRA adapters from: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        
        # Merge adapters for faster inference
        print("   └─ Merging LoRA adapters...")
        model = model.merge_and_unload()
        
        print("✅ Fine-tuned model loaded successfully!\n")
    else:
        print(f"\n🔧 Loading base model (no checkpoint provided): {base_model_name}")
        model, tokenizer = FastVisionModel.from_pretrained(
            base_model_name,
            load_in_4bit=False,
            device_map=device,
            use_safetensors=True
        )
        processor = tokenizer
        print("✅ Base model loaded successfully!\n")
    
    return model, processor


def generate_bboxes(model, processor, image_path, object_class):
    """Generate bounding box predictions for given object class."""
    print(f"🔍 Detecting '{object_class}' in image: {os.path.basename(image_path)}")
    
    # Build prompt
    # question = f"Describe the locations of all visible instances of {object_class} using bounding box coordinates (x1,y1,x2,y2) normalized to 1000. YOU MUST IDENTIFY THE OBJECTS AND ONLY THEM. DO NOT MENTION ANY OTHER OBJECTS. Format: <ref>label</ref><box>(x1,y1),(x2,y2)</box>"

    question = f"""Describe the locations of all visible instances of {object_class} using bounding box coordinates (x1,y1,x2,y2) normalized to 1000.
    YOU MUST IDENTIFY THE OBJECTS AND ONLY THEM. DO NOT MENTION ANY OTHER OBJECTS.

    MISSION CRITICAL FORMATTING INSTRUCTIONS:
    Your response MUST follow this exact format for each object:
    <ref>label</ref><box>(x1,y1),(x2,y2)</box>

    Where:
    - label: descriptive name for the object
    - x1,y1,x2,y2: integer coordinates in range [0,1000]
    - x1,y1 is top-left corner
    - x2,y2 is bottom-right corner

    Example: <ref>yellow bus</ref><box>(120,340),(450,680)</box>

    Output ONLY the bounding box annotations, one per line. No explanations."""



    messages = [
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
    
    print("   └─ Generating predictions...")
    start = time.time()
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    
    generation_time = time.time() - start
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    prediction = output_text[0] if output_text else ""
    
    print(f"   └─ Generation time: {generation_time:.2f}s")
    print(f"\n📝 Raw prediction:\n   {prediction}\n")
    
    return prediction


def main():
    parser = argparse.ArgumentParser(
        description='Draw bounding boxes on image using fine-tuned grounding model'
    )
    
    parser.add_argument(
        '--checkpoint',
        help='Path to fine-tuned model checkpoint (optional, uses base model if not provided)'
    )
    
    parser.add_argument(
        '--image',
        required=True,
        help='Path to input image'
    )
    
    parser.add_argument(
        '--object-class',
        required=True,
        help='Object class to detect (e.g., "ships", "vehicles", "buildings")'
    )
    
    parser.add_argument(
        '--output',
        help='Path to save output image with bounding boxes (default: <image_name>_bbox.png)'
    )
    
    parser.add_argument(
        '--color',
        default='0,255,0',
        help='Box color as R,G,B (default: 0,255,0 for green)'
    )
    
    parser.add_argument(
        '--line-width',
        type=int,
        default=3,
        help='Line width for bounding boxes (default: 3)'
    )
    
    parser.add_argument(
        '--device',
        default='cuda',
        help='Device to use (default: cuda)'
    )
    
    args = parser.parse_args()
    
    # Check image exists
    if not os.path.exists(args.image):
        print(f"❌ Error: Image not found: {args.image}")
        return
    
    # Set output path
    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        output_dir = os.path.dirname(args.image) or '.'
        args.output = os.path.join(output_dir, f"{base_name}_bbox.png")
    
    # Parse color
    try:
        color = tuple(map(int, args.color.split(',')))
        if len(color) != 3 or any(c < 0 or c > 255 for c in color):
            raise ValueError
    except:
        print(f"❌ Error: Invalid color format. Use R,G,B (e.g., 0,255,0)")
        return
    
    print(f"\n{'='*80}")
    print(f"Bounding Box Detection")
    print(f"{'='*80}")
    print(f"Image: {args.image}")
    print(f"Object Class: {args.object_class}")
    print(f"Checkpoint: {args.checkpoint if args.checkpoint else 'Base model (no checkpoint)'}")
    print(f"Output: {args.output}")
    print(f"Box Color: RGB{color}")
    print(f"{'='*80}\n")
    
    # Load model
    model, processor = load_model(args.checkpoint, args.device)
    
    # Generate predictions
    prediction = generate_bboxes(model, processor, args.image, args.object_class)
    
    # Parse bounding boxes
    bboxes = parse_boxes_from_text(prediction)
    
    if not bboxes:
        print(f"⚠️  No bounding boxes detected for '{args.object_class}'")
        print(f"   Make sure the object class matches what's in the image.")
    else:
        # Draw bounding boxes
        draw_bboxes(args.image, bboxes, args.output, color, args.line_width)
        
        print(f"{'='*80}")
        print(f"✅ Done! Found {len(bboxes)} {args.object_class}")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
