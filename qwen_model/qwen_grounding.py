#!/usr/bin/env python3
"""
Qwen3-VL Native Grounding Script
Uses the model's built-in grounding capability to output bounding boxes
for objects in the image.
"""
import re
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import os
import json



OBJECT = "parking lot"
# === Settings ===
model_id = "Qwen/Qwen3-VL-8B-Instruct"
image_path = "/home/samyak/scratch/interiit/skySkript_mini/images/a1000070447_US_21.jpg"
# Path to a SAM checkpoint (.pth). Set to a local path to enable mask generation.
# Example: sam_vit_l_0b3195.pth downloaded from Segment Anything model links.
# sam_checkpoint = "checkpoints/sam/sam_vit_l_0b3195.pth"

# Grounding prompt: ask the model to localize objects and return bounding boxes
# The model is trained to output <ref>object</ref><box>(x1,y1),(x2,y2)</box>
# where coordinates are normalized to [0, 1000)
grounding_query = f"Locate and output bounding boxes for all {OBJECT} in the image. Format: <ref>label</ref><box>(x1,y1),(x2,y2)</box>"

# === Load Model & Processor ===
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading model {model_id} on {device}...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id, dtype=dtype, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)
model.eval()

# === Load Image ===
image = Image.open(image_path).convert("RGB")
image_width, image_height = image.size
print(f"Image loaded: {image_width}x{image_height}")

# === Prepare Conversation ===
# Add a system instruction so the model can combine nearby objects into a single bbox
# system_instruction = (
#     "When multiple objects of the requested class are adjacent or overlapping, "
#     "combine them into a single bounding box that tightly encloses them. "
#     "Consider objects 'close' if the gap between boxes is less than 5% of the image diagonal or if boxes overlap. "
#     "Otherwise, keep separate boxes. Prefer fewer, larger boxes over many tiny boxes when objects are contiguous."
# )

# conversation = [
#     {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
#     {"role": "user", "content": [
#         {"type": "image", "image": image},
#         {"type": "text", "text": grounding_query}
#     ]}
# ]

system_instruction = (
        "Only segment the requested object class directly and completely visible in the image."
    )

    # Use an HF-style chat message list with the *image path* (not PIL.Image object).
    # Strong, example-driven prompt to encourage listing all boxes (one per line)
user_text = (
        f"Locate and output bounding boxes for ALL {OBJECT} in the image. "
        "CRITCAL FORMATTING INSTRUCTIONS: Return one box per line, using this EXACT format: <ref>label</ref><box>(x1,y1),(x2,y2)</box><|box_end|>. "
        "Coordinates must be integer pixels in the range [0,1000]. Do NOT include any extra commentary or summaries. "
        "If there are no such objects, output exactly: NO_BOX."
    )

messages = [
        {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text}
        ]}
    ]

prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)

# === Generate Response ===
print("Generating grounding output...")
with torch.no_grad():
    gen_outputs = model.generate(
        **inputs,
        max_new_tokens=256,  # Allow enough tokens for multiple boxes
        do_sample=False,     # Greedy decoding for deterministic results
    )

generated_ids = gen_outputs[0]
response_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=False)
print("\n=== Model Response ===")
print(response_text)
print("======================\n")

# === Parse Bounding Boxes from Response ===
# Pattern: <ref>label</ref><box>(x1,y1),(x2,y2)</box>
# Coordinates are normalized to [0, 1000)
box_pattern = r'<ref>(.*?)</ref><box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
matches = re.findall(box_pattern, response_text)

if not matches:
    print("Warning: No bounding boxes found in model output.")
    print("The model may not support grounding for Qwen3-VL, or the prompt needs adjustment.")
    print("\nTrying alternative: ask for text description with coordinates...")
    
    # Fallback: try a different prompt style
    conversation_alt = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"Describe the locations of the {OBJECT} using bounding box coordinates (x1,y1,x2,y2) normalized to 1000."}
        ]}
    ]
    prompt_alt = processor.apply_chat_template(conversation_alt, tokenize=False, add_generation_prompt=True)
    inputs_alt = processor(text=prompt_alt, images=[image], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        gen_alt = model.generate(**inputs_alt, max_new_tokens=256, do_sample=False)
    
    response_alt = processor.tokenizer.decode(gen_alt[0], skip_special_tokens=True)
    print("\n=== Alternative Response ===")
    print(response_alt)
    print("============================\n")
    
    # Try to extract any number sequences that look like boxes
    # Pattern: look for sequences of 4 numbers
    number_groups = re.findall(r'\(?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)?', response_alt)
    if number_groups:
        print(f"Found {len(number_groups)} potential coordinate groups in alternative response.")
        matches = [(f"{OBJECT} (inferred)", x1, y1, x2, y2) for x1, y1, x2, y2 in number_groups]
else:
    print(f"Found {len(matches)} bounding boxes in model output.")

# === Draw Bounding Boxes on Image ===
if matches:
    draw = ImageDraw.Draw(image)
    
    # Try to load a font for labels (fallback to default if not available)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    colors = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
        "#FFA500", "#800080", "#008000", "#FFC0CB"
    ]
    
    for idx, (label, x1_str, y1_str, x2_str, y2_str) in enumerate(matches):
        # Convert normalized coords [0, 1000) to pixel coords
        x1 = int(x1_str) * image_width // 1000
        y1 = int(y1_str) * image_height // 1000
        x2 = int(x2_str) * image_width // 1000
        y2 = int(y2_str) * image_height // 1000
        
        # Clamp to image bounds
        x1 = max(0, min(x1, image_width))
        y1 = max(0, min(y1, image_height))
        x2 = max(0, min(x2, image_width))
        y2 = max(0, min(y2, image_height))
        
        print(f"Box {idx+1}: {label} -> ({x1},{y1}), ({x2},{y2})")
        
        # Draw bounding box
        color = colors[idx % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label background and text
        text = f"{label}"
        # Get text bounding box for background
        bbox = draw.textbbox((x1, y1 - 20), text, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 20), text, fill="white", font=font)
    
    # Save annotated image
    output_path = "grounded_output.jpg"
    image.save(output_path)
    print(f"\nSaved grounded image to: {output_path}")


    # Also print normalized AABB coords [xmin, ymin, xmax, ymax] (0-1)
    print("\n=== Normalized AABB (obj_coord) ===")
    for idx, m in enumerate(matches):
        # m expected: (label, x1, y1, x2, y2) or possibly (x1, y1, x2, y2)
        try:
            if len(m) == 5:
                label = m[0]
                x1, y1, x2, y2 = float(m[1]), float(m[2]), float(m[3]), float(m[4])
            elif len(m) == 4:
                label = OBJECT
                x1, y1, x2, y2 = float(m[0]), float(m[1]), float(m[2]), float(m[3])
            else:
                print(f"Skipping unexpected match format: {m}")
                continue
        except Exception as e:
            print(f"Error parsing match {m}: {e}")
            continue

        # Convert from [0,1000) to normalized [0,1], clamp to [0,1]
        nx1 = max(0.0, min(1.0, x1 / 1000.0))
        ny1 = max(0.0, min(1.0, y1 / 1000.0))
        nx2 = max(0.0, min(1.0, x2 / 1000.0))
        ny2 = max(0.0, min(1.0, y2 / 1000.0))

        obj_coord = [round(nx1, 6), round(ny1, 6), round(nx2, 6), round(ny2, 6)]
        print(f"Box {idx+1}: {label} - obj_coord: {obj_coord}")

else:
    print("\nNo valid bounding boxes were extracted. The model may not support grounding in this way.")
    print("Recommendation: Use a dedicated object detector (GroundingDINO, YOLO) for reliable bbox outputs.")
