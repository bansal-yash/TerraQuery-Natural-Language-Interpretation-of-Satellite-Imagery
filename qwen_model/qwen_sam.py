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

# Try optional SAM import (Segment Anything). If not installed, we print instructions
try:
    from segment_anything import sam_model_registry, SamPredictor
    _HAS_SAM = True
except Exception:
    _HAS_SAM = False

OBJECT = "bus"
# === Settings ===
model_id = "Qwen/Qwen3-VL-8B-Instruct"
image_path = "P0003_0002.png"
# Path to a SAM checkpoint (.pth). Set to a local path to enable mask generation.
# Example: sam_vit_l_0b3195.pth downloaded from Segment Anything model links.
sam_checkpoint = "checkpoints/sam/sam_vit_l_0b3195.pth"

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
system_instruction = (
    "When multiple objects of the requested class are adjacent or overlapping, "
    "combine them into a single bounding box that tightly encloses them. "
    "Consider objects 'close' if the gap between boxes is less than 5% of the image diagonal or if boxes overlap. "
    "Otherwise, keep separate boxes. Prefer fewer, larger boxes over many tiny boxes when objects are contiguous."
)

conversation = [
    {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": grounding_query}
    ]}
]

prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
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

    # Run SAM (Segment Anything) on the predicted boxes to produce masks
    # Convert matches into pixel boxes and labels
    pixel_boxes = []
    labels = []
    for (label, x1_str, y1_str, x2_str, y2_str) in matches:
        x1 = int(x1_str) * image_width // 1000
        y1 = int(y1_str) * image_height // 1000
        x2 = int(x2_str) * image_width // 1000
        y2 = int(y2_str) * image_height // 1000
        # clamp
        x1 = max(0, min(x1, image_width))
        y1 = max(0, min(y1, image_height))
        x2 = max(0, min(x2, image_width))
        y2 = max(0, min(y2, image_height))
        pixel_boxes.append([x1, y1, x2, y2])
        labels.append(label)

    if not _HAS_SAM:
        print("\nSAM (segment-anything) is not installed. To enable mask generation run:")
        print("  pip install git+https://github.com/facebookresearch/segment-anything.git")
        print("Optionally also install: opencv-python pycocotools matplotlib\n")
        print("After installing, re-run this script to generate masks for the predicted boxes.")
    else:
        def run_sam_and_save_masks(pil_image, boxes, labels, checkpoint=None, model_type='vit_l',
                                   multimask_output=True, iou_thresh=0.35, out_dir='sam_masks'):
            """For each predicted bbox: crop, upsample+sharpen, run SAM predictor on the crop,
            resize masks back to original crop size, map to full image, then perform
            greedy IoU NMS and subset filtering. Save masks, overlays and metadata.
            """
            os.makedirs(out_dir, exist_ok=True)

            if checkpoint is None:
                print("SAM checkpoint not provided. Please download a SAM checkpoint (e.g. vit_l) and set 'sam_checkpoint' before running.")
                return []

            # instantiate SAM model once
            sam = sam_model_registry.get(model_type)(checkpoint=checkpoint)
            predictor = SamPredictor(sam)

            all_masks = []  # full-image boolean masks
            all_scores = []
            all_meta = []   # dicts with label and source box idx

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                if w <= 0 or h <= 0:
                    continue

                # Choose an upsample scale: upsample small crops to help SAM
                scale = 2 if max(w, h) < 800 else 1
                w_up = max(1, int(w * scale))
                h_up = max(1, int(h * scale))

                # Crop and upsample
                crop = pil_image.crop((x1, y1, x2, y2)).convert('RGB')
                crop_up = crop.resize((w_up, h_up), resample=Image.LANCZOS)
                # Sharpen using UnsharpMask
                crop_up = crop_up.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))

                crop_np = np.array(crop_up)
                predictor.set_image(crop_np)

                # Ask SAM to segment the full crop (box covering the whole upsampled crop)
                box_for_predict = np.array([0, 0, w_up - 1, h_up - 1])
                try:
                    masks_out, scores_out, logits = predictor.predict(box=box_for_predict, multimask_output=multimask_output)
                except TypeError:
                    out = predictor.predict(box=box_for_predict, multimask_output=multimask_output)
                    if isinstance(out, tuple):
                        masks_out = out[0]
                        scores_out = out[1] if len(out) > 1 else [1.0] * len(masks_out)
                    else:
                        masks_out = out
                        scores_out = [1.0] * len(masks_out)

                # Map each mask back to original-image coordinates
                for m_i, mask_up in enumerate(masks_out):
                    score = float(scores_out[m_i]) if (scores_out is not None and len(scores_out) > m_i) else 1.0
                    # mask_up is H_up x W_up boolean
                    mask_img_up = Image.fromarray((mask_up.astype(np.uint8) * 255).astype(np.uint8))
                    # Resize mask back to original crop size using nearest neighbor
                    mask_resized = mask_img_up.resize((w, h), resample=Image.NEAREST)
                    mask_bool = (np.array(mask_resized) > 0)

                    # Place mask into full-image canvas
                    full_mask = np.zeros((image_height, image_width), dtype=bool)
                    full_mask[y1:y2, x1:x2] = mask_bool

                    all_masks.append(full_mask)
                    all_scores.append(score)
                    all_meta.append({'label': labels[i], 'source_box_idx': int(i)})

            if len(all_masks) == 0:
                print('SAM returned no masks for the given boxes.')
                return []

            def mask_iou(a, b):
                inter = np.logical_and(a, b).sum()
                union = np.logical_or(a, b).sum()
                if union == 0:
                    return 0.0
                return float(inter) / float(union)

            # Greedy IoU NMS (by score)
            order = sorted(range(len(all_masks)), key=lambda i: all_scores[i], reverse=True)
            keep = []
            for idx in order:
                cur = all_masks[idx]
                if all(mask_iou(cur, all_masks[k]) <= iou_thresh for k in keep):
                    keep.append(idx)

            # Remove masks that are subsets of another kept mask (subset threshold)
            subset_thresh = 0.90
            final_keep = []
            for i_idx in keep:
                a = all_masks[i_idx]
                a_area = a.sum()
                if a_area == 0:
                    continue
                is_subset = False
                for j_idx in keep:
                    if i_idx == j_idx:
                        continue
                    inter = np.logical_and(a, all_masks[j_idx]).sum()
                    if inter / float(a_area) >= subset_thresh:
                        is_subset = True
                        break
                if not is_subset:
                    final_keep.append(i_idx)

            # Save masks, overlays, metadata
            overlay_dir = os.path.join(out_dir, 'overlays')
            os.makedirs(overlay_dir, exist_ok=True)

            metadata = []
            combined_overlay = pil_image.convert('RGBA')
            for out_idx, kept_idx in enumerate(final_keep):
                mask = all_masks[kept_idx]
                score = float(all_scores[kept_idx])
                meta = all_meta[kept_idx]

                # Save mask PNG
                mask_img = Image.fromarray((mask.astype(np.uint8) * 255).astype(np.uint8))
                mask_fname = f"mask_{out_idx:03d}.png"
                mask_path = os.path.join(out_dir, mask_fname)
                mask_img.save(mask_path)

                # pick color based on out_idx to vary
                color = tuple(int(colors[out_idx % len(colors)].strip('#')[i:i+2], 16) for i in (0, 2, 4))

                overlay = pil_image.convert('RGBA')
                mask_img_rgb = Image.new('RGBA', overlay.size, (0, 0, 0, 0))
                mask_layer = Image.fromarray((mask.astype(np.uint8) * 255).astype(np.uint8)).convert('L')
                solid = Image.new('RGBA', overlay.size, color + (128,))
                mask_img_rgb.paste(solid, (0, 0), mask_layer)
                # per-mask overlay
                overlay_name = f"overlay_{out_idx:03d}.png"
                overlay_path = os.path.join(overlay_dir, overlay_name)
                composed = Image.alpha_composite(overlay, mask_img_rgb)
                composed.save(overlay_path)

                # add mask to combined overlay
                combined_overlay = Image.alpha_composite(combined_overlay, mask_img_rgb)

                metadata.append({
                    'mask_file': mask_fname,
                    'overlay_file': os.path.join('overlays', overlay_name),
                    'score': score,
                    'label': meta.get('label', ''),
                    'source_box_idx': meta.get('source_box_idx', -1)
                })

            # Save combined overlay
            combined_path = os.path.join(out_dir, 'combined_overlay.png')
            combined_overlay.save(combined_path)

            # Write metadata
            with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"Saved {len(metadata)} masks (+overlays) to {out_dir}")
            return metadata

    run_sam_and_save_masks(image, pixel_boxes, labels, checkpoint=sam_checkpoint, model_type='vit_l', multimask_output=True,
                   iou_thresh=0.35, out_dir='sam_masks')
else:
    print("\nNo valid bounding boxes were extracted. The model may not support grounding in this way.")
    print("Recommendation: Use a dedicated object detector (GroundingDINO, YOLO) for reliable bbox outputs.")
