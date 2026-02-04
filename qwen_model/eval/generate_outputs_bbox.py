#!/usr/bin/env python3
"""
Generate outputs_bbox.json for all images in a root directory using the
`qwen_30b` grounding logic. This script loads the Qwen3-VL model once and
processes each image, writing normalized AABB boxes for each image.

Usage:
    python eval/generate_outputs_bbox.py --root /path/to/images --out eval/outputs_bbox.json

Notes:
"""
import argparse
import json
import os
import re
from glob import glob
from PIL import Image
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from collections import Counter


def parse_boxes_from_text(text):
    box_pattern = r'<ref>(.*?)</ref><box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
    matches = re.findall(box_pattern, text)
    return matches


def process_image(model, processor, image_path, object_name):
    image = Image.open(image_path).convert('RGB')
    image_w, image_h = image.size

    system_instruction = (
        "When multiple objects of the requested class are adjacent or overlapping, "
        "combine them into a single bounding box that tightly encloses them. "
        "Only segment the requested object class directly and completely visible in the image."
    )

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"Locate and output bounding boxes for all {object_name} in the image. Format: <ref>label</ref><box>(x1,y1),(x2,y2)</box>"}
        ]}
    ]

    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors='pt').to(model.device)

    with torch.no_grad():
        gen_outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    generated_ids = gen_outputs[0]
    response_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=False)

    matches = parse_boxes_from_text(response_text)
    boxes = []
    for (label, x1s, y1s, x2s, y2s) in matches:
        x1 = int(x1s) * image_w / 1000.0
        y1 = int(y1s) * image_h / 1000.0
        x2 = int(x2s) * image_w / 1000.0
        y2 = int(y2s) * image_h / 1000.0
        # normalize to [0,1]
        boxes.append([x1 / image_w, y1 / image_h, x2 / image_w, y2 / image_h])

    return boxes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, help='Root directory containing images')
    p.add_argument('--out', default='eval/outputs_bbox.json', help='Output JSON path')
    p.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct', help='Model id to use')
    p.add_argument('--gt', default=None, help='Optional ground-truth JSON to extract object class per image')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print('Loading model...', args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, device_map='auto', trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model.eval()

    image_files = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif'):
        image_files.extend(glob(os.path.join(args.root, ext)))
    image_files = sorted(image_files)

    # Load GT map if provided
    gt_map = {}
    if args.gt:
        try:
            with open(args.gt, 'r') as f:
                gt_list = json.load(f)
            # gt_list expected to be list of entries with 'image' and 'objects' each having 'class'
            for entry in gt_list:
                name = entry.get('image')
                classes = [obj.get('class') for obj in entry.get('objects', []) if obj.get('class')]
                gt_map[name] = classes
        except Exception as e:
            print('Failed to load GT json', args.gt, 'error:', e)

    outputs = {}
    for img_path in tqdm(image_files, desc='Images'):
        img_name = os.path.basename(img_path)
        # Determine object name to query
        object_name = 'object'
        if img_name in gt_map and len(gt_map[img_name]) > 0:
            # choose most common class
            object_name = Counter(gt_map[img_name]).most_common(1)[0][0]
        else:
            # fallback single-word guess from filename or default
            object_name = 'bus'

        tqdm.write(f'Processing {img_name} -> query object: {object_name}')
        try:
            boxes = process_image(model, processor, img_path, object_name)
        except Exception as e:
            tqdm.write(f'Failed for {img_name} error: {e}')
            boxes = []

        outputs[img_name] = boxes

        # Write incremental outputs to disk as we go (atomic replace)
        try:
            tmp_path = args.out + '.tmp'
            with open(tmp_path, 'w') as tf:
                json.dump(outputs, tf, indent=2)
            os.replace(tmp_path, args.out)
        except Exception as e:
            tqdm.write(f'Warning: failed to write incremental output for {img_name}: {e}')

    with open(args.out, 'w') as f:
        json.dump(outputs, f, indent=2)
    print('Wrote', args.out)


if __name__ == '__main__':
    main()
