#!/usr/bin/env python3
"""
Prepare EarthMind-Bench dataset for Qwen fine-tuning.

Converts COCO-format annotations to the format expected by the fine-tuning script.
Prioritizes SAR images when available, falls back to RGB.

Usage:
    python qwen/prepare_earthmind_dataset.py \
        --earthmind-root ../EarthMind-Bench \
        --coco-json json/segmentation/instances.json \
        --sar-dir img/test/sar/img \
        --rgb-dir img/test/rgb/img \
        --out-json qwen/train_earthmind.json
"""
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
from PIL import Image

ALLOWED_EXTS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']

def find_image_by_basename(root_dir: Path, basename: str):
    """Find image file by basename, trying common extensions."""
    for ext in ALLOWED_EXTS:
        p = root_dir / (basename + ext)
        if p.exists():
            return str(p)
    # Fallback: scan directory
    if root_dir.exists():
        for f in root_dir.iterdir():
            if f.is_file() and f.stem == basename:
                return str(f)
    return None

def coco_to_train_json(coco_json_path: Path, sar_dir: Path, rgb_dir: Path):
    """Convert COCO annotations to training format."""
    print(f"Loading COCO annotations from: {coco_json_path}")
    
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)
    
    images = {img['id']: img for img in coco.get('images', [])}
    categories = {c['id']: c['name'] for c in coco.get('categories', [])}
    
    # Group annotations by image
    ann_by_image = defaultdict(list)
    for ann in coco.get('annotations', []):
        ann_by_image[ann['image_id']].append(ann)
    
    out_list = []
    skipped = 0
    sar_count = 0
    rgb_count = 0
    
    print(f"Processing {len(images)} images...")
    
    for img_id, img_info in images.items():
        file_name = img_info.get('file_name') or str(img_info.get('id'))
        basename = Path(file_name).stem
        
        # Try SAR first, then RGB
        sar_path = find_image_by_basename(sar_dir, basename)
        rgb_path = find_image_by_basename(rgb_dir, basename)
        
        img_path = None
        is_sar = False
        
        if sar_path:
            img_path = sar_path
            is_sar = True
            sar_count += 1
        elif rgb_path:
            img_path = rgb_path
            rgb_count += 1
        else:
            # Try using original file_name with subdirs
            if (sar_dir / file_name).exists():
                img_path = str(sar_dir / file_name)
                is_sar = True
                sar_count += 1
            elif (rgb_dir / file_name).exists():
                img_path = str(rgb_dir / file_name)
                rgb_count += 1
            else:
                print(f"Warning: image not found for {file_name}, skipping")
                skipped += 1
                continue
        
        # Get image dimensions
        try:
            w = img_info.get('width')
            h = img_info.get('height')
            if not w or not h:
                with Image.open(img_path) as im:
                    w, h = im.size
        except Exception as e:
            print(f"Failed to open {img_path}: {e}, skipping")
            skipped += 1
            continue
        
        # Convert annotations to normalized bboxes
        objects = []
        for ann in ann_by_image.get(img_id, []):
            bbox = ann.get('bbox', [])
            if len(bbox) == 4:
                # COCO format: [x, y, width, height] in pixels
                x, y, bw, bh = bbox
                # Normalize to [0, 1]
                x1 = x / w
                y1 = y / h
                x2 = (x + bw) / w
                y2 = (y + bh) / h
                
                # Clamp to valid range
                x1 = max(0.0, min(1.0, x1))
                y1 = max(0.0, min(1.0, y1))
                x2 = max(0.0, min(1.0, x2))
                y2 = max(0.0, min(1.0, y2))
                
                cat_name = categories.get(ann.get('category_id'), 'object')
                objects.append({
                    "class": cat_name,
                    "bbox_aabb": [x1, y1, x2, y2]
                })
        
        if not objects:
            # Skip images with no objects
            skipped += 1
            continue
        
        out_list.append({
            "image": img_path,  # Use absolute path
            "objects": objects,
            "is_sar": is_sar
        })
    
    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(out_list)}")
    print(f"  SAR images: {sar_count}")
    print(f"  RGB images: {rgb_count}")
    print(f"  Skipped: {skipped}")
    
    return out_list

def main():
    parser = argparse.ArgumentParser(description='Prepare EarthMind-Bench for Qwen fine-tuning')
    parser.add_argument('--earthmind-root', required=True, help='Root of EarthMind-Bench')
    parser.add_argument('--coco-json', required=True, help='Path to COCO instances.json (relative to root)')
    parser.add_argument('--sar-dir', required=True, help='SAR images directory (relative to root)')
    parser.add_argument('--rgb-dir', required=True, help='RGB images directory (relative to root)')
    parser.add_argument('--out-json', required=True, help='Output train.json path')
    
    args = parser.parse_args()
    
    root = Path(args.earthmind_root).resolve()
    
    coco_json_path = root / args.coco_json
    sar_dir = root / args.sar_dir
    rgb_dir = root / args.rgb_dir
    
    if not coco_json_path.exists():
        print(f"Error: COCO JSON not found at {coco_json_path}")
        return
    
    if not sar_dir.exists():
        print(f"Warning: SAR directory not found at {sar_dir}")
    
    if not rgb_dir.exists():
        print(f"Warning: RGB directory not found at {rgb_dir}")
    
    train_list = coco_to_train_json(coco_json_path, sar_dir, rgb_dir)
    
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nWriting {len(train_list)} samples to {out_path}")
    with open(out_path, 'w') as f:
        json.dump(train_list, f, indent=2)
    
    print("Done!")

if __name__ == '__main__':
    main()
