#!/usr/bin/env python3
"""
Bounding-box extractor using Qwen3-VL via vLLM.
Fully replaces llama-cpp version (qwen3-vl is NOT supported in llama.cpp).
"""

import argparse
import base64
import json
import os
import re
from collections import Counter
from glob import glob

from PIL import Image
from tqdm import tqdm

from vllm import LLM, SamplingParams


# ---------------------------------------------------------
# Parse "<ref>label</ref><box>(x1,y1),(x2,y2)</box>"
# ---------------------------------------------------------
BOX_PATTERN = r"<ref>(.*?)</ref><box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>"


def parse_boxes(text):
    return re.findall(BOX_PATTERN, text)


# ---------------------------------------------------------
# Build the multimodal Qwen3-VL prompt
# ---------------------------------------------------------
def build_prompt(image_b64, object_name):
    return f"""
<image>{image_b64}</image>

Locate and output bounding boxes for all {object_name} in the image.

Rules:
- If objects overlap or touch, merge into one tight bounding box.
- Only segment objects of the requested class, fully visible.
- Output ONLY in the format:
  <ref>label</ref><box>(x1,y1),(x2,y2)</box>
- Coordinates MUST be integers from 0 to 1000.

Give only the boxes, nothing else.
"""


# ---------------------------------------------------------
# Process a single image through Qwen3-VL
# ---------------------------------------------------------
def run_qwen(llm, image_path, object_name):
    # Read + encode image
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    image_b64 = base64.b64encode(img_bytes).decode()

    prompt = build_prompt(image_b64, object_name)

    out = llm.generate(
        [prompt],
        sampling_params=SamplingParams(
            temperature=0.0,
            max_tokens=256,
        )
    )

    # Extract text
    return out[0].outputs[0].text.strip()


# ---------------------------------------------------------
# Main script
# ---------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Image directory")
    p.add_argument("--gt", default=None, help="Optional GT JSON (for class extraction)")
    p.add_argument("--model", required=True, help="Qwen3-VL model directory (not GGUF)")
    p.add_argument("--out", default="eval/outputs_bbox.json", help="Output JSON path")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Load GT mapping
    gt_map = {}
    if args.gt:
        try:
            with open(args.gt, "r") as f:
                gt_list = json.load(f)
            for entry in gt_list:
                name = entry.get("image")
                classes = [obj.get("class") for obj in entry.get("objects", []) if obj.get("class")]
                gt_map[name] = classes
        except Exception as e:
            print(f"Failed to load GT JSON: {e}")

    # Collect images
    image_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif"):
        image_files.extend(glob(os.path.join(args.root, ext)))
    image_files = sorted(image_files)

    # -----------------------------------------------------
    # Load Qwen3-VL model (vLLM version)
    # -----------------------------------------------------
    print(f"Loading vLLM model from: {args.model}")
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16"
    )

    outputs = {}

    # -----------------------------------------------------
    # Iterate over images
    # -----------------------------------------------------
    for img_path in tqdm(image_files, desc="Images"):
        img_name = os.path.basename(img_path)

        # Decide class
        if img_name in gt_map and len(gt_map[img_name]) > 0:
            object_name = Counter(gt_map[img_name]).most_common(1)[0][0]
        else:
            object_name = "object"

        tqdm.write(f"Processing {img_name} -> object: {object_name}")

        # Run Qwen3-VL
        try:
            text = run_qwen(llm, img_path, object_name)
        except Exception as e:
            tqdm.write(f"Model failure on {img_name}: {e}")
            outputs[img_name] = []
            continue

        # Parse boxes
        boxes_int = parse_boxes(text)
        try:
            img = Image.open(img_path)
            w, h = img.size
        except:
            w = h = 1

        norm_boxes = []
        for (label, x1s, y1s, x2s, y2s) in boxes_int:
            x1 = int(x1s) * w / 1000
            y1 = int(y1s) * h / 1000
            x2 = int(x2s) * w / 1000
            y2 = int(y2s) * h / 1000
            norm_boxes.append([
                x1 / w, y1 / h, x2 / w, y2 / h
            ])

        outputs[img_name] = norm_boxes

        # Save incremental
        tmp = args.out + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(outputs, f, indent=2)
            os.replace(tmp, args.out)
        except Exception as e:
            tqdm.write(f"Warning: failed incremental write for {img_name}: {e}")

    # Final write
    with open(args.out, "w") as f:
        json.dump(outputs, f, indent=2)

    print("DONE. Wrote", args.out)


if __name__ == "__main__":
    main()
