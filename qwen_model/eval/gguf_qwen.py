#!/usr/bin/env python3
"""
Generate outputs_bbox.json for all images using llama-cpp-python chat handler
for Qwen3-VL GGUF models. Supports optional ground-truth JSON to pick object class.
"""

import argparse
import json
import os
import re
from glob import glob
from collections import Counter
from PIL import Image
from tqdm import tqdm

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler  # use for Qwen-VL

def parse_boxes_from_text(text):
    box_pattern = r'<ref>(.*?)</ref><box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
    matches = re.findall(box_pattern, text)
    return matches

def process_image_with_chat(llm, image_url, object_name, image_w, image_h):
    system_instruction = (
        "When multiple objects of the requested class are adjacent or overlapping, "
        "combine them into a single bounding box that tightly encloses them. "
        "Only segment the requested object class directly and completely visible in the image. "
        "Return boxes exactly as: <ref>label</ref><box>(x1,y1),(x2,y2)</box>."
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": f"Locate and output bounding boxes for all {object_name} in the image. Format exactly: <ref>label</ref><box>(x1,y1),(x2,y2)</box>"}
            ]
        }
    ]

    resp = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0.0,
    )

    choice = resp.get("choices", [{}])[0]
    text = ""
    if "message" in choice and isinstance(choice["message"], dict):
        cont = choice["message"].get("content")
        if isinstance(cont, list):
            for part in cont:
                if isinstance(part, dict) and part.get("type") == "text":
                    text += part.get("text", "")
                elif isinstance(part, str):
                    text += part
        elif isinstance(cont, str):
            text = cont
    elif "text" in choice:
        text = choice["text"]
    else:
        text = str(choice)

    matches = parse_boxes_from_text(text)
    boxes = []
    for (label, x1s, y1s, x2s, y2s) in matches:
        x1 = int(x1s) * image_w / 1000.0
        y1 = int(y1s) * image_h / 1000.0
        x2 = int(x2s) * image_w / 1000.0
        y2 = int(y2s) * image_h / 1000.0
        boxes.append([x1 / image_w, y1 / image_h, x2 / image_w, y2 / image_h])

    return boxes

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, help='Root directory containing images')
    p.add_argument('--out', default='eval/outputs_bbox.json', help='Output JSON path')
    p.add_argument('--model', required=True, help='Path to GGUF model file')
    p.add_argument('--mmproj', required=True, help='Path to mmproj GGUF file')
    p.add_argument('--gt', default=None, help='Optional ground-truth JSON to extract object class per image')
    p.add_argument('--host', default='127.0.0.1', help='Host for image HTTP server (for image_url)')
    p.add_argument('--port', default=8000, type=int, help='Port for image HTTP server')
    p.add_argument('--ctx', default=4096, type=int, help='Context length')
    p.add_argument('--n_gpu_layers', default=-1, type=int, help='Number of GPU layers for llama_cpp (set 0 for CPU-only)')
    p.add_argument('--retry-cpu', action='store_true', help='If loading with GPU fails, retry with CPU (n_gpu_layers=0)')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("Creating chat handler with mmproj:", args.mmproj)
    # Ensure mmproj path exists and resolve absolute paths to avoid confusing llama_cpp
    mmproj_path = os.path.abspath(args.mmproj)
    if not os.path.exists(mmproj_path):
        raise FileNotFoundError(f"mmproj file not found: {mmproj_path}. Provide full path or ensure the file exists.")
    chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path, verbose=False)

    # Ensure model path exists and pass absolute path to llama_cpp for clearer errors
    model_path = os.path.abspath(args.model)
    print("Loading GGUF model:", model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"GGUF model file not found: {model_path}. Use the absolute path to the .gguf file.")

    # Wrap Llama initialization to provide a clearer diagnostic on failure
    # First attempt: use user-provided n_gpu_layers
    try:
        llm = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=args.ctx,
            n_gpu_layers=args.n_gpu_layers,
            verbose=False
        )
    except Exception as e:
        first_exc = e
        # If requested, automatically retry with CPU-only to isolate GPU issues
        if args.retry_cpu or args.n_gpu_layers != 0:
            try:
                print('\nInitial GGUF load failed; retrying with CPU-only (n_gpu_layers=0)')
                llm = Llama(
                    model_path=model_path,
                    chat_handler=chat_handler,
                    n_ctx=args.ctx,
                    n_gpu_layers=0,
                    verbose=False
                )
            except Exception as e2:
                # Both attempts failed — raise rich diagnostic including both exceptions
                msg = (
                    f"Failed to load GGUF model from: {model_path}\n"
                    f"First attempt (n_gpu_layers={args.n_gpu_layers}) error: {first_exc}\n"
                    f"Retry attempt (n_gpu_layers=0) error: {e2}\n\n"
                    "Common causes:\n"
                    " - the file is corrupted or incomplete. Try re-downloading or verifying checksum.\n"
                    " - the model quantization/format is unsupported by your installed `llama-cpp-python` / ggml build.\n"
                    " - insufficient permissions to read the file.\n\n"
                    "Suggested checks:\n"
                    " - ls -l <path_to_gguf> and file size; ensure file exists and size seems correct.\n"
                    " - python -c 'import llama_cpp; print(llama_cpp.__version__)' to verify installation.\n"
                    " - Try loading a known-good small GGUF to verify your runtime works.\n"
                    " - If using GPU, ensure your llama_cpp was built with CUDA support compatible with your GPU and drivers.\n"
                )
                raise RuntimeError(msg) from e2
        else:
            # Single attempt and user explicitly asked for CPU off — just report initial failure
            msg = (
                f"Failed to load GGUF model from: {model_path}\n"
                f"Original error: {first_exc}\n\n"
                "Suggested checks: ensure the file exists, is readable, and your llama_cpp supports the format.\n"
            )
            raise RuntimeError(msg) from first_exc

    # Load GT map if provided
    gt_map = {}
    if args.gt:
        try:
            with open(args.gt, 'r') as f:
                gt_list = json.load(f)
            for entry in gt_list:
                name = entry.get('image')
                classes = [obj.get('class') for obj in entry.get('objects', []) if obj.get('class')]
                gt_map[name] = classes
        except Exception as e:
            print('Failed to load GT json', args.gt, 'error:', e)

    image_files = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif'):
        image_files.extend(glob(os.path.join(args.root, ext)))
    image_files = sorted(image_files)

    outputs = {}

    for img_path in tqdm(image_files, desc='Images'):
        img_name = os.path.basename(img_path)
        if img_name in gt_map and len(gt_map[img_name]) > 0:
            object_name = Counter(gt_map[img_name]).most_common(1)[0][0]
        else:
            object_name = 'object'

        image_url = f"http://{args.host}:{args.port}/{img_name}"
        tqdm.write(f'Processing {img_name} -> query object: {object_name} at URL {image_url}')

        try:
            im = Image.open(img_path).convert("RGB")
            w, h = im.size
            boxes = process_image_with_chat(llm, image_url, object_name, w, h)
        except Exception as e:
            tqdm.write(f'Failed for {img_name} error: {e}')
            boxes = []

        outputs[img_name] = boxes

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
