#!/usr/bin/env python3
"""
Qwen3-VL GGUF Evaluation Pipeline
---------------------------------
- Uses local qwen_mtmd bindings (your qwen3vl_api backend)
- For each image + class in GT, run inference with a class-specific prompt
- Parse <ref>...</ref><box>(x1,y1),(x2,y2)</box> format
- Compute IoU, FN, incorrect predictions
- Save metrics + predictions.json
"""

import os
import json
import argparse
import qwen_mtmd
from PIL import Image
import numpy as np
import re
from tqdm import tqdm
import gc
import sys

# ---------------------------------------------------------
# UTILS
# ---------------------------------------------------------

def compute_iou(boxA, boxB):
    """IoU between two normalized AABBs [x1,y1,x2,y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - interArea

    return (interArea / union) if union > 0 else 0.0


BOX_PATTERN = re.compile(
    r"<ref>(.*?)</ref><box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>"
)


def parse_predicted_boxes(text):
    """Extract boxes from Qwen output. Returns list of normalized boxes."""
    matches = BOX_PATTERN.findall(text)
    boxes = []
    for (_, x1, y1, x2, y2) in matches:
        boxes.append([float(x1)/1000, float(y1)/1000, float(x2)/1000, float(y2)/1000])
    return boxes


# ---------------------------------------------------------
# INFERENCE WRAPPER
# ---------------------------------------------------------

# def infer_single(qwen_handle, image_path, clsname, n_batch, max_new_tokens):
#     """
#     Constructs a prompt for detecting a class.
#     Calls qwen_mtmd.infer(handle, image, prompt, ...)
#     Returns predicted boxes + raw model text
#     """
#     prompt = (
#         "You are a precise detection model.\n"
#         f"Locate ALL instances of class '{clsname}'.\n"
#         "Output ONLY in:\n"
#         "<ref>label</ref><box>(x1,y1),(x2,y2)</box>\n"
#         "Coordinates must be in [0,1000]."
#     )

#     raw = qwen_mtmd.infer(qwen_handle, image_path, prompt, n_batch, max_new_tokens)
#     boxes = parse_predicted_boxes(raw)
#     return boxes, raw

def run_quietly(func, *args, **kwargs):
    """
    Executes func(*args, **kwargs) with ALL stderr from C/C++ fully silenced.
    Works with llama.cpp, mtmd, CUDA, and std::cerr.
    """
    # Flush everything first (required!)
    sys.stderr.flush()
    sys.stdout.flush()

    # Open /dev/null for suppressing
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    # Duplicate original fds
    orig_stderr_fd = os.dup(2)

    try:
        # Redirect C-level stderr → devnull
        os.dup2(devnull_fd, 2)

        # Now run the function
        result = func(*args, **kwargs)

    finally:
        # Restore original FD
        os.dup2(orig_stderr_fd, 2)
        os.close(orig_stderr_fd)
        os.close(devnull_fd)

    return result


def infer_single(handle, image_path, prompt, n_batch, max_new_tokens, quiet=True):
    """
    Safe, KV-reset, GPU-friendly, optionally-silent inference.
    """

    # Reset KV-cache cleanly
    qwen_mtmd.reset_context(handle)

    def run():
        return qwen_mtmd.infer(
            handle,
            image_path,
            prompt,
            int(n_batch),
            int(max_new_tokens),
        )

    if quiet:
        out = run_quietly(run)
    else:
        out = run()

    gc.collect()
    boxes = parse_predicted_boxes(out)
    return boxes, out


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--gt_json", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--log_dir", default="eval_logs")
    ap.add_argument("--iou_threshold", type=float, default=0.5)
    ap.add_argument("--n_batch", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    print("[info] Loading Qwen3-VL GGUF model...")
    handle = qwen_mtmd.load(args.model, args.mmproj, -1, args.threads)
    print("[info] Done loading model.")

    with open(args.gt_json, "r") as f:
        gt_list = json.load(f)

    total_ious = []
    false_negatives = 0
    incorrect = 0
    predictions = {}

    print("[info] Starting evaluation...")

    for img_entry in tqdm(gt_list):
        img_name = img_entry["image"]
        rel_path = img_entry["image_path"]
        img_path = os.path.join(args.image_root, rel_path)

        if not os.path.exists(img_path):
            print(f"❌ Missing: {img_path}")
            continue

        predictions[img_name] = []

        # GT boxes per class
        class_to_gt = {}
        for obj in img_entry["objects"]:
            cls = obj["class"]
            box = obj["bbox_aabb_computed"]
            class_to_gt.setdefault(cls, []).append(box)

        # inference per class
        for clsname, gt_boxes in class_to_gt.items():
            pred_boxes, raw_text = infer_single(
                handle,
                img_path,
                clsname,
                args.n_batch,
                args.max_new_tokens
            )

            predictions[img_name].append({
                "class": clsname,
                "pred_boxes": pred_boxes,
                "raw_text": raw_text
            })

            # evaluate IoU
            for gt in gt_boxes:
                best = 0
                for pb in pred_boxes:
                    best = max(best, compute_iou(gt, pb))
                total_ious.append(best)

                if best < args.iou_threshold:
                    false_negatives += 1
                if best < 0.1:
                    incorrect += 1

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------
    mean_iou = float(np.mean(total_ious)) if total_ious else 0.0
    hist, bins = np.histogram(total_ious, bins=10, range=(0, 1))

    results = {
        "mean_iou": mean_iou,
        "iou_histogram": hist.tolist(),
        "iou_bins": bins.tolist(),
        "false_negatives": false_negatives,
        "incorrect_boxes": incorrect,
        "total_objects": len(total_ious),
    }

    # Save logs
    with open(os.path.join(args.log_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(args.log_dir, "predictions.json"), "w") as f:
        json.dump(predictions, f, indent=2)

    print("\n=== FINAL METRICS ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
