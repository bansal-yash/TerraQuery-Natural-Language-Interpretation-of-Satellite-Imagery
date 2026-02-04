#!/usr/bin/env python3
"""
LM Studio Qwen-VL evaluation script (fixed):
- Uses official LM Studio Python API (no Client() — model = lms.llm(...))
- Reads VRSBench-style JSON
- For each (image, class) pair, runs Qwen-VL detection
- Extracts bounding boxes from <ref>..</ref><box>..</box>
- Computes IoU, histograms, FN/incorrect stats
"""

import os
import json
import argparse
import lmstudio as lms
from PIL import Image
import re
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def compute_iou(boxA, boxB):
    """ IoU for normalized AABBs: [x1,y1,x2,y2] in [0,1] """
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

    if union <= 0:
        return 0.0

    return interArea / union


def parse_boxes(text):
    """
    Extract from model output:
       <ref>class</ref><box>(x1,y1),(x2,y2)</box>
    coordinates are in 0..1000 range.
    """
    pattern = r"<ref>(.*?)</ref><box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>"
    return re.findall(pattern, text)


# ---------------------------------------------------------
# Single-call inference
# ---------------------------------------------------------

def process_single_class(model, image_path, clsname):
    """
    Send one image + one class name to LM Studio (Qwen-VL).
    Returns:
      - list of normalized predicted boxes
      - raw text
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to open image {image_path}: {e}")

    prompt = (
        "You are a precise detection model.\n"
        f"Locate *all* instances of the object class: '{clsname}'.\n"
        "Output ONLY in this format:\n"
        "<ref>label</ref><box>(x1,y1),(x2,y2)</box>\n"
        "Coordinates must be integers in the range [0,1000]."
    )

    try:
        resp = model.respond(prompt, images=[image])
    except Exception as e:
        raise RuntimeError(f"Model failure during respond(): {e}")

    text = resp.content or ""

    matches = parse_boxes(text)
    boxes = []

    for (_, x1, y1, x2, y2) in matches:
        boxes.append([
            float(x1) / 1000.0,
            float(y1) / 1000.0,
            float(x2) / 1000.0,
            float(y2) / 1000.0,
        ])

    return boxes, text


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_json", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--iou_threshold", type=float, default=0.5)
    ap.add_argument("--log_dir", default="eval_logs")
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    print("\n=== Loading LM Studio model ===")
    print(f"Model key: {args.model}")

    # ---- FIXED: Correct official API ---- #
    try:
        model = lms.llm(args.model)
    except Exception as e:
        raise RuntimeError(f"Failed to load LM Studio model '{args.model}': {e}")

    # -------------------------------------- #

    with open(args.gt_json) as f:
        gt_list = json.load(f)

    total_ious = []
    num_false_negatives = 0
    num_incorrect = 0
    prediction_log = {}

    for img_entry in tqdm(gt_list, desc="Images"):
        img_name = img_entry["image"]
        rel_path = img_entry["image_path"]

        img_path = os.path.join(args.image_root, rel_path)
        if not os.path.exists(img_path):
            print(f"\n❌ Missing image file: {img_path}")
            continue

        objects = img_entry["objects"]
        prediction_log[img_name] = []

        # group GT by class
        class_to_gt = {}
        for obj in objects:
            cls = obj["class"]
            class_to_gt.setdefault(cls, []).append(obj["bbox_aabb_computed"])

        # infer per class
        for clsname, gt_boxes in class_to_gt.items():

            try:
                pred_boxes, raw_text = process_single_class(model, img_path, clsname)
            except Exception as e:
                print(f"\n❌ Error inferencing {img_name} class {clsname}: {e}")
                pred_boxes = []
                raw_text = ""

            prediction_log[img_name].append({
                "class": clsname,
                "pred_boxes": pred_boxes,
                "raw_text": raw_text
            })

            # evaluate
            for gt in gt_boxes:
                best_iou = 0.0
                for pb in pred_boxes:
                    best_iou = max(best_iou, compute_iou(gt, pb))

                total_ious.append(best_iou)

                if best_iou < args.iou_threshold:
                    num_false_negatives += 1
                if best_iou < 0.1:
                    num_incorrect += 1

    # ---------------------------------------------
    # Metrics
    # ---------------------------------------------
    mean_iou = float(np.mean(total_ious)) if total_ious else 0.0
    hist, bins = np.histogram(total_ious, bins=10, range=(0, 1))

    metrics = {
        "mean_iou": mean_iou,
        "iou_histogram": hist.tolist(),
        "iou_bins": bins.tolist(),
        "false_negatives": num_false_negatives,
        "incorrect_boxes": num_incorrect,
        "total_gt_objects": len(total_ious)
    }

    with open(os.path.join(args.log_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(args.log_dir, "predictions.json"), "w") as f:
        json.dump(prediction_log, f, indent=2)

    print("\n===== FINAL RESULTS =====")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
