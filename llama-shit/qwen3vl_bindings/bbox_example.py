#!/usr/bin/env python3
# single_image_detect_and_annotate_qwen.py
# Requires: qwen_mtmd (your patched binding), Pillow

import argparse
import re
import json
import os
import sys
from PIL import Image, ImageDraw, ImageFont
import qwen_mtmd

# Robust pattern (allow whitespace)
BOX_PATTERN = re.compile(
    # Accept either the canonical closing tag </box> or the model-produced marker <|box_end|>
    r"<ref>\s*(?P<label>.*?)\s*</ref>\s*<box>\s*\(\s*(?P<x1>\d+)\s*,\s*(?P<y1>\d+)\s*\)\s*,\s*\(\s*(?P<x2>\d+)\s*,\s*(?P<y2>\d+)\s*\)\s*(?:</box>|<\|box_end\|>)",
    flags=re.IGNORECASE
)

# fallback JSON-ish bbox
JSON_BBOX_PATTERN = re.compile(r'\"bbox_2d\"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*,\s*\"label\"\s*:\s*\"([^"]+)\"', re.IGNORECASE)


def parse_boxes_from_text(text):
    """
    Return list of dicts: {"label", "x1","y1","x2","y2"} with coordinates in model space [0,1000].
    """
    boxes = []
    if not text:
        return boxes

    for m in BOX_PATTERN.finditer(text):
        boxes.append({
            "label": m.group("label").strip(),
            "x1": int(m.group("x1")),
            "y1": int(m.group("y1")),
            "x2": int(m.group("x2")),
            "y2": int(m.group("y2"))
        })

    # fallback: JSON blocks
    for m in JSON_BBOX_PATTERN.finditer(text):
        x1, y1, x2, y2, label = m.groups()
        boxes.append({
            "label": label,
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2)
        })

    # If no boxes found, try to detect simple point tags like <points x1="..." y1="..." alt="label">
    pts = re.findall(r'<points\s+[^>]*x1=\"(\d+)\"\s+y1=\"(\d+)\"[^>]*alt=\"([^\"]*)\"', text)
    for x1s, y1s, alt in pts:
        x1 = int(x1s); y1 = int(y1s)
        # make a tiny box (best-effort)
        boxes.append({"label": alt, "x1": x1, "y1": y1, "x2": x1 + 10, "y2": y1 + 10})

    return boxes


def clamp01(v):
    return max(0.0, min(1.0, v))


def model_boxes_to_normalized(boxes):
    """Convert model-space [0,1000] boxes to normalized [0..1] format (x1,y1,x2,y2)."""
    out = []
    for b in boxes:
        x1n = clamp01(b["x1"] / 1000.0)
        y1n = clamp01(b["y1"] / 1000.0)
        x2n = clamp01(b["x2"] / 1000.0)
        y2n = clamp01(b["y2"] / 1000.0)
        out.append({"label": b["label"], "x1": x1n, "y1": y1n, "x2": x2n, "y2": y2n})
    return out


def normalized_to_pixels(box, img_w, img_h):
    x1 = int(round(box["x1"] * img_w))
    y1 = int(round(box["y1"] * img_h))
    x2 = int(round(box["x2"] * img_w))
    y2 = int(round(box["y2"] * img_h))
    # small safety clamp
    x1 = max(0, min(img_w-1, x1))
    x2 = max(0, min(img_w-1, x2))
    y1 = max(0, min(img_h-1, y1))
    y2 = max(0, min(img_h-1, y2))
    return {"label": box["label"], "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def process_image_qwen(handle, image_path, object_name, n_batch=64, max_new_tokens=128, quiet=False):
    """
    Build HF-style conversation with the image path and detection prompt,
    call qwen_mtmd.infer_chat(handle, messages, ...), parse boxes and return normalized boxes + raw text.
    """
    # system instruction for behavior (same as your original)
    system_instruction = (
        "Only segment the requested object class directly and completely visible in the image."
    )

    # Use an HF-style chat message list with the *image path* (not PIL.Image object).
    # Strong, example-driven prompt to encourage listing all boxes (one per line)
    user_text = (
        f"Locate and output bounding boxes for ALL {object_name} in the image. IF THERE ARE MORE THAN ONE INSTANCES OF {object_name}, THEN YOU MUST DETECT THEM ALL."
        "MISSION CRITCAL FORMATTING INSTRUCTIONS: You must give your output in the below format only. Otherwise your output will not be considered." 
        "MISSION CRITCAL INSTRUCTION 1: Return one box per line, using this EXACT format: <ref>label</ref><box>(x1,y1),(x2,y2)</box><|box_end|>. "
        "MISSION CRITICAL INSTRUCTION 2: Coordinates must be integer pixels in the range [0,1000]. Do NOT include any extra commentary or summaries. "
        "If there are no such objects, output exactly: NO_BOX."
        "DEMONSTRATIVE EXAMPLE: If the coordinates you decide are x1=187,y1=567,x2=474,y2=676; then your output should be <ref>label</ref><box>(187,567),(474,676)</box><|box_end|>"
    )

    messages = [
        # {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": f"Describe the locations of the {object_name} using bounding box coordinates (x1,y1,x2,y2) normalised to 1000."}
            # {"type": "text", "text": user_text}
        ]}
    ]

    # Optionally silence C/C++ stderr (if you want), by using a small wrapper; we keep it simple and call directly.
    # If you have a run_quietly helper like earlier, wrap the call with it:
    raw = qwen_mtmd.infer_chat(handle, messages, n_batch, max_new_tokens, do_sample=False)

    parsed = parse_boxes_from_text(raw)
    normalized = model_boxes_to_normalized(parsed)

    return normalized, raw


def draw_and_save_annotation(image_path, boxes_norm, out_path="output_annotated.png", label_font_size=18):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", label_font_size)
    except Exception:
        font = ImageFont.load_default()

    # draw semi-transparent filled rectangle behind labels at bottom if many boxes
    for b in boxes_norm:
        pb = normalized_to_pixels(b, w, h)
        # box style
        draw.rectangle([pb["x1"], pb["y1"], pb["x2"], pb["y2"]], outline=(255, 0, 0, 255), width=2)
        # label background
        label = pb["label"] if pb.get("label") else "obj"
        # robust text-size measurement: prefer draw.textbbox, fall back to older APIs
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            try:
                tw, th = draw.textsize(label, font=font)
            except Exception:
                try:
                    tw, th = font.getsize(label)
                except Exception:
                    tw = int(len(label) * (label_font_size * 0.6))
                    th = label_font_size

        lx = pb["x1"]
        ly = max(0, pb["y1"] - th - 4)
        draw.rectangle([lx, ly, lx + tw + 6, ly + th + 4], fill=(255,0,0,180))
        draw.text((lx+3, ly+2), label, fill=(255,255,255,255), font=font)

    img.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--object", required=True, help="object class name to detect (e.g. 'bridge')")
    ap.add_argument("--n_batch", type=int, default=64)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    print("[info] loading model...")
    handle = qwen_mtmd.load(args.model, args.mmproj, -1, args.threads, True)
    print("[info] model loaded")

    boxes_norm, raw = process_image_qwen(handle, args.image, args.object, args.n_batch, args.max_new_tokens, quiet=False)

    print("[info] raw model output:")
    print(raw[:1000] + ("\n...[truncated]" if len(raw) > 1000 else ""))

    # annotate and save image
    annotated_path = draw_and_save_annotation(args.image, boxes_norm, out_path="output_annotated.png")
    print("[info] saved annotated image:", annotated_path)

    # write boxes (pixel coordinates and normalized)
    img = Image.open(args.image)
    w, h = img.size
    boxes_px = [normalized_to_pixels(b, w, h) for b in boxes_norm]

    out = {
        "image": os.path.basename(args.image),
        "image_size": [w, h],
        "boxes_normalized": boxes_norm,
        "boxes_px": boxes_px,
        "raw_text": raw
    }

    with open("pred_boxes.json", "w") as f:
        json.dump(out, f, indent=2)

    print("[info] saved pred_boxes.json")
    qwen_mtmd.free_handle(handle)


if __name__ == "__main__":
    main()
