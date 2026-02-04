#!/usr/bin/env python3
import json
import os
from pathlib import Path
from typing import List

POSSIBLE_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]

def find_image_for_basename(basename: str, image_dir: str) -> str:
    """
    Try candidate extensions first (basename + ext). If none exist, walk image_dir
    and return the first file whose stem startswith the basename (useful for padded names).
    Returns empty string if not found.
    """
    image_dir = Path(image_dir)
    # try direct candidates
    for ext in POSSIBLE_EXTS:
        cand = image_dir / (basename + ext)
        if cand.exists():
            return str(cand)
    # try pattern match: any file whose stem startswith basename
    for root, _, files in os.walk(image_dir):
        for f in files:
            stem = Path(f).stem
            if stem == basename or stem.startswith(basename):
                return str(Path(root) / f)
    return ""

def normalize_test_json(input_json_path: str, image_dir: str, out_json_path: str, prefer_absolute: bool = False):
    """
    Read input JSON (list of entries), add `image` and `conversations` fields when needed,
    and write to out_json_path.
    """
    with open(input_json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of entries")

    image_dir = str(image_dir)
    missing_images = []
    updated = []

    for i, entry in enumerate(data):
        entry = dict(entry)  # copy
        # 1) Derive a basename from file_name (strip any extension such as .json)
        fname = entry.get("file_name", "") or entry.get("image", "") or entry.get("image_id", "")
        basename = Path(fname).stem if fname else ""

        image_path = ""
        # If entry already contains an 'image' and it exists, keep it
        if entry.get("image"):
            cand = Path(entry["image"])
            if not cand.is_absolute():
                cand_abs = Path(image_dir) / entry["image"]
            else:
                cand_abs = cand
            if cand.exists() or cand_abs.exists():
                image_path = str(cand if cand.exists() else cand_abs)
        # else try basename resolution
        if not image_path and basename:
            image_path = find_image_for_basename(basename, image_dir)

        # As final fallback, if file_name looks like an absolute path to an image, keep it if exists
        if not image_path and fname:
            p = Path(fname)
            if p.exists() and p.suffix.lower() in POSSIBLE_EXTS:
                image_path = str(p)

        if not image_path:
            missing_images.append({"index": i, "file_name": fname})
            # keep the original file_name but set image to fname (dataset will try other heuristics)
            entry["image"] = fname
        else:
            entry["image"] = str(Path(image_path).resolve() if prefer_absolute else os.path.relpath(image_path))

        # 2) Ensure there is a conversations field in the expected format:
        #    [{"from":"human","value": question}, {"from":"gpt","value": answer}]
        convs = entry.get("conversations")
        if not convs:
            question = entry.get("question", "")
            answer = entry.get("answer", "") or entry.get("caption", "")
            convs = []
            if question:
                convs.append({"from": "human", "value": question})
            # place assistant/gpt as final message
            if answer:
                convs.append({"from": "gpt", "value": answer})
            if convs:
                entry["conversations"] = convs

        # 3) Keep backward-compatible direct caption field if present
        if "answer" in entry and "caption" not in entry:
            entry["caption"] = entry["answer"]

        updated.append(entry)

    # Write output
    with open(out_json_path, "w") as f:
        json.dump(updated, f, indent=2)

    # Print report
    print(f"Wrote {len(updated)} entries to {out_json_path}")
    if missing_images:
        print(f"WARNING: {len(missing_images)} entries did not resolve an image automatically. Example(s):")
        for x in missing_images[:10]:
            print(f"  - index {x['index']} file_name: {x['file_name']}")
        print("You should inspect those entries or ensure image filenames in `image_dir` match the file_name basenames.")
    else:
        print("All entries resolved to images successfully.")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Normalize dataset JSON for bbscore.py / QwenBatchDataset")
    p.add_argument("--input-json", required=True, help="Original JSON (list entries like your example)")
    p.add_argument("--image-dir", required=True, help="Directory containing images")
    p.add_argument("--out-json", required=True, help="Path to write normalized JSON")
    p.add_argument("--abs", action="store_true", help="Write absolute image paths instead of relative")
    args = p.parse_args()

    normalize_test_json(args.input_json, args.image_dir, args.out_json, prefer_absolute=args.abs)


