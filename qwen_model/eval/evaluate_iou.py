#!/usr/bin/env python3
"""
Evaluate IoU between predicted boxes (outputs_bbox.json) and ground-truth file.

Produces:
 - eval/iou_stats.json : global mean IoU and histogram counts
 - eval/per_image_results.json : list of per-image entries matching requested schema

Usage:
  python eval/evaluate_iou.py --pred eval/outputs_bbox.json --gt ground_truth.json --images_root /path/to/images

Ground-truth JSON is expected to be a list of entries with 'image' and 'objects' where each object
has 'bbox_aabb' (normalized [xmin,ymin,xmax,ymax]).
"""
import argparse
import json
import os
import math
from collections import Counter


def iou_aabb(a, b):
    # a,b are [xmin,ymin,xmax,ymax] normalized
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    xi1 = max(xa1, xb1)
    yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2)
    yi2 = min(ya2, yb2)
    iw = max(0.0, xi2 - xi1)
    ih = max(0.0, yi2 - yi1)
    inter = iw * ih
    area_a = max(0.0, (xa2 - xa1) * (ya2 - ya1))
    area_b = max(0.0, (xb2 - xb1) * (yb2 - yb1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def choose_position(cx, cy):
    # cx,cy are normalized center coords
    x_pos = ''
    y_pos = ''
    if cx < 0.33:
        x_pos = 'left'
    elif cx > 0.66:
        x_pos = 'right'
    else:
        x_pos = ''

    if cy < 0.33:
        y_pos = 'top'
    elif cy > 0.66:
        y_pos = 'bottom'
    else:
        y_pos = ''

    if x_pos and y_pos:
        return f"{y_pos}-{x_pos}"
    return x_pos or y_pos or ''


def size_category(aabb):
    w = aabb[2] - aabb[0]
    h = aabb[3] - aabb[1]
    area = w * h
    if area < 0.01:
        return 'small'
    if area < 0.05:
        return 'medium'
    return 'large'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pred', required=True, help='Predictions outputs_bbox.json')
    p.add_argument('--gt', required=True, help='Ground-truth JSON file')
    p.add_argument('--images_root', default='.', help='Root path for images (for image_path field)')
    p.add_argument('--out_dir', default='eval', help='Output directory')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.pred, 'r') as f:
        preds = json.load(f)
    with open(args.gt, 'r') as f:
        gts = json.load(f)

    # Index ground truth by image name
    gt_index = {entry['image']: entry for entry in gts}

    all_ious = []
    per_image_results = []

    for img_name, pred_boxes in preds.items():
        gt_entry = gt_index.get(img_name)
        if gt_entry is None:
            print('No ground truth for', img_name)
            continue

        gt_objects = gt_entry.get('objects', [])
        gt_boxes = [obj.get('bbox_aabb') for obj in gt_objects]

        # Matching: greedy by best IoU
        matched = []  # tuples (gt_idx, pred_idx, iou)
        used_preds = set()
        for gi, gbox in enumerate(gt_boxes):
            best_iou = 0.0
            best_pi = None
            for pi, pbox in enumerate(pred_boxes):
                if pi in used_preds:
                    continue
                i = iou_aabb(gbox, pbox)
                if i > best_iou:
                    best_iou = i
                    best_pi = pi
            if best_pi is not None:
                matched.append((gi, best_pi, best_iou))
                used_preds.add(best_pi)
                all_ious.append(best_iou)

        mean_iou = float(sum([m[2] for m in matched]) / len(matched)) if matched else 0.0

        # Build per-image structured output similar to requested format
        image_result = {
            'image': img_name,
            'image_path': os.path.join(args.images_root, img_name),
            'caption': gt_entry.get('caption', ''),
            'objects': []
        }

        # for each gt object create an entry
        for gi, gobj in enumerate(gt_objects):
            gbox = gobj.get('bbox_aabb')
            # find matching pred if any
            matched_pair = next((m for m in matched if m[0] == gi), None)
            if matched_pair:
                _, pred_idx, iou_val = matched_pair
                pbox = preds[img_name][pred_idx]
            else:
                iou_val = 0.0
                pbox = None

            cx = (gbox[0] + gbox[2]) / 2.0
            cy = (gbox[1] + gbox[3]) / 2.0

            obj = {
                'class': gobj.get('class', ''),
                'bbox_aabb': [round(x, 2) for x in gbox],
                'bbox_aabb_computed': [float(x) for x in gbox],
                'obj_id': gi,
                'is_unique': True if sum(1 for other in gt_boxes if iou_aabb(gbox, other) > 0.2) == 1 else False,
                'obj_position': choose_position(cx, cy),
                'obj_size': size_category(gbox)
            }
            image_result['objects'].append(obj)

        per_image_results.append(image_result)

    # Compute global stats
    mean_iou_global = float(sum(all_ious) / len(all_ious)) if all_ious else 0.0
    # histogram with 10 bins
    bins = [0] * 10
    for v in all_ious:
        idx = min(int(v * 10), 9)
        bins[idx] += 1

    stats = {
        'mean_iou': mean_iou_global,
        'num_matches': len(all_ious),
        'histogram_bins': bins
    }

    with open(os.path.join(args.out_dir, 'iou_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    with open(os.path.join(args.out_dir, 'per_image_results.json'), 'w') as f:
        json.dump(per_image_results, f, indent=2)

    print('Wrote stats to', args.out_dir)


if __name__ == '__main__':
    main()
