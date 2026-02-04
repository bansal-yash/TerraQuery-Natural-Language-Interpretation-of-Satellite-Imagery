#!/usr/bin/env python3
"""
GeoNLI evaluation script with visualization (updated)
- Skips attribute sub-queries with empty instruction from final weighted score.
- BERT-BLEU (semantic n-gram precision + LP) with N=4
- Grounding metric with Count Penalty * mean IoU (Hungarian matching)
- Binary exact match
- Numeric exp(-alpha * rel_error)
- Final weighted score per Eq (1) but normalized over only active metrics
- Pretty-prints metrics with 4 decimals (JSON output keeps full precision)
- Generates annotated images with GT (green) and predicted (red) bounding boxes

Dependencies:
  pip install torch transformers shapely scipy numpy pillow requests
"""

import os
import json
import math
import requests
from glob import glob
from typing import List, Tuple
from io import BytesIO

import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw, ImageFont

# ---------- Config ----------
GT_DIR = "ground_truths"
EVAL_DIR = "eval_outputs"
OUT_METRICS = "metrics_per_file.json"
PRED_BOXES_DIR = "pred_boxes_visualization"

# Create visualization directory
os.makedirs(PRED_BOXES_DIR, exist_ok=True)

# BERT-BLEU parameters
BERT_BLEU_N = 4
BERT_BLEU_EPS = 1e-8
BERT_BLEU_ALPHA_LP = 0.5  # length penalty severity

# Grounding CP alpha
GROUNDING_ALPHA = 2.5

# Numeric alpha
NUMERIC_ALPHA = 23

# Device for BERT
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- BERT embedding utilities ----------
print("Loading BERT tokenizer & model ... (this may take a moment)")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()
model.to(_device)

# simple in-memory cache for embeddings
_embedding_cache = {}

def get_embedding(text: str) -> torch.Tensor:
    """
    Return mean-pooled, L2-normalized BERT embedding for `text` as a 1D torch.Tensor (cpu).
    Caches results in _embedding_cache.
    """
    if text is None:
        text = ""
    key = text
    if key in _embedding_cache:
        return _embedding_cache[key]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(_device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state  # (1, seq_len, hidden_dim)
    attention_mask = inputs["attention_mask"].unsqueeze(-1).to(embeddings.dtype)  # (1, seq_len, 1)
    sum_embeddings = (embeddings * attention_mask).sum(dim=1)  # (1, hidden_dim)
    denom = attention_mask.sum(dim=1).clamp(min=1e-9)  # (1, 1)
    mean_pooled = sum_embeddings / denom  # (1, hidden_dim)
    normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1).squeeze(0).cpu()
    _embedding_cache[key] = normalized
    return normalized  # 1D cpu tensor

# ---------- BERT-BLEU implementation ----------
def generate_ngrams(tokens: List[str], n: int) -> List[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def semantic_precision_reference_centered(candidate: str, reference: str, n: int) -> float:
    """
    P_n = (1/|R_n|) * sum_{r in R_n} max_{c in C_n} cos(E(c), E(r))
    If |R_n| == 0 -> return 0.0
    If C_n empty -> max cosine for each r is 0.0
    """
    cand_tokens = candidate.split()
    ref_tokens = reference.split()

    cand_ngrams = generate_ngrams(cand_tokens, n)
    ref_ngrams = generate_ngrams(ref_tokens, n)

    if len(ref_ngrams) == 0:
        return 0.0

    # Precompute embeddings (may be cached)
    cand_embs = [get_embedding(c) for c in cand_ngrams]
    ref_embs = [get_embedding(r) for r in ref_ngrams]

    scores = []
    for r_emb in ref_embs:
        if not cand_embs:
            scores.append(0.0)
            continue
        sims = [float(torch.dot(r_emb, c_emb)) for c_emb in cand_embs]
        scores.append(max(sims))
    return float(np.mean(scores)) if scores else 0.0

def bert_bleu(candidate: str, reference: str, N: int = 4, eps: float = 1e-8, alpha_lp: float = 0.5) -> float:
    """
    Compute BERT-BLEU_N:
      - P_n via semantic_precision_reference_centered for n=1..N
      - geometric mean of P_n (smoothed with eps)
      - length penalty LP = exp(-alpha_lp * |Lc - Lr| / Lr)
      - final = LP * geometric_mean
    """
    P = []
    for n in range(1, N+1):
        Pn = semantic_precision_reference_centered(candidate, reference, n)
        P.append(Pn)

    # geometric mean with smoothing
    log_sum = 0.0
    for p in P:
        log_sum += np.log(p + eps)
    geom_mean = float(np.exp((1.0 / N) * log_sum))

    Lc = len(candidate.split())
    Lr = len(reference.split())
    if Lr == 0:
        LP = 0.0
    else:
        LP = float(math.exp(-alpha_lp * (abs(Lc - Lr) / float(Lr))))

    return LP * geom_mean

# ---------- Grounding utilities (Hungarian matching) ----------
def flatten_to_points(obbox) -> List[Tuple[float, float]]:
    """
    obbox assumed [x1,y1,x2,y2,...,x4,y4] normalized coords in 0..1
    """
    if not obbox or len(obbox) < 8:
        return []
    pts = []
    try:
        for i in range(0, 8, 2):
            pts.append((float(obbox[i]), float(obbox[i+1])))
    except Exception:
        return []
    return pts

def polygon_iou(poly1_pts: List[Tuple[float, float]], poly2_pts: List[Tuple[float, float]]) -> float:
    if not poly1_pts or not poly2_pts:
        return 0.0
    p1 = Polygon(poly1_pts)
    p2 = Polygon(poly2_pts)
    if not p1.is_valid or not p2.is_valid:
        return 0.0
    inter = p1.intersection(p2).area
    union = p1.union(p2).area
    if union <= 0:
        return 0.0
    return float(inter / union)

def compute_grounding_metric(gt_grounding_list, pred_grounding_list, alpha=GROUNDING_ALPHA) -> float:
    """
    Compute S_grounding = CP * mean_IoU
    - CP = exp(-alpha * |Npred - Nref|)
    - mean_IoU computed by matching GT <-> pred polygons to maximize total IoU (Hungarian)
    - Unmatched GTs count as IoU=0 (so mean is divided by Nref)
    """
    gt_polys = [(g.get("object-id"), flatten_to_points(g.get("obbox", []))) for g in gt_grounding_list]
    pred_polys = [(p.get("object-id"), flatten_to_points(p.get("obbox", []))) for p in pred_grounding_list]

    gt_polys = [(gid, pts) for gid, pts in gt_polys if pts]
    pred_polys = [(pid, pts) for pid, pts in pred_polys if pts]

    Nref = len(gt_polys)
    Npred = len(pred_polys)

    # Edge cases
    if Nref == 0 and Npred == 0:
        return 1.0
    if Nref == 0:
        # nothing expected but predicted some -> mean IoU 0
        cp = math.exp(-alpha * abs(Npred - Nref))
        return 0.0 * cp

    # build IoU matrix (shape: Nref x Npred)
    iou_matrix = np.zeros((Nref, Npred), dtype=float)
    for i, (_, gpts) in enumerate(gt_polys):
        for j, (_, ppts) in enumerate(pred_polys):
            iou_matrix[i, j] = polygon_iou(gpts, ppts)

    if iou_matrix.size == 0:
        mean_iou = 0.0
    else:
        # Use Hungarian to maximize sum IoU by minimizing -IoU
        cost = -iou_matrix
        gt_idx, pred_idx = linear_sum_assignment(cost)
        matches = []
        for gi, pi in zip(gt_idx, pred_idx):
            if gi < Nref and pi < Npred:
                val = float(iou_matrix[gi, pi])
                if val > 0:
                    matches.append(val)
        total_iou = sum(matches)
        mean_iou = total_iou / Nref

    cp = math.exp(-alpha * abs(Npred - Nref))

    print("iou: ", mean_iou)
    print("cp: ", cp)
    return cp * mean_iou

# ---------- Visualization utilities ----------
def draw_boxes_on_image(img, gt_boxes, pred_boxes, save_path, resize_to=None):
    """
    img: PIL.Image
    gt_boxes: list of dicts with keys 'object-id' and 'obbox' (flat 8 floats)
    pred_boxes: same structure
    save_path: path to save annotated image
    resize_to: (w,h) or None
    """
    if resize_to:
        img = img.resize(resize_to)
    
    img_width, img_height = img.size
    draw = ImageDraw.Draw(img)
    
    # Attempt to load a small truetype font; fallback to default
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    # helper to draw a polygon with label and thicker lines
    def draw_poly(points, outline="red", width=2, label=None):
        # points: list of (x,y) - handle both normalized (0-1) and pixel coordinates
        scaled_points = []
        coords_outside = False
        
        for x, y in points:
            # Check if coordinates are normalized (0-1 range)
            if 0 <= x <= 1 and 0 <= y <= 1:
                scaled_x = x * img_width
                scaled_y = y * img_height
            else:
                # Assume pixel coordinates
                scaled_x = x
                scaled_y = y
            
            # Check if coordinates are outside image bounds
            if scaled_x < 0 or scaled_x > img_width or scaled_y < 0 or scaled_y > img_height:
                coords_outside = True
            
            # Clamp coordinates to image boundaries (draw to nearest edge)
            scaled_x = max(0, min(img_width, scaled_x))
            scaled_y = max(0, min(img_height, scaled_y))
            
            scaled_points.append((scaled_x, scaled_y))
        
        if len(scaled_points) < 2:
            print(f"Warning: Box with label '{label}' has fewer than 2 points, skipping")
            return
        
        if coords_outside:
            print(f"Info: Box '{label}' has coordinates outside bounds - clamped to edges")
        
        # Draw polygon outline with proper width
        try:
            draw.polygon(scaled_points, outline=outline, width=width)
        except Exception as e:
            print(f"Error drawing polygon for {label}: {e}")
            print(f"  Points: {scaled_points}")
            return
        
        # label near first point
        if label is not None:
            lx, ly = scaled_points[0]
            text = str(label)
            
            # Get text size using textbbox with anchor point
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            
            # Draw background rectangle for readability
            padding = 2
            # Ensure text background stays within image
            text_x = max(0, min(img_width - tw - padding*2, lx))
            text_y = max(th + padding*2, min(img_height, ly))
            
            rect = [text_x, text_y - th - padding*2, text_x + tw + padding*2, text_y]
            draw.rectangle(rect, fill=(0, 0, 0, 180))
            draw.text((text_x + padding, text_y - th - padding), text, fill="white", font=font)

    # Draw GT boxes (green)
    print(f"\n=== Drawing boxes for {save_path} ===")
    print(f"Image size: {img_width}x{img_height}")
    print(f"Number of GT boxes: {len(gt_boxes)}")
    print(f"Number of predicted boxes: {len(pred_boxes)}")
    
    for item in gt_boxes:
        ob = item.get("obbox", []) or item.get("obbox_xy", [])
        obj_id = item.get('object-id', '')
        if not ob or len(ob) < 8:
            print(f"Warning: GT box {obj_id} has invalid obbox: {ob}")
            continue
        try:
            pts = [(float(ob[i]), float(ob[i+1])) for i in range(0, 8, 2)]
            print(f"GT box {obj_id} points: {pts}")
            draw_poly(pts, outline=(0, 255, 0), width=3, label=f"GT:{obj_id}")
        except Exception as e:
            print(f"Error drawing GT box {obj_id}: {e}")
            continue
    
    # Draw predicted boxes (red)
    for item in pred_boxes:
        ob = item.get("obbox", []) or item.get("obbox_xy", [])
        obj_id = item.get('object-id', '')
        if not ob or len(ob) < 8:
            print(f"Warning: Pred box {obj_id} has invalid obbox: {ob}")
            continue
        try:
            pts = [(float(ob[i]), float(ob[i+1])) for i in range(0, 8, 2)]
            print(f"Pred box {obj_id} points: {pts}")
            draw_poly(pts, outline=(255, 0, 0), width=3, label=f"P:{obj_id}")
        except Exception as e:
            print(f"Error drawing predicted box {obj_id}: {e}")
            continue
    
    # save
    img.save(save_path)

# ---------- Other metrics ----------
def normalize_text(s):
    if s is None:
        return ""
    return str(s).strip()

def binary_accuracy(gt_resp, pred_resp) -> float:
    g = normalize_text(gt_resp).lower()
    p = normalize_text(pred_resp).lower()
    return 1.0 if g == p and g != "" else 0.0

def numeric_score(gt_resp, pred_resp, alpha=NUMERIC_ALPHA) -> float:
    try:
        x_gt = float(str(gt_resp).strip())
        x_pred = float(str(pred_resp).strip())
    except Exception:
        return 0.0

    if x_gt == 0:
        return 1.0 if x_pred == 0 else 0.0

    diff_ratio = abs(x_pred - x_gt) / abs(x_gt)
    return math.exp(-alpha * diff_ratio)

# ---------- Instruction detection helper ----------
def has_instruction(query_dict):
    """
    Return True if the query dict has a non-empty 'instruction' string (after stripping).
    query_dict may be None or {}
    """
    if not query_dict:
        return False
    instr = query_dict.get("instruction", "")
    return bool(instr and str(instr).strip())

# ---------- Safe BERT wrapper ----------
def safe_bert_blue(candidate: str, reference: str, N=BERT_BLEU_N, eps=BERT_BLEU_EPS, alpha_lp=BERT_BLEU_ALPHA_LP) -> float:
    try:
        return bert_bleu(candidate, reference, N=N, eps=eps, alpha_lp=alpha_lp)
    except Exception as e:
        print(f"bert_bleu error: {e}")
        return 0.0

# ---------- Main evaluation loop ----------
def main():
    metrics_per_file = {}

    gt_files = sorted(glob(os.path.join(GT_DIR, "*.json")))
    for gt_path in gt_files:
        fname = os.path.basename(gt_path)
        eval_path = os.path.join(EVAL_DIR, fname.replace(".json", "_eval.json"))
        if not os.path.exists(eval_path):
            print(f"Skipping {fname}: eval output not found at {eval_path}")
            continue

        with open(gt_path, "r") as f:
            gt = json.load(f)
        with open(eval_path, "r") as f:
            pred = json.load(f)

        # ---- Caption metric (bert_bleu N=4) ----
        gt_caption = normalize_text(gt.get("queries", {}).get("caption_query", {}).get("response", ""))
        pred_caption = normalize_text(pred.get("queries", {}).get("caption_query", {}).get("response", ""))
        caption_score = safe_bert_blue(pred_caption, gt_caption, N=BERT_BLEU_N)

        # ---- Grounding metric (CP * mean IoU) ----
        gt_grounding = gt.get("queries", {}).get("grounding_query", {}).get("response", [])
        pred_grounding = pred.get("queries", {}).get("grounding_query", {}).get("response", [])
        grounding_score = compute_grounding_metric(gt_grounding, pred_grounding, alpha=GROUNDING_ALPHA)

        # ---- Visualize bounding boxes ----
        try:
            image_url = gt.get("input_image", {}).get("image_url", "")
            if image_url:
                # Download image
                r = requests.get(image_url, timeout=30)
                r.raise_for_status()
                pil_img = Image.open(BytesIO(r.content)).convert("RGB")
                
                # Save raw image
                raw_image_path = os.path.join(PRED_BOXES_DIR, fname.replace(".json", "_raw.png"))
                pil_img.save(raw_image_path)
                
                # Draw boxes and save annotated image
                annotated_image_path = os.path.join(PRED_BOXES_DIR, fname.replace(".json", "_annotated.png"))
                draw_boxes_on_image(pil_img, gt_grounding, pred_grounding, annotated_image_path)
                print(f"Saved annotated image: {annotated_image_path}")
        except Exception as e:
            print(f"Failed to create visualization for {fname}: {e}")

        # ---- Attribute queries (instruction-aware) ----
        # We'll compute each attribute metric only if the GT instruction is non-empty.
        attribute_scores = {
            "binary": None,
            "numeric": None,
            "semantic": None
        }
        # store original weight mapping
        attribute_weights = {
            "binary": 0.10,
            "numeric": 0.20,
            "semantic": 0.20
        }

        attr_qs_gt = gt.get("queries", {}).get("attribute_query", {}) or {}
        attr_qs_pred = pred.get("queries", {}).get("attribute_query", {}) or {}

        # Binary
        bin_q_gt = attr_qs_gt.get("binary", {})
        if has_instruction(bin_q_gt):
            gt_bin_resp = bin_q_gt.get("response", "")
            pred_bin_resp = attr_qs_pred.get("binary", {}).get("response", "")
            binary_score = binary_accuracy(gt_bin_resp, pred_bin_resp)
            attribute_scores["binary"] = binary_score
        else:
            binary_score = None

        # Numeric
        num_q_gt = attr_qs_gt.get("numeric", {})
        if has_instruction(num_q_gt):
            gt_num_resp = num_q_gt.get("response", "")
            pred_num_resp = attr_qs_pred.get("numeric", {}).get("response", "")
            numeric_score_val = numeric_score(gt_num_resp, pred_num_resp, alpha=NUMERIC_ALPHA)
            attribute_scores["numeric"] = numeric_score_val
        else:
            numeric_score_val = None

        # Semantic (treat as attribute_bert_blue_4)
        sem_q_gt = attr_qs_gt.get("semantic", {})
        if has_instruction(sem_q_gt):
            gt_sem_resp = normalize_text(sem_q_gt.get("response", ""))
            pred_sem_resp = normalize_text(attr_qs_pred.get("semantic", {}).get("response", ""))
            attribute_score_semantic = safe_bert_blue(pred_sem_resp, gt_sem_resp, N=1)
            attribute_scores["semantic"] = attribute_score_semantic
        else:
            attribute_score_semantic = None

        # ---- Final weighted score (Equation 1, dynamic attribute removal) ----
        # base weights
        active_weights = {
            "caption": 0.20,
            "grounding": 0.30
        }
        active_scores = {
            "caption": caption_score,
            "grounding": grounding_score
        }

        # add attributes only if present
        if attribute_scores["binary"] is not None:
            active_weights["binary"] = attribute_weights["binary"]
            active_scores["binary"] = attribute_scores["binary"]

        if attribute_scores["numeric"] is not None:
            active_weights["numeric"] = attribute_weights["numeric"]
            active_scores["numeric"] = attribute_scores["numeric"]

        if attribute_scores["semantic"] is not None:
            active_weights["semantic"] = attribute_weights["semantic"]
            active_scores["semantic"] = attribute_scores["semantic"]

        total_w = sum(active_weights.values())
        if total_w > 0:
            final_score = sum(active_scores[k] * active_weights[k] for k in active_scores) / total_w
        else:
            final_score = 0.0

        # ---- Prepare metrics dict (store None for missing attributes) ----
        metrics = {
            "caption_bert_blue_4": caption_score,
            "grounding_cp_meanIoU": grounding_score,
            "binary_exact_match": binary_score,
            "numeric_score_exp_neg_abs_diff": numeric_score_val,
            "attribute_bert_blue_4": attribute_score_semantic,
            "final_weighted_score": final_score
        }
        metrics_per_file[fname] = metrics

        # ---- Pretty print (4 decimals) ----
        print(f"File: {fname}")
        for k, v in metrics.items():
            if v is None:
                print(f"  {k}: N/A")
            elif isinstance(v, (int, float)):
                print(f"  {k}: {float(v):.4f}")
            else:
                print(f"  {k}: {v}")
        print("-" * 40)

    # ---- Average metrics across files: compute averages only over defined values ----
    avg_metrics = {}
    if metrics_per_file:
        # collect keys
        all_keys = list(next(iter(metrics_per_file.values())).keys())
        # prepare accumulators and counts
        acc = {k: 0.0 for k in all_keys}
        cnt = {k: 0 for k in all_keys}

        for m in metrics_per_file.values():
            for k in all_keys:
                v = m.get(k, None)
                if v is None:
                    continue
                try:
                    acc[k] += float(v)
                    cnt[k] += 1
                except Exception:
                    continue

        for k in all_keys:
            if cnt[k] > 0:
                avg_metrics[k] = acc[k] / cnt[k]
            else:
                avg_metrics[k] = None

        print("\n=== AVERAGE METRICS ACROSS ALL FILES ===")
        for k, v in avg_metrics.items():
            if v is None:
                print(f"  {k}: N/A")
            else:
                print(f"  {k}: {v:.4f}")
        print("========================================\n")

    # Save results (full precision)
    with open(OUT_METRICS, "w") as f:
        json.dump(metrics_per_file, f, indent=2)
    print(f"Saved metrics for {len(metrics_per_file)} files to {OUT_METRICS}")

    with open("eval_metrics.json", "w") as f:
        json.dump(avg_metrics, f, indent=2)
    print("Saved average metrics to eval_metrics.json")

if __name__ == "__main__":
    main()
