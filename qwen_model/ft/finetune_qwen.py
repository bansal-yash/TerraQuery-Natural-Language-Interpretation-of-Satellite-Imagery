#!/usr/bin/env python3
"""
Fine-tune Qwen3-VL model for grounding with custom loss using LoRA/PEFT.

Custom loss components:
1. IoU loss: Penalizes low IoU between predicted and ground truth boxes
2. False positive loss: Penalizes predicted boxes that don't match any GT
3. False negative loss: Penalizes GT boxes that have no matching predictions

Usage:
    python ft/finetune_qwen.py --train-json train.json --img-root Images_train/ --output-dir ft/checkpoints
    
Note: Requires unsloth library for efficient LoRA training:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"



"""
import argparse
import json
import os
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from tqdm import tqdm
import numpy as np
from unsloth import FastVisionModel# Import unsloth for efficient LoRA fine-tuning
import sys
    # print("Warning: unsloth not available. Install with:")
    # print('pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
    # print("Falling back to standard transformers (will use more memory)")
    # from transformers import Qwen3VLForConditionalGeneration
    # UNSLOTH_AVAILABLE = False


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes in [x1, y1, x2, y2] format (normalized 0-1).
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def parse_boxes_from_text(text):
    """
    Parse bounding boxes from model output or dataset string.

    Supports two formats:
      1) <ref>label</ref><box>(x1,y1),(x2,y2)</box>  (existing training format)
      2) [x1,y1,x2,y2;x1,y1,x2,y2;...]                (your SARV2 JSON assistant/gpt output)

    Returns: list of tuples. For format (1): (label, x1, y1, x2, y2) where coords are STRINGS (keeps compatibility).
             For format (2): returns list of ('object', x1, y1, x2, y2) where coords are floats (0-1).
    """
    # Try format (1) first (keeps backward compatibility)
    box_pattern = r'<ref>(.*?)</ref><box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
    matches = re.findall(box_pattern, text)
    if matches:
        return matches

    # Try bracketed format: find first [...] group and parse floats separated by semicolons
    br_pattern = r'\[([0-9eE\.\-,;\s]+)\]'
    m = re.search(br_pattern, text)
    if not m:
        return []

    inner = m.group(1).strip()
    # Boxes separated by semicolons (some entries may have trailing ; or spaces)
    raw_boxes = [b.strip() for b in inner.split(';') if b.strip()]
    parsed = []
    for rb in raw_boxes:
        # Expect format x1,y1,x2,y2  (floating point)
        parts = [p.strip() for p in rb.split(',') if p.strip()]
        if len(parts) != 4:
            continue
        try:
            x1, y1, x2, y2 = map(float, parts)
            # Use label 'object' as placeholder (we'll set actual class elsewhere)
            parsed.append(('object', x1, y1, x2, y2))
        except Exception:
            continue
    return parsed



def normalize_boxes_to_01(boxes_1000, image_w, image_h):
    """
    Convert boxes from 0-1000 range to 0-1 normalized by actual image dimensions.
    boxes_1000: list of (label, x1, y1, x2, y2) where coords are in 0-1000 range
    Returns: list of [x1, y1, x2, y2] in 0-1 range
    """
    normalized = []
    for (label, x1s, y1s, x2s, y2s) in boxes_1000:
        x1 = int(x1s) * image_w / 1000.0
        y1 = int(y1s) * image_h / 1000.0
        x2 = int(x2s) * image_w / 1000.0
        y2 = int(y2s) * image_h / 1000.0
        normalized.append([x1 / image_w, y1 / image_h, x2 / image_w, y2 / image_h])
    return normalized


def match_predictions_to_gt(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Match predicted boxes to ground truth boxes using Hungarian matching based on IoU.
    
    Returns:
        matched_pairs: List of (pred_idx, gt_idx, iou) tuples
        unmatched_preds: List of pred indices with no GT match (false positives)
        unmatched_gts: List of GT indices with no pred match (false negatives)
    """
    if len(pred_boxes) == 0:
        return [], [], list(range(len(gt_boxes)))
    if len(gt_boxes) == 0:
        return [], list(range(len(pred_boxes))), []
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou_matrix[i, j] = calculate_iou(pred_box, gt_box)
    
    # Greedy matching (can be replaced with Hungarian algorithm for better results)
    matched_pairs = []
    matched_preds = set()
    matched_gts = set()
    
    # Sort by IoU descending
    pairs = []
    for i in range(len(pred_boxes)):
        for j in range(len(gt_boxes)):
            if iou_matrix[i, j] >= iou_threshold:
                pairs.append((i, j, iou_matrix[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    for pred_idx, gt_idx, iou in pairs:
        if pred_idx not in matched_preds and gt_idx not in matched_gts:
            matched_pairs.append((pred_idx, gt_idx, iou))
            matched_preds.add(pred_idx)
            matched_gts.add(gt_idx)
    
    unmatched_preds = [i for i in range(len(pred_boxes)) if i not in matched_preds]
    unmatched_gts = [j for j in range(len(gt_boxes)) if j not in matched_gts]
    
    return matched_pairs, unmatched_preds, unmatched_gts


class GroundingDataset(Dataset):
    """Dataset for grounding task with bounding boxes."""
    
    def __init__(self, json_path, img_root, processor):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.img_root = img_root
        self.processor = processor
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        # image path may be either relative (to img_root) or already absolute
        img_rel = entry.get('image', '')
        # 1) if it's already an absolute path, use it
        if os.path.isabs(img_rel):
            img_path = img_rel
        else:
            # 2) candidate: join img_root + img_rel (covers the common case where img_rel is relative path under img_root)
            cand1 = os.path.normpath(os.path.join(self.img_root, img_rel))
            if os.path.exists(cand1):
                img_path = cand1
            else:
                # 3) candidate: join img_root + basename(image) (covers cases where json contains full repo-relative path)
                fname = os.path.basename(img_rel)
                cand2 = os.path.normpath(os.path.join(self.img_root, fname))
                if os.path.exists(cand2):
                    img_path = cand2
                else:
                    # 4) last resort: try img_rel as-is (in case img_rel is already a usable relative path w.r.t. CWD)
                    cand3 = os.path.normpath(img_rel)
                    if os.path.exists(cand3):
                        img_path = cand3
                    else:
                        # Provide helpful debug info and raise
                        tried = [cand1, cand2, cand3, img_rel]
                        raise FileNotFoundError(
                            f"Image not found for entry index {idx}. Tried paths:\n" +
                            "\n".join(tried)
                        )

        # Open image once resolved
        image = Image.open(img_path).convert('RGB')
        img_w, img_h = image.size

        # Extract assistant/gpt response from conversations
        convs = entry.get('conversations', [])
        assistant_text = ""
        for c in convs[::-1]:  # search from the end (assistant reply likely last)
            if isinstance(c, dict) and c.get('from', '').lower() in ('gpt', 'assistant'):
                assistant_text = c.get('value', '') or c.get('text', '')
                break
            # older formats may have nested dict with 'value'
            if isinstance(c, dict) and 'value' in c:
                if c.get('from', '').lower() in ('gpt', 'assistant'):
                    assistant_text = c['value']
                    break

        # Parse boxes using the new parser
        parsed = parse_boxes_from_text(assistant_text)

        # Build gt_boxes and gt_classes in normalized 0-1 coords
        gt_boxes = []
        gt_classes = []
        for p in parsed:
            # parsed entries from bracket format are ('object', x1, y1, x2, y2) floats in 0-1
            try:
                label, x1, y1, x2, y2 = p
                # If coords are strings (case of format (1) earlier), convert to int then normalize by image dims
                if isinstance(x1, str):
                    # original format used 0-1000 integers
                    x1n = float(x1) / 1000.0
                    y1n = float(y1) / 1000.0
                    x2n = float(x2) / 1000.0
                    y2n = float(y2) / 1000.0
                else:
                    x1n, y1n, x2n, y2n = float(x1), float(y1), float(x2), float(y2)
                # clamp to [0,1]
                x1n, y1n, x2n, y2n = max(0.0, x1n), max(0.0, y1n), min(1.0, x2n), min(1.0, y2n)
                gt_boxes.append([x1n, y1n, x2n, y2n])
                # Use domain-specific label if you want — default to 'ship' for SARV2
                gt_classes.append('ship')
            except Exception:
                continue

        # Determine primary class (most common) — likely 'ship'
        if gt_classes:
            from collections import Counter
            primary_class = Counter(gt_classes).most_common(1)[0][0]
        else:
            primary_class = 'object'

        # Build system instruction and expected_output in the <ref>..</ref><box> format (0-1000)
        system_instruction = (
            "When multiple objects of the requested class are adjacent or overlapping, "
            "combine them into a single bounding box that tightly encloses them. "
            "Only segment the requested object class directly and completely visible in the image."
        )

        expected_output = ""
        for obj_class, bbox in zip(gt_classes, gt_boxes):
            if obj_class == primary_class:
                x1 = int(bbox[0] * 1000)
                y1 = int(bbox[1] * 1000)
                x2 = int(bbox[2] * 1000)
                y2 = int(bbox[3] * 1000)
                expected_output += f"<ref>{obj_class}</ref><box>({x1},{y1}),({x2},{y2})</box>"

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"Locate and output bounding boxes for all {primary_class} in the image. Format: <ref>label</ref><box>(x1,y1),(x2,y2)</box>"}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": expected_output}]}
        ]

        # Process with tokenizer/processor
        prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(text=prompt, images=[image], return_tensors='pt')

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': inputs['pixel_values'].squeeze(0) if 'pixel_values' in inputs else None,
            'image_grid_thw': inputs['image_grid_thw'].squeeze(0) if 'image_grid_thw' in inputs else None,
            'labels': inputs['input_ids'].squeeze(0),
            'gt_boxes': gt_boxes,
            'gt_classes': gt_classes,
            'primary_class': primary_class,
            'image_path': img_path,
            'image_size': image.size,
        }



def collate_fn(batch):
    """Custom collate function to handle variable-length sequences and metadata."""
    # Pad sequences
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Find max length
    max_len = max(seq.size(0) for seq in input_ids)
    
    # Pad
    input_ids_padded = torch.stack([
        torch.cat([seq, torch.zeros(max_len - seq.size(0), dtype=seq.dtype)]) 
        for seq in input_ids
    ])
    attention_mask_padded = torch.stack([
        torch.cat([seq, torch.zeros(max_len - seq.size(0), dtype=seq.dtype)]) 
        for seq in attention_mask
    ])
    labels_padded = torch.stack([
        torch.cat([seq, torch.full((max_len - seq.size(0),), -100, dtype=seq.dtype)]) 
        for seq in labels
    ])
    
    # Stack pixel values if present
    pixel_values = None
    image_grid_thw = None
    if batch[0]['pixel_values'] is not None:
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
    if batch[0]['image_grid_thw'] is not None:
        image_grid_thw = torch.stack([item['image_grid_thw'] for item in batch])
    
    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'labels': labels_padded,
        'pixel_values': pixel_values,
        'image_grid_thw': image_grid_thw,
        'gt_boxes': [item['gt_boxes'] for item in batch],
        'gt_classes': [item['gt_classes'] for item in batch],
        'primary_class': [item['primary_class'] for item in batch],
        'image_path': [item['image_path'] for item in batch],
        'image_size': [item['image_size'] for item in batch],
    }


class GroundingTrainer(Trainer):
    """Custom trainer with grounding-specific loss matching evaluation criteria."""
    
    def __init__(self, *args, iou_loss_weight=1.0, count_penalty_alpha=1.0, use_grounding_loss=True, grounding_loss_every_n_steps=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.iou_loss_weight = iou_loss_weight
        self.count_penalty_alpha = count_penalty_alpha  # α parameter for count penalty
        self.use_grounding_loss = use_grounding_loss
        self.grounding_loss_every_n_steps = grounding_loss_every_n_steps
        self.step_count = 0
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute combined loss: standard LM loss + grounding loss
        
        Grounding loss matches evaluation metric:
        S_grounding = CP × MeanIoU
        where CP = exp(-α|N_pred - N_ref|)
        
        Loss = -log(S_grounding) = -log(CP) - log(MeanIoU)
              = α|N_pred - N_ref| - log(MeanIoU)
        """
        # Extract metadata
        gt_boxes = inputs.pop('gt_boxes', None)
        gt_classes = inputs.pop('gt_classes', None)
        primary_class = inputs.pop('primary_class', None)
        image_path = inputs.pop('image_path', None)
        image_size = inputs.pop('image_size', None)
        
        # Ensure floating inputs match model parameter dtype/device to avoid mixed-dtype matmuls
        # Determine the majority dtype/device among model parameters. If parameters
        # use mixed dtypes (e.g., some float32 and some bfloat16), prefer float32
        # as a safe fallback to avoid mixed-dtype matmuls in compiled kernels.
        param_dtype_counts = {}
        target_device = None
        param_iter = model.parameters()
        try:
            # Iterate up to a reasonable number of params to build dtype histogram
            for i, p in enumerate(param_iter):
                if i == 0:
                    target_device = p.device
                d = p.dtype
                param_dtype_counts[d] = param_dtype_counts.get(d, 0) + 1
                if i >= 200:
                    break
        except Exception:
            param_dtype_counts = {}

        if param_dtype_counts:
            # Pick the most common dtype (true majority). Do NOT force float32 when
            # the model actually has more bfloat16 parameters — that causes input
            # casts to the wrong dtype and triggers matmul dtype errors.
            majority_dtype = max(param_dtype_counts.items(), key=lambda x: x[1])[0]
        else:
            majority_dtype = None

        if majority_dtype is not None:
            for k, v in list(inputs.items()):
                # Only convert floating tensors (e.g., pixel_values, image_grid_thw)
                if isinstance(v, torch.Tensor) and torch.is_floating_point(v) and v.dtype != majority_dtype:
                    inputs[k] = v.to(device=target_device, dtype=majority_dtype)

        # Optional debug: print param dtype histogram and input tensor dtypes/shapes to diagnose issues
        # Debug printing removed for normal runs. Set DEBUG_COMPUTE_LOSS=1 earlier if
        # you need to re-enable diagnostic prints during debugging.

        # Standard forward pass for language modeling loss. If a dtype mismatch error
        # occurs in compiled kernels, attempt one retry with the alternate float dtype.
        try:
            outputs = model(**inputs)
        except RuntimeError as e:
            msg = str(e)
            if 'mat1 and mat2 must have the same dtype' in msg or 'expected' in msg and 'dtype' in msg:
                # Try the opposite float dtype once (float32 <-> bfloat16)
                alt_dtype = None
                if majority_dtype is not None:
                    if majority_dtype == torch.bfloat16:
                        alt_dtype = torch.float32
                    else:
                        alt_dtype = torch.bfloat16
                else:
                    alt_dtype = torch.bfloat16

                # Debug prints disabled in normal runs; to enable diagnostics set DEBUG_COMPUTE_LOSS=1

                for k, v in list(inputs.items()):
                    if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                        inputs[k] = v.to(device=target_device, dtype=alt_dtype)

                # Retry once
                outputs = model(**inputs)
            else:
                raise
        lm_loss = outputs.loss
        
        # Generate predictions to compute grounding loss
        # Only compute grounding loss every N steps to save time
        grounding_loss = torch.tensor(0.0, device=lm_loss.device)
        
        self.step_count += 1
        should_compute_grounding = (
            self.use_grounding_loss and 
            self.iou_loss_weight > 0 and 
            (self.step_count % self.grounding_loss_every_n_steps == 0)
        )
        
        if should_compute_grounding:
            # Generate outputs for grounding loss computation
            with torch.no_grad():
                gen_outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    pixel_values=inputs.get('pixel_values'),
                    image_grid_thw=inputs.get('image_grid_thw'),
                    max_new_tokens=256,
                    do_sample=False,
                )
            
            # Compute grounding losses for each sample in batch
            batch_size = len(gt_boxes) if gt_boxes else 0
            grounding_scores = []
            
            for i in range(batch_size):
                if not gt_boxes[i]:
                    continue
                
                # Decode generated output
                generated_ids = gen_outputs[i]
                response_text = self.processing_class.decode(generated_ids, skip_special_tokens=False)
                
                # Parse predicted boxes
                matches = parse_boxes_from_text(response_text)
                if image_size[i]:
                    img_w, img_h = image_size[i]
                    pred_boxes = normalize_boxes_to_01(matches, img_w, img_h)
                else:
                    pred_boxes = []
                
                # Filter GT boxes by primary class
                filtered_gt_boxes = [
                    bbox for bbox, cls in zip(gt_boxes[i], gt_classes[i])
                    if cls == primary_class[i]
                ]
                
                if not filtered_gt_boxes:
                    continue
                
                # Compute count penalty: CP = exp(-α|N_pred - N_ref|)
                n_pred = len(pred_boxes)
                n_ref = len(filtered_gt_boxes)
                count_penalty = torch.exp(torch.tensor(-self.count_penalty_alpha * abs(n_pred - n_ref), dtype=torch.float32))
                
                # Match predictions to GT
                matched_pairs, _, _ = match_predictions_to_gt(pred_boxes, filtered_gt_boxes)
                
                # Compute MeanIoU
                if matched_pairs and len(matched_pairs) > 0:
                    mean_iou = sum(iou for _, _, iou in matched_pairs) / len(matched_pairs)
                else:
                    # No matches: IoU = 0 (but add small epsilon to avoid log(0))
                    mean_iou = 1e-6
                
                # Grounding score: S = CP × MeanIoU
                grounding_score = count_penalty * mean_iou
                grounding_scores.append(grounding_score)
            
            # Aggregate loss: -log(S_grounding)
            # This makes the model maximize S_grounding
            if grounding_scores:
                avg_grounding_score = sum(grounding_scores) / len(grounding_scores)
                # Negative log likelihood: want to maximize score = minimize -log(score)
                grounding_loss = -torch.log(torch.clamp(avg_grounding_score, min=1e-6))
                grounding_loss = grounding_loss.to(device=lm_loss.device)
                grounding_loss = self.iou_loss_weight * grounding_loss
        
        # Combined loss
        total_loss = lm_loss + grounding_loss
        
        return (total_loss, outputs) if return_outputs else total_loss


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Qwen3-VL for grounding with LoRA')
    parser.add_argument('--train-json', required=True, help='Path to training JSON file')
    parser.add_argument('--img-root', required=True, help='Root directory containing training images')
    parser.add_argument('--model', default='unsloth/Qwen3-VL-8B-Instruct', help='Base model to fine-tune')
    parser.add_argument('--output-dir', default='ft/checkpoints', help='Output directory for checkpoints')
    parser.add_argument('--batch-size', type=int, default=4, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--iou-loss-weight', type=float, default=1.0, help='Weight for grounding loss (based on IoU and count penalty)')
    parser.add_argument('--count-penalty-alpha', type=float, default=1.0, help='Alpha parameter for count penalty: exp(-α|N_pred - N_ref|)')
    parser.add_argument('--use-grounding-loss', action='store_true', default=True, help='Use grounding loss in addition to LM loss')
    parser.add_argument('--grounding-loss-every-n-steps', type=int, default=5, help='Compute grounding loss every N steps (expensive)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--save-steps', type=int, default=100, help='Save checkpoint every N steps')
    parser.add_argument('--logging-steps', type=int, default=10, help='Log every N steps')
    
    # LoRA configuration arguments
    parser.add_argument('--use-lora', action='store_true', default=True, help='Use LoRA for efficient fine-tuning')
    parser.add_argument('--load-in-4bit', action='store_true', default=False, help='Use 4-bit quantization')
    parser.add_argument('--lora-r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.0, help='LoRA dropout')
    parser.add_argument('--finetune-vision-layers', action='store_true', default=False, 
                       help='Fine-tune vision layers (not recommended for grounding)')
    parser.add_argument('--finetune-language-layers', action='store_true', default=True,
                       help='Fine-tune language layers')
    parser.add_argument('--finetune-attention-modules', action='store_true', default=False,
                       help='Fine-tune attention modules')
    parser.add_argument('--finetune-mlp-modules', action='store_true', default=True,
                       help='Fine-tune MLP modules')
    
    args = parser.parse_args()
    
    # Disable torch.compile for vision layers to avoid dtype mismatch in compiled kernels
    # The vision encoder has mixed float32/bfloat16 params that cause issues with inductor
    os.environ['TORCH_COMPILE_DISABLE'] = '1'
    
    # Load model and processor with LoRA
    print(f'Loading model: {args.model}')
    
    # if UNSLOTH_AVAILABLE and args.use_lora:
    print("Using unsloth FastVisionModel with LoRA for efficient training")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )
    
    # Apply LoRA/PEFT adapters
    print(f"Applying LoRA with r={args.lora_r}, alpha={args.lora_alpha}")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=args.finetune_vision_layers,
        finetune_language_layers=args.finetune_language_layers,
        finetune_attention_modules=args.finetune_attention_modules,
        finetune_mlp_modules=args.finetune_mlp_modules,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Enable training mode
    FastVisionModel.for_training(model)

    # Force the full model to a single dtype (bfloat16) to avoid mixed-dtype math
    # errors inside compiled kernels. L40S supports bfloat16; if you prefer to
    # run in float32, change to torch.float32 (uses more memory).
    try:
        model = model.to(dtype=torch.bfloat16)
    except Exception as _e:
        print('Warning: failed to cast model to bfloat16:', _e)
    
    # Create processor from tokenizer
    processor = tokenizer
        
    # else:
    #     print("Using standard transformers (full fine-tuning - requires more memory)")
    #     from transformers import Qwen3VLForConditionalGeneration
    #     model = Qwen3VLForConditionalGeneration.from_pretrained(
    #         args.model.replace('unsloth/', 'Qwen/'),
    #         device_map='auto',
    #         trust_remote_code=True,
    #     )
    #     processor = AutoProcessor.from_pretrained(
    #         args.model.replace('unsloth/', 'Qwen/'), 
    #         trust_remote_code=True
    #     )
    #     tokenizer = processor.tokenizer
    
    # Print trainable parameters info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Create dataset
    print(f'Loading training data from: {args.train_json}')
    train_dataset = GroundingDataset(args.train_json, args.img_root, processor)
    print(f'Training samples: {len(train_dataset)}')
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=False,  # Disable fp16; we use bf16 for this run
        bf16=torch.cuda.is_available(),
        dataloader_pin_memory=True,
        remove_unused_columns=False,  # Keep metadata
        report_to='none',  # Disable wandb/tensorboard for now
        optim="adamw_8bit" if args.load_in_4bit else "adamw_torch",  # Use 8bit optimizer with 4bit model
    )
    
    # Create trainer
    trainer = GroundingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        processing_class=processor,
        iou_loss_weight=args.iou_loss_weight,
        count_penalty_alpha=args.count_penalty_alpha,
        use_grounding_loss=args.use_grounding_loss,
        grounding_loss_every_n_steps=args.grounding_loss_every_n_steps,
    )
    
    # Train
    print('Starting training...')
    trainer.train()
    
    # Save final model
    final_output_dir = os.path.join(args.output_dir, 'final')
    print(f'Saving final model to: {final_output_dir}')
    
    # if UNSLOTH_AVAILABLE and args.use_lora:
        # Save LoRA adapters
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"Saved LoRA adapters to {final_output_dir}")
    
    # Optionally save merged model (full model with LoRA weights merged)
    merged_output_dir = os.path.join(args.output_dir, 'final_merged')
    print(f"Saving merged model (this may take a while)...")
    model.save_pretrained_merged(merged_output_dir, tokenizer, save_method="merged_16bit")
    print(f"Saved merged 16-bit model to {merged_output_dir}")
    # else:
        # trainer.save_model(final_output_dir)
        # processor.save_pretrained(final_output_dir)
    
    print('Training complete!')
    print(f'Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)')


if __name__ == '__main__':
    main()
