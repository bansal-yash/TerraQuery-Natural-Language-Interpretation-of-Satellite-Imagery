#!/usr/bin/env python3
"""
tim.py - Fine-tune Qwen3-VL for image captioning with:
  - RGB+SAR automatic horizontal merge (single image passed to model)
  - Thinking-in-Modalities (TiM) precompute + injection
  - Optional LoRA (PEFT) support (if 'peft' installed)
  - Option to load a local model directory (avoid re-download)

Usage examples (one-liners):

# Precompute TiM cache (will run model.generate per sample; needs local model dir)
python tim.py --mode precompute_tim --train-json /path/to/json --image-dir /path/to/img --local-model-dir /path/to/local_model --tim-cache-dir /path/to/tim_cache

# Train using precomputed TiM cache
python tim.py --mode train --train-json /path/to/json --image-dir /path/to/img --local-model-dir /path/to/local_model --output-dir /path/to/out --use-lora --tim-cache-dir /path/to/tim_cache --epochs 8 --batch-size 2

Notes:
 - JSON format expected: list of entries, each with keys: task_type, question, answer, file_name (file_name ends with .json)
 - Image locations (per your setup) are ONLY inside:
     <image-dir>/test/rgb/img/<base>.png
     <image-dir>/test/sar/img/<base>.png
 - The script will convert file_name (.json) -> .png automatically.
"""

import argparse
import json
import os
import hashlib
import pickle
import math
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from transformers import AutoProcessor, TrainingArguments, Trainer
import logging
import traceback

# Optional libs
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

# Qwen model class (transformers) - import lazily inside main
QWEN_CLASS_AVAILABLE = True

# -------------------------
# Helpers
# -------------------------
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()

def merge_rgb_sar_horizontal(rgb_img: Image.Image, sar_img: Image.Image) -> Image.Image:
    rgb_img = rgb_img.convert('RGB')
    sar_img = sar_img.convert('RGB')
    h1, h2 = rgb_img.height, sar_img.height
    if h1 != h2:
        if h1 < h2:
            new_w = int(rgb_img.width * (h2 / h1))
            rgb_img = rgb_img.resize((new_w, h2), resample=Image.BICUBIC)
        else:
            new_w = int(sar_img.width * (h1 / h2))
            sar_img = sar_img.resize((new_w, h1), resample=Image.BICUBIC)
    total_w = rgb_img.width + sar_img.width
    max_h = max(rgb_img.height, sar_img.height)
    merged = Image.new('RGB', (total_w, max_h))
    merged.paste(rgb_img, (0, 0))
    merged.paste(sar_img, (rgb_img.width, 0))
    return merged

# -------------------------
# Dataset
# -------------------------
class CaptionDataset(Dataset):
    def __init__(self, json_path: str, processor, image_dir: str, tim_cache_dir: Optional[str] = None):
        with open(json_path, 'r') as f:
            raw = json.load(f)

        self.processor = processor
        self.image_dir = image_dir
        self.tim_cache_dir = tim_cache_dir
        self.data = []

        def to_png(name: str) -> Optional[str]:
            if not isinstance(name, str):
                return None
            return name[:-5] + '.png' if name.endswith('.json') else name

        def exists_any(base_png: str) -> bool:
            if base_png is None:
                return False
            candidates = [
                os.path.join(self.image_dir, 'test', 'rgb', 'img', base_png),
                os.path.join(self.image_dir, 'test', 'sar', 'img', base_png),
            ]
            return any(os.path.exists(p) for p in candidates)

        missing_caption = 0
        missing_image = 0
        for entry in raw:
            caption = entry.get('answer') or entry.get('caption') or entry.get('captions') or entry.get('text')
            if caption is None:
                missing_caption += 1
                continue
            if isinstance(caption, list) and len(caption) == 0:
                missing_caption += 1
                continue
            file_name = entry.get('file_name') or entry.get('image') or entry.get('file')
            if not file_name:
                missing_image += 1
                continue
            base_png = to_png(file_name)
            if not exists_any(base_png):
                missing_image += 1
                continue
            # normalize caption to single string
            if isinstance(caption, list):
                caption = caption[0] if caption else ""
            entry['_base_png'] = base_png
            entry['_caption_str'] = caption
            self.data.append(entry)

        print(f"CaptionDataset initialized: json_path={json_path}, original_entries={len(raw)}, kept={len(self.data)}, dropped_missing_caption={missing_caption}, dropped_missing_image={missing_image}, image_dir={self.image_dir}")
        if len(self.data) == 0:
            raise RuntimeError("No valid (caption + image) entries found in dataset. Check your JSON and image_dir.")

    def __len__(self):
        return len(self.data)

    def _resolve_paths(self, base_png: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        # returns (rgb_path, sar_path, direct_path)
        rgb_candidates = [
            os.path.join(self.image_dir, 'test', 'rgb', 'img', base_png),
            os.path.join(self.image_dir, 'test', 'rgb', base_png),
        ]
        sar_candidates = [
            os.path.join(self.image_dir, 'test', 'sar', 'img', base_png),
            os.path.join(self.image_dir, 'test', 'sar', base_png),
        ]
        rgb_path = next((p for p in rgb_candidates if os.path.exists(p)), None)
        sar_path = next((p for p in sar_candidates if os.path.exists(p)), None)
        direct_path = None
        for p in [os.path.join(self.image_dir, base_png), base_png]:
            if os.path.exists(p):
                direct_path = p
                break
        return rgb_path, sar_path, direct_path

    def __getitem__(self, idx):
        N = len(self.data)
        start = idx % N
        attempts = 0
        while attempts < N:
            entry = self.data[(start + attempts) % N]
            base_png = entry.get('_base_png')
            rgb_path, sar_path, direct_path = self._resolve_paths(base_png)

            rgb_img = None
            sar_img = None
            if rgb_path:
                try:
                    rgb_img = Image.open(rgb_path).convert('RGB')
                except Exception:
                    rgb_img = None
            if sar_path:
                try:
                    tmp = Image.open(sar_path)
                    if tmp.mode == 'L' or tmp.mode.startswith('I'):
                        tmp = tmp.convert('L')
                        sar_img = ImageOps.colorize(tmp, black="black", white="white").convert('RGB')
                    else:
                        sar_img = tmp.convert('RGB')
                except Exception:
                    sar_img = None
            if not rgb_img and not sar_img and direct_path:
                try:
                    rgb_img = Image.open(direct_path).convert('RGB')
                except Exception:
                    rgb_img = None

            if not rgb_img and not sar_img:
                attempts += 1
                continue

            if rgb_img and sar_img:
                final_image = merge_rgb_sar_horizontal(rgb_img, sar_img)
            elif rgb_img:
                final_image = rgb_img
            else:
                final_image = sar_img

            caption = entry.get('_caption_str', "")
            system_instruction = "You are an assistant that generates detailed, accurate descriptions of satellite and aerial imagery."

            # Load TiM cached string if available
            tim_text = None
            if self.tim_cache_dir:
                key = sha1(base_png)
                cache_file = os.path.join(self.tim_cache_dir, f"{key}.txt")
                if os.path.exists(cache_file):
                    try:
                        tim_text = open(cache_file, 'r', encoding='utf-8').read().strip()
                    except Exception:
                        tim_text = None

            user_text = "Please describe this image in detail."
            if tim_text:
                user_text = f"[THINK_MODALITIES] {tim_text}\n\n{user_text}"

            conversation = [
                {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
                {"role": "user", "content": [
                    {"type": "image", "image": final_image},
                    {"type": "text", "text": user_text}
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": caption}]}
            ]

            prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            try:
                inputs = self.processor(text=prompt, images=[final_image], return_tensors='pt')
            except Exception:
                # fallback: downscale and retry
                try:
                    small = final_image.resize((512, int(512 * final_image.height / final_image.width)), resample=Image.BICUBIC)
                    inputs = self.processor(text=prompt, images=[small], return_tensors='pt')
                except Exception:
                    attempts += 1
                    continue

            labels = inputs['input_ids'].clone()
            conversation_without_assistant = [
                {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
                {"role": "user", "content": [
                    {"type": "image", "image": final_image},
                    {"type": "text", "text": user_text}
                ]},
            ]
            try:
                prefix_prompt = self.processor.apply_chat_template(conversation_without_assistant, tokenize=False, add_generation_prompt=True)
                prefix_inputs = self.processor(text=prefix_prompt, images=[final_image], return_tensors='pt')
                prefix_length = prefix_inputs['input_ids'].shape[1]
                labels[0, :prefix_length] = -100
            except Exception:
                try:
                    labels[0, 0] = -100
                except Exception:
                    pass

            image_path_for_log = rgb_path or sar_path or direct_path
            entry['image_path'] = image_path_for_log
            entry['image_size'] = (final_image.width, final_image.height)

            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'pixel_values': inputs['pixel_values'].squeeze(0) if 'pixel_values' in inputs else None,
                'image_grid_thw': inputs['image_grid_thw'].squeeze(0) if 'image_grid_thw' in inputs and inputs['image_grid_thw'] is not None else None,
                'labels': labels.squeeze(0),
                'image_path': image_path_for_log,
                'image_size': (final_image.width, final_image.height),
                '_base_png': base_png,
            }
        raise RuntimeError("No valid image entries found in dataset. Check JSON / image-dir.")

# -------------------------
# Robust Collate
# -------------------------
def _safe_to_int(x):
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if isinstance(x, (str, bytes)):
        return int(x)
    raise TypeError

def _unwrap_singleton_list(x):
    while isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    return x

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    max_len = max(seq.size(0) for seq in input_ids)
    device = input_ids[0].device if hasattr(input_ids[0], 'device') else torch.device('cpu')

    input_ids_padded = torch.stack([
        torch.cat([seq, torch.full((max_len - seq.size(0),), 0, dtype=seq.dtype).to(device)])
        for seq in input_ids
    ])
    attention_mask_padded = torch.stack([
        torch.cat([seq, torch.full((max_len - seq.size(0),), 0, dtype=seq.dtype).to(device)])
        for seq in attention_mask
    ])
    labels_padded = torch.stack([
        torch.cat([seq, torch.full((max_len - seq.size(0),), -100, dtype=seq.dtype).to(device)])
        for seq in labels
    ])

    # Pixel values handling
    pixel_values = None
    if batch[0]['pixel_values'] is not None:
        try:
            pixel_values = torch.cat([item['pixel_values'] for item in batch], dim=0)
        except Exception:
            pixel_values = torch.stack([item['pixel_values'] for item in batch], dim=0)

    # Build image_grid_thw robustly
    image_grid_rows = []
    for i, item in enumerate(batch):
        ig = item.get('image_grid_thw', None)
        ig_list = None
        if isinstance(ig, torch.Tensor):
            try:
                ig_list = ig.detach().cpu().tolist()
            except Exception:
                ig_list = None
        elif isinstance(ig, (list, tuple)):
            ig_list = list(ig)
        ig_list = _unwrap_singleton_list(ig_list)

        t = h = w = None
        if isinstance(ig_list, (list, tuple)) and len(ig_list) >= 1:
            normalized = []
            for el in ig_list:
                el = _unwrap_singleton_list(el)
                if isinstance(el, (list, tuple)) and len(el) > 0:
                    el = el[0]
                normalized.append(el)
            ig_list = normalized
            try:
                if len(ig_list) >= 3:
                    t = _safe_to_int(ig_list[0]); h = _safe_to_int(ig_list[1]); w = _safe_to_int(ig_list[2])
                elif len(ig_list) == 2:
                    t = _safe_to_int(ig_list[0]); h = _safe_to_int(ig_list[1])
                    if h > 0:
                        w = max(1, t // h)
            except Exception:
                t = h = w = None

        if t is None or h is None or w is None:
            item_pv = item.get('pixel_values', None)
            if isinstance(item_pv, torch.Tensor):
                try:
                    num_tokens = item_pv.shape[0]
                    if t is None:
                        t = int(num_tokens)
                    if h is None:
                        guessed_h = int(round(math.sqrt(t))) if t > 0 else 1
                        h = guessed_h
                    if w is None:
                        w = max(1, t // max(1, h))
                except Exception:
                    pass

        if t is None or h is None or w is None:
            # final fallback
            if t is None: t = 1
            if h is None: h = 1
            if w is None: w = max(1, t // max(1, h))
            print(f"Warning: used fallback image_grid_thw for batch index {i}: ({t},{h},{w})")

        image_grid_rows.append(torch.tensor([t, h, w], dtype=torch.long))

    try:
        image_grid_thw = torch.stack(image_grid_rows, dim=0).to(device)
    except Exception as e:
        print(f"Warning: failed to stack image_grid_thw rows: {e}. Setting image_grid_thw=None")
        image_grid_thw = None

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'labels': labels_padded,
        'pixel_values': pixel_values,
        'image_grid_thw': image_grid_thw,
        'image_path': [item['image_path'] for item in batch],
        'image_size': [item['image_size'] for item in batch],
        '_base_png': [item.get('_base_png') for item in batch],
    }

# -------------------------
# Custom Trainer (minimal additions)
# -------------------------
class CustomTrainer(Trainer):
    def __init__(self, *args, eval_samples=None, processor=None, eval_frequency=100, show_predictions_frequency=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_samples = eval_samples or []
        self.processor = processor
        self.eval_frequency = eval_frequency
        self.show_predictions_frequency = show_predictions_frequency
        self.metrics_history = {'bleu': [], 'bert': [], 'steps': []}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # move tensors to model.device handled by Trainer
        outputs = model(**inputs)
        loss = outputs.loss

        # per-sample logging (light)
        try:
            logits = outputs.logits
            labels = inputs.get('labels')
            if logits is not None and labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                batch_size = shift_logits.size(0)
                vocab_size = shift_logits.size(-1)
                predicted_ids = torch.argmax(shift_logits, dim=-1)
                flat_logits = shift_logits.view(-1, vocab_size)
                flat_labels = shift_labels.view(-1)
                per_token_loss = F.cross_entropy(flat_logits, flat_labels, reduction='none')
                per_token_loss = per_token_loss.view(batch_size, -1)
                for i in range(batch_size):
                    sample_labels = shift_labels[i]
                    sample_preds = predicted_ids[i]
                    valid_mask = (sample_labels != -100)
                    if valid_mask.sum() > 0:
                        sample_loss = per_token_loss[i][valid_mask].mean().item()
                        correct = (sample_preds[valid_mask] == sample_labels[valid_mask]).sum().item()
                        total = valid_mask.sum().item()
                        acc = 100 * correct / total if total > 0 else 0.0
                        print(f"Sample {i} loss: {sample_loss:.4f} | acc: {acc:.1f}% (img: {inputs.get('image_path', ['N/A'])[i]})")
        except Exception:
            pass

        return (loss, outputs) if return_outputs else loss

# -------------------------
# TiM precompute helper
# -------------------------
def precompute_tim(train_json: str, image_dir: str, local_model_dir: str, tim_cache_dir: str, tim_max_tokens: int = 48):
    from transformers import Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(local_model_dir)
    model = Qwen3VLForConditionalGeneration.from_pretrained(local_model_dir, device_map="auto",
                                                           torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model.eval()
    os.makedirs(tim_cache_dir, exist_ok=True)
    with open(train_json, 'r') as f:
        data = json.load(f)
    for entry in data:
        file_name = entry.get('file_name') or entry.get('image')
        if not file_name:
            continue
        base_png = file_name[:-5]+'.png' if file_name.endswith('.json') else file_name
        key = sha1(base_png)
        cache_file = os.path.join(tim_cache_dir, f"{key}.txt")
        if os.path.exists(cache_file):
            continue

        # resolve rgb/sar
        rgb_candidates = [
            os.path.join(image_dir, 'test', 'rgb', 'img', base_png),
            os.path.join(image_dir, 'test', 'rgb', base_png),
        ]
        sar_candidates = [
            os.path.join(image_dir, 'test', 'sar', 'img', base_png),
            os.path.join(image_dir, 'test', 'sar', base_png),
        ]
        rgb_path = next((p for p in rgb_candidates if os.path.exists(p)), None)
        sar_path = next((p for p in sar_candidates if os.path.exists(p)), None)

        texts = []
        for name, pth in [('RGB', rgb_path), ('SAR', sar_path)]:
            if not pth:
                texts.append(f"{name}: none")
                continue
            try:
                img = Image.open(pth).convert('RGB')
                conv = [
                    {"role":"system","content":[{"type":"text","text":"You are an assistant that summarizes image modality content in one short phrase."}]},
                    {"role":"user","content":[{"type":"image","image":img},{"type":"text","text":f"Give a compact, 6-15 word summary of salient features for {name} modality."}]}
                ]
                prompt = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=prompt, images=[img], return_tensors='pt')
                inputs = {k:v.to(model.device) if isinstance(v, torch.Tensor) else v for k,v in inputs.items()}
                out_ids = model.generate(**inputs, max_new_tokens=tim_max_tokens, do_sample=False, num_beams=1)
                gen_text = processor.decode(out_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                texts.append(f"{name}:{gen_text}")
            except Exception:
                texts.append(f"{name}:error")
        combined = " || ".join(texts)
        try:
            open(cache_file, 'w', encoding='utf-8').write(combined)
            print(f"Wrote TiM cache {cache_file}")
        except Exception:
            print(f"Failed to write TiM cache {cache_file}")

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'precompute_tim'], default='train')
    parser.add_argument('--train-json', required=True)
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--model', default='unsloth/Qwen3-VL-8B-Instruct')
    parser.add_argument('--local-model-dir', default=None, help='local model dir to load (recommended)')
    parser.add_argument('--output-dir', default='checkpoints_tim')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4)
    parser.add_argument('--save-steps', type=int, default=50)
    parser.add_argument('--logging-steps', type=int, default=20)
    parser.add_argument('--use-lora', action='store_true', default=False)
    parser.add_argument('--r', type=int, default=16)
    parser.add_argument('--alpha', type=int, default=16)
    parser.add_argument('--tim-cache-dir', default=None)
    parser.add_argument('--tim-max-tokens', type=int, default=48)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true', default=False)
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    os.environ['TORCH_COMPILE_DISABLE'] = '1'

    if args.mode == 'precompute_tim':
        if not args.local_model_dir:
            print("Precompute mode requires --local-model-dir")
            sys.exit(1)
        precompute_tim(args.train_json, args.image_dir, args.local_model_dir, args.tim_cache_dir or 'tim_cache', tim_max_tokens=args.tim_max_tokens)
        return

    # TRAIN mode
    # Load model & processor
    local_dir = args.local_model_dir
    local_exists = bool(local_dir and os.path.isdir(local_dir))
    print(f"Loading model: {args.model} (local dir: {local_dir})")

    # Import Qwen class lazily (transformers)
    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    except Exception as e:
        print("Required transformers Qwen3-VL classes not found in this environment. Install a recent transformers.")
        raise

    processor = None
    model = None

    if local_exists:
        print(f"Loading model and processor from local dir: {local_dir}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(local_dir, device_map="auto",
                                                               torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        processor = AutoProcessor.from_pretrained(local_dir)
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, device_map="auto",
                                                               torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        processor = AutoProcessor.from_pretrained(args.model)

    # gradient checkpointing compatibility
    try:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
            print("Set model.config.use_cache = False to be compatible with gradient checkpointing")
    except Exception:
        pass

    # LoRA (PEFT)
    if args.use_lora:
        if not PEFT_AVAILABLE:
            print("PEFT not available. Install 'peft' to enable LoRA. Continuing without LoRA.")
        else:
            try:
                if args.load_in_4bit:
                    try:
                        model = prepare_model_for_kbit_training(model)
                    except Exception:
                        pass
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                lora_config = LoraConfig(
                    r=args.r,
                    lora_alpha=args.alpha,
                    target_modules=target_modules,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, lora_config)
                print("Applied LoRA")
            except Exception:
                print("LoRA application failed; continuing without it.")
                traceback.print_exc()

    # Parameter summary
    try:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")
    except Exception:
        pass

    # Dataset
    train_dataset = CaptionDataset(args.train_json, processor, args.image_dir, tim_cache_dir=args.tim_cache_dir)
    print(f"Training samples: {len(train_dataset)}")

    # prepare small eval set
    eval_samples = []
    for i in range(min(5, len(train_dataset))):
        try:
            item = train_dataset.data[i]
            base = item.get('_base_png')
            rgb_path, sar_path, direct_path = train_dataset._resolve_paths(base)
            img_path = rgb_path or sar_path or direct_path
            im = Image.open(img_path).convert('RGB')
            caption = item.get('_caption_str', "")
            eval_samples.append({'image': im, 'caption': caption, 'image_path': img_path})
        except Exception:
            pass
    print(f"Prepared {len(eval_samples)} eval samples")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=False,
        bf16=torch.cuda.is_available(),
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to='none',
        optim='adamw_torch',
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        eval_samples=eval_samples,
        processor=processor,
    )

    print("Starting training...")
    try:
        trainer.train()
    except Exception:
        print("Training failed:")
        traceback.print_exc()
        raise

    final_out = os.path.join(args.output_dir, 'final')
    print(f"Saving model to {final_out}")
    try:
        trainer.save_model(final_out)
    except Exception:
        traceback.print_exc()
    try:
        processor.save_pretrained(final_out)
    except Exception:
        traceback.print_exc()

if __name__ == '__main__':
    main()
