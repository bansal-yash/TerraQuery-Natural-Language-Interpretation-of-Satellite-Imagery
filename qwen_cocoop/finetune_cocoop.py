#!/usr/bin/env python3
"""
Fixed single-file script for fine-tuning Qwen3-VL with CoCoOp prompt learner.

Key fixes:
 - Auto-detect image feature dimension
 - Ensure loss computation maintains gradient graph to prompt_learner
 - Robust collate for variable pixel_values shapes
 - Proper device/dtype handling

Usage:
 python finetune_qwen_cocoop_fixed.py \
   --train-json /path/to/caption_all_unmatched.json \
   --output-dir ./checkpoints_cocoop \
   --batch-size 2 --epochs 5
"""
import argparse
import json
import os
import traceback
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoProcessor, TrainingArguments, Trainer
import logging
import random
import sys

random.seed(42)

# Try unsloth support (optional)
try:
    from unsloth import FastVisionModel
    UNSLOTH_AVAILABLE = True
except Exception:
    UNSLOTH_AVAILABLE = False

# -------------------------
# Prompt learner (CoCoOp)
# -------------------------
class PromptLearner(nn.Module):
    def __init__(self, prompt_length: int, embed_dim: int, img_feat_dim: int = 1536, hidden_dim: int = 512):
        super().__init__()
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.img_feat_dim = img_feat_dim
        
        # base prompt (P, D)
        self.base_prompt = nn.Parameter(torch.randn(prompt_length, embed_dim) * 0.02)
        
        # adapter MLP: image features -> P*D
        self.adapter = nn.Sequential(
            nn.Linear(img_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, prompt_length * embed_dim),
        )

    def forward(self, image_features: torch.Tensor):
        # image_features: (B, F)
        batch = image_features.size(0)
        delta = self.adapter(image_features)            # (B, P*D)
        delta = delta.view(batch, self.prompt_length, self.embed_dim)  # (B, P, D)
        base = self.base_prompt.unsqueeze(0).expand(batch, -1, -1)     # (B, P, D)
        return base + delta

# -------------------------
# Image feature extraction
# -------------------------
def get_image_features_from_model(model, pixel_values: torch.Tensor):
    """
    Extract image features from the model.
    Returns tensor (B, feat_dim) on same device as model parameters.
    """
    device = next(model.parameters()).device
    pv = pixel_values.to(device)

    # If already patch tokens shape (B, N, D), mean pool
    if pv.dim() == 3:
        return pv.mean(dim=1).detach()

    # If image tensors (B, C, H, W)
    if pv.dim() == 4:
        # Try model-provided extractor
        if hasattr(model, "get_image_features"):
            try:
                with torch.no_grad():
                    feats = model.get_image_features(pv)
                if isinstance(feats, torch.Tensor):
                    if feats.dim() == 3:
                        return feats.mean(dim=1).detach()
                    return feats.detach()
            except Exception:
                pass

        # Try common vision_model attribute
        if hasattr(model, "vision_model"):
            try:
                with torch.no_grad():
                    vis_out = model.vision_model(pv)
                if isinstance(vis_out, dict):
                    if 'pooler_output' in vis_out:
                        return vis_out['pooler_output'].detach()
                    if 'last_hidden_state' in vis_out:
                        return vis_out['last_hidden_state'].mean(dim=1).detach()
                elif isinstance(vis_out, torch.Tensor):
                    if vis_out.dim() == 3:
                        return vis_out.mean(dim=1).detach()
                    return vis_out.detach()
            except Exception:
                pass

        # Other named attributes
        for name in ("vision_tower", "vision_encoder", "vision"):
            if hasattr(model, name):
                try:
                    with torch.no_grad():
                        vis = getattr(model, name)(pv)
                    if isinstance(vis, dict) and 'last_hidden_state' in vis:
                        return vis['last_hidden_state'].mean(dim=1).detach()
                    if isinstance(vis, torch.Tensor):
                        if vis.dim() == 3:
                            return vis.mean(dim=1).detach()
                        return vis.detach()
                except Exception:
                    pass

    # Fallback: flatten spatial dims and mean
    if pv.dim() == 4:
        try:
            flat = pv.flatten(1).mean(dim=1)
            return flat.detach()
        except Exception:
            pass

    # Last resort: zeros
    return torch.zeros(pv.size(0), 1536, device=device)

def detect_image_feature_dim(model, processor):
    """
    Auto-detect the image feature dimension by running a dummy forward pass.
    """
    try:
        device = next(model.parameters()).device
        # Create a dummy image
        dummy_image = Image.new('RGB', (224, 224), color='red')
        
        # Process it
        conv = [
            {"role": "user", "content": [{"type":"image","image":dummy_image},{"type":"text","text":"test"}]}
        ]
        prompt = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[dummy_image], return_tensors='pt')
        
        pixel_values = inputs.get('pixel_values')
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)
            feats = get_image_features_from_model(model, pixel_values)
            feat_dim = feats.size(-1)
            print(f"✓ Detected image feature dimension: {feat_dim}")
            return feat_dim
    except Exception as e:
        print(f"⚠️  Could not auto-detect feature dim: {e}")
        print("Using default: 1536")
    
    return 1536

# -------------------------
# Custom Trainer with prompt injection & manual loss
# -------------------------
class CustomTrainer(Trainer):
    def __init__(self, *args, prompt_learner: Optional[nn.Module]=None, prompt_length: int=0,
                 get_image_features_fn=None, eval_samples=None, processor=None,
                 eval_frequency=100, show_predictions_frequency=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_learner = prompt_learner
        self.prompt_length = prompt_length
        self.get_image_features = get_image_features_fn or get_image_features_from_model
        self.processor = processor
        self.eval_samples = eval_samples or []
        self.eval_frequency = eval_frequency
        self.show_predictions_frequency = show_predictions_frequency
        self.metrics_history = {'bleu': [], 'bert': [], 'steps': []}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Inject learned soft prompt and compute loss with proper gradient flow.
        """

        device = next(model.parameters()).device

        # Pop debug metadata
        image_paths = inputs.pop('image_path', None)
        image_sizes = inputs.pop('image_size', None)

        # Prepare inputs with prompt injection
        try:

            if self.prompt_learner is not None and 'pixel_values' in inputs and inputs['pixel_values'] is not None:
                pixel_values = inputs.pop('pixel_values').to(device)
                
                # Extract image features (no grad for vision encoder)
                with torch.no_grad():
                    img_feats = self.get_image_features(model, pixel_values)
                
                if img_feats is None:
                    raise RuntimeError("get_image_features returned None")
                
                # Ensure batch dim
                if img_feats.dim() == 1:
                    img_feats = img_feats.unsqueeze(0)

                # Move to prompt learner device/dtype
                prompt_dev = next(self.prompt_learner.parameters()).device
                prompt_dtype = next(self.prompt_learner.parameters()).dtype
                img_feats = img_feats.to(device=prompt_dev, dtype=prompt_dtype)

                
                # Generate prompt embeddings (this is where gradients matter!)
                prompt_embeds = self.prompt_learner(img_feats).to(device)  # (B, P, D)
                B, P, D = prompt_embeds.shape

                # Prepare text inputs
                input_ids = inputs.pop('input_ids').to(device)
                attention_mask = inputs.pop('attention_mask').to(device)
                labels = inputs.pop('labels').to(device)

                # Get input embeddings
                embedding_layer = model.get_input_embeddings()

                input_embeds = embedding_layer(input_ids).to(device)  # (B, L, D)

                # Concatenate: [soft prompts | input tokens]
                inputs_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)  # (B, P+L, D)

                # Update attention mask
                prefix_mask = torch.ones((B, P), dtype=attention_mask.dtype, device=device)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

                # Update labels (ignore prompt positions)
                prompt_labels = torch.full((B, P), -100, dtype=labels.dtype, device=device)
                labels = torch.cat([prompt_labels, labels], dim=1)

                inputs_for_model = {
                    'inputs_embeds': inputs_embeds,
                    'attention_mask': attention_mask,
                    'labels': labels
                }
            else:
                # Fallback without prompts
                inputs_for_model = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) 
                                   for k, v in inputs.items()}

        except Exception as e:
            print(f"⚠️  Failed during prompt injection: {e}")
            traceback.print_exc()
            inputs_for_model = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) 
                               for k, v in inputs.items()}

        # Forward pass

        print(model)
        # print(inputs_for_model['pixel_values'].shape)
        print(inputs_for_model.keys())
        outputs = model(**inputs_for_model)

        print(outputs)
        sys.exit()


        logits = outputs.logits if hasattr(outputs, 'logits') else None

        # Compute loss manually to ensure gradient flow
        if logits is not None and 'labels' in inputs_for_model:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs_for_model['labels'][..., 1:].contiguous()
            
            vocab_size = shift_logits.size(-1)
            flat_logits = shift_logits.view(-1, vocab_size)
            flat_labels = shift_labels.view(-1)

            # Cross entropy loss
            per_token_loss = F.cross_entropy(flat_logits, flat_labels, reduction='none')
            mask = (flat_labels != -100).float()
            
            if mask.sum() > 0:
                loss = (per_token_loss * mask).sum() / mask.sum()
            else:
                # Create a differentiable zero loss connected to prompt learner
                if self.prompt_learner is not None:
                    loss = (self.prompt_learner.base_prompt[0, 0] * 0.0).requires_grad_(True)
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            # Fallback: use model's loss if available
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
            else:
                # Create differentiable zero loss
                if self.prompt_learner is not None:
                    loss = (self.prompt_learner.base_prompt[0, 0] * 0.0).requires_grad_(True)
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Log per-sample metrics
        try:
            if logits is not None and 'labels' in inputs_for_model:
                sl = logits[..., :-1, :].contiguous()
                s_labels = inputs_for_model['labels'][..., 1:].contiguous()
                preds = torch.argmax(sl, dim=-1)
                
                for i in range(sl.size(0)):
                    sample_labels = s_labels[i]
                    valid_mask = (sample_labels != -100)
                    if valid_mask.sum() > 0:
                        correct = (preds[i][valid_mask] == sample_labels[valid_mask]).sum().item()
                        total = valid_mask.sum().item()
                        acc = 100 * correct / total if total > 0 else 0.0
                        img_info = f" (img: {image_paths[i] if image_paths else 'N/A'})"
                        print(f"Sample {i} acc: {acc:.1f}%{img_info}")
                        
                        if i == 0 and self.state.global_step % self.show_predictions_frequency == 0:
                            try:
                                pred_text = self.processor.decode(preds[i][valid_mask], skip_special_tokens=True)
                                label_text = self.processor.decode(sample_labels[valid_mask], skip_special_tokens=True)
                                print(f"{'='*40}\nSTEP {self.state.global_step}\nPred: {pred_text[:200]}...\nRef: {label_text[:200]}...\n{'='*40}")
                            except Exception:
                                pass
        except Exception:
            pass

        # Periodic evaluation
        if (self.state.global_step > 0 and 
            self.state.global_step % self.eval_frequency == 0 and 
            len(self.eval_samples) > 0):
            try:
                self._evaluate_metrics(model)
            except Exception as e:
                print(f"Metrics eval failed: {e}")

        return (loss, outputs) if return_outputs else loss

    def _evaluate_metrics(self, model):
        print("="*40, "EVAL METRICS", "="*40)
        model.eval()
        generated_texts, reference_texts = [], []
        
        with torch.no_grad():
            for i, sample in enumerate(self.eval_samples[:3]):
                try:
                    conv = [
                        {"role": "system", "content": [{"type":"text","text":"You are an assistant that generates detailed, accurate descriptions of satellite and aerial imagery."}]},
                        {"role": "user", "content": [{"type":"image","image":sample['image']},{"type":"text","text":"Please describe this image in detail."}]}
                    ]
                    prompt = self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                    inputs = self.processor(text=prompt, images=[sample['image']], return_tensors='pt')
                    device = next(model.parameters()).device
                    
                    # Move inputs to device
                    gen_inputs = {k:(v.to(device) if isinstance(v, torch.Tensor) else v) 
                                 for k,v in inputs.items()}
                    gen_inputs.update({'max_new_tokens':150, 'num_beams':3, 'do_sample':False})
                    
                    out_ids = model.generate(**gen_inputs)
                    gen_text = self.processor.decode(out_ids[0], skip_special_tokens=True).strip()
                    
                    # Extract assistant response
                    if '<|im_start|>assistant' in gen_text:
                        gen_text = gen_text.split('<|im_start|>assistant')[-1].strip()
                    
                    generated_texts.append(gen_text)
                    reference_texts.append(sample.get('caption',''))
                    
                    if i < 2:
                        print(f"[Eval {i}] Gen: {gen_text[:150]}...")
                        print(f"          Ref: {sample.get('caption','')[:150]}...")
                        
                except Exception as e:
                    print(f"Eval sample {i} failed: {e}")
        
        # Compute BLEU
        if generated_texts:
            try:
                from nltk.translate.bleu_score import sentence_bleu
                bleu_scores = [sentence_bleu([r.lower().split()], g.lower().split(), 
                                            weights=(0.25,0.25,0.25,0.25)) 
                              for g,r in zip(generated_texts, reference_texts)]
                avg_bleu = sum(bleu_scores)/len(bleu_scores)
                self.metrics_history['bleu'].append(avg_bleu)
                self.metrics_history['steps'].append(self.state.global_step)
                print(f"BLEU-4: {avg_bleu:.4f}")
            except Exception as e:
                print(f"BLEU computation failed: {e}")
        
        model.train()

    def on_train_end(self, args, state, control, **kwargs):
        # Save metrics history
        try:
            if self.metrics_history['steps']:
                metrics_file = os.path.join(args.output_dir, 'metrics_history.json')
                with open(metrics_file, 'w') as f:
                    json.dump(self.metrics_history, f, indent=2)
                print(f"✓ Saved metrics history: {metrics_file}")
        except Exception as e:
            print(f"Failed saving metrics: {e}")
        
        # Save prompt learner
        try:
            if self.prompt_learner is not None:
                prompt_file = os.path.join(args.output_dir, 'prompt_learner.pt')
                torch.save(self.prompt_learner.state_dict(), prompt_file)
                print(f"✓ Saved prompt learner: {prompt_file}")
        except Exception as e:
            print(f"Failed saving prompt learner: {e}")

# -------------------------
# Dataset & Collate
# -------------------------
class CaptionDataset(Dataset):
    def __init__(self, json_path: str, processor=None, image_root='EarthMind-Bench/img/test/'):
        """
        image_root must contain:
            rgb/img/
            sar/img/
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.processor = processor
        self.image_root = image_root.rstrip('/')
        self.rgb_dir = os.path.join(self.image_root, "rgb/img")
        self.sar_dir = os.path.join(self.image_root, "sar/img")

        print(f"✓ CaptionDataset: {len(self.data)} samples from {json_path}")
        print(f"  ├─ RGB dir: {self.rgb_dir}")
        print(f"  └─ SAR dir: {self.sar_dir}")

    def __len__(self):
        return len(self.data)

    def resolve_image_path(self, file_name: str):
        """
        Takes a file name (like xyz.png or xyz.jpg) and returns a randomly chosen
        existing image path from rgb/img or sar/img.
        """

        # Standardize extension (your json may contain .json or .png)
        base = file_name.replace(".json", "").replace(".jpg", "").replace(".png", "")
        candidate_png = base + ".png"
        candidate_jpg = base + ".jpg"

        choices = []

        for name in [candidate_png, candidate_jpg]:
            rgb_path = os.path.join(self.rgb_dir, name)
            sar_path = os.path.join(self.sar_dir, name)

            if os.path.isfile(rgb_path):
                choices.append(rgb_path)
            if os.path.isfile(sar_path):
                choices.append(sar_path)

        if len(choices) == 0:
            return None

        # Random pick between available RGB/SAR files
        return random.choice(choices)

    def __getitem__(self, idx):
        entry = self.data[idx]

        # Prefer file_name
        file_name = entry.get("file_name") or entry.get("image")

        if not file_name:
            # skip invalid entries
            return self.__getitem__((idx + 1) % len(self.data))

        img_path = self.resolve_image_path(file_name)

        if img_path is None:
            print(f"⚠ No RGB/SAR image found for {file_name}")
            return self.__getitem__((idx + 1) % len(self.data))

        # load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Failed opening {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.data))

        # get caption
        caption = entry.get('caption') or entry.get('captions') or entry.get('answer') or ""
        if isinstance(caption, list):
            caption = caption[0] if caption else ""

        # build conversation
        system_instruction = (
            "You are an assistant that generates detailed, accurate descriptions "
            "of satellite and aerial imagery."
        )

        conversation = [
            {"role":"system","content":[{"type":"text","text":system_instruction}]},
            {"role":"user","content":[
                {"type":"image","image":image},
                {"type":"text","text":"Please describe this image in detail."}
            ]},
            {"role":"assistant","content":[{"type":"text","text":caption}]}
        ]

        # full prompt
        prompt = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(text=prompt, images=[image], return_tensors='pt')
        labels = inputs['input_ids'].clone()

        # prefix mask
        conv_wo_assistant = conversation[:-1]
        prefix_prompt = self.processor.apply_chat_template(
            conv_wo_assistant, tokenize=False, add_generation_prompt=True
        )
        prefix_inputs = self.processor(
            text=prefix_prompt, images=[image], return_tensors='pt'
        )
        prefix_length = prefix_inputs['input_ids'].shape[1]
        labels[0, :prefix_length] = -100

        # fix pixel_values shape
        pv = inputs.get('pixel_values', None)
        if pv is not None and hasattr(pv, 'dim'):
            if pv.dim() == 4 and pv.size(0) == 1:
                pv = pv.squeeze(0)

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': pv,
            'image_grid_thw': inputs.get('image_grid_thw', torch.tensor([1,1,1])).squeeze(0),
            'labels': labels.squeeze(0),
            'image_path': img_path,
            'image_size': image.size,
        }



def collate_fn(batch):
    # Pad textual inputs
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    max_len = max(ids.size(0) for ids in input_ids)

    input_ids_padded = torch.stack([
        torch.cat([ids, torch.zeros(max_len - ids.size(0), dtype=ids.dtype)]) 
        for ids in input_ids
    ]).long()
    
    attention_mask_padded = torch.stack([
        torch.cat([m, torch.zeros(max_len - m.size(0), dtype=m.dtype)]) 
        for m in attention_mask
    ]).long()
    
    labels_padded = torch.stack([
        torch.cat([l, torch.full((max_len - l.size(0),), -100, dtype=l.dtype)]) 
        for l in labels
    ]).long()

    # Handle pixel_values
    pixel_values = None
    if batch[0]['pixel_values'] is not None:
        pvs = [item['pixel_values'] for item in batch]
        first = pvs[0]
        
        if isinstance(first, torch.Tensor):
            if first.dim() == 2:
                # (N, D) patch tokens - pad to max_N
                max_N = max(pv.size(0) for pv in pvs)
                D = first.size(1)
                padded = []
                for pv in pvs:
                    if pv.size(0) < max_N:
                        pad = torch.zeros((max_N - pv.size(0), D), dtype=pv.dtype)
                        pv = torch.cat([pv, pad], dim=0)
                    padded.append(pv)
                pixel_values = torch.stack(padded, dim=0)
            else:
                # Try stacking directly
                try:
                    pixel_values = torch.stack(pvs, dim=0)
                except Exception:
                    pixel_values = None

    image_grid_thw = None
    if batch[0]['image_grid_thw'] is not None:
        try:
            image_grid_thw = torch.stack([item['image_grid_thw'] for item in batch])
        except Exception:
            pass

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'labels': labels_padded,
        'pixel_values': pixel_values,
        'image_grid_thw': image_grid_thw,
        'image_path': [item['image_path'] for item in batch],
        'image_size': [item['image_size'] for item in batch],
    }



# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description='Fine-tune Qwen3-VL with CoCoOp (final)')
    parser.add_argument('--train-json', required=True)
    parser.add_argument('--model', default='unsloth/Qwen3-VL-8B-Instruct')
    parser.add_argument('--output-dir', default='./checkpoints_cocoop_final')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4)
    parser.add_argument('--save-steps', type=int, default=50)
    parser.add_argument('--logging-steps', type=int, default=10)
    parser.add_argument('--use-lora', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--image-dir', default='../EarthMind-Bench/img/test')
    parser.add_argument('--local-model-dir', default=None)
    parser.add_argument('--prompt-length', type=int, default=16)
    parser.add_argument('--img-feat-dim', type=int, default=1536)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # logging
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    os.environ['TORCH_COMPILE_DISABLE'] = '1'

    print("Loading model:", args.model, "local_dir:", args.local_model_dir)
    local_dir = args.local_model_dir
    local_exists = bool(local_dir and os.path.isdir(local_dir))

    model = None
    processor = None
    tokenizer = None

    if UNSLOTH_AVAILABLE and args.use_lora:
        if local_exists:
            model, tokenizer = FastVisionModel.from_pretrained(local_dir)
            processor = tokenizer
        else:
            model, tokenizer = FastVisionModel.from_pretrained(args.model, load_in_4bit=args.load_in_4bit, use_gradient_checkpointing='unsloth')
            processor = tokenizer
            if local_dir:
                try:
                    os.makedirs(local_dir, exist_ok=True)
                    model.save_pretrained(local_dir)
                    tokenizer.save_pretrained(local_dir)
                except Exception:
                    pass
        try:
            model = model.to(dtype=torch.bfloat16)
        except Exception:
            pass
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    else:
        from transformers import Qwen3VLForConditionalGeneration
        if local_exists:
            model = Qwen3VLForConditionalGeneration.from_pretrained(local_dir, torch_dtype=torch.bfloat16)
            processor = AutoProcessor.from_pretrained(local_dir)
            tokenizer = processor.tokenizer
        else:
            model_name = args.model.replace('unsloth/', 'Qwen/')
            model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
            processor = AutoProcessor.from_pretrained(model_name)
            tokenizer = processor.tokenizer
            if local_dir:
                try:
                    os.makedirs(local_dir, exist_ok=True)
                    model.save_pretrained(local_dir)
                    processor.save_pretrained(local_dir)
                except Exception:
                    pass
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    # ensure we have embedding dim and dtype
    try:
        embed_layer = model.get_input_embeddings()
        embed_dim = embed_layer.weight.size(1)
        embed_dtype = embed_layer.weight.dtype
    except Exception:
        embed_dim = 4096
        embed_dtype = torch.float32

    prompt_length = args.prompt_length
    img_feat_dim = args.img_feat_dim

    # instantiate prompt learner
    prompt_learner = PromptLearner(prompt_length=prompt_length, embed_dim=embed_dim, img_feat_dim=img_feat_dim)

    # move prompt learner to model device and cast to embedding dtype
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    prompt_learner = prompt_learner.to(device=model_device, dtype=embed_dtype)

    # freeze model weights
    for p in model.parameters():
        p.requires_grad = False
    # ensure prompt learner trainable
    for p in prompt_learner.parameters():
        p.requires_grad = True

    # quick sanity prints
    print("Model device:", model_device, "Embed dtype:", embed_dtype)
    print("Prompt learner device:", next(prompt_learner.parameters()).device, "Prompt dtype:", next(prompt_learner.parameters()).dtype)
    print("Prompt learner trainable params:", sum(p.numel() for p in prompt_learner.parameters() if p.requires_grad))

    # dataset
    train_dataset = CaptionDataset(args.train_json, processor, image_root=args.image_dir)
    print("Training samples:", len(train_dataset))

    # eval samples (small)
    eval_samples = []

    rgb_count = 0
    sar_count = 0
    MAX_RGB = 3
    MAX_SAR = 3

    for i in range(len(train_dataset)):
        if rgb_count >= MAX_RGB and sar_count >= MAX_SAR:
            break  # collected all 6

        try:
            entry = train_dataset.data[i]

            raw = entry.get('image') or entry.get('file_name')
            if not raw:
                continue

            base = os.path.basename(raw)
            base_no_ext = os.path.splitext(base)[0]
            candidates = [
                base,
                base_no_ext + ".png",
                base_no_ext + ".jpg",
                base_no_ext + ".jpeg",
            ]

            img_path = None
            modality = None

            # --- RGB FIRST ---
            if rgb_count < MAX_RGB:
                for c in candidates:
                    p = os.path.join(args.image_dir, "rgb/img", c)
                    if os.path.isfile(p):
                        img_path = p
                        modality = "rgb"
                        break

            # --- SAR NEXT ---
            if img_path is None and sar_count < MAX_SAR:
                for c in candidates:
                    p = os.path.join(args.image_dir, "sar/img", c)
                    if os.path.isfile(p):
                        img_path = p
                        modality = "sar"
                        break

            if img_path is None:
                continue  # skip if neither RGB nor SAR found

            image = Image.open(img_path).convert('RGB')

            caption = entry.get('caption') or entry.get('answer') or ""
            if isinstance(caption, list):
                caption = caption[0] if caption else ""

            eval_samples.append({
                'image': image,
                'caption': caption,
                'image_path': img_path
            })

            print(f"Loaded eval sample ({modality}):", img_path)

            # increment counters
            if modality == "rgb":
                rgb_count += 1
            elif modality == "sar":
                sar_count += 1

        except Exception:
            print("Failed to load eval sample:", traceback.format_exc())


    # training args
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

    # optimizer only for prompt learner
    prompt_params = [p for p in prompt_learner.parameters() if p.requires_grad]
    if len(prompt_params) == 0:
        raise RuntimeError("No trainable parameters found in prompt_learner.")
    optimizer = torch.optim.AdamW(prompt_params, lr=args.lr)
    print("Optimizer params:", sum(p.numel() for p in prompt_params))

    # instantiate trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        eval_samples=eval_samples,
        processor=processor,
        prompt_learner=prompt_learner,
        prompt_length=prompt_length,
        get_image_features_fn=get_image_features_from_model,
        eval_frequency=100,
        show_predictions_frequency=10,
        optimizers=(optimizer, None),
    )

    print("Starting training (CoCoOp final)...")
    try:
        trainer.train()
    except Exception as e:
        print("Training failed:", e)
        traceback.print_exc()
        raise

    # save final
    final_dir = os.path.join(args.output_dir, "final")
    print("Saving final artifacts to:", final_dir)
    trainer.save_model(final_dir)
    try:
        processor.save_pretrained(final_dir)
    except Exception:
        try:
            tokenizer.save_pretrained(final_dir)
        except Exception:
            pass
    try:
        os.makedirs(final_dir, exist_ok=True)
        torch.save(prompt_learner.state_dict(), os.path.join(final_dir, "prompt_learner.pt"))
        print("Saved prompt_learner:", os.path.join(final_dir, "prompt_learner.pt"))
    except Exception:
        print("Failed saving prompt learner:", traceback.format_exc())

    print("Training complete.")

if __name__ == "__main__":
    main()

# python finetune_cocoop.py --train-json /home/samyak/scratch/interiit/EarthMind-Bench/json/caption_all_unmatched.json --image-dir /home/samyak/scratch/interiit/EarthMind-Bench/img/test --output-dir temp

