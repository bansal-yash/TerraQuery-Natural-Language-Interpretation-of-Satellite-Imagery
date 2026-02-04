#!/usr/bin/env python3
"""
Fine-tune Qwen3-VL model using CoCoOp (Context Optimization with Conditional Prompt Learning).

This script trains ONLY the PromptLearner module while keeping the base model frozen.
The PromptLearner generates image-conditioned soft prompts that are prepended to the input.

Key features:
- Base model is completely frozen
- Only PromptLearner parameters are trained (very few parameters)
- Image features are extracted and used to condition the soft prompts
- Gradient norms are printed for monitoring training progress

Usage example:
    python finetune_cocoop_v2.py \\
        --train-json /path/to/caption.json \\
        --output-dir ./checkpoints_cocoop \\
        --batch-size 2 \\
        --epochs 3 \\
        --use-lora  # Use unsloth for faster loading
"""
import argparse
import json
import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
from typing import List
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from tqdm import tqdm
import logging
import traceback
import sys
import inspect
from typing import Optional
# Using plain prints for debug/trace output per user request
from transformers import TrainerCallback
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image as PILImage
try:
    from unsloth import FastVisionModel
    UNSLOTH_AVAILABLE = True
    print("helll yeahhhhhhhhhhhhhhhhhhhhh")
    
except Exception:
    UNSLOTH_AVAILABLE = False
    print("helll nooooooo")


class PromptLearner(nn.Module):
    def __init__(self, prompt_length: int, embed_dim: int, img_feat_dim: int = 1536, hidden_dim: int = 512,
                 pooling: str = "mean"):
        """
        pooling: "mean" (default) or "max" or "cls"
        img_feat_dim: expected feature dimension (1536 for your case)
        """
        super().__init__()
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.img_feat_dim = img_feat_dim
        self.pooling = pooling

        # base prompt (P, D)
        self.base_prompt = nn.Parameter(torch.randn(prompt_length, embed_dim) * 0.02)

        # adapter MLP: image features -> P*D
        self.adapter = nn.Sequential(
            nn.Linear(img_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, prompt_length * embed_dim),
        )

    def _pool(self, feats: torch.Tensor) -> torch.Tensor:
        # Accepts:
        #  - (B, F)           -> return (B, F)
        #  - (N_patches, F)   -> treat as single image's patches -> return (1, F)
        #  - (B, N_patches, F)-> pool across patches -> return (B, F)
        if feats.dim() == 2:
            # either (B, F) or (N_patches, F) (single image w/out batch dim)
            if feats.size(0) == 1:
                return feats  # already (1, F)
            else:
                # treat as patches for a single image -> pool to (1, F)
                if self.pooling == "mean":
                    return feats.mean(dim=0, keepdim=True)
                elif self.pooling == "max":
                    return feats.max(dim=0, keepdim=True)[0]
                elif self.pooling == "cls":
                    return feats[:1, :]  # first token as CLS
                else:
                    raise ValueError(f"Unknown pooling: {self.pooling}")

        elif feats.dim() == 3:
            # (B, N, F)
            if self.pooling == "mean":
                return feats.mean(dim=1)
            elif self.pooling == "max":
                return feats.max(dim=1)[0]
            elif self.pooling == "cls":
                return feats[:, 0, :]
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")
        else:
            raise ValueError(f"Unsupported feats.dim(): {feats.dim()}")

    def forward(self, image_features: torch.Tensor):
        """
        image_features expected (most common for you): (1024, 1536)
        returns: (B, prompt_length, embed_dim) where B is 1 in the (1024,1536) case
        """
        pooled = self._pool(image_features)   # (B, F)
        if pooled.size(1) != self.img_feat_dim:
            raise ValueError(f"Expected img_feat_dim={self.img_feat_dim}, got {pooled.size(1)}")

        batch = pooled.size(0)
        delta = self.adapter(pooled)                      # (B, P*D)
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
            raise

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
        raise RuntimeError(f"Failed to detect image feature dimension: {e}")
    
    return 1536

class CustomTrainer(Trainer):
    """Custom Trainer with prediction monitoring and BLEU/BERT evaluation."""
    
    def __init__(self, *args, eval_samples=None, processor=None, prompt_learner: Optional[nn.Module]=None, prompt_length: int=0,
                 eval_frequency=1000, show_predictions_frequency=1000, prompt_reg_weight: float=0.01, grad_clip: float=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_samples = eval_samples or []
        self.processor = processor
        self.eval_frequency = eval_frequency
        self.prompt_learner = prompt_learner
        self.prompt_length = prompt_length
        self.show_predictions_frequency = show_predictions_frequency
        self.metrics_history = {'bleu': [], 'bert': [], 'steps': []}
        self.prompt_reg_weight = prompt_reg_weight
        self.grad_clip = grad_clip
        
        # Cache device and dtype for performance (avoid repeated next() calls)
        if prompt_learner is not None:
            self._prompt_device = next(prompt_learner.parameters()).device
            self._prompt_dtype = next(prompt_learner.parameters()).dtype
        else:
            self._prompt_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._prompt_dtype = torch.bfloat16
    
    def _compute_grad_norm(self):
        """Compute gradient norm for prompt learner parameters."""
        if self.prompt_learner is None:
            return 0.0
        total_norm = 0.0
        for p in self.prompt_learner.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, device='cuda'):
        """Override compute_loss - optimized for speed."""
        image_paths = inputs.pop('image_path', None)
        _ = inputs.pop('image_size', None)  # Not used

        pixel_values = inputs['pixel_values'].to(device)
                
        # Extract image features (no grad for vision encoder)
        with torch.no_grad():
            img_feats = get_image_features_from_model(model, pixel_values)
        
        if img_feats is None:
            raise RuntimeError("get_image_features returned None")
        
        # Ensure batch dim
        if img_feats.dim() == 1:
            img_feats = img_feats.unsqueeze(0)

        # Move to prompt learner device/dtype (use cached values for speed)
        img_feats = img_feats.to(device=self._prompt_device, dtype=self._prompt_dtype)

        
        # Generate prompt embeddings (this is where gradients matter!)
        prompt_embeds = self.prompt_learner(img_feats).to(device)  # (B, P, D)
        B, P, D = prompt_embeds.shape

        # Prepare text inputs
        input_ids = inputs.pop('input_ids').to(device)
        attention_mask = inputs.pop('attention_mask').to(device)
        labels = inputs.pop('labels').to(device)
        
        # Pop image_grid_thw if present (not needed when using inputs_embeds)
        _ = inputs.pop('image_grid_thw', None)

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

        # Build inputs for model - NO pixel_values since we're using inputs_embeds
        inputs_for_model = {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
        outputs = model(**inputs_for_model)
        
        # Compute loss manually to ensure gradient flow to prompt_learner
        logits = outputs.logits if hasattr(outputs, 'logits') else None
        
        if logits is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            vocab_size = shift_logits.size(-1)
            flat_logits = shift_logits.view(-1, vocab_size)
            flat_labels = shift_labels.view(-1)

            # Cross entropy loss
            per_token_loss = F.cross_entropy(flat_logits, flat_labels, reduction='none')
            mask = (flat_labels != -100).float()
            
            if mask.sum() > 0:
                ce_loss = (per_token_loss * mask).sum() / mask.sum()
            else:
                # Create a differentiable zero loss connected to prompt learner
                ce_loss = (self.prompt_learner.base_prompt[0, 0] * 0.0).requires_grad_(True)
            
            # L2 regularization on prompt learner parameters
            reg_loss = 0.0
            for p in self.prompt_learner.parameters():
                reg_loss = reg_loss + p.pow(2).sum()
            reg_loss = self.prompt_reg_weight * reg_loss
            
            loss = ce_loss + reg_loss
            
            # Log regularization info periodically
            if self.state.global_step % self.args.logging_steps == 0:
                print(f"  CE Loss: {ce_loss.item():.4f} | Reg Loss: {reg_loss.item():.6f} | Total: {loss.item():.4f}", end = "      ")
        else:
            loss = outputs.loss
        
        # Detailed prediction logging (only occasionally to save time)
        if self.state.global_step > 0 and self.state.global_step % self.show_predictions_frequency == 0:
            try:
                with torch.no_grad():
                    predicted_ids = torch.argmax(shift_logits, dim=-1)
                    valid_mask = (shift_labels != -100)
                    if valid_mask.any():
                        # Just first sample
                        sample_mask = valid_mask[0]
                        if sample_mask.any():
                            pred_tokens = predicted_ids[0][sample_mask]
                            label_tokens = shift_labels[0][sample_mask]
                            correct = (pred_tokens == label_tokens).sum().item()
                            total = sample_mask.sum().item()
                            accuracy = 100 * correct / total
                            
                            pred_text = self.processor.decode(pred_tokens[:50], skip_special_tokens=True)
                            label_text = self.processor.decode(label_tokens[:50], skip_special_tokens=True)
                            
                            print(f"\n[Step {self.state.global_step}] Acc: {accuracy:.1f}%")
                            print(f"Pred: {pred_text[:150]}...")
                            print(f"Ref:  {label_text[:150]}...")
            except Exception:
                pass  # Don't slow down training for logging errors
        
        # Periodic BLEU/BERT evaluation (expensive - do rarely)
        if (self.state.global_step > 0 and 
            self.state.global_step % self.eval_frequency == 0 and 
            len(self.eval_samples) > 0):
            self._evaluate_metrics(model)
        
        # Print grad norm for prompt learner (after backward)
        if self.state.global_step % self.args.logging_steps == 0:
            grad_norm = self._compute_grad_norm()
            print(f"Step {self.state.global_step} | Loss: {loss.item():.4f} | Prompt Learner Grad Norm: {grad_norm:.6f}", end = "   ")
        
        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training_step to apply gradient clipping."""
        model.train()
        self.prompt_learner.train()
        inputs = self._prepare_inputs(inputs)
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        if self.args.n_gpu > 1:
            loss = loss.mean()
        
        # Backward pass
        self.accelerator.backward(loss)
        
        # Gradient clipping for prompt learner
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.prompt_learner.parameters(), self.grad_clip)
        
        # Log grad norm occasionally
        if self.state.global_step % (self.args.logging_steps * 5) == 0:
            grad_norm = self._compute_grad_norm()
            print(f"Step {self.state.global_step} | Grad Norm: {grad_norm:.4f}")
        
        return loss.detach() / self.args.gradient_accumulation_steps
        
    def _evaluate_metrics(self, model):
        """Evaluate BLEU and BERT scores on eval samples."""
        # print(f"\n{'='*80}")
        # print(f"GENERATION METRICS - Step {self.state.global_step}")
        # print(f"{'='*80}")
        
        model.eval()
        generated_texts = []
        reference_texts = []
        
        with torch.no_grad():
            for i, sample in enumerate(self.eval_samples[:2]):  # Evaluate 5 samples
                try:
                    conversation = [
                        {"role": "system", "content": [{"type": "text", "text": "You are an assistant that generates detailed, accurate descriptions of satellite and aerial imagery."}]},
                        {"role": "user", "content": [
                            {"type": "image", "image": sample['image']},
                            {"type": "text", "text": "Please describe this image in detail."}
                        ]},
                    ]
                    
                    prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                    inputs = self.processor(text=prompt, images=[sample['image']], return_tensors='pt')
                    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in inputs.items()}
                    
                    # Generate with greedy decoding (fast)
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False,
                        num_beams=1,  # Greedy is faster than beam search
                    )
                    
                    generated_text = self.processor.decode(
                        output_ids[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    ).strip()
                    
                    generated_texts.append(generated_text)
                    reference_texts.append(sample['caption'])
                    
                    # if i < 2:  # Print first 2 for inspection
                    #     print(f"\n[Sample {i+1}]")
                    #     print(f"Generated: {generated_text[:120]}...")
                    #     print(f"Reference: {sample['caption'][:120]}...")
                
                except Exception as e:
                    print(f"Error generating sample {i}: {e}")
                    continue
        
        # Compute BLEU
        if generated_texts:
            try:
                from nltk.translate.bleu_score import sentence_bleu
                
                bleu_scores = []
                for gen, ref in zip(generated_texts, reference_texts):
                    gen_tokens = gen.lower().split()
                    ref_tokens = ref.lower().split()
                    
                    # BLEU-4 with smoothing
                    bleu = sentence_bleu(
                        [ref_tokens], 
                        gen_tokens, 
                        weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=lambda p, *args, **kwargs: [
                            (p_i.numerator + 1) / (p_i.denominator + 1) for p_i in p
                        ] if hasattr(p[0], 'numerator') else p
                    )
                    bleu_scores.append(bleu)
                
                avg_bleu = sum(bleu_scores) / len(bleu_scores)
                self.metrics_history['bleu'].append(avg_bleu)
                self.metrics_history['steps'].append(self.state.global_step)
                
                # print(f"\n✅ BLEU-4: {avg_bleu:.4f}")
                
            except ImportError:
                pass  # NLTK not installed
            except Exception as e:
                pass  # BLEU failed, don't print to avoid slowdown
            
            # Compute BERT Score
            try:
                from bert_score import score as bert_score_fn
                
                P, R, F1 = bert_score_fn(
                    generated_texts, 
                    reference_texts, 
                    lang='en', 
                    verbose=False,
                    device=model.device
                )
                avg_bert_f1 = F1.mean().item()
                self.metrics_history['bert'].append(avg_bert_f1)
                
                # print(f"✅ BERT-F1: {avg_bert_f1:.4f}")
                
            except ImportError:
                print("⚠️  bert-score not installed. Run: pip install bert-score")
            except Exception as e:
                print(f"❌ BERT Score computation failed: {e}")
        
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


class CaptionDataset(Dataset):
    """Simple dataset for captioning entries.

    Supports loading from two JSON files and two image directories with random mixing.
    Each entry contains at least:
      - 'image': path to image, OR 'file_name': JSON filename to construct image path
      - 'caption' (or 'captions' or 'answer'): reference caption string (used as assistant output)
    """

    def __init__(self, json_path: str, processor, image_dir: str = 'EarthMind-Bench/img/test/rgb',
                 json_path2: str = None, image_dir2: str = None, shuffle: bool = True):
        import random
        
        # Load first JSON
        with open(json_path, 'r') as f:
            data1 = json.load(f)
        
        # Tag each entry with its source image directory
        for entry in data1:
            entry['_image_dir'] = image_dir
            entry['_source'] = 'json1'
        
        self.data = data1
        print(f"Loaded {len(data1)} entries from JSON1: {json_path}")
        
        # Load second JSON if provided
        if json_path2 and os.path.exists(json_path2):
            with open(json_path2, 'r') as f:
                data2 = json.load(f)
            
            # Use image_dir2 if provided, otherwise fall back to image_dir
            img_dir2 = image_dir2 if image_dir2 else image_dir
            for entry in data2:
                entry['_image_dir'] = img_dir2
                entry['_source'] = 'json2'
            
            self.data.extend(data2)
            print(f"Loaded {len(data2)} entries from JSON2: {json_path2}")
        
        # Shuffle the combined dataset
        if shuffle:
            random.shuffle(self.data)
            print(f"Shuffled combined dataset")
        
        self.processor = processor
        print(f"CaptionDataset initialized: total entries={len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        
        # Get the image directory for this entry
        image_dir = entry.get('_image_dir', '')
        
        # Handle different field names: 'image' or construct from 'file_name'
        img_npath = entry.get('image')
        if not img_npath:
            file_name = entry.get('file_name')
            if file_name:
                # Convert 'dfc2023_test_P_0601.json' -> 'dfc2023_test_P_0601.png'
                img_name = file_name.replace('.json', '.png')
                img_path = os.path.join(image_dir, img_name)
            else:
                img_path = None
        else:
            img_path = os.path.join(image_dir, img_npath)
        
        
        if not img_path:
            print(f"No image path found in JSON entry {idx}; skipping to next")
            return self.__getitem__((idx + 1) % len(self.data))
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path} at entry {idx}: {e}\n{traceback.format_exc()}")
            return self.__getitem__((idx + 1) % len(self.data))

        # Extract question and answer from conversations array
        conversations = entry.get('conversations', [])
        human_question = ""
        gpt_answer = ""
        
        for conv in conversations:
            if conv.get('from') == 'human':
                # Remove <image> token if present
                human_question = conv.get('value', '').replace('<image>\n', '').replace('<image>', '').strip()
            elif conv.get('from') == 'gpt':
                gpt_answer = conv.get('value', '').strip()
        
        # Fallback to old caption fields if conversations not found
        if not human_question:
            human_question = "Please describe this image in detail."
        if not gpt_answer:
            gpt_answer = entry.get('caption') or entry.get('captions') or entry.get('answer') or ""
            if isinstance(gpt_answer, list):
                gpt_answer = gpt_answer[0] if gpt_answer else ""

        system_instruction = (
            "You are an assistant that generates detailed, accurate descriptions "
            "of satellite and aerial imagery."
        )

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": human_question}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": gpt_answer}]}
        ]

        # Prepare model inputs using processor's chat template
        prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(text=prompt, images=[image], return_tensors='pt')
        # ✅ CORRECT MASKING: Mask everything EXCEPT assistant response
        labels = inputs['input_ids'].clone()
        
        # Method 1: Find assistant response by building prefix without it
        conversation_without_assistant = [
            {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": human_question}
            ]},
        ]
        
        # Get prompt up to (and including) the assistant token
        prefix_prompt = self.processor.apply_chat_template(
            conversation_without_assistant,
            tokenize=False,
            add_generation_prompt=True  # This adds "<|im_start|>assistant\n"
        )
        
        # Tokenize the prefix (everything we want to MASK)
        prefix_inputs = self.processor(
            text=prefix_prompt,
            images=[image],  # ⚠️ CRITICAL: Include image here too!
            return_tensors='pt'
        )
        
        # Mask everything up to the assistant response
        prefix_length = prefix_inputs['input_ids'].shape[1]
        labels[0, :prefix_length] = -100

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': inputs['pixel_values'].squeeze(0) if 'pixel_values' in inputs else None,
            'image_grid_thw': inputs['image_grid_thw'].squeeze(0) if 'image_grid_thw' in inputs else None,
            'labels': labels.squeeze(0),
            'image_path': img_path,
            'image_size': image.size,
        }


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    max_len = max(seq.size(0) for seq in input_ids)

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

    # For Qwen3-VL with dynamic resolution, concatenate pixel_values along spatial dim
    # Each image creates variable number of tokens, model handles this with image_grid_thw
    pixel_values = None
    image_grid_thw = None
    if batch[0]['pixel_values'] is not None:
        # Concatenate all images' pixel values along the spatial token dimension
        pixel_values = torch.cat([item['pixel_values'] for item in batch], dim=0)
    if batch[0]['image_grid_thw'] is not None:
        # Stack image_grid_thw to track which tokens belong to which image
        image_grid_thw = torch.stack([item['image_grid_thw'] for item in batch])


    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'labels': labels_padded,
        'pixel_values': pixel_values,
        'image_grid_thw': image_grid_thw,
        'image_path': [item['image_path'] for item in batch],
        'image_size': [item['image_size'] for item in batch],
    }

class PromptSaverCallback(TrainerCallback):
    def __init__(self, prompt_learner):
        self.prompt_learner = prompt_learner

    def on_save(self, args, state, control, **kwargs):
        try:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            save_path = os.path.join(ckpt_dir, "prompt_learner.pt")
            torch.save(self.prompt_learner.state_dict(), save_path)
            print(f"✓ Saved prompt_learner to {save_path}")
        except Exception as e:
            print(f"⚠️ Failed to save prompt learner at checkpoint: {e}")



def main():
    parser = argparse.ArgumentParser(description='Fine-tune Qwen3-VL for image captioning')
    parser.add_argument('--train-json', required=True, help='Path to first training JSON file')
    parser.add_argument('--train-json2', default=None, help='Path to second training JSON file (optional)')
    parser.add_argument('--model', default='unsloth/Qwen3-VL-8B-Instruct', help='Base model to fine-tune')
    parser.add_argument('--output-dir', default='qwen/checkpoints_caption', help='Output directory for checkpoints')
    parser.add_argument('--batch-size', type=int, default=1, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--save-steps', type=int, default=1000, help='Save checkpoint every N steps')
    parser.add_argument('--logging-steps', type=int, default=100, help='Log every N steps')
    parser.add_argument('--use-lora', action='store_true', default=False, help='Use unsloth for faster model loading (no LoRA applied in CoCoOp)')
    parser.add_argument('--load-in-4bit', action='store_true', default=False, help='Use 4-bit quantization for memory efficiency')
    parser.add_argument('--image-dir', default='../EarthMind-Bench/img/test/sar/img', help='Directory containing image files for first JSON')
    parser.add_argument('--image-dir2', default=None, help='Directory containing image files for second JSON (optional)')
    parser.add_argument('--local-model-dir', default=None, help='Local directory to load/save the pretrained model')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--prompt-length', type=int, default=16)
    parser.add_argument('--img-feat-dim', type=int, default=1536)
    parser.add_argument('--prompt-reg-weight', type=float, default=0.01, help='L2 regularization weight for prompt learner')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping max norm for prompt learner')

    args = parser.parse_args()

    # If debug requested, prefer simple prints
    global USE_PRINT
    USE_PRINT = bool(getattr(args, 'debug', False))

    # Configure logging; when using print mode, raise logging to WARNING to avoid duplicate output
    log_level = logging.DEBUG if not USE_PRINT and getattr(args, 'debug', False) else logging.WARNING if USE_PRINT else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s %(levelname)s: %(message)s')

    os.environ['TORCH_COMPILE_DISABLE'] = '1'

    print(f'Loading model: {args.model} (local dir: {args.local_model_dir})')

    # Prefer loading from a local model directory if provided and present
    local_dir = args.local_model_dir
    local_exists = bool(local_dir and os.path.isdir(local_dir))

    if UNSLOTH_AVAILABLE and args.use_lora:
        # unsloth path - Note: For CoCoOp we DON'T apply LoRA, just load the model
        if local_exists:
            print(f'Loading unsloth model from local dir: {local_dir}')
            model, tokenizer = FastVisionModel.from_pretrained(local_dir)
            processor = tokenizer
        else:
            print('Using unsloth FastVisionModel (downloading)')
            model, tokenizer = FastVisionModel.from_pretrained(
                args.model,
                load_in_4bit=args.load_in_4bit,
                use_gradient_checkpointing='unsloth',
            )
            processor = tokenizer
            if local_dir:
                try:
                    os.makedirs(local_dir, exist_ok=True)
                    model.save_pretrained(local_dir)
                    tokenizer.save_pretrained(local_dir)
                    print(f'Saved unsloth model to local dir: {local_dir}')
                except Exception as e:
                    print(f'Warning: failed to save unsloth model to {local_dir}: {e}')

        # NOTE: For CoCoOp, we do NOT apply LoRA - base model stays frozen
        # Only the PromptLearner is trained
        print("CoCoOp mode: Base model will be frozen, only PromptLearner is trained")

        # Ensure model dtype compatibility
        try:
            model = model.to(dtype=torch.bfloat16)
        except Exception:
            pass

        # Enable gradient checkpointing to reduce memory (trade compute for memory)
        try:
            model.gradient_checkpointing_enable()
            print('Enabled gradient checkpointing on model to reduce memory usage')
        except Exception:
            pass

    else:
        # standard transformers path
        from transformers import Qwen3VLForConditionalGeneration
        if local_exists:
            print(f'Loading model and processor from local dir: {local_dir}')
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                local_dir,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            processor = AutoProcessor.from_pretrained(local_dir)
            tokenizer = processor.tokenizer
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                args.model,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            processor = AutoProcessor.from_pretrained(args.model)
            tokenizer = processor.tokenizer
            if local_dir:
                try:
                    os.makedirs(local_dir, exist_ok=True)
                    model.save_pretrained(local_dir)
                    processor.save_pretrained(local_dir)
                    print(f'Saved model and processor to local dir: {local_dir}')
                except Exception as e:
                    print(f'Warning: failed to save model to {local_dir}: {e}')

        # Try enabling gradient checkpointing to reduce GPU memory use
        try:
            model.gradient_checkpointing_enable()
            print('Enabled gradient checkpointing on model to reduce memory usage')
        except Exception:
            pass


    # Freeze ALL base model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    print("\n" + "="*60)
    print("BASE MODEL FROZEN - Only training PromptLearner")
    print("="*60)

    train_dataset = CaptionDataset(
        json_path=args.train_json,
        processor=processor,
        image_dir=args.image_dir,
        json_path2=args.train_json2,
        image_dir2=args.image_dir2,
        shuffle=True
    )
    print(f'Training samples: {len(train_dataset)}')
    
    # ✨ Prepare evaluation samples
    print("Preparing evaluation samples...")
    eval_samples = []
    for i in range(min(5, len(train_dataset))):
        try:
            entry = train_dataset.data[i]
            
            # Get the image directory for this entry
            image_dir = entry.get('_image_dir', args.image_dir)
            
            # Get image path
            img_npath = entry.get('image')
            if not img_npath:
                file_name = entry.get('file_name')
                if file_name:
                    img_name = file_name.replace('.json', '.png')
                    img_path = os.path.join(image_dir, img_name)
                else:
                    continue
            else:
                img_path = os.path.join(image_dir, img_npath)
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Get caption from conversations
            conversations = entry.get('conversations', [])
            caption = ""
            for conv in conversations:
                if conv.get('from') == 'gpt':
                    caption = conv.get('value', '').strip()
                    break
            # Fallback
            if not caption:
                caption = entry.get('caption') or entry.get('answer') or ""
                if isinstance(caption, list):
                    caption = caption[0] if caption else ""
            
            eval_samples.append({
                'image': image,
                'caption': caption,
                'image_path': img_path
            })
            print(f"  Loaded eval sample {i+1}: {img_path}")
            
        except Exception as e:
            print(f"  Failed to load eval sample {i}: {e}")
    
    print(f"✅ Prepared {len(eval_samples)} evaluation samples\n")


    prompt_length = args.prompt_length
    img_feat_dim = args.img_feat_dim
    embed_layer = model.get_input_embeddings()
    embed_dim = embed_layer.weight.size(1)

    prompt_learner = PromptLearner(prompt_length=prompt_length, embed_dim=embed_dim, img_feat_dim=img_feat_dim)
    
    # Move prompt learner to model device and dtype
    try:
        model_device = next(model.parameters()).device
        embed_dtype = embed_layer.weight.dtype
    except StopIteration:
        model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        embed_dtype = torch.bfloat16
    
    prompt_learner = prompt_learner.to(device=model_device, dtype=embed_dtype)
    
    # Ensure prompt learner parameters are trainable
    for p in prompt_learner.parameters():
        p.requires_grad = True

    prompt_params = [p for p in prompt_learner.parameters() if p.requires_grad]
    if len(prompt_params) == 0:
        raise RuntimeError("No trainable parameters found in prompt_learner.")
    optimizer = torch.optim.AdamW(prompt_params, lr=args.lr)
    
    # Log parameter counts
    base_model_params = sum(p.numel() for p in model.parameters())
    base_model_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    prompt_learner_params = sum(p.numel() for p in prompt_learner.parameters())
    prompt_learner_trainable = sum(p.numel() for p in prompt_learner.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print(f"PARAMETER SUMMARY")
    print(f"{'='*60}")
    print(f"Base Model: {base_model_params:,} total, {base_model_trainable:,} trainable")
    print(f"PromptLearner: {prompt_learner_params:,} total, {prompt_learner_trainable:,} trainable")
    print(f"Optimizer params: {sum(p.numel() for p in prompt_params):,}")
    print(f"PromptLearner device: {next(prompt_learner.parameters()).device}")
    print(f"PromptLearner dtype: {next(prompt_learner.parameters()).dtype}")
    print(f"{'='*60}\n")



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
        dataloader_num_workers=16,  # Parallel data loading for speed
        remove_unused_columns=False,
        report_to='none',
        optim='adamw_torch',
    )


    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        eval_samples=eval_samples,     # ✨ Add eval samples
        processor=processor,            # ✨ Add processor
        eval_frequency=1000,
        prompt_learner=prompt_learner,
        prompt_length=prompt_length,             # ✨ Evaluate every 100 steps
        prompt_reg_weight=args.prompt_reg_weight,  # L2 regularization
        grad_clip=args.grad_clip,                  # Gradient clipping
        optimizers=(optimizer, None),
        callbacks=[PromptSaverCallback(prompt_learner)] 
    )

    print('Starting captioning training...')
    try:
        trainer.train()
    except Exception as e:
        print(f"Training failed with exception: {e}\n{traceback.format_exc()}")
        raise

    final_output_dir = os.path.join(args.output_dir, 'final')
    os.makedirs(final_output_dir, exist_ok=True)
    print(f'Saving final model to: {final_output_dir}')
    trainer.save_model(final_output_dir)
    try:
        processor.save_pretrained(final_output_dir)
    except Exception:
        # processor might be tokenizer in unsloth path; save tokenizer if present
        try:
            tokenizer.save_pretrained(final_output_dir)
        except Exception:
            pass
    
    # Save prompt learner separately
    try:
        prompt_learner_path = os.path.join(final_output_dir, 'prompt_learner.pt')
        torch.save(prompt_learner.state_dict(), prompt_learner_path)
        print(f"✅ Saved prompt_learner to: {prompt_learner_path}")
    except Exception as e:
        print(f"⚠️  Failed to save prompt_learner: {e}")

    print('\n' + '='*60)
    print('CoCoOp TRAINING COMPLETE!')
    print('='*60)


if __name__ == '__main__':
    main()


# Example CoCoOp usage with single dataset:
# python finetune_cocoop_v2.py --train-json /path/to/caption.json --image-dir /path/to/images --output-dir ./checkpoints_cocoop --use-lora

# Example CoCoOp usage with TWO datasets (mixed/shuffled):
# python finetune_cocoop_v2.py \
#     --train-json /path/to/rgb_captions.json --image-dir /path/to/rgb/images \
#     --train-json2 /path/to/sar_captions.json --image-dir2 /path/to/sar/images \
#     --output-dir ./checkpoints_cocoop --use-lora 