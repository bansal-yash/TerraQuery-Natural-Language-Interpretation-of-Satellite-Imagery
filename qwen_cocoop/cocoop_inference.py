#!/usr/bin/env python3
"""
Inference script (no use of model.generate).

This version adds diagnostic print statements to confirm prompt tokens are being used.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from PIL import Image
from typing import Optional

# ---------------------------
# GLOBALS (edit these)
# ---------------------------
IMAGE_PATH = "/home/samyak/scratch/interiit/EarthMind-Bench/img/test/sar/img/dfc2023_test_P_0421.png"   # <<--- set image path here
USER_PROMPT = "Please describe this image in detail."  # <<--- set prompt here
REFERENCE_TEXT = None  # e.g. "A satellite image of farmland..."  OR None to use greedy autoregressive decoding

# model/weights locations (adjust if your layout differs)
MODEL_NAME_OR_PATH = "unsloth/Qwen3-VL-8B-Instruct"  # or local dir used for training
LOCAL_MODEL_DIR = None  # set to local dir if you saved model/processor locally
PROMPT_LEARNER_WEIGHTS = "qwen/checkpoints_caption/checkpoint-2000/prompt_learner.pt"  # path to prompt learner state_dict
IMG_FEAT_DIM = 1536
PROMPT_LENGTH = 16
MAX_NEW_TOKENS = 120
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EOS_TOKEN_ID = None  # leave None to not stop early on EOS; set if you want early stop
# ---------------------------

# Try unsloth presence like in training
try:
    from unsloth import FastVisionModel
    UNSLOTH_AVAILABLE = True
except Exception:
    UNSLOTH_AVAILABLE = False

from transformers import AutoProcessor

# ---------------------------
# Reuse PromptLearner and helpers (same as training)
# ---------------------------
class PromptLearner(nn.Module):
    def __init__(self, prompt_length: int, embed_dim: int, img_feat_dim: int = 1536, hidden_dim: int = 512,
                 pooling: str = "mean"):
        super().__init__()
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.img_feat_dim = img_feat_dim
        self.pooling = pooling

        self.base_prompt = nn.Parameter(torch.randn(prompt_length, embed_dim) * 0.02)

        self.adapter = nn.Sequential(
            nn.Linear(img_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, prompt_length * embed_dim),
        )

    def _pool(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.dim() == 2:
            if feats.size(0) == 1:
                return feats
            else:
                if self.pooling == "mean":
                    return feats.mean(dim=0, keepdim=True)
                elif self.pooling == "max":
                    return feats.max(dim=0, keepdim=True)[0]
                elif self.pooling == "cls":
                    return feats[:1, :]
                else:
                    raise ValueError(f"Unknown pooling: {self.pooling}")

        elif feats.dim() == 3:
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
        pooled = self._pool(image_features)   # (B, F)
        if pooled.size(1) != self.img_feat_dim:
            raise ValueError(f"Expected img_feat_dim={self.img_feat_dim}, got {pooled.size(1)}")
        batch = pooled.size(0)
        delta = self.adapter(pooled)                      # (B, P*D)
        delta = delta.view(batch, self.prompt_length, self.embed_dim)  # (B, P, D)
        base = self.base_prompt.unsqueeze(0).expand(batch, -1, -1)     # (B, P, D)
        return base + delta


def get_image_features_from_model(model, pixel_values: torch.Tensor):
    device = next(model.parameters()).device
    pv = pixel_values.to(device)

    if pv.dim() == 3:
        return pv.mean(dim=1).detach()

    if pv.dim() == 4:
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

    if pv.dim() == 4:
        try:
            flat = pv.flatten(1).mean(dim=1)
            return flat.detach()
        except Exception:
            raise

    return torch.zeros(pv.size(0), IMG_FEAT_DIM, device=device)


# ---------------------------
# Utilities
# ---------------------------
def load_model_and_processor(model_name_or_path: str, local_dir: Optional[str] = None):
    """
    Load the model and processor/tokenizer. Mirrors behavior in training script.
    Returns (model, processor, tokenizer_if_any)
    """
    tokenizer = None
    if UNSLOTH_AVAILABLE:
        try:
            if local_dir and os.path.isdir(local_dir):
                model, tokenizer = FastVisionModel.from_pretrained(local_dir)
                processor = tokenizer
                print(f"Loaded unsloth model from local dir: {local_dir}")
                return model, processor, tokenizer
            else:
                model, tokenizer = FastVisionModel.from_pretrained(model_name_or_path)
                processor = tokenizer
                print(f"Loaded unsloth model: {model_name_or_path}")
                return model, processor, tokenizer
        except Exception as e:
            print("unsloth load failed, falling back to transformers path:", e)

    # Transformers path (Qwen3-VL)
    from transformers import Qwen3VLForConditionalGeneration
    if local_dir and os.path.isdir(local_dir):
        model = Qwen3VLForConditionalGeneration.from_pretrained(local_dir, device_map="auto", torch_dtype=torch.bfloat16)
        processor = AutoProcessor.from_pretrained(local_dir)
        tokenizer = processor.tokenizer
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16)
        processor = AutoProcessor.from_pretrained(model_name_or_path)
        tokenizer = processor.tokenizer

    return model, processor, tokenizer


def build_conversation_and_inputs(processor, image: Image.Image, user_prompt: str, add_generation_prompt: bool):
    """
    Build conversation with system prompt and user image+prompt.
    add_generation_prompt: pass True to add assistant generation token (i.e., ready for generation)
    Returns tokenized inputs (dict of tensors).
    """
    system_instruction = (
        "You are an assistant that generates detailed, accurate descriptions "
        "of satellite and aerial imagery."
    )
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_prompt}
        ]},
    ]
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=add_generation_prompt)
    inputs = processor(text=prompt, images=[image], return_tensors='pt')
    return inputs


def build_teacher_forcing_inputs(processor, image: Image.Image, user_prompt: str, reference_text: str):
    """
    Build inputs that include the assistant reference text (teacher-forcing).
    Returns:
      inputs: tokens/ pixel_values as returned by processor for conversation+assistant
      prefix_length: length of prefix token portion to be masked (-100) in labels
      labels: labels tensor where prefix tokens are -100 and assistant tokens contain ids
    """
    # conversation with assistant reference included
    system_instruction = (
        "You are an assistant that generates detailed, accurate descriptions "
        "of satellite and aerial imagery."
    )
    conv_with_assistant = [
        {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_prompt}
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": reference_text}]}
    ]
    inputs = processor(text=processor.apply_chat_template(conv_with_assistant, tokenize=False, add_generation_prompt=False),
                       images=[image], return_tensors='pt')

    # compute prefix length (without assistant response) using add_generation_prompt=True
    conv_prefix = [
        {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_prompt}
        ]},
    ]
    prefix_prompt = processor.apply_chat_template(conv_prefix, tokenize=False, add_generation_prompt=True)
    prefix_inputs = processor(text=prefix_prompt, images=[image], return_tensors='pt')
    prefix_len = prefix_inputs['input_ids'].shape[1]

    labels = inputs['input_ids'].clone()
    labels[0, :prefix_len] = -100

    return inputs, prefix_len, labels


def decode_tokens(processor, tokenizer, token_ids):
    """
    Decode token ids to text using tokenizer (or processor).
    token_ids can be list[int] or torch tensor 1D.
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    if tokenizer is not None:
        try:
            return tokenizer.decode(token_ids, skip_special_tokens=True)
        except Exception:
            # fallback to convert tokens one by one
            try:
                toks = tokenizer.convert_ids_to_tokens(token_ids)
                return tokenizer.convert_tokens_to_string(toks)
            except Exception:
                return " ".join(map(str, token_ids))
    elif hasattr(processor, "decode"):
        try:
            return processor.decode(token_ids, skip_special_tokens=True)
        except Exception:
            return " ".join(map(str, token_ids))
    else:
        return "<no tokenizer available>"


def decode_single_token(tokenizer, token_id):
    """
    Decode a single token id to readable string; safe fallback if tokenizer missing.
    """
    try:
        if tokenizer is None:
            return str(token_id)
        # try convert token then decode
        tok = tokenizer.convert_ids_to_tokens([int(token_id)])
        txt = tokenizer.decode([int(token_id)], skip_special_tokens=True)
        if txt is None or txt.strip() == "":
            return tok[0]
        return txt
    except Exception:
        return str(token_id)

# ---------------------------
# Main
# ---------------------------
def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"ERROR: IMAGE_PATH not found: {IMAGE_PATH}")
        sys.exit(1)

    print("Loading model and processor...")
    model, processor, tokenizer = load_model_and_processor(MODEL_NAME_OR_PATH, LOCAL_MODEL_DIR)

    try:
        model.to(DEVICE)
    except Exception:
        pass
    model.eval()

    # Print model device/dtype info
    try:
        model_device = next(model.parameters()).device
        print(f"Model device: {model_device}")
    except StopIteration:
        model_device = DEVICE
        print(f"Model parameters not found; using DEVICE: {DEVICE}")

    # Quick model sanity prints
    try:
        print(f"Model dtype for first param: {next(model.parameters()).dtype}")
    except Exception:
        pass

    image = Image.open(IMAGE_PATH).convert("RGB")

    # For teacher-forcing mode we will need labels; otherwise for generation use prefix inputs
    if REFERENCE_TEXT is not None:
        # build inputs including assistant reference
        inputs_with_ref, prefix_len, labels = build_teacher_forcing_inputs(processor, image, USER_PROMPT, REFERENCE_TEXT)
        pixel_values = inputs_with_ref.get('pixel_values', None)
        if pixel_values is None:
            raise RuntimeError("processor did not return pixel_values for the image; cannot extract vision features")
    else:
        # build prefix inputs (ready for generation) - we'll autoregressively append tokens
        inputs_prefix = build_conversation_and_inputs(processor, image, USER_PROMPT, add_generation_prompt=True)
        pixel_values = inputs_prefix.get('pixel_values', None)
        if pixel_values is None:
            raise RuntimeError("processor did not return pixel_values for the image; cannot extract vision features")

    # extract image features
    pixel_values = pixel_values.to(DEVICE)
    with torch.no_grad():
        img_feats = get_image_features_from_model(model, pixel_values)
    if img_feats is None:
        raise RuntimeError("Failed to extract image features from model")
    if img_feats.dim() == 1:
        img_feats = img_feats.unsqueeze(0)

    print(f"Image features shape: {img_feats.shape}, dtype: {img_feats.dtype}, device: {img_feats.device}")

    # prepare prompt learner and load weights
    embedding_layer = model.get_input_embeddings()
    embed_dim = embedding_layer.weight.size(1)
    print(f"Embedding layer found. embed_dim={embed_dim}, vocab_size={embedding_layer.weight.size(0)}")

    prompt_learner = PromptLearner(prompt_length=PROMPT_LENGTH, embed_dim=embed_dim, img_feat_dim=IMG_FEAT_DIM)
    if PROMPT_LEARNER_WEIGHTS and os.path.exists(PROMPT_LEARNER_WEIGHTS):
        state = torch.load(PROMPT_LEARNER_WEIGHTS, map_location="cpu")
        try:
            prompt_learner.load_state_dict(state)
            print(f"Loaded PromptLearner weights from {PROMPT_LEARNER_WEIGHTS}")
        except Exception as e:
            try:
                missing, unexpected = prompt_learner.load_state_dict(state, strict=False)
                print(f"Warning: loaded prompt learner with strict=False. missing keys: {missing}; unexpected keys: {unexpected}")
            except Exception as e2:
                print("Failed to load prompt learner state_dict:", e2)
    else:
        print("Warning: PromptLearner weights not found; using randomly initialized PromptLearner")

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = DEVICE
    embed_dtype = embedding_layer.weight.dtype
    # ensure prompt learner is on same device/dtype as embeddings
    adapter_weight_dtype = prompt_learner.adapter[0].weight.dtype if hasattr(prompt_learner.adapter[0], 'weight') else embed_dtype
    prompt_learner = prompt_learner.to(device=model_device, dtype=embed_dtype)
    prompt_learner.eval()

    # compute prompt embeddings
    img_feats = img_feats.to(device=model_device, dtype=prompt_learner.adapter[0].weight.dtype if hasattr(prompt_learner.adapter[0], 'weight') else embed_dtype)
    with torch.no_grad():
        prompt_embeds = prompt_learner(img_feats)  # (B, P, D)
    B, P, D = prompt_embeds.shape

    # Diagnostic prints for prompt embeddings
    print(f"Prompt embeddings shape: {prompt_embeds.shape}, dtype: {prompt_embeds.dtype}, device: {prompt_embeds.device}")
    # print norm stats
    try:
        norms = prompt_embeds.norm(dim=-1)  # (B, P)
        print(f"Prompt embeddings norms: mean={norms.mean().item():.6f}, std={norms.std().item():.6f}, min={norms.min().item():.6f}, max={norms.max().item():.6f}")
    except Exception:
        pass
    # print a sample of the first prompt vector values (small slice)
    try:
        sample_first = prompt_embeds[0, 0, :min(8, D)].detach().cpu().numpy().tolist()
        print(f"Sample prompt vector (first prompt token, first 8 dims): {sample_first}")
    except Exception:
        pass

    if REFERENCE_TEXT is not None:
        # -----------------------
        # Teacher-forcing evaluation (use logits -> argmax like your training logging)
        # -----------------------
        input_ids = inputs_with_ref['input_ids'].to(model_device)
        attention_mask = inputs_with_ref['attention_mask'].to(model_device)
        labels = labels.to(model_device)

        print(f"Teacher-forcing: input_ids shape {input_ids.shape}, attention_mask shape {attention_mask.shape}, labels shape {labels.shape}")
        # show first few token ids (prefix)
        try:
            print("First 10 input_ids (prefix+assistant+...):", input_ids[0, :10].tolist())
        except Exception:
            pass

        # input embeddings for full sequence (system+user+assistant reference)
        input_embeds = embedding_layer(input_ids).to(model_device)
        print(f"Input embeddings shape: {input_embeds.shape}, dtype: {input_embeds.dtype}, device: {input_embeds.device}")

        # Prepend prompt embeds to input embeddings (same as training)
        if prompt_embeds.size(0) != input_embeds.size(0):
            if prompt_embeds.size(0) == 1:
                prompt_embeds_exp = prompt_embeds.expand(input_embeds.size(0), -1, -1)
                print("Expanded prompt_embeds to match batch size.")
            else:
                raise RuntimeError("Batch size mismatch between prompt embeddings and token embeddings")
        else:
            prompt_embeds_exp = prompt_embeds

        # Check that prompt embeds and token embeddings share same dim
        if prompt_embeds_exp.size(-1) != input_embeds.size(-1):
            raise RuntimeError(f"Embedding dim mismatch: prompt {prompt_embeds_exp.size(-1)} vs input {input_embeds.size(-1)}")

        inputs_embeds = torch.cat([prompt_embeds_exp, input_embeds], dim=1)  # (B, P + L, D)
        print(f"Inputs_embeds after concat shape: {inputs_embeds.shape}")

        # attention mask: prefix ones + original
        prefix_mask = torch.ones((input_embeds.size(0), P), dtype=attention_mask.dtype, device=model_device)
        attention_mask_with_prefix = torch.cat([prefix_mask, attention_mask], dim=1)
        print(f"Attention mask after prefix concat shape: {attention_mask_with_prefix.shape}")

        # labels need to include -100 for prompt positions as done in training
        prompt_labels = torch.full((labels.size(0), P), -100, dtype=labels.dtype, device=model_device)
        labels_with_prefix = torch.cat([prompt_labels, labels], dim=1)
        print(f"Labels with prefix shape: {labels_with_prefix.shape}. Prefix length (P) = {P}")

        # forward pass (no generation) to get logits
        with torch.no_grad():
            outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask_with_prefix)
        logits = outputs.logits  # (B, S, V)
        print(f"Logits shape: {logits.shape}")

        # compute shift logits/labels exactly as in training
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels_with_prefix[..., 1:].contiguous()
        print(f"shift_logits shape: {shift_logits.shape}, shift_labels shape: {shift_labels.shape}")

        # predicted ids (greedy)
        predicted_ids = torch.argmax(shift_logits, dim=-1)  # (B, S-1)

        # valid mask where shift_labels != -100
        valid_mask = (shift_labels != -100)
        if valid_mask.any():
            sample_mask = valid_mask[0]
            if sample_mask.any():
                pred_tokens = predicted_ids[0][sample_mask]
                label_tokens = shift_labels[0][sample_mask]

                # Use tokenizer/processor to decode first up to 512 tokens
                try:
                    pred_text = decode_tokens(processor, tokenizer, pred_tokens[:512])
                    label_text = decode_tokens(processor, tokenizer, label_tokens[:512])
                except Exception:
                    # fallback: join ids
                    pred_text = " ".join(map(str, pred_tokens.tolist()))
                    label_text = " ".join(map(str, label_tokens.tolist()))

                correct = (pred_tokens == label_tokens).sum().item()
                total = sample_mask.sum().item()
                accuracy = 100.0 * correct / total

                print(f"\n[Teacher-forcing evaluation] Acc: {accuracy:.2f}% ({correct}/{total})")
                print("Pred (first 1000 chars):")
                print(pred_text[:1000])
                print("Ref (first 1000 chars):")
                print(label_text[:1000])

                # Additional diagnostics: print first 10 predicted token ids and their decoded tokens
                try:
                    print("\nFirst up to 20 predicted token ids and decoded tokens:")
                    for i, tid in enumerate(pred_tokens[:20].tolist()):
                        dec = decode_single_token(tokenizer, tid)
                        lab = decode_single_token(tokenizer, label_tokens[:20].tolist()[i])
                        print(f"  idx {i}: id={tid} -> '{dec}'   (label='{lab}')")
                except Exception:
                    pass
        else:
            print("No valid label tokens found to evaluate against (mask all -100).")

    else:
        # -----------------------
        # Manual greedy autoregressive decoding (no model.generate)
        # -----------------------
        input_ids = inputs_prefix['input_ids'].to(model_device)
        attention_mask = inputs_prefix['attention_mask'].to(model_device)

        print(f"Generation mode: prefix input_ids shape {input_ids.shape}, attention_mask shape {attention_mask.shape}")
        try:
            print("Prefix first 20 token ids:", input_ids[0, :20].tolist())
            if tokenizer is not None:
                try:
                    print("Prefix decoded (first 200 chars):", decode_tokens(processor, tokenizer, input_ids[0, :input_ids.shape[1]].tolist())[:200])
                except Exception:
                    pass
        except Exception:
            pass

        # starting input embeddings for prefix (system+user+assistant token)
        input_embeds = embedding_layer(input_ids).to(model_device)  # (B, L, D)
        print(f"Input embeddings shape: {input_embeds.shape}, dtype: {input_embeds.dtype}")

        # prepend prompt embeddings
        if prompt_embeds.size(0) != input_embeds.size(0):
            if prompt_embeds.size(0) == 1:
                prompt_embeds_exp = prompt_embeds.expand(input_embeds.size(0), -1, -1)
                print("Expanded prompt_embeds to match batch size.")
            else:
                raise RuntimeError("Batch size mismatch between prompt embeddings and token embeddings")
        else:
            prompt_embeds_exp = prompt_embeds

        # Check dims before concat
        print(f"prompt_embeds_exp shape: {prompt_embeds_exp.shape}, input_embeds shape: {input_embeds.shape}")
        if prompt_embeds_exp.size(-1) != input_embeds.size(-1):
            raise RuntimeError(f"Embedding dimension mismatch: prompt {prompt_embeds_exp.size(-1)} vs input {input_embeds.size(-1)}")

        current_inputs_embeds = torch.cat([prompt_embeds_exp, input_embeds], dim=1).to(model_device)  # (B, P+L, D)
        prefix_mask = torch.ones((input_embeds.size(0), P), dtype=attention_mask.dtype, device=model_device)
        current_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1).to(model_device)

        print(f"current_inputs_embeds shape after concat: {current_inputs_embeds.shape}")
        print(f"current_attention_mask shape after concat: {current_attention_mask.shape}")

        # Optional sanity: ensure prompt positions correspond to prefix ones in mask
        print("Mask first 10 entries of attention mask (row 0):", current_attention_mask[0, :min(50, current_attention_mask.shape[1])].tolist()[:50])

        # manual greedy decode
        generated_ids = []
        for step in range(MAX_NEW_TOKENS):
            with torch.no_grad():
                out = model(inputs_embeds=current_inputs_embeds, attention_mask=current_attention_mask)
                logits = out.logits  # (B, S, V)

            # get last token logits
            last_logits = logits[:, -1, :]  # (B, V)
            next_ids = torch.argmax(last_logits, dim=-1)  # (B,)

            # append generated token ids
            next_id = int(next_ids[0].item())
            generated_ids.append(next_id)

            # Decode the token to string for logging
            decoded_token_str = decode_single_token(tokenizer, next_id)
            # Also show top-k token ids for context (k=5)
            try:
                topk_vals, topk_ids = torch.topk(last_logits[0], k=5)
                topk_info = [(int(tid.item()), float(val.item())) for val, tid in zip(topk_vals, topk_ids)]
            except Exception:
                topk_info = None

            # print(f"[step {step}] next_id={next_id} -> '{decoded_token_str}' | topk={topk_info}")

            # if EOS and EOS_TOKEN_ID set, break
            if EOS_TOKEN_ID is not None and next_id == EOS_TOKEN_ID:
                print(f"Hit EOS_TOKEN_ID at step {step}. Stopping decode.")
                break

            # append embedding of next token to inputs_embeds
            next_token_embed = embedding_layer(torch.tensor([next_id], device=model_device)).unsqueeze(1)  # (1, 1, D)
            # Ensure dtype/device match
            next_token_embed = next_token_embed.to(device=model_device, dtype=current_inputs_embeds.dtype)
            current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embed], dim=1)

            # update attention mask
            new_mask_col = torch.ones((current_attention_mask.size(0), 1), dtype=current_attention_mask.dtype, device=model_device)
            current_attention_mask = torch.cat([current_attention_mask, new_mask_col], dim=1)

            # small diagnostic: print shapes occasionally
            if (step + 1) % 20 == 0:
                print(f"After step {step+1}: current_inputs_embeds.shape = {current_inputs_embeds.shape}, attention_mask.shape = {current_attention_mask.shape}")

        # decode generated ids
        try:
            gen_text = decode_tokens(processor, tokenizer, generated_ids)
        except Exception:
            gen_text = " ".join(map(str, generated_ids))

        print("\n=== Generated (greedy manual) ===")
        print(gen_text)
        print("===============================")


if __name__ == "__main__":
    main()
