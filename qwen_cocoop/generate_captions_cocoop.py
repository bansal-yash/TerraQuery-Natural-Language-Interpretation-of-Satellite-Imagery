#!/usr/bin/env python3
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import os
import sys

# ---------------------------------------------------------
# PromptLearner (must match EXACTLY the training definition)
# ---------------------------------------------------------
class PromptLearner(nn.Module):
    def __init__(self, prompt_length: int, embed_dim: int, img_feat_dim: int = 1536, hidden_dim: int = 512):
        super().__init__()
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.img_feat_dim = img_feat_dim
        
        self.base_prompt = nn.Parameter(torch.randn(prompt_length, embed_dim) * 0.02)
        self.adapter = nn.Sequential(
            nn.Linear(img_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, prompt_length * embed_dim),
        )

    def forward(self, image_features: torch.Tensor):
        # image_features: (B, F)
        batch = image_features.size(0)

        print("image features shape: ", image_features.shape)
        delta = self.adapter(image_features)            # (B, P*D)
        delta = delta.view(batch, self.prompt_length, self.embed_dim)  # (B, P, D)
        base = self.base_prompt.unsqueeze(0).expand(batch, -1, -1)     # (B, P, D)
        return base + delta


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


# ---------------------------------------------------------
# Load model + processor + prompt learner
# ---------------------------------------------------------
def load_cocoop_model(model_dir, prompt_length=16, img_feat_dim=1536):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Qwen3-VL model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16
    ).to(device)

    
    processor = AutoProcessor.from_pretrained(model_dir)

    embed_dim = model.get_input_embeddings().weight.size(1)
    embed_dtype = model.get_input_embeddings().weight.dtype

    print("Loading PromptLearner...")
    prompt_learner = PromptLearner(prompt_length, embed_dim, img_feat_dim)
    prompt_learner.load_state_dict(torch.load(os.path.join(model_dir, "prompt_learner.pt")))
    prompt_learner = prompt_learner.to(device=device, dtype=torch.bfloat16)
    prompt_learner.eval()

    # freeze model (as in training)
    for p in model.parameters():
        p.requires_grad = False

    return model, processor, prompt_learner


# ---------------------------------------------------------
# Generate caption using CoCoOp prompts
# ---------------------------------------------------------
@torch.no_grad()
def generate_caption(model, processor, prompt_learner, image_path):
    device = next(model.parameters()).device

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Build evaluation conversation
    conv = [
        {"role": "system", "content": [{"type":"text","text":"You are an assistant that generates detailed, accurate descriptions of satellite and aerial imagery."}]},
        {"role": "user", "content": [{"type":"image","image":image}, {"type":"text","text":"Please describe this image in detail."}]}
    ]

    # Tokenize without assistant answer
    prompt = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    # inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = processor(
        images=[image],
        text=prompt,
        return_tensors="pt"
    ).to(model.device)


    print(inputs["pixel_values"].shape)


    
    # device = next(model.parameters()).device

    # # Pop debug metadata
    # image_paths = inputs.pop('image_path', None)
    # image_sizes = inputs.pop('image_size', None)


    # pixel_values = inputs.pop('pixel_values').to(device)

    
    # # Extract image features (no grad for vision encoder)
    # with torch.no_grad():
    #     img_feats = get_image_features_from_model(model, pixel_values)
    
    # if img_feats is None:
    #     raise RuntimeError("get_image_features returned None")
    
    # # Ensure batch dim
    # if img_feats.dim() == 1:
    #     img_feats = img_feats.unsqueeze(0)

    # # Move to prompt learner device/dtype
    # prompt_dev = next(prompt_learner.parameters()).device
    # prompt_dtype = next(prompt_learner.parameters()).dtype
    # img_feats = img_feats.to(device=prompt_dev, dtype=prompt_dtype)


    # # Generate prompt embeddings (this is where gradients matter!)
    # prompt_embeds = prompt_learner(img_feats).to(device)  # (B, P, D)
    # B, P, D = prompt_embeds.shape

    # # Prepare text inputs
    # input_ids = inputs.pop('input_ids').to(device)
    # attention_mask = inputs.pop('attention_mask').to(device)

    # # Get input embeddings
    # embedding_layer = model.get_input_embeddings()

    # input_embeds = embedding_layer(input_ids).to(device)  # (B, L, D)

    # # Concatenate: [soft prompts | input tokens]
    # inputs_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)  # (B, P+L, D)

    # # Concatenate soft prompts + text
    # # inputs_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)

    # # Adjust attention mask
    # B, P, _ = prompt_embeds.shape
    # prefix_mask = torch.ones((B, P), dtype=inputs["attention_mask"].dtype, device=device)
    # attn_mask = torch.cat([prefix_mask, inputs["attention_mask"].to(device)], dim=1)

    # Run generation using the injected embeddings

    print(model)
    inputs.pop('pixel_values')

    # print(inputs['pixel_values'].shape)
    print(inputs.keys())

    out = model.generate(
        # inputs_embeds=inputs_embeds,
        # attention_mask=attn_mask,
        **inputs,
        max_new_tokens=150,
        num_beams=3,
        do_sample=False
    )
    print(out)
    sys.exit()

    # Decode
    text = processor.decode(out[0], skip_special_tokens=True)
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1].strip()

    return text


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    model_dir = "/home/samyak/scratch/interiit/qwen_cocoop/checkpoints_cocoop/final/"
    image_path = "/home/samyak/scratch/interiit/EarthMind-Bench/img/test/sar/img/dfc2023_test_P_0421.png"

    model, processor, prompt_learner = load_cocoop_model(model_dir)

    caption = generate_caption(model, processor, prompt_learner, image_path)
    print("\n=== GENERATED CAPTION ===\n")
    print(caption)
