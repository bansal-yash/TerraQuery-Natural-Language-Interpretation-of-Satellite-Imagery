#!/usr/bin/env python3
"""
type_classifier_safe.py

Safe model loader & predictor:
 - Loads checkpoint to CPU (avoid CUDA allocations during torch.load)
 - Attempts to move model to requested GPU index (or finds first usable GPU)
 - Falls back to CPU on failure
 - CLI flags: --checkpoint, --image, --gpu-index, --force-cpu
"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# -------------------------
# CONFIG
# -------------------------
CLASS_NAMES = ["sar", "rgb", "falsecolor"]

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------
# UTIL: check GPU usability
# -------------------------
def is_gpu_usable(index=0, try_alloc_bytes=1024 * 1024):
    """
    Check if GPU index is usable by:
      - verifying index < device_count
      - attempting a tiny allocation on that device
    Returns True if usable, False otherwise.
    """
    if not torch.cuda.is_available():
        return False
    try:
        count = torch.cuda.device_count()
    except Exception:
        return False
    if index < 0 or index >= count:
        return False
    try:
        # try a tiny tensor allocation and sync
        dev = torch.device(f"cuda:{index}")
        t = torch.empty(1, device=dev)
        # optional: cuda synchronize to surface errors early
        torch.cuda.synchronize(dev)
        del t
        return True
    except Exception:
        return False


# -------------------------
# MODEL BUILDING & LOADING
# -------------------------
def build_model(num_classes=len(CLASS_NAMES)):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model


def clean_state_dict_keys(state_dict):
    # common replacements people need when saving with wrappers
    new_state = {}
    for k, v in state_dict.items():
        new_k = k
        # strip some common prefixes
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        new_k = new_k.replace("_orig_mod.", "")
        new_state[new_k] = v
    return new_state


def load_model(checkpoint_path, target_device=None, prefer_gpu_index=None):
    """
    Load checkpoint safely:
      - load checkpoint to CPU first
      - extract state_dict if necessary
      - load into model
      - then move model to target_device (which can be 'cpu' or 'cuda:X')
    Returns: model, device_used (torch.device)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # build model architecture
    model = build_model(num_classes=len(CLASS_NAMES))

    # load checkpoint to CPU (prevent attempt to allocate on CUDA)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # some checkpoints store dict with 'state_dict'
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        state_dict = checkpoint["state_dict"]
    else:
        # assume checkpoint itself is a state_dict
        state_dict = checkpoint

    # clean keys
    state_dict = clean_state_dict_keys(state_dict)

    # load weights
    model.load_state_dict(state_dict)

    # Decide device to move model to
    device_used = torch.device("cpu")
    if target_device is None:
        # default behaviour: prefer GPU if available and usable
        if prefer_gpu_index is not None and is_gpu_usable(prefer_gpu_index):
            device_used = torch.device(f"cuda:{prefer_gpu_index}")
        else:
            # find first usable GPU
            if torch.cuda.is_available():
                for idx in range(torch.cuda.device_count()):
                    if is_gpu_usable(idx):
                        device_used = torch.device(f"cuda:{idx}")
                        break
    else:
        # target_device can be "cpu" or a torch.device
        if isinstance(target_device, torch.device):
            device_used = target_device
        else:
            try:
                device_used = torch.device(target_device)
            except Exception:
                device_used = torch.device("cpu")

    # try moving model to chosen device; fallback to cpu on exception
    try:
        model.to(device_used)
    except Exception as e:
        print(f"Warning: failed to move model to {device_used}. Falling back to CPU. Error: {e}", file=sys.stderr)
        device_used = torch.device("cpu")
        model.to(device_used)

    model.eval()
    return model, device_used


# -------------------------
# PREDICTION
# -------------------------
def predict(image_path, model, device):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()

    pred_idx = int(np.argmax(probs))
    return CLASS_NAMES[pred_idx], float(probs[pred_idx]), probs


# -------------------------
# CLI / MAIN
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Classify SAR / RGB / False-Color images (safe loader)")
    p.add_argument("-c", "--checkpoint", required=True, help="Path to model checkpoint")
    p.add_argument("-i", "--image", required=True, help="Path to input image")
    p.add_argument("--gpu-index", type=int, default=None, help="Preferred GPU index (e.g. 0). If omitted, script will pick the first usable GPU.")
    p.add_argument("--force-cpu", action="store_true", help="Force CPU only (do not attempt to use GPU)")
    p.add_argument("--no-sync", action="store_true", help="Do not attempt cuda.synchronize during gpu check (slightly faster)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.force_cpu:
        target_device = torch.device("cpu")
        print("Force CPU mode enabled (--force-cpu).")
    else:
        # Try to pick a GPU if available and usable
        if args.gpu_index is not None:
            if is_gpu_usable(args.gpu_index):
                target_device = torch.device(f"cuda:{args.gpu_index}")
                print(f"Preferred GPU {args.gpu_index} is usable; will attempt to use it.")
            else:
                print(f"Preferred GPU {args.gpu_index} is NOT usable. Falling back to searching for other GPUs or CPU.")
                target_device = None
        else:
            # no explicit preference; let load_model pick the first usable GPU
            target_device = None

    # Load model (safe: checkpoint loaded to CPU first)
    try:
        model, device_used = load_model(args.checkpoint, target_device=target_device, prefer_gpu_index=args.gpu_index)
    except Exception as e:
        # If something unexpected fails, try a strict CPU-only retry once
        print(f"Error loading model: {e}", file=sys.stderr)
        print("Retrying load on CPU only...")
        model, device_used = load_model(args.checkpoint, target_device=torch.device("cpu"))

    print(f"Model loaded. Device in use: {device_used}")

    # Predict
    pred_class, conf, full_probs = predict(args.image, model, device_used)

    # Print results
    print("\n====================================")
    print("📌 PREDICTION RESULT")
    print("====================================")
    print(f"Image Path        : {args.image}")
    print(f"Checkpoint        : {args.checkpoint}")
    print(f"Device used       : {device_used}")
    print(f"Predicted Class   : {pred_class}")
    print(f"Confidence        : {conf:.6f}")
    print(f"All Probabilities : {full_probs}")

    # If running on GPU, optionally show memory usage
    if str(device_used).startswith("cuda"):
        try:
            idx = int(str(device_used).split(":")[-1])
            mem_total = torch.cuda.get_device_properties(idx).total_memory
            mem_reserved = torch.cuda.memory_reserved(idx)
            mem_alloc = torch.cuda.memory_allocated(idx)
            print("\nGPU memory (bytes):")
            print(f"  total   : {mem_total:,}")
            print(f"  reserved: {mem_reserved:,}")
            print(f"  allocated: {mem_alloc:,}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
