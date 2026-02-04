#!/usr/bin/env python3
"""Utility to build a "null" LoRA adapter for Qwen3-VL models.

The generated adapter mirrors the target LoRA configuration but contains
zero-initialized updates. Loading it keeps the base model's behavior while
allowing fast adapter swaps (via set_adapter/delete_adapter) without cloning
full model weights in memory.

Example:
    python scripts/create_null_adapter.py \
        --base-model Qwen/Qwen3-VL-8B-Instruct \
        --template-adapter ../qwen_model/ft/checkpoints/SARV2/final \
        --output ./artifacts/null_adapter \
        --dtype bf16 --device cpu
"""

import argparse
import os
import sys
import torch
from transformers import Qwen3VLForConditionalGeneration
from peft import PeftConfig, LoraConfig, get_peft_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a zeroed/base LoRA adapter")
    parser.add_argument("--base-model", required=True, help="HF repo or local path of the base model")
    parser.add_argument(
        "--template-adapter",
        required=True,
        help="Existing LoRA adapter path whose config should be mirrored",
    )
    parser.add_argument("--output", required=True, help="Destination directory for the null adapter")
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Precision to use when instantiating the base model",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to instantiate the base weights on while exporting",
    )
    parser.add_argument(
        "--rank",
        type=int,
        help="Optional override for LoRA rank (defaults to template value)",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        help="Optional override for lora_alpha (defaults to template value)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        help="Optional override for lora_dropout (defaults to template value)",
    )
    parser.add_argument(
        "--target-modules",
        nargs="+",
        help="Override list of target modules if you are not mirroring the template",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code when loading the base model (needed for Qwen3-VL)",
    )
    return parser.parse_args()


def build_lora_config(args: argparse.Namespace) -> LoraConfig:
    cfg = PeftConfig.from_pretrained(args.template_adapter)
    if not isinstance(cfg, LoraConfig):
        raise ValueError("Template adapter is not LoRA-based; cannot build null adapter")

    cfg_dict = cfg.to_dict()
    if args.rank is not None:
        cfg_dict["r"] = args.rank
    if args.alpha is not None:
        cfg_dict["lora_alpha"] = args.alpha
    if args.dropout is not None:
        cfg_dict["lora_dropout"] = args.dropout
    if args.target_modules is not None:
        cfg_dict["target_modules"] = args.target_modules

    cfg_dict["inference_mode"] = True
    cfg_dict["bias"] = cfg_dict.get("bias", "none")
    cfg_dict.pop("base_model_name_or_path", None)
    return LoraConfig(**cfg_dict)


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    marker = os.path.join(path, "adapter_config.json")
    if os.path.exists(marker):
        print(f"[WARN] Output directory {path} already contains an adapter; files will be overwritten.")


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output)

    lora_cfg = build_lora_config(args)

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]
    device_map = "auto" if args.device == "cuda" else None

    print(f"[NullAdapter] Loading base model {args.base_model} on {args.device} with dtype={args.dtype} …")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    print("[NullAdapter] Injecting zero-initialized LoRA modules …")
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.eval()

    print(f"[NullAdapter] Saving adapter to {args.output} …")
    peft_model.save_pretrained(args.output)

    print("[NullAdapter] Done. Use this directory as --base-adapter for LocalVLM to enable fast swaps.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
