#!/usr/bin/env python3
"""
Generate a caption / answer for an image using your fine-tuned Qwen3-VL (LoRA) model.

Usage:
    python generate_caption_simple.py \
      --checkpoint /path/to/checkpoint_or_final_merged \
      --image /path/to/image.png \
      --question "Describe the scene in one sentence." \
      --output ./caption.txt \
      --device cuda

      python infer.py \
  --checkpoint /home/samyak/scratch/interiit/qwen_model/ft/checkpoints/SARV2/checkpoint-540 \
  --image /home/samyak/scratch/interiit/GAURAV_BIG_DATA/SAR_BIG/pair_data/test/sar/img/dfc2023_test_P_0434.png \
  --question "Describe the image in detail." \
  --output /home/samyak/scratch/interiit/qwen_model/ft/caption.txt 

"""

import argparse
import os
import time
import torch
from PIL import Image, ImageDraw, ImageFont
from unsloth import FastVisionModel
from peft import PeftModel


def load_model(checkpoint_path=None, device="cuda"):
    """
    Load model and processor.
    - If checkpoint_path points to a LoRA adapter folder, load base model + adapter and merge.
    - If checkpoint_path points to a merged model folder (final_merged), load that directly.
    - If checkpoint_path is None or empty, load the base model.
    Returns: model, processor
    """
    base_model_name = "unsloth/Qwen3-VL-8B-Instruct"
    device_map = device if device != "cpu" else None

    if checkpoint_path:
        # Heuristic: if directory name contains 'merged' assume it's the merged model.
        if "merged" in os.path.basename(checkpoint_path).lower() or "final_merged" in checkpoint_path.lower():
            print(f"Loading merged model from: {checkpoint_path}")
            model, processor = FastVisionModel.from_pretrained(
                checkpoint_path,
                load_in_4bit=False,
                device_map=device_map,
                use_safetensors=True
            )
            model.eval()
            return model, processor

        # Otherwise load base model then adapter
        print(f"Loading base model: {base_model_name}")
        model, processor = FastVisionModel.from_pretrained(
            base_model_name,
            load_in_4bit=False,
            device_map=device_map,
            use_safetensors=True
        )

        print(f"Loading LoRA adapters from: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)

        # Merge adapters if available
        if hasattr(model, "merge_and_unload"):
            print("Merging LoRA adapters for faster inference...")
            model = model.merge_and_unload()
        else:
            print("Warning: merge_and_unload() not available; running with adapters attached.")

        model.eval()
        return model, processor

    # No checkpoint => baseline
    print(f"Loading baseline base model: {base_model_name}")
    model, processor = FastVisionModel.from_pretrained(
        base_model_name,
        load_in_4bit=False,
        device_map=device_map,
        use_safetensors=True
    )
    model.eval()
    return model, processor


def generate_caption(model, processor, image_path, question, device="cuda", max_new_tokens=250):
    """
    Generate caption/answer for provided image + question.
    Returns the decoded text (string).
    """
    # load image
    image = Image.open(image_path).convert("RGB")

    # Build chat-style multimodal message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    # Apply chat template and tokenize for generation
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    # Move inputs to model device
    inputs = inputs.to(model.device)

    # Generate
    start = time.time()
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    elapsed = time.time() - start

    # Trim prompt tokens from generated ids
    # Some processor variants return inputs.input_ids as tensor inside the returned BatchEncoding-like object
    try:
        prompt_len = inputs.input_ids.shape[-1]
    except Exception:
        # fallback if structure differs
        prompt_len = gen_ids.shape[1] // 2

    output_ids = gen_ids[:, prompt_len:]
    decoded = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    caption = decoded[0] if decoded else ""

    return caption.strip(), elapsed


def optionally_draw_caption_on_image(image_path, caption, output_image_path, font_size=20):
    """Draw caption text at bottom of image and save."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    W, H = image.size
    # wrap text if too long
    margin = 8
    text = caption
    # compute text size
    tw, th = draw.textsize(text, font=font)
    # background rectangle
    rect_h = th + 2 * margin
    rect_y0 = H - rect_h
    draw.rectangle([(0, rect_y0), (W, H)], fill=(0, 0, 0))
    draw.text((margin, rect_y0 + margin // 2), text, fill=(255, 255, 255), font=font)
    image.save(output_image_path)
    return output_image_path


def main():
    parser = argparse.ArgumentParser(description="Generate caption/answer for image with fine-tuned Qwen3-VL")
    parser.add_argument("--checkpoint", help="Path to LoRA checkpoint or merged model folder (omit for baseline)", default=None)
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--question", required=False, default="Describe the image in one sentence.", help="Caption/question text")
    parser.add_argument("--output", required=False, help="Path to save caption text (default: <image>_caption.txt)")
    parser.add_argument("--output-image", required=False, help="Optional: save image with caption overlaid")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--max-new-tokens", type=int, default=250)
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    # set default output path if not provided
    if not args.output:
        base = os.path.splitext(os.path.basename(args.image))[0]
        args.output = os.path.join(os.path.dirname(args.image) or ".", f"{base}_caption.txt")

    model, processor = load_model(args.checkpoint, device=args.device)

    print(f"Running generation for: {args.image}")
    caption, elapsed = generate_caption(model, processor, args.image, args.question, device=args.device, max_new_tokens=args.max_new_tokens)

    print(f"\nGeneration time: {elapsed:.2f}s")
    print("Caption / Answer:")
    print(caption)

    # save caption to file
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(caption + "\n")
    print(f"Saved caption to: {args.output}")

    # optionally write caption on image
    if args.output_image:
        out_img = optionally_draw_caption_on_image(args.image, caption, args.output_image)
        print(f"Saved image with caption: {out_img}")


if __name__ == "__main__":
    main()
