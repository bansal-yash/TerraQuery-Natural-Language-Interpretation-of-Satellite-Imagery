import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import cv2
import numpy as np
import supervision as sv
from transformers import (
    CLIPProcessor,
    CLIPModel,
    AutoProcessor, 
    Qwen3VLForConditionalGeneration
)
from groundingdino.util.inference import Model
from segment_anything import build_sam, SamPredictor

# ---------------- CONFIG ----------------
CONFIG_PATH = "/home/scratch/samyak/interiit/samyak/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "/home/samyak/scratch/interiit/samyak/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
SAM_PATH = "/home/samyak/scratch/interiit/samyak/Grounded-Segment-Anything/sam_vit_h_4b8939.pth"
IMAGE_PATH = "Inputs/WhatsApp Image 2025-11-11 at 01.21.33_43ef1d0c.jpg"

DEVICE = "cpu" if torch.cuda.is_available() else "cpu"
os.makedirs("outputs", exist_ok=True)

# ---------------- MODEL LOADERS ----------------
print("🚀 Loading models (please wait)...")

# 1. CLIP for attribute extraction (safetensors = no torch 2.6 requirement)
vlm_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
vlm_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. Grounding DINO + SAM
dino_model = Model(CONFIG_PATH, WEIGHTS_PATH, device=DEVICE)
sam_model = build_sam(checkpoint=SAM_PATH).to(DEVICE)
sam_predictor = SamPredictor(sam_model)

# 3. Qwen LLM for caption generation
QWEN_VL_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
)
model.eval()

print("✅ All models loaded successfully.\n")

# ---------------- 1. VLM Attribute Extraction ----------------
def extract_attributes(image_bgr, candidates):
    inputs = vlm_processor(text=candidates, images=image_bgr, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = vlm_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
    top_indices = probs.topk(min(5, len(candidates))).indices.tolist()
    return [candidates[i] for i in top_indices]

# ---------------- 2. Detection + Segmentation ----------------
def detect_and_segment(objects, image_bgr, iter_idx=0):
    detections, phrases = dino_model.predict_with_caption(
        image=image_bgr,
        caption=", ".join(objects),
        box_threshold=0.25,
        text_threshold=0.20,
    )

    if len(phrases) == 0 or detections.xyxy.shape[0] == 0:
        print("⚠ No detections found for:", objects)
        return None, None, []

    sam_predictor.set_image(image_bgr, image_format="BGR")
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        torch.tensor(detections.xyxy), image_bgr.shape[:2]
    ).to(DEVICE)

    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False
    )

    # --- Annotate and Save Masked Image ---
    final_masks = masks.squeeze(1).cpu().numpy()
    detections.mask = final_masks

    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
    annotated = box_annotator.annotate(scene=annotated, detections=detections)

    out_path = f"outputs/segmented_iter_{iter_idx+1}.png"
    cv2.imwrite(out_path, annotated)
    print(f"🖼 Saved segmented image: {out_path}")

    return detections, masks, phrases

# ---------------- 3. Context Builder ----------------
def build_context(detections, phrases):
    ctx = []
    for (x1, y1, x2, y2), label in zip(detections.xyxy, phrases):
        ctx.append(f"{label} at ({int(x1)}, {int(y1)})")
    return " | ".join(ctx)

# ---------------- 4. Caption Generation (Qwen) ----------------
def generate_caption_with_qwen(image_bgr, context: str = "", max_new_tokens: int = 1000):
    """
    Generate a caption using Qwen3-VL-8B-Instruct with both image and text context.
    """

    # Convert image to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Compose instruction
    # if context.strip():
    user_prompt = f"Describe this image carefully, and center it only around the following context: {context}"
    # else:
    #     user_prompt = "Describe this image in detail."

    # Qwen3 expects messages in chat format (like ChatGPT)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_rgb,
                },
                {"type": "text", "text": user_prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

# ---------------- 5. Iterative Loop ----------------
def iterative_captioning(image_bgr, candidates, iters=2):
    caption = ""
    for i in range(iters):
        print(f"\n🔁 Iteration {i+1}")

        # Step 1: Extract attributes
        attributes = extract_attributes(image_bgr, candidates)
        print("🎯 VLM attributes:", attributes)

        # Step 2: Detect + Segment
        detections, masks, phrases = detect_and_segment(attributes, image_bgr, iter_idx=i)
        if detections is None:
            print("Skipping iteration (no valid detections).")
            caption = generate_caption_with_qwen(image_bgr, context=candidates)
            print("💬 Generated caption:", caption)
            tokens = [t for t in caption.split() if t.isalpha()]
            candidates = list(set(candidates + tokens))
            continue

        # Step 3: Build context
        context = build_context(detections, phrases)
        print("🧩 Context:", context)

        # Step 4: Generate caption
        caption = generate_caption_with_qwen(image_bgr, context=candidates)
        print("💬 Generated caption:", caption)

        # Step 5: Update candidate list
        tokens = [t for t in caption.split() if t.isalpha()]
        candidates = list(set(candidates + tokens))

    return caption

# ---------------- RUN ----------------
image_bgr = cv2.imread(IMAGE_PATH)
base_candidates = [""]

final_caption = iterative_captioning(image_bgr, base_candidates, iters=5)

print("\n✅ Final caption:", final_caption)
