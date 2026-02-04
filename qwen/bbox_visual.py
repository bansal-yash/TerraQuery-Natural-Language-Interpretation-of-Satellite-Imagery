import re
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

# =====================================================================
# CONFIG – EDIT THESE PATHS
# =====================================================================
BASE_MODEL_DIR = "/home/spandan/scratch/interiit/qwen/small_spandan"
LORA_DIR = "/home/spandan/scratch/interiit/qwen/checkpoints_tim2/checkpoint-195"

IMAGE_PATH = "/home/spandan/scratch/interiit/EarthMind-Bench/img/test/rgb/img/dfc2025_TrainArea_333.png"
OUTPUT_PATH = "bbox_output.png"

QUESTION = "bounding box of central open space."

USE_4BIT = False   # set True only if your LoRA was trained with 4-bit


# =====================================================================
# 1. Load ONLY fine-tuned model (base + LoRA)
# =====================================================================
def load_model_only_lora():
    print("\nLoading base model...")
    if USE_4BIT:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            BASE_MODEL_DIR,
            device_map="auto",
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            BASE_MODEL_DIR,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    print("Loading LoRA adapter...")
    # model = PeftModel.from_pretrained(model, LORA_DIR)
    model.eval()

    processor = AutoProcessor.from_pretrained(BASE_MODEL_DIR)

    print("Model + LoRA loaded successfully.")
    return model, processor


# =====================================================================
# 2. Ask the model for bounding boxes
# =====================================================================
def ask(model, processor, image, question):
    system_prompt = (
        "You output ONLY bounding boxes in the format: "
        "<ref>label</ref><box>(x1,y1),(x2,y2)</box>"
    )

    user_prompt = f"""
{question}

Follow EXACT format:
<ref>label</ref><box>(x1,y1),(x2,y2)</box>
Coordinates MUST be integers in [0,1000].
One object per line.
No extra text.
""".strip()
    # object_name = "all digits"

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_prompt},
        ]}
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        images=[image],
        text=[prompt],
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            num_beams=1
        )

    # remove prompt tokens
    trimmed = out[:, inputs["input_ids"].shape[-1]:]
    text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    print("\nRaw model output:")
    print(text)
    return text


# =====================================================================
# 3. Parse bounding boxes
# =====================================================================
BOX_PAREN_RE = re.compile(
    r"<ref>(.*?)</ref>\s*<box>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*</box>",
    re.IGNORECASE,
)

BOX_SIMPLE_RE = re.compile(
    r"<ref>(.*?)</ref>\s*<box>\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*</box>",
    re.IGNORECASE,
)


# =====================================================================
# 3. Parse bounding boxes (more robust)
# =====================================================================
def parse_boxes(output_text):
    boxes = []
    
    # Clean up the text first
    text = output_text.strip()
    
    # Find all <ref>...</ref><box>...</box> patterns
    pattern = r'<ref>(.*?)</ref>\s*<box>(.*?)</box>'
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    
    for label, box_content in matches:
        # Clean up the box content
        box_content = box_content.strip()
        
        # Try different parsing strategies
        coords = None
        
        # Strategy 1: Try (x1,y1),(x2,y2) format
        match1 = re.search(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', box_content)
        if match1:
            coords = list(map(int, match1.groups()))
        
        # Strategy 2: Try x1,y1,x2,y2 format
        if not coords:
            match2 = re.search(r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', box_content)
            if match2:
                coords = list(map(int, match2.groups()))
        
        # Strategy 3: Try to extract any 4 numbers in various formats
        if not coords:
            # Find all numbers in the box content
            numbers = re.findall(r'\b(\d+)\b', box_content)
            if len(numbers) >= 4:
                coords = list(map(int, numbers[:4]))
        
        if coords:
            x1, y1, x2, y2 = coords
            # Ensure coordinates are in the correct order
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
                
            boxes.append(dict(label=label, x1=x1, y1=y1, x2=x2, y2=y2))
            print(f"Parsed: label='{label}', box={box_content} -> ({x1},{y1},{x2},{y2})")
    
    return boxes


# =====================================================================
# Alternative: Even simpler parser
# =====================================================================
def parse_boxes_simple(output_text):
    """Simpler parser that just looks for 4 numbers after each label"""
    boxes = []
    
    # Split by lines or by </ref> tags
    parts = re.split(r'</ref>', output_text)
    
    for part in parts:
        # Extract label
        label_match = re.search(r'<ref>(.*)', part)
        if not label_match:
            continue
            
        label = label_match.group(1).strip()
        
        # Extract all numbers from this part
        numbers = re.findall(r'\b(\d{1,3}|1000)\b', part)
        if len(numbers) >= 4:
            # Take the first 4 numbers as coordinates
            x1, y1, x2, y2 = map(int, numbers[:4])
            
            # Ensure coordinates are in the correct order
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
                
            boxes.append(dict(label=label, x1=x1, y1=y1, x2=x2, y2=y2))
            print(f"Parsed: '{label}' -> ({x1},{y1},{x2},{y2})")
    
    return boxes


# =====================================================================
# Use this in main()
# =====================================================================
def main():
    model, processor = load_model_only_lora()
    image = Image.open(IMAGE_PATH).convert("RGB")

    out_text = ask(model, processor, image, QUESTION)
    
    # Try both parsing methods
    boxes = parse_boxes(out_text)
    if not boxes:
        print("\nTrying alternative parser...")
        boxes = parse_boxes_simple(out_text)
    
    print("\nParsed boxes:")
    for b in boxes:
        print(b)

    if not boxes:
        print("\nNo valid bounding boxes detected.")
        print(f"Raw output: {out_text}")
        # Try to extract any coordinates manually
        numbers = re.findall(r'\b(\d+)\b', out_text)
        if len(numbers) >= 4:
            print(f"Found numbers: {numbers}")
            # Try to use them as coordinates
            x1, y1, x2, y2 = map(int, numbers[:4])
            boxes.append(dict(label="object", x1=x1, y1=y1, x2=x2, y2=y2))
            print(f"Using as box: {boxes[0]}")

    if boxes:
        draw(image, boxes, OUTPUT_PATH)


# =====================================================================
# 4. Draw bounding boxes on the image
# =====================================================================
# =====================================================================
# 4. Draw bounding boxes on the image (corrected)
# =====================================================================
def draw(image, boxes, save_path):
    w, h = image.size
    img = image.copy()
    d = ImageDraw.Draw(img)

    for b in boxes:
        # convert from [0–1000] to pixel coords
        x1 = int(b["x1"] / 1000 * w)
        y1 = int(b["y1"] / 1000 * h)
        x2 = int(b["x2"] / 1000 * w)
        y2 = int(b["y2"] / 1000 * h)  # Fixed: was using x2 for y2
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        
        # Ensure x1 <= x2 and y1 <= y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        d.rectangle([x1, y1, x2, y2], outline="red", width=3)
        d.text((x1, max(0, y1 - 12)), b["label"], fill="red")

    img.save(save_path)
    print(f"\nSaved annotated image to: {save_path}")
    return img  # Return the image for display if needed


# =====================================================================
# MAIN
# =====================================================================
# def main():
#     model, processor = load_model_only_lora()
#     image = Image.open(IMAGE_PATH).convert("RGB")

#     out_text = ask(model, processor, image, QUESTION)
#     boxes = parse_boxes(out_text)

#     print("\nParsed boxes:")
#     for b in boxes:
#         print(b)

#     if not boxes:
#         print("\nNo valid bounding boxes detected.")
#         return

#     draw(image, boxes, OUTPUT_PATH)


if __name__ == "__main__":
    main()
