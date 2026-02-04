from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

BASE_DIR = "/home/samyak/scratch/interiit/qwen/small_spandan"
CKPT_DIR = "/home/samyak/scratch/interiit/qwen/checkpoints_tim2/checkpoint-195"
IMG_PATH = "/home/samyak/scratch/interiit/GAURAV_BIG_DATA/SAR_BIG/pair_data/test/sar/img/dfc2023_test_P_0471.png"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(BASE_DIR)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    BASE_DIR,
    device_map="auto",
    load_in_4bit=True,          # same as training
    torch_dtype=torch.bfloat16, # or float16 if your GPU supports it
)
model = PeftModel.from_pretrained(model, CKPT_DIR)
model.eval()

image = Image.open(IMG_PATH).convert("RGB")

# 1) Build multimodal chat-style prompt with an image placeholder
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe the SAR image in detail."},
        ],
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# 2) Let the processor align image features and image tokens
inputs = processor(
    text=[text],
    images=[image],
    return_tensors="pt"
).to(device)

with torch.no_grad():
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=250,
        do_sample=False,
    )

# 3) Decode only the newly generated tokens (skip the prompt part)
output_ids = gen_ids[:, inputs["input_ids"].shape[-1]:]
caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
print("Caption:", caption)


# /home/spandan/scratch/interiit/spandan/EarthMind/demo_images/sar/000014.jpg