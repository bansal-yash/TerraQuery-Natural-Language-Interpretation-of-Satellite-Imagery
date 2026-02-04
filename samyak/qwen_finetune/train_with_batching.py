"""
Fine-tuning script for Qwen3-VL with optimized batch processing.

Key batch size parameters:
- per_device_train_batch_size: Number of samples per GPU during training
- gradient_accumulation_steps: Accumulate gradients over N steps before updating
- Effective batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus

Memory-speed tradeoffs:
- Larger batch_size → faster training but more VRAM
- Larger gradient_accumulation → slower but same effective batch size with less VRAM
"""

from unsloth import FastVisionModel
import torch
import os
import json
from PIL import Image
from datasets import Dataset as HFDataset
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# ============================================
# CONFIGURATION
# ============================================
# Adjust these based on your GPU memory:
# - 24GB GPU: per_device_train_batch_size=2, gradient_accumulation_steps=4
# - 40GB GPU: per_device_train_batch_size=4, gradient_accumulation_steps=2
# - 80GB GPU: per_device_train_batch_size=8, gradient_accumulation_steps=1

BATCH_SIZE = 2              # Samples per GPU per step
GRAD_ACCUM_STEPS = 4        # Accumulate gradients over 4 steps
EFFECTIVE_BATCH = BATCH_SIZE * GRAD_ACCUM_STEPS  # = 8 total

MAX_STEPS = 30              # Quick test run (set higher for full training)
# NUM_EPOCHS = 3            # Alternative: use epochs instead of max_steps
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 2048

TRAIN_IMAGES_DIR = "/home/samyak/scratch/interiit/samyak/GeoPixel/VRSBench/Images_train"
TRAIN_ANNOTATIONS_DIR = "/home/samyak/scratch/interiit/samyak/GeoPixel/VRSBench/Annotations_train"
OUTPUT_DIR = "outputs"

# ============================================
# 1. LOAD MODEL
# ============================================
print("Loading Qwen3-VL model...")
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-8B-Instruct",
    load_in_4bit=True,  # Use 4bit quantization to save memory
    use_gradient_checkpointing="unsloth",  # Enable gradient checkpointing
)

# ============================================
# 2. CONFIGURE LORA
# ============================================
print("Configuring LoRA adapters...")
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,      # Train vision encoder
    finetune_language_layers=True,    # Train language model
    finetune_attention_modules=True,  # Train attention layers
    finetune_mlp_modules=True,        # Train MLP layers
    
    r=16,                   # LoRA rank (higher = more params but better accuracy)
    lora_alpha=16,          # LoRA alpha (recommended: same as r)
    lora_dropout=0,         # No dropout
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# ============================================
# 3. LOAD TRAINING DATA
# ============================================
def create_hf_dataset(images_dir, annotations_dir, max_samples=None):
    """Create HuggingFace Dataset from VRSBench data"""
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    image_files.sort()
    
    if max_samples:
        image_files = image_files[:max_samples]
    
    data = {
        'image': [],
        'text': []
    }
    
    for img_filename in image_files:
        img_path = os.path.join(images_dir, img_filename)
        image = Image.open(img_path).convert('RGB')
        
        json_filename = img_filename.replace('.png', '.json')
        json_path = os.path.join(annotations_dir, json_filename)
        
        with open(json_path, 'r') as f:
            annotation = json.load(f)
        
        data['image'].append(image)
        data['text'].append(annotation['caption'])
    
    return HFDataset.from_dict(data)

print("Loading training dataset...")
dataset = create_hf_dataset(
    TRAIN_IMAGES_DIR, 
    TRAIN_ANNOTATIONS_DIR, 
    max_samples=None  # Set to a number for quick testing (e.g., 100)
)
print(f"Loaded {len(dataset)} training samples")

# ============================================
# 4. CONVERT TO CONVERSATION FORMAT
# ============================================
instruction = "Describe this satellite image in detail."

def convert_to_conversation(sample):
    """Convert a dataset sample to Qwen3-VL conversation format"""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": sample["text"]}
            ]
        },
    ]
    return {"messages": conversation}

print("Converting dataset to conversation format...")
converted_dataset = [convert_to_conversation(sample) for sample in dataset]

# ============================================
# 5. CONFIGURE TRAINER
# ============================================
print("Setting up trainer...")
FastVisionModel.for_training(model)  # Enable training mode

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=converted_dataset,
    args=SFTConfig(
        # === BATCH SIZE CONFIGURATION ===
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        
        # === TRAINING SCHEDULE ===
        warmup_steps=5,
        max_steps=MAX_STEPS,
        # num_train_epochs=NUM_EPOCHS,  # Uncomment for epoch-based training
        
        # === OPTIMIZER ===
        learning_rate=LEARNING_RATE,
        optim="adamw_8bit",           # Memory-efficient optimizer
        weight_decay=0.001,
        lr_scheduler_type="linear",
        
        # === LOGGING ===
        logging_steps=1,
        save_steps=10,                # Save checkpoint every 10 steps
        save_total_limit=3,           # Keep only last 3 checkpoints
        
        # === OUTPUT ===
        output_dir=OUTPUT_DIR,
        report_to="none",             # Change to "wandb" for W&B logging
        
        # === VISION MODEL SPECIFIC ===
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=MAX_SEQ_LENGTH,
        
        # === MISC ===
        seed=3407,
        fp16=False,                   # Use bf16 if supported
        bf16=torch.cuda.is_bf16_supported(),
    ),
)

# ============================================
# 6. TRAIN
# ============================================
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)
print(f"Effective batch size: {EFFECTIVE_BATCH}")
print(f"  - per_device_train_batch_size: {BATCH_SIZE}")
print(f"  - gradient_accumulation_steps: {GRAD_ACCUM_STEPS}")
print(f"Total training samples: {len(dataset)}")
print(f"Max steps: {MAX_STEPS}")
print(f"Learning rate: {LEARNING_RATE}")
print("="*60 + "\n")

trainer_stats = trainer.train()

print("\n" + "="*60)
print("TRAINING COMPLETED")
print("="*60)
print(f"Final loss: {trainer_stats.training_loss:.4f}")
print(f"Total training time: {trainer_stats.metrics['train_runtime']:.2f}s")
print(f"Samples/second: {trainer_stats.metrics['train_samples_per_second']:.2f}")
print("="*60)

# ============================================
# 7. SAVE FINAL MODEL
# ============================================
print("\nSaving final model...")
model.save_pretrained(f"{OUTPUT_DIR}/final_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")
print(f"Model saved to {OUTPUT_DIR}/final_model")

print("\n✅ Training complete!")
print(f"\nTo use the trained model:")
print(f"  model, tokenizer = FastVisionModel.from_pretrained('{OUTPUT_DIR}/final_model')")
