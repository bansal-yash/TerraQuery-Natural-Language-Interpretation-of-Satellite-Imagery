# Fine-tuning Qwen3-VL for Grounding with Custom Loss + LoRA

This directory contains scripts for fine-tuning the Qwen3-VL model with custom loss functions designed for bounding box grounding tasks, using **LoRA (Low-Rank Adaptation)** for memory-efficient training.

## Features

### Custom Loss Components

The fine-tuning script includes three custom loss components on top of the standard language modeling loss:

1. **IoU Loss**: Penalizes low Intersection over Union (IoU) between predicted and ground truth bounding boxes
2. **False Positive Loss**: Penalizes predicted bounding boxes that don't match any ground truth box
3. **False Negative Loss**: Penalizes ground truth boxes that have no matching predictions (missed detections)

### LoRA Efficient Fine-Tuning

- **Memory Efficient**: Uses 4-bit quantization + LoRA to fine-tune on consumer GPUs (24GB VRAM)
- **Fast Training**: Only trains ~1-2% of parameters vs full fine-tuning
- **Comparable Results**: LoRA achieves similar performance to full fine-tuning
- **Configurable**: Choose which layers to fine-tune (vision, language, attention, MLP)

## Requirements

```bash
# Install unsloth for efficient LoRA training
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Or use standard transformers (requires more memory)
pip install torch transformers pillow tqdm numpy peft bitsandbytes
```

## Usage

### Basic Training (with LoRA - Recommended)

```bash
python ft/finetune_qwen.py \
  --train-json /path/to/train.json \
  --img-root /path/to/Images_train/ \
  --output-dir ft/checkpoints \
  --use-lora \
  --load-in-4bit \
  --lora-r 16 \
  --lora-alpha 16
```

### Advanced Options with LoRA Configuration

```bash
python ft/finetune_qwen.py \
  --train-json /path/to/train.json \
  --img-root /path/to/Images_train/ \
  --model unsloth/Qwen3-VL-8B-Instruct \
  --output-dir ft/checkpoints \
  --batch-size 2 \
  --epochs 3 \
  --lr 2e-4 \
  --use-lora \
  --load-in-4bit \
  --lora-r 16 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --finetune-vision-layers \
  --finetune-language-layers \
  --finetune-mlp-modules \
  --iou-loss-weight 1.0 \
  --fp-loss-weight 0.5 \
  --fn-loss-weight 1.0 \
  --gradient-accumulation-steps 4 \
  --save-steps 100 \
  --logging-steps 10
```

### Full Fine-Tuning (Not Recommended - Requires >80GB VRAM)

```bash
python ft/finetune_qwen.py \
  --train-json /path/to/train.json \
  --img-root /path/to/Images_train/ \
  --output-dir ft/checkpoints \
  --no-use-lora \
  --no-load-in-4bit
```

### Parameters

#### Basic Parameters
- `--train-json`: Path to the training JSON file in the expected format (see below)
- `--img-root`: Root directory containing the training images
- `--model`: HuggingFace model ID (default: `unsloth/Qwen3-VL-8B-Instruct`)
- `--output-dir`: Directory to save checkpoints and final model
- `--batch-size`: Training batch size per device (default: 2)
- `--epochs`: Number of training epochs (default: 3)
- `--lr`: Learning rate (default: 2e-4 for LoRA, 2e-5 for full fine-tuning)

#### LoRA Configuration
- `--use-lora`: Use LoRA for efficient fine-tuning (default: True)
- `--load-in-4bit`: Use 4-bit quantization to reduce memory (default: True)
- `--lora-r`: LoRA rank - higher = more capacity but more memory (default: 16)
- `--lora-alpha`: LoRA alpha - typically set equal to r (default: 16)
- `--lora-dropout`: Dropout for LoRA layers (default: 0.0)
- `--finetune-vision-layers`: Fine-tune vision encoder (default: False)
- `--finetune-language-layers`: Fine-tune language model layers (default: True)
- `--finetune-attention-modules`: Fine-tune attention layers (default: False)
- `--finetune-mlp-modules`: Fine-tune MLP/FFN layers (default: True)

#### Custom Loss Weights
- `--iou-loss-weight`: Weight for IoU loss component (default: 1.0)
- `--fp-loss-weight`: Weight for false positive loss (default: 0.5)
- `--fn-loss-weight`: Weight for false negative loss (default: 1.0)

#### Training Configuration
- `--gradient-accumulation-steps`: Accumulate gradients over N steps (default: 4)
- `--save-steps`: Save checkpoint every N steps (default: 100)
- `--logging-steps`: Log metrics every N steps (default: 10)


## Expected JSON Format

The training JSON should be a list of objects with the following structure:

```json
[
  {
    "image": "00006_0000.png",
    "image_path": "Images_train/00006_0000.png",
    "caption": "Description of the image...",
    "objects": [
      {
        "class": "ship",
        "bbox_aabb": [0.55, 0.36, 0.7, 0.56],
        "obj_id": 0,
        "is_unique": true,
        "obj_position": "",
        "obj_size": "small"
      }
    ]
  }
]
```

Key fields:
- `image`: Filename of the image
- `objects`: List of objects with:
  - `class`: Object class name (e.g., "ship", "airplane", "vehicle")
  - `bbox_aabb`: Bounding box in normalized [x1, y1, x2, y2] format (0-1 range)

## How It Works

### Loss Computation

The total loss is:
```
Total Loss = LM Loss + (α × IoU Loss) + (β × FP Loss) + (γ × FN Loss)
```

Where:
- **LM Loss**: Standard language modeling loss (cross-entropy on token predictions)
- **IoU Loss**: `1 - average_iou` for matched prediction-GT pairs
- **FP Loss**: Count of predicted boxes with no GT match
- **FN Loss**: Count of GT boxes with no prediction match
- α, β, γ: Configurable weights

### Matching Strategy

Predictions are matched to ground truth boxes using:
1. Compute IoU matrix between all predictions and GT boxes
2. Greedy matching: match highest IoU pairs first (IoU ≥ 0.5 threshold)
3. Each prediction/GT box can only be matched once
4. Unmatched predictions → false positives
5. Unmatched GT boxes → false negatives

### Training Process

1. For each training sample:
   - Load image and ground truth boxes
   - Determine primary object class (most common class in the image)
   - Format conversation with system instruction, user query, and expected assistant response
   - Tokenize and process inputs

2. During training:
   - Standard forward pass computes LM loss
   - Generate predictions for grounding loss computation
   - Parse predicted bounding boxes from generated text
   - Match predictions to GT boxes
   - Compute IoU, FP, and FN losses
   - Backpropagate combined loss

## Performance Notes

### Memory Requirements

| Configuration | VRAM Required | Trainable Params | Notes |
|--------------|---------------|------------------|-------|
| **LoRA + 4-bit (Recommended)** | ~18-24GB | ~1-2% | Fits on RTX 3090/4090, A6000 |
| LoRA + 16-bit | ~40GB | ~1-2% | Requires A100 40GB |
| Full Fine-tuning | >80GB | 100% | Requires A100 80GB or multi-GPU |

### Training Speed

- **LoRA with 4-bit**: ~2-3 samples/sec on RTX 4090
- **Full fine-tuning**: ~0.5-1 samples/sec (if you have enough VRAM)
- Computing grounding loss requires generation at each step, which is slow

### Optimization Tips

1. **Start with LoRA**: Always use `--use-lora --load-in-4bit` unless you have >80GB VRAM
2. **Reduce batch size**: If OOM, set `--batch-size 1` and increase `--gradient-accumulation-steps 8`
3. **Choose layers wisely**: 
   - For grounding: `--finetune-language-layers --finetune-mlp-modules` (default)
   - For vision understanding: Also add `--finetune-vision-layers`
4. **Adjust LoRA rank**: Lower `--lora-r 8` for less memory, higher `--lora-r 32` for more capacity
5. **Use gradient checkpointing**: Automatically enabled with unsloth

## Output

The training will produce:

### With LoRA (default):
1. **Checkpoints**: `{output_dir}/checkpoint-{step}/` - LoRA adapter weights only (~50-100MB each)
2. **Final LoRA adapters**: `{output_dir}/final/` - Can be loaded on top of base model
3. **Merged model**: `{output_dir}/final_merged/` - Full model with LoRA weights merged (16-bit, ~16GB)

### Without LoRA:
1. **Checkpoints**: `{output_dir}/checkpoint-{step}/` - Full model weights (~16GB each)
2. **Final model**: `{output_dir}/final/` - Complete fine-tuned model

### Training Logs
- Metrics logged every `--logging-steps` to console
- Loss components: `loss`, `iou_loss`, `fp_loss`, `fn_loss`

## Using the Fine-tuned Model

### Using LoRA Adapters (Recommended)

```python
from unsloth import FastVisionModel

# Load base model + LoRA adapters
model, tokenizer = FastVisionModel.from_pretrained(
    model_name="ft/checkpoints/final",  # Your LoRA adapter path
    load_in_4bit=True,
)
FastVisionModel.for_inference(model)

# Use for inference...
```

### Using Merged Model

```bash
python eval/generate_outputs_bbox.py \
  --root /path/to/test/images \
  --model ft/checkpoints/final_merged \
  --out eval/outputs_bbox.json
```

### Using with Transformers (No Unsloth)

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# Load base model
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    device_map="auto",
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "ft/checkpoints/final")
processor = AutoProcessor.from_pretrained("ft/checkpoints/final")
```

## Tips

### Choosing What to Fine-tune

For **grounding/bounding box tasks** (your use case):
```bash
--finetune-language-layers    # ✅ YES - needed for generating box coordinates
--finetune-mlp-modules         # ✅ YES - helps with spatial reasoning
--finetune-attention-modules   # ⚠️  OPTIONAL - may help but uses more memory
--finetune-vision-layers       # ❌ NO - vision encoder is already good
```

For **general caption improvement**:
```bash
--finetune-language-layers    # ✅ YES
--finetune-vision-layers      # ✅ YES - if domain-specific images
```

### LoRA Rank Selection

- **r=8**: Fastest, lowest memory, good for small datasets (<1000 samples)
- **r=16**: Balanced (recommended) - good for most tasks
- **r=32**: More capacity, better for large datasets (>10000 samples)
- **r=64**: Approaching full fine-tuning performance but uses more memory

### Loss Weight Tuning

Start with defaults and adjust based on validation:
- **High IoU errors**: Increase `--iou-loss-weight`
- **Too many false positives**: Increase `--fp-loss-weight`
- **Missing objects**: Increase `--fn-loss-weight`

### Common Issues

1. **OOM (Out of Memory)**: 
   - Reduce `--batch-size` to 1
   - Increase `--gradient-accumulation-steps` to 8
   - Lower `--lora-r` to 8
   - Ensure `--load-in-4bit` is enabled

2. **Slow training**: 
   - Grounding loss requires generation, which is slow by design
   - Use smaller `--lora-r` for faster training
   - Reduce `--save-steps` and `--logging-steps`

3. **Poor convergence**: 
   - Adjust learning rate: try 1e-4 or 3e-4
   - Adjust loss weights
   - Ensure training data is diverse

4. **Model not learning**: 
   - Check that loss is decreasing
   - Verify data format matches expected JSON structure
   - Try fine-tuning more layers (add `--finetune-attention-modules`)

## Comparison: LoRA vs Full Fine-tuning

| Aspect | LoRA + 4-bit | Full Fine-tuning |
|--------|--------------|------------------|
| **Memory** | 18-24GB | >80GB |
| **Speed** | 2-3x faster | 1x baseline |
| **Disk space** | ~50MB adapters | ~16GB full model |
| **Trainable params** | 1-2% | 100% |
| **Performance** | 95-99% of full FT | 100% |
| **Overfitting risk** | Lower | Higher |
| **Recommended** | ✅ YES | ❌ Only if necessary |

## Troubleshooting

- **OOM (Out of Memory)**: Reduce `--batch-size` or use the smaller 8B model
- **Slow training**: Increase `--gradient-accumulation-steps`, reduce dataset size
- **Poor convergence**: Adjust learning rate or loss weights
