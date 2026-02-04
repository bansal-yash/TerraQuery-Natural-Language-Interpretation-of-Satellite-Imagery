# Samyak/Qwen Finetune - Detailed Documentation

## General Purpose
Experimental Qwen3-VL fine-tuning workspace with optimized batching strategies for VRSBench satellite imagery dataset, including notebooks and production scripts.

## Files

### `fine_tune.py`
Initial fine-tuning implementation using Unsloth FastVisionModel with 4-bit quantization and LoRA. Creates HuggingFace datasets from VRSBench annotations and implements custom data collation for vision-language training.

### `train_with_batching.py`
Optimized training script with configurable batch size and gradient accumulation for memory-efficient fine-tuning. Includes detailed memory-speed tradeoff documentation and per-device batch size tuning for different GPU configurations (24GB/40GB/80GB).

### `fine_tune.ipynb`
Interactive Jupyter notebook for experimental fine-tuning with visualization and step-by-step execution.

### `main.ipynb`
Main experimentation notebook for testing various fine-tuning configurations and hyperparameters.

### `evaluation_results.json`
Saved evaluation metrics from fine-tuning experiments.

### `outputs/`
Directory containing training outputs, checkpoints, and logs.

### `unsloth_compiled_cache/`
Cached Unsloth compiled trainers for faster startup.
