# Qwen Model

## General Purpose
Core Qwen3-VL model implementations with grounding and segmentation integration. Contains evaluation scripts and fine-tuning resources.

## Files

### `qwen_grounding.py`
Native grounding implementation using Qwen3-VL to detect objects and output normalized bounding boxes in <ref>label</ref><box>(x1,y1),(x2,y2)</box> format.

### `qwen_sam.py`
Integrated pipeline combining Qwen3-VL grounding with Segment Anything Model (SAM) for detecting bounding boxes and generating segmentation masks.

### `eval.sh`, `eval_gguf.sh`
Shell scripts automating model evaluation on test sets, with separate script for GGUF-quantized models.

### `eval/`
Directory containing evaluation scripts and utilities.

### `ft/`
Fine-tuning resources including training configurations and adapter checkpoints.

### `eval_logs/`
Logged outputs from evaluation runs for analysis and debugging.
