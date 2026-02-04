# Qwen Model - Detailed Documentation

## General Purpose
Core Qwen3-VL model implementations with grounding capabilities, SAM integration, evaluation scripts, and fine-tuning resources for satellite imagery tasks.

## Root Files

### `qwen_grounding.py`
Standalone script using Qwen3-VL's native grounding to detect objects and output normalized bounding boxes (0-1000 scale) in `<ref>label</ref><box>(x1,y1),(x2,y2)</box>` format.

### `qwen_sam.py`
Integrated pipeline combining Qwen3-VL grounding with Segment Anything Model (SAM) for end-to-end object detection and segmentation mask generation.

### `eval.sh`
Shell script automating model evaluation on test datasets using PyTorch models.

### `eval_gguf.sh`
Evaluation script specifically for GGUF-quantized Qwen models using llama.cpp backend.

## eval/ Directory

### `evaluate_iou.py`
Computes IoU metrics between predicted and ground-truth bounding boxes. Outputs global mean IoU statistics, histogram counts, and per-image results with position classification (left/right/top/bottom).

### `generate_outputs_bbox.py`
Batch generation script loading Qwen3-VL once to process all images in a directory, outputting normalized AABB boxes for evaluation using native grounding with system instructions for merging overlapping objects.

### `gguf_qwen.py`
GGUF model inference using llama-cpp-python's Qwen25VLChatHandler for efficient bounding box generation with quantized models, supporting ground-truth JSON for object class extraction.

### `vllm_qwen.py`
High-performance bounding box extraction using vLLM for batched Qwen3-VL inference with base64 image encoding and multimodal prompt construction.

### `lmkks.py`
Utility script for evaluation tasks (specific functionality would need code inspection).

### `iou_stats.json`
Cached IoU statistics from evaluation runs.

### `outputs_bbox.json`
Generated bounding box predictions in evaluation format.

### `per_image_results.json`
Detailed per-image evaluation results with IoU scores and position classifications.

## ft/ Directory (Fine-tuning)

### `finetune_qwen.py`
Main fine-tuning script with custom loss function combining IoU loss, false positive penalty, and false negative penalty. Uses Unsloth FastVisionModel for efficient LoRA training with PEFT.

### `bbox_eval.py`
Comprehensive comparison script evaluating fine-tuned vs base Qwen models using BERTScore and BERT-BLEU metrics with batched inference support for parallel GPU evaluation.

### `evaluate.py`
Simple bounding box visualization script loading fine-tuned grounding models to draw predicted boxes on images. Handles multiple bbox output formats.

### `infer.py`
Caption/answer generation script for fine-tuned Qwen3-VL LoRA models. Supports loading adapters or merged models with configurable generation parameters.

### `normal.py`
Standard inference utility for baseline model evaluation.

### `fix_save.py`
Utility to fix or convert saved model checkpoints.

### `README.md`
Documentation for the fine-tuning workflow.

### `checkpoints/`
Directory storing LoRA adapter checkpoints from fine-tuning runs.

### Visualization Outputs
- `normal_rgb_bbox.png`: RGB imagery bbox results
- `out_ships.png`, `out_false_ships.png`: Ship detection results and false positives
- `out_base_ships.png`, `out_base_false_ships.png`: Base model comparison
- `output.png`, `output_base.png`, `test_out.png`, `last.png`: Various test outputs

### `bbox_check_trained_VRSBench.log`
Training logs for VRSBench bbox fine-tuning.

### `bbox_eval.txt`, `caption.txt`, `normal.json`
Evaluation outputs and results files.
