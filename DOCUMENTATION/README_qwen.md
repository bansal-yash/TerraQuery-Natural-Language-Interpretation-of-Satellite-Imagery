# Qwen

## General Purpose
Central hub for Qwen3-VL model fine-tuning, benchmarking, and evaluation. Contains scripts for training on satellite imagery datasets, BERT-BLEU evaluation, bounding box scoring, and dataset preparation.

## Files

### `finetune_qwen_all.py`
Main fine-tuning script for Qwen3-VL on combined datasets with custom trainer monitoring per-item losses, token accuracy, and BLEU/BERT metrics during training.

### `finetune_qwen_caption_earthmind.py`
Caption-focused fine-tuning variant for EarthMind dataset using language modeling loss without box/IoU losses.

### `qwen_run.sh`
Shell script for launching Qwen training jobs with predefined configurations.

### `bbscore.py`
Comprehensive benchmarking script comparing fine-tuned vs base Qwen models using BERTScore and BERT-BLEU metrics with batched inference support.

### `bbscore_2.py`, `bbscore_tim.py`, `bbscore_vqa.py`, `bbscore_all.py`
Specialized variants of bbscore for different evaluation scenarios: TiM (Thinking-in-Modalities) training, VQA tasks, and comprehensive all-task evaluation.

### `bert_bleu.py`
Implementation of semantic n-gram precision metric using BERT embeddings for evaluating caption quality beyond surface-level matching.

### `bbox_visual.py`
Visualization script loading fine-tuned Qwen+LoRA models to generate and overlay bounding boxes on satellite images.

### `tim.py`, `tim_2.py`
Thinking-in-Modalities (TiM) training scripts with automatic RGB+SAR horizontal merging, TiM precomputation/injection, and optional LoRA fine-tuning.

### `tim_infer.py`
Inference script using TiM-trained models for caption generation with cached modality-specific reasoning.

### `generate_captions_from_checkpoints.py`
Evaluation script loading fine-tuned checkpoints (LoRA or full) to generate captions on test sets with exact generation matching training configuration.

### `prepare_earthmind_dataset.py`
Dataset converter transforming COCO-format EarthMind annotations to Qwen training format, prioritizing SAR over RGB when available.

### `iter.py`
Multi-model iterative pipeline combining CLIP, GroundingDINO, SAM, and Qwen for attribute extraction, grounding, segmentation, and caption generation.

### `test_adapter_loading.py`
Testing utility comparing vanilla Qwen outputs vs adapter-loaded outputs with timing benchmarks to verify LoRA integration.

### `extract_bert_bleu.py`
Utility script for extracting and analyzing BERT-BLEU scores from evaluation outputs.

### `qwen_test.py`
General testing script for Qwen model functionality and sanity checks.

### `RESULTS.md`
Markdown documentation of experimental results, performance comparisons, and benchmark findings.

### `TODOS.txt`
Task tracking file for ongoing experiments and planned improvements.
