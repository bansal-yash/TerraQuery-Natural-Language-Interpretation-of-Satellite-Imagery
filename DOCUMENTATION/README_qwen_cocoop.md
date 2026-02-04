# Qwen CoCoOp

## General Purpose
Integration of Context Optimization (CoCoOp) prompt learning with Qwen3-VL for improved few-shot adaptation. Implements learnable prompt tokens conditioned on image features.

## Files

### `finetune_cocoop.py`
Fine-tuning script implementing CoCoOp PromptLearner module with auto-detection of image feature dimensions. Injects learned prompts into input embeddings while maintaining gradient flow.

### `finetune_cocoop_v2.py`
Enhanced version of CoCoOp training with improved prompt injection and training stability.

### `cocoop_inference.py`
Inference script loading trained PromptLearner weights and injecting learned prompts at inference time without using model.generate() for controlled decoding.

### `generate_captions_cocoop.py`
Caption generation utility specifically for CoCoOp-trained models.

### `bbscore.py`
BERTScore evaluation adapted for CoCoOp model outputs.

### `finetune_qwen_all.py`
Alternative training script variant with CoCoOp integration.
