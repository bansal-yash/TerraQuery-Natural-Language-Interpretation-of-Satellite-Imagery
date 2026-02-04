# Spandan/Models to Test - Detailed Documentation

## General Purpose
Collection of research models and papers for evaluation on satellite imagery tasks, including RingMo framework and academic literature on Earth observation vision-language models.

## RingMo/
Chinese Academy of Sciences and Huawei's self-supervised pre-training framework for visual domain. Full domestic (China-native) toolkit featuring MAE, SimMIM, SimCLR architectures with ViT, ViT-MoE, Swin transformers. Optimized for Ascend chips and MindSpore framework with distributed parallel strategies, MoE expert systems, and zero-labeled pre-training capabilities.

### Key Files in RingMo:
- `pretrain.py`: Pre-training script for self-supervised learning
- `finetune.py`: Fine-tuning for downstream tasks
- `eval.py`: Evaluation utilities
- `README.md`: Comprehensive Chinese documentation on architecture and features
- `ringmo_framework/`: Core framework implementation
- `config/`: Configuration files for various model architectures
- `scripts/`: Training and evaluation scripts

## model_papers/

### `EarthMind.pdf`
Academic paper describing EarthMind benchmark and models for satellite imagery understanding.

### `EarthDial.pdf`
Research paper on EarthDial: dialogue-based Earth observation analysis system.

### `EarthGPT.pdf`
Paper on EarthGPT: Large language model for geospatial applications.

### `GeoLang.pdf`
Research on GeoLang: language models for geographic information processing.

### `REO_VLM.pdf`
Paper on Remote Earth Observation Vision-Language Models.

### `RING-MO.pdf`
Detailed paper on RingMo architecture and methodology.

### `EarthMind/`
Subdirectory potentially containing EarthMind-related code or supplementary materials.
