#!/usr/bin/env bash
set -euo pipefail



IMAGE_ROOT="/home/gaurav/scratch/interiit/data/VRSBench/Images_val/"
GT_JSON="/home/gaurav/scratch/interiit/data/VRSBench/val.json"
MODEL_DIR="/home/gaurav/scratch/interiit/qwen_model/Qwen3VL-32B-Instruct-Q4_K_M.gguf"

python eval/vllm_qwen.py \
  --root "${IMAGE_ROOT}" \
  --gt "${GT_JSON}" \
  --model "${MODEL_DIR}" \
  --out eval/outputs_bbox.json

python eval/evaluate_iou.py \
  --pred eval/outputs_bbox.json \
  --gt "${GT_JSON}" \
  --images_root "${IMAGE_ROOT}" \
  --out_dir eval
