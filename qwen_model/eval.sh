#!/usr/bin/env bash
set -euo pipefail

IMAGE_ROOT="/home/samyak/scratch/interiit/data/VRSBench/Images_val/"
GT_JSON="/home/samyak/scratch/interiit/data/VRSBench/val.json"
MODEL_DIR="ft/checkpoints_1/final_merged"   # or Qwen/Qwen3-VL-8B-Instruct

python eval/generate_outputs_bbox.py \
  --root "${IMAGE_ROOT}" \
  --gt "${GT_JSON}" \
  --out eval/outputs_bbox.json \
  --model "${MODEL_DIR}" \
  --device "cuda"

python eval/evaluate_iou.py \
  --pred eval/outputs_bbox.json \
  --gt "${GT_JSON}" \
  --images_root "${IMAGE_ROOT}" \
  --out_dir eval

echo "Eval complete. See eval/iou_stats.json and eval/per_image_results.json"