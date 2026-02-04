#!/usr/bin/env bash
# run_tim_run.sh
# Picks a GPU with at least MIN_FREE MiB free and runs the provided training command.

MIN_FREE=25000   # required free memory (MiB) -> 20GB
CMD="python finetune_qwen_caption_earthmind.py --train-json /home/spandan/scratch/interiit/EarthMind-Bench/json/caption_all_unmatched.json --local-model-dir /home/spandan/scratch/interiit/qwen/small_spandan --image-dir /home/spandan/scratch/interiit/EarthMind-Bench/img/test/rgb/img --use-lora "

# find first GPU with memory.free >= MIN_FREE
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
  awk -v min=$MIN_FREE -F',' '{gsub(/^ +| +$/,"",$2); if($2+0>=min){print $1; exit}}')

if [ -z "$GPU" ]; then
  echo "No GPU with >= ${MIN_FREE} MiB free found. Abort."
  nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader,nounits
  exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU
echo "Using GPU $GPU (seen by this job as CUDA device 0)."
echo "Running command:"
echo "$CMD"
exec $CMD
