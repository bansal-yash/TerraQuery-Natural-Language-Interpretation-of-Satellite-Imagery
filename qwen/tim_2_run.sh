#!/usr/bin/env bash
# run_tim_run.sh
# Picks a GPU with at least MIN_FREE MiB free and runs the provided training command.

#!/bin/bash
# run_training.sh - Script to run training on GPU with sufficient free memory

MIN_FREE=20000  # Minimum free memory in MiB (adjust based on your needs)

# Your training command - UPDATE THESE PATHS
CMD=(
  python tim_2.py
  --train-json /home/spandan/scratch/interiit/EarthMind-Bench/json/caption_all_unmatched.json
  --image-dir /home/spandan/scratch/interiit/EarthMind-Bench/img
  --local-model-dir /home/spandan/scratch/interiit/qwen/small_spandan
  --output-dir /home/spandan/scratch/interiit/qwen/checkpoints_tim2
  --use-lora
  --target-trainable 15000000
  
)

# Find first GPU with sufficient free memory
echo "Checking GPU availability..."
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
  awk -v min=$MIN_FREE -F',' '{gsub(/^ +| +$/,"",$2); if($2+0>=min){print $1; exit}}')

if [ -z "$GPU" ]; then
  echo "No GPU with >= ${MIN_FREE} MiB free found. Current GPU status:"
  nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader,nounits
  echo ""
  echo "Would you like to:"
  echo "1. Wait 30 seconds and retry"
  echo "2. Run on a GPU with less memory (may cause OOM)"
  echo "3. Exit"
  read -p "Enter choice (1-3): " choice
  
  case $choice in
    1)
      echo "Waiting 30 seconds..."
      sleep 30
      # Retry finding GPU
      GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
        awk -v min=$MIN_FREE -F',' '{gsub(/^ +| +$/,"",$2); if($2+0>=min){print $1; exit}}')
      if [ -z "$GPU" ]; then
        echo "Still no GPU available. Exiting."
        exit 1
      fi
      ;;
    2)
      echo "Selecting GPU with most free memory..."
      GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
        awk -F',' '{gsub(/^ +| +$/,"",$2); print $1","$2}' | \
        sort -t',' -k2 -nr | head -1 | cut -d',' -f1)
      echo "Using GPU $GPU (may be insufficient memory)"
      ;;
    3)
      echo "Exiting..."
      exit 1
      ;;
    *)
      echo "Invalid choice. Exiting..."
      exit 1
      ;;
  esac
fi

export CUDA_VISIBLE_DEVICES=$GPU
echo "Using GPU $GPU (seen by this job as CUDA device 0)."
echo "Current GPU memory status:"
nvidia-smi --id=$GPU --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits

echo ""
echo "Running training command:"
printf '  %s \\\n' "${CMD[@]}"
echo ""

# Create a log file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="training_${TIMESTAMP}_gpu${GPU}.log"
echo "Logging output to: $LOGFILE"
echo "Start time: $(date)" | tee -a "$LOGFILE"

# Run the command and log output
exec "${CMD[@]}" 2>&1 | tee -a "$LOGFILE"


# python tim.py --mode train --train-json /home/spandan/scratch/interiit/EarthMind-Bench/json/caption_all_unmatched.json --image-dir /home/spandan/scratch/interiit/EarthMind-Bench/img --local-model-dir /home/spandan/scratch/interiit/qwen/small_spandan --output-dir /home/spandan/scratch/interiit/qwen/checkpoints_tim --batch-size 2 --epochs 5 --lr 2e-4 --gradient-accumulation-steps 4 --use-lora --load-in-4bit --tim-cache-dir /home/spandan/scratch/interiit/qwen/tim_cache --save-steps 50 --logging-steps 20


