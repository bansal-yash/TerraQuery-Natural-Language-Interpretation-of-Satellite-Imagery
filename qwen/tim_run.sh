#!/usr/bin/env bash
# run_tim_run.sh
# Picks a GPU with at least MIN_FREE MiB free and runs the provided training command.

#!/usr/bin/env bash
set -euo pipefail

MIN_FREE=23000

# Python command and args to run (edit if you want different args)
CMD=(
  python tim.py
  --mode train
  --train-json /home/samyak/scratch/interiit/EarthMind-Bench/json/caption_all_unmatched.json
  --image-dir /home/samyak/scratch/interiit/EarthMind-Bench/img
  --local-model-dir /home/samyak/scratch/interiit/qwen/small_spandan
  --output-dir /home/samyak/scratch/interiit/qwen/checkpoints_tim2
  --batch-size 1
  --epochs 3
  --use-lora
  --r 2
  --alpha 8
  --tim-cache-dir /home/samyak/scratch/interiit/qwen/tim_cache
  --load-in-4bit
)
# python tim.py --mode train --train-json /home/samyak/scratch/interiit/EarthMind-Bench/json/caption_all_unmatched.json --image-dir /home/samyak/scratch/interiit/EarthMind-Bench/img --local-model-dir /home/samyak/scratch/interiit/qwen/small_samyak --output-dir /home/samyak/scratch/interiit/qwen/checkpoints_tim2 --use-lora --r 2 --alpha 16 --load-in-4bit --tim-cache-dir /home/samyak/scratch/interiit/qwen/tim_cache --epochs 3 --batch-size 1
# find first GPU with >= MIN_FREE MiB free
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
  awk -v min=$MIN_FREE -F',' '{gsub(/^ +| +$/,"",$2); if($2+0>=min){print $1; exit}}')

if [ -z "$GPU" ]; then
  echo "No GPU with >= ${MIN_FREE} MiB free found. Abort."
  echo "GPU states:"
  nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader,nounits
  exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU
# optional: reduce fragmentation (helpful on some systems)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Using GPU $GPU (seen by this job as CUDA device 0)."
echo "Running command:"
printf ' %q' "${CMD[@]}"
echo

# exec to replace shell with the python process (preserves signals)
exec "${CMD[@]}"



# python tim.py --mode train --train-json /home/samyak/scratch/interiit/EarthMind-Bench/json/caption_all_unmatched.json --image-dir /home/samyak/scratch/interiit/EarthMind-Bench/img --local-model-dir /home/samyak/scratch/interiit/qwen/small_samyak --output-dir /home/samyak/scratch/interiit/qwen/checkpoints_tim --batch-size 2 --epochs 5 --lr 2e-4 --gradient-accumulation-steps 4 --use-lora --load-in-4bit --tim-cache-dir /home/samyak/scratch/interiit/qwen/tim_cache --save-steps 50 --logging-steps 20


