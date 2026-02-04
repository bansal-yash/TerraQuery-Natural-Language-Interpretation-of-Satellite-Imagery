#!/bin/bash

TARGET_PID=4051934

echo "Waiting for process $TARGET_PID to finish..."

# Loop until the process exits
while kill -0 $TARGET_PID 2> /dev/null; do
    sleep 5   # check every 5 seconds
done

echo "Process $TARGET_PID finished. Running command..."

cd /home/samyak/scratch/interiit/qwen

source /home/samyak/scratch/micromamba/etc/profile.d/mamba.sh

micromamba activate qwen

python /home/samyak/scratch/interiit/qwen/finetune_qwen_all.py --train-json /home/samyak/scratch/interiit/GAURAV_BIG_DATA/SAR_BIG/pair_data/train/final_all.json --image-dir /home/samyak/scratch/interiit/GAURAV_BIG_DATA/SAR_BIG/pair_data/train/sar/img --use-lora
