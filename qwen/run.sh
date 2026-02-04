#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

conda activate inter

cd /home/spandan/scratch/interiit/qwen

python finetune_qwen_caption_earthmind.py --train-json /home/spandan/scratch/interiit/EarthMind-Bench/json/caption_all_unmatched.json --local-model-dir /home/spandan/scratch/interiit/qwen/small_spandan --image-dir /home/spandan/scratch/interiit/EarthMind-Bench/img/test/rgb/img --use-lora 