#!/bin/bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
pip install -r requirements.txt
pip install unsloth==2025.11.6 unsloth_zoo==2025.11.6
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install '.[torch]'
cd ..
rm -rf transformers