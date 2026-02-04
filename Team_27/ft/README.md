# Finetuning Guide (ft)

This folder contains training and evaluation scripts for Qwen-based models on SAR/remote-sensing tasks. Use the commands below as a quick start for bbox grounding and paired caption-style finetuning.

## Additional Installations

```bash
conda activate geonli
pip install peft==0.18.0
pip install bert_score
```

## Key scripts
- `finetune_qwen_bbox.py`: Train Qwen for bounding-box grounding with GroundingDINO-style loss.
- `finetune_qwen_caption.py`: Finetune for captioning/attribute-style outputs on paired image-text data.
- `evaluate.py`, `bbox_eval.py`: Basic evaluation helpers for bbox/caption outputs.
- `scripts/create_null_adapter.py`: Utility to build a null LoRA adapter scaffold.

## Data sources
- **MMRS SARV2 detection split** (HBB JSON + images). Refer to the MMRS resources cited in the EarthGPT project for download/licensing details: https://github.com/wivizhang/EarthGPT
- **EarthMind / EarthBench paired data** (RGB image + text pairs) on Hugging Face: https://huggingface.co/datasets/sy1998/EarthMind-data

## Quickstart commands

### Bounding boxes (SARV2 HBB)
```bash
python ft/finetune_qwen.py \
  --train-json path/to/MMRS/data/json/detection/SARV2/SARv2_trainnval_hbb.json \
  --img-root path/to/MMRS/data/detection/SARV2/images \
  --output-dir ft/checkpoints/SARV2 \
  --batch-size 2 --epochs 3 \
  --use-grounding-loss --grounding-loss-every-n-steps 5
```

### Paired instruction/caption tuning (EarthBench pairs)
```bash
python finetune_qwen_all.py \
  --use-lora \
  --train-json path/to/pair_data/train/final_all.json \
  --image-dir path/to/pair_data/train/rgb/img \
  --use-lora
```

## Notes
- Create and activate the `geonli` Conda env from the root install instructions before running.
- Adjust batch size/epochs to fit your GPU memory; keep `--use-grounding-loss` for bbox tasks that need alignment.
- Replace the dataset paths with your local copies. Ensure you have permission to use MMRS SARV2 and EarthBench data in your environment.
- Outputs (checkpoints, logs) are written under `--output-dir`; keep that directory on a fast disk.
