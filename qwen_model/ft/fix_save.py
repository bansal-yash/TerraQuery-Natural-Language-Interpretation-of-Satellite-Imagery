#!/usr/bin/env python3
"""
Fix/save helper for fine-tune checkpoints.

What it does:
- Locates a checkpoint directory (explicit or the latest in `--checkpoints-dir`).
- Loads the model and processor/tokenizer from the checkpoint using local files only.
- Saves a `final` folder with `save_pretrained` (model + tokenizer/processor).
- Attempts to write a merged (16-bit) model in `final_merged` using
  `model.save_pretrained_merged(..., save_method="merged_16bit")` if available.

Usage:
    python ft/fix_save.py --checkpoints-dir ft/checkpoints_1
    python ft/fix_save.py --checkpoints-dir ft/checkpoints_1 --checkpoint checkpoint-12670

Notes:
- This script prefers local files and will avoid contacting Hugging Face when possible
  (uses local_files_only where supported).
- It handles the common shapes returned by `FastVisionModel.from_pretrained` which may
  return (model, tokenizer) or just model. When tokenizer isn't returned it tries to
  use the processor/tokenizer loaded from the same folder.
"""

import argparse
import os
import re
import sys
import traceback
from pathlib import Path


def find_latest_checkpoint(checkpoints_dir: Path) -> Path:
    # Prefer 'final_merged' already present (nothing to do), then 'final', then highest checkpoint-
    if (checkpoints_dir / 'final_merged').is_dir():
        return checkpoints_dir / 'final_merged'
    if (checkpoints_dir / 'final').is_dir():
        return checkpoints_dir / 'final'

    # Find checkpoint-* directories with numeric suffix
    candidates = []
    for child in checkpoints_dir.iterdir():
        if child.is_dir():
            m = re.match(r'checkpoint-(\d+)', child.name)
            if m:
                candidates.append((int(m.group(1)), child))
    if not candidates:
        # fallback to any directory
        dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir()]
        if not dirs:
            raise FileNotFoundError(f'No checkpoint directories found in {checkpoints_dir}')
        # return lexicographically last
        return sorted(dirs)[-1]

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def has_model_files(dirpath: Path) -> bool:
    # crude check for tokenizer/model files
    keys = [
        'config.json', 'tokenizer.json', 'tokenizer_config.json',
        'adapter_model.safetensors', 'pytorch_model.bin', 'model.safetensors'
    ]
    for k in keys:
        if (dirpath / k).exists():
            return True
    # also check for subfolders typical of HF snapshots
    if any((dirpath / p).exists() for p in ('tokenizer', 'model')):
        return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints-dir', default='ft/checkpoints_1', help='Directory containing checkpoints')
    p.add_argument('--checkpoint', default=None, help='Specific checkpoint directory name to use (e.g. checkpoint-12670)')
    p.add_argument('--final-name', default='final', help='Name for final save folder inside checkpoints dir')
    p.add_argument('--merged-name', default='final_merged', help='Name for merged output folder inside checkpoints dir')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    base = Path(args.checkpoints_dir)
    if not base.exists():
        print(f'Checkpoints dir not found: {base}', file=sys.stderr)
        sys.exit(2)

    # Decide which checkpoint to load
    if args.checkpoint:
        ckpt = base / args.checkpoint
        if not ckpt.exists():
            print(f'Specified checkpoint does not exist: {ckpt}', file=sys.stderr)
            sys.exit(2)
    else:
        ckpt = find_latest_checkpoint(base)

    print('Using checkpoint dir:', ckpt)

    # If ckpt is already the final_merged, nothing to do
    final_dir = base / args.final_name
    merged_dir = base / args.merged_name

    if ckpt.name == args.merged_name and has_model_files(ckpt):
        print(f'Checkpoint is already a merged model ({ckpt}). Nothing to do.')
        return

    # choose source for loading: prefer 'final' inside dir if present, else ckpt
    source = ckpt
    if (ckpt / 'final').is_dir():
        source = ckpt / 'final'

    # attempt to import packages
    try:
        from unsloth import FastVisionModel
    except Exception as e:
        print('Failed to import unsloth.FastVisionModel:', e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(3)

    try:
        from transformers import AutoProcessor
    except Exception as e:
        print('Failed to import transformers.AutoProcessor:', e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(3)

    # Load processor/tokenizer locally
    processor = None
    tokenizer = None
    model = None

    load_errors = []
    try:
        print('Loading processor (local files only) from', str(source))
        processor = AutoProcessor.from_pretrained(str(source), local_files_only=True)
    except Exception as e:
        print('AutoProcessor.from_pretrained(local) failed:', e)
        load_errors.append(e)
        processor = None

    # Load FastVisionModel. It may return (model, tokenizer) or model only.
    try:
        print('Loading model from', str(source), '(local files only)')
        res = FastVisionModel.from_pretrained(str(source), local_files_only=True)
        # support either (model, tokenizer) or model
        if isinstance(res, tuple) or isinstance(res, list):
            model = res[0]
            if len(res) > 1:
                tokenizer = res[1]
        else:
            model = res
    except Exception as e:
        print('FastVisionModel.from_pretrained(local) failed:', e)
        traceback.print_exc()
        # As a final attempt, try without local_files_only (may hit network)
        try:
            print('Retrying FastVisionModel.from_pretrained without local_files_only (may require network)')
            res = FastVisionModel.from_pretrained(str(source))
            if isinstance(res, tuple) or isinstance(res, list):
                model = res[0]
                if len(res) > 1:
                    tokenizer = res[1]
            else:
                model = res
        except Exception as e2:
            print('Retry also failed:', e2)
            traceback.print_exc()
            sys.exit(4)

    # If tokenizer wasn't returned, try to get from processor
    if tokenizer is None and processor is not None:
        tok = getattr(processor, 'tokenizer', None)
        if tok is not None:
            tokenizer = tok

    # Prepare final output dirs
    final_output_dir = final_dir
    merged_output_dir = merged_dir

    try:
        print('Saving model and processor to', final_output_dir)
        final_output_dir.mkdir(parents=True, exist_ok=True)
        # model.save_pretrained may exist on FastVisionModel
        try:
            model.save_pretrained(str(final_output_dir))
        except Exception as e:
            print('model.save_pretrained failed:', e)
            # attempt to fallback to HF-compatible save if available
            try:
                save_fn = getattr(model, 'save_pretrained', None)
                if save_fn is not None:
                    save_fn(str(final_output_dir))
            except Exception:
                print('Fallback model.save_pretrained also failed; continuing')

        # save tokenizer/processor
        if tokenizer is not None:
            try:
                tokenizer.save_pretrained(str(final_output_dir))
            except Exception as e:
                print('tokenizer.save_pretrained failed:', e)
        if processor is not None:
            try:
                processor.save_pretrained(str(final_output_dir))
            except Exception as e:
                print('processor.save_pretrained failed:', e)

        # Now attempt merged save
        print('Attempting merged save to', merged_output_dir)
        merged_output_dir.mkdir(parents=True, exist_ok=True)

        save_merged_fn = getattr(model, 'save_pretrained_merged', None)
        if save_merged_fn is None:
            print('Model does not implement save_pretrained_merged; skipping merged save.')
        else:
            # choose tokenizer/processor to pass
            saveto = tokenizer or processor
            if saveto is None:
                print('No tokenizer/processor instance available to pass to save_pretrained_merged, attempting to pass None')
            try:
                save_merged_fn(str(merged_output_dir), saveto, save_method='merged_16bit')
                print('Merged model saved to', merged_output_dir)
            except TypeError:
                # try without named arg
                try:
                    save_merged_fn(str(merged_output_dir), saveto)
                    print('Merged model saved to', merged_output_dir)
                except Exception as e:
                    print('save_pretrained_merged failed:', e)
                    traceback.print_exc()

    except Exception as e:
        print('Error during save operations:', e)
        traceback.print_exc()
        sys.exit(5)

    print('Done. Saved final ->', final_output_dir, 'and merged ->', merged_output_dir)


if __name__ == '__main__':
    main()
