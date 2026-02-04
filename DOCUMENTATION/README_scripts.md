# Scripts

## General Purpose
Utility scripts for model initialization, performance profiling, and development tools.

## Files

### `create_null_adapter.py`
Creates zero-initialized LoRA adapters mirroring target adapter configurations. Useful for fast adapter swapping without memory overhead when maintaining base model behavior.

### `profile_local_vlm.py`
Benchmark utility measuring LocalVLM inference latency including warmup runs, percentile statistics, and end-to-end timing for prompt construction and decoding.

### `artifacts/`
Directory storing generated artifacts like null adapters.
