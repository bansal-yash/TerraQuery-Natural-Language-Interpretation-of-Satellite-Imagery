# Qwen API

## General Purpose
FastAPI-based REST API serving Qwen3-VL model for bounding box detection, image captioning, and feature extraction. Supports both GGUF (llama.cpp) and PyTorch backends with optional adapter loading.

## Files

### `api.py`
Main FastAPI application implementing endpoints for bbox detection, captioning, and feature extraction. Supports multipart file uploads, configurable via environment variables for model paths and adapter usage.

### `qwen_grounding.py`
Standalone grounding script using Qwen3-VL's native grounding capability to output normalized bounding boxes (0-1000 scale) for specified objects in images.

### `test.py`
API smoke test suite sending requests to bbox, caption, and features endpoints with image uploads. Includes bbox overlay visualization demo.

### `api_test_results.json`
Saved test results from API endpoint verification runs.
