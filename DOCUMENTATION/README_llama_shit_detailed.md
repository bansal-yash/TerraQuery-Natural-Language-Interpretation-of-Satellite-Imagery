# Llama-shit - Detailed Documentation

## General Purpose
Experimental infrastructure for efficient LLM inference using llama.cpp and custom Python bindings for Qwen3-VL GGUF models with C++ performance optimization.

## Root Files

### `export`
Script or utility for exporting/converting models to GGUF format compatible with llama.cpp.

## llama.cpp/
Standard llama.cpp repository clone for efficient CPU/GPU inference with GGUF quantized models. Contains comprehensive tooling for model conversion, benchmarking, and inference. See llama.cpp official documentation for detailed usage.

## qwen3vl_bindings/

### `py_qwen3vl.cpp`
Patched C++ Python bindings using pybind11 for Qwen3-VL MTMD (Multi-Turn Multi-Domain) support. Implements legacy single-image `infer()` and new multi-image `infer_chat()` with sequential block tokenization, correct n_past progression to avoid M-RoPE errors, and audio file filtering.

### `setup.py`
Python package setup for qwen-mtmd with compiled shared object distribution for Python 3.10+ on Linux x86_64.

### `api.py`
FastAPI REST service for Qwen3-VL GGUF models with endpoints for bbox detection, captioning, and feature extraction. Uses qwen_mtmd module with configurable model/mmproj paths via environment variables.

### `qwen3vl_api.py`
Command-line inference script for qwen_mtmd with configurable batch size, max tokens, and automatic memory failure handling with batch size reduction.

### `example.py`
Example multi-image chat inference demonstrating SAR image analysis with per-image descriptions and combined summary parsing.

### `qwen_grounding.py`
Grounding inference using qwen_mtmd bindings.

### `eval_qwen3vl.py`
Evaluation script for GGUF model performance testing.

### `bbox_example.py`
Example bbox detection implementation.

### `test.py`
Testing utilities for qwen_mtmd bindings.

### `draw.py`
Visualization script for drawing predicted boxes on images.

### `compile.sh`, `maker.sh`
Build scripts for compiling C++ extensions.

### `CMakeLists.txt`
CMake configuration for building the Python extension module.

### `cpu.cpp`, `gpu_test.cpp`, `gpu_test_robust.cpp`
C++ testing utilities for CPU and GPU inference performance.

### `stb_image.h`, `stb_image.c`
STB image loading library for C++ image processing.

### `qwen_mtmd.cpython-*.so`
Compiled Python extension modules for Python 3.10 and 3.12.

### `pyproject.toml`
Python project configuration.

## qwen_mtmd/

### `__init__.py`
Python package initialization for qwen_mtmd module.

### `qwen_mtmd.cpython-310-x86_64-linux-gnu.so`
Compiled shared library providing Python interface to C++ qwen_mtmd implementation.
