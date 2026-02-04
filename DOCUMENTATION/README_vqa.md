# VQA

## General Purpose
Comprehensive Visual Question Answering system for satellite imagery combining Qwen3-VL, GroundingDINO, SAM, and geometric analysis. Implements unified API with task routing, adapter swapping, and multi-modal reasoning.

## Files

### `unified_api.py`
FastAPI server implementing task classification router, orchestrator integration, band classifier, and dynamic adapter management for caption/VQA tasks with SAR/RGB-specific models.

### `orchestrator.py`
Core VQA orchestration pipeline: extracts object classes from queries, runs GroundingDINO detection, SAM segmentation, computes geometric properties, and uses VLM for reasoning with tool-calling capabilities.

### `local_vlm.py`
LocalVLM class wrapping Qwen3-VL with Unsloth optimizations. Supports adapter loading/swapping, tool calling, LangChain integration, image preprocessing, and streaming responses for VQA tasks.

### `gsam.py`
Helper utilities for loading GroundingDINO and SAM models, running grounded detection with NMS, and generating segmentation masks with cropped outputs.

### `sam3_api.py`
Client for SAM3 API providing methods for mask generation, bbox detection, and merged mask requests with filtering capabilities.

### `geometric_utils.py`
Computes mask properties including orientation, aspect ratio, area, centroid, perimeter, and distances between masks for spatial reasoning.

### `image_type_classifier.py`
BandClassifier wrapper for ResNet50-based SAR/RGB/false-color classification with checkpoint loading and inference methods.

### `vlm_adapter.py`
Adapter class making LocalVLM compatible with LangGraph agent frameworks by providing standardized generate() interface.

### `mask_merging.py`
Intelligent mask merging using IoM (Intersection over Minimum) graph algorithm with geodesic dilation, connected components, and visualization utilities.

### `groq_client.py`
Lightweight Groq API client for LLM calls with configurable endpoints and robust response parsing.

### `langgraph_tools.py`
Tool definitions for LangGraph integration enabling multi-step reasoning with object detection, segmentation, and analysis.

### `test_vqa.py`
VQA system test suite.

### `test_api_integration.py`, `test_api_mode.py`, `test_unified_api.py`
API integration tests for various endpoints and modes.

### `example_usage.py`
Example code demonstrating VQA system usage patterns.

### `count.py`, `a.py`
Utility scripts for analysis and quick experiments.

### `API_MODE_README.md`, `IMPLEMENTATION_SUMMARY.md`, `API_MIGRATION_SUMMARY.md`, `ALSO.md`
Documentation files describing API modes, implementation details, and migration guides.

### `README.md`
Main documentation for the VQA system.

### `systemprompt*.txt`
System prompts for different VQA task types: tools, attributes, binary, filtering, and numerical reasoning.

### `instructs.txt`
Instruction templates for model prompting.
