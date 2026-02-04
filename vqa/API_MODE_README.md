# API Mode Support for LocalVLM

## Summary of Changes

The `LocalVLM` class now supports **two modes** of operation:

1. **Local Mode** (default): Uses transformers/torch to run Qwen3-VL locally
2. **API Mode**: Makes HTTP requests to a remote Qwen API server

This allows you to:
- Run inference without needing GPU/transformers installed locally
- Scale horizontally by using a dedicated inference server
- Switch between local and remote modes easily

## Changes Made

### 1. `local_vlm.py`
- Added `api_base_url` and `api_timeout` parameters to `__init__`
- Added `_call_api_general_inference()` method for API communication
- Modified `caption_image()` and `answer_question()` to check `self.use_api` and route accordingly
- **All existing local transformer code is preserved and functional**

### 2. `orchestrator.py`
- Already had `qwen_api_url` parameter (from previous work)
- Updated `_ensure_models()` to support both API and local VLM initialization
- Fixed default behavior: if no API URL is provided, uses local mode

### 3. `qwen_api/api.py`
- Already has `/general_inference` endpoint (from previous work)
- Supports multi-image inference with arbitrary text prompts
- Perfect for VQA use cases

## Usage

### Option 1: Local Mode (Default)
```python
from local_vlm import LocalVLM

# Uses transformers/torch locally
vlm = LocalVLM(
    device="cuda",
    system_prompt="You are a helpful assistant."
)

answer = vlm.answer_question("image.jpg", "What's in this image?")
```

### Option 2: API Mode
```python
from local_vlm import LocalVLM

# Makes HTTP requests to API server
vlm = LocalVLM(
    api_base_url="http://localhost:8001",  # or remote server
    system_prompt="You are a helpful assistant."
)

answer = vlm.answer_question("image.jpg", "What's in this image?")
```

### Option 3: Using Environment Variable
```bash
export QWEN_API_URL=http://your-server:8001
```

```python
vlm = LocalVLM(
    api_base_url=os.environ.get("QWEN_API_URL"),
    system_prompt="You are a helpful assistant."
)
```

### Option 4: Via Orchestrator CLI
```bash
# Local mode (default)
python orchestrator.py \
    --image image.jpg \
    --question "What's in this image?"

# API mode
python orchestrator.py \
    --image image.jpg \
    --question "What's in this image?" \
    --qwen_api_url http://localhost:8001
```

## API Requirements

When using API mode, the API server must have the following endpoint:

**POST /general_inference**
- `images`: List of image files (multipart/form-data)
- `user_prompt`: Text question/instruction
- `system_prompt`: System-level instructions (optional)
- `max_new_tokens`: Maximum tokens to generate
- `n_batch`: Batch size for processing

Returns:
```json
{
  "response": "The model's generated text answer"
}
```

The `qwen_api/api.py` already implements this endpoint correctly.

## Starting the API Server

```bash
cd qwen_api
uvicorn api:app --host 0.0.0.0 --port 8001
```

## Dependencies

### Local Mode
- torch
- transformers
- PIL
- All other existing dependencies

### API Mode
- requests (only this is needed!)
- PIL (for image handling)
- No torch/transformers required!

## Backward Compatibility

✅ **100% backward compatible**
- If you don't pass `api_base_url`, everything works exactly as before
- All existing code continues to work without modifications
- Agent mode and tools still work in local mode (API mode doesn't support tools yet)

## Limitations

### API Mode Current Limitations:
1. **No tool support yet** - Agent mode with geometric tools only works in local mode
2. **No streaming** - API responses come back as complete text (not token-by-token)
3. **File I/O** - Images must be on disk or PIL.Image objects (can't use arbitrary image data)

### Future Enhancements:
- Add tool calling support to API
- Implement streaming responses
- Add batch inference endpoint
- Support for base64-encoded images

## Testing

Run the test script to verify both modes:

```bash
cd vqa
python test_api_mode.py
```

## Architecture Diagram

```
┌─────────────────────────────────────────┐
│         Orchestrator (VQA)              │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │         LocalVLM                  │ │
│  │                                   │ │
│  │  if api_base_url:                │ │
│  │    ┌─────────────────────┐       │ │
│  │    │  API Mode           │       │ │
│  │    │  - HTTP requests    │       │ │
│  │    │  - No torch needed  │       │ │
│  │    └─────────────────────┘       │ │
│  │  else:                           │ │
│  │    ┌─────────────────────┐       │ │
│  │    │  Local Mode         │       │ │
│  │    │  - transformers     │       │ │
│  │    │  - torch/CUDA       │       │ │
│  │    └─────────────────────┘       │ │
│  └───────────────────────────────────┘ │
└─────────────────────────────────────────┘
                 │
                 │ API Mode HTTP
                 ▼
    ┌─────────────────────────────┐
    │   Qwen API Server           │
    │   (qwen_api/api.py)         │
    │                             │
    │   POST /general_inference   │
    │   - Multi-image support     │
    │   - Arbitrary prompts       │
    │   - System prompts          │
    └─────────────────────────────┘
```

## Files Modified

1. ✅ `vqa/local_vlm.py` - Added API mode support
2. ✅ `vqa/orchestrator.py` - Updated to properly support optional API URL
3. ✅ `qwen_api/api.py` - Already has general_inference endpoint

## Files Created

1. `vqa/test_api_mode.py` - Test script for both modes
2. `vqa/API_MODE_README.md` - This documentation

## No Files Deleted or Broken

- ✅ All original functionality preserved
- ✅ No breaking changes
- ✅ Backward compatible
- ✅ Original transformer-based code intact
