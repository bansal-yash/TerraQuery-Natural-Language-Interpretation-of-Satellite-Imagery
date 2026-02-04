"""Test script to verify API mode integration in local_vlm.py"""

import os
import sys

# Test 1: Import and check modes
print("=" * 60)
print("TEST 1: Import LocalVLM and check both modes")
print("=" * 60)

from local_vlm import LocalVLM

# Test local mode (should work without API)
print("\n[Test 1a] Initializing LocalVLM in LOCAL mode (no api_base_url)...")
try:
    vlm_local = LocalVLM(
        system_prompt="Test local mode",
        stream_thoughts=False
    )
    print(f"✓ Local mode initialized: use_api={vlm_local.use_api}, device={vlm_local.device}")
except Exception as e:
    print(f"✗ Local mode failed: {e}")

# Test API mode (initialization only, no actual API call yet)
print("\n[Test 1b] Initializing LocalVLM in API mode...")
try:
    vlm_api = LocalVLM(
        api_base_url="http://localhost:8001",
        system_prompt="Test API mode",
        stream_thoughts=False
    )
    print(f"✓ API mode initialized: use_api={vlm_api.use_api}, device={vlm_api.device}")
    print(f"  API base URL: {vlm_api.api_base_url}")
except Exception as e:
    print(f"✗ API mode failed: {e}")

# Test 2: Check method signatures
print("\n" + "=" * 60)
print("TEST 2: Verify method signatures")
print("=" * 60)

methods_to_check = ['caption_image', 'answer_question', '_call_api_general_inference']
for method_name in methods_to_check:
    if hasattr(vlm_api, method_name):
        print(f"✓ Method '{method_name}' exists")
    else:
        print(f"✗ Method '{method_name}' missing")

# Test 3: Show usage examples
print("\n" + "=" * 60)
print("TEST 3: Usage Examples")
print("=" * 60)

print("""
# Example 1: Using LOCAL mode (with transformers)
vlm = LocalVLM(
    device="cuda",
    system_prompt="You are a helpful assistant."
)
answer = vlm.answer_question("image.jpg", "What's in this image?")

# Example 2: Using API mode (no transformers needed)
vlm = LocalVLM(
    api_base_url="http://localhost:8001",  # or use env var QWEN_API_URL
    system_prompt="You are a helpful assistant."
)
answer = vlm.answer_question("image.jpg", "What's in this image?")

# Example 3: In orchestrator.py
orchestrator = Orchestrator(
    ...,
    qwen_api_url="http://localhost:8001"  # This gets passed to LocalVLM
)
""")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
