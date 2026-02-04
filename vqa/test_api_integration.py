#!/usr/bin/env python3
"""Test script to verify API-based VQA system."""

import os
import sys
import requests

# Test configuration
QWEN_API_URL = os.environ.get("QWEN_API_URL", "http://localhost:8001")
TEST_IMAGE = "P0003_0002.png"  # Sample image in vqa directory

def test_api_availability():
    """Test if Qwen API server is running."""
    print("=" * 80)
    print("TEST 1: API Server Availability")
    print("=" * 80)
    
    try:
        response = requests.get(f"{QWEN_API_URL}/docs", timeout=5)
        if response.status_code == 200:
            print(f"✅ API server is running at {QWEN_API_URL}")
            return True
        else:
            print(f"❌ API server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot reach API server: {e}")
        print(f"   Make sure server is running: uvicorn api:app --host 0.0.0.0 --port 8001")
        return False

def test_general_inference_endpoint():
    """Test the new /general_inference endpoint."""
    print("\n" + "=" * 80)
    print("TEST 2: /general_inference Endpoint")
    print("=" * 80)
    
    if not os.path.exists(TEST_IMAGE):
        print(f"❌ Test image not found: {TEST_IMAGE}")
        return False
    
    try:
        with open(TEST_IMAGE, "rb") as f:
            files = [("images", (TEST_IMAGE, f.read(), "image/jpeg"))]
        
        data = {
            "user_prompt": "Describe this image briefly.",
            "system_prompt": "You are a helpful visual assistant.",
            "max_new_tokens": 100
        }
        
        print(f"Sending request to {QWEN_API_URL}/general_inference...")
        response = requests.post(
            f"{QWEN_API_URL}/general_inference",
            data=data,
            files=files,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Endpoint responded successfully")
            print(f"\nResponse preview:")
            print("-" * 80)
            print(result.get("response", "")[:200] + "...")
            print("-" * 80)
            return True
        else:
            print(f"❌ Endpoint returned status {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_local_vlm_import():
    """Test importing the new LocalVLM."""
    print("\n" + "=" * 80)
    print("TEST 3: LocalVLM Import (No torch/transformers)")
    print("=" * 80)
    
    try:
        # This should work without torch/transformers installed
        from local_vlm import LocalVLM
        print("✅ LocalVLM imported successfully")
        
        # Check that torch is NOT imported
        import sys
        torch_imported = 'torch' in sys.modules
        transformers_imported = 'transformers' in sys.modules
        
        if torch_imported:
            print("⚠️  WARNING: torch is imported (should not be needed)")
        else:
            print("✅ torch is NOT imported (good)")
        
        if transformers_imported:
            print("⚠️  WARNING: transformers is imported (should not be needed)")
        else:
            print("✅ transformers is NOT imported (good)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import LocalVLM: {e}")
        return False

def test_vlm_initialization():
    """Test initializing LocalVLM with API URL."""
    print("\n" + "=" * 80)
    print("TEST 4: LocalVLM Initialization")
    print("=" * 80)
    
    try:
        from local_vlm import LocalVLM
        
        vlm = LocalVLM(
            api_base_url=QWEN_API_URL,
            system_prompt="You are a test assistant.",
            stream_thoughts=False,
            timeout=30
        )
        
        print("✅ LocalVLM initialized successfully")
        print(f"   API URL: {vlm.api_base_url}")
        print(f"   Timeout: {vlm.timeout}s")
        return True
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vlm_caption_image():
    """Test image captioning through API."""
    print("\n" + "=" * 80)
    print("TEST 5: LocalVLM.caption_image()")
    print("=" * 80)
    
    if not os.path.exists(TEST_IMAGE):
        print(f"❌ Test image not found: {TEST_IMAGE}")
        return False
    
    try:
        from local_vlm import LocalVLM
        
        vlm = LocalVLM(
            api_base_url=QWEN_API_URL,
            stream_thoughts=False
        )
        
        print(f"Captioning image: {TEST_IMAGE}")
        caption = vlm.caption_image(TEST_IMAGE)
        
        print("✅ Captioning successful")
        print(f"\nCaption:")
        print("-" * 80)
        print(caption)
        print("-" * 80)
        return True
        
    except Exception as e:
        print(f"❌ Captioning failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("VQA API INTEGRATION TEST SUITE")
    print("=" * 80)
    print(f"API URL: {QWEN_API_URL}")
    print(f"Test Image: {TEST_IMAGE}")
    print()
    
    results = []
    
    # Run tests
    results.append(("API Availability", test_api_availability()))
    
    if results[-1][1]:  # Only continue if API is available
        results.append(("General Inference Endpoint", test_general_inference_endpoint()))
    
    results.append(("LocalVLM Import", test_local_vlm_import()))
    results.append(("LocalVLM Initialization", test_vlm_initialization()))
    
    if results[0][1]:  # Only if API is available
        results.append(("LocalVLM Caption", test_vlm_caption_image()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 80)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
