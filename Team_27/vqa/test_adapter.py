#!/usr/bin/env python3
"""
Comprehensive test script for LoRA adapter loading/unloading in LocalVLM.

Tests:
1. Load LoRA adapter and verify it's loaded
2. Generate output with adapter loaded
3. Unload adapter and verify it's unloaded
4. Generate output with base model (post-unload)
5. GPU memory tracking throughout
6. Timing measurements for all operations
"""

import unsloth
import argparse
import os
import sys
import time
import torch
import gc
from pathlib import Path
from PIL import Image
from typing import Optional

# Add vqa directory to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from local_vlm import LocalVLM


def get_gpu_memory_mb():
    """Get current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024**2  # MB
    return 0


def print_separator(title=""):
    """Print a visual separator."""
    print("\n" + "="*80)
    if title:
        print(f"  {title}")
        print("="*80)
    print()


def test_adapter_loading(
    adapter_path: str,
    test_image: str,
    test_question: str,
    base_model: str = "Qwen/Qwen3-VL-8B-Instruct",
    precision: str = "bf16",
    attn_impl: Optional[str] = "sdpa",
    prefer_unsloth: bool = False,
    base_adapter: Optional[str] = None,
    perf_logs: bool = False,
    null_adapter: Optional[str] = None,
):
    """
    Main test function for adapter loading/unloading.
    
    Args:
        adapter_path: Path to LoRA adapter checkpoint
        test_image: Path to test image
        test_question: Question to ask about the image
        base_model: Base model name/path
    """
    
    print_separator("ADAPTER LOADING/UNLOADING TEST")
    
    # Verify inputs exist
    if not os.path.exists(test_image):
        raise FileNotFoundError(f"Test image not found: {test_image}")
    
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    
    print(f"Base Model: {base_model}")
    print(f"Adapter Path: {adapter_path}")
    print(f"Test Image: {test_image}")
    print(f"Test Question: {test_question}")
    print(f"Precision: {precision}")
    print(f"Attention impl: {attn_impl or 'transformers-default'}")
    print(f"Prefer FastVisionModel: {prefer_unsloth}")
    print(f"Base adapter: {base_adapter or 'not provided'}")
    print(f"Null adapter: {null_adapter or base_adapter or 'not provided'}")
    
    # Initialize VLM
    print_separator("Step 1: Initialize Base Model")
    init_start = time.time()
    initial_memory = get_gpu_memory_mb()
    
    vlm = LocalVLM(
        model_name=base_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        stream_thoughts=False,
        precision=precision,
        attn_implementation=attn_impl,
        prefer_unsloth=prefer_unsloth,
        base_adapter_path=base_adapter,
        null_adapter_path=null_adapter,
        enable_perf_logs=perf_logs,
    )
    
    init_time = time.time() - init_start
    post_init_memory = get_gpu_memory_mb()
    
    print(f"✓ Model initialized in {init_time:.2f}s")
    print(f"✓ GPU Memory: {initial_memory:.0f}MB → {post_init_memory:.0f}MB "
          f"(+{post_init_memory - initial_memory:.0f}MB)")
    
    # Test 1: Generate with base model
    print_separator("Step 2: Generate with Base Model (Before Adapter)")
    gen1_start = time.time()
    pre_gen1_memory = get_gpu_memory_mb()
    
    base_output = vlm.answer_question(test_image, test_question, max_length=128)
    
    gen1_time = time.time() - gen1_start
    post_gen1_memory = get_gpu_memory_mb()
    
    print(f"\nBase Model Output:\n{base_output}\n")
    print(f"✓ Generation time: {gen1_time:.2f}s")
    print(f"✓ GPU Memory: {pre_gen1_memory:.0f}MB → {post_gen1_memory:.0f}MB")
    
    # Test 2: Load adapter
    print_separator("Step 3: Load LoRA Adapter")
    pre_load_memory = get_gpu_memory_mb()
    
    load_result = vlm.load_lora(adapter_path)
    
    post_load_memory = get_gpu_memory_mb()
    
    # Assertions for load
    assert load_result.get("status") in ["success", "already_loaded"], \
        f"Adapter load failed: {load_result.get('error')}"
    assert vlm._lora_loaded == True, "Adapter not marked as loaded"
    assert vlm._lora_path == adapter_path, "Adapter path mismatch"
    
    print(f"✓ Adapter loaded successfully")
    print(f"✓ Load time: {load_result.get('load_time', 0):.2f}s")
    print(f"✓ GPU Memory: {pre_load_memory:.0f}MB → {post_load_memory:.0f}MB "
          f"(+{post_load_memory - pre_load_memory:.0f}MB)")
    print(f"✓ Assertions passed: adapter marked as loaded")
    
    # Test 3: Generate with adapter
    print_separator("Step 4: Generate with Adapter Loaded")
    gen2_start = time.time()
    pre_gen2_memory = get_gpu_memory_mb()
    
    adapter_output = vlm.answer_question(test_image, test_question, max_length=128)
    
    gen2_time = time.time() - gen2_start
    post_gen2_memory = get_gpu_memory_mb()
    
    print(f"\nAdapter Model Output:\n{adapter_output}\n")
    print(f"✓ Generation time: {gen2_time:.2f}s")
    print(f"✓ GPU Memory: {pre_gen2_memory:.0f}MB → {post_gen2_memory:.0f}MB")
    
    # Compare outputs
    if base_output.strip() != adapter_output.strip():
        print("✓ Outputs differ (expected when adapter modifies behavior)")
    else:
        print("⚠ Outputs identical (adapter may not be affecting inference)")
    
    # Test 4: Unload adapter
    print_separator("Step 5: Unload LoRA Adapter")
    pre_unload_memory = get_gpu_memory_mb()
    
    unload_result = vlm.unload_lora(null_path=null_adapter or base_adapter)
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    post_unload_memory = get_gpu_memory_mb()
    
    # Assertions for unload
    assert unload_result.get("status") == "success", \
        f"Adapter unload failed: {unload_result.get('error')}"
    assert vlm._lora_loaded == False, "Adapter still marked as loaded"
    assert vlm._lora_path is None, "Adapter path not cleared"
    
    print(f"✓ Adapter unloaded successfully")
    print(f"✓ Unload time: {unload_result.get('unload_time', 0):.2f}s")
    print(f"✓ GPU Memory: {pre_unload_memory:.0f}MB → {post_unload_memory:.0f}MB "
          f"(-{pre_unload_memory - post_unload_memory:.0f}MB)")
    print(f"✓ Assertions passed: adapter marked as unloaded")
    
    # Test 5: Generate after unload (should revert to base behavior)
    print_separator("Step 6: Generate After Unload (Should Match Base)")
    gen3_start = time.time()
    pre_gen3_memory = get_gpu_memory_mb()
    
    post_unload_output = vlm.answer_question(test_image, test_question, max_length=128)
    
    gen3_time = time.time() - gen3_start
    post_gen3_memory = get_gpu_memory_mb()
    
    print(f"\nPost-Unload Output:\n{post_unload_output}\n")
    print(f"✓ Generation time: {gen3_time:.2f}s")
    print(f"✓ GPU Memory: {pre_gen3_memory:.0f}MB → {post_gen3_memory:.0f}MB")
    
    # Summary
    print_separator("SUMMARY")
    
    print("Timing Summary:")
    print(f"  Model initialization:  {init_time:.2f}s")
    print(f"  Base generation:       {gen1_time:.2f}s")
    print(f"  Adapter load:          {load_result.get('load_time', 0):.2f}s")
    print(f"  Adapter generation:    {gen2_time:.2f}s")
    print(f"  Adapter unload:        {unload_result.get('unload_time', 0):.2f}s")
    print(f"  Post-unload generation: {gen3_time:.2f}s")
    
    print("\nMemory Summary:")
    print(f"  Initial:               {initial_memory:.0f}MB")
    print(f"  After init:            {post_init_memory:.0f}MB (+{post_init_memory - initial_memory:.0f}MB)")
    print(f"  After adapter load:    {post_load_memory:.0f}MB (+{post_load_memory - pre_load_memory:.0f}MB)")
    print(f"  After adapter unload:  {post_unload_memory:.0f}MB (-{pre_unload_memory - post_unload_memory:.0f}MB)")
    print(f"  Final:                 {post_gen3_memory:.0f}MB")
    
    print("\nOutput Comparison:")
    print(f"  Base ≠ Adapter:        {base_output.strip() != adapter_output.strip()}")
    print(f"  Base ≈ Post-unload:    {base_output.strip() == post_unload_output.strip()}")
    
    print("\n✅ ALL TESTS PASSED")
    
    return {
        "base_output": base_output,
        "adapter_output": adapter_output,
        "post_unload_output": post_unload_output,
        "timings": {
            "init": init_time,
            "gen_base": gen1_time,
            "load": load_result.get('load_time', 0),
            "gen_adapter": gen2_time,
            "unload": unload_result.get('unload_time', 0),
            "gen_post": gen3_time
        },
        "memory": {
            "initial": initial_memory,
            "post_init": post_init_memory,
            "post_load": post_load_memory,
            "post_unload": post_unload_memory,
            "final": post_gen3_memory
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test LoRA adapter loading/unloading in LocalVLM"
    )
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="Path to LoRA adapter checkpoint directory"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to test image"
    )
    parser.add_argument(
        "--question",
        default="What do you see in this image?",
        help="Question to ask about the image"
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Base model name or path (use Qwen/Qwen3-VL-8B-Instruct or unsloth/Qwen3-VL-8B-Instruct)"
    )
    parser.add_argument(
        "--precision",
        choices=["bf16", "fp16"],
        default="bf16",
        help="Force LocalVLM to load in the requested precision"
    )
    parser.add_argument(
        "--attn-impl",
        choices=["sdpa", "flash_attention_2", "none"],
        default="sdpa",
        help="Attention implementation hint for transformers loader (use 'none' to skip)"
    )
    parser.add_argument(
        "--prefer-unsloth",
        action="store_true",
        help="Force FastVisionModel fallback (disables SDPA)"
    )
    parser.add_argument(
        "--base-adapter",
        help="Path to a pre-generated null/base adapter for faster swapping",
        default=None,
    )
    parser.add_argument(
        "--perf-logs",
        action="store_true",
        help="Enable LocalVLM performance logging"
    )
    parser.add_argument(
        "--null-adapter",
        help="Path to the null/base adapter used when unloading",
        default=None,
    )
    
    args = parser.parse_args()
    
    try:
        results = test_adapter_loading(
            adapter_path=args.adapter_path,
            test_image=args.image,
            test_question=args.question,
            base_model=args.base_model,
            precision=args.precision,
            attn_impl=None if args.attn_impl == "none" else args.attn_impl,
            prefer_unsloth=args.prefer_unsloth,
            base_adapter=args.base_adapter,
            perf_logs=args.perf_logs,
            null_adapter=args.null_adapter,
        )
        
        # Exit successfully
        sys.exit(0)
        
    except AssertionError as e:
        print(f"\n❌ ASSERTION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
