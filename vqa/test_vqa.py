#!/usr/bin/env python3
"""
Test script for VQA system components.
Tests each module independently before running full pipeline.
"""

import sys
import os

def test_local_llm():
    """Test local LLM class extraction."""
    print("\n" + "="*80)
    print("TEST 1: Local LLM Class Extraction")
    print("="*80)
    
    try:
        from local_llm import LocalLLM
        
        llm = LocalLLM(device="cuda")
        
        test_queries = [
            "count all red and yellow buses",
            "how many cars are in the parking lot?",
            "is there a person walking?",
            "what is the area of the largest building?",
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            classes = llm.extract_classes_from_query(query, max_classes=5)
            print(f"Extracted: {classes}")
            
        print("\n✅ Local LLM test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Local LLM test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_geometric_utils():
    """Test geometric utilities."""
    print("\n" + "="*80)
    print("TEST 2: Geometric Utilities")
    print("="*80)
    
    try:
        import numpy as np
        import geometric_utils
        
        # Create test masks
        mask1 = np.zeros((100, 100), dtype=bool)
        mask1[20:60, 30:80] = True  # Rectangle
        
        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[50:80, 10:40] = True  # Another rectangle
        
        # Test property computation
        props1 = geometric_utils.compute_mask_properties(mask1)
        print("\nMask 1 properties:")
        for k, v in props1.items():
            print(f"  {k}: {v:.2f}")
        
        # Test distance
        dist = geometric_utils.compute_min_distance_between_masks(mask1, mask2)
        print(f"\nDistance between masks: {dist:.2f} pixels")
        
        # Test relative position
        pos = geometric_utils.get_relative_position(mask1, mask2)
        print(f"Relative position: mask2 is '{pos}' of mask1")
        
        # Test overlap
        overlap = geometric_utils.compute_mask_overlap(mask1, mask2)
        print(f"\nOverlap metrics:")
        for k, v in overlap.items():
            print(f"  {k}: {v:.2f}")
        
        print("\n✅ Geometric utilities test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Geometric utilities test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_system_prompt():
    """Test system prompt loading."""
    print("\n" + "="*80)
    print("TEST 3: System Prompt")
    print("="*80)
    
    try:
        prompt_path = "systemprompt.txt"
        
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r') as f:
                prompt = f.read()
            
            print(f"\nLoaded system prompt ({len(prompt)} characters)")
            print("\nFirst 500 characters:")
            print(prompt[:500])
            print("...")
            
            print("\n✅ System prompt test PASSED")
            return True
        else:
            print(f"\n❌ System prompt not found at: {prompt_path}")
            return False
            
    except Exception as e:
        print(f"\n❌ System prompt test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vlm_init():
    """Test VLM initialization (without full inference)."""
    print("\n" + "="*80)
    print("TEST 4: VLM Initialization")
    print("="*80)
    
    try:
        from local_vlm import LocalVLM
        
        print("\nInitializing VLM (this may take a while)...")
        vlm = LocalVLM(
            device="cuda",
            stream_thoughts=True,
        )
        
        print(f"✅ VLM loaded successfully")
        print(f"   Model device: {vlm.device}")
        print(f"   Streaming enabled: {vlm.stream_thoughts}")
        
        print("\n✅ VLM initialization test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ VLM initialization test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_orchestrator_init():
    """Test orchestrator initialization."""
    print("\n" + "="*80)
    print("TEST 5: Orchestrator Initialization")
    print("="*80)
    
    try:
        from orchestrator import Orchestrator
        
        orch = Orchestrator(
            grounding_config="../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            grounding_checkpoint="../groundingdino_swint_ogc.pth",
            sam_encoder="vit_h",
            sam_checkpoint="../sam_vit_h_4b8939.pth",
            device="cuda",
            system_prompt_path="systemprompt.txt",
        )
        
        print("\n✅ Orchestrator initialized successfully")
        print(f"   Grounding config: {orch.grounding_config}")
        print(f"   SAM encoder: {orch.sam_encoder}")
        print(f"   Device: {orch.device}")
        
        print("\n✅ Orchestrator initialization test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Orchestrator initialization test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("VQA SYSTEM TEST SUITE")
    print("="*80)
    
    tests = [
        ("Geometric Utilities", test_geometric_utils),
        ("System Prompt", test_system_prompt),
        ("Local LLM", test_local_llm),
        ("VLM Initialization", test_vlm_init),
        ("Orchestrator Initialization", test_orchestrator_init),
    ]
    
    results = []
    for test_name, test_func in tests:
        passed = test_func()
        results.append((test_name, passed))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:40s} {status}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! System is ready.")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed. Please fix errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
