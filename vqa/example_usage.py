#!/usr/bin/env python3
"""
Example usage of the VQA system.

This demonstrates how to use the orchestrator for various types of questions.
"""

from orchestrator import Orchestrator
import sys

def main():
    # Initialize orchestrator
    print("\n🚀 Initializing VQA System...")
    
    orch = Orchestrator(
        grounding_config="../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        grounding_checkpoint="../groundingdino_swint_ogc.pth",
        sam_encoder="vit_h",
        sam_checkpoint="../sam_vit_h_4b8939.pth",
        device="cuda",  # Change to "cpu" if no GPU
        system_prompt_path="systemprompt.txt",
    )
    
    # Example questions
    examples = [
        {
            "image": "../demo_images/example1.jpg",
            "questions": [
                "How many red buses are visible?",
                "What is the color of the largest vehicle?",
                "Is there a person in the image?",
            ]
        },
        {
            "image": "../demo_images/example2.jpg",
            "questions": [
                "Count all the cars",
                "What is the area of the largest building?",
            ],
            "custom_classes": ["car", "building"]  # Optional: override LLM extraction
        }
    ]
    
    # Process each example
    for idx, example in enumerate(examples, 1):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {idx}")
        print(f"{'='*80}")
        print(f"Image: {example['image']}")
        
        # Check if custom classes are provided
        custom_classes = example.get("custom_classes", None)
        
        for question in example["questions"]:
            print(f"\n{'─'*80}")
            print(f"Question: {question}")
            print(f"{'─'*80}")
            
            try:
                answer = orch.run(
                    image_path=example["image"],
                    question=question,
                    classes=custom_classes,
                    score_threshold=0.35
                )
                
                print(f"\n📝 Answer:\n{answer}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("✅ All examples processed!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
