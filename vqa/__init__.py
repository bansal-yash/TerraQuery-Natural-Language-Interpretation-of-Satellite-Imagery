"""VQA orchestration package for Grounded-SAM project.

This package contains a small orchestration that: generates a base query
for GroundingDINO (via a small LLM call), runs grounding to get boxes,
runs SAM on the chosen boxes for fine-grained masks/crops, and then
produces an answer using the VLM + LLM.

The code is intentionally lightweight and acts as an orchestration layer.
"""

__all__ = ["orchestrator", "groq_client", "local_vlm"]
