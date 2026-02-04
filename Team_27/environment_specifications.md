# Environment Specifications

OS: Ubuntu 20.04LTS, 22.04LTS, 24.04LTS.
Python: Version 3.10 (conda/micromamba recommended)
Linux Packages: conda nginx gunicorn uvicorn python3-fastapi python3-httpx python3-dotenv python3-openai python3-pil
Python Packages: torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 unsloth==2025.11.6 unsloth_zoo==2025.11.6 numpy==2.2.6 pillow==12.0.0 opencv-python-headless==4.12.0.88 torch==2.9.0 torchvision==0.24.0 transformers==4.57.3 accelerate==1.12.0 einops==0.8.1 fastapi==0.122.0 uvicorn==0.38.0 python multipart==0.0.20 pydantic==2.12.5 requests==2.32.5 langchain-core==1.1.0 langgraph==1.0.4
Hardware: GPU node with NVIDIA drivers and CUDA>=12.6 support for SAM and VLM models. Atleast 24GB, recommended 32GB
VPN: We used our institutes vpn to link the deployment server to the gpu node
