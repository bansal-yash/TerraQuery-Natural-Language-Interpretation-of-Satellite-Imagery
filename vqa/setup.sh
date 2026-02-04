#!/bin/bash
# Setup script for VQA system
# Run this to install all required dependencies

set -e  # Exit on error

echo "======================================================================"
echo "VQA System Setup"
echo "======================================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"
echo ""

# Check CUDA availability
echo "Checking CUDA availability..."
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "✓ CUDA version: $cuda_version"
else
    echo "⚠ CUDA not found - will use CPU mode (slower)"
fi
echo ""

# Install PyTorch (adjust for your CUDA version)
echo "Installing PyTorch..."
echo "Note: Modify this command based on your CUDA version"
echo "Visit: https://pytorch.org/get-started/locally/"
echo ""

# For CUDA 11.8 (adjust as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Installing Transformers and related packages..."
pip install transformers>=4.35.0
pip install accelerate
pip install sentencepiece
pip install protobuf

echo ""
echo "Installing vision packages..."
pip install opencv-python>=4.8.0
pip install pillow>=10.0.0
pip install numpy>=1.24.0

echo ""
echo "Installing GroundingDINO..."
# Assumes GroundingDINO is in parent directory
if [ -d "../GroundingDINO" ]; then
    cd ../GroundingDINO
    pip install -e .
    cd ../vqa
    echo "✓ GroundingDINO installed"
else
    echo "⚠ GroundingDINO directory not found at ../GroundingDINO"
    echo "  Please install manually: https://github.com/IDEA-Research/GroundingDINO"
fi

echo ""
echo "Installing Segment Anything..."
if [ -d "../segment_anything" ]; then
    cd ../segment_anything
    pip install -e .
    cd ../vqa
    echo "✓ Segment Anything installed"
else
    echo "⚠ segment_anything directory not found at ../segment_anything"
    echo "  Please install manually: pip install git+https://github.com/facebookresearch/segment-anything.git"
fi

echo ""
echo "======================================================================"
echo "Downloading model checkpoints..."
echo "======================================================================"
echo ""

# Create checkpoints directory
mkdir -p ../checkpoints

# GroundingDINO checkpoint
if [ ! -f "../groundingdino_swint_ogc.pth" ]; then
    echo "Downloading GroundingDINO checkpoint..."
    wget -P .. https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
else
    echo "✓ GroundingDINO checkpoint already exists"
fi

# SAM checkpoint
if [ ! -f "../sam_vit_h_4b8939.pth" ]; then
    echo "Downloading SAM ViT-H checkpoint..."
    wget -P .. https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
else
    echo "✓ SAM checkpoint already exists"
fi

echo ""
echo "======================================================================"
echo "Verifying installation..."
echo "======================================================================"
echo ""

python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

echo ""
echo "======================================================================"
echo "Setup complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "1. Run tests: python test_vqa.py"
echo "2. Try examples: python example_usage.py"
echo "3. Run orchestrator: python orchestrator.py --image <path> --question <question>"
echo ""
echo "For detailed usage, see README.md"
echo ""
