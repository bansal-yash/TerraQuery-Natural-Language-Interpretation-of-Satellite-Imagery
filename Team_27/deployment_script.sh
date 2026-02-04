#!/bin/bash

set -e  # Exit on error

echo "==================================="
echo "Starting GeoNLI Installation"
echo "==================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if script is run as root
if [ "$EUID" -eq 0 ]; then 
    print_error "Please do not run this script as root (without sudo)"
    exit 1
fi

# Install system packages
print_status "Installing system packages..."
sudo apt update
sudo apt install -y nginx python3-pip python3-venv conda gunicorn uvicorn python3-fastapi python3-httpx python3-dotenv python3-openai python3-pil

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create conda environment
print_status "Creating conda environment..."
conda create -n geonli python=3.10 -y

# Initialize conda for bash (if not already done)
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell hook --shell bash)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # Fallback: source the profile script if available
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    print_error "Conda is installed but shell hook could not be loaded"
    exit 1
fi

# Activate environment
print_status "Activating conda environment..."
conda activate geonli

# Install PyTorch
print_status "Installing PyTorch..."
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1

# Install requirements (prefer installation/requirements.txt if present)
REQ_FILE="installation/requirements.txt"
if [ ! -f "$REQ_FILE" ]; then
    REQ_FILE="requirements.txt"
fi

if [ ! -f "$REQ_FILE" ]; then
    print_error "requirements file not found (looked for installation/requirements.txt or requirements.txt)"
    exit 1
fi

print_status "Installing Python requirements from $REQ_FILE..."
pip install -r "$REQ_FILE"

# Install unsloth packages
print_status "Installing unsloth packages..."
pip install unsloth==2025.11.6 unsloth_zoo==2025.11.6

# Clone and install transformers
print_status "Installing transformers from source..."
if [ -d "transformers" ]; then
    rm -rf transformers
fi
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install '.[torch]'
cd ..
rm -rf transformers

# Create backend .env
print_status "Creating backend/.env configuration..."
mkdir -p backend
cat > backend/.env << 'EOF'
ROUTER_URL=http://localhost:8001/router
CAPTION_URL=http://localhost:8001/caption
GROUND_URL=http://localhost:8001/bbox
VQA_URL=http://localhost:8001/vqa
EOF

# Create VQA .env
print_status "Creating vqa/.env configuration..."
mkdir -p vqa
cat > vqa/.env << 'EOF'
ALLOWED_HOSTS=localhost,127.0.0.1
SAM_URL=http://localhost:6767
QWEN_ADAPTER_ROOT=checkpoints
QWEN_NULL_ADAPTER_ROOT=artifacts
# BAND_CLASSIFIER_CHECKPOINT=path/to/ckpt  # Uncomment and set if needed
EOF

# Create SAM3 .env
print_status "Creating sam3/.env configuration..."
mkdir -p sam3
cat > sam3/.env << 'EOF'
ALLOWED_HOSTS=localhost,127.0.0.1
SAM_URL=http://localhost:6767
QWEN_URL=http://localhost:8001
EOF

# Check if ports are available
print_status "Checking if required ports are available..."
for port in 80 6767 8001; do
    if sudo lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_warning "Port $port is already in use. Please free it before continuing."
    fi
done

# Start SAM3 service
print_status "Starting SAM3 service on port 6767..."
cd sam3
if [ ! -f "api.py" ]; then
    print_error "sam3/api.py not found"
    exit 1
fi
nohup uvicorn api:app --host 0.0.0.0 --port 6767 > sam3.log 2>&1 &
echo $! > sam3.pid
cd ..

# Wait a bit for SAM3 to start
sleep 2

# Start VQA service
print_status "Starting VQA service on port 8001..."
cd vqa
if [ ! -f "unified_api.py" ]; then
    print_error "vqa/unified_api.py not found"
    exit 1
fi
nohup uvicorn unified_api:app --host 0.0.0.0 --port 8001 > vqa.log 2>&1 &
echo $! > vqa.pid
cd ..

# Wait for VQA to start
sleep 2

# Copy frontend to web directory
print_status "Deploying frontend..."
sudo rm -rf /var/www/frontend
sudo cp -r frontend /var/www/

# Get server IP (or use localhost)
SERVER_IP=$(hostname -I | awk '{print $1}')
if [ -z "$SERVER_IP" ]; then
    SERVER_IP="127.0.0.1"
fi

print_status "Configuring Nginx for IP: $SERVER_IP"

# Configure Nginx
sudo tee /etc/nginx/sites-available/interiit > /dev/null << EOF
server {
    listen 80;
    server_name $SERVER_IP localhost;

    client_max_body_size 50M;

    root /var/www/frontend/dist;
    index index.html;

    location / {
        try_files \$uri \$uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable Nginx site
sudo rm -f /etc/nginx/sites-enabled/default
sudo ln -sf /etc/nginx/sites-available/interiit /etc/nginx/sites-enabled/interiit

# Test Nginx configuration
print_status "Testing Nginx configuration..."
if sudo nginx -t; then
    print_status "Nginx configuration is valid"
else
    print_error "Nginx configuration has errors"
    exit 1
fi

# Restart Nginx
print_status "Restarting Nginx..."
sudo systemctl enable nginx
sudo systemctl restart nginx

# Update backend .env with correct IPs
print_status "Updating backend configuration..."
cat > backend/.env << EOF
API_MODE=custom
ROUTER_URL=http://127.0.0.1:8001/router
CAPTION_URL=http://127.0.0.1:8001/caption
GROUND_URL=http://127.0.0.1:8001/bbox
VQA_URL=http://127.0.0.1:8001/vqa
EOF

# Start backend service
print_status "Starting backend service..."
cd backend
if [ ! -f "api.py" ]; then
    print_error "backend/api.py not found"
    exit 1
fi
nohup gunicorn -w 2 -k uvicorn.workers.UvicornWorker api:app --bind 127.0.0.1:8000 > backend.log 2>&1 &
echo $! > backend.pid
cd ..

print_status "Waiting for services to start..."
sleep 3

# Print completion message
echo ""
echo "==================================="
print_status "Installation complete!"
echo "==================================="
echo ""
echo "Services started:"
echo "  - SAM3:    http://localhost:6767"
echo "  - VQA:     http://localhost:8001"
echo "  - Backend: http://localhost:8000"
echo "  - Nginx:   http://$SERVER_IP/ (or http://localhost/)"
echo ""
echo "Process IDs saved in:"
echo "  - sam3/sam3.pid"
echo "  - vqa/vqa.pid"
echo "  - backend/backend.pid"
echo ""
echo "Logs available at:"
echo "  - sam3/sam3.log"
echo "  - vqa/vqa.log"
echo "  - backend/backend.log"
echo ""
print_warning "To stop services, run: kill \$(cat sam3/sam3.pid vqa/vqa.pid backend/backend.pid)"
echo ""
