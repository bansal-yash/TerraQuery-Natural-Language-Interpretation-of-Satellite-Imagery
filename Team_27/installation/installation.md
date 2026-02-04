# Deployment Guide

This document outlines deployment steps for the  application.

Sections:
* Loading Models
* Frontend (React + Nginx)
* Backend (FastAPI + Gunicorn)

---
# 1. Models

## 1.1 Installation
On every machine please install using 

```bash
conda create -n geonli python=3.10 -y
conda activate geonli
chmod +x install.sh
./install.sh
```

## 1.2 Exposing APIs
In one terminal (or one machine)
```bash
cd sam3
uvicorn api:app --host 0.0.0.0 --port 6767
```

In another terminal (or machine) run 
```bash
cd vqa
uvicorn unified_api:app --host 0.0.0.0 --port 8001
```

Ensure that the corresponding ports are free

## 1.3 setting .envs

### backend (.env in `backend/`)
Use the sample env file from `backend/.env.example` and copy it to the hosts that run the backend services. Adjust host/port values to match your deployment.

```bash
cd /home/ubuntu/backend
cp .env.example .env
# Update these to point to your running VQA stack
ROUTER_URL=http://<ip_vqa>:8001/router
CAPTION_URL=http://<ip_vqa>:8001/caption
GROUND_URL=http://<ip_vqa>:8001/bbox
VQA_URL=http://<ip_vqa>:8001/vqa
```

- `ip_vqa` is the machine running `uvicorn unified_api:app` (Section 1.2). If reverse-proxying on the same host, you can use `127.0.0.1`.
- Restart the backend service after edits (Section 3.4).

### vqa service (.env in `vqa/`)
Base on `vqa/.env.example`:
```bash
ALLOWED_HOSTS=localhost,127.0.0.1,<ip_sam3>
SAM_URL=http://<ip_sam3>:6767            # sam3 API endpoint
QWEN_ADAPTER_ROOT=checkpoints            # optional: override adapter root
QWEN_NULL_ADAPTER_ROOT=artifacts         # optional: override null adapter root
BAND_CLASSIFIER_CHECKPOINT=path/to/ckpt  # optional: SAR/false-color classifier
```

### sam3 service (.env in `sam3/`)
Base on `sam3/.env.example`:
```bash
ALLOWED_HOSTS=localhost,127.0.0.1,<ip_sam3>
SAM_URL=http://<ip_sam3>:6767            # sam3 self URL
QWEN_URL=http://<ip_vqa>:8001            # VQA service URL (if sam3 calls qwen)
```

Ensure `SAM_URL` in vqa points to the sam3 host/port above. Restart both services after updating envs.


# 2. Frontend

## 2.1 Enter & Move Frontend

```bash
cd frontend
npm i
npm run build
cd ..
sudo mv frontend /var/www/
```

## 2.2 Install & Configure Nginx

```bash
sudo apt install nginx
```

Create site config:

```bash
sudo tee /etc/nginx/sites-available/interiit << EOF > /dev/null
server {
    listen 80;
    server_name <YOUR-SERVER-IP>;

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
    }
}
EOF
```

Enable site:

```bash
sudo rm /etc/nginx/sites-enabled/default
sudo ln -s /etc/nginx/sites-available/interiit /etc/nginx/sites-enabled/interiit
sudo systemctl enable --now nginx
```

---

# 3. Backend

## 3.1 Enter Backend

```bash
cd backend
```

## 3.2 Environment Setup

```bash
sudo tee /home/ubuntu/backend/.env << EOF > /dev/null
API_MODE=custom
ROUTER_URL=http://{ip_vqa}:8001/router
CAPTION_URL=http://{ip_vqa}:8001/caption
GROUND_URL=http://{ip_vqa}:8001/bbox
VQA_URL=http://{ip_vqa}:8001/vqa
EOF
```

## 3.3 Install Dependencies

```bash
sudo apt install gunicorn uvicorn python3-fastapi python3-httpx python3-dotenv python3-openai python3-pil
```

## 3.4 Gunicorn Systemd Service

```bash
sudo tee /etc/systemd/system/interiit-backend.service << 'EOF'
[Unit]
Description=FastAPI Backend
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/backend
ExecStart=/usr/bin/gunicorn -w 2 -k uvicorn.workers.UvicornWorker api:app --bind 127.0.0.1:8000

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now interiit-backend
```
