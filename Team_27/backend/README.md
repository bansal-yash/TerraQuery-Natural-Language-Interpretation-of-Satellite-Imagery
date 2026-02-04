# Backend (backend)

FastAPI proxy (root_path `/api`) that fronts the VQA services and handles CORS for the frontend.

## Features
- FastAPI proxy with `/api` root_path that forwards router/caption/bbox/VQA calls to the VQA node.
- Optional OpenAI-compatible router mode (`API_MODE=openai`) plus default raw HTTP mode.
- Supports multipart uploads or `image_url` fetch, light image compression, and bbox overlay rendering helpers.
- Normalizes certain answers (binary yes/no, numerical with pixel-to-meter conversion when spatial resolution is provided).
- CORS enabled for local dev origins.

## Environment
Copy `.env.example` to `.env` and set the VQA endpoints:
```
ROUTER_URL=http://<ip_vqa>:8000/router
CAPTION_URL=http://<ip_vqa>:8000/caption
GROUND_URL=http://<ip_vqa>:8000/bbox
VQA_URL=http://<ip_vqa>:8000/vqa
API_MODE=custom
```

## Run
```bash
cd backend
conda activate geonli
uvicorn api:app --host 0.0.0.0 --port 8000 --root-path /api
```
For production, see the systemd + gunicorn snippet in `installation/installation.md`.

## API (proxied)
Base URL (after Nginx): `http://<frontend_host>/api`
- `POST /router` — image+prompt → VQA router classification
- `POST /caption` — image → VQA caption
- `POST /bbox` — image + `object_name` → VQA bbox
- `POST /vqa/{mode}` — image + `question`, `mode` in `{attribute,binary,numerical,filtering}`

## Notes
- Ensure VPN/allowlist from frontend to backend, and backend to VQA node.
- Restart after `.env` changes.

## 📁 Back-End Directory Structure
```
backend/
├── api.py                          # Main API server file (Flask/FastAPI)
├── .env.example                    # Example environment variables template
└── README.md                       # Backend documentation and setup guide
```
