"""Smoke tests for the qwen_mtmd API with bbox overlay demo."""

import json
import os
from pathlib import Path
from typing import Dict

import requests
from PIL import Image, ImageDraw, ImageFont

BASE_URL = os.environ.get("QWEN_MTMD_API_URL", "http://visionl40.cse.iitd.ac.in:8001")
# IMAGES_DIR = Path(os.environ.get("QWEN_MTMD_IMAGE_DIR", Path(__file__).resolve().parent))
# IMAGE_NAME = os.environ.get("QWEN_MTMD_IMAGE_NAME", "image1.png")
IMAGES_DIR = Path("/home/samyak/scratch/interiit/llama-shit/qwen3vl_bindings")
IMAGE_NAME = "P0003_0002.png"
OVERLAY_PATH = Path("bbox_overlay.png")

REQUESTS: Dict[str, Dict] = {
    "bbox": {
        "image_name": IMAGE_NAME,
        "object_name": "yellow buses",
    },
    "caption": {
        "image_name": IMAGE_NAME,
    },
    "features": {
        "image_name": IMAGE_NAME,
        "describer": "vehicle",
    },
}


def _post(endpoint: str, payload: Dict) -> Dict:
    url = f"{BASE_URL}/{endpoint}"
    image_path = (IMAGES_DIR / payload.get("image_name", IMAGE_NAME)).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Test image not found: {image_path}")

    files = {"image": open(image_path, "rb")}
    data = {}
    # map expected form fields per endpoint
    if endpoint == "bbox":
        data["object_name"] = payload.get("object_name")
        data["n_batch"] = str(payload.get("n_batch", 64))
        data["max_new_tokens"] = str(payload.get("max_new_tokens", 128))
    elif endpoint == "caption":
        data["n_batch"] = str(payload.get("n_batch", 64))
        data["max_new_tokens"] = str(payload.get("max_new_tokens", 256))
    elif endpoint == "features":
        data["describer"] = payload.get("describer")
        data["n_batch"] = str(payload.get("n_batch", 64))
        data["max_new_tokens"] = str(payload.get("max_new_tokens", 128))

    resp = requests.post(url, files=files, data=data)
    files["image"].close()
    resp.raise_for_status()
    return resp.json()


def _draw_overlay(image_path: Path, boxes: Dict[str, Dict]) -> Path:
    image = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for idx, box in enumerate(boxes):
        bounds = (box["x1"], box["y1"], box["x2"], box["y2"])
        draw.rectangle(bounds, outline=(255, 0, 0, 255), width=3)
        label = box.get("label") or f"object_{idx+1}"
        draw.text((box["x1"], max(0, box["y1"] - 12)), label, fill=(255, 255, 255, 255), font=font)
    image.save(OVERLAY_PATH)
    return OVERLAY_PATH


def main() -> None:
    requests_log: Dict[str, Dict] = {}
    for endpoint, payload in REQUESTS.items():
        try:
            resp = _post(endpoint, payload)
        except requests.HTTPError as exc:
            print(f"{endpoint} failed: {exc}")
            continue
        requests_log[endpoint] = resp
        print(f"{endpoint} response: {json.dumps(resp, indent=2)}")
        if endpoint == "bbox":
            image_path = (IMAGES_DIR / IMAGE_NAME).resolve()
            if not image_path.exists():
                print(f"Image for overlay not found: {image_path}")
                continue
            overlay_path = _draw_overlay(image_path, resp.get("boxes", []))
            print(f"Saved bbox overlay to {overlay_path}")

    with open("api_test_results.json", "w") as fp:
        json.dump(requests_log, fp, indent=2)
    print("Test run complete; responses dumped to api_test_results.json")


if __name__ == "__main__":
    main()
