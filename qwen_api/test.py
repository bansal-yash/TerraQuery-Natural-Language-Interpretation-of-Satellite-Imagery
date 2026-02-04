"""Smoke tests for the qwen_mtmd API with bbox overlay demo."""

import json
import os
from pathlib import Path
from typing import Dict, List

import requests
from PIL import Image, ImageDraw, ImageFont

BASE_URL = os.environ.get("QWEN_MTMD_API_URL", "http://aih.cse.iitd.ac.in:8000")
# IMAGES_DIR = Path(os.environ.get("QWEN_MTMD_IMAGE_DIR", Path(__file__).resolve().parent))
# IMAGE_NAME = os.environ.get("QWEN_MTMD_IMAGE_NAME", "image1.png")
IMAGES_DIR = Path(".")
IMAGE_NAME = "/home/samyak/scratch/interiit/sample_dataset_inter_iit_v1_2/sample1.png"
# IMAGE_NAME = "P0003_0002.png"
OVERLAY_PATH = Path("bbox_overlay.png")

REQUESTS: Dict[str, Dict] = {
    "bbox": {
        "image_name": IMAGE_NAME,
        "object_name": "digit",
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

    #resp = requests.post(url, files=files, data=data)
    resp = requests.post(url, files=files, data=data, proxies={"http": None, "https": None})
    files["image"].close()
    resp.raise_for_status()
    return resp.json()


def _draw_overlay(image_path: Path, boxes: List[Dict]) -> Path:
    """Draw bounding boxes on image following the API's JSON schema.
    
    Expected box format per MISSION CRITICAL FORMATTING INSTRUCTIONS:
    {
        "label": str,
        "x1": int [0,1000],
        "y1": int [0,1000],
        "x2": int [0,1000],
        "y2": int [0,1000]
    }
    """
    image = Image.open(image_path).convert("RGBA")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    
    if not boxes:
        print("Warning: No boxes to draw")
        image.save(OVERLAY_PATH)
        return OVERLAY_PATH
    
    for idx, box in enumerate(boxes):
        # Validate box structure per schema
        required_keys = ["x1", "y1", "x2", "y2"]
        if not all(key in box for key in required_keys):
            print(f"Warning: Box {idx} missing required keys: {box}")
            continue
        
        # Denormalize coordinates from [0,1000] to pixel coordinates
        try:
            x1 = int(box["x1"] * width / 1000)
            y1 = int(box["y1"] * height / 1000)
            x2 = int(box["x2"] * width / 1000)
            y2 = int(box["y2"] * height / 1000)
        except (ValueError, TypeError) as e:
            print(f"Warning: Invalid coordinate values in box {idx}: {e}")
            continue
        
        bounds = (x1, y1, x2, y2)
        draw.rectangle(bounds, outline=(255, 0, 0, 255), width=3)
        label = box.get("label") or f"object_{idx+1}"
        draw.text((x1, max(0, y1 - 12)), label, fill=(255, 255, 255, 255), font=font)
    
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
            
            boxes = resp.get("boxes", [])
            if not isinstance(boxes, list):
                print(f"Error: 'boxes' is not a list. Got: {type(boxes)}")
                continue
            
            overlay_path = _draw_overlay(image_path, boxes)
            print(f"Saved bbox overlay to {overlay_path}")

    with open("api_test_results.json", "w") as fp:
        json.dump(requests_log, fp, indent=2)
    print("Test run complete; responses dumped to api_test_results.json")


if __name__ == "__main__":
    main()
