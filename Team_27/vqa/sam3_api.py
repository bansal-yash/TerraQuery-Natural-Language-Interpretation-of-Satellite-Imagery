"""Example client requests against a running SAM3 API deployment."""

from pathlib import Path
from typing import Any, Dict
import os

# Load .env file early
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests


def _get_base_url() -> str:
    """Get SAM_URL at runtime (after .env is loaded)."""
    return os.environ.get("SAM_URL", "")


BASE_URL = _get_base_url()  # For backward compat, but prefer reading at init time
IMAGE_PATH = Path("../P0003_0002.png")
DEFAULT_PROMPT = "yellow buses"
REQUEST_TIMEOUT = 120


class Sam3ApiClient:
    """Helper to perform /masks, /boxes, and /merged_masks requests against the API."""

    def __init__(self, base_url: str = None, timeout: int = REQUEST_TIMEOUT):
        # Read SAM_URL at runtime if not provided (ensures .env is loaded)
        if base_url is None:
            base_url = os.environ.get("SAM_URL", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, endpoint: str, payload: Dict[str, Any], image_path: Path, image_type: str = None) -> Dict[str, Any]:
        if image_type:
            payload["image_type"] = image_type
        files = {
            "image": (image_path.name, image_path.read_bytes(), "image/jpeg")
        }
        response = requests.post(
            f"{self.base_url}{endpoint}",
            data=payload,
            files=files,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_masks(self, prompt: str = DEFAULT_PROMPT, image_path: Path = IMAGE_PATH, image_type: str = None, apply_filtering: bool = False, iom_threshold: float = 0.3) -> Dict[str, Any]:
        payload = {"prompt": prompt}
        if apply_filtering:
            payload["apply_filtering"] = "true"
            payload["iom_threshold"] = str(iom_threshold)
        return self._post("/masks", payload, image_path, image_type)

    def get_boxes(self, prompt: str = DEFAULT_PROMPT, image_path: Path = IMAGE_PATH, image_type: str = None) -> Dict[str, Any]:
        return self._post("/boxes", {"prompt": prompt}, image_path, image_type)

    def get_masks_with_boxes(
        self,
        prompt: str = DEFAULT_PROMPT,
        boxes: list = None,
        image_path: Path = IMAGE_PATH,
        image_type: str = None,
    ) -> Dict[str, Any]:
        """Get masks using text prompt and multiple box prompts."""
        import json
        payload = {
            "prompt": prompt,
            "boxes": json.dumps(boxes) if boxes else "[]",
        }
        return self._post("/masks_with_boxes", payload, image_path, image_type)

    def get_merged_masks(
        self,
        object_name: str = DEFAULT_PROMPT,
        image_path: Path = IMAGE_PATH,
        debug_visualization: bool = False,
        iom_threshold: float = 0.5,
        coverage_threshold: float = 0.7,
        n_batch: int = 64,
        max_new_tokens: int = 128,
        image_type: str = None,
    ) -> Dict[str, Any]:
        payload = {
            "object_name": object_name,
            "debug_visualization": str(debug_visualization).lower(),
            "iom_threshold": str(iom_threshold),
            "coverage_threshold": str(coverage_threshold),
            "n_batch": str(n_batch),
            "max_new_tokens": str(max_new_tokens),
        }
        return self._post("/merged_masks", payload, image_path, image_type)


if __name__ == "__main__":
    client = Sam3ApiClient()
    print("Requesting merged masks from", client.base_url)
    try:
        merged_response = client.get_merged_masks()
        print("Merged response keys:", list(merged_response.keys()))
        detections = merged_response.get("detections", [])
        print("Detections count:", len(detections))
        print("Merged masks available:", len(merged_response.get("masks", [])))
    except requests.exceptions.RequestException as exc:
        print("Failed to fetch merged masks:", exc)
