"""lightweight client for sam3 api interactions"""

from pathlib import Path
from typing import Any, Dict, List
import requests


class Sam3ApiClient:
    """helper for sam3 api requests"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
    
    def get_masks(
        self, 
        prompt: str, 
        image_path: Path,
        apply_filtering: bool = False,
        use_nms: bool = False,
        iom_threshold: float = 0.3,
        nms_iou_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """get segmentation masks from text prompt with optional filtering"""
        return self._post(
            "/masks", prompt, image_path,
            apply_filtering=apply_filtering,
            use_nms=use_nms,
            iom_threshold=iom_threshold,
            nms_iou_threshold=nms_iou_threshold
        )
    
    def get_boxes(self, prompt: str, image_path: Path) -> Dict[str, Any]:
        """get bounding boxes from text prompt"""
        return self._post("/boxes", prompt, image_path)
    
    def get_masks_with_box(self, prompt: str, box: List[float], image_path: Path) -> Dict[str, Any]:
        """get segmentation masks using text prompt + box hint"""
        return self._post_with_box("/masks_with_box", prompt, box, image_path)
    
    def _post(
        self, 
        endpoint: str, 
        prompt: str, 
        image_path: Path,
        apply_filtering: bool = False,
        use_nms: bool = False,
        iom_threshold: float = 0.3,
        nms_iou_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """internal: post request without box"""
        files = {"image": (image_path.name, image_path.read_bytes(), "image/jpeg")}
        payload = {
            "prompt": prompt,
            "apply_filtering": str(apply_filtering).lower(),
            "use_nms": str(use_nms).lower(),
            "iom_threshold": str(iom_threshold),
            "nms_iou_threshold": str(nms_iou_threshold)
        }
        response = requests.post(
            f"{self.base_url}{endpoint}",
            data=payload,
            files=files,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
    
    def _post_with_box(self, endpoint: str, prompt: str, box: List[float], image_path: Path) -> Dict[str, Any]:
        """internal: post request with box hint"""
        files = {"image": (image_path.name, image_path.read_bytes(), "image/jpeg")}
        box_str = ",".join(str(coord) for coord in box)
        payload = {"prompt": prompt, "box": box_str}
        response = requests.post(
            f"{self.base_url}{endpoint}",
            data=payload,
            files=files,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
    
    def get_masks_with_boxes(self, prompt: str, boxes: List[List[float]], image_path: Path) -> Dict[str, Any]:
        """get segmentation masks using text prompt + multiple box hints"""
        files = {"image": (image_path.name, image_path.read_bytes(), "image/jpeg")}
        
        # convert list of boxes to semicolon-separated string
        # format: "x1,y1,x2,y2;x1,y1,x2,y2;..."
        boxes_str = ";".join(",".join(str(coord) for coord in box) for box in boxes)
        
        payload = {"prompt": prompt, "boxes": boxes_str}
        response = requests.post(
            f"{self.base_url}/masks_with_boxes",
            data=payload,
            files=files,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()