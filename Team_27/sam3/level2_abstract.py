"""client-side integration with intelligent mask merging"""

import os

# Load .env file early before any env vars are read
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

allowed_hosts = os.environ.get("ALLOWED_HOSTS", "")
sam_url = os.environ.get("SAM_URL", "")
qwen_url = os.environ.get("QWEN_URL", "")
os.environ["NO_PROXY"] = allowed_hosts
os.environ["no_proxy"] = allowed_hosts
for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(proxy_var, None)

from typing import List, Dict, Any
from pathlib import Path
from collections import defaultdict
from sam3_client import Sam3ApiClient
from mask_merging import merge_iom_graph, visualize_merge_decision
import requests


class VisionPipeline:
    """orchestrates qwen bbox detection with sam3 segmentation"""
    
    def __init__(self, qwen_url: str = qwen_url, sam3_url: str = sam_url):
        self.qwen_url = qwen_url.rstrip('/')
        self.sam3_client = Sam3ApiClient(base_url=sam3_url)
    
    def detect_and_segment_dual(
        self, 
        image_path: str, 
        object_name: str,
        iom_threshold: float = 0.5,
        coverage_threshold: float = 0.7,
        debug_visualization: bool = False,
        n_batch: int = 64, 
        max_new_tokens: int = 128,
        image_type: str = None
    ) -> Dict[str, Any]:
        """run both models and intelligently merge"""
        image_path = Path(image_path)
        
        # step 1: get detections from qwen
        qwen_response = self._call_qwen_bbox(image_path, object_name, n_batch, max_new_tokens, image_type)
        
        if not qwen_response.get("boxes"):
            return {
                "detections": [], 
                "masks": [],
                "model1_boxes": [],
                "model2_boxes": [],
            }
        
        qwen_boxes = qwen_response["boxes"]
        
        # step 2: run model1 (no box guidance) - ENABLE FILTERING to reduce over-segmentation
        model1_response = self.sam3_client.get_masks(
            prompt=object_name, 
            image_path=image_path,
            apply_filtering=True,  # NEW: enable IoM filtering for Model1
            iom_threshold=0.3      # NEW: 30% overlap removal threshold
        )
        
        # step 3: run model2 (with ALL qwen boxes as guidance)
        # convert all qwen boxes to normalized format
        normalized_boxes = [self._convert_box_format(box) for box in qwen_boxes]
        
        model2_response = self.sam3_client.get_masks_with_boxes(  # plural!
            prompt=object_name,
            boxes=normalized_boxes,  # pass all boxes
            image_path=image_path
        )
        
        # extract masks and boxes from both models
        masks1 = model1_response.get("masks", [])
        masks2 = model2_response.get("masks", [])
        
        # extract bounding boxes from both models for debug output
        model1_boxes = [mask.get("box") for mask in masks1 if mask.get("box")]
        model2_boxes = [mask.get("box") for mask in masks2 if mask.get("box")]
        
        # step 4: intelligent merge using IoM-based graph algorithm
        merged_masks, debug_info = merge_iom_graph(
            masks1, masks2, 
            iom_threshold=iom_threshold,
            coverage_threshold=coverage_threshold,
            debug=debug_visualization
        )
        
        # step 5: create debug visualization if requested
        if debug_visualization:
            # comprehensive visualization with actual masks
            visualize_merge_decision(
                image_path=str(image_path),
                masks1=masks1,
                masks2=masks2,
                merged=merged_masks,
                debug_info=debug_info,
                output_path=f"merge_debug_{object_name}.png"
            )
        
        return {
            "detections": qwen_boxes,  # qwen detections
            "model1_count": len(masks1),
            "model2_count": len(masks2),
            "merged_count": len(merged_masks),
            "model1_boxes": model1_boxes,  # boxes from model1
            "model2_boxes": model2_boxes,  # boxes from model2
            "masks": merged_masks,
            "image_size": model1_response.get("image_size")
        }
    
    def _visualize_model_boxes(
        self, 
        image_path: str, 
        boxes: List[List[float]], 
        output_path: str,
        color: str = "red",
        label: str = "box"
    ):
        """visualize bounding boxes from one model"""
        from PIL import Image, ImageDraw, ImageFont
        
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # CHANGE: smaller font size for compact labels
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)  # was 30
        except Exception:
            font = ImageFont.load_default()
        
        img_w, img_h = img.size
        
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            x1_px = int(x1 * img_w)
            y1_px = int(y1 * img_h)
            x2_px = int(x2 * img_w)
            y2_px = int(y2 * img_h)
            
            # draw box
            draw.rectangle([x1_px, y1_px, x2_px, y2_px], outline=color, width=4)
            
            # CHANGE: use simple number instead of "model_1_1" etc
            text = str(idx + 1)  # just "1", "2", "3" etc (was f"{label}_{idx}")
            draw.text((x1_px + 5, y1_px + 5), text, fill=color, font=font)
        
        img.save(output_path)
        print(f"saved {label} boxes visualization to {output_path}")
    
    def _call_qwen_bbox(self, image_path: Path, object_name: str, 
                       n_batch: int, max_new_tokens: int, image_type: str = None) -> Dict[str, Any]:
        """call qwen bbox endpoint"""
        data = {
            "object_name": object_name,
            "n_batch": n_batch,
            "max_new_tokens": max_new_tokens
        }
        if image_type:
            data["image_type"] = image_type
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{self.qwen_url}/bbox",
                files={"image": f},
                data=data
            )
        response.raise_for_status()
        return response.json()
    
    def _convert_box_format(self, qwen_box: Dict[str, int]) -> List[float]:
        """convert qwen box (0-1000 int) to sam3 format (0-1 float list)"""
        return [
            qwen_box["x1"] / 1000.0,
            qwen_box["y1"] / 1000.0,
            qwen_box["x2"] / 1000.0,
            qwen_box["y2"] / 1000.0,
        ]