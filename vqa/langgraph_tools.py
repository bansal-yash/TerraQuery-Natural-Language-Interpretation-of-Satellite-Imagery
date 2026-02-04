# langgraph_tools.py
from langchain_core.tools import tool, Tool
from pydantic import BaseModel
import geometric_utils
import cv2
import numpy as np
from typing import List, Dict

# We define small Pydantic schemas for input validation (optional but recommended)
class MaskPathSchema(BaseModel):
    mask_path: str

class TwoMasksSchema(BaseModel):
    mask1_path: str
    mask2_path: str

class MaskListSchema(BaseModel):
    mask_paths: List[str]

# Use the @tool decorator to register a function as a LangGraph/Tool.
# The docstring becomes the tool description shown to the model.
@tool
def compute_mask_properties_tool(mask_path: str) -> Dict:
    """
    Compute geometric properties of a segmentation mask stored at mask_path (PNG).
    Returns a JSON-serializable dict of properties (area, centroid_x, centroid_y, orientation, aspect_ratio, bbox_width, bbox_height, perimeter).
    """
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return {"error": f"mask not found: {mask_path}"}
    mask = (m > 0)
    props = geometric_utils.compute_mask_properties(mask)
    return props

@tool
def compute_min_distance_between_masks_tool(mask1_path: str, mask2_path: str) -> Dict:
    """
    Compute centroid distance (pixels) between two mask images.
    Returns { "distance": float }.
    """
    m1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
    m2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)
    if m1 is None or m2 is None:
        return {"error": "one or both mask files not found"}
    dist = geometric_utils.compute_min_distance_between_masks(m1 > 0, m2 > 0)
    return {"distance": float(dist)}

@tool
def compute_mask_overlap_tool(mask1_path: str, mask2_path: str) -> Dict:
    """
    Compute overlap metrics (intersection, union, iou).
    """
    m1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
    m2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)
    if m1 is None or m2 is None:
        return {"error": "one or both mask files not found"}
    overlap = geometric_utils.compute_mask_overlap(m1 > 0, m2 > 0)
    return overlap

@tool
def get_relative_position_tool(mask1_path: str, mask2_path: str) -> Dict:
    """
    Return relative position of mask2 wrt mask1 (string: 'above', 'below-left', etc.)
    """
    m1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
    m2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)
    if m1 is None or m2 is None:
        return {"error": "one or both mask files not found"}
    pos = geometric_utils.get_relative_position(m1 > 0, m2 > 0)
    return {"position": pos}

@tool
def compute_total_area_tool(mask_paths: List[str]) -> Dict:
    """
    Compute total union area (pixels) of a list of mask image paths.
    """
    masks = []
    for p in mask_paths:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            return {"error": f"mask not found: {p}"}
        masks.append(m > 0)
    total = geometric_utils.compute_total_area(masks)
    return {"total_area": float(total)}
