# api.py
import os
import re
import json
import math
import base64
import shutil
import tempfile

# Disable proxy for local requests (must be before other imports that use requests/httpx)
os.environ["NO_PROXY"] = "localhost,127.0.0.1,aih.cse.iitd.ac.in,.iitd.ac.in"
os.environ["no_proxy"] = "localhost,127.0.0.1,aih.cse.iitd.ac.in,.iitd.ac.in"

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Response, status
from fastapi.concurrency import run_in_threadpool
from orchestrator import Orchestrator
from typing import Optional, List, Dict, Any, Tuple, Union
import logging
from contextlib import contextmanager
from image_type_classifier import BandClassifier, find_checkpoint_candidate

# Config
DEVICE = "cuda"
CROPS_MANIFEST_PATH = "vqa_outputs_fresh/crops_manifest.json"

ADAPTER_ROOT = os.environ.get("QWEN_ADAPTER_ROOT", "checkpoints")
NULL_ADAPTER_ROOT = os.environ.get("QWEN_NULL_ADAPTER_ROOT", "artifacts")


def _default_path(root: Optional[str], subdir: str) -> str:
    if not root:
        return subdir
    return os.path.join(root, subdir)


CAPTION_DEFAULT_ADAPTER_PATH = os.environ.get(
    "CAPTION_DEFAULT_ADAPTER_PATH",
    _default_path(ADAPTER_ROOT, "normal_caption"),
)
CAPTION_DEFAULT_NULL_ADAPTER_PATH = os.environ.get(
    "CAPTION_DEFAULT_NULL_ADAPTER_PATH",
    _default_path(NULL_ADAPTER_ROOT, "normal_caption_null_adapter"),
)
CAPTION_SAR_ADAPTER_PATH = os.environ.get(
    "CAPTION_SAR_ADAPTER_PATH",
    _default_path(ADAPTER_ROOT, "sar_caption"),
)
CAPTION_SAR_NULL_ADAPTER_PATH = os.environ.get(
    "CAPTION_SAR_NULL_ADAPTER_PATH",
    _default_path(NULL_ADAPTER_ROOT, "sar_caption_null_adapter"),
)
VQA_SAR_ADAPTER_PATH = os.environ.get(
    "VQA_SAR_ADAPTER_PATH",
    _default_path(ADAPTER_ROOT, "sar_bbox"),
)
VQA_SAR_NULL_ADAPTER_PATH = os.environ.get(
    "VQA_SAR_NULL_ADAPTER_PATH",
    _default_path(NULL_ADAPTER_ROOT, "sar_bbox_null_adapter"),
)
DEFAULT_NULL_ADAPTER_PATH = (
    os.environ.get("NULL_ADAPTER_PATH")
    or CAPTION_DEFAULT_NULL_ADAPTER_PATH
    or VQA_SAR_NULL_ADAPTER_PATH
)

# Router classification system prompt
ROUTER_SYSTEM_PROMPT = """You are a task classifier for visual question answering. Given an image and a user request, classify which task is needed.

Any user request that involves locating objects, grounding, drawing bounding boxes, highlighting instances, or otherwise selecting subsets of objects MUST be classified as VQA_FILTERING.

Reply with ONLY one of these exact words (nothing else):
- CAPTION: User wants a description/caption of the image
- VQA_ATTRIBUTE: User asks about properties/attributes of objects (color, shape, material, etc.)
- VQA_NUMERICAL: User asks "how many" or wants to count objects
- VQA_BINARY: User asks a yes/no question about the image
- VQA_FILTERING: User wants to locate/find objects, draw bounding boxes, or filter objects matching certain criteria

Respond with exactly one word from the list above."""

app = FastAPI()
orchestrator: Optional[Orchestrator] = None
band_classifier: Optional[BandClassifier] = None
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup():
    global orchestrator
    # LOADS 8B MODEL ONCE
    orchestrator = Orchestrator(
        device=DEVICE
    )
    orchestrator._ensure_models()
    # Initialize band classifier if checkpoint available
    global band_classifier
    try:
        ckpt = os.environ.get("BAND_CLASSIFIER_CHECKPOINT") or find_checkpoint_candidate()
        if ckpt:
            band_classifier = BandClassifier(ckpt)
            logger.info("Initialized band classifier from %s", ckpt)
        else:
            logger.info("No band classifier checkpoint found; image type classification disabled.")
            band_classifier = None
    except Exception as e:
        logger.warning("Failed to initialize band classifier: %s", e)
        band_classifier = None

def save_temp(file):
    """Save uploaded file to a persistent temp location (caller must clean up)."""
    # Use a dedicated temp directory that persists
    temp_dir = os.path.join(tempfile.gettempdir(), "vqa_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate unique filename
    import uuid
    ext = os.path.splitext(file.filename or ".jpg")[1] or ".jpg"
    tmp_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}{ext}")
    
    with open(tmp_path, "wb") as tmp:
        shutil.copyfileobj(file.file, tmp)
    return tmp_path

def cleanup_temp(path: str):
    """Safely remove temp file."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

# --- ENDPOINTS ---

@app.post("/router")
async def router(bg_tasks: BackgroundTasks, image: UploadFile = File(...), prompt: str = Form(...)):
    """Classify the user prompt into one of the task categories."""
    tmp = save_temp(image)
    
    # Save original VLM state
    original_system_prompt = orchestrator.vlm.system_prompt
    original_suppress = orchestrator.vlm._suppress_debug_prints
    
    # Configure VLM for clean classification output
    prefix = _maybe_image_type_prefix(tmp)
    if prefix:
        orchestrator.vlm.system_prompt = prefix + ROUTER_SYSTEM_PROMPT
    else:
        orchestrator.vlm.system_prompt = ROUTER_SYSTEM_PROMPT
    orchestrator.vlm._suppress_debug_prints = True
    
    try:
        # Call VLM with our router system prompt (short max_length for classification)
        response = orchestrator.vlm.answer_question(tmp, f"User request: {prompt}", max_length=64)
    finally:
        # Restore original VLM state
        orchestrator.vlm.system_prompt = original_system_prompt
        orchestrator.vlm._suppress_debug_prints = original_suppress
        # Clean up temp file after processing is complete
        cleanup_temp(tmp)
    
    # Extract just the classification word
    classification = response.strip().upper()
    
    # Validate and normalize the response
    valid_classes = ["CAPTION", "VQA_ATTRIBUTE", "VQA_NUMERICAL", "VQA_BINARY", "VQA_FILTERING"]
    
    for valid in valid_classes:
        if valid in classification:
            return {"classification": valid}
    
    # Default fallback
    return {"classification": "VQA_ATTRIBUTE"}

def _parse_bool(value: Optional[str], default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"false", "0", "no", "off"}


def _load_bbox_lookup(base_image_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Read crops_manifest.json (or configured path) and build lookup by mask filename."""
    manifest_path = CROPS_MANIFEST_PATH
    if not manifest_path or not os.path.exists(manifest_path):
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest_data = json.load(fh)
    except Exception:
        return {}
    # Allow manifest to be either a dict (single entry) or list of dicts
    entries: List[Dict[str, Any]]
    if isinstance(manifest_data, list):
        entries = [entry for entry in manifest_data if isinstance(entry, dict)]
    elif isinstance(manifest_data, dict):
        entries = [manifest_data]
    else:
        return {}
    if not entries:
        return {}
    image_basename = os.path.basename(base_image_path)
    matching_entries = [
        entry
        for entry in entries
        if os.path.basename(entry.get("image_path", "")) == image_basename
    ]
    if matching_entries:
        entries_to_use = matching_entries
    else:
        entries_to_use = entries[-1:]
    lookup: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries_to_use:
        for prompt in entry.get("prompts", []):
            for crop in prompt.get("crops", []):
                if not isinstance(crop, dict):
                    continue
                bbox_pixels = crop.get("bbox_pixels")
                if not bbox_pixels or len(bbox_pixels) != 4:
                    continue
                props = crop.get("properties", {}) or {}
                orientation = float(props.get("orientation", 0.0) or 0.0)
                mask_path = crop.get("mask_path", "") or ""
                crop_index = crop.get("index")
                sam_index = crop.get("sam_index")
                sam_prompt = (crop.get("sam_prompt") or "").strip()
                identifiers = []
                if mask_path:
                    identifiers.append(os.path.basename(mask_path))
                if crop_index is not None:
                    identifiers.append(f"mask_{crop_index}.png")
                if sam_index is not None:
                    identifiers.append(f"mask_{sam_index}.png")
                prompt_label = (prompt.get("prompt") or "").strip()
                entry_payload = {
                    "bbox": bbox_pixels,
                    "orientation": orientation,
                    "sam_prompt": sam_prompt,
                    "prompt_label": prompt_label,
                    "mask_path": mask_path,
                }

                for ident in identifiers:
                    if not ident:
                        continue
                    norm_ident = ident.lower()
                    lookup.setdefault(norm_ident, []).append(entry_payload)
    return lookup


def _compute_oriented_box_points(
    bbox_pixels: List[int],
    orientation: float,
    img_width: int,
    img_height: int,
) -> List[tuple]:
    """Return four (x, y) tuples for the rotated rectangle defined in manifest."""
    if not bbox_pixels or len(bbox_pixels) != 4:
        return []
    x1, y1, x2, y2 = bbox_pixels
    width = max(float(x2 - x1), 1.0)
    height = max(float(y2 - y1), 1.0)
    cx = x1 + width / 2.0
    cy = y1 + height / 2.0
    target_orientation = float(orientation or 0.0)
    initial_orientation = 90.0 if height >= width else 0.0
    angle_rad = math.radians(target_orientation - initial_orientation)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    half_w = width / 2.0
    half_h = height / 2.0
    raw_points = [
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    ]
    points = []
    for dx, dy in raw_points:
        rx = cx + dx * cos_a - dy * sin_a
        ry = cy + dx * sin_a + dy * cos_a
        rx = int(round(min(max(rx, 0.0), img_width - 1)))
        ry = int(round(min(max(ry, 0.0), img_height - 1)))
        points.append((rx, ry))
    return points


def _compute_mask_box_points(
    mask_path: Optional[str],
    img_width: int,
    img_height: int,
) -> Optional[List[tuple]]:
    """Return four (x, y) tuples for the mask's minimum-area rectangle."""
    if not mask_path:
        return None
    candidates = [mask_path]
    if not os.path.isabs(mask_path):
        base_dir = os.path.dirname(CROPS_MANIFEST_PATH) or "."
        candidates.append(os.path.join(base_dir, mask_path))
    import cv2

    mask = None
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            mask = cv2.imread(candidate, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                break
    if mask is None:
        return None
    _, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    points = cv2.findNonZero(thresh)
    if points is None:
        return None
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    ordered = []
    for x, y in box:
        x_clamped = int(round(min(max(x, 0.0), img_width - 1)))
        y_clamped = int(round(min(max(y, 0.0), img_height - 1)))
        ordered.append((x_clamped, y_clamped))
    return ordered


def _order_points_clockwise(points: List[tuple]) -> List[tuple]:
    """Return 4 points ordered clockwise, starting at highest-y (rightmost on tie)."""
    if len(points) != 4:
        return points
    import numpy as np

    pts_np = np.array(points, dtype=np.float32)
    centroid = pts_np.mean(axis=0)
    angles = np.arctan2(pts_np[:, 1] - centroid[1], pts_np[:, 0] - centroid[0])
    order = np.argsort(-angles)  # clockwise order
    ordered = pts_np[order]

    max_y = ordered[:, 1].max()
    start_candidates = [idx for idx, val in enumerate(ordered[:, 1]) if abs(val - max_y) <= 1e-3]
    if start_candidates:
        # choose rightmost among candidates
        start_idx = max(start_candidates, key=lambda idx: ordered[idx, 0])
        ordered = np.concatenate([ordered[start_idx:], ordered[:start_idx]])

    return [tuple(map(float, pt)) for pt in ordered]


def _points_to_obbox(
    points: List[tuple],
    img_width: int,
    img_height: int,
) -> Optional[Dict[str, Any]]:
    """Convert polygon points into normalized point list and orientation angle."""
    if not points or len(points) < 4 or img_width <= 0 or img_height <= 0:
        return None
    import math

    ordered = _order_points_clockwise(points)
    normalized: List[float] = []
    for x, y in ordered[:4]:
        normalized.extend([
            float(x) / float(img_width),
            float(y) / float(img_height),
        ])

    start = ordered[0]
    end = ordered[3]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.degrees(math.atan2(dy, dx))
    while angle < -90.0:
        angle += 180.0
    while angle >= 0.0:
        angle -= 180.0

    return {
        "points": normalized,
        "orientation": float(angle),
    }


def _normalize_label(value: Optional[str]) -> str:
    """Return a normalized label for reliable dictionary lookups."""
    if value is None:
        return ""
    return value.strip().lower()


def _select_bbox_candidate(
    raw_entry: Any,
    obj: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve a lookup entry that might be a dict or list of dicts."""
    if raw_entry is None:
        return None
    if isinstance(raw_entry, dict):
        return raw_entry
    if not isinstance(raw_entry, list) or not raw_entry:
        return None

    if obj:
        category_hint = _normalize_label(
            obj.get("category")
            or obj.get("sam_prompt")
            or obj.get("label")
            or obj.get("reason")
        )
        if category_hint:
            for candidate in raw_entry:
                if _normalize_label(candidate.get("sam_prompt")) == category_hint:
                    return candidate
            for candidate in raw_entry:
                cand_label = _normalize_label(candidate.get("sam_prompt"))
                if cand_label and (
                    cand_label in category_hint or category_hint in cand_label
                ):
                    return candidate

    return raw_entry[0]


def _build_objects_from_manifest() -> List[Dict[str, Any]]:
    """Read crops_manifest.json and build a list of object dicts from all crops."""
    manifest_path = CROPS_MANIFEST_PATH
    if not manifest_path or not os.path.exists(manifest_path):
        return []
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest_data = json.load(fh)
    except Exception:
        return []
    
    objects_list = []
    if isinstance(manifest_data, dict):
        for prompt_info in manifest_data.get("prompts", []):
            sam_prompt = prompt_info.get("prompt", "object")
            for crop in prompt_info.get("crops", []):
                if not isinstance(crop, dict):
                    continue
                objects_list.append({
                    "index": crop.get("index"),
                    "mask_path": crop.get("mask_path", ""),
                    "category": sam_prompt,
                    "reason": sam_prompt,
                    "bbox_pixels": crop.get("bbox_pixels"),
                    "properties": crop.get("properties", {}),
                    "sam_prompt": crop.get("sam_prompt", sam_prompt),
                })
    return objects_list


def _parse_filtering_response(raw_answer: str, base_image_path: str) -> Dict[str, Any]:
    """Parse VQA filtering response, draw bounding boxes on base image with legend.
    
    Returns a dict with:
    - answer: clean final answer (the JSON array as string)
    - thinking: the thinking/reasoning part
    - image_base64: base image with bounding boxes and legend overlaid
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    
    result = {
        "answer": raw_answer,
        "thinking": None,
        "image_base64": None,
        "objects": [],
        "obboxes": [],
    }
    
    # Extract thinking section
    thinking_match = re.search(r'\[THINKING\]\s*(.*?)\s*\[FINAL ANSWER\]', raw_answer, re.DOTALL | re.IGNORECASE)
    if thinking_match:
        result["thinking"] = thinking_match.group(1).strip()
    
    # Check if response is mask paths (filtering mode returns newline-separated paths)
    is_mask_paths = False
    if raw_answer and not raw_answer.strip().startswith('['):
        # Check if it looks like file paths
        lines = [l.strip() for l in raw_answer.strip().split('\n') if l.strip()]
        if lines and all(l.endswith('.png') or l.endswith('.jpg') for l in lines):
            is_mask_paths = True
    
    objects_list = []
    
    if is_mask_paths:
        # Build objects from crops_manifest.json directly
        objects_list = _build_objects_from_manifest()
        if objects_list:
            result["answer"] = json.dumps([{"index": o.get("index"), "category": o.get("category")} for o in objects_list])
    else:
        # Try to extract [FINAL ANSWER] JSON array
        final_answer_match = re.search(r'\[FINAL ANSWER\]\s*(\[.*\])', raw_answer, re.DOTALL)
        if not final_answer_match:
            # Try without the tag - just find a JSON array
            final_answer_match = re.search(r'(\[\s*\{.*?\}\s*\])', raw_answer, re.DOTALL)
        
        if not final_answer_match:
            return result
        
        json_str = final_answer_match.group(1)
        result["answer"] = json_str.strip()
        
        try:
            objects_list = json.loads(json_str)
        except json.JSONDecodeError:
            try:
                fixed_json = re.sub(r',\s*]', ']', json_str)
                fixed_json = re.sub(r',\s*}', '}', fixed_json)
                objects_list = json.loads(fixed_json)
            except:
                return result
    
    if not objects_list:
        return result
    
    # Load base image
    try:
        base_img_bgr = cv2.imread(base_image_path)
        if base_img_bgr is None:
            return result
        base_height, base_width = base_img_bgr.shape[:2]
    except Exception:
        return result
    
    # Color palette for different categories (RGB)
    colors = [
        (255, 0, 0),      # red
        (0, 255, 0),      # green
        (0, 0, 255),      # blue
        (255, 255, 0),    # yellow
        (255, 0, 255),    # magenta
        (0, 255, 255),    # cyan
        (255, 128, 0),    # orange
        (128, 0, 255),    # purple
    ]
    
    bbox_lookup = _load_bbox_lookup(base_image_path)
    # Group objects by category (prefer SAM prompt) and track their polygons
    category_objects: Dict[str, Dict[str, Any]] = {}
    objects_response: List[Dict[str, Any]] = []
    
    for obj in objects_list:
        if not isinstance(obj, dict):
            continue
        
        index = obj.get("index")
        mask_path = obj.get("mask_path", "")
        reason = obj.get("reason", "unknown")
        
        # For manifest-sourced objects, use embedded data directly
        bbox_pixels_direct = obj.get("bbox_pixels")
        properties_direct = obj.get("properties", {})
        
        mask_filename = os.path.basename(mask_path) if mask_path else f"mask_{index}.png"
        mask_filename = (mask_filename or "").lower()
        bbox_info = _select_bbox_candidate(bbox_lookup.get(mask_filename), obj)
        if bbox_info is None and index is not None:
            fallback_key = f"mask_{index}.png".lower()
            bbox_info = _select_bbox_candidate(bbox_lookup.get(fallback_key), obj)
        
        # Use direct bbox_pixels if bbox_info not found
        if bbox_info is None and bbox_pixels_direct:
            bbox_info = {
                "bbox": bbox_pixels_direct,
                "orientation": properties_direct.get("orientation", 0.0),
                "sam_prompt": obj.get("sam_prompt", ""),
                "mask_path": mask_path,
            }
        
        if bbox_info is None:
            continue
        bbox_pixels = bbox_info.get("bbox")
        if not bbox_pixels:
            continue
        mask_path_candidate = (
            bbox_info.get("mask_path")
            or obj.get("mask_path")
            or mask_path
        )
        points = _compute_mask_box_points(
            mask_path_candidate,
            base_width,
            base_height,
        )
        if not points:
            points = _compute_oriented_box_points(
                bbox_pixels,
                bbox_info.get("orientation", 0.0),
                base_width,
                base_height,
            )
        if not points:
            continue
        obbox_payload = _points_to_obbox(points, base_width, base_height)
        if not obbox_payload:
            continue
        category_label = (
            obj.get("category")
            or bbox_info.get("sam_prompt")
            or reason
            or "unknown"
        )
        category_key = _normalize_label(category_label) or "unknown"
        object_identifier = (
            obj.get("object_id")
            or obj.get("id")
            or obj.get("index")
            or mask_filename
            or len(objects_response)
        )
        object_payload = {
            "object_id": str(object_identifier),
            "category": category_label,
            "obbox": obbox_payload.get("points", []),
            "orientation": obbox_payload.get("orientation"),
            "mask": mask_filename,
            "mask_path": mask_path_candidate,
        }
        objects_response.append(object_payload)
        if category_key not in category_objects:
            category_objects[category_key] = {
                "label": category_label,
                "items": [],
            }
        category_objects[category_key]["items"].append({
            "index": index,
            "points": points,
            "obbox": object_payload["obbox"],
            "object_id": object_payload["object_id"],
        })
    
    # Assign colors to categories that have at least one valid mask
    category_colors = {}
    color_idx = 0
    categories_with_masks = []
    
    for category, data in category_objects.items():
        items = data.get("items", [])
        has_valid = any(o.get("points") for o in items)
        if has_valid:
            category_colors[category] = colors[color_idx % len(colors)]
            categories_with_masks.append(category)
            color_idx += 1
    
    # Image buffer for drawing via OpenCV
    annotated_img = base_img_bgr.copy()
    
    # Collect legend font up front (used later once we convert to PIL)
    try:
        legend_font = ImageFont.truetype("arial.ttf", 14)
    except:
        legend_font = ImageFont.load_default()
    
    for category, data in category_objects.items():
        if category not in category_colors:
            continue
        
        color = category_colors[category]
        objs = data.get("items", [])
        for obj in objs:
            points = obj.get("points")
            if not points:
                continue
            pts_np = np.array(points, dtype=np.int32)
            color_bgr = (color[2], color[1], color[0])
            cv2.polylines(annotated_img, [pts_np], True, (0, 0, 0), thickness=7)
            cv2.polylines(annotated_img, [pts_np], True, color_bgr, thickness=4)

    # Convert back to PIL for legend drawing and encoding
    pil_img = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    # Draw legend in bottom right corner
    if categories_with_masks:
        legend_padding = 10
        legend_item_height = 25
        legend_width = 200
        legend_height = len(categories_with_masks) * legend_item_height + legend_padding * 2
        
        img_width, img_height = pil_img.size
        legend_x = img_width - legend_width - legend_padding
        legend_y = img_height - legend_height - legend_padding
        
        # Draw legend background
        draw.rectangle(
            [legend_x, legend_y, img_width - legend_padding, img_height - legend_padding],
            fill=(255, 255, 255, 200),
            outline=(0, 0, 0),
            width=1
        )
        
        # Draw legend items
        for i, category in enumerate(categories_with_masks):
            color = category_colors[category]
            label_text = category_objects.get(category, {}).get("label", category)
            item_y = legend_y + legend_padding + i * legend_item_height
            
            # Color box
            draw.rectangle(
                [legend_x + legend_padding, item_y, legend_x + legend_padding + 20, item_y + 15],
                fill=color,
                outline=(0, 0, 0)
            )
            
            # Category label (truncate if too long)
            label = label_text[:25] + "..." if len(label_text) > 25 else label_text
            draw.text(
                (legend_x + legend_padding + 25, item_y),
                label,
                fill=(0, 0, 0),
                font=legend_font
            )
    
    # Convert to base64
    import io
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    result["image_base64"] = base64.b64encode(buffer.read()).decode("utf-8")
    result["objects"] = objects_response
    result["obboxes"] = [entry["obbox"] for entry in objects_response]
    
    return result

def _detect_image_type(image_path: str) -> Optional[str]:
    if band_classifier is None:
        return None
    try:
        value = band_classifier.classify(image_path)
        return str(value).lower() if value else None
    except Exception as exc:
        logger.debug("Image type classification failed for %s: %s", image_path, exc)
        return None


def _image_type_prefix_for(image_type: Optional[str]) -> Optional[str]:
    if image_type == "sar":
        return """The image provided to you is a SAR (Synthetic Aperture Radar) image. Assume this to be true.

SAR IMAGE CHARACTERISTICS:
- Grayscale imagery (no color information available)
- Brightness indicates radar reflectivity, not visible light
- Smooth surfaces (water, roads, paved areas) appear dark/black
- Rough/textured surfaces and metal objects appear bright/white
- Shadows indicate tall structures or terrain features
- Geometric distortions may occur due to radar viewing angle
- Speckle noise is common (grainy texture)

When analyzing SAR imagery:
- Do NOT attempt color-based descriptions or identification
- Focus on texture, brightness patterns, and geometric shapes
- Consider radar reflection properties when identifying objects
- Tall structures cast distinctive radar shadows (appear as dark areas)
- Water bodies are typically very dark due to smooth surface
- Urban areas show bright returns due to buildings and infrastructure

"""
    if image_type == "falsecolor":
        return """The image provided to you is a false-color composite created using the 
Near-Infrared (NIR), Red, and Green spectral bands. Assume this to be true.

FALSE-COLOR (NIR–R–G) IMAGE CHARACTERISTICS:
- The Red channel represents Near-Infrared reflectance (NIR)
- The Green channel represents Red reflectance (visible)
- The Blue channel represents Green reflectance (visible)
- Colors do NOT correspond to natural human vision
- Healthy vegetation appears bright red or pink due to strong NIR reflectance
- Stressed or sparse vegetation appears dull red or brownish
- Water bodies appear dark or black because they absorb NIR
- Urban areas, concrete, and bare soil appear in cyan, blue, tan, or brown tones
- Clouds, snow, and sand often appear very bright (white or light cyan)
- Different land-cover types exhibit distinct color patterns due to spectral signatures

WHEN ANALYZING NIR–R–G FALSE-COLOR IMAGERY:
- Do NOT assume natural/visible coloration
- Use red/pink tones to assess vegetation presence and health
- Use cyan/blue/grey tones to identify urban or man-made surfaces
- Use dark areas to identify water, deep shadows, or certain vegetation types
- Consider both texture and spatial patterns alongside color
- Note that variations in red intensity often correlate with vegetation density
- Spectral color differences are key to distinguishing materials and land cover


"""
    return None


def _maybe_image_type_prefix(
    image_path: str,
    *,
    return_type: bool = False,
    known_type: Optional[str] = None,
) -> Union[Optional[str], Tuple[Optional[str], Optional[str]]]:
    image_type = known_type or _detect_image_type(image_path)
    prefix = _image_type_prefix_for(image_type)
    if return_type:
        return prefix, image_type
    return prefix


def _adapter_paths_for_job(job: str, image_type: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    job_key = (job or "").strip().lower()
    img_type = (image_type or "").strip().lower() if image_type else None
    if job_key == "caption":
        if img_type == "sar":
            return CAPTION_SAR_ADAPTER_PATH, CAPTION_SAR_NULL_ADAPTER_PATH
        return CAPTION_DEFAULT_ADAPTER_PATH, CAPTION_DEFAULT_NULL_ADAPTER_PATH
    if job_key == "vqa" and img_type == "sar":
        return VQA_SAR_ADAPTER_PATH, VQA_SAR_NULL_ADAPTER_PATH
    return None, None


@contextmanager
def _adapter_session(job: str, image_type: Optional[str]):
    if orchestrator is None or orchestrator.vlm is None:
        yield
        return

    adapter_path, null_path = _adapter_paths_for_job(job, image_type)
    adapter_loaded = False
    selected_adapter = adapter_path if adapter_path and os.path.isdir(adapter_path) else None
    selected_null = null_path if null_path and os.path.isdir(null_path) else DEFAULT_NULL_ADAPTER_PATH

    if adapter_path and not selected_adapter:
        logger.warning("Adapter path for job '%s' missing: %s", job, adapter_path)

    if selected_adapter:
        try:
            load_result = orchestrator.vlm.load_lora(selected_adapter)
            adapter_loaded = load_result.get("status") in {"success", "already_loaded"}
            if not adapter_loaded:
                logger.warning(
                    "Failed to load adapter for job '%s' (%s): %s",
                    job,
                    selected_adapter,
                    load_result.get("error"),
                )
        except Exception as exc:
            logger.warning("Exception while loading adapter '%s': %s", selected_adapter, exc)

    try:
        yield
    finally:
        if adapter_loaded:
            try:
                orchestrator.vlm.unload_lora(selected_null)
            except Exception as exc:
                logger.warning(
                    "Exception while unloading adapter for job '%s' with null '%s': %s",
                    job,
                    selected_null,
                    exc,
                )


@app.post("/vqa/{mode}")
async def vqa(
    mode: str,
    image: UploadFile = File(...),
    question: str = Form(...),
    use_sam: Optional[str] = Form("true"),
):
    """Run VQA with specified mode (attribute, numerical, binary, filtering)."""
    tmp = save_temp(image)
    try:
        use_sam_flag = _parse_bool(use_sam, default=True)
        # Inject image type prefix if needed
        original_prompt = orchestrator.vlm.system_prompt
        prefix, image_type = _maybe_image_type_prefix(tmp, return_type=True)
        if prefix:
            orchestrator.vlm.system_prompt = prefix + (original_prompt or "")
        try:
            with _adapter_session("vqa", image_type):
                result = await run_in_threadpool(
                    orchestrator.run,
                    tmp,
                    question,
                    max_candidates=10,
                    use_sam3_api=use_sam_flag,
                    mode=mode,
                )
        
            # For filtering mode, parse the response and draw bboxes on image
            if mode.lower() == "filtering":
                parsed = _parse_filtering_response(result, tmp)
                return parsed
       
            return {"answer": result}
        finally:
            orchestrator.vlm.system_prompt = original_prompt
    finally:
        cleanup_temp(tmp)

@app.post("/bbox")
async def bbox(image: UploadFile = File(...), object_name: str = Form(...), image_type: str = Form(None)):
    """Detect objects and return bounding boxes via the shared Qwen VLM."""
    tmp = save_temp(image)
    try:
        original_prompt = orchestrator.vlm.system_prompt
        # Use provided image_type or always detect it (detection is more reliable)
        prefix = _maybe_image_type_prefix(tmp)
        if prefix:
            orchestrator.vlm.system_prompt = prefix + (original_prompt or "")
        try:
            result = await run_in_threadpool(
                orchestrator.detect_objects_via_sam,
                image_path=tmp,
                prompt=object_name,
            )
        finally:
            orchestrator.vlm.system_prompt = original_prompt
        # Ensure backward compatibility for clients expecting top-level bboxes list
        response_payload = dict(result or {})
        response_payload.setdefault("bboxes", response_payload.get("bboxes_normalized", []))
        response_payload.setdefault("object_name", object_name)
        return response_payload
    finally:
        cleanup_temp(tmp)

@app.post("/caption")
async def caption(image: UploadFile = File(...)):
    """Generate image caption."""
    tmp = save_temp(image)
    try:
        original_prompt = orchestrator.vlm.system_prompt
        prefix, image_type = _maybe_image_type_prefix(tmp, return_type=True)
        if prefix:
            
            orchestrator.vlm.system_prompt = prefix + (original_prompt or "")
        try:
            with _adapter_session("caption", image_type):
                result = orchestrator.vlm.caption_image(tmp)
            return {"caption": result}
        finally:
            orchestrator.vlm.system_prompt = original_prompt
    finally:
        cleanup_temp(tmp)

@app.post("/features")
async def features(
    image: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
):
    """Extract feature list from the image using the VLM."""
    tmp = save_temp(image)
    try:
        original_prompt = orchestrator.vlm.system_prompt
        prefix = _maybe_image_type_prefix(tmp)
        if prefix:
            orchestrator.vlm.system_prompt = prefix + (original_prompt or "")
        try:
            result = await run_in_threadpool(
                orchestrator.get_feature_list,
                tmp,
                prompt,
            )
            return {"features": result}
        finally:
            orchestrator.vlm.system_prompt = original_prompt
    finally:
        cleanup_temp(tmp)

@app.get("/health")
async def health(response: Response):
    """Check if the Qwen VLM model is loaded and ready."""
    if orchestrator is None:
        response.status_code = status.HTTP_410_GONE
        return {"status": "No", "message": "Orchestrator not initialized"}
    
    if orchestrator.vlm is None:
        response.status_code = status.HTTP_410_GONE
        return {"status": "No", "message": "VLM not loaded"}
    
    # Check if the model is loaded by verifying the shared model exists
    if orchestrator.vlm.model is not None:
        return {"status": "Yes", "message": "Qwen VLM is loaded and ready"}
    
    response.status_code = status.HTTP_410_GONE
    return {"status": "No", "message": "VLM model not loaded"}