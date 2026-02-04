"""Simple FastAPI endpoints for running SAM3 inference."""

import os

# Load .env file early before any env vars are read
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import base64
import shutil
allowed_hosts = os.environ.get("ALLOWED_HOSTS", "")

os.environ["NO_PROXY"] = allowed_hosts
os.environ["no_proxy"] = allowed_hosts
for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(proxy_var, None)

import io
import logging
import tempfile
from pathlib import Path
import threading
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, Response, status
from fastapi.concurrency import run_in_threadpool
from PIL import Image
import numpy as np
from level2_abstract import VisionPipeline

# Import NMS and overlap removal utilities
try:
    from sam3.perflib.nms import nms_masks
    from sam3.perflib.masks_ops import mask_iou
    NMS_AVAILABLE = True
except ImportError:
    NMS_AVAILABLE = False
    logging.warning("NMS utilities not available, masks will not be filtered")

from transformers import Sam3Processor, Sam3Model
import torch


logger = logging.getLogger(__name__)
app = FastAPI()

pipeline = VisionPipeline()


def _parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def _save_pil_image_to_temp(image: Image.Image) -> Path:
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    try:
        image.save(tmp_file, format="PNG")
    finally:
        tmp_file.close()
    return Path(tmp_file.name)


def _cleanup_temp_file(path: Path) -> None:
    try:
        path.unlink()
    except Exception as exc:
        logger.warning("failed to delete temporary file %s: %s", path, exc)


def _level2_response(result: Dict, object_name: str) -> Dict:
    return {
        "prompt": object_name,
        "detections": result.get("detections", []),
        "model1_count": result.get("model1_count", 0),
        "model2_count": result.get("model2_count", 0),
        "merged_count": result.get("merged_count", 0),
        "model1_boxes": result.get("model1_boxes", []),
        "model2_boxes": result.get("model2_boxes", []),
        "masks": result.get("masks", []),
        "image_size": result.get("image_size"),
    }


def mask_to_box(mask: np.ndarray) -> Optional[List[float]]:
    a = np.asarray(mask)
    if a.ndim > 2:
        lead_axes = tuple(range(a.ndim - 2))
        try:
            a = a.any(axis=lead_axes)
        except Exception:
            a = a.squeeze()
    if a.dtype != bool:
        a = a.astype(bool)
    ys, xs = np.where(a)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    h, w = a.shape
    return [float(x1) / w, float(y1) / h, float(x2) / w, float(y2) / h]


def fix_mask_shape(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    m = np.squeeze(m)
    if m.ndim > 2:
        m = m.reshape(m.shape[-2], m.shape[-1])
    if m.ndim != 2:
        m = m.reshape(m.shape[-2], m.shape[-1])
    return m


def encode_mask(mask: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def apply_iom_overlap_removal(mask_images: List[np.ndarray], scores: List[float], 
                               iom_threshold: float = 0.3) -> tuple:
    """
    Remove overlapping masks using IoM (Intersection over Minimum) metric.
    Greedy algorithm: sort by score descending, keep mask if IoM to all kept masks <= threshold.
    
    Args:
        mask_images: List of binary mask arrays (H, W)
        scores: List of scores for each mask
        iom_threshold: IoM threshold (default 0.3)
    
    Returns:
        (kept_mask_images, kept_scores, kept_indices)
    """
    if len(mask_images) <= 1:
        return mask_images, scores, list(range(len(mask_images)))
    
    # Sort by score descending
    order = sorted(range(len(mask_images)), key=lambda i: scores[i], reverse=True)
    
    kept_indices = []
    kept_masks = []
    
    for i in order:
        if len(kept_masks) == 0:
            kept_indices.append(i)
            kept_masks.append(mask_images[i])
            continue
        
        # Check IoM with all kept masks
        current_mask = mask_images[i] > 0
        current_area = current_mask.sum()
        
        should_keep = True
        for kept_mask in kept_masks:
            kept_bool = kept_mask > 0
            kept_area = kept_bool.sum()
            
            intersection = np.logical_and(current_mask, kept_bool).sum()
            min_area = min(current_area, kept_area)
            
            if min_area > 0:
                iom = float(intersection) / float(min_area)
                if iom > iom_threshold:
                    should_keep = False
                    break
        
        if should_keep:
            kept_indices.append(i)
            kept_masks.append(mask_images[i])
    
    # Sort indices back to original order
    kept_indices_sorted = sorted(kept_indices)
    
    return (
        [mask_images[i] for i in kept_indices_sorted],
        [scores[i] for i in kept_indices_sorted],
        kept_indices_sorted
    )


def apply_nms_filtering(mask_images: List[np.ndarray], scores: List[float],
                        prob_threshold: float = 0.7, iou_threshold: float = 0.5) -> tuple:
    """
    Apply NMS (Non-Maximum Suppression) to filter overlapping masks.
    
    Args:
        mask_images: List of binary mask arrays
        scores: List of scores for each mask
        prob_threshold: Minimum score threshold
        iou_threshold: IoU threshold for suppression
    
    Returns:
        (kept_mask_images, kept_scores, kept_indices)
    """
    if not NMS_AVAILABLE or len(mask_images) <= 1:
        return mask_images, scores, list(range(len(mask_images)))
    
    try:
        # Convert to tensors
        masks_tensor = torch.from_numpy(np.stack([m > 0 for m in mask_images])).float()
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        
        # Apply NMS
        keep_mask = nms_masks(
            pred_probs=scores_tensor,
            pred_masks=masks_tensor,
            prob_threshold=prob_threshold,
            iou_threshold=iou_threshold
        )
        
        kept_indices = torch.where(keep_mask)[0].tolist()
        
        return (
            [mask_images[i] for i in kept_indices],
            [scores[i] for i in kept_indices],
            kept_indices
        )
    except Exception as e:
        logging.warning(f"NMS failed: {e}, falling back to no filtering")
        return mask_images, scores, list(range(len(mask_images)))


def extract_masks_and_boxes(results: Dict, img_h: int, img_w: int, 
                            apply_filtering: bool = True,
                            use_nms: bool = False,
                            iom_threshold: float = 0.3,
                            nms_iou_threshold: float = 0.5) -> Dict:
    """Extract masks and boxes from HF processor results with optional filtering."""
    boxes = []
    mask_images = []
    scores = []
    
    masks = results.get("masks")
    boxes_raw = results.get("boxes")
    scores_raw = results.get("scores")
    
    if masks is not None:
        if torch is not None and isinstance(masks, torch.Tensor):
            masks_np = masks.detach().cpu().numpy()
        else:
            masks_np = np.asarray(masks)
        
        # Extract scores if available
        if scores_raw is not None:
            if torch is not None and isinstance(scores_raw, torch.Tensor):
                scores = scores_raw.detach().cpu().numpy().tolist()
            else:
                scores = list(scores_raw)
        else:
            # Default scores (equal confidence)
            scores = [1.0] * len(masks_np)
        
        for m in masks_np:
            m = fix_mask_shape(m)
            m_bool = m.astype(bool)
            mask_images.append((m_bool.astype("uint8") * 255))
            b = mask_to_box(m_bool)
            if b is not None:
                boxes.append(b)
        
        # Apply filtering if requested
        if apply_filtering and len(mask_images) > 1:
            original_count = len(mask_images)
            
            if use_nms and NMS_AVAILABLE:
                mask_images, scores, kept_indices = apply_nms_filtering(
                    mask_images, scores, 
                    prob_threshold=0.0,  # already filtered by post_process
                    iou_threshold=nms_iou_threshold
                )
                logging.info(f"NMS: {original_count} → {len(mask_images)} masks")
            else:
                mask_images, scores, kept_indices = apply_iom_overlap_removal(
                    mask_images, scores, iom_threshold=iom_threshold
                )
                logging.info(f"IoM filtering: {original_count} → {len(mask_images)} masks")
            
            # Filter boxes to match kept masks
            boxes = [boxes[i] for i in kept_indices if i < len(boxes)]
    
    # If we have boxes from model, use them (already in absolute pixel coordinates)
    if boxes_raw is not None and len(boxes) == 0:
        if torch is not None and isinstance(boxes_raw, torch.Tensor):
            boxes_np = boxes_raw.detach().cpu().numpy()
        else:
            boxes_np = np.asarray(boxes_raw)
        
        for bx in boxes_np:
            x1, y1, x2, y2 = bx.tolist()
            # Normalize to 0-1 range
            boxes.append([x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h])
    
    return {
        "boxes": boxes,
        "mask_images": mask_images,
        "original_height": img_h,
        "original_width": img_w,
        "scores": scores,
    }


class Sam3InferenceClient:
    """Wraps SAM3 model loading and inference calls using HuggingFace transformers."""

    def __init__(self):
        self._model: Optional[Sam3Model] = None
        self._processor: Optional[Sam3Processor] = None
        self._device: Optional[str] = None
        self._lock = threading.Lock()

    def _ensure_model_and_processor(self):
        """Lazy load model and processor."""
        if Sam3Model is None or Sam3Processor is None or torch is None:
            raise RuntimeError("transformers and torch are not installed or failed to import")
        
        if self._model is None or self._processor is None:
                    self._device = "cuda" if torch.cuda.is_available() else "cpu"
                    
                    # --- NEW FAST LOADING LOGIC ---
                    # 1. Define paths based on your file structure image
                    # api.py is in 'sam3/', so parent.parent is the root 'l40/' where 'models--facebook--sam3' is.
                    slow_source = Path(__file__).parent.parent / "models--facebook--sam3"
                    
                    # If the folder in your image is a raw HF cache (has 'snapshots' inside), we need to go deeper.
                    # This check tries to find the actual model files.
                    if (slow_source / "snapshots").exists():
                        # Grab the most recent snapshot
                        snapshots = sorted((slow_source / "snapshots").iterdir())
                        if snapshots:
                            slow_source = snapshots[-1]

                    fast_dest = Path("/tmp/facebook-sam3-fast")

                    # 2. Copy to local /tmp if it doesn't exist there yet
                    if not fast_dest.exists():
                        logging.info(f"Speedup: Copying model from {slow_source} to {fast_dest}...")
                        if slow_source.exists():
                            try:
                                shutil.copytree(str(slow_source), str(fast_dest))
                                logging.info("Copy complete. Loading from fast local storage.")
                            except Exception as e:
                                logging.warning(f"Copy failed ({e}), falling back to slow load.")
                        else:
                            logging.warning(f"Source {slow_source} not found. Downloading/Loading from default cache.")

                    # 3. Load from the fast path if it exists, otherwise fallback to default string
                    model_path = str(fast_dest) if fast_dest.exists() else "facebook/sam3"
                    
                    self._model = Sam3Model.from_pretrained(model_path).to(self._device)
                    self._processor = Sam3Processor.from_pretrained(model_path)
                    # ------------------------------
        
        return self._model, self._processor, self._device

    def _run_inference(self, image: Image.Image, prompt: str) -> Dict:
        """Run inference with text prompt only."""
        model, processor, device = self._ensure_model_and_processor()
        
        img = image.convert("RGB")
        img_w, img_h = img.size
        
        # Prepare inputs
        inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process results - INCREASED thresholds to reduce over-segmentation
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,  # was 0.7, still high but not too strict
            mask_threshold=0.5,  # was 0.7
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        print(results)
        
        return {
            "results": results,
            "img_h": img_h,
            "img_w": img_w,
        }

    def _run_inference_with_box(self, image: Image.Image, prompt: str, box: List[float]) -> Dict:
        """Run inference with text prompt and box prompt."""
        model, processor, device = self._ensure_model_and_processor()
        
        img = image.convert("RGB")
        img_w, img_h = img.size
        
        # Convert normalized coordinates to pixel coordinates
        x1, y1, x2, y2 = box
        box_pixels = [x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h]
        
        # Prepare inputs with box prompt
        input_boxes = [[box_pixels]]  # [batch, num_boxes, 4]
        input_boxes_labels = [[1]]  # 1 = positive box
        
        inputs = processor(
            images=img,
            text=prompt,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
            return_tensors="pt"
        ).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process results
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.70,
            mask_threshold=0.70,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        
        print(results)

        return {
            "results": results,
            "img_h": img_h,
            "img_w": img_w,
        }

    def generate(
        self, 
        image: Image.Image, 
        prompt: str,
        apply_filtering: bool = True,
        use_nms: bool = False,
        iom_threshold: float = 0.3,
        nms_iou_threshold: float = 0.5
    ) -> Dict:
        """Generate masks from text prompt with optional filtering."""
        inference_result = self._run_inference(image, prompt)
        results = inference_result["results"]
        img_h = inference_result["img_h"]
        img_w = inference_result["img_w"]
        
        result = extract_masks_and_boxes(
            results, img_h, img_w,
            apply_filtering=apply_filtering,
            use_nms=use_nms,
            iom_threshold=iom_threshold,
            nms_iou_threshold=nms_iou_threshold
        )
        result["image_size"] = {"width": img_w, "height": img_h}
        return result

    def generate_with_box(
        self, 
        image: Image.Image, 
        prompt: str, 
        box: List[float],
        apply_filtering: bool = False,
        use_nms: bool = False,
        iom_threshold: float = 0.3,
        nms_iou_threshold: float = 0.5
    ) -> Dict:
        """Generate masks from text prompt and box prompt with optional filtering."""
        inference_result = self._run_inference_with_box(image, prompt, box)
        results = inference_result["results"]
        img_h = inference_result["img_h"]
        img_w = inference_result["img_w"]
        
        result = extract_masks_and_boxes(
            results, img_h, img_w,
            apply_filtering=apply_filtering,
            use_nms=use_nms,
            iom_threshold=iom_threshold,
            nms_iou_threshold=nms_iou_threshold
        )
        result["image_size"] = {"width": img_w, "height": img_h}
        return result

    def generate_with_boxes(
        self, 
        image: Image.Image, 
        prompt: str, 
        boxes: List[List[float]],
        apply_filtering: bool = False,
        use_nms: bool = False,
        iom_threshold: float = 0.3,
        nms_iou_threshold: float = 0.5
    ) -> Dict:
        """generate masks from text prompt and multiple box prompts with optional filtering"""
        model, processor, device = self._ensure_model_and_processor()
        
        img = image.convert("RGB")
        img_w, img_h = img.size
        
        # convert all normalized coordinates to pixel coordinates
        box_pixels_list = []
        for box in boxes:
            x1, y1, x2, y2 = box
            box_pixels = [x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h]
            box_pixels_list.append(box_pixels)
        
        # prepare inputs with multiple box prompts
        input_boxes = [box_pixels_list]  # [batch=1, num_boxes, 4]
        input_boxes_labels = [[1] * len(box_pixels_list)]  # all positive boxes
        
        inputs = processor(
            images=img,
            text=prompt,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
            return_tensors="pt"
        ).to(device)
        
        # run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # post-process results
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.70,
            mask_threshold=0.70,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        
        print(results)

        result = extract_masks_and_boxes(
            results, img_h, img_w,
            apply_filtering=apply_filtering,
            use_nms=use_nms,
            iom_threshold=iom_threshold,
            nms_iou_threshold=nms_iou_threshold
        )
        result["image_size"] = {"width": img_w, "height": img_h}
        return result

inference_client = Sam3InferenceClient()


def get_inference_client() -> Sam3InferenceClient:
    return inference_client


async def _read_image(image: UploadFile) -> Image.Image:
    data = await image.read()
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Failed to read uploaded image") from exc


@app.post("/masks")
async def get_masks(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    apply_filtering: str = Form("false"),
    use_nms: str = Form("false"),
    iom_threshold: str = Form("0.3"),
    nms_iou_threshold: str = Form("0.5"),
    image_type: str = Form(None),
    client: Sam3InferenceClient = Depends(get_inference_client),
):
    pil_image = await _read_image(image)
    
    # parse boolean and float parameters
    apply_filtering_bool = apply_filtering.lower() == "true"
    use_nms_bool = use_nms.lower() == "true"
    iom_threshold_float = float(iom_threshold)
    nms_iou_threshold_float = float(nms_iou_threshold)
    
    try:
        result = await run_in_threadpool(
            client.generate, 
            pil_image, 
            prompt,
            apply_filtering_bool,
            use_nms_bool,
            iom_threshold_float,
            nms_iou_threshold_float
        )
    except RuntimeError as exc:
        logger.exception("SAM3 masks inference failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while generating masks")
        raise HTTPException(status_code=500, detail="SAM3 inference failed") from exc

    boxes = result.get("boxes", [])
    serialized_masks = []
    mask_images = result.get("mask_images", [])
    for idx, mask in enumerate(mask_images):
        serialized_masks.append(
            {
                "id": idx,
                "box": boxes[idx] if idx < len(boxes) else None,
                "png": encode_mask(mask),
            }
        )

    return {
        "prompt": prompt,
        "image_size": result.get("image_size"),
        "masks": serialized_masks,
    }


@app.post("/boxes")
async def get_boxes(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    image_type: str = Form(None),
    client: Sam3InferenceClient = Depends(get_inference_client),
):
    pil_image = await _read_image(image)
    try:
        result = await run_in_threadpool(client.generate, pil_image, prompt)
    except RuntimeError as exc:
        logger.exception("SAM3 boxes inference failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while generating boxes")
        raise HTTPException(status_code=500, detail="SAM3 inference failed") from exc

    return {
        "prompt": prompt,
        "image_size": result.get("image_size"),
        "boxes": result.get("boxes", []),
    }


@app.post("/masks_with_box")
async def get_masks_with_box(
    prompt: str = Form(...),
    box: str = Form(...),  # Expected format: "x1,y1,x2,y2" (normalized 0-1)
    image: UploadFile = File(...),
    client: Sam3InferenceClient = Depends(get_inference_client),
):
    pil_image = await _read_image(image)
    
    # Parse box coordinates
    try:
        box_coords = [float(x) for x in box.split(",")]
        if len(box_coords) != 4:
            raise ValueError("Box must have exactly 4 coordinates")
        if not all(0 <= coord <= 1 for coord in box_coords):
            raise ValueError("Box coordinates must be normalized (0-1)")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid box format: {str(exc)}") from exc
    
    try:
        result = await run_in_threadpool(client.generate_with_box, pil_image, prompt, box_coords)
    except RuntimeError as exc:
        logger.exception("SAM3 masks with box inference failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while generating masks with box")
        raise HTTPException(status_code=500, detail="SAM3 inference failed") from exc

    boxes = result.get("boxes", [])
    serialized_masks = []
    mask_images = result.get("mask_images", [])
    for idx, mask in enumerate(mask_images):
        serialized_masks.append(
            {
                "id": idx,
                "box": boxes[idx] if idx < len(boxes) else None,
                "png": encode_mask(mask),
            }
        )

    return {
        "prompt": prompt,
        "input_box": box_coords,
        "image_size": result.get("image_size"),
        "masks": serialized_masks,
    }

@app.post("/masks_with_boxes")
async def get_masks_with_boxes(
    prompt: str = Form(...),
    boxes: str = Form(...),  # format: "x1,y1,x2,y2;x1,y1,x2,y2;..." (semicolon-separated)
    image: UploadFile = File(...),
    client: Sam3InferenceClient = Depends(get_inference_client),
):
    """generate masks using text prompt and multiple box hints"""
    pil_image = await _read_image(image)
    
    # parse multiple boxes
    try:
        box_list = []
        for box_str in boxes.split(";"):
            box_coords = [float(x) for x in box_str.strip().split(",")]
            if len(box_coords) != 4:
                raise ValueError("each box must have exactly 4 coordinates")
            if not all(0 <= coord <= 1 for coord in box_coords):
                raise ValueError("box coordinates must be normalized (0-1)")
            box_list.append(box_coords)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid boxes format: {str(exc)}") from exc
    
    try:
        result = await run_in_threadpool(client.generate_with_boxes, pil_image, prompt, box_list)
    except RuntimeError as exc:
        logger.exception("SAM3 masks with boxes inference failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while generating masks with boxes")
        raise HTTPException(status_code=500, detail="SAM3 inference failed") from exc

    boxes_out = result.get("boxes", [])
    serialized_masks = []
    mask_images = result.get("mask_images", [])
    for idx, mask in enumerate(mask_images):
        serialized_masks.append(
            {
                "id": idx,
                "box": boxes_out[idx] if idx < len(boxes_out) else None,
                "png": encode_mask(mask),
            }
        )

    return {
        "prompt": prompt,
        "input_boxes": box_list,
        "image_size": result.get("image_size"),
        "masks": serialized_masks,
    }


@app.post("/merged_masks")
async def get_merged_masks(
    object_name: str = Form(...),
    image: UploadFile = File(...),
    debug_visualization: str = Form("false"),
    iom_threshold: str = Form("0.5"),
    coverage_threshold: str = Form("0.7"),
    n_batch: str = Form("64"),
    max_new_tokens: str = Form("128"),
    image_type: str = Form(None),
    client: VisionPipeline = Depends(lambda: pipeline),
):
    """Run the Level2 pipeline (Qwen + dual SAM3 + merge)."""
    pil_image = await _read_image(image)
    temp_path = _save_pil_image_to_temp(pil_image)
    debug_flag = _parse_bool(debug_visualization)
    try:
        result = await run_in_threadpool(
            client.detect_and_segment_dual,
            str(temp_path),
            object_name,
            float(iom_threshold),
            float(coverage_threshold),
            debug_flag,
            int(n_batch),
            int(max_new_tokens),
            image_type,
        )
    except Exception as exc:
        logger.exception("Level2 pipeline failed")
        raise HTTPException(status_code=500, detail=f"Level2 pipeline error: {exc}") from exc
    finally:
        _cleanup_temp_file(temp_path)

    return _level2_response(result, object_name)


@app.get("/health")
async def health(response: Response):
    """Check if SAM3 model is loaded and ready."""
    client = get_inference_client()
    
    # Check if model and processor are loaded
    if client._model is not None and client._processor is not None:
        return {"status": "Yes", "message": "SAM3 model is loaded and ready"}
    
    response.status_code = status.HTTP_410_GONE
    return {"status": "No", "message": "SAM3 model not yet loaded"}