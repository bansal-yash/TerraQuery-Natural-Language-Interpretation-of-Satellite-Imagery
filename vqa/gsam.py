"""Grounding + SAM helper utilities.

This module wraps the common logic used in the demo scripts to:
- load GroundingDINO model
- load SAM and build a SamPredictor
- run grounding with a short class prompt
- apply NMS and convert outputs into a simple Python list
- run SAM to obtain per-box masks and cropped images

The functions are intentionally small and importable from the
orchestrator.
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
import torch

try:
    import torchvision
except Exception:
    torchvision = None

try:
    from groundingdino.util.inference import Model as GroundingModel
except ModuleNotFoundError:
    # If GroundingDINO is present in the repository sibling folder (project root),
    # add the repository root to sys.path so the local copy can be imported when
    # running scripts from inside the `vqa/` directory.
    import sys
    import os

    repo_root = os.path.dirname(os.path.dirname(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from groundingdino.util.inference import Model as GroundingModel
from segment_anything import sam_model_registry, SamPredictor


def load_grounding_model(config_path: str, checkpoint_path: str) -> GroundingModel:
    """Load a GroundingDINO model from config & checkpoint path."""
    model = GroundingModel(model_config_path=config_path, model_checkpoint_path=checkpoint_path)
    return model


def load_sam_predictor(sam_encoder: str, sam_checkpoint: str, device: Optional[torch.device] = None) -> SamPredictor:
    """Create a SAM predictor instance ready for segmentation."""
    sam = sam_model_registry[sam_encoder](checkpoint=sam_checkpoint)
    if device is not None:
        try:
            sam.to(device=device)
        except Exception:
            # if device move fails, continue — SamPredictor will handle CPU fallback
            pass
    predictor = SamPredictor(sam)
    return predictor


def predict_with_classes(
    grounding_model: GroundingModel,
    image: np.ndarray,
    classes: List[str],
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
) -> Dict[str, Any]:
    """Run grounding on an image with a list of class phrases.

    Returns a dict with keys: 'xyxy' (N,4 numpy), 'scores' (N,), 'phrases' (N,)
    """
    detections = grounding_model.predict_with_classes(
        image=image,
        classes=classes,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    # convert to numpy arrays for downstream usage
    xyxy = np.array(detections.xyxy)
    scores = np.array(detections.confidence)
    # detections.class_id may be present but we will map with provided classes
    try:
        phrases = [classes[int(cid)] for cid in detections.class_id]
    except Exception:
        # fallback: use class index order
        phrases = classes[: xyxy.shape[0]]

    # apply a lightweight NMS if torchvision available
    if torchvision is not None and xyxy.size != 0:
        try:
            boxes = torch.from_numpy(xyxy).float()
            scores_t = torch.from_numpy(scores).float()
            keep = torchvision.ops.nms(boxes, scores_t, iou_threshold=0.5).numpy().tolist()
            xyxy = xyxy[keep]
            scores = scores[keep]
            phrases = [phrases[i] for i in keep]
        except Exception:
            pass

    return {"xyxy": xyxy, "scores": scores, "phrases": phrases}


def segment_with_sam(predictor: SamPredictor, image_rgb: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """Given a SamPredictor and boxes (N,4) in xyxy pixel coordinates, return masks array (N, H, W boolean).

    The predictor will be given the original RGB image and will predict
    best-of-multi masks for each box.
    """
    predictor.set_image(image_rgb)
    result_masks = []
    for box in xyxy:
        try:
            masks, scores, logits = predictor.predict(box=box, multimask_output=True)
            index = int(np.argmax(scores))
            result_masks.append(masks[index])
        except Exception:
            # if predict fails for a box, append an empty mask
            h, w = image_rgb.shape[:2]
            result_masks.append(np.zeros((h, w), dtype=bool))
    if len(result_masks) == 0:
        return np.zeros((0, image_rgb.shape[0], image_rgb.shape[1]), dtype=bool)
    return np.stack(result_masks, axis=0)


def crop_by_mask(image_bgr: np.ndarray, mask: np.ndarray, padding: int = 8) -> np.ndarray:
    """Return a tight crop (BGR numpy image) around the mask with optional padding.

    If mask is empty, returns an empty array.
    """
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    y0, y1 = max(int(ys.min()) - padding, 0), min(int(ys.max()) + padding, image_bgr.shape[0] - 1)
    x0, x1 = max(int(xs.min()) - padding, 0), min(int(xs.max()) + padding, image_bgr.shape[1] - 1)
    crop = image_bgr[y0 : y1 + 1, x0 : x1 + 1].copy()
    return crop


def detections_to_list(d: Dict[str, Any], image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """Convert the grounding output dict to a list of dicts with masks/crops placeholders.

    The masks and crops are left empty; caller can call segment_with_sam and fill them.
    """
    out = []
    xyxy = d.get("xyxy", np.zeros((0, 4)))
    scores = d.get("scores", np.zeros((xyxy.shape[0],)))
    phrases = d.get("phrases", [""] * xyxy.shape[0])
    for i in range(xyxy.shape[0]):
        out.append({
            "box": xyxy[i].tolist(),
            "score": float(scores[i]) if i < len(scores) else 0.0,
            "phrase": phrases[i] if i < len(phrases) else "",
            "mask": None,
            "crop": None,
        })
    return out


if __name__ == "__main__":
    """
    Example usage demonstrating the complete pipeline:
    1. Load GroundingDINO and SAM models
    2. Run grounding detection on an image
    3. Segment detected objects with SAM
    4. Extract crops and display all outputs
    
    Usage:
        python gsam.py --image path/to/image.jpg --classes "red bus|yellow bus|car"
    """
    import argparse
    import os
    
    parser = argparse.ArgumentParser("GSAM Example - Grounding + Segmentation Pipeline")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--classes", required=True, help="Pipe-separated class phrases (e.g. 'red bus|yellow bus')")
    parser.add_argument("--grounding_config", default="../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--grounding_checkpoint", default="../groundingdino_swint_ogc.pth")
    parser.add_argument("--sam_encoder", default="vit_h")
    parser.add_argument("--sam_checkpoint", default="../sam_vit_h_4b8939.pth")
    parser.add_argument("--box_threshold", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="Text similarity threshold")
    parser.add_argument("--output_dir", default="./gsam_outputs", help="Directory to save outputs")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("GSAM EXAMPLE - Grounding + Segmentation Pipeline")
    print("="*80)
    print(f"Image: {args.image}")
    print(f"Classes: {args.classes}")
    print(f"Output Directory: {args.output_dir}")
    print("="*80 + "\n")
    
    # Step 1: Load image
    print("[1/5] Loading image...")
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f"❌ ERROR: Could not load image from {args.image}")
        exit(1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"✓ Image loaded: {img_bgr.shape[1]}x{img_bgr.shape[0]} pixels\n")
    
    # Step 2: Load models
    print("[2/5] Loading GroundingDINO model...")
    grounding_model = load_grounding_model(args.grounding_config, args.grounding_checkpoint)
    print("✓ GroundingDINO loaded\n")
    
    print("[3/5] Loading SAM model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_predictor = load_sam_predictor(args.sam_encoder, args.sam_checkpoint, device=device)
    print(f"✓ SAM loaded on {device}\n")
    
    # Step 3: Run grounding detection
    print("[4/5] Running grounding detection...")
    classes_list = [c.strip() for c in args.classes.split("|")]
    print(f"Target classes: {classes_list}")
    
    detections = predict_with_classes(
        grounding_model=grounding_model,
        image=img_rgb,
        classes=classes_list,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    
    print(f"✓ Found {len(detections['xyxy'])} detections after NMS\n")
    
    # Convert to list format
    det_list = detections_to_list(detections, img_bgr)
    
    # Step 4: Segment with SAM
    print("[5/5] Segmenting with SAM...")
    if len(detections['xyxy']) > 0:
        masks = segment_with_sam(sam_predictor, img_rgb, detections['xyxy'])
        print(f"✓ Generated {len(masks)} segmentation masks\n")
        
        # Add masks and crops to detection list
        for i, det in enumerate(det_list):
            det['mask'] = masks[i]
            det['crop'] = crop_by_mask(img_bgr, masks[i], padding=10)
    else:
        print("⚠ No detections to segment\n")
    
    # Display all outputs
    print("="*80)
    print("DETECTION RESULTS")
    print("="*80)
    
    for i, det in enumerate(det_list):
        print(f"\nDetection #{i+1}:")
        print(f"  Phrase: {det['phrase']}")
        print(f"  Confidence: {det['score']:.3f}")
        print(f"  Bounding Box: {det['box']}")
        
        if det['mask'] is not None:
            mask_area = np.sum(det['mask'])
            mask_shape = det['mask'].shape
            print(f"  Mask Shape: {mask_shape}")
            print(f"  Mask Area: {mask_area} pixels")
            
            if det['crop'] is not None:
                crop_shape = det['crop'].shape
                print(f"  Crop Shape: {crop_shape[1]}x{crop_shape[0]} (WxH)")
                
                # Save crop
                crop_path = os.path.join(args.output_dir, f"crop_{i+1}_{det['phrase'].replace(' ', '_')}.png")
                cv2.imwrite(crop_path, det['crop'])
                print(f"  Saved crop: {crop_path}")
                
                # Save mask visualization
                mask_vis = np.zeros_like(img_bgr)
                mask_vis[det['mask']] = [0, 255, 0]  # Green mask
                mask_overlay = cv2.addWeighted(img_bgr, 0.7, mask_vis, 0.3, 0)
                mask_path = os.path.join(args.output_dir, f"mask_{i+1}_{det['phrase'].replace(' ', '_')}.png")
                cv2.imwrite(mask_path, mask_overlay)
                print(f"  Saved mask: {mask_path}")
    
    # Save annotated full image
    print("\n" + "="*80)
    print("SAVING ANNOTATED IMAGE")
    print("="*80)
    
    annotated = img_bgr.copy()
    for i, det in enumerate(det_list):
        # Draw bounding box
        box = det['box']
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{det['phrase']} {det['score']:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Overlay mask with transparency
        if det['mask'] is not None:
            # Create colored mask (different color for each detection)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            color = colors[i % len(colors)]
            mask_colored = np.zeros_like(img_bgr)
            mask_colored[det['mask']] = color
            annotated = cv2.addWeighted(annotated, 0.7, mask_colored, 0.3, 0)
    
    annotated_path = os.path.join(args.output_dir, "annotated.png")
    cv2.imwrite(annotated_path, annotated)
    print(f"✓ Saved annotated image: {annotated_path}\n")
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total Detections: {len(det_list)}")
    print(f"Classes Found: {set(d['phrase'] for d in det_list)}")
    print(f"All outputs saved to: {args.output_dir}")
    print("="*80 + "\n")
    
