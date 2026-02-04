"""VQA Orchestration using Local LLM + GroundingDINO + SAM + VLM.

Architecture:
1. Local small LLM extracts object classes from user query
2. GroundingDINO detects objects based on extracted classes
3. SAM segments each detection and computes geometric properties
4. VLM analyzes segments and answers question with reasoning
5. Returns final answer with thinking process and most relevant image

Usage (programmatic):
  from vqa.orchestrator import Orchestrator
  orch = Orchestrator(...)
  answer = orch.run(image_path, question)

There's also a minimal CLI at the bottom.
"""

import os

# Load .env file early before any module reads env vars
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on system env vars

import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Set

import numpy as np
from PIL import Image
import cv2
from time import perf_counter

#from local_llm import LocalLLM
from local_vlm import LocalVLM
#import gsam
import geometric_utils
import sam3_api
import mask_merging as mm

# Import image type classifier
try:
    from image_type_classifier import BandClassifier, find_checkpoint_candidate
    BAND_CLASSIFIER_AVAILABLE = True
except ImportError:
    BAND_CLASSIFIER_AVAILABLE = False

DEFAULT_NULL_ADAPTER_PATH = os.environ.get("NULL_ADAPTER_PATH") or os.path.join(
    "artifacts",
    "sar_bbox_null_adapter",
)

SYSTEM_PROMPT_FILES = {
    "attribute": "systemprompt_attribute.txt",
    "numerical": "systemprompt_numerical.txt",
    "binary": "systemprompt_binary.txt",
    "filtering": "systemprompt_filtering.txt",
}

BBOX_SYSTEM_PROMPT = (
    "You are Qwen3-VL, an exact object localization assistant. "
    "When the user asks for bounding boxes you MUST respond strictly with the requested format. "
    "Never include explanations, only the annotations."
)

BBOX_USER_PROMPT_TEMPLATE = """Describe the locations of all visible instances of {object_name} using bounding box coordinates (x1,y1,x2,y2) normalized to 1000.
YOU MUST IDENTIFY THE OBJECTS AND ONLY THEM. DO NOT MENTION ANY OTHER OBJECTS.

MISSION CRITICAL FORMATTING INSTRUCTIONS:
Your response MUST follow this exact format for each object:
<ref>label</ref><box>(x1,y1),(x2,y2)</box>

Where:
- label: descriptive name for the object
- x1,y1,x2,y2: integer coordinates in range [0,1000]
- x1,y1 is top-left corner
- x2,y2 is bottom-right corner

Example: <ref>yellow bus</ref><box>(120,340),(450,680)</box>

Output ONLY the bounding box annotations, one per line. No explanations."""

_BOX_PATTERN = re.compile(
    r"<ref>\s*(?P<label>.+?)\s*</ref>\s*<box>\s*\(\s*(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*(?P<y1>-?\d+(?:\.\d+)?)\s*\)\s*,\s*\(\s*(?P<x2>-?\d+(?:\.\d+)?)\s*,\s*(?P<y2>-?\d+(?:\.\d+)?)\s*\)\s*(?:</box>|<\|box_end\|>)",
    flags=re.IGNORECASE | re.DOTALL,
)

_BOX_JSON_PATTERN = re.compile(
    (
        r'"label"\s*:\s*"(?P<label>[^\"]+)"[^{}]*?'
        r'"bbox_2d"\s*:\s*\[\s*(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*(?P<y1>-?\d+(?:\.\d+)?)\s*,\s*(?P<x2>-?\d+(?:\.\d+)?)\s*,\s*(?P<y2>-?\d+(?:\.\d+)?)\s*\]'
        r'|"bbox_2d"\s*:\s*\[\s*(?P<x1_alt>-?\d+(?:\.\d+)?)\s*,\s*(?P<y1_alt>-?\d+(?:\.\d+)?)\s*,\s*(?P<x2_alt>-?\d+(?:\.\d+)?)\s*,\s*(?P<y2_alt>-?\d+(?:\.\d+)?)\s*\]'
        r'[^{}]*?"label"\s*:\s*"(?P<label_alt>[^\"]+)"'
    ),
    flags=re.IGNORECASE,
)


def _coerce_coord(value: str) -> int:
    number = int(round(float(value)))
    return max(0, min(1000, number))


def _parse_bbox_text(reply: str) -> List[Dict[str, Any]]:
    boxes: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, int, int, int, int]] = set()

    def _append_box(label: str, x1: str, y1: str, x2: str, y2: str) -> None:
        if not all((label, x1, y1, x2, y2)):
            return
        label_clean = str(label).strip() or "object"
        try:
            x1_i = _coerce_coord(x1)
            y1_i = _coerce_coord(y1)
            x2_i = _coerce_coord(x2)
            y2_i = _coerce_coord(y2)
        except (TypeError, ValueError):
            return
        if x1_i >= x2_i or y1_i >= y2_i:
            return
        key = (label_clean, x1_i, y1_i, x2_i, y2_i)
        if key in seen:
            return
        seen.add(key)
        boxes.append({
            "label": label_clean,
            "x1": x1_i,
            "y1": y1_i,
            "x2": x2_i,
            "y2": y2_i,
        })

    for match in _BOX_PATTERN.finditer(reply or ""):
        _append_box(
            match.group("label"),
            match.group("x1"),
            match.group("y1"),
            match.group("x2"),
            match.group("y2"),
        )

    for match in _BOX_JSON_PATTERN.finditer(reply or ""):
        label = match.group("label") or match.group("label_alt")
        x1 = match.group("x1") or match.group("x1_alt")
        y1 = match.group("y1") or match.group("y1_alt")
        x2 = match.group("x2") or match.group("x2_alt")
        y2 = match.group("y2") or match.group("y2_alt")
        _append_box(label, x1, y1, x2, y2)

    return boxes


def _boxes_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return not (
        a["x2"] <= b["x1"]
        or b["x2"] <= a["x1"]
        or a["y2"] <= b["y1"]
        or b["y2"] <= a["y1"]
    )


def _merge_boxes(boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(boxes) < 2:
        return boxes

    parent = list(range(len(boxes)))

    def find(idx: int) -> int:
        if parent[idx] != idx:
            parent[idx] = find(parent[idx])
        return parent[idx]

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _boxes_overlap(boxes[i], boxes[j]):
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[root_j] = root_i

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for idx, box in enumerate(boxes):
        root = find(idx)
        grouped.setdefault(root, []).append(box)

    merged: List[Dict[str, Any]] = []
    for group in grouped.values():
        merged.append({
            "label": ", ".join(sorted({str(b.get("label", "object")) for b in group})),
            "x1": min(b["x1"] for b in group),
            "y1": min(b["y1"] for b in group),
            "x2": max(b["x2"] for b in group),
            "y2": max(b["y2"] for b in group),
        })

    return merged


class Orchestrator:
    def __init__(
        self,
        device: Optional[str] = None,
        sam3_base_url: Optional[str] = None,
        sam3_timeout: int = sam3_api.REQUEST_TIMEOUT,
        mode: str = "attribute",
        system_prompt_files: Optional[Dict[str, str]] = None,
    ):

        self.device = device
        self.mode = mode.lower()
        self.system_prompt_files = (system_prompt_files or {}).copy()
        for key, path in SYSTEM_PROMPT_FILES.items():
            self.system_prompt_files.setdefault(key, path)

        # load models lazily
        self._grounding = None
        self._sam_predictor = None
        self.vlm = None
        self.llm = None
        self.system_prompt = None
        self._system_prompt_cache: Dict[str, str] = {}
        self.null_adapter_path = DEFAULT_NULL_ADAPTER_PATH
        
        # SAM3 API client - base_url=None will read SAM_URL env var at runtime
        self.sam3_client = sam3_api.Sam3ApiClient(
            base_url=sam3_base_url,  # None means read from SAM_URL env var
            timeout=sam3_timeout
        )
        
        # Band classifier for image type detection
        self.band_classifier = None
        if BAND_CLASSIFIER_AVAILABLE:
            try:
                ckpt = os.environ.get("BAND_CLASSIFIER_CHECKPOINT") or find_checkpoint_candidate()
                if ckpt:
                    self.band_classifier = BandClassifier(ckpt)
            except Exception:
                pass

    def _get_system_prompt(self) -> str:
        mode = self.mode
        if mode not in self.system_prompt_files:
            raise ValueError(f"Unsupported mode '{mode}'. Expected one of {list(self.system_prompt_files.keys())}.")

        if mode not in self._system_prompt_cache:
            prompt_path = self.system_prompt_files[mode]
            # try:
            with open(prompt_path, 'r') as f:
                prompt = f.read().strip()
            print(f"[Orchestrator] Loaded system prompt for mode '{mode}' from {prompt_path}")
            # except Exception as e:
            #     print(f"[WARNING] Failed to load system prompt from {prompt_path}: {e}")
            #     prompt = "You are a helpful visual assistant."
            self._system_prompt_cache[mode] = prompt
        return self._system_prompt_cache[mode]

    def _ensure_models(self):
        #if self._grounding is None:
            #self._grounding = gsam.load_grounding_model(self.grounding_config, self.grounding_checkpoint)
        #if self._sam_predictor is None:
            #self._sam_predictor = gsam.load_sam_predictor(self.sam_encoder, self.sam_checkpoint)
        
        # Load system prompt first
        self.system_prompt = self._get_system_prompt()
        
        # Load VLM with system prompt
        if self.vlm is None:
            try:
                self.vlm = LocalVLM(
                    device=self.device,
                    system_prompt=self.system_prompt,
                    stream_thoughts=True,
                    base_adapter_path=self.null_adapter_path,
                    null_adapter_path=self.null_adapter_path,
                )
                print("vlm loaded")
            except Exception as e:
                print(f"[WARNING] Failed to load VLM: {e}")
                self.vlm = None
        else:
            # Update prompt if mode changed between runs
            self.vlm.system_prompt = self.system_prompt

    def detect_objects_via_sam(
        self,
        image_path: str,
        prompt: str,
        max_candidates: int = 10,
        n_batch: int = 64,
        max_new_tokens: int = 192,
        combine_boxes: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Detect bounding boxes using the already loaded Qwen VLM (no extra model load)."""

        path_obj = Path(image_path).expanduser().resolve()
        if not path_obj.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        self._ensure_models()
        if self.vlm is None:
            raise RuntimeError("VLM unavailable for bbox detection")

        with Image.open(path_obj) as pil_img:
            width, height = pil_img.size

        prev_prompt = self.vlm.system_prompt
        prev_stream = self.vlm.stream_thoughts
        prev_suppress = getattr(self.vlm, "_suppress_debug_prints", False)

        try:
            self.vlm.system_prompt = BBOX_SYSTEM_PROMPT
            self.vlm.stream_thoughts = False
            self.vlm._suppress_debug_prints = True

            user_prompt = BBOX_USER_PROMPT_TEMPLATE.format(object_name=prompt)
            raw_response = self.vlm.answer_question(
                [str(path_obj)],
                user_prompt,
                max_length=max_new_tokens,
            )
        finally:
            self.vlm.system_prompt = prev_prompt
            self.vlm.stream_thoughts = prev_stream
            self.vlm._suppress_debug_prints = prev_suppress

        boxes = _parse_bbox_text(raw_response)
        if combine_boxes is None:
            combine_boxes = bool(int(os.environ.get("QWEN_COMBINE_BOXES", "0")))
        if combine_boxes:
            boxes = _merge_boxes(boxes)

        boxes = boxes[:max_candidates]

        def to_normalized(box: Dict[str, Any]) -> List[float]:
            return [
                box["x1"] / 1000.0,
                box["y1"] / 1000.0,
                box["x2"] / 1000.0,
                box["y2"] / 1000.0,
            ]

        def _clamp(value: int, limit: int) -> int:
            return max(0, min(value, max(limit - 1, 0)))

        def to_pixels(box: Dict[str, Any]) -> List[int]:
            return [
                _clamp(int(round(box["x1"] * width / 1000.0)), width),
                _clamp(int(round(box["y1"] * height / 1000.0)), height),
                _clamp(int(round(box["x2"] * width / 1000.0)), width),
                _clamp(int(round(box["y2"] * height / 1000.0)), height),
            ]

        items = []
        for idx, box in enumerate(boxes):
            items.append({
                "index": idx,
                "label": box.get("label", prompt),
                "bbox_normalized": to_normalized(box),
                "bbox_pixels": to_pixels(box),
            })

        result = {
            "raw": (raw_response or "").strip(),
            "boxes": boxes,
            "items": items,
            "image_size": {"width": width, "height": height},
            "object_name": prompt,
        }
        result["bboxes"] = [item["bbox_pixels"] for item in items]
        result["bboxes_normalized"] = [item["bbox_normalized"] for item in items]
        return result
        

    def run(
        self,
        image_path: str,
        question: str,
        max_candidates: int = 10,
        use_sam3_api: bool = True,
        mode: Optional[str] = None,
    ) -> str:

        if mode:
            requested_mode = mode.lower()
            if requested_mode not in self.system_prompt_files:
                return f"Invalid mode '{mode}'. Expected one of {list(self.system_prompt_files.keys())}.\n"
            if requested_mode != self.mode:
                self.mode = requested_mode
                self.system_prompt = None  # trigger reload on next ensure

        # For filtering mode, skip VLM/system prompt loading since we return mask paths directly
        if self.mode != "filtering":
            self._ensure_models()

        # ensure output dir
        os.makedirs("vqa_outputs_fresh", exist_ok=True)

        # read image
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        H, W = img_bgr.shape[:2]

        # Auto-generate SAM prompts from detected features for every VQA mode that benefits from localization
        sam_prompts = None  # Always auto-generated from get_feature_list
        auto_prompt_modes = {"filtering", "attribute", "numerical", "binary"}
        if self.mode in auto_prompt_modes:
            auto_prompts = None
            try:
                features_output = self.get_feature_list(image_path, custom_prompt=question)
                if isinstance(features_output, str):
                    normalized = re.sub(r"\s*,\s*", "|", features_output.strip())
                    normalized = re.sub(r"\|+", "|", normalized).strip("| ")
                    if normalized:
                        # Deduplicate prompts while preserving order
                        prompts_list = normalized.split("|")
                        seen = set()
                        unique_prompts = []
                        for p in prompts_list:
                            p_clean = p.strip().lower()
                            if p_clean and p_clean not in seen:
                                seen.add(p_clean)
                                unique_prompts.append(p_clean)
                        if unique_prompts:
                            auto_prompts = "|".join(unique_prompts)
            except Exception as feature_err:
                print(f"[WARNING] Failed to derive SAM prompts from features: {feature_err}")
            if auto_prompts:
                sam_prompts = auto_prompts
                print(f"[Orchestrator] Auto-generated SAM prompts: {sam_prompts}")
        # Classify image type once for entire request
        image_type = None
        if self.band_classifier:
            try:
                image_type = self.band_classifier.classify(image_path)
                if image_type and image_type.lower() in {"sar", "falsecolor"}:
                    print(f"[Orchestrator] Detected image type: {image_type}")
            except Exception as e:
                print(f"[WARNING] Image classification failed: {e}")

        # Get masks from SAM3 API
        bounding_boxes = None
        masks_data = None
        mask_items = []  # Initialize at top level to avoid scoping issues
        using_existing_data = False  # Flag to track if we're using existing crops/masks
        
        if use_sam3_api:
            try:
                from pathlib import Path
                
                # Clean up old artifacts BEFORE creating new ones
                print("[Orchestrator] Cleaning up old artifacts before SAM processing...")
                try:
                    import glob
                    for old in glob.glob(os.path.join("vqa_outputs_fresh", "mask_*.png")):
                        try:
                            os.remove(old)
                        except Exception:
                            pass
                    for old in glob.glob(os.path.join("vqa_outputs_fresh", "crop_*.png")):
                        try:
                            os.remove(old)
                        except Exception:
                            pass
                    # Clean old crops_manifest if it exists
                    old_manifest_path = os.path.join("vqa_outputs_fresh", "crops_manifest.json")
                    if os.path.exists(old_manifest_path):
                        try:
                            os.remove(old_manifest_path)
                        except Exception:
                            pass
                except Exception:
                    pass
                
                # Support multiple SAM prompts
                prompts_to_process = sam_prompts.split("|") if sam_prompts else [question]

                # Collect raw masks (base64 png) across prompts for inter-class merging
                raw_masks = []
                for prompt_idx, sam_prompt in enumerate(prompts_to_process):
                    sam_prompt = sam_prompt.strip()
                    print(f"[Orchestrator] Requesting masks for prompt {prompt_idx + 1}/{len(prompts_to_process)}: '{sam_prompt}'")
                    response = self.sam3_client.get_merged_masks(
                        object_name=sam_prompt,
                        image_path=Path(image_path),
                        image_type=image_type,
                    )
                    masks_data = response.get("masks", [])
                    if not masks_data:
                        print(f"[WARNING] SAM3 API returned no masks for prompt '{sam_prompt}'")
                        continue

                    for sam_idx, mask_data in enumerate(masks_data[:max_candidates]):
                        # Normalize to dict form with base64 'png' key
                        entry: Dict[str, Any] = {}
                        if isinstance(mask_data, dict) and mask_data.get("png"):
                            entry["png"] = mask_data.get("png")
                        else:
                            # try to extract segmentation/mask array and encode to png
                            mask_array = None
                            if isinstance(mask_data, dict):
                                mask_array = mask_data.get("segmentation") or mask_data.get("mask") or mask_data.get("data")
                            else:
                                mask_array = mask_data

                            if mask_array is None:
                                print(f"[WARNING] Mask {sam_idx} for prompt '{sam_prompt}' has no valid data, skipping")
                                continue

                            try:
                                arr = np.array(mask_array)
                                # binarize
                                bin_arr = (arr > 0)
                                entry["png"] = mm.encode_mask(bin_arr)
                            except Exception as e:
                                print(f"[WARNING] Failed to encode mask {sam_idx} for prompt '{sam_prompt}': {e}")
                                continue

                        # preserve provenance fields so merging can keep metadata
                        entry["sam_prompt"] = sam_prompt
                        entry["sam_prompt_index"] = prompt_idx
                        entry["sam_index"] = sam_idx
                        raw_masks.append(entry)

                # If no masks found, continue but do not fail
                if len(raw_masks) == 0:
                    print("[Orchestrator] No masks returned by SAM3 for any prompt; proceeding with base image only.")
                else:
                    # Merge inter-class masks using IoM graph merging
                    try:
                        iom_thresh = float(os.environ.get("MASK_MERGE_IOM", "0.5"))
                        cov_thresh = float(os.environ.get("MASK_MERGE_COVERAGE", "0.5"))
                        merged_masks, merge_debug = mm.merge_iom_graph(raw_masks, [], iom_threshold=iom_thresh, coverage_threshold=cov_thresh, debug=True)
                        print(f"[Orchestrator] Merged {len(raw_masks)} masks -> {len(merged_masks)} masks (IOM={iom_thresh}, COV={cov_thresh})")
                    except Exception as e:
                        print(f"[WARNING] Mask merging failed: {e}; falling back to raw masks")
                        merged_masks = raw_masks

                    # Save merged masks and build mask_items
                    global_saved_idx = 0
                    for m in merged_masks:
                        png_b64 = m.get("png")
                        if not png_b64:
                            continue
                        try:
                            mask_arr = mm.decode_mask(png_b64)
                        except Exception as e:
                            print(f"[WARNING] Failed to decode merged mask: {e}")
                            continue

                        # ensure binary uint8 0/255 for saving
                        mask_bin = (mask_arr > 0).astype(np.uint8) * 255

                        coords = np.argwhere(mask_bin > 0)
                        if coords.size == 0:
                            continue
                        y_min, x_min = coords.min(axis=0)
                        y_max, x_max = coords.max(axis=0)

                        mask_path = os.path.join("vqa_outputs_fresh", f"mask_{global_saved_idx}.png")
                        cv2.imwrite(mask_path, mask_bin)

                        bbox_normalized = [float(x_min) / W, float(y_min) / H, float(x_max) / W, float(y_max) / H]
                        bbox_pixels = [int(x_min), int(y_min), int(x_max), int(y_max)]

                        try:
                            properties = geometric_utils.compute_mask_properties(mask_bin)
                        except Exception:
                            properties = {}

                        mask_items.append({
                            "index": global_saved_idx,
                            "mask_path": os.path.abspath(mask_path),
                            "bbox_normalized": bbox_normalized,
                            "bbox_pixels": bbox_pixels,
                            "sam_prompt": m.get("sam_prompt"),
                            "sam_prompt_index": m.get("sam_prompt_index"),
                            "sam_index": m.get("sam_index"),
                            "properties": properties,
                        })
                        global_saved_idx += 1
                
                if len(mask_items) == 0:
                    print("[WARNING] SAM3 API returned no valid masks for any prompt. Will proceed with base image only.")
                else:
                    print("[Orchestrator] SAM3 API successful - masks and bboxes ready.")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[WARNING] SAM3 API encountered an error: {e}")
                print("[Orchestrator] Will proceed with base image only (no crops).")
                # Continue with empty mask_items - no need to abort
        else:
            # Fallback to hardcoded boxes for testing
            bounding_boxes = [[0.0, 0.5859375, 0.01171875, 0.66015625], 
                    [0.02734375, 0.453125, 0.09375, 0.646484375], 
                    [0.14453125, 0.427734375, 0.216796875, 0.62890625], 
                    [0.083984375, 0.4453125, 0.15625, 0.63671875], 
                    [0.251953125, 0.404296875, 0.326171875, 0.59765625]]

        # If we used the API path above, construct bounding_boxes from manifest for downstream logic
        if use_sam3_api:
            # Derive bounding_boxes list in the exact saved order
            bounding_boxes = [it["bbox_normalized"] for it in mask_items]

        if not bounding_boxes:
            print("[Orchestrator] No bounding boxes found by SAM. Proceeding with base image only...")
            # Continue with empty crops - just use the base image
            cropped_objects = []
        else:
            # If using existing data, skip to VLM invocation
            # if using_existing_data:
            #     print("[Orchestrator] Skipping crop/manifest regeneration, using existing files...")
            #     # Build image_inputs from existing crop files
            #     image_inputs = [image_path]
            #     try:
            #         import glob
            #         crop_files = sorted(glob.glob(os.path.join("vqa_outputs_fresh", "crop_*.png")),
            #                           key=lambda x: int(x.split('_')[-1].split('.')[0]))
            #         image_inputs.extend([os.path.abspath(f) for f in crop_files])
            #         print(f"[Orchestrator] Loaded {len(crop_files)} existing crop images")
            #     except Exception as e:
            #         print(f"[WARNING] Failed to load existing crop files: {e}")
            # else:
                # Only regenerate crops and manifest if NOT using existing data
                # Store crop information without saving crop images
            cropped_objects = []
            for idx, box in enumerate(bounding_boxes):
                try:
                    xmin_n, ymin_n, xmax_n, ymax_n = box
                except Exception:
                    continue

                xmin = int(xmin_n * W)
                ymin = int(ymin_n * H)
                xmax = int(xmax_n * W)
                ymax = int(ymax_n * H)

                # Add 10% padding for context (5% on each side)
                width = xmax - xmin
                height = ymax - ymin
                pad_x = int(width * 0.5)  # 5% padding on each side
                pad_y = int(height * 0.5)
                
                xmin_padded = xmin - pad_x
                ymin_padded = ymin - pad_y
                xmax_padded = xmax + pad_x
                ymax_padded = ymax + pad_y

                # clamp to image boundaries
                xmin_padded = max(0, min(xmin_padded, W - 1))
                xmax_padded = max(0, min(xmax_padded, W - 1))
                ymin_padded = max(0, min(ymin_padded, H - 1))
                ymax_padded = max(0, min(ymax_padded, H - 1))

                if xmax_padded <= xmin_padded or ymax_padded <= ymin_padded:
                    continue

                # Build crop object; prefer manifest path if available
                manifest_mask_path = None
                try:
                    # mask_items exists only when use_sam3_api is True
                    if use_sam3_api and idx < len(mask_items):
                        manifest_mask_path = mask_items[idx]["mask_path"]
                except Exception:
                    manifest_mask_path = None

                mask_path = manifest_mask_path or os.path.join("vqa_outputs_fresh", f"mask_{idx}.png")
                has_mask = os.path.exists(mask_path)

                # Save crop image with padding for better context
                crop_img = img_bgr[ymin_padded:ymax_padded, xmin_padded:xmax_padded]
                crop_path = os.path.join("vqa_outputs_fresh", f"crop_{idx}.png")
                cv2.imwrite(crop_path, crop_img)

                cropped_objects.append({
                    "index": idx,
                    "box": [xmin, ymin, xmax, ymax],  # Original bbox (without padding)
                    "box_normalized": [xmin_n, ymin_n, xmax_n, ymax_n],
                    "crop_path": os.path.abspath(crop_path),
                    "mask_path": mask_path if has_mask else None,
                    "has_mask": has_mask,
                })

        if len(cropped_objects) == 0:
            print("[Orchestrator] No valid crops produced. Will proceed with base image only.")
            # Don't abort - continue with empty crops

        # Create crops manifest with all object information (crops + masks unified)
        # Structure: group crops by prompt for better organization
        try:
            prompts_to_process = sam_prompts.split("|") if sam_prompts else [question]
            
            # Group crops by prompt
            prompts_data = []
            for prompt_idx, prompt in enumerate(prompts_to_process):
                prompt = prompt.strip()  # Clean whitespace
                prompt_crops = []
                for obj in cropped_objects:
                    idx = obj['index']
                    # Check if this crop belongs to this prompt
                    if use_sam3_api and idx < len(mask_items):
                        if mask_items[idx].get("sam_prompt_index") == prompt_idx:
                            crop_item = {
                                "index": idx,
                                "bbox_pixels": obj['box'],
                                "bbox_normalized": obj['box_normalized'],
                                "crop_path": obj.get('crop_path', ''),
                                "mask_path": mask_items[idx].get("mask_path", ''),
                                "properties": mask_items[idx].get("properties", {}),
                                "sam_prompt": mask_items[idx].get("sam_prompt", prompt),
                                "sam_index": mask_items[idx].get("sam_index", 0),
                            }
                            prompt_crops.append(crop_item)
                    else:
                        # Fallback for non-SAM mode
                        crop_item = {
                            "index": idx,
                            "bbox_pixels": obj['box'],
                            "bbox_normalized": obj['box_normalized'],
                            "crop_path": obj.get('crop_path', ''),
                            "mask_path": obj.get('mask_path'),
                        }
                        prompt_crops.append(crop_item)
                
                if prompt_crops:
                    prompts_data.append({
                        "prompt": prompt,
                        "prompt_index": prompt_idx,
                        "crops": prompt_crops
                    })
            
            crops_manifest = {
                "image_path": os.path.abspath(image_path),
                "image_size": {"width": int(W), "height": int(H)},
                "prompts": prompts_data,
            }
            
            with open(os.path.join("vqa_outputs_fresh", "crops_manifest.json"), "w") as f:
                import json as _json
                _json.dump(crops_manifest, f, indent=2)
            print(f"[Orchestrator] Saved crops_manifest.json with {len(cropped_objects)} crops")
        except Exception as e:
            print(f"[WARNING] Failed to write crops_manifest.json: {e}")

        # Save annotated image with bounding boxes and indices for visual verification
        try:
            annotated = img_bgr.copy()
            # Simple color palette (BGR)
            colors = [
                (0, 255, 0),     # green
                (0, 0, 255),     # red
                (255, 0, 0),     # blue
                (0, 255, 255),   # yellow
                (255, 0, 255),   # magenta
                (255, 255, 0),   # cyan
                (128, 128, 255),
                (128, 255, 128),
                (255, 128, 128),
                (200, 200, 50),
            ]

            for obj in cropped_objects:
                xmin, ymin, xmax, ymax = obj['box']
                idx = int(obj['index'])
                color = colors[idx % len(colors)]
                cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 2)

                label = f"#{idx}"
                # Also show SAM index if available to help diagnose ordering
                if use_sam3_api:
                    try:
                        if 'mask_items' in locals() and 0 <= idx < len(mask_items):
                            sam_idx = mask_items[idx].get('sam_index')
                            label += f"/sam:{sam_idx}"
                    except Exception:
                        pass

                y_text = max(15, ymin - 5)
                # Outline for readability
                cv2.putText(annotated, label, (xmin, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(annotated, label, (xmin, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            base = os.path.splitext(os.path.basename(image_path))[0]
            anno_path = os.path.join("vqa_outputs_fresh", f"{base}_annotated.png")
            cv2.imwrite(anno_path, annotated)
            print(f"[Orchestrator] Saved annotated image: {anno_path}")
        except Exception as e:
            print(f"[WARNING] Failed to save annotated image: {e}")

        # Prepare images for agent: base image + all crop images
        image_inputs: List[str] = []
        if os.path.exists(image_path):
            image_inputs.append(os.path.abspath(image_path))
        else:
            return f"Base image not found: {image_path}\n"
        
        # Add crop images so VLM can see each region clearly
        for obj in cropped_objects:
            crop_path = obj.get('crop_path')
            if crop_path and os.path.exists(crop_path):
                image_inputs.append(crop_path)

        print(f"[Orchestrator] Passing 1 base image + {len(cropped_objects)} crop images to VLM")
        
        # Common VLM invocation (works for both new and existing data)
        print(f"[Orchestrator] Crops manifest: vqa_outputs_fresh/crops_manifest.json")
        print(f"[Orchestrator] Total images to VLM: {len(image_inputs)}")


        # For filtering mode, skip VLM and return merged mask paths directly
        if self.mode == "filtering":
            print("[Orchestrator] Filtering mode detected - skipping VLM, returning mask paths directly")
            
            # Build response with mask paths
            mask_paths = []
            for obj in cropped_objects:
                mask_path = obj.get('mask_path')
                if mask_path and os.path.exists(mask_path):
                    mask_paths.append(mask_path)
            
            if mask_paths:
                # Return paths as newline-separated list
                response = "\n".join(mask_paths)
                print(f"[Orchestrator] Returning {len(mask_paths)} mask path(s)")
                return response
            else:
                print("[Orchestrator] No valid masks found")
                return "No masks found.\n"

        # ensure VLM exists
        if self.vlm is None:
            return "VLM unavailable.\n"

        # Modify question if no crops were detected to avoid hallucination
        vlm_question = question
        if len(cropped_objects) == 0:
            vlm_question = (
                f"[NOTE: No specific objects were detected in this image by the object detection system. "
                f"You are viewing only the full base image with no crop images. "
                f"Answer based solely on what you can observe in the full image.]\n\n{question}"
            )
            print("[Orchestrator] No crops detected - added clarification prefix to question")

        # Call VLM agent with base image and bbox-enriched question
        try:
            if self.mode == "numerical":
                vlm_response = self.vlm.run_agent(image_inputs, vlm_question)
            else:
                vlm_response = self.vlm.answer_question(image_inputs, vlm_question)
            
            return vlm_response

        except Exception as e:
            return f"VLM query failed: {e}\n"
 
    def get_feature_list(
        self,
        image_path: str,
        custom_prompt: Optional[str] = None,
        max_new_tokens: int = 128,
    ) -> str:
        """Call the VLM with a custom prompt to extract a feature list from the image.
        
        Args:
            image_path: Path to the input image.
            custom_prompt: The user's original question (e.g., "How many stadiums are there?").
                           Used to scope the feature extraction to relevant objects only.
            max_new_tokens: Maximum number of tokens to generate in the response.
        
        Returns:
            The VLM's response containing the feature list.
        """
        path_obj = Path(image_path).expanduser().resolve()
        if not path_obj.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        self._ensure_models()
        if self.vlm is None:
            raise RuntimeError("VLM unavailable for feature extraction")

        # Store current VLM settings
        prev_prompt = self.vlm.system_prompt
        prev_stream = self.vlm.stream_thoughts
        prev_suppress = getattr(self.vlm, "_suppress_debug_prints", False)

        try:
            # System prompt: strict, minimal output format
            self.vlm.system_prompt = (
                "You are an object type extractor. Output ONLY a comma-separated list of object types. "
                "No explanations, no counts, no sentences - just the list."
            )
            self.vlm.stream_thoughts = False
            self.vlm._suppress_debug_prints = True

            # User prompt: clearly instructs what to extract based on the question
            user_question = (custom_prompt or "").strip()
            user_prompt = f"""USER QUESTION: "{user_question}"

Extract ONLY the object type(s) that are DIRECTLY mentioned or required to answer the question above.

RULES:
- Only include objects explicitly mentioned or directly needed for the question
- Ignore all background/unrelated objects in the image
- Lowercase, plural form, one word per type
- No explanations, no counts

Output:"""

            response = self.vlm.answer_question(
                [str(path_obj)],
                user_prompt,
                max_length=max_new_tokens,
            )
            return response

        finally:
            # Restore previous VLM settings
            self.vlm.system_prompt = prev_prompt
            self.vlm.stream_thoughts = prev_stream
            self.vlm._suppress_debug_prints = prev_suppress
 

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("VQA Orchestrator - Local LLM + GroundingDINO + SAM + VLM")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--question", required=True, help="Question to answer about the image")
    parser.add_argument("--device", default='cuda', help="Device to run models on (cuda/cpu)")
    parser.add_argument("--classes", default=None, help="Optional pipe-separated list of class phrases (e.g. 'red bus|yellow bus')")
    parser.add_argument("--score_threshold", type=float, default=0.35, help="Minimum detection confidence score")
    parser.add_argument("--no_sam3", action="store_true", help="Disable SAM3 API and use hardcoded boxes")
    parser.add_argument("--mode", choices=["attribute", "numerical", "binary", "filtering"], default="attribute", help="Task mode controlling prompt + tool usage")

    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("VQA ORCHESTRATOR")
    print("="*80)
    print(f"Image: {args.image}")
    print(f"Question: {args.question}")
    print(f"Device: {args.device}")
    print(f"Score Threshold: {args.score_threshold}")
    if args.classes:
        print(f"Custom Classes: {args.classes}")
    print("="*80 + "\n")
    
    orch = Orchestrator(
        device=args.device,
        mode=args.mode,
    )

    classes_list = args.classes.split("|") if args.classes else None
    ans = orch.run(
        args.image,
        args.question,
        use_sam3_api=not args.no_sam3,
        mode=args.mode,
    )