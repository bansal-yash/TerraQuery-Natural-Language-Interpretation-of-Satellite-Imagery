import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# import qwen_mtmd
import typing
from PIL import Image, ImageDraw, ImageFont

# Toggle this to False to use the torch (HF) version of Qwen instead of the mtmd loader.
# By default respect an environment variable if provided, otherwise True.
mtmd = bool(int(os.environ.get("QWEN_USE_MTMD", "0")))
combine = bool(int(os.environ.get("QWEN_COMBINE_BOXES", "0")))
use_adapter = bool(int(os.environ.get("QWEN_USE_ADAPTER", "0")))  # Set to 0 to skip adapter and use base model only

# Model id to use when `mtmd` is False (torch/HF path).
# Note: The adapter was trained on 4-bit base, but we load full precision and it usually works
BASE_MODEL_ID = os.environ.get("QWEN_TORCH_BASE_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
ADAPTER_PATH = os.environ.get("QWEN_ADAPTER_PATH", "sar+rgb_new_para")
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field


MODEL_PATH = Path(os.environ.get("QWEN_MTMD_MODEL_PATH", "/home/samyak/scratch/interiit/gguf/8b/Qwen3VL-8B-Instruct-F16.gguf")).expanduser()
MMPROJ_PATH = Path(os.environ.get("QWEN_MTMD_MMPROJ_PATH", "/home/samyak/scratch/interiit/gguf/8b/mmproj-Qwen3VL-8B-Instruct-F16.gguf")).expanduser()


class _PathsConfig:
    @staticmethod
    def validate() -> None:
        missing = []
        for name, path in (("MODEL", MODEL_PATH), ("MMProj", MMPROJ_PATH)):
            if not path.exists():
                missing.append(f"{name} path missing: {path}")
        if missing:
            raise RuntimeError("; ".join(missing))


class BBoxRequest(BaseModel):
    # kept for documentation but API will accept multipart/form-data instead
    image_name: str = Field(None, description="Not used when uploading image; keep for compatibility")
    object_name: str = Field(..., min_length=1)
    n_batch: int = Field(64, ge=1)
    max_new_tokens: int = Field(128, ge=1)


class CaptionRequest(BaseModel):
    image_name: str = Field(None, description="Not used when uploading image; keep for compatibility")
    n_batch: int = Field(64, ge=1)
    max_new_tokens: int = Field(256, ge=1)


class FeaturesRequest(BaseModel):
    image_name: str = Field(None, description="Not used when uploading image; keep for compatibility")
    describer: str = Field(..., min_length=1, max_length=64)
    n_batch: int = Field(64, ge=1)
    max_new_tokens: int = Field(256, ge=1)


class GeneralInferenceRequest(BaseModel):
    # Multipart form data will be used, so this is for documentation
    system_prompt: str = Field(None, description="Optional system prompt")
    user_prompt: str = Field(..., min_length=1)
    n_batch: int = Field(64, ge=1)
    max_new_tokens: int = Field(512, ge=1)


app = FastAPI(title="qwen_mtmd_api")
HANDLE_CACHE: Dict[Tuple[str, str], Any] = {}


def _handle_key() -> Tuple[str, str]:
    return str(MODEL_PATH), str(MMPROJ_PATH)


def open_handle(n_threads: int = 8) -> Any:
    key = _handle_key()
    if key not in HANDLE_CACHE:
        if mtmd:
            _PathsConfig.validate()
            HANDLE_CACHE[key] = qwen_mtmd.load(key[0], key[1], -1, n_threads, False)
        else:
            # Torch / HF path: load base model + LoRA adapter lazily
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            from peft import PeftModel
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            
            # Check if we should load adapter
            adapter_path = Path(ADAPTER_PATH)
            should_load_adapter = (
                use_adapter 
                and adapter_path.exists() 
                and (adapter_path / "adapter_config.json").exists()
            )
            
            if should_load_adapter:
                print(f"Loading base model {BASE_MODEL_ID} on {device}...")
                base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    BASE_MODEL_ID,
                    torch_dtype=dtype,
                    device_map="auto",
                )
                print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
                model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
                model = model.merge_and_unload()  # Merge for faster inference
                print("✅ LoRA adapter merged successfully")
                # Load processor from adapter path (has tokenizer config)
                processor = AutoProcessor.from_pretrained(ADAPTER_PATH)
            else:
                if not use_adapter:
                    print(f"Adapter disabled (QWEN_USE_ADAPTER=0), loading base model only...")
                else:
                    print(f"No adapter found at {ADAPTER_PATH}, loading base model only...")
                print(f"Loading base model {BASE_MODEL_ID} on {device}...")
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    BASE_MODEL_ID,
                    torch_dtype=dtype,
                    device_map="auto",
                )
                processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
            
            model.eval()
            # store a dict so shutdown can detect and free appropriately
            HANDLE_CACHE[key] = {"type": "torch", "model": model, "processor": processor}
    return HANDLE_CACHE[key]


@app.on_event("shutdown")
def _shutdown() -> None:
    for handle in HANDLE_CACHE.values():
        try:
            if mtmd:
                qwen_mtmd.free_handle(handle)
            else:
                # torch handle: try to delete model and clear CUDA cache
                if isinstance(handle, dict) and handle.get("type") == "torch":
                    try:
                        model = handle.get("model")
                        del model
                    except Exception:
                        pass
                    try:
                        import torch

                        torch.cuda.empty_cache()
                    except Exception:
                        pass
        except Exception:
            pass
    HANDLE_CACHE.clear()


def _run_chat(handle: Any, messages: List[dict], n_batch: int, max_new_tokens: int) -> str:
    if mtmd:
        return qwen_mtmd.infer_chat(handle, messages, n_batch, max_new_tokens)
    # Torch/HF path
    # `handle` expected to be a dict with keys: model, processor
    if isinstance(handle, dict) and handle.get("type") == "torch":
        model = handle["model"]
        processor = handle["processor"]
        import torch
        from PIL import Image

        # Build prompt using processor's chat template
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Collect images from messages (open as PIL.Image)
        images = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        img_obj = item.get("image")
                        try:
                            # image may be a path string
                            if isinstance(img_obj, str):
                                images.append(Image.open(img_obj).convert("RGB"))
                            else:
                                images.append(img_obj)
                        except Exception:
                            # skip images we can't open
                            pass

        # IMPORTANT: images FIRST, then text (matches training code)
        inputs = processor(
            images=images if images else None,
            text=prompt,
            return_tensors="pt"
        ).to(model.device)
        
        input_token_len = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            gen_outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        
        # Extract only the newly generated tokens (after the input prompt)
        generated_ids = gen_outputs[0][input_token_len:]
        
        # decode using processor tokenizer
        response_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response_text
    # Fallback: if handle shape unexpected, raise
    raise RuntimeError("Invalid handle for non-mtmd mode")


class _DSU:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        if self.parent[index] != index:
            self.parent[index] = self.find(self.parent[index])
        return self.parent[index]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


BOX_PATTERN = re.compile(
    r"<ref>\s*(?P<label>.+?)\s*</ref>\s*<box>\s*\(\s*(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*(?P<y1>-?\d+(?:\.\d+)?)\s*\)\s*,\s*\(\s*(?P<x2>-?\d+(?:\.\d+)?)\s*,\s*(?P<y2>-?\d+(?:\.\d+)?)\s*\)\s*(?:</box>|<\|box_end\|>)",
    flags=re.IGNORECASE | re.DOTALL,
)
BOX_JSON_PATTERN = re.compile(
    (
        r'"label"\s*:\s*"(?P<label>[^\"]+)"[^{}]*?'
        r'"bbox_2d"\s*:\s*\[\s*(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*(?P<y1>-?\d+(?:\.\d+)?)\s*,\s*(?P<x2>-?\d+(?:\.\d+)?)\s*,\s*(?P<y2>-?\d+(?:\.\d+)?)\s*\]'
        r'|"bbox_2d"\s*:\s*\[\s*(?P<x1_alt>-?\d+(?:\.\d+)?)\s*,\s*(?P<y1_alt>-?\d+(?:\.\d+)?)\s*,\s*(?P<x2_alt>-?\d+(?:\.\d+)?)\s*,\s*(?P<y2_alt>-?\d+(?:\.\d+)?)\s*\]'
        r'[^{}]*?"label"\s*:\s*"(?P<label_alt>[^\"]+)"'
    ),
    flags=re.IGNORECASE,
)


def _parse_boxes(reply: str) -> List[dict]:
    boxes: List[dict] = []
    seen: Set[Tuple[str, int, int, int, int]] = set()

    def _coerce(value: str) -> int:
        number = int(round(float(value)))
        return max(0, min(1000, number))

    def _append_box(label: str, x1: str, y1: str, x2: str, y2: str) -> None:
        if not all((label, x1, y1, x2, y2)):
            return
        try:
            x1_i = _coerce(x1)
            y1_i = _coerce(y1)
            x2_i = _coerce(x2)
            y2_i = _coerce(y2)
        except (TypeError, ValueError):
            return
        if x1_i >= x2_i or y1_i >= y2_i:
            return
        key = (label.strip() or "object", x1_i, y1_i, x2_i, y2_i)
        if key in seen:
            return
        seen.add(key)
        boxes.append({
            "label": label.strip() or "object",
            "x1": x1_i,
            "y1": y1_i,
            "x2": x2_i,
            "y2": y2_i,
        })

    for match in BOX_PATTERN.finditer(reply):
        _append_box(
            match.group("label"),
            match.group("x1"),
            match.group("y1"),
            match.group("x2"),
            match.group("y2"),
        )

    for match in BOX_JSON_PATTERN.finditer(reply):
        label = match.group("label") or match.group("label_alt")
        x1 = match.group("x1") or match.group("x1_alt")
        y1 = match.group("y1") or match.group("y1_alt")
        x2 = match.group("x2") or match.group("x2_alt")
        y2 = match.group("y2") or match.group("y2_alt")
        _append_box(label, x1, y1, x2, y2)

    return boxes


def _boxes_overlap(a: dict, b: dict) -> bool:
    return not (
        a["x2"] <= b["x1"]
        or b["x2"] <= a["x1"]
        or a["y2"] <= b["y1"]
        or b["y2"] <= a["y1"]
    )


def _merge_boxes(boxes: List[dict]) -> List[dict]:
    if len(boxes) < 2:
        return boxes
    dsu = _DSU(len(boxes))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _boxes_overlap(boxes[i], boxes[j]):
                dsu.union(i, j)
    groups: Dict[int, List[dict]] = {}
    for idx, box in enumerate(boxes):
        root = dsu.find(idx)
        groups.setdefault(root, []).append(box)
    merged: List[dict] = []
    for group in groups.values():
        merged.append({
            "label": ", ".join(sorted({b.get("label", "object") for b in group})),
            "x1": min(b["x1"] for b in group),
            "y1": min(b["y1"] for b in group),
            "x2": max(b["x2"] for b in group),
            "y2": max(b["y2"] for b in group),
        })
    return merged


def _save_upload_to_temp(upload: UploadFile) -> Path:
    suffix = Path(upload.filename).suffix if upload.filename else ".jpg"
    tmp = tempfile.NamedTemporaryFile(prefix="qwen_img_", suffix=suffix, delete=False)
    try:
        data = upload.file.read()
        tmp.write(data)
        tmp.flush()
        tmp_path = Path(tmp.name).resolve()
    finally:
        try:
            tmp.close()
        except Exception:
            pass
    return tmp_path


def _run_bbox(handle: Any, image_path: Path, object_name: str, n_batch: int, max_new_tokens: int) -> Dict[str, Any]:
    image = Image.open(image_path).convert("RGB")

    # MISSION CRITICAL FORMATTING INSTRUCTIONS:
    # The response MUST follow this exact JSON schema:
    # {
    #   "raw": "string - the raw model output",
    #   "boxes": [
    #     {
    #       "label": "string - descriptive label for the object",
    #       "x1": integer - left coordinate normalized to [0,1000],
    #       "y1": integer - top coordinate normalized to [0,1000],
    #       "x2": integer - right coordinate normalized to [0,1000],
    #       "y2": integer - bottom coordinate normalized to [0,1000]
    #     }
    #   ]
    # }
    # All coordinates MUST be integers in the range [0, 1000].
    # The boxes array may be empty if no objects are detected.

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"""Describe the locations of the {object_name} using bounding box coordinates (x1,y1,x2,y2) normalized to 1000.
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

Output ONLY the bounding box annotations, one per line. No explanations."""}
        ]}
    ]
    reply = _run_chat(handle, messages, n_batch, max_new_tokens)
    boxes = _parse_boxes(reply)
    print(f"Detected {len(boxes)} boxes for object '{object_name}'")
    if combine:
        boxes = _merge_boxes(boxes)
    return {"raw": reply, "boxes": boxes}


@app.post("/bbox")
def bbox_endpoint(
    image: UploadFile = File(...),
    object_name: str = Form(...),
    n_batch: int = Form(64),
    max_new_tokens: int = Form(128),
) -> Dict[str, Any]:
    handle = open_handle()
    tmp_path = _save_upload_to_temp(image)
    try:
        return _run_bbox(handle, tmp_path, object_name, n_batch, max_new_tokens)
    finally:
        try:
            os.unlink(str(tmp_path))
        except Exception:
            pass


@app.post("/caption")
def caption_endpoint(
    image: UploadFile = File(...),
    n_batch: int = Form(64),
    max_new_tokens: int = Form(256),
) -> Dict[str, str]:
    handle = open_handle()
    tmp_path = _save_upload_to_temp(image)
    try:
        system_prompt = (
            "You are Qwen3-VL, the most articulate multimodal assistant available. "
            "Provide a vivid multi-sentence description that lays out the dominant subjects, textures, lighting, color palette, and any suggested activity."
        )
        user_prompt = (
            "Describe the scene in 100-120 words. Mention the foremost objects, ground treatment, lighting, depth, and any implied motion while avoiding hallucination. "
            "If humans or vehicles are present, note their posture or trajectory."
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(tmp_path)},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        return {"caption": _run_chat(handle, messages, n_batch, max_new_tokens).strip()}
    finally:
        try:
            os.unlink(str(tmp_path))
        except Exception:
            pass


@app.post("/features")
def features_endpoint(
    image: UploadFile = File(...),
    describer: str = Form(...),
    n_batch: int = Form(64),
    max_new_tokens: int = Form(128),
) -> Dict[str, List[str]]:
    handle = open_handle()
    tmp_path = _save_upload_to_temp(image)
    try:
        system_prompt = (
            "You are Qwen3-VL, an exact inventory assistant. "
            "Return only plural nouns describing features that visibly match the user prompt. Only mention features that are clearly present in the image; do not hallucinate or over presume."
        )
        user_prompt = (
            f"Given the description '{describer}', list every matching plural noun you can visibly verify from the image. Do not mention any object that is not clearly present. Output format: comma-separated plurals."
            "No explanations, only a comma-separated series of plurals."
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(tmp_path)},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        raw = _run_chat(handle, messages, n_batch, max_new_tokens)
        features = [item.strip().lower().rstrip('.') for item in raw.replace('\n', ',').split(',') if item.strip()]
        return {"features": features}
    finally:
        try:
            os.unlink(str(tmp_path))
        except Exception:
            pass


@app.post("/general_inference")
def general_inference_endpoint(
    images: List[UploadFile] = File(...),
    user_prompt: str = Form(...),
    system_prompt: str = Form(None),
    n_batch: int = Form(64),
    max_new_tokens: int = Form(512),
) -> Dict[str, str]:
    """General VLM inference endpoint supporting multiple images and arbitrary text prompts.
    
    This endpoint handles any text+image query, making it suitable for:
    - Visual question answering
    - Image description with specific instructions
    - Multi-image reasoning
    - Tool-based reasoning (when using agent frameworks)
    
    Args:
        images: One or more image files to process
        user_prompt: The question or instruction
        system_prompt: Optional system-level instructions for the model
        n_batch: Batch size for processing
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Dictionary with "response" key containing model's generated text
    """
    handle = open_handle()
    tmp_paths = []
    try:
        # Save all uploaded images to temporary files
        for img in images:
            tmp_paths.append(_save_upload_to_temp(img))
        
        # Build messages with system prompt (if provided) and user content
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })
        
        # Construct user message with all images followed by text
        user_content = []
        for tmp_path in tmp_paths:
            user_content.append({"type": "image", "image": str(tmp_path)})
        user_content.append({"type": "text", "text": user_prompt})
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # Run inference
        response = _run_chat(handle, messages, n_batch, max_new_tokens)
        return {"response": response.strip()}
        
    finally:
        # Clean up temporary files
        for tmp_path in tmp_paths:
            try:
                os.unlink(str(tmp_path))
            except Exception:
                pass
