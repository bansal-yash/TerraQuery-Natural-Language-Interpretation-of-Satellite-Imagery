import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import qwen_mtmd
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
    max_new_tokens: int = Field(128, ge=1)


app = FastAPI(title="qwen_mtmd_api")
HANDLE_CACHE: Dict[Tuple[str, str], Any] = {}


def _handle_key() -> Tuple[str, str]:
    return str(MODEL_PATH), str(MMPROJ_PATH)


def open_handle(n_threads: int = 8) -> Any:
    key = _handle_key()
    if key not in HANDLE_CACHE:
        _PathsConfig.validate()
        HANDLE_CACHE[key] = qwen_mtmd.load(key[0], key[1], -1, n_threads, False)
    return HANDLE_CACHE[key]


@app.on_event("shutdown")
def _shutdown() -> None:
    for handle in HANDLE_CACHE.values():
        qwen_mtmd.free_handle(handle)
    HANDLE_CACHE.clear()


def _run_chat(handle: Any, messages: List[dict], n_batch: int, max_new_tokens: int) -> str:
    return qwen_mtmd.infer_chat(handle, messages, n_batch, max_new_tokens)


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
    r"<ref>\s*(?P<label>.*?)\s*</ref>\s*<box>\s*\(\s*(?P<x1>\d+)\s*,\s*(?P<y1>\d+)\s*\)\s*,\s*\(\s*(?P<x2>\d+)\s*,\s*(?P<y2>\d+)\s*\)\s*(?:</box>|<\|box_end\|>)",
    flags=re.IGNORECASE,
)
BOX_JSON_PATTERN = re.compile(
    r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*,\s*"label"\s*:\s*"([^\"]+)"',
    flags=re.IGNORECASE,
)


def _parse_boxes(reply: str) -> List[dict]:
    boxes: List[dict] = []
    for match in BOX_PATTERN.finditer(reply):
        boxes.append({
            "label": match.group("label").strip(),
            "x1": int(match.group("x1")),
            "y1": int(match.group("y1")),
            "x2": int(match.group("x2")),
            "y2": int(match.group("y2")),
        })
    for match in BOX_JSON_PATTERN.finditer(reply):
        x1, y1, x2, y2, label = match.groups()
        boxes.append({
            "label": label,
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
        })
    return boxes


def _boxes_overlap(a: dict, b: dict) -> bool:
    return not (
        a["x2"] < b["x1"]
        or b["x2"] < a["x1"]
        or a["y2"] < b["y1"]
        or b["y2"] < a["y1"]
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
    system_prompt = (
        "You are Qwen3-VL, a precise multimodal assistant responsible for detection output. "
        "Emit only bounding boxes wrapped in <ref>label</ref><box>(x1,y1),(x2,y2)</box> with coordinates in [0,1000]; do not add commentary."
    )
    user_prompt = (
        f"Identify every {object_name} that is fully visible, merge overlapping or touching instances into a single box, "
        "and ensure the coordinate order obeys x1<=x2 and y1<=y2. Reply 'none' if no matching object is present."
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    reply = _run_chat(handle, messages, n_batch, max_new_tokens)
    boxes = _parse_boxes(reply)
    merged = _merge_boxes(boxes)
    return {"raw": reply, "boxes": merged}


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
