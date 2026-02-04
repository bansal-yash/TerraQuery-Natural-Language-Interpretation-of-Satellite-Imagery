##imports
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
import httpx
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
#from openai import OpenAI
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import re

load_dotenv()

app = FastAPI(root_path="/api")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

##config

URLS: Dict[str, str] = {
    "ROUTER": os.getenv("ROUTER_URL") or "", 
    "CAPTION": os.getenv("CAPTION_URL") or "",
    "GROUND": os.getenv("GROUND_URL") or "",
    "VQA": os.getenv("VQA_URL") or ""  # Base VQA URL, we'll append /filtering, /attribute, etc.
}


ROUTER_TIMEOUT = 300.0
CAPTION_TIMEOUT = 300.0
GROUND_TIMEOUT = 300.0
DEFAULT_VQA_TIMEOUT = 300.0
VQA_TIMEOUTS = {
    "attribute": 300.0,
    "binary": 300.0,
    "numerical": 300.0,
    "filtering": 300.0,
}

for key, value in URLS.items():
    if not value:
        raise RuntimeError(f"Missing environment variable for {key}")

# API_MODE: "openai" uses OpenAI-compatible API for router, "custom" uses raw HTTP POST
API_MODE = os.getenv("API_MODE", "openai")

if API_MODE == "openai":
    client = OpenAI(base_url=os.getenv("ROUTER_URL") or "", api_key=os.getenv("OPENAI_API_KEY", "lm-studio"))
else:
    client = None

##image handling

def encode_image_bytes(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

def compress_image(image_bytes: bytes, max_size: int = 512) -> tuple[bytes, float]:
    """Compress/downsample image to max_size and return (compressed_bytes, scale_factor).
    
    Args:
        image_bytes: Original image bytes
        max_size: Maximum dimension (width or height) for the output image
    
    Returns:
        Tuple of (compressed_image_bytes, scale_factor)
        scale_factor is the ratio of original size to new size (e.g., 2.0 means image was halved)
    """
    img = Image.open(io.BytesIO(image_bytes))
    original_width, original_height = img.size
    
    # Calculate scale factor before thumbnail
    # thumbnail scales to fit within max_size x max_size while preserving aspect ratio
    max_original_dim = max(original_width, original_height)
    if max_original_dim > max_size:
        scale_factor = max_original_dim / max_size
    else:
        scale_factor = 1.0
    
    img.thumbnail((max_size, max_size))
    
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue(), scale_factor


def draw_bboxes_on_image(image_bytes: bytes, boxes: List[List[int]], label: str = "Object") -> str:
    """Draw bounding boxes on an image and return as base64 encoded string.
    
    Args:
        image_bytes: Original image as bytes
        boxes: List of bounding boxes [[x1,y1,x2,y2], ...]
        label: Label text for the objects
    
    Returns:
        Base64 encoded image with bounding boxes drawn
    """
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode == "RGBA":
        img = img.convert("RGB")
    
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fall back to default if not available
    try:
        # Try common font paths
        font_size = max(12, min(img.width, img.height) // 30)
        font = ImageFont.truetype("arial.ttf", font_size)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()
    
    # Colors for different boxes (cycle through if more boxes than colors)
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
    ]
    
    for i, box in enumerate(boxes):
        if len(box) >= 4:
            x1, y1, x2, y2 = box[:4]
            color = colors[i % len(colors)]
            
            # Draw rectangle with thick border
            line_width = max(2, min(img.width, img.height) // 150)
            for offset in range(line_width):
                draw.rectangle(
                    [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                    outline=color
                )
            
            # Draw label background
            box_label = f"{label} {i + 1}"
            
            # Get text bounding box
            text_bbox = draw.textbbox((0, 0), box_label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Position label above the box, or inside if no room above
            label_x = x1
            label_y = y1 - text_height - 4
            if label_y < 0:
                label_y = y1 + 2
            
            # Draw label background rectangle
            draw.rectangle(
                [label_x, label_y, label_x + text_width + 6, label_y + text_height + 4],
                fill=color
            )
            
            # Draw label text in white
            draw.text((label_x + 3, label_y + 2), box_label, fill=(255, 255, 255), font=font)
    
    # Convert back to bytes and encode as base64
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


async def _fetch_image_from_url(image_url: str) -> bytes:
    if not image_url:
        raise ValueError("image_url is required")
    async with httpx.AsyncClient() as http_client:
        response = await http_client.get(image_url, timeout=120.0)
        response.raise_for_status()
        return response.content


def _format_binary_answer(answer: str) -> str:
    if not isinstance(answer, str):
        return "Unknown"
    if re.search(r"\byes\b", answer, re.IGNORECASE):
        return "Yes"
    if re.search(r"\bno\b", answer, re.IGNORECASE):
        return "No"
    return answer.strip() or "Unknown"


def _format_numeric_answer(answer: str, spatial_resolution: Optional[float] = None, scale_factor: float = 1.0) -> str:
    """Format numeric answer, applying scale factor and optionally converting pixels to meters.
    
    Args:
        answer: Raw answer string that may contain a number and optional suffix (pixels, square pixels, px, px²)
        spatial_resolution: Spatial resolution in meters per pixel (at original image resolution). If provided, converts:
            - "pixels" or "px" -> multiply by spatial_resolution
            - "square pixels" or "px²" -> multiply by spatial_resolution^2
        scale_factor: The downsampling scale factor (original_size / downsampled_size).
            For linear measurements (pixels), multiply by scale_factor.
            For area measurements (square pixels), multiply by scale_factor^2.
    
    Returns:
        Formatted numeric answer string
    """
    if not isinstance(answer, str):
        return "0"
    
    # Debug: print raw answer to see what we're working with
    print(f"[DEBUG] Raw answer: {repr(answer)}")
    print(f"[DEBUG] scale_factor: {scale_factor}, spatial_resolution: {spatial_resolution}")
    
    # Try to extract number and suffix
    # Match patterns like:
    # - "123.45 square pixels", "100 pixels", "50.5 px", "474.0 px²"
    # - "[FINAL ANSWER]\n511.0 pixels\n..."
    # - "The area is 474.0 px²."
    pattern = r"(-?\d+(?:\.\d+)?)\s*(square\s*pixels?|pixels?|px²|px\u00b2|px)"
    match = re.search(pattern, answer, re.IGNORECASE)
    
    print(f"[DEBUG] Regex match: {match}")
    if match:
        print(f"[DEBUG] Groups: {match.groups()}")
    
    if not match:
        # Fallback: just find any number
        matches = re.findall(r"-?\d+(?:\.\d+)?", answer)
        if not matches:
            return answer.strip() or "0"
        try:
            value = abs(float(matches[-1]))
            return ("%.6f" % value).rstrip("0").rstrip(".") or "0"
        except ValueError:
            return matches[-1]
    
    try:
        value = abs(float(match.group(1)))
        suffix = (match.group(2) or "").lower().strip()
        
        print(f"[DEBUG] Value: {value}, Suffix: {repr(suffix)}")
        
        # Determine if it's square (area) or linear measurement
        is_square = "square" in suffix or "²" in suffix or "\u00b2" in suffix
        is_pixel = "pixel" in suffix or "px" in suffix
        
        print(f"[DEBUG] is_square: {is_square}, is_pixel: {is_pixel}")
        
        # First, apply scale factor to convert from downsampled pixels to original pixels
        if is_pixel and scale_factor > 1.0:
            if is_square:
                # Area: multiply by scale_factor squared
                value = value * (scale_factor ** 2)
                print(f"[DEBUG] After scale_factor (area): {value}")
            else:
                # Linear: multiply by scale_factor
                value = value * scale_factor
                print(f"[DEBUG] After scale_factor (linear): {value}")
        
        # Then, apply spatial resolution conversion if provided
        if spatial_resolution is not None and spatial_resolution > 0 and is_pixel:
            if is_square:
                # Square pixels -> square meters
                value = value * (spatial_resolution ** 2)
                formatted_value = ("%.6f" % value).rstrip("0").rstrip(".") or "0"
                print(f"[DEBUG] Final value (m²): {formatted_value}")
                return f"{formatted_value}"
            else:
                # Linear pixels -> meters
                value = value * spatial_resolution
                formatted_value = ("%.6f" % value).rstrip("0").rstrip(".") or "0"
                print(f"[DEBUG] Final value (m): {formatted_value}")
                return f"{formatted_value}"
        
        return ("%.6f" % value).rstrip("0").rstrip(".") or "0"
    except ValueError:
        return match.group(1)


def _format_semantic_answer(answer: str) -> str:
    if not isinstance(answer, str):
        return ""
    return answer.strip()


##wrapper
        
async def call_vlm(
    prompt: str,
    image_bytes: bytes,
    context: str = "",
    temperature: float = 0.7
) -> Dict[str, Any]:
    """Call the router VLM to classify the request."""
    if API_MODE == "openai":
        return _call_openai_style(prompt, image_bytes, context, temperature)
    else:
        return await _call_custom_router(prompt, image_bytes, context)
    
##api structure handlers

def _call_openai_style(
    prompt: str,
    image_bytes: bytes,
    context: str,
    temperature: float
) -> Dict[str, Any]:
    if client is None:
        raise RuntimeError("OpenAI client is not configured for openai API mode")
    
    encoded_image = encode_image_bytes(image_bytes)
    
    full_prompt = prompt
    if context:
        full_prompt = f"Previous context: {context}\n\nCurrent question: {prompt}"
    
    completion = client.chat.completions.create(
        model=os.getenv("ROUTER_MODEL", "qwen/qwen3-vl-8b"),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            }
        ],
        temperature=temperature,
        timeout=ROUTER_TIMEOUT
    )
    
    answer = completion.choices[0].message.content or ""
    
    return {
        "answer": answer,
        "boxes": None,
        "labels": None
    }


async def _call_custom_router(
    prompt: str,
    image_bytes: bytes,
    context: str
) -> Dict[str, Any]:
    """Custom HTTP POST to router API (non-OpenAI style)."""
    full_prompt = prompt
    if context:
        full_prompt = f"Previous context: {context}\n\nCurrent question: {prompt}"
    
    async with httpx.AsyncClient() as http_client:
        try:
            # Send as multipart form data (file + form field)
            files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
            data = {"prompt": full_prompt}
            
            response = await http_client.post(
                URLS["ROUTER"],
                files=files,
                data=data,
                timeout=ROUTER_TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            return {
                "answer": result.get("classification", result.get("answer", "")),
                "boxes": result.get("boxes", result.get("bboxes")),
                "labels": result.get("labels")
            }
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Router API failed: {str(e)}")



async def _call_caption_api(image_bytes: bytes, prompt: str) -> Dict[str, Any]:
    """Upstream caption API.
    Request: multipart form with image file + prompt form field
    Response: {"caption": str}
    """
    async with httpx.AsyncClient() as http_client:
        try:
            files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
            data = {"prompt": prompt}
            response = await http_client.post(
                URLS["CAPTION"],
                files=files,
                data=data,
                timeout=CAPTION_TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            return {"answer": result.get("caption", ""), "boxes": None, "labels": None}
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Caption API failed: {str(e)}")


async def _call_ground_api(image_bytes: bytes, text: str, original_image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
    """Upstream grounding/bbox API.
    Request: multipart form with image file + object_name form field
    Response: {"bboxes": [[x1,y1,x2,y2], ...], "bboxes_normalized": [...], "object_name": str}
    """
    async with httpx.AsyncClient() as http_client:
        try:
            files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
            data = {"object_name": text}
            response = await http_client.post(
                URLS["GROUND"],
                files=files,
                data=data,
                timeout=GROUND_TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            boxes = result.get("bboxes", result.get("bboxes_normalized", []))
            
            # Extract object name from prompt for labeling
            # Try to get a cleaner label from common patterns
            label = text
            for pattern in [r'bounding box(?:es)? (?:for |of )?(?:the )?(.+)', 
                           r'locate (?:the )?(.+)', 
                           r'find (?:the )?(.+)',
                           r'where (?:is|are) (?:the )?(.+)',
                           r'show (?:me )?(?:the )?(.+)']:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    label = match.group(1).strip()
                    break
            
            # Capitalize first letter of label
            label = label.capitalize() if label else "Object"
            
            # Draw bounding boxes on the image
            image_with_boxes = None
            if boxes and len(boxes) > 0:
                img_to_draw = original_image_bytes if original_image_bytes else image_bytes
                image_with_boxes = draw_bboxes_on_image(img_to_draw, boxes, label)
            
            num_boxes = len(boxes) if boxes else 0
            answer = f"Found {num_boxes} {label.lower()}" + ("s" if num_boxes != 1 else "")
            
            return {
                "answer": answer, 
                "boxes": boxes, 
                "labels": [f"{label} {i+1}" for i in range(num_boxes)],
                "image_with_boxes": image_with_boxes
            }
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Ground API failed: {str(e)}")


async def _call_vqa_api(image_bytes: bytes, question: str, vqa_type: str) -> Dict[str, Any]:
    """Upstream VQA API for attribute/numerical/binary/filtering.
    Endpoints: VQA_URL/attribute, VQA_URL/numerical, VQA_URL/binary, VQA_URL/filtering
    Request: multipart form with image file + question form field
    Response varies by type:
      - attribute: {"answer": str}
      - numerical: {"answer": int}
      - binary: {"answer": bool}
      - filtering: {"bboxes": [[x1,y1,x2,y2], ...]}
    """
    # Build the endpoint URL: e.g., http://vqa-service/vqa/attribute
    vqa_endpoint = f"{URLS['VQA'].rstrip('/')}/{vqa_type}"
    
    timeout_seconds = VQA_TIMEOUTS.get(vqa_type, DEFAULT_VQA_TIMEOUT)

    async with httpx.AsyncClient() as http_client:
        try:
            files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
            data = {"question": question}
            response = await http_client.post(
                vqa_endpoint,
                files=files,
                data=data,
                timeout=timeout_seconds
            )
            response.raise_for_status()
            result = response.json()
            
            if vqa_type == "filtering":
                # Filtering returns annotated image + mask metadata
                raw_answer = result.get("answer", "")
                masks = []
                thinking = result.get("thinking")
                image_with_boxes = None
                
                if isinstance(raw_answer, str):
                    # Extract thinking if present
                    if not thinking:
                        thinking_match = re.search(r'\[THINKING\]\s*(.*?)\s*\[FINAL ANSWER\]', raw_answer, re.IGNORECASE | re.DOTALL)
                        if thinking_match:
                            thinking = thinking_match.group(1).strip()
                    
                    # Extract JSON array after [FINAL ANSWER]
                    json_match = re.search(r'\[FINAL ANSWER\]\s*(\[.*\])', raw_answer, re.IGNORECASE | re.DOTALL)
                    if json_match:
                        try:
                            import json
                            masks = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            masks = []
                    else:
                        # Failsafe: try to find any JSON array in the response
                        json_array_match = re.search(r'(\[\s*\{.*?\}\s*\])', raw_answer, re.DOTALL)
                        if json_array_match:
                            try:
                                import json
                                masks = json.loads(json_array_match.group(1))
                            except json.JSONDecodeError:
                                masks = []

                image_b64 = result.get("image_base64")
                if isinstance(image_b64, str) and image_b64.strip():
                    prefix = "data:image/png;base64,"
                    image_with_boxes = image_b64 if image_b64.startswith("data:") else f"{prefix}{image_b64}"
                
                return {
                    "answer": raw_answer or question,
                    "masks": masks,
                    "thinking": thinking,
                    "boxes": None,
                    "labels": None,
                    "image_with_boxes": image_with_boxes,
                    "objects": result.get("objects", []),
                    "obboxes": result.get("obboxes", []),
                }
            else:
                raw_answer = result.get("answer", "")
                thinking = None
                
                # Extract thinking/reasoning if present
                if isinstance(raw_answer, str):
                    thinking_match = re.search(r'\[THINKING\]\s*(.*?)\s*\[FINAL ANSWER\]', raw_answer, re.IGNORECASE | re.DOTALL)
                    if thinking_match:
                        thinking = thinking_match.group(1).strip()
                
                # For numerical VQA, extract the number and unit after [FINAL ANSWER]
                if vqa_type == "numerical" and isinstance(raw_answer, str):
                    # Extract number with optional suffix (pixels, px, px², square pixels, etc.)
                    match = re.search(r'\[FINAL ANSWER\]\s*(\d+(?:\.\d+)?)\s*(square\s*pixels?|pixels?|px²|px\u00b2|px)?', raw_answer, re.IGNORECASE)
                    if match:
                        number = match.group(1)
                        suffix = match.group(2) or ""
                        raw_answer = f"{number} {suffix}".strip()
                    else:
                        # Fallback: find any number with optional suffix
                        fallback_match = re.search(r'(\d+(?:\.\d+)?)\s*(square\s*pixels?|pixels?|px²|px\u00b2|px)?', raw_answer, re.IGNORECASE)
                        if fallback_match:
                            number = fallback_match.group(1)
                            suffix = fallback_match.group(2) or ""
                            raw_answer = f"{number} {suffix}".strip()
                        else:
                            raw_answer = "0"
                    return {"answer": raw_answer, "thinking": thinking, "boxes": None, "labels": None}
                # For binary VQA, extract Yes/No after [FINAL ANSWER]
                elif vqa_type == "binary" and isinstance(raw_answer, str):
                    match = re.search(r'\[FINAL ANSWER\]\s*(yes|no)', raw_answer, re.IGNORECASE)
                    if match:
                        raw_answer = match.group(1).capitalize()
                    else:
                        # Failsafe: search for yes/no anywhere in the response
                        if re.search(r'\byes\b', raw_answer, re.IGNORECASE):
                            raw_answer = "Yes"
                        elif re.search(r'\bno\b', raw_answer, re.IGNORECASE):
                            raw_answer = "No"
                        else:
                            raw_answer = "Unknown"
                    return {"answer": raw_answer, "thinking": thinking, "boxes": None, "labels": None}
                # For attribute VQA, extract text after [FINAL ANSWER]
                elif vqa_type == "attribute" and isinstance(raw_answer, str):
                    match = re.search(r'\[FINAL ANSWER\]\s*(.+?)(?:\n|<|$)', raw_answer, re.IGNORECASE | re.DOTALL)
                    if match:
                        raw_answer = match.group(1).strip()
                    return {"answer": raw_answer, "thinking": thinking, "boxes": None, "labels": None}
                return {"answer": raw_answer, "thinking": thinking, "boxes": None, "labels": None}
        except httpx.HTTPError as e:
            error_detail = f"{e.__class__.__name__}: {repr(e)}"
            print(f"[VQA API ERROR] type={vqa_type} detail={error_detail}")
            raise HTTPException(status_code=502, detail=f"VQA API ({vqa_type}) failed: {error_detail}")


def _round_obbox_values(obbox: List[Any]) -> List[float]:
    rounded = []
    for value in obbox or []:
        try:
            rounded.append(round(float(value), 6))
        except (TypeError, ValueError):
            rounded.append(0.0)
    return rounded


##endpoints
@app.get("/")
async def root():
    return {"message": "Running", "api_mode": API_MODE}

@app.post("/handle")
async def handler(
    prompt: str = Form(...),
    context: str = Form(""),
    image: UploadFile = File(...)
):
    try:
        image_bytes = await image.read()
        image_bytes, scale_factor = compress_image(image_bytes)

        router_prompt = f"""Given this image and user request, classify which task is needed.
Any request involving locating, grounding, drawing bounding boxes, or filtering objects MUST be classified as VQA_FILTERING.
Reply with ONLY one of these exact words:
- CAPTION: User wants a description/caption of the image
- VQA_ATTRIBUTE: User asks about properties/attributes of objects (color, shape, material, etc.)
- VQA_NUMERICAL: User asks "how many" or wants to count objects
- VQA_BINARY: User asks a yes/no question about the image
- VQA_FILTERING: User wants to locate/find objects, draw bounding boxes, or filter objects matching criteria

User request: {prompt}"""
        
        router_response = await call_vlm(router_prompt, image_bytes, context, temperature=0.1)
        route_decision = router_response["answer"].strip().upper()

        # Determine the route
        if "CAPTION" in route_decision:
            route_type = "CAPTION"
        elif "GROUND" in route_decision:
            route_type = "GROUND"
        elif "VQA_ATTRIBUTE" in route_decision:
            route_type = "VQA_ATTRIBUTE"
        elif "VQA_NUMERICAL" in route_decision:
            route_type = "VQA_NUMERICAL"
        elif "VQA_BINARY" in route_decision:
            route_type = "VQA_BINARY"
        elif "VQA_FILTERING" in route_decision:
            route_type = "VQA_FILTERING"
        else:
            # Default fallback
            route_type = "VQA_ATTRIBUTE"

        # Store original image bytes before compression for drawing
        original_image_bytes = image_bytes
        
        # Call the appropriate upstream API
        if route_type == "CAPTION":
            result = await _call_caption_api(image_bytes, prompt)
        elif route_type == "GROUND":
            result = await _call_ground_api(image_bytes, prompt, original_image_bytes)
        elif route_type.startswith("VQA_"):
            vqa_type = route_type.replace("VQA_", "").lower()  # attribute, numerical, binary, filtering
            result = await _call_vqa_api(image_bytes, prompt, vqa_type)
        else:
            result = await _call_vqa_api(image_bytes, prompt, "attribute")

        return {
            "routed_to": route_type,
            "result": result
        }
    except HTTPException as http_exc:
        detail = http_exc.detail
        if isinstance(detail, dict):
            error_message = detail
        else:
            error_message = {"error": str(detail) if detail else str(http_exc)}
        return {
            "routed_to": "ERROR",
            "result": error_message,
        }
    except Exception as exc:
        return {
            "routed_to": "ERROR",
            "result": {"error": str(exc)},
        }


@app.post("/eval")
async def evaluate(payload: Dict[str, Any]):
    """Directly execute the provided queries without using the router."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload format")

    input_image = payload.get("input_image") or {}
    image_url = input_image.get("image_url")
    if not image_url:
        raise HTTPException(status_code=400, detail="input_image.image_url is required")

    try:
        original_bytes = await _fetch_image_from_url(image_url)
    except Exception as download_err:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {download_err}")

    # Compress/downsample image and get scale factor
    image_bytes, scale_factor = compress_image(original_bytes)
    print(f"[DEBUG] Image scale_factor: {scale_factor}")
    
    queries_section = payload.get("queries") or {}
    queries_response: Dict[str, Any] = {}

    # Caption query
    caption_query = queries_section.get("caption_query")
    if isinstance(caption_query, dict):
        instruction = caption_query.get("instruction", "")
        caption_prompt = instruction or "Provide a detailed caption for the image."
        caption_result = await _call_caption_api(image_bytes, caption_prompt)
        queries_response["caption_query"] = {
            "instruction": instruction,
            "response": caption_result.get("answer", "")
        }

    # Grounding query (expects oriented bounding boxes)
    grounding_query = queries_section.get("grounding_query")
    if isinstance(grounding_query, dict):
        instruction = grounding_query.get("instruction", "")
        grounding_result = await _call_vqa_api(image_bytes, instruction, "filtering")
        objects = grounding_result.get("objects") or []
        response_items = []
        object_counter = 1
        for obj in objects:
            obbox = obj.get("obbox")
            if not obbox:
                continue
            response_items.append({
                "object-id": str(object_counter),
                "obbox": _round_obbox_values(obbox),
            })
            object_counter += 1
        queries_response["grounding_query"] = {
            "instruction": instruction,
            "response": response_items,
        }

    # Extract spatial resolution from metadata if available
    metadata = input_image.get("metadata") or {}
    spatial_resolution = metadata.get("spatial_resolution_m")
    print(spatial_resolution)
    if spatial_resolution is not None:
        try:
            spatial_resolution = float(spatial_resolution)
        except (TypeError, ValueError):
            spatial_resolution = None

    # Attribute queries (binary/numeric/semantic all routed to attribute VQA)
    attribute_queries = queries_section.get("attribute_query") or {}
    attribute_response: Dict[str, Any] = {}
    attr_mode_map = {
        "binary": "binary",
        "numeric": "numerical",
        "semantic": "attribute",
    }
    for attr_type in ("binary", "numeric", "semantic"):
        entry = attribute_queries.get(attr_type)
        if not isinstance(entry, dict):
            continue
        instruction = entry.get("instruction", "")
        vqa_mode = attr_mode_map.get(attr_type, "attribute")
        attr_result = await _call_vqa_api(image_bytes, instruction, vqa_mode)
        raw_answer = attr_result.get("answer", "")
        if attr_type == "binary":
            formatted = _format_binary_answer(raw_answer)
        elif attr_type == "numeric":
            print(raw_answer)
            formatted = _format_numeric_answer(raw_answer, spatial_resolution, scale_factor)
            print(formatted)
        else:
            formatted = _format_semantic_answer(raw_answer)
        attribute_response[attr_type] = {
            "instruction": instruction,
            "response": formatted,
        }

    if attribute_response:
        queries_response["attribute_query"] = attribute_response

    return {
        "input_image": input_image,
        "queries": queries_response,
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "message": "Service is running"}