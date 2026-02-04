import os
import json
import random
import requests
from tqdm import tqdm
import time
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

random.seed(42)

num_files = 1

def classify_answer(ans):
    try:
        int(str(ans).strip())
        return "numeric"
    except:
        pass
    if str(ans).strip().lower() in {"yes", "no", "true", "false"}:
        return "binary"
    return "semantic"

def flatten_obbox(obj_corner):
    """
    Convert obj_corner to a flat list of 8 floats:
    Accepts either [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] or [x1,y1,x2,y2,...]
    Returns list of floats rounded to 6 decimals.
    If input is missing or invalid returns [].
    """
    if not obj_corner:
        return []
    # If nested lists
    try:
        if isinstance(obj_corner, list) and len(obj_corner) == 4 and all(isinstance(p, (list, tuple)) for p in obj_corner):
            flat = []
            for p in obj_corner:
                if len(p) >= 2:
                    flat.extend([float(p[0]), float(p[1])])
                else:
                    # invalid - bail
                    return []
            return [round(float(x), 6) for x in flat]
        # If already flat list
        if isinstance(obj_corner, list) and len(obj_corner) >= 8:
            flat = [float(x) for x in obj_corner[:8]]
            return [round(x, 6) for x in flat]
    except Exception:
        return []
    return []

def convert_json(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    image_id = data["image"]
    image_url = f"http://visionl40.cse.iitd.ac.in:5000/{image_id}"

    objs = data.get("objects", [])
    if objs:
        first_obj = objs[0]
        grounding_instruction = (
            f"Locate and return oriented bounding boxes for the "
            f"{first_obj.get('obj_cls','object')} in the image."
        )
    else:
        grounding_instruction = (
            "Locate and return oriented bounding boxes for objects seen in the image."
        )

    picked = {"binary": None, "numeric": None, "semantic": None}
    for qa in data.get("qa_pairs", []):
        if all(picked.values()):
            break
        q = qa.get("question", "").strip()
        ans = qa.get("answer", "")
        cat = classify_answer(ans)
        if picked[cat] is None:
            picked[cat] = q

    attr_queries = {
        "binary": {"instruction": picked["binary"] or ""},
        "numeric": {"instruction": picked["numeric"] or ""},
        "semantic": {"instruction": picked["semantic"] or ""},
    }

    output = {
        "input_image": {
            "image_id": image_id,
            "image_url": image_url,
            "metadata": {
                "width": 512,
                "height": 512,
                "spatial_resolution_m": 1.57,
            },
        },
        "queries": {
            "caption_query": {
                "instruction": (
                    "Generate a detailed caption describing all visible elements in "
                    "the satellite image, including object types, counts, relative "
                    "locations, and overall scene context."
                )
            },
            "grounding_query": {"instruction": grounding_instruction},
            "attribute_query": attr_queries,
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    return output, data


# ----------------------------
# Helper to draw oriented boxes
# ----------------------------
def draw_boxes_on_image(img, gt_boxes, pred_boxes, save_path, resize_to=None):
    """
    img: PIL.Image
    gt_boxes: list of dicts with keys 'object-id' and 'obbox' (flat 8 floats)
    pred_boxes: same structure
    save_path: path to save annotated image
    resize_to: (w,h) or None
    """
    if resize_to:
        img = img.resize(resize_to)
    
    img_width, img_height = img.size
    draw = ImageDraw.Draw(img)
    
    # Attempt to load a small truetype font; fallback to default
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    # helper to draw a polygon with label and thicker lines
    def draw_poly(points, outline="red", width=2, label=None):
        # points: list of (x,y) - check if normalized (0-1) and scale to pixels
        scaled_points = []
        all_valid = True
        
        for x, y in points:
            # Check if coordinates are normalized (between 0 and 1)
            # if 0 <= x <= 1 and 0 <= y <= 1:
            #     scaled_x = x * img_width
            #     scaled_y = y * img_height
            # else:
                # Assume pixel coordinates
            scaled_x = x
            scaled_y = y
            
            # Check if coordinates are within image bounds (with some tolerance)
            # if scaled_x < -200 or scaled_x > img_width + 200 or scaled_y < -200 or scaled_y > img_height + 200:
            #     all_valid = False
            
            # Clamp coordinates to image bounds
            # scaled_x = max(0, min(img_width - 1, scaled_x))
            # scaled_y = max(0, min(img_height - 1, scaled_y))
            
            scaled_points.append((scaled_x, scaled_y))
        
        if len(scaled_points) < 2:
            return
        
        # Skip drawing if coordinates are way outside bounds (likely invalid)
        if not all_valid:
            print(f"Warning: Box with label '{label}' has coordinates outside image bounds - clamped to edges")
        
        # Draw polygon outline with proper width
        try:
            draw.polygon(scaled_points, outline=outline, width=width)
        except Exception as e:
            print(f"Error drawing polygon for {label}: {e}")
            return
        
        # label near first point
        if label is not None:
            lx, ly = scaled_points[0]
            text = str(label)
            
            # Get text size using textbbox with anchor point
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            
            # Draw background rectangle for readability
            padding = 2
            # Ensure text background stays within image
            text_x = max(0, min(img_width - tw - padding*2, lx))
            text_y = max(th + padding*2, min(img_height, ly))
            
            rect = [text_x, text_y - th - padding*2, text_x + tw + padding*2, text_y]
            draw.rectangle(rect, fill=(0, 0, 0, 180))
            draw.text((text_x + padding, text_y - th - padding), text, fill="white", font=font)
    # Draw GT boxes (green)
    for item in gt_boxes:
        ob = item.get("obbox", []) or item.get("obbox_xy", [])
        if not ob or len(ob) < 8:
            continue
        try:
            pts = [(float(ob[i]), float(ob[i+1])) for i in range(0, 8, 2)]
            draw_poly(pts, outline=(0, 255, 0), width=3, label=f"GT:{item.get('object-id','')}")
        except Exception as e:
            print(f"Error drawing GT box: {e}")
            continue
    
    # Draw predicted boxes (red)
    for item in pred_boxes:
        ob = item.get("obbox", []) or item.get("obbox_xy", [])
        if not ob or len(ob) < 8:
            continue
        try:
            pts = [(float(ob[i]), float(ob[i+1])) for i in range(0, 8, 2)]
            draw_poly(pts, outline=(255, 0, 0), width=3, label=f"P:{item.get('object-id','')}")
        except Exception as e:
            print(f"Error drawing predicted box: {e}")
            continue
    
    # save
    img.save(save_path)

# ----------------------------
# Loop random JSONs (main)
# ----------------------------

ann_dir = "/home/samyak/scratch/interiit/data/VRSBench/Annotations_val/"
save_dir = "eval_outputs"
gt_dir = "ground_truths"
pred_boxes_dir = "pred_boxes"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(gt_dir, exist_ok=True)
os.makedirs(pred_boxes_dir, exist_ok=True)

input_json_dir = "converted_inputs"
os.makedirs(input_json_dir, exist_ok=True)

all_files = [f for f in os.listdir(ann_dir) if f.endswith(".json")]
picked_files = random.sample(all_files, num_files)

iter_num = 1
for fname in tqdm(picked_files):
    start_time = time.time()

    in_path = os.path.join(ann_dir, fname)
    tmp_out_path = os.path.join(input_json_dir, fname)

    payload, original = convert_json(in_path, tmp_out_path)

    grounding_instr = payload["queries"]["grounding_query"]["instruction"]

    cls = None
    if "for the " in grounding_instr:
        try:
            cls = grounding_instr.split("for the ",1)[1].split(" in",1)[0].strip().lower()
        except:
            cls = None
    if not cls:
        objs = original.get("objects", [])
        cls = objs[0].get("obj_cls","").strip().lower() if objs else None

    # select reference objects: prefer those matching cls, else all objects
    ref_objs = []
    for obj in original.get("objects", []):
        if cls and obj.get("obj_cls","").strip().lower() == cls:
            ref_objs.append({
                "obj_id": obj.get("obj_id"),
                "obj_cls": obj.get("obj_cls"),
                "obj_coord": obj.get("obj_coord"),
                "obj_corner": obj.get("obj_corner")
            })
    if not ref_objs:
        for obj in original.get("objects", []):
            ref_objs.append({
                "obj_id": obj.get("obj_id"),
                "obj_cls": obj.get("obj_cls"),
                "obj_coord": obj.get("obj_coord"),
                "obj_corner": obj.get("obj_corner")
            })

    qa_list = original.get("qa_pairs", [])

    def find_answer(instr, category):
        # prefer exact instruction match
        if instr:
            for qa in qa_list:
                if qa.get("question","").strip() == instr:
                    return qa.get("answer","")
        # otherwise fallback to first answer of the requested category
        for qa in qa_list:
            ans = qa.get("answer","")
            if category == "numeric":
                try:
                    int(str(ans).strip())
                    return ans
                except:
                    pass
            elif category == "binary":
                if str(ans).strip().lower() in {"yes","no","true","false"}:
                    return ans
            else:  # semantic
                a = str(ans).strip()
                if a and a.lower() not in {"yes","no","true","false"}:
                    try:
                        int(a)
                    except:
                        return ans
        return ""

    attr_pred = payload["queries"]["attribute_query"]
    gt_attrs = {
        "binary": find_answer(attr_pred["binary"]["instruction"], "binary"),
        "numeric": find_answer(attr_pred["numeric"]["instruction"], "numeric"),
        "semantic": find_answer(attr_pred["semantic"]["instruction"], "semantic"),
    }

    # ----------------------------
    # ADDING CAPTION GROUND TRUTH
    # ----------------------------
    gt_caption = original.get("caption", "")

    # Build grounding response list with the selected objects only
    grounding_response = []
    for obj in ref_objs:
        obbox = flatten_obbox(obj.get("obj_corner"))
        grounding_response.append({
            "object-id": str(obj.get("obj_id", "")),
            "obbox": obbox
        })

    # Build final GT structure in requested format
    ground_truth = {
        "input_image": {
            "image_id": payload["input_image"]["image_id"],
            "image_url": payload["input_image"]["image_url"],
            "metadata": payload["input_image"]["metadata"]
        },
        "queries": {
            "caption_query": {
                "instruction": payload["queries"]["caption_query"]["instruction"],
                "response": gt_caption
            },
            "grounding_query": {
                "instruction": payload["queries"]["grounding_query"]["instruction"],
                "response": grounding_response
            },
            "attribute_query": {
                "binary": {
                    "instruction": attr_pred["binary"]["instruction"],
                    "response": gt_attrs["binary"]
                },
                "numeric": {
                    "instruction": attr_pred["numeric"]["instruction"],
                    "response": gt_attrs["numeric"]
                },
                "semantic": {
                    "instruction": attr_pred["semantic"]["instruction"],
                    "response": gt_attrs["semantic"]
                }
            }
        }
    }

    # Save ground truth file
    gt_path = os.path.join(gt_dir, fname)
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=4)

    # ----------------------------
    # Send payload to eval API and save response
    # ----------------------------
    try:
        response = requests.post(
            "http://52.66.150.171/api/eval",
            json=payload,
            headers={"accept": "application/json", "Content-Type": "application/json"},
            timeout=180
        )
    except Exception as e:
        print(f"Request failed for {fname}: {e}")
        elapsed = time.time() - start_time
        print(f"Image {iter_num} : {fname}: {elapsed:.3f} seconds")
        iter_num += 1
        continue

    out_path = os.path.join(save_dir, fname.replace(".json", "_eval.json"))
    resp_json = None
    try:
        resp_json = response.json()
        with open(out_path, "w") as f:
            json.dump(resp_json, f, indent=4)
    except Exception:
        # fallback, write text
        with open(out_path, "w") as f:
            f.write(response.text)

    # ----------------------------
    # Prepare to draw boxes: fetch image, parse predicted boxes
    # ----------------------------
    image_url = payload["input_image"]["image_url"]
    local_image_filename = os.path.join(pred_boxes_dir, fname.replace(".json", "_raw.png"))
    annotated_image_filename = os.path.join(pred_boxes_dir, fname.replace(".json", "_annotated.png"))

    # Download image
    pil_img = None
    try:
        r = requests.get(image_url, timeout=30)
        r.raise_for_status()
        pil_img = Image.open(BytesIO(r.content)).convert("RGB")
        # save raw copy (optional)
        pil_img.save(local_image_filename)
    except Exception as e:
        print(f"Could not download image {image_url} for {fname}: {e}")
        # skip drawing but continue
        elapsed = time.time() - start_time
        print(f"Image {iter_num} : {fname}: {elapsed:.3f} seconds")
        iter_num += 1
        continue

    # Parse predicted grounding response from the API output
    pred_grounding = []
    # Try multiple likely keys where predictions may be returned
    if isinstance(resp_json, dict):
        # common path: resp_json["queries"]["grounding_query"]["response"]
        try:
            pred_grounding = resp_json.get("queries", {}).get("grounding_query", {}).get("response", [])
            if not pred_grounding:
                # maybe top-level grounding
                pred_grounding = resp_json.get("grounding_response", []) or resp_json.get("predictions", {}).get("grounding", []) or []
        except Exception:
            pred_grounding = []
    else:
        pred_grounding = []

    # If API returned as string or single dict, normalize to list
    if isinstance(pred_grounding, dict):
        pred_grounding = [pred_grounding]
    # Ensure each predicted item uses keys 'object-id' and 'obbox'
    norm_pred = []
    for item in pred_grounding:
        if not isinstance(item, dict):
            continue
        # try common variants
        if "obbox" in item and (isinstance(item["obbox"], list) and len(item["obbox"]) >= 8):
            norm_pred.append({"object-id": item.get("object-id", item.get("id", "")), "obbox": item["obbox"]})
        else:
            # maybe xy pairs under 'polygon' or 'coords'
            for k in ["polygon","coords","points","points_xy"]:
                if k in item:
                    val = item[k]
                    flat = []
                    # if list of points
                    if isinstance(val, list) and len(val) and isinstance(val[0], (list,tuple)):
                        for p in val:
                            if len(p) >= 2:
                                flat.extend([float(p[0]), float(p[1])])
                    elif isinstance(val, list) and len(val) >= 8:
                        flat = [float(x) for x in val[:8]]
                    if flat:
                        norm_pred.append({"object-id": item.get("object-id", item.get("id", "")), "obbox": flat})
                        break
    pred_grounding = norm_pred

    # Build GT boxes list in same format
    gt_boxes = []
    for g in grounding_response:
        gt_boxes.append({"object-id": g.get("object-id", ""), "obbox": g.get("obbox", [])})

    # Draw and save annotated image
    try:
        draw_boxes_on_image(pil_img, gt_boxes, pred_grounding, annotated_image_filename)
    except Exception as e:
        print(f"Failed drawing boxes for {fname}: {e}")

    elapsed = time.time() - start_time
    print(f"Image {iter_num} : {fname}: {elapsed:.3f} seconds")
    iter_num += 1
