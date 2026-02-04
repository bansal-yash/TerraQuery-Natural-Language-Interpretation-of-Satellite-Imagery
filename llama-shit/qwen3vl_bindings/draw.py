from PIL import Image, ImageDraw, ImageFont

def normalized_to_pixels(box, img_w, img_h):
    x1 = int(round(box["x1"] * img_w))
    y1 = int(round(box["y1"] * img_h))
    x2 = int(round(box["x2"] * img_w))
    y2 = int(round(box["y2"] * img_h))
    # small safety clamp
    x1 = max(0, min(img_w-1, x1))
    x2 = max(0, min(img_w-1, x2))
    y1 = max(0, min(img_h-1, y1))
    y2 = max(0, min(img_h-1, y2))
    return {"label": box["label"], "x1": x1, "y1": y1, "x2": x2, "y2": y2}

def draw_and_save_annotation(image_path, boxes_norm, out_path="output_annotated.png", label_font_size=18):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", label_font_size)
    except Exception:
        font = ImageFont.load_default()

    # draw semi-transparent filled rectangle behind labels at bottom if many boxes
    for b in boxes_norm:
        pb = normalized_to_pixels(b, w, h)
        # box style
        draw.rectangle([pb["x1"], pb["y1"], pb["x2"], pb["y2"]], outline=(255, 0, 0, 255), width=2)
        # label background
        label = pb["label"] if pb.get("label") else "obj"
        # robust text-size measurement: prefer draw.textbbox, fall back to older APIs
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            try:
                tw, th = draw.textsize(label, font=font)
            except Exception:
                try:
                    tw, th = font.getsize(label)
                except Exception:
                    tw = int(len(label) * (label_font_size * 0.6))
                    th = label_font_size

        lx = pb["x1"]
        ly = max(0, pb["y1"] - th - 4)
        draw.rectangle([lx, ly, lx + tw + 6, ly + th + 4], fill=(255,0,0,180))
        draw.text((lx+3, ly+2), label, fill=(255,255,255,255), font=font)

    img.save(out_path)
    return out_path

draw_and_save_annotation("/home/samyak/scratch/interiit/skySkript_mini/images/a1000070447_US_21.jpg", [{"label": "ground track","x1": 0.437,"y1":0.752,"x2":0.700,"y2":0.875},])
# 0,0,312,550