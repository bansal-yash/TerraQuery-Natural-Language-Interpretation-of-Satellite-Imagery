# example_chat_three_images_run.py
import argparse
import qwen_mtmd
from PIL import Image, ImageDraw, ImageFont

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--mmproj", required=True)
    p.add_argument("--img1", required=True)
    # p.add_argument("--img2", required=True)
    p.add_argument("--n_batch", type=int, default=64)
    p.add_argument("--max_new_tokens", type=int, default=256)
    args = p.parse_args()

    handle = qwen_mtmd.load(args.model, args.mmproj, -1, 8, True)

    messages = [
        {"role":"system", "content":[{"type":"text","text":"You are Qwen3-VL, a helpful multimodal assistant. I wil provide a SAR image and you must try to anyalze it carefully."}]},
        {"role":"user", "content":[
            {"type":"image","image": args.img1},
            # {"type":"image","image": args.img2},
            {"type":"text","text":"Look at this SAR image carefully and provide a detailed analysis of the scene, identifying key features, objects, and any notable patterns you observe."}
        ]}
    ]

    reply = qwen_mtmd.infer_chat(handle, messages, args.n_batch, args.max_new_tokens)
    print("=== RAW REPLY ===")
    print(reply)

    # parse per-image outputs
    parts = reply.split("<|combined_summary|>")
    pre = parts[0]
    summary = parts[1].strip() if len(parts) > 1 else ""
    descs = []
    for seg in pre.split("<|image_description_"):
        if not seg.strip(): continue
        # seg starts with 'i|>' like '1|>\nDESCRIPTION\n\n...'
        idx_end = seg.find("|>")
        if idx_end == -1: continue
        rest = seg[idx_end+2:].strip()
        descs.append(rest.splitlines()[0:10])  # take first lines

    # Simple save: overlay the per-image descriptions onto copies of images
    imgs = [args.img1]
    for i, imgpath in enumerate(imgs):
        img = Image.open(imgpath).convert("RGB")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except:
            font = ImageFont.load_default()
        text = " ".join(descs[i]) if i < len(descs) else "No description"
        draw.rectangle([0, img.height-120, img.width, img.height], fill=(0,0,0,160))
        draw.text((8, img.height-112), text, fill=(255,255,255), font=font)
        outname = f"annotated_{i+1}.png"
        img.save(outname)
        print("Saved", outname)

    print("\nCombined summary:\n", summary)

    qwen_mtmd.free_handle(handle)

if __name__ == "__main__":
    main()
