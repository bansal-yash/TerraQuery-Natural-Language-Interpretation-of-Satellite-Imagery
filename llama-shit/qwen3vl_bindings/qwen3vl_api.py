# qwen3vl_api.py
import argparse
import qwen_mtmd
import sys
import time

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to model gguf")
    p.add_argument("--mmproj", required=True, help="path to mmproj gguf")
    p.add_argument("--image", required=True, help="image path")
    p.add_argument("--prompt", default="describe the image", help="prompt")
    p.add_argument("--n_batch", default=256, type=int, help="requested mtmd batch (will be halved on memory failure)")
    p.add_argument("--max_new_tokens", default=128, type=int)
    args = p.parse_args()

    print("[info] loading model and mmproj (this may take a while)...")
    handle = qwen_mtmd.load(args.model, args.mmproj, -1, 8)  # n_gpu_layers=-1 (auto), threads=8
    print("[info] loaded; running inference... (requested n_batch=%d)" % args.n_batch)
    t0 = time.time()
    try:
        out = qwen_mtmd.infer(handle, args.image, args.prompt, int(args.n_batch), int(args.max_new_tokens))
    except Exception as e:
        print("[error] inference failed:", e)
        sys.exit(1)
    t1 = time.time()
    print("[info] inference done in %.2fs" % (t1 - t0))
    print("=== OUTPUT ===")
    print(out)

    # optionally free handle (module also frees on capsule GC)
    # qwen_mtmd.free_handle(handle)

if __name__ == "__main__":
    main()
