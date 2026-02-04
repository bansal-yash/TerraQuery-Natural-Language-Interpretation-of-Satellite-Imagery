#!/usr/bin/env python3
"""Benchmark helper for LocalVLM inference.

This script mirrors the standalone benchmark but routes requests through
LocalVLM, making it easier to compare end-to-end timings (including prompt
construction, streaming toggles, etc.).
"""

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
VQA_ROOT = ROOT / "vqa"
if str(VQA_ROOT) not in sys.path:
    sys.path.insert(0, str(VQA_ROOT))

from local_vlm import LocalVLM  # noqa: E402


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile LocalVLM answer latency")
    parser.add_argument("--image", required=True, help="Image path to feed into the model")
    parser.add_argument("--question", required=True, help="Question to ask about the image")
    parser.add_argument("--iters", type=int, default=5, help="Number of timed iterations")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs before timing")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Decode budget")
    parser.add_argument("--model", default="unsloth/Qwen3-VL-8B-Instruct", help="Base model name or path")
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--device", default="cuda", help="Device identifier passed to LocalVLM")
    parser.add_argument(
        "--attn-impl",
        choices=["sdpa", "flash_attention_2", "none"],
        default="sdpa",
        help="Attention implementation hint for transformers loader",
    )
    parser.add_argument("--base-adapter", help="Path to a null/base adapter for fast swaps")
    parser.add_argument("--prefer-unsloth", action="store_true", help="Force FastVisionModel backend")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming tokens")
    parser.add_argument("--perf-logs", action="store_true", help="Enable verbose perf logs inside LocalVLM")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attn_impl = None if args.attn_impl == "none" else args.attn_impl

    vlm = LocalVLM(
        model_name=args.model,
        device=args.device,
        stream_thoughts=not args.no_stream,
        precision=args.precision,
        attn_implementation=attn_impl,
        prefer_unsloth=args.prefer_unsloth,
        base_adapter_path=args.base_adapter,
        enable_perf_logs=args.perf_logs,
    )

    def run_once() -> float:
        start = time.perf_counter()
        _ = vlm.answer_question(args.image, args.question, max_length=args.max_new_tokens)
        return time.perf_counter() - start

    print(f"[Profile] Warmup x{args.warmup}")
    for _ in range(max(args.warmup, 0)):
        run_once()

    latencies = []
    tokens = []
    print("[Profile] Measuring …")
    for idx in range(args.iters):
        lat = run_once()
        latencies.append(lat)
        perf = vlm.last_perf or {}
        tokens.append(perf.get("new_tokens", 0.0))
        throughput = perf.get("throughput_tps", 0.0)
        print(
            f"iter {idx + 1}/{args.iters}  {lat * 1000:.1f} ms  new_tokens={tokens[-1]:.0f}  throughput={throughput:.1f} tok/s"
        )

    lat_ms = [x * 1000 for x in latencies]
    print("\n=== LOCALVLM REPORT ===")
    print(f"mean {statistics.mean(lat_ms):.2f} ms  p50 {percentile(lat_ms, 0.5):.2f}  p90 {percentile(lat_ms, 0.9):.2f}")
    print(f"min {min(lat_ms):.2f}  max {max(lat_ms):.2f}")
    total_tokens = sum(tokens)
    total_time = sum(latencies) if latencies else 1e-9
    print(f"throughput: {total_tokens / total_time:.2f} tokens/sec")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
