"""
Baseline vs EAGLE3 speculative decoding on a 2-GPU node (A6000/A100/etc).

What uses both GPUs:
  - Target model: tensor_parallel_size=2  -> layers split across GPU 0 & 1
  - EAGLE3 draft: draft_tensor_parallel_size=1 -> runs on GPU 0
    (GPU 1 still busy verifying draft tokens via the TP target model)

Run:
  python gpu_check.py          # sanity check
  python gpu_watch.py          # optional, second terminal
  python benchmark.py
  python benchmark.py --quick  # 2 prompts, for a fast smoke test
"""

from __future__ import annotations

import argparse
import gc
import time

import torch
from vllm import LLM, SamplingParams

import config
from metrics import print_spec_decode_report, spec_decode_stats
from prompts import PROMPTS


def build_llm(*, use_speculative: bool) -> LLM:
    kwargs: dict = {
        "model": config.TARGET_MODEL,
        "tensor_parallel_size": config.TENSOR_PARALLEL_SIZE,
        "gpu_memory_utilization": config.GPU_MEMORY_UTILIZATION,
        "trust_remote_code": True,
        "disable_log_stats": False,
    }
    if use_speculative:
        kwargs["speculative_config"] = {
            "method": "eagle3",
            "model": config.EAGLE3_MODEL,
            "num_speculative_tokens": config.NUM_SPECULATIVE_TOKENS,
            "draft_tensor_parallel_size": config.DRAFT_TENSOR_PARALLEL_SIZE,
        }
    return LLM(**kwargs)


def run_mode(name: str, use_speculative: bool, prompts: list[str]) -> dict:
    print(f"\n{'=' * 60}")
    print(f"Loading: {name}")
    print(f"  target TP={config.TENSOR_PARALLEL_SIZE}, draft TP={config.DRAFT_TENSOR_PARALLEL_SIZE if use_speculative else 'n/a'}")
    print(f"{'=' * 60}")

    load_start = time.perf_counter()
    llm = build_llm(use_speculative=use_speculative)
    load_s = time.perf_counter() - load_start
    print(f"Engine ready in {load_s:.1f}s")

    sampling = SamplingParams(
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_OUTPUT_TOKENS,
    )

    # Warmup — first batch includes CUDA graph capture
    _ = llm.generate([prompts[0]], sampling)

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.perf_counter() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tokens_per_sec = total_tokens / elapsed if elapsed else 0.0
    sec_per_request = elapsed / len(prompts)

    result = {
        "name": name,
        "prompts": len(prompts),
        "output_tokens": total_tokens,
        "elapsed_s": elapsed,
        "tokens_per_sec": tokens_per_sec,
        "sec_per_request": sec_per_request,
        "load_s": load_s,
    }

    print(f"\n--- {name} results ---")
    print(f"  prompts:         {len(prompts)}")
    print(f"  output tokens:   {total_tokens}")
    print(f"  wall time:       {elapsed:.2f}s")
    print(f"  throughput:      {tokens_per_sec:.1f} tok/s")
    print(f"  latency/request: {sec_per_request:.2f}s")

    if use_speculative:
        stats = spec_decode_stats(llm, config.NUM_SPECULATIVE_TOKENS)
        print_spec_decode_report(stats)
        result["spec_stats"] = stats

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)  # let NCCL workers exit cleanly before next engine

    return result


def print_comparison(baseline: dict, spec: dict) -> None:
    tp_gain = (spec["tokens_per_sec"] / baseline["tokens_per_sec"] - 1) * 100
    lat_gain = (1 - spec["sec_per_request"] / baseline["sec_per_request"]) * 100

    print(f"\n{'=' * 60}")
    print("COMPARISON (speculative vs baseline)")
    print(f"{'=' * 60}")
    print(f"  Throughput:  {baseline['tokens_per_sec']:.1f} -> {spec['tokens_per_sec']:.1f} tok/s  ({tp_gain:+.1f}%)")
    print(f"  Latency/req: {baseline['sec_per_request']:.2f}s -> {spec['sec_per_request']:.2f}s  ({lat_gain:+.1f}%)")
    if "spec_stats" in spec:
        print(f"  Mean accept length: {spec['spec_stats']['mean_acceptance_length']:.2f}")
    print()
    print("Resume line example:")
    print(
        f'  "Implemented EAGLE3 speculative decoding in vLLM on {config.HARDWARE_LABEL} '
        f"(TP={config.TENSOR_PARALLEL_SIZE}), improving decode throughput by "
        f'{max(tp_gain, 0):.0f}% with mean acceptance length '
        f'{spec.get("spec_stats", {}).get("mean_acceptance_length", 0):.2f}."'
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use 2 prompts only")
    args = parser.parse_args()

    prompts = PROMPTS[:2] if args.quick else PROMPTS

    if torch.cuda.device_count() < config.TENSOR_PARALLEL_SIZE:
        raise SystemExit(
            f"Need {config.TENSOR_PARALLEL_SIZE} GPU(s) for TP={config.TENSOR_PARALLEL_SIZE}. "
            "Run: python gpu_check.py"
        )

    baseline = run_mode("baseline (no speculative decoding)", False, prompts)
    spec = run_mode("EAGLE3 speculative decoding", True, prompts)
    print_comparison(baseline, spec)


if __name__ == "__main__":
    main()
