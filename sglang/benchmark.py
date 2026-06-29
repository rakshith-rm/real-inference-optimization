"""
Baseline vs EAGLE3 (static) vs EAGLE3 (runtime-adaptive) on Llama-3.1-8B (SGLang).

SGLang runs as an HTTP server (launched as a subprocess in server.py). Unlike
vLLM, it can change the draft length (speculative_num_steps) AT RUNTIME from an
EMA of acceptance. --adaptive adds that third mode and compares it to static.

Run:
  python gpu_check.py
  python benchmark.py                 # baseline vs static EAGLE3
  python benchmark.py --adaptive      # + runtime-adaptive mode
  python benchmark.py --quick         # 2 prompts (smoke test)
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import requests

import config
from metrics import print_spec_decode_report, server_spec_stats
from prompts import PROMPTS
from server import SGLangServer, base_url

MODE_LABELS = {
    "baseline": "baseline (no speculative decoding)",
    "eagle3": f"EAGLE3 static (k={config.SPEC_NUM_STEPS})",
    "adaptive": "EAGLE3 adaptive (runtime draft length)",
}


def _chat(prompt: str) -> dict:
    t0 = time.perf_counter()
    r = requests.post(f"{base_url()}/v1/chat/completions", timeout=600, json={
        "model": config.TARGET_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.TEMPERATURE,
        "max_tokens": config.MAX_OUTPUT_TOKENS,
    })
    r.raise_for_status()
    return {
        "latency": time.perf_counter() - t0,
        "completion_tokens": r.json().get("usage", {}).get("completion_tokens", 0),
    }


def run_load(prompts: list[str]) -> dict:
    _chat(prompts[0])  # warmup
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        results = list(pool.map(_chat, prompts))
    elapsed = time.perf_counter() - start
    total_tokens = sum(r["completion_tokens"] for r in results)
    return {
        "elapsed_s": elapsed,
        "output_tokens": total_tokens,
        "tokens_per_sec": total_tokens / elapsed if elapsed else 0.0,
        "mean_latency_s": sum(r["latency"] for r in results) / len(results),
        "prompts": len(prompts),
    }


def run_mode(mode: str, prompts: list[str]) -> dict:
    name = MODE_LABELS[mode]
    print(f"\n{'=' * 64}\nLoading: {name}\n{'=' * 64}")
    with SGLangServer(mode):
        result = run_load(prompts)
        if mode != "baseline":
            result["spec_stats"] = server_spec_stats(base_url())
    result["name"] = name

    print(f"\n--- {name} results ---")
    print(f"  prompts:       {result['prompts']}")
    print(f"  output tokens: {result['output_tokens']}")
    print(f"  wall time:     {result['elapsed_s']:.2f}s")
    print(f"  throughput:    {result['tokens_per_sec']:.1f} tok/s")
    print(f"  mean latency:  {result['mean_latency_s']:.2f}s")
    if "spec_stats" in result:
        print_spec_decode_report(result["spec_stats"])
    return result


def print_comparison(results: dict[str, dict]) -> None:
    base = results.get("baseline")
    base_tps = base["tokens_per_sec"] if base else None

    print(f"\n{'=' * 64}\nCOMPARISON\n{'=' * 64}")
    print(f"  {'mode':<28} {'tok/s':>8} {'vs base':>9} {'accept_len':>11}")
    print(f"  {'-' * 28} {'-' * 8} {'-' * 9} {'-' * 11}")
    for mode in ("baseline", "eagle3", "adaptive"):
        r = results.get(mode)
        if not r:
            continue
        gain = f"{(r['tokens_per_sec'] / base_tps - 1) * 100:+.1f}%" if base_tps else "  n/a"
        al = r.get("spec_stats", {}).get("mean_acceptance_length")
        al_str = f"{al:.2f}" if al is not None else "n/a"
        print(f"  {r['name']:<28} {r['tokens_per_sec']:>8.1f} {gain:>9} {al_str:>11}")

    static, adaptive = results.get("eagle3"), results.get("adaptive")
    if static and adaptive:
        delta = (adaptive["tokens_per_sec"] / static["tokens_per_sec"] - 1) * 100
        print(f"\n  Adaptive vs static: {delta:+.1f}%  "
              f"({'adaptive wins' if delta > 0 else 'static wins'})")
        final_k = adaptive.get("spec_stats", {}).get("current_num_steps")
        if final_k is not None:
            print(f"  Adaptive settled at draft steps = {final_k}")

    best = max((r for r in results.values() if r.get("spec_stats")),
              key=lambda r: r["tokens_per_sec"], default=None)
    if best and base:
        best_gain = (best["tokens_per_sec"] / base["tokens_per_sec"] - 1) * 100
        print("\nResume line example:")
        print(f'  "Benchmarked EAGLE3 speculative decoding in SGLang on '
              f'{config.HARDWARE_LABEL} with Llama-3.1-8B, improving decode '
              f'throughput by {max(best_gain, 0):.0f}% over baseline."')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use 2 prompts only")
    parser.add_argument("--adaptive", action="store_true",
                        help="Also run runtime-adaptive draft length")
    args = parser.parse_args()

    prompts = PROMPTS[:2] if args.quick else PROMPTS
    modes = ["baseline", "eagle3"] + (["adaptive"] if args.adaptive else [])

    results = {mode: run_mode(mode, prompts) for mode in modes}
    print_comparison(results)


if __name__ == "__main__":
    main()
