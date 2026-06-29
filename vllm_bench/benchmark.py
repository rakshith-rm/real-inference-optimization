"""
Baseline vs EAGLE3 vs selective speculation scheduler (vLLM).

Default config: TP=1, k=2 (best on A6000). See config.py.

Run:
  python gpu_check.py
  python benchmark.py --quick
  python benchmark.py
"""

from __future__ import annotations

import argparse
import gc
import time
from collections import Counter

import torch
from vllm import LLM, SamplingParams

import config
from metrics import print_spec_decode_report, spec_decode_stats
from prompts import QUICK_WORKLOAD, WORKLOAD, BenchmarkRequest
from scheduler import SelectiveSpecScheduler, prompt_token_count


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


def sampling_params_for(requests: list[BenchmarkRequest]) -> list[SamplingParams]:
    return [
        SamplingParams(temperature=r.temperature, max_tokens=r.max_tokens)
        for r in requests
    ]


def count_output_tokens(outputs) -> int:
    return sum(len(o.outputs[0].token_ids) for o in outputs)


def generate_requests(
    llm: LLM,
    requests: list[BenchmarkRequest],
    *,
    sequential: bool = False,
):
    """Run generation. Sequential mode avoids vLLM 0.11 prometheus spec-metrics crash on batches."""
    prompts = [r.prompt for r in requests]
    params = sampling_params_for(requests)
    if not sequential or len(requests) <= 1:
        return llm.generate(prompts, params)
    outputs = []
    for prompt, param in zip(prompts, params):
        outputs.extend(llm.generate([prompt], param))
    return outputs


def release_gpu() -> None:
    """Best-effort VRAM cleanup between vLLM engine loads."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    time.sleep(5)


def unload_llm(llm: LLM) -> None:
    del llm
    release_gpu()


def run_homogeneous(
    name: str,
    *,
    use_speculative: bool,
    requests: list[BenchmarkRequest],
) -> dict:
    print(f"\n{'=' * 60}")
    print(f"Loading: {name}")
    print(
        f"  target TP={config.TENSOR_PARALLEL_SIZE}, "
        f"draft TP={config.DRAFT_TENSOR_PARALLEL_SIZE if use_speculative else 'n/a'}"
    )
    print(f"{'=' * 60}")

    load_start = time.perf_counter()
    llm = build_llm(use_speculative=use_speculative)
    load_s = time.perf_counter() - load_start
    print(f"Engine ready in {load_s:.1f}s")

    prompts = [r.prompt for r in requests]
    params = sampling_params_for(requests)

    _ = llm.generate([prompts[0]], params[0])

    start = time.perf_counter()
    outputs = generate_requests(llm, requests, sequential=use_speculative)
    elapsed = time.perf_counter() - start

    total_tokens = count_output_tokens(outputs)
    tokens_per_sec = total_tokens / elapsed if elapsed else 0.0
    sec_per_request = elapsed / len(requests) if requests else 0.0

    result = {
        "name": name,
        "prompts": len(requests),
        "output_tokens": total_tokens,
        "elapsed_s": elapsed,
        "tokens_per_sec": tokens_per_sec,
        "sec_per_request": sec_per_request,
        "load_s": load_s,
    }

    print(f"\n--- {name} results ---")
    print(f"  prompts:         {len(requests)}")
    print(f"  output tokens:   {total_tokens}")
    print(f"  wall time:       {elapsed:.2f}s")
    print(f"  throughput:      {tokens_per_sec:.1f} tok/s")
    print(f"  latency/request: {sec_per_request:.2f}s")

    if use_speculative:
        stats = spec_decode_stats(llm, config.NUM_SPECULATIVE_TOKENS)
        print_spec_decode_report(stats)
        result["spec_stats"] = stats

    unload_llm(llm)
    return result


def run_selective(requests: list[BenchmarkRequest]) -> dict:
    """Route each request to baseline or EAGLE3 using the selective scheduler."""
    name = "selective speculation scheduler"
    print(f"\n{'=' * 60}")
    print(f"Loading: {name}")
    print(f"{'=' * 60}")

    scheduler = SelectiveSpecScheduler()
    baseline_queue: list[tuple[int, BenchmarkRequest]] = []
    spec_queue: list[tuple[int, BenchmarkRequest]] = []

    print("\n--- Initial routing (static rules + rolling accept rate) ---")
    for idx, req in enumerate(requests):
        tokens = prompt_token_count(req.prompt)
        decision = scheduler.decide(tokens, req.temperature)
        scheduler.log_decision(decision)
        target = spec_queue if decision.use_speculation else baseline_queue
        target.append((idx, req))
        print(
            f"  [{req.category:18}] tokens={tokens:3d} temp={req.temperature:.2f} "
            f"-> {'SPEC' if decision.use_speculation else 'base':4s} ({decision.reason})"
        )

    outputs_by_idx: dict[int, object] = {}
    total_load_s = 0.0
    total_elapsed_s = 0.0
    spec_stats_agg: dict | None = None
    executed_spec = 0
    executed_baseline = 0

    if spec_queue:
        print(f"\n--- Spec engine: {len(spec_queue)} request(s), chunked adaptive routing ---")
        load_start = time.perf_counter()
        llm = build_llm(use_speculative=True)
        total_load_s += time.perf_counter() - load_start
        print(f"Spec engine ready in {time.perf_counter() - load_start:.1f}s")

        _ = llm.generate([spec_queue[0][1].prompt], sampling_params_for([spec_queue[0][1]])[0])

        remaining = list(spec_queue)
        demoted: list[tuple[int, BenchmarkRequest]] = []

        while remaining:
            if scheduler.has_acceptance_samples and scheduler.rolling_accept_rate < config.SCHED_MIN_ROLLING_ACCEPT_RATE:
                print(
                    f"  Rolling accept rate {scheduler.rolling_accept_rate:.2f} "
                    f"< {config.SCHED_MIN_ROLLING_ACCEPT_RATE} — demoting {len(remaining)} request(s) to baseline"
                )
                demoted.extend(remaining)
                remaining = []
                break

            chunk = remaining[: config.SCHED_SPEC_CHUNK_SIZE]
            remaining = remaining[config.SCHED_SPEC_CHUNK_SIZE :]

            chunk_reqs = [r for _, r in chunk]
            start = time.perf_counter()
            chunk_out = []
            for req in chunk_reqs:
                chunk_out.extend(
                    llm.generate([req.prompt], sampling_params_for([req])[0])
                )
            total_elapsed_s += time.perf_counter() - start

            stats = spec_decode_stats(llm, config.NUM_SPECULATIVE_TOKENS)
            scheduler.record_acceptance(stats["mean_acceptance_length"])
            spec_stats_agg = stats

            for (idx, _), out in zip(chunk, chunk_out):
                outputs_by_idx[idx] = out
            executed_spec += len(chunk)

            print(
                f"  chunk done: rolling accept={scheduler.rolling_accept_rate:.2f}, "
                f"remaining={len(remaining)}"
            )

        baseline_queue.extend(demoted)
        unload_llm(llm)
        print("GPU memory settling before baseline engine load...")
        release_gpu()

    if baseline_queue:
        print(f"\n--- Baseline engine: {len(baseline_queue)} request(s) ---")
        load_start = time.perf_counter()
        llm = build_llm(use_speculative=False)
        total_load_s += time.perf_counter() - load_start
        print(f"Baseline engine ready in {time.perf_counter() - load_start:.1f}s")

        baseline_queue.sort(key=lambda x: x[0])
        base_reqs = [r for _, r in baseline_queue]
        _ = llm.generate([base_reqs[0].prompt], sampling_params_for([base_reqs[0]])[0])

        start = time.perf_counter()
        base_out = llm.generate(
            [r.prompt for r in base_reqs],
            sampling_params_for(base_reqs),
        )
        total_elapsed_s += time.perf_counter() - start

        for (idx, _), out in zip(baseline_queue, base_out):
            outputs_by_idx[idx] = out
        executed_baseline = len(baseline_queue)

        unload_llm(llm)

    ordered_outputs = [outputs_by_idx[i] for i in range(len(requests))]
    total_tokens = count_output_tokens(ordered_outputs)
    tokens_per_sec = total_tokens / total_elapsed_s if total_elapsed_s else 0.0
    sec_per_request = total_elapsed_s / len(requests) if requests else 0.0

    routed_spec_initial = sum(1 for d in scheduler.routing_log if d.use_speculation)
    routed_base_initial = len(scheduler.routing_log) - routed_spec_initial
    reasons = scheduler.reason_counts()

    result = {
        "name": name,
        "prompts": len(requests),
        "output_tokens": total_tokens,
        "elapsed_s": total_elapsed_s,
        "tokens_per_sec": tokens_per_sec,
        "sec_per_request": sec_per_request,
        "load_s": total_load_s,
        "routed_spec": routed_spec_initial,
        "routed_base": routed_base_initial,
        "executed_spec": executed_spec,
        "executed_baseline": executed_baseline,
        "routing_reasons": dict(reasons),
        "final_rolling_accept_rate": scheduler.rolling_accept_rate,
    }
    if spec_stats_agg:
        result["spec_stats"] = spec_stats_agg

    print(f"\n--- {name} results ---")
    print(f"  prompts:         {len(requests)}")
    print(f"  routed spec (initial): {routed_spec_initial}")
    print(f"  routed baseline (initial): {routed_base_initial}")
    print(f"  executed on spec engine: {executed_spec}")
    print(f"  executed on baseline engine: {executed_baseline}")
    print(f"  routing reasons: {dict(reasons)}")
    print(f"  output tokens:   {total_tokens}")
    print(f"  inference time:  {total_elapsed_s:.2f}s  (excludes engine reload)")
    print(f"  engine load:     {total_load_s:.1f}s")
    print(f"  throughput:      {tokens_per_sec:.1f} tok/s")
    print(f"  latency/request: {sec_per_request:.2f}s")
    if spec_stats_agg:
        print_spec_decode_report(spec_stats_agg)

    return result


def print_comparison(baseline: dict, always_spec: dict, selective: dict) -> None:
    spec_gain = (always_spec["tokens_per_sec"] / baseline["tokens_per_sec"] - 1) * 100
    sel_gain = (selective["tokens_per_sec"] / baseline["tokens_per_sec"] - 1) * 100
    sel_vs_spec = (selective["tokens_per_sec"] / always_spec["tokens_per_sec"] - 1) * 100

    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")
    print(
        f"  Throughput (tok/s):  baseline {baseline['tokens_per_sec']:.1f} | "
        f"always-spec {always_spec['tokens_per_sec']:.1f} ({spec_gain:+.1f}%) | "
        f"selective {selective['tokens_per_sec']:.1f} ({sel_gain:+.1f}% vs base, {sel_vs_spec:+.1f}% vs always-spec)"
    )
    print(
        f"  Latency/request (s): baseline {baseline['sec_per_request']:.2f} | "
        f"always-spec {always_spec['sec_per_request']:.2f} | "
        f"selective {selective['sec_per_request']:.2f}"
    )
    if "spec_stats" in always_spec:
        print(f"  Always-spec mean accept length: {always_spec['spec_stats']['mean_acceptance_length']:.2f}")
    print(f"  Selective routing: {selective['executed_spec']} spec / {selective['executed_baseline']} baseline (executed)")
    print(f"  Selective routing reasons: {selective['routing_reasons']}")
    print()
    print("Resume line examples:")
    print(
        f'  "Implemented EAGLE3 speculative decoding in vLLM on {config.HARDWARE_LABEL} '
        f"(TP={config.TENSOR_PARALLEL_SIZE}), improving decode throughput by "
        f'{max(spec_gain, 0):.0f}% with mean acceptance length '
        f'{always_spec.get("spec_stats", {}).get("mean_acceptance_length", 0):.2f}."'
    )
    print(
        f'  "Designed a runtime scheduler that selectively enabled speculative decoding '
        f"based on prompt length, sampling temperature, and rolling draft acceptance rate, "
        f'routing {selective["executed_spec"]}/{selective["prompts"]} requests to EAGLE3."'
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use a small mixed workload (7 prompts)")
    args = parser.parse_args()

    requests = QUICK_WORKLOAD if args.quick else WORKLOAD

    if torch.cuda.device_count() < config.TENSOR_PARALLEL_SIZE:
        raise SystemExit(
            f"Need {config.TENSOR_PARALLEL_SIZE} GPU(s) for TP={config.TENSOR_PARALLEL_SIZE}."
        )

    print(f"Workload: {len(requests)} requests")
    print(f"  model: {config.TARGET_MODEL}")
    cats = Counter(r.category for r in requests)
    print(f"  categories: {dict(cats)}")

    baseline = run_homogeneous("baseline (no speculative decoding)", use_speculative=False, requests=requests)
    always_spec = run_homogeneous("EAGLE3 always-on", use_speculative=True, requests=requests)
    selective = run_selective(requests)
    print_comparison(baseline, always_spec, selective)


if __name__ == "__main__":
    main()
