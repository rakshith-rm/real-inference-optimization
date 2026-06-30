"""
Baseline vs EAGLE3 vs selective speculation scheduler (vLLM).

Each engine runs in its own subprocess (GPU memory freed on exit).
All three modes generate in the same small batched chunks (GENERATE_CHUNK_SIZE).

Run:
  python gpu_check.py
  python benchmark.py --quick
  python benchmark.py
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from collections import Counter

import config
from prompts import QUICK_WORKLOAD, WORKLOAD
from scheduler import SelectiveSpecScheduler, prompt_token_count


def _build_llm(use_speculative: bool):
    from vllm import LLM

    kwargs = {
        "model": config.TARGET_MODEL,
        "tensor_parallel_size": config.TENSOR_PARALLEL_SIZE,
        "gpu_memory_utilization": config.GPU_MEMORY_UTILIZATION,
        "max_num_seqs": config.GENERATE_CHUNK_SIZE,
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


def _params(reqs: list[tuple]):
    from vllm import SamplingParams

    return [SamplingParams(temperature=t, max_tokens=m) for _, _, t, m in reqs]


def _tokens(outs) -> int:
    return sum(len(o.outputs[0].token_ids) for o in outs)


def _iter_batches(reqs: list[tuple]):
    """Chunk by size, never mixing temp bands (EAGLE3 hangs on mixed temp in one batch)."""
    chunk = config.GENERATE_CHUNK_SIZE
    cutoff = config.SCHED_MAX_SPEC_TEMPERATURE
    low = [r for r in reqs if r[2] <= cutoff]
    high = [r for r in reqs if r[2] > cutoff]
    for label, group in (("low-temp", low), ("high-temp", high)):
        for i in range(0, len(group), chunk):
            yield label, group[i : i + chunk]


def _batched_generate(llm, reqs: list[tuple]) -> tuple[int, float]:
    total_tokens = 0
    elapsed = 0.0
    batches = list(_iter_batches(reqs))
    for n, (label, batch) in enumerate(batches, 1):
        t0 = time.perf_counter()
        outs = llm.generate([p for _, p, _, _ in batch], _params(batch))
        elapsed += time.perf_counter() - t0
        total_tokens += _tokens(outs)
        print(f"    batch {n}/{len(batches)} ({label}, {len(batch)} prompts)", flush=True)
    return total_tokens, elapsed


def _run_engine(use_speculative: bool, reqs: list[tuple], q) -> None:
    from metrics import spec_decode_stats

    t0 = time.perf_counter()
    llm = _build_llm(use_speculative)
    load_s = time.perf_counter() - t0

    llm.generate([reqs[0][1]], _params(reqs[:1])[0])  # warmup

    tokens, elapsed = _batched_generate(llm, reqs)
    result = {
        "output_tokens": tokens,
        "elapsed_s": elapsed,
        "load_s": load_s,
        "tokens_per_sec": tokens / elapsed if elapsed else 0.0,
        "sec_per_request": elapsed / len(reqs) if reqs else 0.0,
    }
    if use_speculative:
        result["spec_stats"] = spec_decode_stats(llm, config.NUM_SPECULATIVE_TOKENS)
    q.put(result)


def _run_spec_phase(reqs: list[tuple], q) -> None:
    from metrics import spec_decode_stats

    sched = SelectiveSpecScheduler()
    t0 = time.perf_counter()
    llm = _build_llm(True)
    load_s = time.perf_counter() - t0
    llm.generate([reqs[0][1]], _params(reqs[:1])[0])

    tokens = executed = 0
    elapsed = 0.0
    demoted, logs, stats = [], [], None
    remaining = list(reqs)
    while remaining:
        if sched.has_acceptance_samples and sched.rolling_accept_rate < config.SCHED_MIN_ROLLING_ACCEPT_RATE:
            demoted = [idx for idx, _, _, _ in remaining]
            logs.append(
                f"  rolling accept {sched.rolling_accept_rate:.2f} < "
                f"{config.SCHED_MIN_ROLLING_ACCEPT_RATE} — demoting {len(remaining)} to baseline"
            )
            break
        chunk = remaining[: config.SCHED_SPEC_CHUNK_SIZE]
        remaining = remaining[config.SCHED_SPEC_CHUNK_SIZE :]
        t1 = time.perf_counter()
        outs = llm.generate([p for _, p, _, _ in chunk], _params(chunk))
        elapsed += time.perf_counter() - t1
        stats = spec_decode_stats(llm, config.NUM_SPECULATIVE_TOKENS)
        sched.record_acceptance(stats["mean_acceptance_length"])
        tokens += _tokens(outs)
        executed += len(chunk)
        logs.append(f"  chunk done: rolling accept={sched.rolling_accept_rate:.2f}, remaining={len(remaining)}")

    q.put({
        "output_tokens": tokens,
        "executed": executed,
        "demoted": demoted,
        "elapsed_s": elapsed,
        "load_s": load_s,
        "spec_stats": stats,
        "logs": logs,
    })


def _spawn(target, *args):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=target, args=(*args, q))
    proc.start()
    result = q.get()
    proc.join()
    return result


def _print_spec(stats: dict | None) -> None:
    if not stats:
        return
    print("\n--- Speculative decoding stats ---")
    print(f"  mean accept len: {stats['mean_acceptance_length']:.2f}  draft accept %: {stats['draft_accept_rate'] * 100:.1f}")
    for i, c in enumerate(stats["acceptance_by_pos"]):
        rate = c / stats["num_drafts"] if stats["num_drafts"] else 0.0
        print(f"  pos {i} accept rate: {rate * 100:.1f}%")


def run_homogeneous(name: str, use_speculative: bool, reqs: list[tuple]) -> dict:
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    r = _spawn(_run_engine, use_speculative, reqs)
    print(f"  engine load:     {r['load_s']:.1f}s")
    print(f"  output tokens:   {r['output_tokens']}  wall: {r['elapsed_s']:.2f}s")
    print(f"  throughput:      {r['tokens_per_sec']:.1f} tok/s  latency/req: {r['sec_per_request']:.2f}s")
    _print_spec(r.get("spec_stats"))
    return r


def run_selective(requests) -> dict:
    print(f"\n{'=' * 60}\nselective speculation scheduler\n{'=' * 60}")
    sched = SelectiveSpecScheduler()
    spec, base = [], []
    for idx, req in enumerate(requests):
        decision = sched.decide(prompt_token_count(req.prompt), req.temperature)
        sched.log_decision(decision)
        entry = (idx, req.prompt, req.temperature, req.max_tokens)
        (spec if decision.use_speculation else base).append(entry)
    routed_spec = sum(1 for d in sched.routing_log if d.use_speculation)
    print(f"  routing: {routed_spec} spec / {len(sched.routing_log) - routed_spec} baseline  {dict(sched.reason_counts())}")

    tokens = elapsed = load = 0.0
    executed_spec = 0
    stats = None
    if spec:
        sr = _spawn(_run_spec_phase, spec)
        for line in sr["logs"]:
            print(line)
        tokens += sr["output_tokens"]
        elapsed += sr["elapsed_s"]
        load += sr["load_s"]
        executed_spec = sr["executed"]
        stats = sr["spec_stats"]
        base += [(i, requests[i].prompt, requests[i].temperature, requests[i].max_tokens) for i in sr["demoted"]]
    if base:
        br = _spawn(_run_engine, False, base)
        tokens += br["output_tokens"]
        elapsed += br["elapsed_s"]
        load += br["load_s"]

    tps = tokens / elapsed if elapsed else 0.0
    r = {
        "output_tokens": tokens,
        "elapsed_s": elapsed,
        "load_s": load,
        "tokens_per_sec": tps,
        "sec_per_request": elapsed / len(requests) if requests else 0.0,
        "executed_spec": executed_spec,
        "executed_baseline": len(base),
        "routing_reasons": dict(sched.reason_counts()),
        "spec_stats": stats,
    }
    print(f"  executed:        {executed_spec} spec / {len(base)} baseline")
    print(f"  output tokens:   {tokens}  inference: {elapsed:.2f}s  load: {load:.1f}s")
    print(f"  throughput:      {tps:.1f} tok/s  latency/req: {r['sec_per_request']:.2f}s")
    _print_spec(stats)
    return r


def print_comparison(baseline: dict, spec: dict, sel: dict) -> None:
    def gain(a, b):
        return (a / b - 1) * 100 if b else 0.0

    print(f"\n{'=' * 60}\nCOMPARISON (chunk={config.GENERATE_CHUNK_SIZE})\n{'=' * 60}")
    print(
        f"  throughput tok/s:  baseline {baseline['tokens_per_sec']:.1f} | "
        f"always-spec {spec['tokens_per_sec']:.1f} ({gain(spec['tokens_per_sec'], baseline['tokens_per_sec']):+.1f}%) | "
        f"selective {sel['tokens_per_sec']:.1f} ({gain(sel['tokens_per_sec'], baseline['tokens_per_sec']):+.1f}% vs base)"
    )
    if spec.get("spec_stats"):
        print(f"  always-spec mean accept length: {spec['spec_stats']['mean_acceptance_length']:.2f}")
    print(f"  selective routed: {sel['executed_spec']} spec / {sel['executed_baseline']} baseline  {sel['routing_reasons']}")


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small 7-prompt workload")
    args = parser.parse_args()

    if torch.cuda.device_count() < config.TENSOR_PARALLEL_SIZE:
        raise SystemExit(f"Need {config.TENSOR_PARALLEL_SIZE} GPU(s). Run: python gpu_check.py")

    requests = QUICK_WORKLOAD if args.quick else WORKLOAD
    reqs = [(i, r.prompt, r.temperature, r.max_tokens) for i, r in enumerate(requests)]
    print(f"Workload: {len(requests)} requests  chunk={config.GENERATE_CHUNK_SIZE}")
    print(f"  categories: {dict(Counter(r.category for r in requests))}")

    baseline = run_homogeneous("baseline (no speculative decoding)", False, reqs)
    spec = run_homogeneous("EAGLE3 always-on", True, reqs)
    sel = run_selective(requests)
    print_comparison(baseline, spec, sel)


if __name__ == "__main__":
    main()
