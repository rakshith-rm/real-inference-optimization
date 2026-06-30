"""
Baseline vs EAGLE3 vs selective speculation scheduler (vLLM).

Each engine runs in its own subprocess. All modes use one prompt per generate()
call (vLLM 0.11 EAGLE3 hangs on batched spec).

  python gpu_check.py
  python benchmark.py [--quick]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from collections import Counter

import config
from metrics import print_spec_summary, spec_decode_stats
from prompts import QUICK_WORKLOAD, WORKLOAD
from scheduler import Request, SelectiveSpecScheduler, prompt_token_count


def _build_llm(use_speculative: bool):
    from vllm import LLM

    kwargs = {
        "model": config.TARGET_MODEL,
        "tensor_parallel_size": config.TENSOR_PARALLEL_SIZE,
        "gpu_memory_utilization": config.GPU_MEMORY_UTILIZATION,
        "max_num_seqs": 1,
        "trust_remote_code": True,
        "disable_log_stats": False,  # required for spec_decode_stats() / rolling demotion
    }
    if use_speculative:
        kwargs["speculative_config"] = {
            "method": "eagle3",
            "model": config.EAGLE3_MODEL,
            "num_speculative_tokens": config.NUM_SPECULATIVE_TOKENS,
            "draft_tensor_parallel_size": config.DRAFT_TENSOR_PARALLEL_SIZE,
        }
    return LLM(**kwargs)


def _sp(temperature: float, max_tokens: int):
    from vllm import SamplingParams

    return SamplingParams(temperature=temperature, max_tokens=max_tokens)


def _out_tokens(outs) -> int:
    return sum(len(o.outputs[0].token_ids) for o in outs)


def _generate_all(llm, reqs: list[Request]) -> tuple[int, float]:
    tokens, elapsed = 0, 0.0
    for n, (_, prompt, temp, max_tok) in enumerate(reqs, 1):
        t0 = time.perf_counter()
        outs = llm.generate([prompt], _sp(temp, max_tok))
        elapsed += time.perf_counter() - t0
        tokens += _out_tokens(outs)
        if n % 5 == 0 or n == len(reqs):
            print(f"    {n}/{len(reqs)} prompts done", flush=True)
    return tokens, elapsed


def _load_llm(use_speculative: bool, warmup: Request) -> tuple:
    t0 = time.perf_counter()
    llm = _build_llm(use_speculative)
    load_s = time.perf_counter() - t0
    _, prompt, temp, max_tok = warmup
    llm.generate([prompt], _sp(temp, max_tok))
    return llm, load_s


def _run_engine(use_speculative: bool, reqs: list[Request], q) -> None:
    llm, load_s = _load_llm(use_speculative, reqs[0])
    tokens, elapsed = _generate_all(llm, reqs)
    q.put({
        "output_tokens": tokens,
        "elapsed_s": elapsed,
        "load_s": load_s,
        "tokens_per_sec": tokens / elapsed if elapsed else 0.0,
        "sec_per_request": elapsed / len(reqs) if reqs else 0.0,
        "spec_stats": spec_decode_stats(llm, config.NUM_SPECULATIVE_TOKENS) if use_speculative else None,
    })


def _run_spec_phase(reqs: list[Request], q) -> None:
    sched = SelectiveSpecScheduler()
    llm, load_s = _load_llm(True, reqs[0])

    tokens = executed = 0
    elapsed = 0.0
    demoted: list[Request] = []
    logs: list[str] = []
    stats = None
    remaining = list(reqs)
    while remaining:
        req = remaining.pop(0)
        _, prompt, temp, max_tok = req
        t0 = time.perf_counter()
        outs = llm.generate([prompt], _sp(temp, max_tok))
        elapsed += time.perf_counter() - t0
        stats = spec_decode_stats(llm, config.NUM_SPECULATIVE_TOKENS)
        sched.record_acceptance(stats["mean_acceptance_length"])
        tokens += _out_tokens(outs)
        executed += 1
        logs.append(f"  spec {executed}/{len(reqs)}: rolling accept={sched.rolling_accept_rate:.2f}")
        if (
            stats["num_drafts"]
            and sched.has_acceptance_samples
            and sched.rolling_accept_rate < config.SCHED_MIN_ROLLING_ACCEPT_RATE
        ):
            demoted.extend(remaining)
            logs.append(
                f"  rolling accept {sched.rolling_accept_rate:.2f} < "
                f"{config.SCHED_MIN_ROLLING_ACCEPT_RATE} — demoting {len(remaining)} to baseline"
            )
            break

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
    p = ctx.Process(target=target, args=(*args, q))
    p.start()
    result = q.get()
    p.join()
    return result


def _print_results(tokens, elapsed, load_s, n_reqs, spec_stats=None, *, inference_label="wall"):
    print(f"  engine load:     {load_s:.1f}s")
    print(f"  output tokens:   {tokens}  {inference_label}: {elapsed:.2f}s")
    tps = tokens / elapsed if elapsed else 0.0
    print(f"  throughput:      {tps:.1f} tok/s  latency/req: {elapsed / n_reqs:.2f}s")
    print_spec_summary(spec_stats)
    return tps


def run_homogeneous(name: str, use_speculative: bool, reqs: list[Request]) -> dict:
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    r = _spawn(_run_engine, use_speculative, reqs)
    tps = _print_results(r["output_tokens"], r["elapsed_s"], r["load_s"], len(reqs), r.get("spec_stats"))
    return {**r, "tokens_per_sec": tps}


def run_selective(requests) -> dict:
    print(f"\n{'=' * 60}\nselective speculation scheduler\n{'=' * 60}")
    sched = SelectiveSpecScheduler()
    spec, base = [], []
    for idx, req in enumerate(requests):
        use_spec, _ = sched.decide(prompt_token_count(req.prompt), req.temperature)
        entry: Request = (idx, req.prompt, req.temperature, req.max_tokens)
        (spec if use_spec else base).append(entry)
    print(f"  routing: {len(spec)} spec / {len(base)} baseline  {sched.reason_counts()}")

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
        base.extend(sr["demoted"])
    if base:
        br = _spawn(_run_engine, False, base)
        tokens += br["output_tokens"]
        elapsed += br["elapsed_s"]
        load += br["load_s"]

    print(f"  executed:        {executed_spec} spec / {len(base)} baseline")
    tps = _print_results(tokens, elapsed, load, len(requests), stats, inference_label="inference")
    return {
        "output_tokens": tokens,
        "elapsed_s": elapsed,
        "load_s": load,
        "tokens_per_sec": tps,
        "sec_per_request": elapsed / len(requests) if requests else 0.0,
        "executed_spec": executed_spec,
        "executed_baseline": len(base),
        "routing_reasons": sched.reason_counts(),
        "spec_stats": stats,
    }


def print_comparison(baseline: dict, spec: dict, sel: dict) -> None:
    def pct(a, b):
        return (a / b - 1) * 100 if b else 0.0

    b, s, x = baseline["tokens_per_sec"], spec["tokens_per_sec"], sel["tokens_per_sec"]
    print(f"\n{'=' * 60}\nCOMPARISON (batch=1, fair)\n{'=' * 60}")
    print(f"  throughput tok/s:  baseline {b:.1f} | always-spec {s:.1f} ({pct(s, b):+.1f}%) | selective {x:.1f} ({pct(x, b):+.1f}% vs base)")
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
    print(f"Workload: {len(requests)} requests  (batch=1, all modes)")
    print(f"  categories: {dict(Counter(r.category for r in requests))}")

    baseline = run_homogeneous("baseline (no speculative decoding)", False, reqs)
    spec = run_homogeneous("EAGLE3 always-on", True, reqs)
    sel = run_selective(requests)
    print_comparison(baseline, spec, sel)


if __name__ == "__main__":
    main()
