# Real Inference Optimization

A **selective speculation scheduler** for vLLM that routes each request to baseline decode or EAGLE3 speculative decoding based on prompt length, sampling temperature, and rolling draft acceptance.

Validated on **Llama-3.1-8B** + **EAGLE3** (k=2) with vLLM 0.11 on 1× NVIDIA A6000 (48GB), TP=1, against always-on speculation and a no-spec baseline.

## Validation results

Mixed workload (35 requests: short, high-temp, long analytical, long context). Generation at batch=1 for fair comparison and stable EAGLE3 on vLLM 0.11.

| Mode | Throughput | vs baseline | Inference | Tokens |
|------|------------|-------------|-----------|--------|
| Baseline | 41.1 tok/s | — | 167s | 6,859 |
| Always EAGLE3 | 50.5 tok/s | **+22.8%** | 139s | 7,012 |
| Selective scheduler | 43.7 tok/s | **+6.5%** | 157s | 6,859 |

**Always-on spec:** mean accept length 1.59, draft accept 29.7%  
**Selective spec (6 routed requests):** mean accept length 1.75, draft accept 37.3%  
**Routing:** 6 spec / 29 baseline — `short_prompt: 29`, `spec_enabled: 6`

### What we learned

- Always-on EAGLE3 delivers ~23% higher throughput than baseline on this stack at low batch / TP=1.
- The scheduler improves over baseline with conservative defaults, but trails always-on because only **6/35** requests met the routing criteria (long context, prompt ≥64 tokens).
- Routed traffic shows **higher acceptance** (1.75 vs 1.59) — the policy selects well; widening thresholds to match production traffic should close the gap with always-on.
- Evaluation ran cleanly (no hangs, no OOM). Each engine phase uses a **subprocess** so GPU memory is released between baseline and spec runs.

## Selective scheduler

### Design goal

Always-on speculation pays draft-model cost on every request — including short prompts and high-temperature traffic where acceptance is poor. This scheduler **routes away** from those requests and enables EAGLE3 only where drafts are likely to pay off.

Used with thresholds tuned to your workload, it can sit as a **default routing layer in a serving engine**: one target model, optional spec path, per-request decision. You keep most of the always-on speedup without wasting draft work on requests that will never benefit.

Current defaults are intentionally conservative (prompt-length gate dominates), which explains the throughput gap vs always-on. The higher accept rate on routed requests is the signal to tune before rollout.

### Routing rules (`config.py`)

| Check | Parameter | Value | Effect |
|-------|-----------|-------|--------|
| Prompt too short | `SCHED_MIN_PROMPT_TOKENS` | **64** | Baseline if input under 64 tokens |
| Temperature too high | `SCHED_MAX_SPEC_TEMPERATURE` | **0.8** | Baseline if `temperature > 0.8` |
| Rolling accept too low | `SCHED_MIN_ROLLING_ACCEPT_RATE` | **1.2** | Demote remaining requests if mean accept length over the last window falls below 1.2 |
| Rolling window | `SCHED_ROLLING_WINDOW` | **8** | Last 8 spec requests for rolling accept rate |
| Initial accept (no samples) | `SCHED_INITIAL_ACCEPT_RATE` | **2.0** | Prior before first measured accept length |

**Decision order:** short prompt → high temperature → low rolling accept → otherwise **spec enabled**.

vLLM locks `speculative_config` at engine init, so validation runs spec and baseline traffic on **separate engine subprocesses** (production would use a gateway or dual-worker setup for the same effect).

### Tuning for production

Lower `SCHED_MIN_PROMPT_TOKENS` or factor in output length / request class to route more decode-heavy traffic through spec. Raise `SCHED_MAX_SPEC_TEMPERATURE` only after measuring acceptance on creative endpoints. Set `SCHED_MIN_ROLLING_ACCEPT_RATE` from live `mean acceptance length` — demotion guards against draft drift when traffic shifts.

## Comparison modes

The evaluation harness runs three configurations in sequence:

1. **Baseline** — Llama-3.1-8B-Instruct, no speculation  
2. **Always EAGLE3** — same model + `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` on every request  
3. **Selective** — scheduler routing with runtime demotion on low rolling acceptance

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login   # accept Llama license on HuggingFace
```

## Run evaluation

```bash
cd vllm_bench
python gpu_check.py
python benchmark.py --quick   # 7-prompt smoke test (~2 min)
python benchmark.py           # full 35-prompt run (~15 min)
```

Optional — GPU monitor in a second terminal: `python gpu_watch.py`

## Layout

```
vllm_bench/
  scheduler.py   # routing policy (core)
  benchmark.py   # validation harness: baseline → always-EAGLE3 → selective
  config.py      # models, TP, k, memory, scheduler thresholds
  prompts.py     # mixed workload (35 + quick 7)
  metrics.py     # vLLM spec-decode counters
  gpu_check.py   # pre-flight GPU check
  run.sh         # setup + evaluation one-liner
```

Edit `vllm_bench/config.py` for models, `NUM_SPECULATIVE_TOKENS` (k), or scheduler thresholds.

## Troubleshooting

- **OOM on reload** — each phase uses a fresh subprocess; lower `GPU_MEMORY_UTILIZATION` if needed
- **EAGLE3 hang** — keep batch=1 on vLLM 0.11; batched spec is unstable on mixed workloads
- **Spec stats show 0%** — `disable_log_stats` must be `False` on the LLM (set in `benchmark.py`)
- **Llama 403** — `huggingface-cli login` + accept model license
- **transformers error** — keep `transformers<5.0` (pinned in requirements.txt)

## License

MIT License — see [LICENSE](LICENSE). Free to use, modify, and distribute.
