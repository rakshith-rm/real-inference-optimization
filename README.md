# Real Inference Optimization

Benchmark comparing **baseline decode**, **always-on EAGLE3**, and a **selective speculation scheduler** on Llama-3.1-8B via vLLM 0.11.

**Hardware:** 1× NVIDIA A6000 (48GB), TP=1, EAGLE3 k=2  
**Workload:** 35 mixed prompts (short, high-temp, long analytical, long context)  
**Method:** one prompt per `generate()` call (batch=1) — fair comparison and stable on vLLM 0.11 EAGLE3

## Results

| Mode | Throughput | vs baseline | Inference | Tokens |
|------|------------|-------------|-----------|--------|
| Baseline | 41.1 tok/s | — | 167s | 6,859 |
| Always EAGLE3 | 50.5 tok/s | **+22.8%** | 139s | 7,012 |
| Selective scheduler | 43.7 tok/s | **+6.5%** | 157s | 6,859 |

**Always-on spec:** mean accept length 1.59, draft accept 29.7%  
**Selective spec (6 routed requests):** mean accept length 1.75, draft accept 37.3%  
**Routing:** 6 spec / 29 baseline — `short_prompt: 29`, `spec_enabled: 6`

### Takeaways

- **EAGLE3 always-on** gives a solid ~23% gain on this workload at low batch / TP=1.
- **Selective routing** improves over baseline but trails always-on because only **6/35** requests (long context, prompt ≥64 tokens) use speculation; the rest run baseline.
- On routed traffic, acceptance is **higher** (1.75 vs 1.59) — the policy picks good candidates but is conservative today.
- Run completed cleanly: no hangs, no OOM. Each engine runs in a **subprocess** so GPU memory is freed between phases.

## What it does

1. **Baseline** — Llama-3.1-8B-Instruct, no speculation  
2. **Always EAGLE3** — same model + `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` draft head on every request  
3. **Selective** — scheduler routes by prompt length, temperature, and rolling draft acceptance; demotes to baseline when acceptance drops

## Selective scheduler

### Why use it

Always-on speculation pays draft-model cost on every request, including short prompts and high-temperature creative traffic where acceptance is poor. The selective scheduler **skips those requests** and only runs EAGLE3 where drafts are likely to pay off.

That makes it a practical **default routing layer in a serving stack**: one target model, optional spec path, per-request decision. With thresholds tuned to your traffic (prompt length distribution, temperature mix, output budgets), you can capture most of the always-on speedup while avoiding wasted draft work on requests that would never benefit.

This benchmark’s defaults are conservative (prompt-length gate only), so always-on wins on throughput here — but the routed 6 requests show **better acceptance** (1.75 vs 1.59), which is what you want before widening the policy.

### Routing rules (current `config.py`)

| Check | Parameter | Value | Effect |
|-------|-----------|-------|--------|
| Prompt too short | `SCHED_MIN_PROMPT_TOKENS` | **64** | Route to baseline if input under 64 tokens |
| Temperature too high | `SCHED_MAX_SPEC_TEMPERATURE` | **0.8** | Route to baseline if `temperature > 0.8` |
| Rolling accept too low | `SCHED_MIN_ROLLING_ACCEPT_RATE` | **1.2** | During spec phase, demote remaining requests if mean accept length over last window falls below 1.2 |
| Rolling window | `SCHED_ROLLING_WINDOW` | **8** | Last 8 spec requests used for rolling accept rate |
| Initial accept (no samples yet) | `SCHED_INITIAL_ACCEPT_RATE` | **2.0** | Optimistic prior before first measured accept length |

**Decision order:** short prompt → high temperature → low rolling accept → otherwise **spec enabled**.

vLLM locks `speculative_config` at engine init, so the benchmark runs spec and baseline requests on **separate engine subprocesses** (not simultaneous per-request switching in one process).

### Tuning for production

Lower `SCHED_MIN_PROMPT_TOKENS` or add output-length / category rules to route more decode-heavy traffic through spec. Raise `SCHED_MAX_SPEC_TEMPERATURE` only if you measure acceptable acceptance on creative endpoints. Set `SCHED_MIN_ROLLING_ACCEPT_RATE` from observed `mean acceptance length` on a representative trace — demotion protects you when the draft head stops matching live traffic.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login   # accept Llama license on HuggingFace
```

## Run

```bash
cd vllm_bench
python gpu_check.py
python benchmark.py --quick   # 7-prompt smoke test (~2 min)
python benchmark.py           # full 35-prompt benchmark (~15 min)
```

Optional — GPU monitor in a second terminal: `python gpu_watch.py`

## Layout

```
vllm_bench/
  benchmark.py   # baseline → always-EAGLE3 → selective
  config.py      # models, TP, k, memory, scheduler thresholds
  prompts.py     # WORKLOAD (35) + QUICK_WORKLOAD (7)
  scheduler.py   # routing policy
  metrics.py     # vLLM spec-decode counters
  gpu_check.py   # pre-flight GPU check
  run.sh         # setup + benchmark one-liner
```

Edit `vllm_bench/config.py` to change models, `NUM_SPECULATIVE_TOKENS` (k), or scheduler thresholds.

## Troubleshooting

- **OOM on reload** — each phase uses a fresh subprocess; if it still fails, lower `GPU_MEMORY_UTILIZATION`
- **EAGLE3 hang** — do not batch spec requests on vLLM 0.11; keep batch=1
- **Spec stats show 0%** — `disable_log_stats` must be `False` on the LLM (set in `benchmark.py`)
- **Llama 403** — `huggingface-cli login` + accept model license
- **transformers error** — keep `transformers<5.0` (pinned in requirements.txt)
