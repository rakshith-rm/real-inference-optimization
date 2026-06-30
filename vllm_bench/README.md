# EAGLE3 Speculative Decoding Benchmark (vLLM)

Baseline vs always-on EAGLE3 vs **selective speculation scheduler** on Llama-3.1-8B.
Default: **TP=1, k=2** (tuned for A6000). Workload mixes short prompts, high-temperature
creative requests, and long-context analytical prompts.

All three modes generate in a **single batch** so the comparison is fair, and
**each engine runs in its own subprocess** so its GPU memory is fully released when
it exits (no reload OOM).

## Run

```bash
# from repo root (after pip install -r requirements.txt)
cd vllm_bench
python gpu_check.py
python benchmark.py --quick   # 7-prompt smoke test
python benchmark.py           # full 35-prompt workload
```

Optional — GPU utilization in a second terminal:

```bash
python gpu_watch.py
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | Models, TP, k, memory, scheduler thresholds |
| `prompts.py` | Mixed workload (`WORKLOAD`, `QUICK_WORKLOAD`) |
| `scheduler.py` | Per-request speculation routing policy |
| `benchmark.py` | Baseline → always-EAGLE3 → selective scheduler |
| `metrics.py` | vLLM spec-decode counters |
| `gpu_check.py` | Verify GPU before load |

## Reading the results

Speculative decoding helps most on long, low-temperature prompts (higher draft
acceptance) and hurts on short/high-temperature ones (draft overhead with little
acceptance). The selective scheduler routes per request to capture the gains while
avoiding the losses, and the `mean accept length` reported per run quantifies why.

## Troubleshooting

- **OOM:** lower `GPU_MEMORY_UTILIZATION` in `config.py`
- **Llama 403:** `huggingface-cli login` + accept model license
- **transformers error:** keep `transformers<5.0` (pinned in requirements.txt)
