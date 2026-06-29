# Speculative Decoding Benchmark (vLLM + EAGLE3)

Baseline vs EAGLE3 on Llama-3.1-8B. Default config: **TP=1, k=2** (best on A6000).

## Setup

```bash
# from repo root (after: pip install -r requirements.txt)
python gpu_check.py
python benchmark.py --quick
python benchmark.py
```

In a **second terminal** while benchmarking:

```bash
python gpu_watch.py
```

You should see **both GPUs** with high utilization during generation.

## Files

| File | Purpose |
|------|---------|
| `config.py` | Model names, TP sizes, token counts |
| `benchmark.py` | Loads baseline → EAGLE3, prints throughput + acceptance stats |
| `metrics.py` | Reads vLLM spec-decode counters |
| `gpu_check.py` | Verify 2 GPUs before loading models |
| `gpu_watch.py` | Live `nvidia-smi` poll |

## Expected results (2× A100 80GB, 8 prompts, 256 max tokens)

Rough ballpark — your numbers will vary:

- **Throughput:** ~30–60% higher with EAGLE3
- **Mean acceptance length:** ~2.5–3.0 (with `num_speculative_tokens=3`)
- **Draft accept rate:** ~60–75%

## Troubleshooting

- **NCCL timeout with `draft_tensor_parallel_size=2`:** keep draft TP at 1 (already set in `config.py`).
- **OOM:** lower `GPU_MEMORY_UTILIZATION` in `config.py` to `0.75`.
- **Slow first run:** models download from HuggingFace; warmup run is intentional.
