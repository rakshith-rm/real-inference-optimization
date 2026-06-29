# EAGLE3 Speculative Decoding Benchmark (vLLM)

Baseline vs EAGLE3 on Llama-3.1-8B. Default: **TP=1, k=2** (tuned for A6000).

## Run

```bash
# from repo root (after pip install -r requirements.txt)
cd vllm_bench
python gpu_check.py
python benchmark.py --quick
python benchmark.py
```

Optional — GPU utilization in a second terminal:

```bash
python gpu_watch.py
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | Models, TP, k, memory |
| `benchmark.py` | Baseline → EAGLE3, throughput + acceptance stats |
| `metrics.py` | vLLM spec-decode counters |
| `gpu_check.py` | Verify GPU before load |

## Results (A6000, TP=1, k=2)

- Throughput: **+8.8%** vs baseline
- Mean accept length: **~1.64**

## Troubleshooting

- **OOM:** lower `GPU_MEMORY_UTILIZATION` in `config.py`
- **Llama 403:** `huggingface-cli login` + accept model license
- **transformers error:** keep `transformers<5.0` (pinned in requirements.txt)
