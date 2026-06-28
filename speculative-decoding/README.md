# Speculative Decoding Benchmark (vLLM + EAGLE3, 2× A100)

Small project: **baseline vs EAGLE3** on Llama-3.1-8B with **tensor parallel = 2** so both GPUs participate.

## How both GPUs are used

| Component | GPUs | Config |
|-----------|------|--------|
| Target model (Llama 3.1 8B) | **0 + 1** | `tensor_parallel_size=2` |
| EAGLE3 draft head | **0** | `draft_tensor_parallel_size=1` |

GPU 1 is not idle — it runs half the target layers during draft **verification** (the expensive step speculative decoding saves).

## Setup (cloud node)

```bash
# 1. HuggingFace token (Llama is gated)
export HF_TOKEN=hf_...
huggingface-cli login --token "$HF_TOKEN"

# 2. Run
cd speculative-decoding
bash run.sh          # full benchmark (~8 prompts)
bash run.sh --quick  # smoke test (2 prompts)
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
