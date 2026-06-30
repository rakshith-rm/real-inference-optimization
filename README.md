# Real Inference Optimization

**Baseline vs EAGLE3 speculative decoding** on Llama-3.1-8B using vLLM, plus a **selective speculation scheduler** that routes requests to baseline or EAGLE3 based on prompt length, temperature, and rolling draft acceptance rate.

Runs on **1× A6000 48GB** (TP=1, k=2). All modes generate in a single batch for a
fair comparison, and each engine runs in its own subprocess so its GPU memory is
fully released when it exits.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login   # + accept Llama license on HuggingFace
```

## Run

```bash
cd vllm_bench
python gpu_check.py
python benchmark.py --quick   # smoke test
python benchmark.py           # full benchmark
```

Needs a cloud GPU with ~24GB+ VRAM (e.g. A6000).

## Config

Edit `vllm_bench/config.py` — model names, `TENSOR_PARALLEL_SIZE`, `NUM_SPECULATIVE_TOKENS` (k), scheduler thresholds.
