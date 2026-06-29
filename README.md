# Real Inference Optimization

**Baseline vs EAGLE3 speculative decoding** on Llama-3.1-8B using vLLM.

Proven on **1× A6000 48GB** (TP=1, k=2): **+8.8% throughput** over baseline.

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
python benchmark.py --quick   # 2 prompts, smoke test
python benchmark.py           # full prompt set
```

Needs a cloud GPU with ~24GB+ VRAM (won't fit on 8GB laptop).

## Config

Edit `vllm_bench/config.py` — model names, `TENSOR_PARALLEL_SIZE`, `NUM_SPECULATIVE_TOKENS` (k).
