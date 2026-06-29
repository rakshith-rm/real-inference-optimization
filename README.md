# Real Inference Optimization

Two implementations of the same benchmark: **baseline vs EAGLE3 speculative decoding** on Llama-3.1-8B.

| Folder | Engine | How it runs |
|--------|--------|-------------|
| [`vllm/`](vllm/) | vLLM 0.11.0 | In-process `LLM().generate()` — ran on A6000 (+8.8% with TP=1, k=2) |
| [`sglang/`](sglang/) | SGLang | HTTP server + client — same comparison, different serving stack |

```bash
# once, from repo root
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login

# vLLM (proven A6000 results)
cd vllm && python benchmark.py --quick

# SGLang
cd ../sglang && python benchmark.py --quick
```

Both need a cloud GPU with ~16GB+ VRAM (Llama-8B + EAGLE3 head). Won't fit on 8GB laptop.

**Important:** install **one stack per venv** — vLLM and SGLang conflict if installed together.

```bash
# SGLang (sglang/ benchmark)
pip install -r requirements.txt

# vLLM (vllm/ benchmark) — separate venv recommended
pip install -r requirements-vllm.txt
```
