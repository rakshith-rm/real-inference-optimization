# EAGLE3 Speculative Decoding (SGLang)

Baseline vs EAGLE3 on Llama-3.1-8B. SGLang serves the model over HTTP.

```bash
# from repo root (after: pip install -r requirements.txt)
cd sglang
python gpu_check.py
python benchmark.py --quick       # baseline vs static EAGLE3
python benchmark.py               # full prompt set
python benchmark.py --adaptive    # + runtime-adaptive draft length
```

SGLang is used as a **server** (launched as a subprocess: `python -m sglang.launch_server`
in `server.py`), and the benchmark talks to it over HTTP — not via `import sglang`.

Draft model: `jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B` (SGLang format, not the vLLM checkpoint).

### Adaptive draft length (`--adaptive`)

SGLang switches `speculative_num_steps` at runtime from an EMA of accepted draft
length, choosing among `ADAPTIVE_CANDIDATE_STEPS` in `config.py`. Rules:

- Candidate tiers (e.g. `[1,2,3,4]`) set the allowed draft lengths
- After each verify round, accepted length feeds an EMA (`ema_alpha`)
- `warmup_batches` observe-only before the first switch
- Re-evaluate every `update_interval` batches; hysteresis avoids thrashing
- Switch happens before the next draft round; requires `topk=1`; output is lossless
