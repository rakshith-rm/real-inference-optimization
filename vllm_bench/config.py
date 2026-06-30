"""All knobs in one place — edit here before running."""

# Target LLM (needs HF access: huggingface-cli login + Llama license)
TARGET_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# EAGLE3 draft head matched to Llama 3.1 8B
EAGLE3_MODEL = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"

# Shown in the resume line — set to whatever you actually ran on.
HARDWARE_LABEL = "A6000"

# Llama-8B fits on ONE A6000 (48GB). TP=1 avoids PCIe all-reduce overhead
# that kills speculative decoding gains. Spec decoding wins at low batch / TP=1.
TENSOR_PARALLEL_SIZE = 1
DRAFT_TENSOR_PARALLEL_SIZE = 1

# k=2: pos-2 acceptance was only ~12% on this workload, so drafting 3 wasted work.
NUM_SPECULATIVE_TOKENS = 2
MAX_OUTPUT_TOKENS = 256
GPU_MEMORY_UTILIZATION = 0.85

# Batch size for llm.generate() — same for all three modes (fair comparison).
# Batches are split by temperature band first (vLLM 0.11 EAGLE3 hangs if temp=0
# and temp=1.0 prompts share one batch), then chunked to this size.
GENERATE_CHUNK_SIZE = 4

# Sampling: temperature=0 keeps benchmark numbers reproducible
TEMPERATURE = 0.0

# Selective speculation scheduler (see scheduler.py)
SCHED_MIN_PROMPT_TOKENS = 64
SCHED_MAX_SPEC_TEMPERATURE = 0.8
SCHED_MIN_ROLLING_ACCEPT_RATE = 1.2
SCHED_ROLLING_WINDOW = 8
SCHED_SPEC_CHUNK_SIZE = 4
SCHED_INITIAL_ACCEPT_RATE = 2.0
