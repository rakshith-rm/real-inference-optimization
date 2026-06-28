"""All knobs in one place — edit here before running."""

# Target LLM (needs HF access: huggingface-cli login)
TARGET_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# EAGLE3 draft head matched to Llama 3.1 8B
EAGLE3_MODEL = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"

# Shown in the resume line — set to whatever you actually ran on.
HARDWARE_LABEL = "2x A6000"

# Target model is tensor-parallel across 2 GPUs.
# Draft stays on 1 GPU (stable on vLLM; target still uses both).
TENSOR_PARALLEL_SIZE = 2
DRAFT_TENSOR_PARALLEL_SIZE = 1

NUM_SPECULATIVE_TOKENS = 3
MAX_OUTPUT_TOKENS = 256
GPU_MEMORY_UTILIZATION = 0.85

# Sampling: temperature=0 keeps benchmark numbers reproducible
TEMPERATURE = 0.0
