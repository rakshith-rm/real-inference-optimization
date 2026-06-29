"""All knobs in one place — edit here before running."""

TARGET_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
EAGLE3_DRAFT_MODEL = "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B"
HARDWARE_LABEL = "A6000"

TENSOR_PARALLEL_SIZE = 1

# Static EAGLE3 (fixed draft length). topk=1 is required for adaptive mode,
# so we keep it here too for an apples-to-apples comparison.
SPEC_NUM_STEPS = 3
SPEC_EAGLE_TOPK = 1
SPEC_NUM_DRAFT_TOKENS = 4  # = SPEC_NUM_STEPS + 1 when topk=1

# Adaptive EAGLE3 (--adaptive): SGLang switches speculative_num_steps at runtime
# from an EMA of accepted draft length. These tiers are written to
# adaptive_config.json when the adaptive server launches.
ADAPTIVE_CANDIDATE_STEPS = [1, 2, 3, 4]
ADAPTIVE_EMA_ALPHA = 0.2        # EMA smoothing of accepted draft length
ADAPTIVE_WARMUP_BATCHES = 5     # observe-only before first switch
ADAPTIVE_UPDATE_INTERVAL = 3    # re-evaluate every N verify batches

MAX_OUTPUT_TOKENS = 256
TEMPERATURE = 0.0
GPU_MEMORY_FRACTION = 0.85

HOST = "127.0.0.1"
PORT = 30000
SERVER_STARTUP_TIMEOUT_S = 1200
