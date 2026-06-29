"""Launch / stop an SGLang server subprocess for one benchmark mode."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

import config

ADAPTIVE_CONFIG_PATH = Path(__file__).with_name("adaptive_config.json")


def base_url() -> str:
    return f"http://{config.HOST}:{config.PORT}"


def _write_adaptive_config() -> Path:
    ADAPTIVE_CONFIG_PATH.write_text(json.dumps({
        "ema_alpha": config.ADAPTIVE_EMA_ALPHA,
        "warmup_batches": config.ADAPTIVE_WARMUP_BATCHES,
        "update_interval": config.ADAPTIVE_UPDATE_INTERVAL,
        "1": {"candidate_steps": config.ADAPTIVE_CANDIDATE_STEPS,
              "up_hysteresis": 0.0, "down_hysteresis": -0.25, "ceiling_coeff": 0},
    }, indent=2))
    return ADAPTIVE_CONFIG_PATH


def build_command(mode: str) -> list[str]:
    """mode: 'baseline' | 'eagle3' | 'adaptive'."""
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", config.TARGET_MODEL,
        "--host", config.HOST, "--port", str(config.PORT),
        "--tp-size", str(config.TENSOR_PARALLEL_SIZE),
        "--mem-fraction-static", str(config.GPU_MEMORY_FRACTION),
        "--trust-remote-code",
        # Avoid flashinfer JIT — needs nvcc, missing on many cloud VMs
        "--attention-backend", config.ATTENTION_BACKEND,
        "--cuda-graph-backend-prefill", config.CUDA_GRAPH_BACKEND_PREFILL,
    ]
    if mode == "baseline":
        return cmd
    cmd += [
        "--speculative-algorithm", "EAGLE3",
        "--speculative-draft-model-path", config.EAGLE3_DRAFT_MODEL,
        "--speculative-num-steps", str(config.SPEC_NUM_STEPS),
        "--speculative-eagle-topk", str(config.SPEC_EAGLE_TOPK),
        "--speculative-num-draft-tokens", str(config.SPEC_NUM_DRAFT_TOKENS),
    ]
    if mode == "eagle3":
        return cmd
    if mode == "adaptive":
        return cmd + ["--speculative-adaptive",
                      "--speculative-adaptive-config", str(_write_adaptive_config())]
    raise ValueError(f"unknown mode: {mode}")


def _wait_until_ready(timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            if requests.get(f"{base_url()}/health_generate", timeout=5).status_code == 200:
                return
        except requests.RequestException as e:
            last_err = e
        time.sleep(3)
    raise TimeoutError(f"SGLang server not ready after {timeout_s}s. Last error: {last_err}")


class SGLangServer:
    def __init__(self, mode: str):
        self.mode = mode
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> "SGLangServer":
        cmd = build_command(self.mode)
        print(f"\nLaunching SGLang server [{self.mode}]:\n  {' '.join(cmd)}")
        kwargs = {"preexec_fn": os.setsid} if os.name == "posix" else {}
        self.proc = subprocess.Popen(cmd, **kwargs)
        start = time.perf_counter()
        _wait_until_ready(config.SERVER_STARTUP_TIMEOUT_S)
        print(f"Server ready in {time.perf_counter() - start:.1f}s")
        return self

    def __exit__(self, *exc) -> None:
        if self.proc is None:
            return
        print(f"Stopping SGLang server [{self.mode}]...")
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            else:
                self.proc.terminate()
            self.proc.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            self.proc.kill()
        time.sleep(3)
