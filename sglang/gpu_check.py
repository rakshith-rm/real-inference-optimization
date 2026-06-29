"""Verify enough GPUs are visible before launching the SGLang server."""

import subprocess
import sys

import torch

import config


def main() -> None:
    count = torch.cuda.device_count()
    print(f"PyTorch sees {count} GPU(s)")
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {props.name} — {props.total_memory // (1024**3)} GiB")

    need = config.TENSOR_PARALLEL_SIZE
    if count < need:
        print(f"\nERROR: need {need} GPU(s) for TP={need}.")
        sys.exit(1)

    print("\nnvidia-smi snapshot:")
    subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,utilization.gpu", "--format=csv"],
        check=False,
    )


if __name__ == "__main__":
    main()
