"""Verify 2 GPUs are visible before spending time loading models."""

import subprocess
import sys

import torch


def main() -> None:
    count = torch.cuda.device_count()
    print(f"PyTorch sees {count} GPU(s)")
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {props.name} — {props.total_memory // (1024**3)} GiB")

    if count < 2:
        print("\nERROR: need at least 2 GPUs for this project (tensor_parallel_size=2).")
        sys.exit(1)

    print("\nnvidia-smi snapshot:")
    subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,utilization.gpu", "--format=csv"], check=False)


if __name__ == "__main__":
    main()
