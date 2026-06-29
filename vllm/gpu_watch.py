"""Run in a second terminal while benchmark.py runs: watch both GPUs light up."""

import subprocess
import time


def main() -> None:
    print("Polling nvidia-smi every 2s (Ctrl+C to stop)\n")
    while True:
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader",
            ],
            check=False,
        )
        print("-" * 60)
        time.sleep(2)


if __name__ == "__main__":
    main()
