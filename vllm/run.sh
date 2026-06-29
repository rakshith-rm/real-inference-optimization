#!/usr/bin/env bash
set -euo pipefail

# One-shot setup + benchmark on a fresh cloud node.
# Usage: bash run.sh [--quick]

python gpu_check.py
pip install -r ../requirements.txt -q

python benchmark.py "$@"
