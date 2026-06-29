#!/usr/bin/env bash
set -euo pipefail

# One-shot setup + benchmark on a fresh cloud node.
# Usage: bash run.sh [--quick]

python gpu_check.py
pip install -r ../requirements.txt -q

echo ""
echo "Tip: open a second terminal and run:  python gpu_watch.py"
echo ""

python benchmark.py "$@"
