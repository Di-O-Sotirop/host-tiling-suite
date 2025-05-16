#!/bin/bash

# Move to the root directory of host-tiling-lab if not already there
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Path to the benchmark config
CONFIG_PATH="config/gemver.yaml"

# Run the benchmark using Python module execution
echo "[INFO] Running GEMVER benchmark..."
python3 -m scripts.run_benchmarks "$CONFIG_PATH"

# Confirm completion
echo "[INFO] Benchmark finished."
