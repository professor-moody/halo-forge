#!/bin/bash
# Run the full pipeline example

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "halo-forge Full Pipeline Example"
echo "=============================================="
echo

# Check if in toolbox
if ! command -v halo-forge &> /dev/null; then
    echo "Note: Running without halo-forge CLI"
    echo "Make sure you're in the toolbox: toolbox enter halo-forge"
    echo
fi

# Run training
python train.py --step all

echo
echo "Pipeline complete!"
echo "Check results/ for benchmark results"

