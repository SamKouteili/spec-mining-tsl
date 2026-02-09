#!/bin/bash
#
# TSL_f Specification Mining Demo
#
# This script demonstrates the complete specification mining pipeline:
#   1. Generate traces (play the game or auto-generate)
#   2. Mining a TSL_f specification from traces
#   3. Synthesizing a controller from the mined specification
#   4. Replaying the controller's moves on the original board
#
# Usage:
#   bash demo.sh frozen_lake
#   bash demo.sh frozen_lake --gen-traces 10
#   bash demo.sh frozen_lake --random-placements
#

set -e

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: bash demo.sh <game> [options]"
    echo ""
    echo "Games:"
    echo "  frozen_lake  - Navigate a frozen lake to reach the goal"
    echo ""
    echo "Options:"
    echo "  --gen-traces N       - Auto-generate N positive and N negative traces"
    echo "  --random-placements  - Randomize goal and hole positions"
    echo "  --debug              - Show detailed output from mining and synthesis"
    echo ""
    echo "Examples:"
    echo "  bash demo.sh frozen_lake                     # Interactive play"
    echo "  bash demo.sh frozen_lake --gen-traces 10    # Auto-generate traces"
    echo "  bash demo.sh frozen_lake --random-placements"
    exit 1
fi

# Make sure conda environment is activated
if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" != "tlsf" ]]; then
    echo "Warning: conda environment 'tlsf' may not be activated"
    echo "Run: conda activate tlsf"
    echo ""
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the Python demo script
python "$SCRIPT_DIR/demo.py" "$@"
