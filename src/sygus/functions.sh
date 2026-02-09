#!/bin/bash
set -e  # stop on any error

#########################################
# ARGUMENT PARSING
#########################################

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
SELF_INPUTS_ONLY=""
POS_DIR=""
NEG_DIR=""
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --self-inputs-only)
            SELF_INPUTS_ONLY="--self-inputs-only"
            shift
            ;;
        --pos)
            POS_DIR="$2"
            shift 2
            ;;
        --neg)
            NEG_DIR="$2"
            shift 2
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional arguments
set -- "${POSITIONAL_ARGS[@]}"

if [ "$#" != 2 ]; then
    echo "Usage: ./functions.sh <input_dir> <output_dir> [options]"
    echo ""
    echo "Options:"
    echo "  --self-inputs-only    Only use self-inputs (playerX <- f(playerX)),"
    echo "                        skip cross-updates and alternative input swapping."
    echo "                        This shrinks the search space and avoids spurious functions."
    echo "  --pos <dir>           Explicit path to positive traces directory"
    echo "  --neg <dir>           Explicit path to negative traces directory"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

echo "==========================================="
echo " Running SyGuS Function Synthesis Pipeline"
echo " Input Dir:  $INPUT_DIR"
echo " Output Dir: $OUTPUT_DIR"
if [ -n "$SELF_INPUTS_ONLY" ]; then
    echo " Mode:       Self-inputs only"
fi
if [ -n "$POS_DIR" ]; then
    echo " Pos Dir:    $POS_DIR"
fi
if [ -n "$NEG_DIR" ]; then
    echo " Neg Dir:    $NEG_DIR"
fi
echo "==========================================="

# Build pos/neg flags
POS_NEG_FLAGS=""
if [ -n "$POS_DIR" ]; then
    POS_NEG_FLAGS="$POS_NEG_FLAGS --pos $POS_DIR"
fi
if [ -n "$NEG_DIR" ]; then
    POS_NEG_FLAGS="$POS_NEG_FLAGS --neg $NEG_DIR"
fi

#########################################
# STEP 1 — groupings.py (I/O extraction + groupings)
#########################################

echo ""
echo "=== Step 1: groupings.py (I/O extraction + groupings) ==="
python3 "$SCRIPT_DIR/groupings.py" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    $SELF_INPUTS_ONLY $POS_NEG_FLAGS

#########################################
# STEP 2 — sygus.py (function synthesis)
#########################################

echo ""
echo "=== Step 2: sygus.py (function synthesis) ==="
python3 "$SCRIPT_DIR/sygus.py" \
    --root_dir "$OUTPUT_DIR" \
    $SELF_INPUTS_ONLY

#########################################
# STEP 3 — metadata.py (create metadata)
#########################################

echo ""
echo "=== Step 3: metadata.py (create metadata) ==="
python3 "$SCRIPT_DIR/metadata.py" \
    --trace_dir "$INPUT_DIR" \
    --function_dir "$OUTPUT_DIR" \
    $POS_NEG_FLAGS

echo ""
echo "==========================================="
echo " SyGuS Pipeline Complete!"
echo " Results generated under: $OUTPUT_DIR"
echo "==========================================="
