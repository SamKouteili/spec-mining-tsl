#!/usr/bin/env bash
# mine.sh - Main entry point for TSL_f specification mining pipeline
#
# Orchestrates:
#   1. SyGuS function synthesis (functions.sh)
#   2. Bolt LTL mining (bolt_safety_liveness.sh or bolt.sh)
#   3. (Optional) Spec transformation for game-specific format
#
# Usage:
#   bash mine.sh <trace_dir> [options]
#
# Examples:
#   bash mine.sh games/Logs/frozen_lake/20260110 --mode safety-liveness --collect-all --max-size 7
#   bash mine.sh traces/ --mode safety-liveness --first-all --game frozen_lake --prune --self-inputs-only

set -euo pipefail

#########################################
# PATHS
#########################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYGUS_DIR="$SCRIPT_DIR/sygus"
BOLT_DIR="$SCRIPT_DIR/bolt"
SYNT_DIR="$(dirname "$SCRIPT_DIR")/games/synt"

#########################################
# DEFAULTS
#########################################

MODE="safety-liveness"
MAX_SIZE=10
COLLECT_ALL=""
FIRST_ALL=""
SELF_INPUTS_ONLY=""
PRUNE=""
GAME=""
POS_DIR=""
NEG_DIR=""
SAFETY_ONLY=""
LIVENESS_ONLY=""
MAX_SOLUTIONS=""

#########################################
# ARGUMENT PARSING
#########################################

print_usage() {
    echo "Usage: $0 <trace_dir> [options]"
    echo ""
    echo "Arguments:"
    echo "  trace_dir         Directory containing pos/ and neg/ trace subdirectories"
    echo ""
    echo "Options:"
    echo "  --mode <mode>     Mining mode: bolt, safety-liveness (default), safety, liveness"
    echo "  --max-size N      Maximum formula size for enumeration (default: 10)"
    echo "  --collect-all     Collect ALL solutions up to max-size"
    echo "  --first-all       Find ALL specs at FIRST size where solutions found"
    echo "  --self-inputs-only Only consider self-updates (playerX <- f(playerX))"
    echo "  --prune           Remove spurious/tautological safety specs"
    echo "  --game <name>     Transform specs for specific game (frozen_lake, taxi, etc.)"
    echo "  --pos <dir>       Explicit path to positive traces directory"
    echo "  --neg <dir>       Explicit path to negative traces directory"
    echo "  --safety-only     Mine ONLY safety specs (G-rooted)"
    echo "  --liveness-only   Mine ONLY liveness specs (F-rooted)"
    echo "  --max-solutions N Limit number of solutions to collect"
    echo ""
    echo "Output files (in <trace_dir>/out/):"
    echo "  metadata.py       - Synthesized functions and predicates"
    echo "  bolt.json         - Boolean traces for Bolt"
    echo "  updates.tsl       - Updates grouped by variable"
    echo "  liveness.tsl      - Best liveness spec"
    echo "  safety.tsl        - Best safety spec"
    echo "  spec.tsl          - Combined specification"
    echo "  all_liveness.tsl  - All liveness specs (with --first-all/--collect-all)"
    echo "  all_safety.tsl    - All safety specs (with --first-all/--collect-all)"
}

# Handle --help before requiring positional args
if [[ $# -lt 1 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    print_usage
    [[ $# -ge 1 ]] && exit 0 || exit 1
fi

# First positional argument is trace_dir
TRACE_DIR="$1"
shift

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --max-size)
            MAX_SIZE="$2"
            shift 2
            ;;
        --collect-all)
            COLLECT_ALL="yes"
            shift
            ;;
        --first-all)
            FIRST_ALL="yes"
            shift
            ;;
        --self-inputs-only)
            SELF_INPUTS_ONLY="yes"
            shift
            ;;
        --prune)
            PRUNE="yes"
            shift
            ;;
        --game)
            GAME="$2"
            shift 2
            ;;
        --pos)
            POS_DIR="$2"
            shift 2
            ;;
        --neg)
            NEG_DIR="$2"
            shift 2
            ;;
        --safety-only)
            SAFETY_ONLY="yes"
            shift
            ;;
        --liveness-only)
            LIVENESS_ONLY="yes"
            shift
            ;;
        --max-solutions)
            MAX_SOLUTIONS="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            print_usage
            exit 1
            ;;
    esac
done

#########################################
# VALIDATION
#########################################

if [[ ! -d "$TRACE_DIR" ]]; then
    echo "[mine.sh] Error: Trace directory '$TRACE_DIR' does not exist" >&2
    exit 1
fi

# Convert to absolute path
TRACE_DIR="$(cd "$TRACE_DIR" && pwd)"
OUT_DIR="$TRACE_DIR/out"

echo "=============================================="
echo "[mine.sh] TSL_f Specification Mining Pipeline"
echo "=============================================="
echo "Trace directory: $TRACE_DIR"
echo "Output directory: $OUT_DIR"
echo "Mode: $MODE"
echo "Max size: $MAX_SIZE"
[[ -n "$COLLECT_ALL" ]] && echo "Search: collect-all"
[[ -n "$FIRST_ALL" ]] && echo "Search: first-all"
[[ -n "$SELF_INPUTS_ONLY" ]] && echo "Input mode: self-inputs-only"
[[ -n "$PRUNE" ]] && echo "Pruning: enabled"
[[ -n "$GAME" ]] && echo "Game transformation: $GAME"
[[ -n "$POS_DIR" ]] && echo "Positive traces: $POS_DIR"
[[ -n "$NEG_DIR" ]] && echo "Negative traces: $NEG_DIR"
[[ -n "$SAFETY_ONLY" ]] && echo "Filter: safety-only"
[[ -n "$LIVENESS_ONLY" ]] && echo "Filter: liveness-only"
[[ -n "$MAX_SOLUTIONS" ]] && echo "Max solutions: $MAX_SOLUTIONS"
echo ""

#########################################
# BUILD FLAG STRINGS
#########################################

# Flags for functions.sh
SYGUS_FLAGS=""
[[ -n "$SELF_INPUTS_ONLY" ]] && SYGUS_FLAGS="$SYGUS_FLAGS --self-inputs-only"
[[ -n "$POS_DIR" ]] && SYGUS_FLAGS="$SYGUS_FLAGS --pos $POS_DIR"
[[ -n "$NEG_DIR" ]] && SYGUS_FLAGS="$SYGUS_FLAGS --neg $NEG_DIR"

# Flags for bolt scripts
BOLT_FLAGS=""
[[ -n "$COLLECT_ALL" ]] && BOLT_FLAGS="$BOLT_FLAGS --collect-all"
[[ -n "$FIRST_ALL" ]] && BOLT_FLAGS="$BOLT_FLAGS --first-all"
[[ -n "$SELF_INPUTS_ONLY" ]] && BOLT_FLAGS="$BOLT_FLAGS --self-inputs-only"
[[ -n "$PRUNE" ]] && BOLT_FLAGS="$BOLT_FLAGS --prune"
[[ -n "$POS_DIR" ]] && BOLT_FLAGS="$BOLT_FLAGS --pos $POS_DIR"
[[ -n "$NEG_DIR" ]] && BOLT_FLAGS="$BOLT_FLAGS --neg $NEG_DIR"
[[ -n "$SAFETY_ONLY" ]] && BOLT_FLAGS="$BOLT_FLAGS --safety-only"
[[ -n "$LIVENESS_ONLY" ]] && BOLT_FLAGS="$BOLT_FLAGS --liveness-only"
[[ -n "$MAX_SOLUTIONS" ]] && BOLT_FLAGS="$BOLT_FLAGS --max-solutions $MAX_SOLUTIONS"

#########################################
# STEP 1: SyGuS Function Synthesis
#########################################

echo ""
echo "=============================================="
echo "[Step 1] Running SyGuS Function Synthesis"
echo "=============================================="

if [[ ! -f "$SYGUS_DIR/functions.sh" ]]; then
    echo "[mine.sh] Error: functions.sh not found at $SYGUS_DIR/functions.sh" >&2
    exit 1
fi

# Create output directory
mkdir -p "$OUT_DIR"

# Run functions.sh
bash "$SYGUS_DIR/functions.sh" "$TRACE_DIR" "$OUT_DIR" $SYGUS_FLAGS

# Verify metadata was created
METADATA_FILE="$OUT_DIR/metadata.py"
if [[ ! -f "$METADATA_FILE" ]]; then
    echo "[mine.sh] Error: metadata.py not generated at $METADATA_FILE" >&2
    exit 1
fi

echo "[Step 1] SyGuS synthesis complete. Metadata: $METADATA_FILE"

#########################################
# STEP 2: Bolt LTL Mining
#########################################

echo ""
echo "=============================================="
echo "[Step 2] Running Bolt TSLf Mining"
echo "=============================================="

if [[ "$MODE" == "bolt" ]]; then
    # Use basic bolt.sh
    if [[ ! -f "$BOLT_DIR/bolt.sh" ]]; then
        echo "[mine.sh] Error: bolt.sh not found at $BOLT_DIR/bolt.sh" >&2
        exit 1
    fi
    bash "$BOLT_DIR/bolt.sh" "$TRACE_DIR" "$METADATA_FILE" "$OUT_DIR" $BOLT_FLAGS
else
    # Use bolt_safety_liveness.sh for all other modes
    if [[ ! -f "$BOLT_DIR/bolt_safety_liveness.sh" ]]; then
        echo "[mine.sh] Error: bolt_safety_liveness.sh not found at $BOLT_DIR/bolt_safety_liveness.sh" >&2
        exit 1
    fi
    bash "$BOLT_DIR/bolt_safety_liveness.sh" "$TRACE_DIR" "$METADATA_FILE" "$OUT_DIR" "$MAX_SIZE" $BOLT_FLAGS
fi

echo "[Step 2] Bolt mining complete."

#########################################
# STEP 3: (Optional) Spec Transformation
#########################################

if [[ -n "$GAME" ]]; then
    # echo ""
    # echo "=============================================="
    # echo "[Step 3] Transforming Specs for Game: $GAME"
    # echo "=============================================="

    TRANSFORMER="$SYNT_DIR/spec_transformer.py"
    if [[ ! -f "$TRANSFORMER" ]]; then
        echo "[mine.sh] Warning: spec_transformer.py not found at $TRANSFORMER" >&2
        # echo "[mine.sh] Skipping spec transformation."
    else
        # Transform each spec file
        for spec_file in "$OUT_DIR/liveness.tsl" "$OUT_DIR/safety.tsl" "$OUT_DIR/spec.tsl"; do
            if [[ -f "$spec_file" ]] && [[ -s "$spec_file" ]]; then
                spec_content=$(cat "$spec_file")
                if [[ -n "$spec_content" ]]; then
                    # Determine if this is a safety spec
                    is_safety=""
                    if [[ "$spec_file" == *"safety"* ]]; then
                        is_safety="--is-safety"
                    fi

                    # Transform and overwrite
                    transformed=$(python3 "$TRANSFORMER" --spec "$spec_content" --game "$GAME" $is_safety 2>/dev/null || echo "$spec_content")
                    echo "$transformed" > "$spec_file"
                    echo "  Transformed: $spec_file"
                fi
            fi
        done

        # Transform all_liveness.tsl and all_safety.tsl if they exist
        for all_file in "$OUT_DIR/all_liveness.tsl" "$OUT_DIR/all_safety.tsl"; do
            if [[ -f "$all_file" ]] && [[ -s "$all_file" ]]; then
                is_safety=""
                if [[ "$all_file" == *"safety"* ]]; then
                    is_safety="--is-safety"
                fi

                # Transform each line
                temp_file=$(mktemp)
                while IFS= read -r line; do
                    if [[ -n "$line" ]]; then
                        transformed=$(python3 "$TRANSFORMER" --spec "$line" --game "$GAME" $is_safety 2>/dev/null || echo "$line")
                        echo "$transformed" >> "$temp_file"
                    fi
                done < "$all_file"
                mv "$temp_file" "$all_file"
                echo "  Transformed: $all_file"
            fi
        done

        # echo "[Step 3] Spec transformation complete."
    fi
fi

#########################################
# SUMMARY
#########################################

echo ""
echo "=============================================="
echo "[mine.sh] Pipeline Complete!"
echo "=============================================="
echo "Output files:"
[[ -f "$OUT_DIR/metadata.py" ]] && echo "  $OUT_DIR/metadata.py"
[[ -f "$OUT_DIR/bolt.json" ]] && echo "  $OUT_DIR/bolt.json"
[[ -f "$OUT_DIR/updates.tsl" ]] && echo "  $OUT_DIR/updates.tsl"
[[ -f "$OUT_DIR/liveness.tsl" ]] && echo "  $OUT_DIR/liveness.tsl"
[[ -f "$OUT_DIR/safety.tsl" ]] && echo "  $OUT_DIR/safety.tsl"
[[ -f "$OUT_DIR/spec.tsl" ]] && echo "  $OUT_DIR/spec.tsl"
[[ -f "$OUT_DIR/all_liveness.tsl" ]] && echo "  $OUT_DIR/all_liveness.tsl"
[[ -f "$OUT_DIR/all_safety.tsl" ]] && echo "  $OUT_DIR/all_safety.tsl"
echo ""

# Print final spec if it exists
if [[ -f "$OUT_DIR/spec.tsl" ]] && [[ -s "$OUT_DIR/spec.tsl" ]]; then
    echo "Final specification:"
    cat "$OUT_DIR/spec.tsl"
    echo ""
fi
