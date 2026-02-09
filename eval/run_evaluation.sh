#!/bin/bash
# run_evaluation.sh - Generate traces and run pipeline for TSL_f evaluation
#
# Generates traces for:
#   - Fixed board (default 4x4)
#   - Random positions (randomized goal/hole placements)
#   - Random size (randomized board dimensions)
#
# For each condition: n=5,10,15,20,25 traces x 5 independent sets

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
GAMES_DIR="$PROJECT_DIR/games"
EVAL_DIR="$PROJECT_DIR/eval"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate tlsf

echo "=============================================="
echo "TSL_f Specification Mining Evaluation"
echo "=============================================="
echo "Project directory: $PROJECT_DIR"
echo "Evaluation directory: $EVAL_DIR"
echo ""

# Configuration
TRACE_COUNTS=(5 10 15 20 25)
NUM_SETS=5
MAX_SIZE=10

# Conditions
CONDITIONS=("fixed" "random_pos" "random_size")

# Generate traces for a given condition
generate_traces() {
    local condition=$1
    local n=$2
    local set_id=$3
    local out_dir="$EVAL_DIR/$condition/n_${n}/set_${set_id}"

    mkdir -p "$out_dir"

    local extra_args=""
    case $condition in
        "fixed")
            # No extra args - default fixed board
            ;;
        "random_pos")
            extra_args="--random-placements"
            ;;
        "random_size")
            extra_args="--random-size"
            ;;
    esac

    echo "  Generating: $condition n=$n set=$set_id"

    # Run the game to generate traces
    cd "$GAMES_DIR"
    python tfrozen_lake_game.py --gen $n $extra_args 2>/dev/null

    # Find the most recent log directory
    local latest_log=$(ls -td Logs/tfrozen_lake/*/ 2>/dev/null | head -1)

    if [[ -z "$latest_log" ]]; then
        echo "    ERROR: No traces generated!"
        return 1
    fi

    # Move traces to evaluation directory
    mv "$latest_log/pos" "$out_dir/"
    mv "$latest_log/neg" "$out_dir/"
    rmdir "$latest_log" 2>/dev/null || true

    echo "    Generated: $(ls "$out_dir/pos" | wc -l | tr -d ' ') pos, $(ls "$out_dir/neg" | wc -l | tr -d ' ') neg traces"
}

# Run pipeline on a trace set
run_pipeline() {
    local condition=$1
    local n=$2
    local set_id=$3
    local trace_dir="$EVAL_DIR/$condition/n_${n}/set_${set_id}"

    if [[ ! -d "$trace_dir/pos" ]]; then
        echo "  Skipping: $condition n=$n set=$set_id (no traces)"
        return 1
    fi

    echo "  Processing: $condition n=$n set=$set_id"

    cd "$PROJECT_DIR/src"
    ./mine.sh "$trace_dir" --mode first-all --max-size $MAX_SIZE > "$trace_dir/mine.log" 2>&1 || true

    # Check if specs were generated
    if [[ -f "$trace_dir/out/spec.tsl" ]]; then
        local spec=$(cat "$trace_dir/out/spec.tsl" | head -1)
        echo "    Spec: ${spec:0:80}..."
    else
        echo "    No spec generated"
    fi
}

# Extract results summary
extract_results() {
    local condition=$1
    local results_file="$EVAL_DIR/$condition/results.csv"

    echo "n,set,liveness,safety,spec" > "$results_file"

    for n in "${TRACE_COUNTS[@]}"; do
        for set_id in $(seq 1 $NUM_SETS); do
            local trace_dir="$EVAL_DIR/$condition/n_${n}/set_${set_id}"
            local liveness=""
            local safety=""
            local spec=""

            if [[ -f "$trace_dir/out/liveness.tsl" ]]; then
                liveness=$(cat "$trace_dir/out/liveness.tsl" | head -1 | tr ',' ';')
            fi
            if [[ -f "$trace_dir/out/safety.tsl" ]]; then
                safety=$(cat "$trace_dir/out/safety.tsl" | head -1 | tr ',' ';')
            fi
            if [[ -f "$trace_dir/out/spec.tsl" ]]; then
                spec=$(cat "$trace_dir/out/spec.tsl" | head -1 | tr ',' ';')
            fi

            echo "$n,$set_id,\"$liveness\",\"$safety\",\"$spec\"" >> "$results_file"
        done
    done

    echo "Results saved to: $results_file"
}

# Main execution
main() {
    local action=${1:-"all"}

    case $action in
        "generate")
            echo "[Phase 1] Generating traces..."
            for condition in "${CONDITIONS[@]}"; do
                echo ""
                echo "=== Condition: $condition ==="
                mkdir -p "$EVAL_DIR/$condition"
                for n in "${TRACE_COUNTS[@]}"; do
                    for set_id in $(seq 1 $NUM_SETS); do
                        generate_traces "$condition" "$n" "$set_id"
                    done
                done
            done
            ;;

        "pipeline")
            echo "[Phase 2] Running pipeline..."
            for condition in "${CONDITIONS[@]}"; do
                echo ""
                echo "=== Condition: $condition ==="
                for n in "${TRACE_COUNTS[@]}"; do
                    for set_id in $(seq 1 $NUM_SETS); do
                        run_pipeline "$condition" "$n" "$set_id"
                    done
                done
            done
            ;;

        "results")
            echo "[Phase 3] Extracting results..."
            for condition in "${CONDITIONS[@]}"; do
                echo ""
                echo "=== Condition: $condition ==="
                extract_results "$condition"
            done
            ;;

        "all")
            main "generate"
            echo ""
            main "pipeline"
            echo ""
            main "results"
            ;;

        *)
            echo "Usage: $0 [generate|pipeline|results|all]"
            exit 1
            ;;
    esac

    echo ""
    echo "=============================================="
    echo "Evaluation complete!"
    echo "=============================================="
}

main "$@"
