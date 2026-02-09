#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <logs_dir> <metadata.py> <output_dir> [--self-inputs-only]" >&2
  exit 1
fi

LOGS_DIR=$1
META_FILE=$2
OUT_DIR=$3
shift 3

# Parse optional flags
SELF_INPUTS_ONLY=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --self-inputs-only)
            SELF_INPUTS_ONLY="--self-inputs-only"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

if [[ ! -d "$LOGS_DIR" ]]; then
  echo "[bolt.sh] Error: Logs directory '$LOGS_DIR' does not exist" >&2
  exit 1
fi

if [[ ! -f "$META_FILE" ]]; then
  echo "[bolt.sh] Error: Metadata file '$META_FILE' does not exist" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROCESS_PY="$SCRIPT_DIR/log2tslf.py"

if [[ ! -f "$PROCESS_PY" ]]; then
  echo "[bolt.sh] Error: log2tslf.py not found at $PROCESS_PY" >&2
  exit 1
fi

if ! command -v bolt >/dev/null 2>&1; then
  echo "[bolt.sh] Error: 'bolt' command not found in PATH" >&2
  exit 1
fi

MAX_SIZE_LTL=10        # Maximum size explored by the LTL enumerator
DOMIN_NB=10            # Number of candidates retained during domination checking
MAX_SIZE_BOOL=100      # Maximum size for boolean enumeration

echo "[bolt.sh] Generating BOLT traces with log2tslf.py"
python "$PROCESS_PY" "$LOGS_DIR" "$META_FILE" --out "$OUT_DIR" $SELF_INPUTS_ONLY

shopt -s nullglob
trace_files=("$OUT_DIR"/*.json)
shopt -u nullglob

if [[ ${#trace_files[@]} -eq 0 ]]; then
  echo "[bolt.sh] No trace files were generated in $OUT_DIR" >&2
  exit 1
fi

declare -a specs=()
for trace in "${trace_files[@]}"; do
  echo "[bolt.sh] Mining specification from $trace"
  spec_output=$(bolt "$trace" "$MAX_SIZE_LTL" "$DOMIN_NB" enum "$MAX_SIZE_BOOL" "$DOMIN_NB")
  specs+=("$spec_output")
done

if [[ ${#specs[@]} -eq 0 ]]; then
  echo "[bolt.sh] No specifications were produced by bolt" >&2
  exit 1
fi

final_spec=""
for spec in "${specs[@]}"; do
  cleaned_spec=$(printf '%s\n' "$spec" | sed '/^[[:space:]]*$/d')
  if [[ -z "$cleaned_spec" ]]; then
    continue
  fi
  if [[ -z "$final_spec" ]]; then
    final_spec="($cleaned_spec)"
  else
    final_spec="${final_spec} | ($cleaned_spec)"
  fi
done

if [[ -z "$final_spec" ]]; then
  echo "[bolt.sh] Warning: All specifications were empty; final.tsl will be empty" >&2
fi

FINAL_SPEC_PATH="$OUT_DIR/final.tsl"
printf '%s\n' "$final_spec" > "$FINAL_SPEC_PATH"
echo "[bolt.sh] Combined specification saved to $FINAL_SPEC_PATH"
cat "$FINAL_SPEC_PATH"
