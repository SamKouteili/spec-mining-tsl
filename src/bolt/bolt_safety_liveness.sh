#!/usr/bin/env bash
# bolt_safety_liveness.sh - Mine both safety and liveness specifications
#
# Uses modified Bolt with --safety and --liveness filters to find:
#   - F-rooted (liveness) discriminant: "eventually reach goal"
#   - G-rooted (safety) discriminant: "never hit obstacles"
#
# Bolt CLI structure:
#   Filters (mutually exclusive): --liveness, --safety
#   Search options: (default=first), --first-all, --collect-all, --max-solutions N
#
# Safety properties exclude F, U, R operators, END predicate, and constants
# to ensure they represent pure "nothing bad happens" patterns.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <logs_dir> <metadata.py> [output_dir] [max_size] [flags]" >&2
  echo "" >&2
  echo "Arguments:" >&2
  echo "  logs_dir    - Directory containing pos/ and neg/ trace subdirectories" >&2
  echo "  metadata.py - Metadata file with VARS, FUNCTIONS, PREDICATES" >&2
  echo "  output_dir  - Optional: output directory (default: logs_dir/out)" >&2
  echo "  max_size    - Optional: max formula size for enumeration (default: 10)" >&2
  echo "" >&2
  echo "Flags:" >&2
  echo "  --first-all        - Find ALL specs at FIRST size where solutions found (recommended)" >&2
  echo "  --collect-all      - Collect ALL solutions up to max_size (instead of first)" >&2
  echo "  --max-solutions N  - Limit number of solutions to collect" >&2
  echo "  --self-inputs-only - Only consider self-updates (playerX <- f(playerX)), no cross-updates" >&2
  echo "  --prune            - Remove spurious/tautological safety specs:" >&2
  echo "                       * Tautologies: a->a, a<->a, a|!a, !a|a" >&2
  echo "                       * Specs with all non-identity updates for a variable" >&2
  echo "  --safety-only      - Mine ONLY safety specs (G-rooted), skip liveness" >&2
  echo "  --liveness-only    - Mine ONLY liveness specs (F-rooted), skip safety" >&2
  echo "  --pos <dir>        - Explicit path to positive traces directory (default: logs_dir/pos)" >&2
  echo "  --neg <dir>        - Explicit path to negative traces directory (default: logs_dir/neg)" >&2
  echo "" >&2
  echo "Examples:" >&2
  echo "  $0 games/Logs/frozen_lake/20260110_224101 metadata.py" >&2
  echo "  $0 games/Logs/frozen_lake/20260110_224101 metadata.py out 10 --first-all" >&2
  echo "  $0 games/Logs/frozen_lake/20260110_224101 metadata.py out 8 --collect-all" >&2
  echo "  $0 games/Logs/frozen_lake/20260110_224101 metadata.py out 8 --collect-all --max-solutions 50" >&2
  echo "  $0 games/Logs/frozen_lake/20260110_224101 metadata.py out 10 --first-all --prune" >&2
  echo "  $0 games/Logs/frozen_lake/20260110_224101 metadata.py out 10 --safety-only" >&2
  echo "  $0 games/Logs/frozen_lake/20260110_224101 metadata.py out 10 --liveness-only" >&2
  exit 1
fi

LOGS_DIR=$1
META_FILE=$2
OUT_DIR=${3:-"$LOGS_DIR/out"}
MAX_SIZE=${4:-10}
shift 4 2>/dev/null || shift $#

# Parse optional flags
FIRST_ALL=""
COLLECT_ALL=""
MAX_SOLUTIONS_FLAG=""
SELF_INPUTS_ONLY=""
PRUNE=""
SAFETY_ONLY=""
LIVENESS_ONLY=""
POS_DIR=""
NEG_DIR=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --first-all)
      FIRST_ALL="yes"
      shift
      ;;
    --collect-all)
      COLLECT_ALL="yes"
      shift
      ;;
    --max-solutions)
      MAX_SOLUTIONS_FLAG="--max-solutions $2"
      shift 2
      ;;
    --self-inputs-only)
      SELF_INPUTS_ONLY="--self-inputs-only"
      shift
      ;;
    --prune)
      PRUNE="yes"
      shift
      ;;
    --safety-only)
      SAFETY_ONLY="yes"
      shift
      ;;
    --liveness-only)
      LIVENESS_ONLY="yes"
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
      shift
      ;;
  esac
done

# Validate inputs
if [[ ! -d "$LOGS_DIR" ]]; then
  echo "[bolt_safety_liveness.sh] Error: Logs directory '$LOGS_DIR' does not exist" >&2
  exit 1
fi

if [[ ! -f "$META_FILE" ]]; then
  echo "[bolt_safety_liveness.sh] Error: Metadata file '$META_FILE' does not exist" >&2
  exit 1
fi

# Find Bolt binary - prioritize locally built version with --first-each support
BOLT_BIN=""
if [[ -f "/Users/samkouteili/rose/tsl-f/Bolt/target/release/bolt" ]]; then
  BOLT_BIN="/Users/samkouteili/rose/tsl-f/Bolt/target/release/bolt"
elif command -v bolt >/dev/null 2>&1; then
  BOLT_BIN="bolt"
else
  echo "[bolt_safety_liveness.sh] Error: 'bolt' not found. Build with: cd Bolt && cargo build --release" >&2
  exit 1
fi

# Find log2tslf.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG2TSLF_PY="$SCRIPT_DIR/log2tslf.py"

if [[ ! -f "$LOG2TSLF_PY" ]]; then
  echo "[bolt_safety_liveness.sh] Error: log2tslf.py not found at $LOG2TSLF_PY" >&2
  exit 1
fi

# Create output directory
mkdir -p "$OUT_DIR"

# ============================================
# Helper Functions (must be defined before use)
# ============================================

# Convert safety spec from "φ U (END)" to "G(φ)"
convert_safety_to_g() {
  local spec="$1"
  if [[ "$spec" =~ ^(.+)\ U\ \(END\)$ ]]; then
    local inner="${BASH_REMATCH[1]}"
    echo "G $inner"
  else
    echo "$spec"
  fi
}

# Count number of update arrows (<-) in a formula
# Use '<- ' (with space) to avoid matching <-> (biconditional)
count_updates() {
  local spec="$1"
  local count
  # grep returns 1 if no match, so we capture and handle it
  count=$(echo "$spec" | grep -o '<- ' 2>/dev/null | wc -l | tr -d ' ') || count=0
  echo "$count"
}

# Count number of unique predicates (eqC, neqC, etc.) in a formula
# E.g., "F ((eqC player goal) & (eqC player goal))" counts as 1, not 2
count_predicates() {
  local spec="$1"
  local count
  # Match full predicate expressions like "(eqC player goal)", then deduplicate
  count=$(echo "$spec" | grep -oE '\((eqC|neqC|ltC|gtC|leC|geC)[^)]+\)' 2>/dev/null | sort -u | wc -l | tr -d ' ') || count=0
  echo "$count"
}

# Select the best spec (works for both safety and liveness):
# 1. Primary: fewest updates (prefer predicate-based over update-based)
# 2. Secondary: most predicates (prefer specs that mention more state predicates)
select_best_spec() {
  local best_spec=""
  local best_updates=999999
  local best_predicates=-1

  while IFS= read -r spec; do
    if [[ -z "$spec" ]]; then
      continue
    fi
    local updates=$(count_updates "$spec")
    local predicates=$(count_predicates "$spec")

    # Prefer: fewest updates, then most predicates
    if [[ $updates -lt $best_updates ]] || \
       [[ $updates -eq $best_updates && $predicates -gt $best_predicates ]]; then
      best_updates=$updates
      best_predicates=$predicates
      best_spec="$spec"
    fi
  done

  echo "$best_spec"
}

# Alias for backwards compatibility
select_best_safety() {
  select_best_spec
}

# Parse updates.tsl file and return non-identity updates for each variable
# Format: "[var <- expr]" where expr != var (space-trimmed)
# Output: Newline-separated list of "variable:update_term" pairs
parse_non_identity_updates() {
  local updates_file="$1"
  if [[ ! -f "$updates_file" ]]; then
    return
  fi

  while IFS= read -r line; do
    if [[ -z "$line" ]]; then
      continue
    fi
    # Parse line format: [var <-  var] || [var <- func var] || ...
    # Split by ||
    IFS='|' read -ra terms <<< "$line"
    for term in "${terms[@]}"; do
      term="${term## }"  # trim leading spaces
      term="${term%% }"  # trim trailing spaces
      if [[ -z "$term" ]]; then
        continue
      fi
      # Extract variable and expression from [var <- expr]
      if [[ "$term" =~ ^\[([^\]]+)\ *\<-\ *([^\]]+)\]$ ]]; then
        local var="${BASH_REMATCH[1]}"
        local expr="${BASH_REMATCH[2]}"
        var="${var## }"  # trim
        var="${var%% }"
        expr="${expr## }"  # trim
        expr="${expr%% }"
        # Check if this is NOT an identity update (expr != var)
        if [[ "$expr" != "$var" ]]; then
          # Output the non-identity update
          echo "${var}:${term}"
        fi
      fi
    done
  done < "$updates_file"
}

# Check if a spec is spurious (contains ALL non-identity updates for any variable)
# Args: spec, non_identity_updates (newline-separated "var:update" pairs)
# Returns: 0 if spurious (should be pruned), 1 if not spurious
is_spurious_spec() {
  local spec="$1"
  local non_identity_updates="$2"

  # Group updates by variable
  declare -A var_updates
  declare -A var_update_count

  while IFS= read -r line; do
    if [[ -z "$line" ]]; then
      continue
    fi
    local var="${line%%:*}"
    local update="${line#*:}"

    if [[ -z "${var_updates[$var]:-}" ]]; then
      var_updates[$var]=""
      var_update_count[$var]=0
    fi
    var_updates[$var]+="$update"$'\n'
    ((var_update_count[$var]++)) || true
  done <<< "$non_identity_updates"

  # For each variable, check if ALL its non-identity updates are in the spec
  for var in "${!var_updates[@]}"; do
    local updates="${var_updates[$var]}"
    local count="${var_update_count[$var]}"

    if [[ $count -lt 2 ]]; then
      # Need at least 2 non-identity updates to be spurious
      continue
    fi

    local found_count=0
    while IFS= read -r update; do
      if [[ -z "$update" ]]; then
        continue
      fi
      # Check if this update appears in the spec
      if [[ "$spec" == *"$update"* ]]; then
        ((found_count++)) || true
      fi
    done <<< "$updates"

    # If ALL non-identity updates for this variable are in the spec, it's spurious
    if [[ $found_count -eq $count ]]; then
      return 0  # spurious
    fi
  done

  return 1  # not spurious
}

# Check if a spec contains tautological patterns like:
#   - (X) -> (X)        : implication with same antecedent and consequent
#   - (X) <-> (X)       : biconditional with same left and right
#   - (X) | (! (X))     : disjunction of atom and its negation
#   - (! (X)) | (X)     : disjunction of negated atom and atom
# Returns: 0 if tautological, 1 if not
is_tautological_spec() {
  local spec="$1"

  # Use Python for reliable parsing of nested parentheses
  python3 - "$spec" <<'PYEOF'
import sys
import re

def find_matching_paren(s, start):
    """Find the index of the closing paren matching the open paren at start."""
    if start >= len(s) or s[start] != '(':
        return -1
    depth = 1
    i = start + 1
    while i < len(s) and depth > 0:
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
        i += 1
    return i - 1 if depth == 0 else -1

def extract_paren_content(s, start):
    """Extract content inside parens starting at start, return (content, end_index)."""
    if start >= len(s) or s[start] != '(':
        return None, start
    end = find_matching_paren(s, start)
    if end == -1:
        return None, start
    return s[start+1:end], end

def is_tautological(spec):
    # Pattern 1 & 2: (X) -> (X) or (X) <-> (X)
    # Find ") -> (" or ") <-> (" patterns
    for op in [' -> ', ' <-> ']:
        idx = spec.find(op)
        while idx != -1:
            # Find the opening paren for left side (scan backwards)
            left_end = idx - 1
            while left_end >= 0 and spec[left_end] == ' ':
                left_end -= 1
            if left_end >= 0 and spec[left_end] == ')':
                # Find matching open paren
                depth = 1
                left_start = left_end - 1
                while left_start >= 0 and depth > 0:
                    if spec[left_start] == ')':
                        depth += 1
                    elif spec[left_start] == '(':
                        depth -= 1
                    left_start -= 1
                left_start += 1
                if depth == 0:
                    left_content = spec[left_start+1:left_end]
                    # Find right side
                    right_start = idx + len(op)
                    while right_start < len(spec) and spec[right_start] == ' ':
                        right_start += 1
                    if right_start < len(spec) and spec[right_start] == '(':
                        right_content, right_end = extract_paren_content(spec, right_start)
                        if right_content is not None and left_content == right_content:
                            return True
            idx = spec.find(op, idx + 1)

    # Pattern 3: (X) | (! (X))
    idx = spec.find(') | (! (')
    while idx != -1:
        # Extract left side
        left_end = idx
        depth = 1
        left_start = left_end - 1
        while left_start >= 0 and depth > 0:
            if spec[left_start] == ')':
                depth += 1
            elif spec[left_start] == '(':
                depth -= 1
            left_start -= 1
        left_start += 1
        if depth == 0:
            left_content = spec[left_start+1:left_end]
            # Extract right side (after "! (")
            right_start = idx + len(') | (! (') - 1
            if right_start < len(spec) and spec[right_start] == '(':
                right_content, _ = extract_paren_content(spec, right_start)
                if right_content is not None and left_content == right_content:
                    return True
        idx = spec.find(') | (! (', idx + 1)

    # Pattern 4: (! (X)) | (X)
    # Look for "(! (" which starts a negated expression
    idx = spec.find('(! (')
    while idx != -1:
        # Find the inner content of the negation
        inner_start = idx + 3  # position of inner "("
        inner_content, inner_end = extract_paren_content(spec, inner_start)
        if inner_content is not None:
            # After (! (X)) we expect ") | ("
            # inner_end is at the closing ")" of inner, then we have another ")" for the (! ...) wrapper
            after_neg = inner_end + 1
            if after_neg < len(spec) and spec[after_neg] == ')':
                # Now look for " | (" starting at position after_neg+1
                rest_start = after_neg + 1
                if spec[rest_start:rest_start+4] == ' | (':
                    right_start = rest_start + 3  # position of "(" in " | ("
                    right_content, _ = extract_paren_content(spec, right_start)
                    if right_content is not None and inner_content == right_content:
                        return True
        idx = spec.find('(! (', idx + 1)

    return False

spec = sys.argv[1]
sys.exit(0 if is_tautological(spec) else 1)
PYEOF
}

# Prune spurious safety specs from an array
# Args: array of specs (passed as positional args), updates_file path
# Output: Non-spurious specs, one per line
prune_spurious_specs() {
  local updates_file="$1"
  shift

  # Get non-identity updates
  local non_identity_updates
  non_identity_updates=$(parse_non_identity_updates "$updates_file")

  local spurious_count=0
  local tautology_count=0

  for spec in "$@"; do
    # Check for tautologies first (doesn't need updates file)
    if is_tautological_spec "$spec"; then
      ((tautology_count++)) || true
      continue  # Skip this spec
    fi

    # Check for spurious specs (needs updates file)
    if [[ -n "$non_identity_updates" ]] && is_spurious_spec "$spec" "$non_identity_updates"; then
      ((spurious_count++)) || true
      continue  # Skip this spec
    fi

    # Spec passed both checks
    echo "$spec"
  done

  if [[ $tautology_count -gt 0 ]]; then
    echo "[prune] Removed $tautology_count tautological specs (a->a, a<->a, a|!a)" >&2
  fi
  if [[ $spurious_count -gt 0 ]]; then
    echo "[prune] Removed $spurious_count spurious specs (all non-id updates for a variable)" >&2
  fi
}

# ============================================

echo "=============================================="
echo "[bolt_safety_liveness.sh] Mining Safety + Liveness Specifications"
echo "=============================================="
echo "Logs directory: $LOGS_DIR"
echo "Metadata file:  $META_FILE"
echo "Output dir:     $OUT_DIR"
echo "Max formula size: $MAX_SIZE"
echo "Bolt binary:    $BOLT_BIN"
if [[ -n "$SAFETY_ONLY" ]]; then
  echo "Search mode:    safety-only (G-rooted specs only)"
elif [[ -n "$LIVENESS_ONLY" ]]; then
  echo "Search mode:    liveness-only (F-rooted specs only)"
elif [[ -n "$FIRST_ALL" ]]; then
  echo "Search mode:    first-all (all specs at first size)"
elif [[ -n "$COLLECT_ALL" ]]; then
  echo "Search mode:    collect-all"
  if [[ -n "$MAX_SOLUTIONS_FLAG" ]]; then
    echo "Max solutions:  ${MAX_SOLUTIONS_FLAG#--max-solutions }"
  fi
else
  echo "Search mode:    first-each"
fi
if [[ -n "$SELF_INPUTS_ONLY" ]]; then
  echo "Input mode:     self-inputs-only (no cross-updates)"
fi
if [[ -n "$PRUNE" ]]; then
  echo "Pruning:        enabled (remove spurious safety specs)"
fi
if [[ -n "$POS_DIR" ]]; then
  echo "Positive traces: $POS_DIR"
fi
if [[ -n "$NEG_DIR" ]]; then
  echo "Negative traces: $NEG_DIR"
fi
echo ""

# Build pos/neg flags for log2tslf.py
POS_NEG_FLAGS=""
if [[ -n "$POS_DIR" ]]; then
  POS_NEG_FLAGS="$POS_NEG_FLAGS --pos $POS_DIR"
fi
if [[ -n "$NEG_DIR" ]]; then
  POS_NEG_FLAGS="$POS_NEG_FLAGS --neg $NEG_DIR"
fi

# Step 1: Generate bolt.json with log2tslf.py
echo "[Step 1] Generating boolean traces with log2tslf.py..."
python "$LOG2TSLF_PY" "$LOGS_DIR" "$META_FILE" --out "$OUT_DIR" $SELF_INPUTS_ONLY $POS_NEG_FLAGS

BOLT_JSON="$OUT_DIR/bolt.json"
if [[ ! -f "$BOLT_JSON" ]]; then
  echo "[bolt_safety_liveness.sh] Error: bolt.json not generated at $BOLT_JSON" >&2
  exit 1
fi

echo ""
if [[ -n "$SAFETY_ONLY" ]]; then
  echo "[Step 2] Mining ONLY safety specifications (G-rooted)..."
  if [[ -n "$COLLECT_ALL" ]]; then
    echo ""
    echo "Running: $BOLT_BIN $BOLT_JSON $MAX_SIZE $MAX_SIZE --safety --collect-all $MAX_SOLUTIONS_FLAG"
    BOLT_OUTPUT_SAFETY=$("$BOLT_BIN" "$BOLT_JSON" "$MAX_SIZE" "$MAX_SIZE" --safety --collect-all $MAX_SOLUTIONS_FLAG 2>&1)
  elif [[ -n "$FIRST_ALL" ]]; then
    echo ""
    echo "Running: $BOLT_BIN $BOLT_JSON $MAX_SIZE $MAX_SIZE --safety --first-all"
    BOLT_OUTPUT_SAFETY=$("$BOLT_BIN" "$BOLT_JSON" "$MAX_SIZE" "$MAX_SIZE" --safety --first-all 2>&1)
  else
    echo ""
    echo "Running: $BOLT_BIN $BOLT_JSON $MAX_SIZE $MAX_SIZE --safety"
    BOLT_OUTPUT_SAFETY=$("$BOLT_BIN" "$BOLT_JSON" "$MAX_SIZE" "$MAX_SIZE" --safety 2>&1)
  fi
  echo "$BOLT_OUTPUT_SAFETY"
  echo ""
elif [[ -n "$LIVENESS_ONLY" ]]; then
  echo "[Step 2] Mining ONLY liveness specifications (F-rooted)..."
  if [[ -n "$COLLECT_ALL" ]]; then
    echo ""
    echo "Running: $BOLT_BIN $BOLT_JSON $MAX_SIZE $MAX_SIZE --liveness --collect-all $MAX_SOLUTIONS_FLAG"
    BOLT_OUTPUT_LIVENESS=$("$BOLT_BIN" "$BOLT_JSON" "$MAX_SIZE" "$MAX_SIZE" --liveness --collect-all $MAX_SOLUTIONS_FLAG 2>&1)
  elif [[ -n "$FIRST_ALL" ]]; then
    echo ""
    echo "Running: $BOLT_BIN $BOLT_JSON $MAX_SIZE $MAX_SIZE --liveness --first-all"
    BOLT_OUTPUT_LIVENESS=$("$BOLT_BIN" "$BOLT_JSON" "$MAX_SIZE" "$MAX_SIZE" --liveness --first-all 2>&1)
  else
    echo ""
    echo "Running: $BOLT_BIN $BOLT_JSON $MAX_SIZE $MAX_SIZE --liveness"
    BOLT_OUTPUT_LIVENESS=$("$BOLT_BIN" "$BOLT_JSON" "$MAX_SIZE" "$MAX_SIZE" --liveness 2>&1)
  fi
  echo "$BOLT_OUTPUT_LIVENESS"
  echo ""
elif [[ -n "$FIRST_ALL" ]]; then
  echo "[Step 2] Mining specifications with --first-all..."
  echo "  Finding ALL specs at FIRST size where solutions are found"
  echo ""

  # Run Bolt twice with --first-all for both liveness and safety
  echo "  [2a] Mining liveness (F-rooted) formulas..."
  echo "  Running: $BOLT_BIN $BOLT_JSON $MAX_SIZE $MAX_SIZE --liveness --first-all"
  BOLT_OUTPUT_LIVENESS=$("$BOLT_BIN" "$BOLT_JSON" "$MAX_SIZE" "$MAX_SIZE" --liveness --first-all 2>&1)
  echo "$BOLT_OUTPUT_LIVENESS"
  echo ""

  echo "  [2b] Mining safety (G-rooted) formulas..."
  echo "  Running: $BOLT_BIN $BOLT_JSON $MAX_SIZE $MAX_SIZE --safety --first-all"
  BOLT_OUTPUT_SAFETY=$("$BOLT_BIN" "$BOLT_JSON" "$MAX_SIZE" "$MAX_SIZE" --safety --first-all 2>&1)
  echo "$BOLT_OUTPUT_SAFETY"
  echo ""
elif [[ -n "$COLLECT_ALL" ]]; then
  echo "[Step 2] Mining specifications with --collect-all..."
  echo "  Running two passes: --first-all for liveness, --collect-all for safety"
  echo ""

  # Run Bolt twice:
  # - Liveness: always uses --first-all (find all at first size, then rank)
  # - Safety: uses --collect-all (collect all up to max size)
  echo "  [2a] Mining liveness (F-rooted) formulas with --first-all..."
  echo "  Running: $BOLT_BIN $BOLT_JSON $MAX_SIZE $MAX_SIZE --liveness --first-all"
  BOLT_OUTPUT_LIVENESS=$("$BOLT_BIN" "$BOLT_JSON" "$MAX_SIZE" "$MAX_SIZE" --liveness --first-all 2>&1)
  echo "$BOLT_OUTPUT_LIVENESS"
  echo ""

  echo "  [2b] Mining safety (G-rooted, pure) formulas with --collect-all..."
  echo "  Running: $BOLT_BIN $BOLT_JSON $MAX_SIZE $MAX_SIZE --safety --collect-all $MAX_SOLUTIONS_FLAG"
  BOLT_OUTPUT_SAFETY=$("$BOLT_BIN" "$BOLT_JSON" "$MAX_SIZE" "$MAX_SIZE" --safety --collect-all $MAX_SOLUTIONS_FLAG 2>&1)
  echo "$BOLT_OUTPUT_SAFETY"
  echo ""
else
  # Default: find first of each (liveness and safety)
  echo "[Step 2] Mining specifications (first of each)..."
  echo ""

  echo "  [2a] Mining first liveness (F-rooted) formula..."
  echo "  Running: $BOLT_BIN $BOLT_JSON $MAX_SIZE $MAX_SIZE --liveness"
  BOLT_OUTPUT_LIVENESS=$("$BOLT_BIN" "$BOLT_JSON" "$MAX_SIZE" "$MAX_SIZE" --liveness 2>&1)
  echo "$BOLT_OUTPUT_LIVENESS"
  echo ""

  echo "  [2b] Mining first safety (G-rooted) formula..."
  echo "  Running: $BOLT_BIN $BOLT_JSON $MAX_SIZE $MAX_SIZE --safety"
  BOLT_OUTPUT_SAFETY=$("$BOLT_BIN" "$BOLT_JSON" "$MAX_SIZE" "$MAX_SIZE" --safety 2>&1)
  echo "$BOLT_OUTPUT_SAFETY"
  echo ""
fi

# Output files
LIVENESS_FILE="$OUT_DIR/liveness.tsl"
SAFETY_FILE="$OUT_DIR/safety.tsl"
COMBINED_FILE="$OUT_DIR/spec.tsl"
ALL_LIVENESS_FILE="$OUT_DIR/all_liveness.tsl"
ALL_SAFETY_FILE="$OUT_DIR/all_safety.tsl"

if [[ -n "$SAFETY_ONLY" ]]; then
  # Parse safety-only output
  echo "[Step 3] Parsing safety-only solutions..."

  ALL_SAFETY=()
  while IFS= read -r line; do
    if [[ "$line" =~ ^\[SOLUTION\ [0-9]+\]\ (.+)$ ]]; then
      ALL_SAFETY+=("${BASH_REMATCH[1]}")
    fi
  done <<< "$BOLT_OUTPUT_SAFETY"

  echo "  Found ${#ALL_SAFETY[@]} safety specs (G-rooted)"

  # Prune spurious safety specs if --prune is enabled
  if [[ -n "$PRUNE" && ${#ALL_SAFETY[@]} -gt 0 ]]; then
    UPDATES_FILE="$OUT_DIR/updates.tsl"
    echo ""
    echo "[Step 3.5] Pruning spurious safety specs..."
    PRUNED_SAFETY=()
    while IFS= read -r spec; do
      if [[ -n "$spec" ]]; then
        PRUNED_SAFETY+=("$spec")
      fi
    done < <(prune_spurious_specs "$UPDATES_FILE" "${ALL_SAFETY[@]}")
    echo "  Before pruning: ${#ALL_SAFETY[@]} specs"
    echo "  After pruning: ${#PRUNED_SAFETY[@]} specs"
    ALL_SAFETY=("${PRUNED_SAFETY[@]}")
  fi

  # Write safety specs
  echo "[Step 4] Writing specification files..."

  if [[ ${#ALL_SAFETY[@]} -gt 0 ]]; then
    printf '%s\n' "${ALL_SAFETY[@]}" > "$ALL_SAFETY_FILE"
    echo "  All safety: $ALL_SAFETY_FILE (${#ALL_SAFETY[@]} formulas)"

    # Select best safety spec (fewest updates = most predicate-based)
    BEST_SAFETY=$(printf '%s\n' "${ALL_SAFETY[@]}" | select_best_safety)
    BEST_SAFETY_UPDATES=$(count_updates "$BEST_SAFETY")

    echo "$BEST_SAFETY" > "$SAFETY_FILE"
    echo "  Best safety (fewest updates): $SAFETY_FILE"
    echo "    $BEST_SAFETY"
    echo "    (${BEST_SAFETY_UPDATES} update arrows)"

    # Convert to G format and write as combined spec
    SAFETY_G=$(convert_safety_to_g "$BEST_SAFETY")
    echo "$SAFETY_G" > "$COMBINED_FILE"
    echo "  Combined spec: $COMBINED_FILE"
  else
    echo "" > "$SAFETY_FILE"
    echo "" > "$ALL_SAFETY_FILE"
    echo "" > "$COMBINED_FILE"
    echo "  Safety: NOT FOUND"
  fi

  # Clear liveness files since we're safety-only
  echo "" > "$LIVENESS_FILE"
  echo "" > "$ALL_LIVENESS_FILE"

elif [[ -n "$LIVENESS_ONLY" ]]; then
  # Parse liveness-only output (uses [SOLUTION N] format)
  echo "[Step 3] Parsing liveness-only solutions..."

  ALL_LIVENESS=()
  while IFS= read -r line; do
    if [[ "$line" =~ ^\[SOLUTION\ [0-9]+\]\ (.+)$ ]]; then
      ALL_LIVENESS+=("${BASH_REMATCH[1]}")
    fi
  done <<< "$BOLT_OUTPUT_LIVENESS"

  echo "  Found ${#ALL_LIVENESS[@]} liveness specs at first size (F-rooted)"

  # Write liveness specs
  echo "[Step 4] Writing specification files..."

  if [[ ${#ALL_LIVENESS[@]} -gt 0 ]]; then
    printf '%s\n' "${ALL_LIVENESS[@]}" > "$ALL_LIVENESS_FILE"
    echo "  All liveness: $ALL_LIVENESS_FILE (${#ALL_LIVENESS[@]} formulas)"

    # Select best liveness spec (fewest updates = most predicate-based)
    BEST_LIVENESS=$(printf '%s\n' "${ALL_LIVENESS[@]}" | select_best_spec)
    BEST_LIVENESS_UPDATES=$(count_updates "$BEST_LIVENESS")

    echo "$BEST_LIVENESS" > "$LIVENESS_FILE"
    echo "  Best liveness (fewest updates): $LIVENESS_FILE"
    echo "    $BEST_LIVENESS"
    echo "    (${BEST_LIVENESS_UPDATES} update arrows)"

    # Show ranking of all liveness specs
    echo ""
    echo "  Liveness specs ranked (fewest updates, then most predicates):"
    for spec in "${ALL_LIVENESS[@]}"; do
      updates=$(count_updates "$spec")
      predicates=$(count_predicates "$spec")
      if [[ "$spec" == "$BEST_LIVENESS" ]]; then
        echo "    [SELECTED] ($updates updates, $predicates predicates) $spec"
      else
        echo "    ($updates updates, $predicates predicates) $spec"
      fi
    done

    # Write as combined spec
    echo "$BEST_LIVENESS" > "$COMBINED_FILE"
    echo "  Combined spec: $COMBINED_FILE"
  else
    echo "" > "$LIVENESS_FILE"
    echo "" > "$ALL_LIVENESS_FILE"
    echo "" > "$COMBINED_FILE"
    echo "  Liveness: NOT FOUND"
  fi

  # Clear safety files since we're liveness-only
  echo "" > "$SAFETY_FILE"
  echo "" > "$ALL_SAFETY_FILE"

elif [[ -n "$FIRST_ALL" ]]; then
  # Parse --first-all outputs from the two separate Bolt runs
  echo "[Step 3] Parsing first-all solutions..."

  ALL_LIVENESS=()
  ALL_SAFETY=()

  # Parse liveness output (uses [SOLUTION N] format)
  while IFS= read -r line; do
    if [[ "$line" =~ ^\[SOLUTION\ [0-9]+\]\ (.+)$ ]]; then
      ALL_LIVENESS+=("${BASH_REMATCH[1]}")
    fi
  done <<< "$BOLT_OUTPUT_LIVENESS"

  # Parse safety output (uses [SOLUTION N] format)
  while IFS= read -r line; do
    if [[ "$line" =~ ^\[SOLUTION\ [0-9]+\]\ (.+)$ ]]; then
      ALL_SAFETY+=("${BASH_REMATCH[1]}")
    fi
  done <<< "$BOLT_OUTPUT_SAFETY"

  echo "  Found ${#ALL_LIVENESS[@]} liveness specs at first size (F-rooted)"
  echo "  Found ${#ALL_SAFETY[@]} safety specs at first size (G-rooted)"

  # Prune spurious safety specs if --prune is enabled
  if [[ -n "$PRUNE" && ${#ALL_SAFETY[@]} -gt 0 ]]; then
    UPDATES_FILE="$OUT_DIR/updates.tsl"
    echo ""
    echo "[Step 3.5] Pruning spurious safety specs..."
    PRUNED_SAFETY=()
    while IFS= read -r spec; do
      if [[ -n "$spec" ]]; then
        PRUNED_SAFETY+=("$spec")
      fi
    done < <(prune_spurious_specs "$UPDATES_FILE" "${ALL_SAFETY[@]}")
    echo "  Before pruning: ${#ALL_SAFETY[@]} specs"
    echo "  After pruning: ${#PRUNED_SAFETY[@]} specs"
    ALL_SAFETY=("${PRUNED_SAFETY[@]}")
  fi

  # Write all specs to files
  echo "[Step 4] Writing specification files..."

  # Write all liveness and select best
  if [[ ${#ALL_LIVENESS[@]} -gt 0 ]]; then
    printf '%s\n' "${ALL_LIVENESS[@]}" > "$ALL_LIVENESS_FILE"
    echo "  All liveness: $ALL_LIVENESS_FILE (${#ALL_LIVENESS[@]} formulas)"

    # Select best liveness spec (fewest updates = most predicate-based)
    BEST_LIVENESS=$(printf '%s\n' "${ALL_LIVENESS[@]}" | select_best_spec)
    BEST_LIVENESS_UPDATES=$(count_updates "$BEST_LIVENESS")

    echo "$BEST_LIVENESS" > "$LIVENESS_FILE"
    echo "  Best liveness (fewest updates): $LIVENESS_FILE"
    echo "    $BEST_LIVENESS"
    echo "    (${BEST_LIVENESS_UPDATES} update arrows)"

    # Show ranking of all liveness specs
    echo ""
    echo "  Liveness specs ranked (fewest updates, then most predicates):"
    for spec in "${ALL_LIVENESS[@]}"; do
      updates=$(count_updates "$spec")
      predicates=$(count_predicates "$spec")
      if [[ "$spec" == "$BEST_LIVENESS" ]]; then
        echo "    [SELECTED] ($updates updates, $predicates predicates) $spec"
      else
        echo "    ($updates updates, $predicates predicates) $spec"
      fi
    done
  else
    echo "" > "$LIVENESS_FILE"
    echo "" > "$ALL_LIVENESS_FILE"
    echo "  Liveness: NOT FOUND"
    BEST_LIVENESS=""
  fi

  # Write all safety and select best
  if [[ ${#ALL_SAFETY[@]} -gt 0 ]]; then
    printf '%s\n' "${ALL_SAFETY[@]}" > "$ALL_SAFETY_FILE"
    echo "  All safety: $ALL_SAFETY_FILE (${#ALL_SAFETY[@]} formulas)"

    # Select best safety spec (fewest updates = most predicate-based)
    BEST_SAFETY=$(printf '%s\n' "${ALL_SAFETY[@]}" | select_best_spec)
    BEST_SAFETY_UPDATES=$(count_updates "$BEST_SAFETY")

    echo "$BEST_SAFETY" > "$SAFETY_FILE"
    echo "  Best safety (fewest updates): $SAFETY_FILE"
    echo "    $BEST_SAFETY"
    echo "    (${BEST_SAFETY_UPDATES} update arrows)"

    # Show ranking of all safety specs
    echo ""
    echo "  Safety specs ranked (fewest updates, then most predicates):"
    for spec in "${ALL_SAFETY[@]}"; do
      updates=$(count_updates "$spec")
      predicates=$(count_predicates "$spec")
      if [[ "$spec" == "$BEST_SAFETY" ]]; then
        echo "    [SELECTED] ($updates updates, $predicates predicates) $spec"
      else
        echo "    ($updates updates, $predicates predicates) $spec"
      fi
    done
  else
    echo "" > "$SAFETY_FILE"
    echo "" > "$ALL_SAFETY_FILE"
    echo "  Safety: NOT FOUND"
    BEST_SAFETY=""
  fi

  # Combined spec (best of each)
  echo ""
  echo "[Step 5] Combining best specifications..."
  LIVENESS_SPEC="${BEST_LIVENESS:-}"
  SAFETY_SPEC="${BEST_SAFETY:-}"

elif [[ -n "$COLLECT_ALL" ]]; then
  # Parse collect-all outputs from the two separate Bolt runs
  # Both use [SOLUTION N] format
  echo "[Step 3] Parsing collected solutions..."

  ALL_LIVENESS=()
  ALL_SAFETY=()

  # Parse liveness output (uses [SOLUTION N] format)
  while IFS= read -r line; do
    if [[ "$line" =~ ^\[SOLUTION\ [0-9]+\]\ (.+)$ ]]; then
      ALL_LIVENESS+=("${BASH_REMATCH[1]}")
    fi
  done <<< "$BOLT_OUTPUT_LIVENESS"

  # Parse safety output (uses [SOLUTION N] format)
  while IFS= read -r line; do
    if [[ "$line" =~ ^\[SOLUTION\ [0-9]+\]\ (.+)$ ]]; then
      ALL_SAFETY+=("${BASH_REMATCH[1]}")
    fi
  done <<< "$BOLT_OUTPUT_SAFETY"

  echo "  Found ${#ALL_LIVENESS[@]} liveness specs at first size (F-rooted)"
  echo "  Found ${#ALL_SAFETY[@]} safety specs (G-rooted, pure - no F/U/R/END)"

  # Prune spurious safety specs if --prune is enabled
  if [[ -n "$PRUNE" && ${#ALL_SAFETY[@]} -gt 0 ]]; then
    UPDATES_FILE="$OUT_DIR/updates.tsl"
    echo ""
    echo "[Step 3.5] Pruning spurious safety specs..."
    PRUNED_SAFETY=()
    while IFS= read -r spec; do
      if [[ -n "$spec" ]]; then
        PRUNED_SAFETY+=("$spec")
      fi
    done < <(prune_spurious_specs "$UPDATES_FILE" "${ALL_SAFETY[@]}")
    echo "  Before pruning: ${#ALL_SAFETY[@]} specs"
    echo "  After pruning: ${#PRUNED_SAFETY[@]} specs"
    ALL_SAFETY=("${PRUNED_SAFETY[@]}")
  fi

  # Write all specs to files
  echo "[Step 4] Writing specification files..."

  # Write all liveness and select best
  if [[ ${#ALL_LIVENESS[@]} -gt 0 ]]; then
    printf '%s\n' "${ALL_LIVENESS[@]}" > "$ALL_LIVENESS_FILE"
    echo "  All liveness: $ALL_LIVENESS_FILE (${#ALL_LIVENESS[@]} formulas)"

    # Select best liveness spec (fewest updates = most predicate-based)
    BEST_LIVENESS=$(printf '%s\n' "${ALL_LIVENESS[@]}" | select_best_spec)
    BEST_LIVENESS_UPDATES=$(count_updates "$BEST_LIVENESS")

    echo "$BEST_LIVENESS" > "$LIVENESS_FILE"
    echo "  Best liveness (fewest updates): $LIVENESS_FILE"
    echo "    $BEST_LIVENESS"
    echo "    (${BEST_LIVENESS_UPDATES} update arrows)"

    # Show ranking of all liveness specs
    echo ""
    echo "  Liveness specs ranked (fewest updates, then most predicates):"
    for spec in "${ALL_LIVENESS[@]}"; do
      updates=$(count_updates "$spec")
      predicates=$(count_predicates "$spec")
      if [[ "$spec" == "$BEST_LIVENESS" ]]; then
        echo "    [SELECTED] ($updates updates, $predicates predicates) $spec"
      else
        echo "    ($updates updates, $predicates predicates) $spec"
      fi
    done
  else
    echo "" > "$LIVENESS_FILE"
    echo "" > "$ALL_LIVENESS_FILE"
    echo "  Liveness: NOT FOUND"
    BEST_LIVENESS=""
  fi

  # Write all safety and select best
  if [[ ${#ALL_SAFETY[@]} -gt 0 ]]; then
    printf '%s\n' "${ALL_SAFETY[@]}" > "$ALL_SAFETY_FILE"
    echo "  All safety: $ALL_SAFETY_FILE (${#ALL_SAFETY[@]} formulas)"

    # Select best safety spec (fewest updates = most predicate-based)
    BEST_SAFETY=$(printf '%s\n' "${ALL_SAFETY[@]}" | select_best_spec)
    BEST_SAFETY_UPDATES=$(count_updates "$BEST_SAFETY")

    echo "$BEST_SAFETY" > "$SAFETY_FILE"
    echo "  Best safety (fewest updates): $SAFETY_FILE"
    echo "    $BEST_SAFETY"
    echo "    (${BEST_SAFETY_UPDATES} update arrows)"

    # Show ranking of all safety specs
    echo ""
    echo "  Safety specs ranked (fewest updates, then most predicates):"
    for spec in "${ALL_SAFETY[@]}"; do
      updates=$(count_updates "$spec")
      predicates=$(count_predicates "$spec")
      if [[ "$spec" == "$BEST_SAFETY" ]]; then
        echo "    [SELECTED] ($updates updates, $predicates predicates) $spec"
      else
        echo "    ($updates updates, $predicates predicates) $spec"
      fi
    done
  else
    echo "" > "$SAFETY_FILE"
    echo "" > "$ALL_SAFETY_FILE"
    echo "  Safety: NOT FOUND"
    BEST_SAFETY=""
  fi

  # Combined spec (best of each)
  echo ""
  echo "[Step 5] Combining best specifications..."
  LIVENESS_SPEC="${BEST_LIVENESS:-}"
  SAFETY_SPEC="${BEST_SAFETY:-}"

else
  # Parse first-of-each outputs from the two separate Bolt runs
  echo "[Step 3] Parsing first-of-each solutions..."

  LIVENESS_SPEC=""
  SAFETY_SPEC=""

  # Parse liveness output - look for [SOLUTION 0] (first formula)
  while IFS= read -r line; do
    if [[ "$line" =~ ^\[SOLUTION\ 0\]\ (.+)$ ]]; then
      LIVENESS_SPEC="${BASH_REMATCH[1]}"
      break
    fi
  done <<< "$BOLT_OUTPUT_LIVENESS"

  # Parse safety output - look for [SOLUTION 0] (first formula)
  while IFS= read -r line; do
    if [[ "$line" =~ ^\[SOLUTION\ 0\]\ (.+)$ ]]; then
      SAFETY_SPEC="${BASH_REMATCH[1]}"
      break
    fi
  done <<< "$BOLT_OUTPUT_SAFETY"

  # Step 3: Write individual spec files
  echo "[Step 4] Writing specification files..."

  if [[ -n "$LIVENESS_SPEC" ]]; then
    echo "$LIVENESS_SPEC" > "$LIVENESS_FILE"
    echo "  Liveness spec: $LIVENESS_FILE"
    echo "    $LIVENESS_SPEC"
  else
    echo "  Liveness spec: NOT FOUND"
    echo "" > "$LIVENESS_FILE"
  fi

  if [[ -n "$SAFETY_SPEC" ]]; then
    echo "$SAFETY_SPEC" > "$SAFETY_FILE"
    echo "  Safety spec:   $SAFETY_FILE"
    echo "    $SAFETY_SPEC"
  else
    echo "  Safety spec:   NOT FOUND"
    echo "" > "$SAFETY_FILE"
  fi

  echo ""
  echo "[Step 5] Combining specifications..."
fi

# Write combined spec (only for modes that set LIVENESS_SPEC/SAFETY_SPEC)
# Skip for safety-only and liveness-only modes (they handle their own output)
if [[ -z "$SAFETY_ONLY" && -z "$LIVENESS_ONLY" ]]; then
  if [[ -n "$LIVENESS_SPEC" && -n "$SAFETY_SPEC" ]]; then
    # Convert safety from "φ U END" to "G(φ)"
    SAFETY_G=$(convert_safety_to_g "$SAFETY_SPEC")
    echo "  Converting safety: φ U END → G(φ)"
    echo "    Original: $SAFETY_SPEC"
    echo "    Converted: $SAFETY_G"
    COMBINED="($LIVENESS_SPEC) & ($SAFETY_G)"
    echo "$COMBINED" > "$COMBINED_FILE"
    echo "  Combined spec: $COMBINED_FILE"
    echo "    $COMBINED"
  elif [[ -n "$LIVENESS_SPEC" ]]; then
    echo "$LIVENESS_SPEC" > "$COMBINED_FILE"
    echo "  Combined spec (liveness only): $COMBINED_FILE"
  elif [[ -n "$SAFETY_SPEC" ]]; then
    SAFETY_G=$(convert_safety_to_g "$SAFETY_SPEC")
    echo "$SAFETY_G" > "$COMBINED_FILE"
    echo "  Combined spec (safety only, converted to G): $COMBINED_FILE"
  else
    echo "  WARNING: No specifications found!"
    echo "" > "$COMBINED_FILE"
  fi
fi

echo ""
echo "=============================================="
echo "[bolt_safety_liveness.sh] Complete!"
echo "=============================================="
echo "Output files:"
echo "  $LIVENESS_FILE"
echo "  $SAFETY_FILE"
echo "  $COMBINED_FILE"
if [[ -n "$FIRST_ALL" || -n "$COLLECT_ALL" ]]; then
  echo "  $ALL_LIVENESS_FILE"
  echo "  $ALL_SAFETY_FILE"
fi
echo "  $BOLT_JSON"
