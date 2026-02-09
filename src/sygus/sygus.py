import json
import subprocess
import tempfile
import os
import sys
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
from itertools import product

# ========================================================================
#   SYGUS TEMPLATES (unchanged)
# ========================================================================

## full cover
SINGLE_ARITY_TEMPLATE = """(set-logic NIA)

(synth-fun f ((x Int)) Int
  ((I Int) (C Int))
  (
    (I Int (
        x
        (+ I I)
        (- I I)
        (+ I C)
        (- I C)
    ))
    (C Int (
        0
        1
        (- 0 1)
        (+ C C)
        (- C C)
    ))
  )
)

{constraints}

(check-synth)
"""

BINARY_ARITY_TEMPLATE = """(set-logic NIA)

(synth-fun f ((x Int) (y Int)) Int
  ((I Int) (C Int))
  (
    (I Int (
        x
        y
        (+ I I)
        (- I I)
        (+ I C)
        (- I C)
        (+ C I)
        (- C I)
    ))
    (C Int (
        0
        1
        (- 0 1)
        (+ C C)
        (- C C)
    ))
  )
)

{constraints}

(check-synth)
"""

TIMEOUT = 0.1  # seconds per solver call

# ========================================================================
#   HELPERS
# ========================================================================

def format_sygus_int(n):
    """Format an integer for SyGuS - negative numbers need (- 0 n) syntax."""
    if n < 0:
        return f"(- 0 {abs(n)})"
    return str(n)


def generate_constraints(records: List[Dict]) -> str:
    """Generate SyGuS constraints from a list of records."""
    constraints = []
    for rec in records:
        inp = rec["input"]
        out = rec["output"]

        if isinstance(inp, list):
            inp0 = format_sygus_int(inp[0])
            inp1 = format_sygus_int(inp[1])
            out_str = format_sygus_int(out)
            constraints.append(f"(constraint (= (f {inp0} {inp1}) {out_str}))")
        else:
            inp_str = format_sygus_int(inp)
            out_str = format_sygus_int(out)
            constraints.append(f"(constraint (= (f {inp_str}) {out_str}))")

    return "\n".join(constraints)


def extract_define_fun(out: str) -> Optional[str]:
    """Extract the full (define-fun ...) s-expression."""
    lines = out.split("\n")

    start = None
    paren_count = 0
    buf = []

    for i, line in enumerate(lines):
        if "(define-fun" in line:
            start = i
            paren_count = line.count("(") - line.count(")")
            buf.append(line)
            break

    if start is None:
        return None

    for line in lines[start+1:]:
        paren_count += line.count("(")
        paren_count -= line.count(")")
        buf.append(line)
        if paren_count == 0:
            break

    return "\n".join(buf)


def is_identity(rec: Dict) -> bool:
    """Check if a record is an identity mapping."""
    return rec["input"] == rec["output"]


def get_arity(rec: Dict) -> int:
    """Determine arity from record."""
    return 2 if isinstance(rec["input"], list) else 1


def synthesize_function(records: List[Dict], timeout: float = TIMEOUT) -> Optional[str]:
    """
    Synthesize a function for a list of records using CVC5.
    Returns the define-fun string or None if synthesis fails.
    """
    if not records:
        return None

    # Check if all records are identity
    if all(is_identity(rec) for rec in records):
        # Return identity function
        arity = get_arity(records[0])
        if arity == 1:
            return "(define-fun f ((x Int)) Int x)"
        else:
            return "(define-fun f ((x Int) (y Int)) Int x)"  # or y?

    # Check arity consistency
    arity = get_arity(records[0])
    if not all(get_arity(r) == arity for r in records):
        return None  # Inconsistent arity

    # Generate constraints for ALL records (including identity)
    constraints = generate_constraints(records)
    if not constraints:
        # Only identity constraints
        if arity == 1:
            return "(define-fun f ((x Int)) Int x)"
        else:
            return "(define-fun f ((x Int) (y Int)) Int x)"

    sygus = (SINGLE_ARITY_TEMPLATE if arity == 1 else BINARY_ARITY_TEMPLATE).format(
        constraints=constraints
    )

    with tempfile.NamedTemporaryFile(mode='w', suffix='.sy', delete=False) as f:
        f.write(sygus)
        path = f.name

    try:
        result = subprocess.run(
            ['cvc5', '--lang=sygus2', path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        out = result.stdout.strip()

        if "(define-fun" in out and "error" not in out.lower():
            func = extract_define_fun(out)
            return func
        return None

    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        print("ERROR: CVC5 not found in PATH")
        return None
    except Exception:
        return None
    finally:
        os.unlink(path)


def normalize_function(func_str: str) -> str:
    """
    Normalize function string for comparison.
    Simplify whitespace and format for consistent matching.
    """
    return " ".join(func_str.split())


# ========================================================================
#   BOTTOM-UP SYNTHESIS WITH INPUT SWAPPING
# ========================================================================

def load_baseline_and_alternatives(trace_dir: str) -> Tuple[Dict, Dict]:
    """
    Load baseline grouping and alternatives map from new format.

    Returns:
        baseline_grouping: dict mapping "time_X__var" to record
        alternatives_map: {(time_part, var_part): [list of input→output records]}
    """
    groupings_path = os.path.join(trace_dir, "groupings.jsonl")
    alternatives_path = os.path.join(trace_dir, "alternatives.json")

    # Load baseline grouping (only one line)
    with open(groupings_path) as f:
        baseline_grouping = json.loads(f.readline())

    # Load alternatives map
    with open(alternatives_path) as f:
        alternatives_raw = json.load(f)

    # Convert keys from "time_X__var" to ("time_X", "var") tuples
    alternatives_map = {}
    for key, records in alternatives_raw.items():
        parts = key.split("__")
        if len(parts) == 2:
            time_part, var_part = parts
            alternatives_map[(time_part, var_part)] = records

    return baseline_grouping, alternatives_map


def bottom_up_synthesis(trace_dir: str, timeout: float = TIMEOUT, self_inputs_only: bool = False) -> Optional[Dict]:
    """
    Main bottom-up synthesis algorithm with optional input swapping.

    Args:
        trace_dir: Path to trace directory containing groupings.jsonl and alternatives.json
        timeout: CVC5 timeout per call
        self_inputs_only: If True, skip alternative input swapping (only use self-inputs)

    Returns dict with:
        - functions: list of synthesized define-fun strings
        - assignments: dict mapping keys to {function_id, record}
    """
    # Step 1: Load baseline grouping and alternatives map
    base_grouping, alternatives_map = load_baseline_and_alternatives(trace_dir)

    if not base_grouping:
        print("No baseline grouping found")
        return None

    total_alternatives = sum(len(alts) for alts in alternatives_map.values())
    print(f"Loaded baseline grouping with {len(base_grouping)} slots")
    print(f"Total alternatives available: {total_alternatives}")

    # Step 2: Start with baseline grouping
    current_records = dict(base_grouping)  # key → record (mutable, we'll swap inputs)

    # Step 3: Singleton synthesis
    print("\n=== Singleton Synthesis ===")
    singleton_functions = {}  # key → function_str

    for key, record in current_records.items():
        if is_identity(record):
            singleton_functions[key] = "IDENTITY"
            print(f"  {key}: IDENTITY")
        else:
            func = synthesize_function([record], timeout)
            if func:
                singleton_functions[key] = normalize_function(func)
                print(f"  {key}: {singleton_functions[key][:50]}...")
            else:
                print(f"  {key}: FAILED")
                singleton_functions[key] = None

    # Step 4: Group by function signature
    print("\n=== Grouping by Function ===")
    function_groups = defaultdict(list)  # function_str → [keys]

    for key, func in singleton_functions.items():
        if func is not None:
            function_groups[func].append(key)

    for func_sig, keys in sorted(function_groups.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {func_sig[:60]:60s} → {len(keys)} keys")

    # Step 5: Bottom-up merging
    print("\n=== Bottom-Up Merging ===")
    max_iterations = 20
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\nIteration {iteration}:")

        # Sort groups by size (smallest first)
        sorted_groups = sorted(function_groups.items(), key=lambda x: len(x[1]))

        if not sorted_groups:
            break

        merged_any = False

        # Try to merge smallest groups
        for i, (small_func, small_keys) in enumerate(sorted_groups[:-1]):  # Skip largest
            if not small_keys:  # Empty group
                continue

            print(f"  Trying to merge group '{small_func[:40]}' (size {len(small_keys)})")

            # Try merging with larger groups
            merge_succeeded = False
            for large_func, large_keys in sorted_groups[i+1:]:
                if not large_keys:
                    continue

                # Try merge
                combined_keys = small_keys + large_keys
                combined_records = [current_records[k] for k in combined_keys]

                merged_func = synthesize_function(combined_records, timeout)

                if merged_func:
                    print(f"    ✓ Merged with '{large_func[:40]}' (size {len(large_keys)})")
                    # Merge successful!
                    function_groups[large_func].extend(small_keys)
                    function_groups[small_func] = []  # Clear small group
                    merged_any = True
                    merge_succeeded = True
                    break

            if merge_succeeded:
                continue

            # Merge failed with current inputs, try alternative combinations
            # Skip alternatives if self_inputs_only mode is enabled
            if self_inputs_only:
                print(f"    Merge failed (self-inputs-only mode, skipping alternatives)")
                continue

            print(f"    Merge failed, trying alternative input combinations...")

            # Collect alternatives for each key in small group
            key_alternatives = []  # List of [alternatives for key1, alternatives for key2, ...]
            for key in small_keys:
                parts = key.split("__")
                if len(parts) != 2:
                    key_alternatives.append([current_records[key]])
                    continue

                lookup_key = (parts[0], parts[1])
                alternatives = alternatives_map.get(lookup_key, [current_records[key]])
                key_alternatives.append(alternatives)

            # Calculate Cartesian product size
            product_size = 1
            for alts in key_alternatives:
                product_size *= len(alts)

            print(f"    Trying {product_size} unique combinations ({' × '.join(str(len(a)) for a in key_alternatives)})")

            if product_size > 64:
                print(f"    Product size {product_size} too large (>64), skipping alternatives")
                continue

            # Try all combinations in Cartesian product
            swap_succeeded = False
            for combination in product(*key_alternatives):
                # combination is a tuple of records, one for each key
                # Update current_records with this combination
                old_records = {}
                for key, new_record in zip(small_keys, combination):
                    old_records[key] = current_records[key]
                    current_records[key] = new_record

                # Try merging with each larger group
                for large_func, large_keys in sorted_groups[i+1:]:
                    if not large_keys:
                        continue

                    combined_keys = small_keys + large_keys
                    combined_records = [current_records[k] for k in combined_keys]

                    merged_func = synthesize_function(combined_records, timeout)

                    if merged_func:
                        print(f"    ✓ Found working combination, merged with '{large_func[:40]}'")
                        # Success! Update groups
                        for key in small_keys:
                            function_groups[small_func].remove(key)
                            function_groups[large_func].append(key)
                        merged_any = True
                        swap_succeeded = True
                        break

                if swap_succeeded:
                    break

                # Restore old records if this combination didn't work
                for key in small_keys:
                    current_records[key] = old_records[key]

            if not swap_succeeded:
                # Restore original records
                for key in small_keys:
                    if key in old_records:
                        current_records[key] = old_records[key]

        if not merged_any:
            print("  No merges succeeded, stopping")
            break

    # Step 6: Collect final functions
    print("\n=== Final Functions ===")
    final_functions = []
    final_assignments = {}
    function_id_map = {}  # function_str → function_id

    for func_str, keys in function_groups.items():
        if not keys:
            continue

        if func_str not in function_id_map:
            function_id_map[func_str] = len(final_functions)
            final_functions.append(func_str)

        func_id = function_id_map[func_str]

        for key in keys:
            final_assignments[key] = {
                "function_id": func_id,
                "record": current_records[key]
            }

    for i, func in enumerate(final_functions):
        count = sum(1 for a in final_assignments.values() if a["function_id"] == i)
        print(f"  f_{i}: {func[:60]} ({count} keys)")

    return {
        "functions": final_functions,
        "assignments": final_assignments
    }


# ========================================================================
#   PROCESS TRACE DIRECTORY
# ========================================================================

def process_single_trace(trace_dir: str, timeout: float = TIMEOUT, self_inputs_only: bool = False):
    """Process a single trace directory."""
    groupings_path = os.path.join(trace_dir, "groupings.jsonl")
    alternatives_path = os.path.join(trace_dir, "alternatives.json")
    output_path = os.path.join(trace_dir, "output_funcs.jsonl")

    if not os.path.exists(groupings_path) or not os.path.exists(alternatives_path):
        print(f"[SKIP] Missing groupings.jsonl or alternatives.json in {trace_dir}")
        return

    print(f"\n{'='*80}")
    print(f"Processing: {trace_dir}")
    print('='*80)

    result = bottom_up_synthesis(trace_dir, timeout, self_inputs_only)

    if result:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Success! Wrote {output_path}")
    else:
        print(f"\n✗ Failed to synthesize functions for {trace_dir}")


def process_all_traces(root_dir: str, timeout: float = TIMEOUT, self_inputs_only: bool = False):
    """Process all trace directories in root_dir."""
    for name in sorted(os.listdir(root_dir)):
        trace_dir = os.path.join(root_dir, name)
        if os.path.isdir(trace_dir):
            process_single_trace(trace_dir, timeout, self_inputs_only)


# ========================================================================
#   MAIN
# ========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", required=True,
        help="Directory containing subdirectories with groupings.jsonl files"
    )
    parser.add_argument(
        "--timeout", type=float, default=TIMEOUT,
        help="Timeout per CVC5 call in seconds"
    )
    parser.add_argument(
        "--self-inputs-only", action="store_true",
        help="Only use self-inputs (playerX <- f(playerX)), skip alternative input swapping"
    )
    args = parser.parse_args()

    process_all_traces(args.root_dir, args.timeout, getattr(args, 'self_inputs_only', False))
