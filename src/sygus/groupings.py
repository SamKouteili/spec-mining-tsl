"""
Groupings Module - Extract I/O pairs and create groupings for SyGuS synthesis.

This module combines the functionality of:
- IOSeparation: Extract input→output pairs from traces
- CreateGroupings: Create baseline groupings and alternatives for bottom-up synthesis

Usage:
    python groupings.py --input_dir <traces> --output_dir <out> [--self-inputs-only]
"""

import os
import json
import argparse
import itertools
import shutil
from collections import defaultdict


# ============================================================================
# Utility Functions
# ============================================================================

def clean_line(raw):
    return (
        raw.replace("\ufeff", "")
           .replace("\xa0", " ")
           .strip()
    )


def is_tuple_value(value):
    """Check if a value is a tuple (represented as a list in JSON)."""
    return isinstance(value, list) and len(value) >= 2 and all(isinstance(x, (int, float)) for x in value)


def is_boolean_value(value):
    """Check if a value is a boolean (true/false in JSON becomes Python bool)."""
    return isinstance(value, bool)


def expand_tuple_keys(obj: dict) -> dict:
    """
    Expand tuple values into separate element keys.
    e.g., {"player": [0, 1]} -> {"player[0]": 0, "player[1]": 1}
    Non-tuple values are kept as-is.
    """
    expanded = {}
    for key, value in obj.items():
        if is_tuple_value(value):
            for i, elem in enumerate(value):
                expanded[f"{key}[{i}]"] = elem
        else:
            expanded[key] = value
    return expanded


def get_tuple_structure(file_path) -> dict[str, int]:
    """
    Detect which variables are tuples and their arities.
    Returns a dict mapping tuple variable names to their lengths.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            clean = clean_line(raw)
            if not clean:
                continue
            try:
                obj = json.loads(clean)
            except Exception:
                continue
            if isinstance(obj, dict):
                tuples = {}
                for key, value in obj.items():
                    if is_tuple_value(value):
                        tuples[key] = len(value)
                return tuples
    return {}


def extract_first_line_keys(file_path, expand_tuples=True) -> set[str]:
    """
    Parse the first non-empty JSON line in file_path and return its keys as a set.
    If expand_tuples=True, tuple values are expanded to element keys.
    Returns an empty set if no valid JSON object line is found.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            clean = clean_line(raw)
            if not clean:
                continue
            try:
                obj = json.loads(clean)
            except Exception:
                continue
            if isinstance(obj, dict):
                if expand_tuples:
                    obj = expand_tuple_keys(obj)
                return set(obj.keys())
            return set()
    return set()


def find_boolean_variables(trace_paths: list[str]) -> set[str]:
    """
    Return a set of variables that are boolean (true/false) in all traces.
    A variable is boolean if ALL its values across ALL traces are Python bools.
    """
    first_keys = set()
    with open(trace_paths[0], "r", encoding="utf-8") as f:
        for raw in f:
            clean = clean_line(raw)
            if not clean:
                continue
            try:
                obj = json.loads(clean)
                first_keys = set(obj.keys())
                break
            except Exception:
                continue

    if not first_keys:
        return set()

    boolean_vars = set()
    for var in first_keys:
        is_bool = True
        for path in trace_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for raw in f:
                        clean = clean_line(raw)
                        if not clean:
                            continue
                        try:
                            obj = json.loads(clean)
                            if var in obj and not is_boolean_value(obj[var]):
                                is_bool = False
                                break
                        except Exception:
                            continue
                if not is_bool:
                    break
            except FileNotFoundError:
                is_bool = False
                break
        if is_bool:
            boolean_vars.add(var)

    return boolean_vars


def find_constant_variables(trace_paths: list[str], expand_tuples=True) -> set[str]:
    """
    Return a set of variables that are constant within every trace file in trace_paths.
    Values may differ between traces but must not change inside a single trace.
    If expand_tuples=True, tuple values are expanded to element keys.
    """
    constant_vars = extract_first_line_keys(trace_paths[0], expand_tuples=expand_tuples)
    if not constant_vars:
        return set()

    for path in trace_paths:
        first_values: dict = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for raw in f:
                    clean = clean_line(raw)
                    if not clean:
                        continue
                    try:
                        obj = json.loads(clean)
                        if expand_tuples:
                            obj = expand_tuple_keys(obj)
                    except Exception:
                        continue

                    for var in list(constant_vars):
                        if var not in obj:
                            constant_vars.discard(var)
                            continue
                        if var not in first_values:
                            first_values[var] = obj[var]
                        elif obj[var] != first_values[var]:
                            constant_vars.discard(var)
        except FileNotFoundError:
            constant_vars.clear()
            break

    return constant_vars


# ============================================================================
# I/O Separation Functions
# ============================================================================

def generate_mapping_classes(variables, arity=None, self_inputs_only=False) -> dict[str, list]:
    """
    Generate mapping classes without duplicates.
    Uses combinations (unordered) instead of permutations (ordered).
    Ensures XY1 is only XY1, never YX1.

    If self_inputs_only=True, only generates self-mappings (playerX -> playerX),
    skipping cross-mappings (playerY -> playerX).
    """
    classes = {}
    rs = [arity] if arity is not None else range(1, len(variables) + 1)

    for r in rs:
        for combo in itertools.combinations(variables, r):
            inp_sorted = sorted(combo)
            inp_prefix = "_".join(inp_sorted)

            for out in variables:
                if self_inputs_only:
                    if r == 1 and combo[0] != out:
                        continue
                    elif r > 1:
                        continue

                print("Generating class for input:", inp_sorted, "-> output:", out)
                cls_name = f"{inp_prefix}toNext{out}"
                classes[cls_name] = []

    return classes


def classify_and_store(prev_obj, next_obj, source, pairs):
    """
    Adds mapping instance to the correct class.
    Ensures order of variables in mapping is consistent (sorted).
    """
    for clause in pairs.keys():
        inp_block, out_var = clause.split("toNext")
        inp_vars = inp_block.split("_")

        if len(inp_vars) == 1:
            inp_value = prev_obj[inp_vars[0]]
        else:
            inp_value = tuple(prev_obj[v] for v in inp_vars)

        out_value = next_obj[out_var]

        pairs[clause].append({
            "source": source,
            "input": inp_value,
            "output": out_value
        })


def extract_io_pairs(input_dir, output_dir, self_inputs_only=False, pos_dir=None, neg_dir=None):
    """
    Extract I/O pairs from trace files.

    Args:
        input_dir: Directory containing pos/ and neg/ subdirectories with traces
        output_dir: Output directory for I/O pair files
        self_inputs_only: Only generate self-mappings (playerX -> playerX)
        pos_dir: Explicit path to positive traces directory (default: input_dir/pos)
        neg_dir: Explicit path to negative traces directory (default: input_dir/neg)
    """
    assert os.path.exists(input_dir), "Invalid input directory"
    if input_dir == output_dir:
        output_dir = os.path.join(output_dir, "out")

    pos_path = pos_dir if pos_dir else os.path.join(input_dir, "pos")
    neg_path = neg_dir if neg_dir else os.path.join(input_dir, "neg")
    assert os.path.exists(pos_path) and os.path.exists(neg_path), "Traces not properly bucketed into pos & neg"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    trace_paths = []
    for path in [pos_path, neg_path]:
        for name in os.listdir(path):
            trace_paths.append(os.path.join(path, name))

    print(list(trace_paths))

    # Detect boolean variables (true/false values) - skip for function synthesis
    boolean_vars = find_boolean_variables(trace_paths)
    print("Found following boolean variables (skipping for function synthesis):", boolean_vars)

    # Save boolean variables to file for log2tslf.py to use
    boolean_vars_path = os.path.join(output_dir, "boolean_vars.json")
    with open(boolean_vars_path, "w") as f:
        json.dump(list(boolean_vars), f)
    print(f"Saved boolean variables to: {boolean_vars_path}")

    constants = find_constant_variables(trace_paths)
    print("Found following constant variables:", constants)

    for fpath in trace_paths:
        if not fpath.endswith(".jsonl"):
            continue

        all_vars = extract_first_line_keys(fpath)
        variables = sorted(all_vars - constants - boolean_vars)
        fname = os.path.basename(fpath)

        trace_dir = os.path.join(output_dir, fname.replace(".jsonl", ""))
        os.makedirs(trace_dir, exist_ok=True)

        pairs = generate_mapping_classes(variables, arity=1, self_inputs_only=self_inputs_only)
        print(pairs)

        lines = []
        with open(fpath, "r") as f:
            for raw_line in f:
                clean = clean_line(raw_line)
                if not clean:
                    continue
                try:
                    obj = json.loads(clean)
                    obj = expand_tuple_keys(obj)
                    lines.append(obj)
                except:
                    print("Bad JSON:", fpath, "line:", repr(raw_line))

        for i in range(len(lines) - 1):
            classify_and_store(
                lines[i],
                lines[i + 1],
                f"{fname}:line_{i}_to_{i+1}",
                pairs
            )

        for cls_name, items in pairs.items():
            out_path = os.path.join(trace_dir, f"{cls_name}.jsonl")
            with open(out_path, "w") as f:
                for obj in items:
                    f.write(json.dumps(obj) + "\n")

        print(f"Processed: {fname}")

    print("I/O extraction complete")


# ============================================================================
# Grouping Functions
# ============================================================================

def extract_time_step(source):
    part = source.split("line_")[1]
    i = int(part.split("_to_")[0])
    return i


def output_var_from_filename(filename):
    base = os.path.basename(filename)
    return base.split("toNext")[1].replace(".jsonl", "")


def extract_input_vars_from_filename(filename):
    """
    Extract input variable names from filename.
    E.g., "playerXtoNextplayerY.jsonl" -> ["playerX"]
    E.g., "playerX_playerYtoNextgoalX.jsonl" -> ["playerX", "playerY"]
    """
    base = os.path.basename(filename).replace(".jsonl", "")
    if "toNext" not in base:
        return []
    input_part = base.split("toNext")[0]
    return input_part.split("_")


def load_trace_mapping_dir(trace_dir):
    """
    Buckets by (out_var, time_step).
    Each bucket should have multiple rules.
    Records are annotated with their source mapping class.
    """
    buckets = defaultdict(list)

    for filename in sorted(os.listdir(trace_dir)):
        if not filename.endswith(".jsonl"):
            continue

        out_var = output_var_from_filename(filename)
        input_vars = extract_input_vars_from_filename(filename)
        filepath = os.path.join(trace_dir, filename)

        with open(filepath) as f:
            for line in f:
                rec = json.loads(line)
                rec["_input_vars"] = input_vars
                time_step = extract_time_step(rec["source"])
                slot = (out_var, time_step)
                buckets[slot].append(rec)

    return buckets


def create_baseline_and_alternatives(buckets, self_inputs_only=False):
    """
    Create ONE baseline grouping (first choice per slot) and
    an alternatives map for the bottom-up algorithm.

    Args:
        buckets: dict mapping (out_var, time_step) to list of records
        self_inputs_only: if True, only include self-updates (playerX <- f(playerX))

    Returns:
        baseline: dict mapping "time_X__var" to one record
        alternatives: dict mapping "time_X__var" to list of ALL possible records
    """
    baseline = {}
    alternatives = {}

    for slot, records in buckets.items():
        out_var, time_step = slot
        key = f"time_{time_step}__{out_var}"

        if self_inputs_only:
            filtered_records = []
            for rec in records:
                input_vars = rec.get("_input_vars", [])
                is_self = (len(input_vars) == 1 and input_vars[0] == out_var)
                if is_self:
                    filtered_records.append(rec)
            if filtered_records:
                records = filtered_records

        def priority(rec):
            inp = rec["input"]
            input_vars = rec.get("_input_vars", [])
            is_self = (len(input_vars) == 1 and input_vars[0] == out_var)

            if is_self and inp == rec["output"]:
                return (0, 0)
            if is_self:
                return (1, 0)
            if not isinstance(inp, (list, tuple)):
                return (2, 0)
            return (3, len(inp) if isinstance(inp, (list, tuple)) else 1)

        sorted_records = sorted(records, key=priority)
        baseline[key] = sorted_records[0]

        clean_records = []
        for rec in records:
            clean_rec = {k: v for k, v in rec.items() if k != "_input_vars"}
            clean_records.append(clean_rec)
        alternatives[key] = clean_records

    return baseline, alternatives


def create_groupings(root_dir, self_inputs_only=False):
    """
    Create groupings from I/O pair files.

    Args:
        root_dir: Directory containing per-trace subdirectories with I/O pairs
        self_inputs_only: Only include self-updates (playerX <- f(playerX))
    """
    for trace in sorted(os.listdir(root_dir)):
        trace_dir = os.path.join(root_dir, trace)
        if not os.path.isdir(trace_dir):
            continue

        print(f"\nProcessing trace directory: {trace}")

        buckets = load_trace_mapping_dir(trace_dir)
        print(f"  Found {len(buckets)} slots")

        baseline, alternatives = create_baseline_and_alternatives(buckets, self_inputs_only=self_inputs_only)

        groupings_path = os.path.join(trace_dir, "groupings.jsonl")
        with open(groupings_path, "w") as f:
            f.write(json.dumps(baseline) + "\n")

        alternatives_path = os.path.join(trace_dir, "alternatives.json")
        with open(alternatives_path, "w") as f:
            json.dump(alternatives, f, indent=2)

        print(f"  Wrote baseline grouping to {groupings_path}")
        print(f"  Wrote alternatives map to {alternatives_path}")
        print(f"  Total alternatives: {sum(len(alts) for alts in alternatives.values())}")

    print("\nGrouping creation complete")


# ============================================================================
# Main Entry Point
# ============================================================================

def main(input_dir, output_dir, self_inputs_only=False, pos_dir=None, neg_dir=None):
    """
    Run full I/O extraction and grouping pipeline.

    Args:
        input_dir: Directory containing pos/ and neg/ subdirectories with traces
        output_dir: Output directory for results
        self_inputs_only: Only generate/use self-mappings
        pos_dir: Explicit path to positive traces directory (default: input_dir/pos)
        neg_dir: Explicit path to negative traces directory (default: input_dir/neg)
    """
    print("=" * 50)
    print("Step 1: Extracting I/O pairs")
    print("=" * 50)
    extract_io_pairs(input_dir, output_dir, self_inputs_only, pos_dir=pos_dir, neg_dir=neg_dir)

    print("\n" + "=" * 50)
    print("Step 2: Creating groupings")
    print("=" * 50)
    create_groupings(output_dir, self_inputs_only)

    print("\n" + "=" * 50)
    print("Groupings pipeline complete!")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract I/O pairs and create groupings for SyGuS synthesis")
    parser.add_argument("--input_dir", required=True, help="Directory containing pos/ and neg/ trace subdirectories")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--self-inputs-only", action="store_true",
                        help="Only generate self-mappings (playerX -> playerX), skipping cross-mappings")
    parser.add_argument("--pos", help="Explicit path to positive traces directory (default: input_dir/pos)")
    parser.add_argument("--neg", help="Explicit path to negative traces directory (default: input_dir/neg)")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir,
         self_inputs_only=getattr(args, 'self_inputs_only', False),
         pos_dir=args.pos, neg_dir=args.neg)
