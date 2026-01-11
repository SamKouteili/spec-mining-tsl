import os
import json
import argparse
from collections import defaultdict


def extract_time_step(source):
    # source looks like: "pos_1:line_3_to_4"
    part = source.split("line_")[1]
    i = int(part.split("_to_")[0])
    return i


def output_var_from_filename(filename):
    base = os.path.basename(filename)
    # print("base:", base.split("toNext")[1])
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
    # Input vars are separated by underscores
    return input_part.split("_")


def load_trace_mapping_dir(trace_dir):
    """
    Buckets by (out_var, time_step).
    Each bucket should have multiple rules.
    Records are annotated with their source mapping class.
    """
    buckets = defaultdict(list)

    for filename in sorted(os.listdir(trace_dir)):  # Sort for determinism
        if not filename.endswith(".jsonl"):
            continue

        out_var = output_var_from_filename(filename)
        input_vars = extract_input_vars_from_filename(filename)
        filepath = os.path.join(trace_dir, filename)

        with open(filepath) as f:
            for line in f:
                rec = json.loads(line)
                # Annotate record with input variable names
                rec["_input_vars"] = input_vars
                time_step = extract_time_step(rec["source"])
                slot = (out_var, time_step)
                buckets[slot].append(rec)

    return buckets


def create_baseline_and_alternatives(buckets):
    """
    Create ONE baseline grouping (first choice per slot) and
    an alternatives map for the bottom-up algorithm.

    Returns:
        baseline: dict mapping "time_X__var" to one record
        alternatives: dict mapping "time_X__var" to list of ALL possible records
    """
    baseline = {}
    alternatives = {}

    for slot, records in buckets.items():
        out_var, time_step = slot
        key = f"time_{time_step}__{out_var}"

        # Baseline: prioritize self-updates (playerX <- f(playerX))
        # Sort records by priority:
        # 1. Self-update with identity (playerX <- playerX where input == output)
        # 2. Self-update with transformation (playerX <- f(playerX))
        # 3. Unary cross-update (playerX <- f(playerY))
        # 4. Binary and higher arity
        def priority(rec):
            inp = rec["input"]
            input_vars = rec.get("_input_vars", [])

            # Check if self-update (input vars contain output var)
            is_self = (len(input_vars) == 1 and input_vars[0] == out_var)

            # Priority 0: Self-update with identity
            if is_self and inp == rec["output"]:
                return (0, 0)

            # Priority 1: Self-update with transformation
            if is_self:
                return (1, 0)

            # Priority 2: Unary (scalar input)
            if not isinstance(inp, (list, tuple)):
                return (2, 0)

            # Priority 3: Binary and higher arity
            return (3, len(inp) if isinstance(inp, (list, tuple)) else 1)

        sorted_records = sorted(records, key=priority)
        baseline[key] = sorted_records[0]

        # Alternatives: store all options for this slot (without metadata)
        # Remove the _input_vars annotation from alternatives
        clean_records = []
        for rec in records:
            clean_rec = {k: v for k, v in rec.items() if k != "_input_vars"}
            clean_records.append(clean_rec)
        alternatives[key] = clean_records

    return baseline, alternatives


def main(root_dir):
    for trace in sorted(os.listdir(root_dir)):
        trace_dir = os.path.join(root_dir, trace)
        if not os.path.isdir(trace_dir):
            continue

        print(f"\nProcessing trace directory: {trace}")

        buckets = load_trace_mapping_dir(trace_dir)
        print(f"  Found {len(buckets)} slots")

        # Create baseline grouping and alternatives map
        baseline, alternatives = create_baseline_and_alternatives(buckets)

        # Write baseline grouping (just one line!)
        groupings_path = os.path.join(trace_dir, "groupings.jsonl")
        with open(groupings_path, "w") as f:
            f.write(json.dumps(baseline) + "\n")

        # Write alternatives map
        alternatives_path = os.path.join(trace_dir, "alternatives.json")
        with open(alternatives_path, "w") as f:
            json.dump(alternatives, f, indent=2)

        print(f"  Wrote baseline grouping to {groupings_path}")
        print(f"  Wrote alternatives map to {alternatives_path}")
        print(f"  Total alternatives: {sum(len(alts) for alts in alternatives.values())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True,
                        help="Directory of per-trace mapping subdirectories.")

    args = parser.parse_args()
    main(args.root_dir)
