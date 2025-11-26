import os
import json
from itertools import product
from collections import defaultdict


def extract_time_step(source):
    # source looks like: "pos_1:line_3→4"
    part = source.split("line_")[1]
    i = int(part.split("→")[0])
    return i


def output_var_from_filename(filename):
    base = os.path.basename(filename)
    return base.split("_to_")[1].replace(".jsonl", "")


def load_trace_mapping_dir(trace_dir):
    """
    Buckets by (out_var, time_step).
    Each bucket should have 3 rules (unary, binary, xy).
    """
    buckets = defaultdict(list)

    for filename in os.listdir(trace_dir):
        if not filename.endswith(".jsonl"):
            continue

        out_var = output_var_from_filename(filename)
        filepath = os.path.join(trace_dir, filename)

        with open(filepath) as f:
            for line in f:
                rec = json.loads(line)
                time_step = extract_time_step(rec["source"])
                slot = (out_var, time_step)
                buckets[slot].append(rec)

    return buckets


def enumerate_exactly_one_per_slot(buckets):
    """
    Choose exactly ONE rule per slot.
    Slots = (out_var, time_step)
    """
    slots = list(buckets.keys())
    choices = [buckets[s] for s in slots]  # each should have 3

    for combo in product(*choices):
        subset = {}
        for slot, rec in zip(slots, combo):
            out_var, time_step = slot
            key = f"time_{time_step}__{out_var}"
            subset[key] = rec
        yield subset


if __name__ == "__main__":

    ROOT = "output_pairs"

    for trace in sorted(os.listdir(ROOT)):
        trace_dir = os.path.join(ROOT, trace)
        if not os.path.isdir(trace_dir):
            continue

        print(f"\nProcessing trace directory: {trace}")

        buckets = load_trace_mapping_dir(trace_dir)
        
        # Expect 10 slots (5 time steps × X2/Y2)
        print("Found slots:", buckets.keys())

        out_path = os.path.join(trace_dir, "groupings.jsonl")

        count = 0
        with open(out_path, "w") as out_file:
            for subset in enumerate_exactly_one_per_slot(buckets):
                out_file.write(json.dumps(subset) + "\n")
                count += 1

        print(f"  Wrote {count} subsets to {out_path}")
        print("  Expected:", 3 ** len(buckets))
