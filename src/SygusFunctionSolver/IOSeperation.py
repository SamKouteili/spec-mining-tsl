import os
import json
import argparse
import itertools


def clean_line(raw):
    return (
        raw.replace("\ufeff", "")
           .replace("\xa0", " ")
           .strip()
    )


def extract_first_line_keys(file_path):
    """
    Parse the first non-empty JSON line in file_path and return its keys as a list.
    Returns an empty list if no valid JSON object line is found.
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
                return list(obj.keys())
            return []
    return []


def generate_mapping_classes(vars_list):
    """
    Generate mapping classes without duplicates.
    Uses combinations (unordered) instead of permutations (ordered).
    Ensures XY1 is only XY1, never YX1.
    """

    classes = {}

    # All possible input subsets (size >= 1)
    for r in range(1, len(vars_list) + 1):
        for combo in itertools.combinations(vars_list, r):
            # Sort for determinism (avoid XY vs YX)
            inp_sorted = sorted(combo)
            inp_prefix = "_".join(inp_sorted)

            # Map to each possible output var
            for out in vars_list:
                cls_name = f"{inp_prefix}2X{out}"
                classes[cls_name] = []

    return classes


def classify_and_store(prev_obj, next_obj, source, pairs, vars_list):
    """
    Adds mapping instance to the correct class.
    Ensures order of variables in mapping is consistent (sorted).
    """

    for clause in pairs.keys():
        # example: "XY1_to_Z2"
        inp_block, out_var = clause.split("2X")

        # print(f"clause: {clause}, input_block: {inp_block}, out_block: {out_block}")

        # input var string inside "XY1"
        inp_vars = inp_block.split("_")  # remove trailing 1

        # print("inp_vars->outvar", inp_vars, out_var)

        # extract input tuple or scalar
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


def main(input_dir, output_dir):

    assert os.path.exists(input_dir), "Invalid input directory"
    pos_path = os.path.join(input_dir, "pos")
    neg_path = os.path.join(input_dir, "neg")
    assert os.path.exists(pos_path) and os.path.exists(neg_path), "Traces not properly bucketed into pos & neg"
    # os.removedirs(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for d, traces in [(pos_path, os.listdir(pos_path)), (neg_path, os.listdir(neg_path))]:
        for fname in traces: 
            if not fname.endswith(".jsonl"):
                continue

            file_path = os.path.join(d, fname)

            vars_list = extract_first_line_keys(file_path)
            # print(vars_list)

            # Per-trace output folder
            trace_dir = os.path.join(output_dir, fname.replace(".jsonl", ""))
            os.makedirs(trace_dir, exist_ok=True)

            pairs = generate_mapping_classes(vars_list)
            # print(pairs)

            # load trace
            lines = []
            with open(file_path, "r") as f:
                for raw_line in f:
                    clean = clean_line(raw_line)
                    if not clean:
                        continue
                    try:
                        obj = json.loads(clean)
                        lines.append(obj)
                    except:
                        print("Bad JSON:", fname, "line:", repr(raw_line))

            # print("lines", lines)

            # process transitions
            for i in range(len(lines) - 1):
                # print(lines[i], lines[i+1], pairs, vars_list)
                classify_and_store(
                    lines[i],
                    lines[i + 1],
                    f"{fname}:line_{i}_to_{i+1}",
                    pairs,
                    vars_list,
                )

            # write files
            for cls_name, items in pairs.items():
                out_path = os.path.join(trace_dir, f"{cls_name}.jsonl")
                with open(out_path, "w") as f:
                    for obj in items:
                        f.write(json.dumps(obj) + "\n")

            print(f"Processed: {fname}")

    print("Done")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    # parser.add_argument("--vars", nargs="+", required=True)
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
