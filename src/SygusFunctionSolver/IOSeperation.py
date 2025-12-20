import os
import json
import argparse
import itertools
import shutil


def clean_line(raw):
    return (
        raw.replace("\ufeff", "")
           .replace("\xa0", " ")
           .strip()
    )


def extract_first_line_keys(file_path) -> set[str]:
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
                return set(obj.keys())
            return set()
    return set()


def find_constant_variables(trace_paths: list[str]) -> set[str]:
    """
    Return a set of variables that are constant within every trace file in trace_paths.
    Values may differ between traces but must not change inside a single trace.
    """
    # Start from the keys of the first trace file
    constant_vars = extract_first_line_keys(trace_paths[0])
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
            # If a trace doesn't exist, it cannot contribute constants
            constant_vars.clear()
            break

    return constant_vars
        

def generate_mapping_classes(vars_list: list[str], arity = None) -> dict[str, list]:
    """
    Generate mapping classes without duplicates.
    Uses combinations (unordered) instead of permutations (ordered).
    Ensures XY1 is only XY1, never YX1.
    """

    classes = {}

    rs = [arity] if arity is not None else range(1, len(vars_list) + 1)

    # All possible input subsets (size >= 1)
    for r in rs:
        for combo in itertools.combinations(vars_list, r):
            # Sort for determinism (avoid XY vs YX)
            inp_sorted = sorted(combo)
            inp_prefix = "_".join(inp_sorted)



            # Map to each possible output var
            for out in vars_list:
                # NOTE: TESTING FOR NOW TO BE REMOVED
                # if inp_sorted == ['playerX'] and out == 'playerY' or inp_sorted == ['playerY'] and out == 'playerX':
                #     # Skip trivial identity mappings between X and Y
                #     continue
                print("Generating class for input:", inp_sorted, "-> output:", out)
                cls_name = f"{inp_prefix}toNext{out}"
                classes[cls_name] = []

    return classes


# NOTE: vars_list unused
def classify_and_store(prev_obj, next_obj, source, pairs, vars_list):
    """
    Adds mapping instance to the correct class.
    Ensures order of variables in mapping is consistent (sorted).
    """

    for clause in pairs.keys():
        # print(clause)
        # example: "XY1_to_Z2"
        inp_block, out_var = clause.split("toNext")

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

# TODO: check if conflicting trace file names create problems
def main(input_dir, output_dir):

    assert os.path.exists(input_dir), "Invalid input directory"
    if input_dir == output_dir :
        output_dir = os.path.join(output_dir, "out")
        
    pos_path = os.path.join(input_dir, "pos")
    neg_path = os.path.join(input_dir, "neg")
    assert os.path.exists(pos_path) and os.path.exists(neg_path), "Traces not properly bucketed into pos & neg"
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    trace_paths = []
    for path in [pos_path, neg_path]:
        for name in os.listdir(path):
            trace_paths.append(os.path.join(path, name))

    print(list(trace_paths))

    constants = find_constant_variables(trace_paths)
    print("Found following constant variables:", constants)

    for fpath in trace_paths:
        if not fpath.endswith(".jsonl"):
            continue

        # file_path = os.path.join(d, fname)

        all_vars = extract_first_line_keys(fpath)
        # print(all_vars)
        variables = all_vars - constants
        # print(variables)

        fname = os.path.basename(fpath)

        # Per-trace output folder
        trace_dir = os.path.join(output_dir, fname.replace(".jsonl", ""))
        os.makedirs(trace_dir, exist_ok=True)

        pairs = generate_mapping_classes(variables, arity=1)
        print(pairs)

        # load trace
        lines = []
        with open(fpath, "r") as f:
            for raw_line in f:
                clean = clean_line(raw_line)
                if not clean:
                    continue
                try:
                    obj = json.loads(clean)
                    lines.append(obj)
                except:
                    print("Bad JSON:", fpath, "line:", repr(raw_line))

        # print("lines", lines)

        # process transitions
        for i in range(len(lines) - 1):
            # print(lines[i], lines[i+1], pairs, vars_list)
            classify_and_store(
                lines[i],
                lines[i + 1],
                f"{fname}:line_{i}_to_{i+1}",
                pairs,
                variables,
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
