import json
import os
import argparse


def get_arity(rec):
    return 2 if isinstance(rec["input"], list) else 1


def partitions(set_list):
    """
    Generate all partitions of a list into disjoint non-empty subsets.
    Bell number growth but manageable for sets ≤ 5.
    """
    if not set_list:
        yield []
        return

    first = set_list[0]
    for rest in partitions(set_list[1:]):
        # Add "first" to an existing block
        for i in range(len(rest)):
            new_block = rest[i] + [first]
            yield rest[:i] + [new_block] + rest[i+1:]
        # Create a new block containing only "first"
        yield [[first]] + rest


def compute_full_block_partitions(full_set):
    unary = []
    binary = []

    for key, rec in full_set.items():
        if get_arity(rec) == 1:
            unary.append(key)
        else:
            binary.append(key)

    unary_parts = list(partitions(unary)) if unary else [[]]
    binary_parts = list(partitions(binary)) if binary else [[]]

    all_results = []

    for up in unary_parts:
        for bp in binary_parts:
            blocks = []
            for block in up:
                blocks.append({k: full_set[k] for k in block})
            for block in bp:
                blocks.append({k: full_set[k] for k in block})
            all_results.append(blocks)

    all_results.sort(key=lambda p: len(p))
    return all_results


def main(root_dir):
    for trace in sorted(os.listdir(root_dir)):
        tdir = os.path.join(root_dir, trace)
        if not os.path.isdir(tdir):
            continue

        in_path = os.path.join(tdir, "groupings.jsonl")
        if not os.path.exists(in_path):
            print("Skipping", trace)
            continue

        out_path = os.path.join(tdir, "final_partitions.jsonl")
        count = 0

        with open(in_path) as fin, open(out_path, "w") as fout:
            for line in fin:
                full_set = json.loads(line)
                results = compute_full_block_partitions(full_set)
                for r in results:
                    fout.write(json.dumps(r) + "\n")
                    count += 1

        print(f"Trace {trace}: Wrote {count} partitions to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True,
                        help="Directory where trace subfolders (with groupings.jsonl) are located.")
    args = parser.parse_args()
    main(args.root_dir)
