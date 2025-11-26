import json
import os

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
        # Try putting "first" in an existing block
        for i in range(len(rest)):
            new_block = rest[i] + [first]
            yield rest[:i] + [new_block] + rest[i+1:]
        # Try creating a new block with just "first"
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

    # Debug: verify sorting
    sizes = [len(p) for p in all_results[:10]]
    print(f"First 10 partition sizes: {sizes}")
    return all_results

if __name__ == "__main__":
    ROOT = "output_pairs"

    for trace in sorted(os.listdir(ROOT)):
        tdir = os.path.join(ROOT, trace)
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