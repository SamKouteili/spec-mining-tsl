import json
import subprocess
import tempfile
import os
import sys
import argparse

# ========================================================================
#   UPDATED: NO CONSTANTS ALLOWED IN GRAMMAR
# ========================================================================

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
#TODO: WHY - 0 1 and + C C ???
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

TIMEOUT = 0.02  # 0.1 seconds per solver call


# ========================================================================
#   HELPERS
# ========================================================================

def get_arity(rec):
    return 2 if isinstance(rec["input"], list) else 1


def format_block(block):
    """Format block as input->output pairs for debugging."""
    pairs = []
    for key, rec in block.items():
        pairs.append(f"{rec['input']}->{rec['output']}")
    return "{" + ", ".join(pairs) + "}"


def format_sygus_int(n):
    """Format an integer for SyGuS - negative numbers need (- 0 n) syntax."""
    if n < 0:
        return f"(- 0 {abs(n)})"
    return str(n)


def generate_constraints(block):
    """Generate SyGuS constraints from a block of input-output pairs."""
    out = []
    for key, rec in block.items():
        inp = rec["input"]
        outv = rec["output"]
        if isinstance(inp, list):
            inp0 = format_sygus_int(inp[0])
            inp1 = format_sygus_int(inp[1])
            outv_str = format_sygus_int(outv)
            out.append(f"(constraint (= (f {inp0} {inp1}) {outv_str}))")
        else:
            inp_str = format_sygus_int(inp)
            outv_str = format_sygus_int(outv)
            out.append(f"(constraint (= (f {inp_str}) {outv_str}))")
    return "\n".join(out)


def extract_define_fun(out):
    """
    Extracts the full (define-fun ...) s-expression
    even if it's multi-line and contains nested lets.
    """
    lines = out.split("\n")
    
    start = None
    paren_count = 0
    buf = []

    for i, line in enumerate(lines):
        if "(define-fun" in line:
            start = i
            # count parentheses on this line
            paren_count = line.count("(") - line.count(")")
            buf.append(line)
            break

    if start is None:
        return None  # no define-fun found

    # Read forward until parentheses balance to zero
    for line in lines[start+1:]:
        paren_count += line.count("(")
        paren_count -= line.count(")")
        buf.append(line)
        if paren_count == 0:
            break

    return "\n".join(buf)


def solve_block(block, block_idx, timeout=TIMEOUT):
    """Solve a single block of constraints."""
    if not block:
        print(f"      Block {block_idx}: EMPTY")
        return None

    # Determine arity
    first_rec = next(iter(block.values()))
    arity = get_arity(first_rec)

    # Ensure consistent arity across block
    for rec in block.values():
        if get_arity(rec) != arity:
            print(f"      Block {block_idx}: INCONSISTENT ARITY")
            return None

    constraints = generate_constraints(block)
    sygus = (
        SINGLE_ARITY_TEMPLATE if arity == 1 else BINARY_ARITY_TEMPLATE
    ).format(constraints=constraints)

    # Temp file I/O
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
        err = result.stderr.strip()

        # Successful synthesis?
        if "(define-fun" in out and "error" not in out.lower():
            # Extract just the function definition
            lines = [l for l in out.split('\n') if 'define-fun' in l]
            func = lines[0] if lines else out
            print(f"      Block {block_idx}: SUCCESS -> {func}")

            func = extract_define_fun(out)
            return func

        print(f"      Block {block_idx}: NO SOLUTION (stderr: {err[:80]})")
        return None

    except subprocess.TimeoutExpired:
        print(f"      Block {block_idx}: TIMEOUT")
        return None
    except FileNotFoundError:
        print(f"      Block {block_idx}: CVC5 NOT FOUND")
        return None
    except Exception as e:
        print(f"      Block {block_idx}: EXCEPTION {type(e).__name__}: {e}")
        return None
    finally:
        os.unlink(path)


def solve_partition(partition, line_num):
    """All blocks must be solvable or partition fails."""
    print(f"\n  Line {line_num}: {len(partition)} blocks")
    for block_idx, block in enumerate(partition):
        print(f"    Block {block_idx}: {format_block(block)}")
    
    fns = []
    for block_idx, block in enumerate(partition):
        fn = solve_block(block, block_idx)
        if fn is None:
            print(f"    -> PARTITION FAILED at block {block_idx}")
            return None
        fns.append(fn)
    
    print(f"    -> ALL BLOCKS SOLVED!")
    return fns



# ========================================================================
#   PARTITION SUBSETS
# ========================================================================

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

# def compute_full_block_partitions(full_set):
#     keys = list(full_set.keys())
#     parts = list(partitions(keys))
#     results = []

#     for p in parts:
#         block_dicts = []
#         for block in p:
#             block_dicts.append({k: full_set[k] for k in block})
#         results.append(block_dicts)

#     # sort by number of blocks
#     results.sort(key=lambda p: len(p))
#     return results

# ========================================================================
#   PROCESS ONE TRACE
# ========================================================================

def process_single_trace(trace_dir):
    """
    Load final_partitions.jsonl and try to solve each partition until success.
    Write result to output_funcs.jsonl.
    """
    in_path = os.path.join(trace_dir, "groupings.jsonl")
    out_path = os.path.join(trace_dir, "output_funcs.jsonl")

    if not os.path.exists(in_path):
        print(f"  [SKIP] No groupings.jsonl in {trace_dir}")
        return

    print(f"  Solving partitions in {trace_dir} ...")


    with open(in_path) as f:
        for line_num, line in enumerate(f):
            # NOW THIS IS A LINE IN groupings.jsonl
            full_set = json.loads(line)

            keys = list(full_set.keys())
            # parts = list(partitions(keys))

            for p in partitions(keys):
                # block_dicts here is the partition of full_set into blocks
                block_dicts = []
                for block in p:
                    block_dicts.append({k: full_set[k] for k in block})

                fns = solve_partition(block_dicts, line_num)

                if fns is not None:
                    result = {
                        "line": line_num,
                        "partition": block_dicts,
                        "functions": fns
                    }
                    with open(out_path, "w") as out:
                        json.dump(result, out, indent=2)
                    print(f"\n  SUCCESS for trace {trace_dir} at partition {line_num}")
                    print(f"  Output written to {out_path}\n")
                    return

    print(f"  No solvable partition found for {trace_dir}\n")


# ========================================================================
#   PROCESS ENTIRE DIRECTORY TREE
# ========================================================================

def process_all_traces(root_dir):
    """
    For each subdirectory in root_dir:
       trace_dir/final_partitions.jsonl → trace_dir/output_funcs.jsonl
    """
    print(f"\n=== Processing all traces in {root_dir} ===\n")
    for name in sorted(os.listdir(root_dir)):
        trace_dir = os.path.join(root_dir, name)
        if os.path.isdir(trace_dir):
            process_single_trace(trace_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", required=True,
        help="Directory containing subdirectories with final_partitions.jsonl files"
    )
    args = parser.parse_args()
    process_all_traces(args.root_dir)