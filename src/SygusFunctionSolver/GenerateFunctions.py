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
  ((I Int))
  (
    (I Int (
            x
            (+ I I)
            (- I I)
            (* I I)
            (div I I)
            (mod I I)
    ))
  )
)

{constraints}

(check-synth)
"""

BINARY_ARITY_TEMPLATE = """(set-logic NIA)

(synth-fun f ((x Int) (y Int)) Int
  ((I Int))
  (
    (I Int (
            x
            y
            (+ I I)
            (- I I)
            (* I I)
            (div I I)
            (mod I I)
    ))
  )
)

{constraints}

(check-synth)
"""

TIMEOUT = 0.05  # 50ms per solver call


# ========================================================================
#   HELPERS
# ========================================================================

def get_arity(rec):
    return 2 if isinstance(rec["input"], list) else 1


def generate_constraints(block):
    """Generate SyGuS constraints from a block of input-output pairs."""
    out = []
    for key, rec in block.items():
        inp = rec["input"]
        outv = rec["output"]
        if isinstance(inp, list):
            out.append(f"(constraint (= (f {inp[0]} {inp[1]}) {outv}))")
        else:
            out.append(f"(constraint (= (f {inp}) {outv}))")
    return "\n".join(out)


def solve_block(block, timeout=TIMEOUT):
    """Solve a single block of constraints."""
    if not block:
        return None

    # Determine arity
    first_rec = next(iter(block.values()))
    arity = get_arity(first_rec)

    # Ensure consistent arity across block
    for rec in block.values():
        if get_arity(rec) != arity:
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

        # Successful synthesis?
        if "(define-fun" in out and "error" not in out.lower():
            return out

        return None

    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None
    finally:
        os.unlink(path)


def solve_partition(partition):
    """All blocks must be solvable or partition fails."""
    fns = []
    for block in partition:
        fn = solve_block(block)
        if fn is None:
            return None
        fns.append(fn)
    return fns


# ========================================================================
#   PROCESS ONE TRACE
# ========================================================================

def process_single_trace(trace_dir):
    """
    Load final_partitions.jsonl and try to solve each partition until success.
    Write result to output_funcs.jsonl.
    """
    in_path = os.path.join(trace_dir, "final_partitions.jsonl")
    out_path = os.path.join(trace_dir, "output_funcs.jsonl")

    if not os.path.exists(in_path):
        print(f"  [SKIP] No final_partitions.jsonl in {trace_dir}")
        return

    print(f"  Solving partitions in {trace_dir} ...")

    with open(in_path) as f:
        for line_num, line in enumerate(f):
            if line_num % 100 == 0:
                print(f"    Checking partition {line_num}...")

            partition = json.loads(line)
            fns = solve_partition(partition)

            if fns is not None:
                result = {
                    "line": line_num,
                    "partition": partition,
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
