import json
import subprocess
import tempfile
import os
import sys

SINGLE_ARITY_TEMPLATE = """(set-logic NIA)

(synth-fun f ((x Int)) Int
  ((I Int))
  (
    (I Int (
            x
            (Constant Int)
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
            (Constant Int)
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

TIMEOUT = 0.05  # 50ms timeout


def get_arity(rec):
    return 2 if isinstance(rec["input"], list) else 1


def generate_constraints(block):
    """Generate SyGuS constraints from a block of input-output pairs."""
    constraints = []
    for key, rec in block.items():
        inp = rec["input"]
        out = rec["output"]
        if isinstance(inp, list):
            constraints.append(f"(constraint (= (f {inp[0]} {inp[1]}) {out}))")
        else:
            constraints.append(f"(constraint (= (f {inp}) {out}))")
    return "\n".join(constraints)


def solve_block(block, timeout=TIMEOUT):
    """
    Try to synthesize a function for this block.
    Returns the function if found within timeout, None otherwise.
    """
    if not block:
        return None
    
    # Determine arity from first record
    first_rec = next(iter(block.values()))
    arity = get_arity(first_rec)
    
    # Check all records have same arity
    for rec in block.values():
        if get_arity(rec) != arity:
            return None  # Mixed arity in block - shouldn't happen
    
    constraints = generate_constraints(block)
    
    if arity == 1:
        sygus_code = SINGLE_ARITY_TEMPLATE.format(constraints=constraints)
    else:
        sygus_code = BINARY_ARITY_TEMPLATE.format(constraints=constraints)
    
    # Write to temp file and run cvc5
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sy', delete=False) as f:
        f.write(sygus_code)
        temp_path = f.name
    
    try:
        result = subprocess.run(
            ['cvc5', '--lang=sygus2', temp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        output = result.stdout.strip()
        
        # Check if synthesis succeeded
        if output and not output.startswith('(error') and 'unsat' not in output.lower():
            # Parse out the function definition
            if '(define-fun' in output:
                return output
        
        return None
        
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        return None
    finally:
        os.unlink(temp_path)


def solve_partition(partition):
    """
    Try to solve all blocks in a partition.
    Returns list of functions if ALL blocks succeed, None otherwise.
    """
    functions = []
    
    for block in partition:
        func = solve_block(block)
        if func is None:
            return None  # One block failed, entire partition fails
        functions.append(func)
    
    return functions


def process_partitions_file(input_path, output_path):
    """
    Process partitions until FIRST success, then exit.
    """
    with open(input_path) as f:
        for line_num, line in enumerate(f):
            partition = json.loads(line)
            
            if line_num % 100 == 0:
                print(f"Processing line {line_num}...")
            
            functions = solve_partition(partition)
            
            if functions is not None:
                result = {
                    "line": line_num,
                    "partition": partition,
                    "functions": functions
                }
                
                with open(output_path, 'w') as out:
                    json.dump(result, out, indent=2)
                
                print(f"\nSUCCESS at line {line_num} ({len(partition)} blocks)")
                for i, func in enumerate(functions):
                    print(f"\nBlock {i}: {func}")
                
                print(f"\nResult written to {output_path}")
                sys.exit(0)  # EXIT IMMEDIATELY
    
    print("No solution found")
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python solve_partitions.py <final_partitions.jsonl> [output.json]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "solved_partitions.json"
    
    process_partitions_file(input_path, output_path)