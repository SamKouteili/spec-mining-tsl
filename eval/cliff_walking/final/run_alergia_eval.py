#!/usr/bin/env python3
"""
Run Alergia baseline evaluation for CliffWalking to match existing results table.

Runs on both fixed and var_config training modes with 50 test configs.
Tests on var_conf (variable cliff height and width configurations).
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cliff_walking.baselines import alergia_baseline


def generate_var_size_test_configs(n: int = 50, seed: int = 42) -> list:
    """
    Generate var_size test configs for CliffWalking.

    Matches the table description:
    - cliff height h ∈ [1,3]
    - board width w ∈ [3,12]
    - board length (height) l ∈ [4,6]

    Config format matches what BC/DT baselines expect:
    - width: board width
    - height: board height (4-6)
    - goalX, goalY: goal position (bottom-right)
    - cliffXMin, cliffXMax: cliff x-range
    - cliffHeight: cliff height (1-indexed, cliff occupies y < cliffHeight)
    """
    random.seed(seed)

    configs = []
    for i in range(n):
        # Variable width (3-12), variable height (4-6)
        width = random.randint(3, 12)
        height = random.randint(4, 6)

        # Variable cliff height (1-3), but must be less than height
        max_cliff_height = min(3, height - 1)
        cliff_height = random.randint(1, max_cliff_height)

        # Goal is always bottom-right (in logged coords where y=0 is bottom)
        goal_x = width - 1
        goal_y = 0

        # Cliff spans from x=1 to x=width-2 (excluding start and goal columns)
        cliff_x_min = 1
        cliff_x_max = width - 2

        configs.append({
            'width': width,
            'height': height,
            'goalX': goal_x,
            'goalY': goal_y,
            'cliffXMin': cliff_x_min,
            'cliffXMax': cliff_x_max,
            'cliffHeight': cliff_height,
            'name': f'var_size_{i}'
        })

    return configs


def run_alergia_on_traces(trace_dir: Path, test_configs: list, eps: float = 2.0) -> dict:
    """Run Alergia on a trace directory."""
    try:
        result = alergia_baseline.train_and_evaluate(trace_dir, test_configs, eps=eps)
        return {
            'successes': result['successes'],
            'total': result['total'],
            'success_rate': result['success_rate'],
            'num_states': result.get('num_states'),
            'train_time': result.get('train_time')
        }
    except Exception as e:
        return {
            'successes': 0,
            'total': len(test_configs),
            'success_rate': 0.0,
            'error': str(e)
        }


def main():
    base_dir = Path(__file__).parent.parent / "full_eval"
    output_file = Path(__file__).parent / "alergia_results.json"

    # Generate test configs
    test_configs = generate_var_size_test_configs(50, seed=42)
    print(f"Generated {len(test_configs)} var_size test configs")

    # N values matching the table columns
    n_values = [4, 8, 12, 16, 20, 50, 100, 200, 500, 1000]

    results = {
        'method': 'alergia',
        'test_condition': 'var_size',
        'num_test_configs': len(test_configs),
        'fixed': {},
        'var_size': {},
        'test_configs': test_configs
    }

    # === fixed → var_size ===
    print(f"\n{'='*60}")
    print("Training mode: fixed → var_size")
    print('='*60)

    for n in n_values:
        trace_dir = base_dir / "fixed" / f"n_{n}"

        if not trace_dir.exists():
            print(f"  n={n:4d}: trace dir not found, skipping")
            continue

        pos_dir = trace_dir / "pos"
        if not pos_dir.exists() or not list(pos_dir.glob("*.jsonl")):
            print(f"  n={n:4d}: no traces found, skipping")
            continue

        result = run_alergia_on_traces(trace_dir, test_configs)
        results['fixed'][n] = result

        rate = result['success_rate']
        states = result.get('num_states', '?')
        print(f"  n={n:4d}: {result['successes']}/{result['total']} ({rate:.1%}) | {states} states")

    # === var_size → var_size (using var_config training data) ===
    print(f"\n{'='*60}")
    print("Training mode: var_size → var_size")
    print('='*60)

    for n in n_values:
        # var_size training uses var_config directory
        trace_dir = base_dir / "var_config" / f"n_{n}"

        if not trace_dir.exists():
            print(f"  n={n:4d}: trace dir not found, skipping")
            continue

        pos_dir = trace_dir / "pos"
        if not pos_dir.exists() or not list(pos_dir.glob("*.jsonl")):
            print(f"  n={n:4d}: no traces found, skipping")
            continue

        result = run_alergia_on_traces(trace_dir, test_configs)
        results['var_size'][n] = result

        rate = result['success_rate']
        states = result.get('num_states', '?')
        print(f"  n={n:4d}: {result['successes']}/{result['total']} ({rate:.1%}) | {states} states")

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY: Alergia Results (successes out of 50)")
    print("="*70)
    print(f"{'n':>6} | {'fixed→var_size':>15} | {'var_size→var_size':>18}")
    print("-"*50)
    for n in n_values:
        fixed_succ = results['fixed'].get(n, {}).get('successes', None)
        var_succ = results['var_size'].get(n, {}).get('successes', None)

        fixed_str = f"{fixed_succ}" if fixed_succ is not None else "--"
        var_str = f"{var_succ}" if var_succ is not None else "--"

        print(f"{n:>6} | {fixed_str:>15} | {var_str:>18}")


if __name__ == "__main__":
    main()
