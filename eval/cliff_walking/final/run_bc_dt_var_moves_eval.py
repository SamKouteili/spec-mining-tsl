#!/usr/bin/env python3
"""
Run BC and DT baseline evaluation for CliffWalking with variant moves.

Uses 50 test configs with var_moves enabled.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from cliff_walking.baselines import bc_baseline, dt_baseline


def generate_var_conf_test_configs(n: int = 50, seed: int = 42) -> list:
    """
    Generate var_conf test configs for CliffWalking with var_moves.

    Variable cliff height (1-3) and variable width (3-12).
    Height fixed at 4 to allow cliff heights 1-3.
    """
    random.seed(seed)

    configs = []
    for i in range(n):
        # Variable width (3-12), fixed height 4
        width = random.randint(3, 12)
        height = 4

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
            'var_moves': True,
            'name': f'var_conf_{i}'
        })

    return configs


def run_baseline_on_traces(trace_dir: Path, test_configs: list, method: str) -> dict:
    """Run BC or DT baseline with var_moves on a trace directory."""
    try:
        if method == "bc":
            result = bc_baseline.train_and_evaluate(trace_dir, test_configs, var_moves=True)
        else:
            result = dt_baseline.train_and_evaluate(trace_dir, test_configs, var_moves=True)
        return {
            'successes': result['successes'],
            'total': result['total'],
            'success_rate': result['success_rate'],
        }
    except Exception as e:
        import traceback
        return {
            'successes': 0,
            'total': len(test_configs),
            'success_rate': 0.0,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def main():
    base_dir = Path(__file__).parent.parent / "full_eval" / "var_moves"
    output_file = Path(__file__).parent / "bc_dt_var_moves_results.json"

    # Generate test configs (same as Alergia eval for consistency)
    test_configs = generate_var_conf_test_configs(50, seed=42)
    print(f"Generated {len(test_configs)} var_conf test configs for var_moves")

    # N values matching the table columns
    n_values = [4, 8, 12, 16, 20, 50, 100, 200, 500, 1000]

    results = {
        'test_condition': 'var_conf',
        'var_moves': True,
        'num_test_configs': len(test_configs),
        'bc': {'fixed': {}, 'var_conf': {}},
        'dt': {'fixed': {}, 'var_conf': {}},
    }

    for method in ['bc', 'dt']:
        print(f"\n{'='*60}")
        print(f"Method: {method.upper()}")
        print('='*60)

        # === fixed → var_conf (with var_moves) ===
        print(f"\nTraining mode: fixed → var_conf (var_moves)")

        for n in n_values:
            trace_dir = base_dir / "fixed" / f"n_{n}"

            if not trace_dir.exists():
                print(f"  n={n:4d}: trace dir not found, skipping")
                continue

            pos_dir = trace_dir / "pos"
            if not pos_dir.exists() or not list(pos_dir.glob("*.jsonl")):
                print(f"  n={n:4d}: no traces found, skipping")
                continue

            result = run_baseline_on_traces(trace_dir, test_configs, method)
            results[method]['fixed'][n] = result

            rate = result['success_rate']
            print(f"  n={n:4d}: {result['successes']}/{result['total']} ({rate:.1%})")

        # === var_conf → var_conf (with var_moves) ===
        print(f"\nTraining mode: var_conf → var_conf (var_moves)")

        for n in n_values:
            # var_conf training with var_moves
            trace_dir = base_dir / "var_config_moves" / f"n_{n}"

            if not trace_dir.exists():
                # Try alternative path
                trace_dir = base_dir / f"n_{n}"

            if not trace_dir.exists():
                print(f"  n={n:4d}: trace dir not found, skipping")
                continue

            pos_dir = trace_dir / "pos"
            if not pos_dir.exists() or not list(pos_dir.glob("*.jsonl")):
                print(f"  n={n:4d}: no traces found, skipping")
                continue

            result = run_baseline_on_traces(trace_dir, test_configs, method)
            results[method]['var_conf'][n] = result

            rate = result['success_rate']
            print(f"  n={n:4d}: {result['successes']}/{result['total']} ({rate:.1%})")

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY: BC/DT Results - var_moves (successes out of 50)")
    print("="*70)
    print(f"{'n':>6} | {'BC fixed':>10} | {'BC var_conf':>12} | {'DT fixed':>10} | {'DT var_conf':>12}")
    print("-"*70)
    for n in n_values:
        bc_fixed = results['bc']['fixed'].get(n, {}).get('successes', None)
        bc_var = results['bc']['var_conf'].get(n, {}).get('successes', None)
        dt_fixed = results['dt']['fixed'].get(n, {}).get('successes', None)
        dt_var = results['dt']['var_conf'].get(n, {}).get('successes', None)

        bc_fixed_str = f"{bc_fixed}" if bc_fixed is not None else "--"
        bc_var_str = f"{bc_var}" if bc_var is not None else "--"
        dt_fixed_str = f"{dt_fixed}" if dt_fixed is not None else "--"
        dt_var_str = f"{dt_var}" if dt_var is not None else "--"

        print(f"{n:>6} | {bc_fixed_str:>10} | {bc_var_str:>12} | {dt_fixed_str:>10} | {dt_var_str:>12}")


if __name__ == "__main__":
    main()
