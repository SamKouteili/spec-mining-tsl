#!/usr/bin/env python3
"""
Run Alergia baseline evaluation for FrozenLake to match existing results.

Runs on both fixed and var_config training modes with 50 test configs.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from frozen_lake.baselines import alergia_baseline


def load_test_configs(json_path):
    """Load test configs from existing results file."""
    with open(json_path) as f:
        data = json.load(f)

    # Test configs should be stored, or we regenerate them
    if 'test_configs_var_config' in data:
        return data['test_configs_var_config']

    # Generate same test configs (seed 42 for reproducibility)
    import random
    random.seed(42)

    configs = []
    for i in range(50):
        while True:
            gx, gy = random.randint(0, 3), random.randint(0, 3)
            if (gx, gy) != (0, 0):
                break

        holes = []
        used = {(0, 0), (gx, gy)}
        for _ in range(3):
            while True:
                hx, hy = random.randint(0, 3), random.randint(0, 3)
                if (hx, hy) not in used:
                    holes.append({'x': hx, 'y': hy})
                    used.add((hx, hy))
                    break

        configs.append({
            'grid_size': 4,
            'goal': {'x': gx, 'y': gy},
            'holes': holes,
            'name': f'test{i}'
        })

    return configs


def run_alergia_on_traces(trace_dir, test_configs, eps=2.0):
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

    # Load test configs
    var_config_json = Path(__file__).parent / "tslf_bc_dt_var_config.json"
    test_configs = load_test_configs(var_config_json)
    print(f"Using {len(test_configs)} test configs")

    n_values = [2, 4, 8, 12, 16, 20, 50, 100, 200, 500, 1000]

    results = {
        'method': 'alergia',
        'test_condition': 'var_config',
        'num_test_configs': len(test_configs),
        'fixed': {},
        'var_config': {}
    }

    for train_mode in ['fixed', 'var_config']:
        print(f"\n{'='*60}")
        print(f"Training mode: {train_mode}")
        print('='*60)

        for n in n_values:
            trace_dir = base_dir / train_mode / f"n_{n}"

            if not trace_dir.exists():
                print(f"  n={n:4d}: trace dir not found, skipping")
                continue

            pos_dir = trace_dir / "pos"
            if not pos_dir.exists() or not list(pos_dir.glob("*.jsonl")):
                print(f"  n={n:4d}: no traces found, skipping")
                continue

            result = run_alergia_on_traces(trace_dir, test_configs)
            results[train_mode][n] = result

            rate = result['success_rate']
            states = result.get('num_states', '?')
            print(f"  n={n:4d}: {result['successes']}/{result['total']} ({rate:.1%}) | {states} states")

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY: Alergia Results")
    print("="*70)
    print(f"{'n':>6} | {'fixed':>12} | {'var_config':>12}")
    print("-"*40)
    for n in n_values:
        fixed_rate = results['fixed'].get(n, {}).get('success_rate', None)
        var_rate = results['var_config'].get(n, {}).get('success_rate', None)

        fixed_str = f"{fixed_rate:.1%}" if fixed_rate is not None else "--"
        var_str = f"{var_rate:.1%}" if var_rate is not None else "--"

        print(f"{n:>6} | {fixed_str:>12} | {var_str:>12}")


if __name__ == "__main__":
    main()
