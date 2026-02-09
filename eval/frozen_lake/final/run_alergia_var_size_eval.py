#!/usr/bin/env python3
"""
Run Alergia baseline evaluation for FrozenLake on var_size traces.

Tests on var_size test configs (variable grid sizes 3-5).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from frozen_lake.baselines import alergia_baseline


def load_var_size_test_configs(json_path):
    """Load var_size test configs from existing results file."""
    with open(json_path) as f:
        data = json.load(f)

    if 'test_configs_var_size' in data:
        return data['test_configs_var_size']

    # Fallback: generate var_size test configs
    import random
    random.seed(42)

    configs = []
    for i in range(10):
        grid_size = random.choice([3, 4, 5])

        while True:
            gx, gy = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
            if (gx, gy) != (0, 0):
                break

        holes = []
        used = {(0, 0), (gx, gy)}
        for _ in range(3):
            while True:
                hx, hy = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
                if (hx, hy) not in used:
                    holes.append({'x': hx, 'y': hy})
                    used.add((hx, hy))
                    break

        configs.append({
            'grid_size': grid_size,
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
    output_file = Path(__file__).parent / "alergia_var_size_results.json"

    # Load var_size test configs from existing results
    var_size_json = Path(__file__).parent / "tslf_bc_dt_var_to_var_size.json"
    test_configs = load_var_size_test_configs(var_size_json)
    print(f"Using {len(test_configs)} var_size test configs")

    # Print grid size distribution
    sizes = [c['grid_size'] for c in test_configs]
    print(f"Grid sizes: {sizes}")

    # Available n values for var_size training
    # Check what actually exists
    var_size_dir = base_dir / "var_size"
    available_n = []
    for subdir in var_size_dir.iterdir():
        if subdir.is_dir() and subdir.name.startswith('n_'):
            n = int(subdir.name[2:])
            pos_dir = subdir / "pos"
            if pos_dir.exists() and list(pos_dir.glob("*.jsonl")):
                available_n.append(n)

    available_n = sorted(available_n)
    print(f"Available n values: {available_n}")

    results = {
        'method': 'alergia',
        'test_condition': 'var_size',
        'num_test_configs': len(test_configs),
        'var_size': {}
    }

    print(f"\n{'='*60}")
    print(f"Training mode: var_size")
    print('='*60)

    for n in available_n:
        trace_dir = var_size_dir / f"n_{n}"

        result = run_alergia_on_traces(trace_dir, test_configs)
        results['var_size'][n] = result

        rate = result['success_rate']
        states = result.get('num_states', '?')
        err = result.get('error', '')
        if err:
            print(f"  n={n:4d}: ERROR - {err[:50]}")
        else:
            print(f"  n={n:4d}: {result['successes']}/{result['total']} ({rate:.1%}) | {states} states")

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY: Alergia var_size Results")
    print("="*70)
    print(f"{'n':>6} | {'var_size':>12}")
    print("-"*30)
    for n in available_n:
        rate = results['var_size'].get(n, {}).get('success_rate', None)
        rate_str = f"{rate:.1%}" if rate is not None else "--"
        print(f"{n:>6} | {rate_str:>12}")


if __name__ == "__main__":
    main()
