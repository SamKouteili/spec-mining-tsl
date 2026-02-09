#!/usr/bin/env python3
"""
Run complete Alergia evaluation for FrozenLake matching the paper table format.

Evaluates all 4 permutations:
- fixed → var_conf
- var_conf → var_conf
- fixed → var_size
- var_size → var_size

Tests on 50 configurations for each test condition.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from frozen_lake.baselines import alergia_baseline


def generate_var_conf_test_configs(n=50, seed=42):
    """Generate 50 var_conf test configs (4x4 board, random hole/goal positions)."""
    random.seed(seed)

    configs = []
    for i in range(n):
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
            'name': f'var_conf_{i}'
        })

    return configs


def generate_var_size_test_configs(n=50, seed=42):
    """Generate 50 var_size test configs (3-5 board, random hole/goal positions)."""
    random.seed(seed)

    configs = []
    for i in range(n):
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
            'name': f'var_size_{i}'
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
    output_file = Path(__file__).parent / "alergia_full_results.json"

    # Generate test configs
    var_conf_test_configs = generate_var_conf_test_configs(50, seed=42)
    var_size_test_configs = generate_var_size_test_configs(50, seed=43)  # Different seed

    print(f"Generated {len(var_conf_test_configs)} var_conf test configs")
    print(f"Generated {len(var_size_test_configs)} var_size test configs")

    # N values to evaluate (matching table columns)
    n_values = [4, 8, 12, 16, 20, 50, 100, 500, 1000]

    results = {
        'method': 'alergia',
        'num_test_configs': 50,
        'fixed_to_var_conf': {},
        'var_conf_to_var_conf': {},
        'fixed_to_var_size': {},
        'var_size_to_var_size': {},
        'test_configs_var_conf': var_conf_test_configs,
        'test_configs_var_size': var_size_test_configs
    }

    # === fixed → var_conf ===
    print(f"\n{'='*60}")
    print("fixed → var_conf")
    print('='*60)
    for n in n_values:
        trace_dir = base_dir / "fixed" / f"n_{n}"
        if not trace_dir.exists():
            print(f"  n={n:4d}: trace dir not found")
            continue
        pos_dir = trace_dir / "pos"
        if not pos_dir.exists() or not list(pos_dir.glob("*.jsonl")):
            print(f"  n={n:4d}: no traces found")
            continue

        result = run_alergia_on_traces(trace_dir, var_conf_test_configs)
        results['fixed_to_var_conf'][n] = result
        print(f"  n={n:4d}: {result['successes']}/{result['total']} ({result['success_rate']:.1%})")

    # === var_conf → var_conf ===
    print(f"\n{'='*60}")
    print("var_conf → var_conf")
    print('='*60)
    for n in n_values:
        trace_dir = base_dir / "var_config" / f"n_{n}"
        if not trace_dir.exists():
            print(f"  n={n:4d}: trace dir not found")
            continue
        pos_dir = trace_dir / "pos"
        if not pos_dir.exists() or not list(pos_dir.glob("*.jsonl")):
            print(f"  n={n:4d}: no traces found")
            continue

        result = run_alergia_on_traces(trace_dir, var_conf_test_configs)
        results['var_conf_to_var_conf'][n] = result
        print(f"  n={n:4d}: {result['successes']}/{result['total']} ({result['success_rate']:.1%})")

    # === fixed → var_size ===
    print(f"\n{'='*60}")
    print("fixed → var_size")
    print('='*60)
    for n in n_values:
        trace_dir = base_dir / "fixed" / f"n_{n}"
        if not trace_dir.exists():
            print(f"  n={n:4d}: trace dir not found")
            continue
        pos_dir = trace_dir / "pos"
        if not pos_dir.exists() or not list(pos_dir.glob("*.jsonl")):
            print(f"  n={n:4d}: no traces found")
            continue

        result = run_alergia_on_traces(trace_dir, var_size_test_configs)
        results['fixed_to_var_size'][n] = result
        print(f"  n={n:4d}: {result['successes']}/{result['total']} ({result['success_rate']:.1%})")

    # === var_size → var_size ===
    print(f"\n{'='*60}")
    print("var_size → var_size")
    print('='*60)
    # Check available var_size n values
    var_size_dir = base_dir / "var_size"
    available_var_size_n = []
    for subdir in var_size_dir.iterdir():
        if subdir.is_dir() and subdir.name.startswith('n_'):
            n = int(subdir.name[2:])
            pos_dir = subdir / "pos"
            if pos_dir.exists() and list(pos_dir.glob("*.jsonl")):
                available_var_size_n.append(n)
    available_var_size_n = sorted(available_var_size_n)
    print(f"  Available var_size n values: {available_var_size_n}")

    for n in n_values:
        if n not in available_var_size_n:
            print(f"  n={n:4d}: no var_size training data available")
            continue

        trace_dir = var_size_dir / f"n_{n}"
        result = run_alergia_on_traces(trace_dir, var_size_test_configs)
        results['var_size_to_var_size'][n] = result
        print(f"  n={n:4d}: {result['successes']}/{result['total']} ({result['success_rate']:.1%})")

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE (successes out of 50)")
    print("="*80)
    print(f"{'Permutation':<25} | " + " | ".join(f"{n:>5}" for n in n_values))
    print("-"*80)

    for key, label in [
        ('fixed_to_var_conf', 'fixed → var_conf'),
        ('var_conf_to_var_conf', 'var_conf → var_conf'),
        ('fixed_to_var_size', 'fixed → var_size'),
        ('var_size_to_var_size', 'var_size → var_size'),
    ]:
        row = []
        for n in n_values:
            if n in results[key]:
                row.append(f"{results[key][n]['successes']:>5}")
            else:
                row.append("    -")
        print(f"{label:<25} | " + " | ".join(row))


if __name__ == "__main__":
    main()
