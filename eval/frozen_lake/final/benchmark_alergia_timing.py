#!/usr/bin/env python3
"""Quick benchmark to collect Alergia training times."""

import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from frozen_lake.baselines import alergia_baseline


def benchmark_alergia_timing():
    base_dir = Path(__file__).parent.parent / "full_eval"
    n_values = [4, 8, 12, 16, 20, 50, 100, 500, 1000]

    # Dummy test configs (we only care about training time)
    dummy_configs = [{'grid_size': 4, 'goal': {'x': 3, 'y': 3},
                      'holes': [{'x': 1, 'y': 1}, {'x': 2, 'y': 2}, {'x': 3, 'y': 2}]}]

    results = {
        'fixed': {},
        'var_config': {},
        'var_size': {}
    }

    for train_mode in ['fixed', 'var_config', 'var_size']:
        print(f"\n=== {train_mode} ===")
        mode_dir = base_dir / train_mode if train_mode != 'var_config' else base_dir / 'var_config'

        for n in n_values:
            trace_dir = mode_dir / f"n_{n}"
            pos_dir = trace_dir / "pos"

            if not pos_dir.exists() or not list(pos_dir.glob("*.jsonl")):
                print(f"  n={n:4d}: no traces")
                continue

            # Time only training (not evaluation)
            try:
                start = time.time()
                model, meta = alergia_baseline.train(trace_dir, eps=2.0)
                train_time = time.time() - start

                results[train_mode][n] = {
                    'train_time': round(train_time, 4),
                    'num_states': meta.get('num_states'),
                    'num_traces': meta.get('num_traces')
                }
                print(f"  n={n:4d}: {train_time:.4f}s ({meta.get('num_states', '?')} states)")
            except Exception as e:
                print(f"  n={n:4d}: ERROR - {e}")
                results[train_mode][n] = {'error': str(e)}

    # Save results
    output_file = Path(__file__).parent / "alergia_timing.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print summary table
    print("\n" + "="*70)
    print("ALERGIA TRAINING TIME (seconds)")
    print("="*70)
    print(f"{'Train Mode':<15} | " + " | ".join(f"{n:>6}" for n in n_values))
    print("-"*70)

    for mode in ['fixed', 'var_config', 'var_size']:
        row = []
        for n in n_values:
            if n in results[mode] and 'train_time' in results[mode][n]:
                row.append(f"{results[mode][n]['train_time']:>6.3f}")
            else:
                row.append("     -")
        print(f"{mode:<15} | " + " | ".join(row))


if __name__ == "__main__":
    benchmark_alergia_timing()
