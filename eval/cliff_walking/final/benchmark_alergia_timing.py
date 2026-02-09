#!/usr/bin/env python3
"""Quick benchmark to collect Alergia training times for CliffWalking."""

import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cliff_walking.baselines import alergia_baseline


def benchmark_alergia_timing():
    base_dir = Path(__file__).parent.parent / "full_eval"
    n_values = [4, 8, 12, 16, 20, 50, 100, 500, 1000]

    results = {
        'fixed_moves': {},
        'var_config_moves': {}
    }

    for train_mode in ['var_moves', 'var_config_moves']:
        key = 'fixed_moves' if train_mode == 'var_moves' else 'var_config_moves'
        print(f"\n=== {train_mode} ===")
        mode_dir = base_dir / train_mode

        if not mode_dir.exists():
            print(f"  Directory not found: {mode_dir}")
            continue

        for n in n_values:
            trace_dir = mode_dir / f"n_{n}"
            pos_dir = trace_dir / "pos"

            if not pos_dir.exists() or not list(pos_dir.glob("*.jsonl")):
                print(f"  n={n:4d}: no traces")
                continue

            try:
                start = time.time()
                model, meta = alergia_baseline.train(trace_dir, eps=2.0)
                train_time = time.time() - start

                results[key][n] = {
                    'train_time': round(train_time, 4),
                    'num_states': meta.get('num_states'),
                    'num_traces': meta.get('num_traces')
                }
                print(f"  n={n:4d}: {train_time:.4f}s ({meta.get('num_states', '?')} states)")
            except Exception as e:
                print(f"  n={n:4d}: ERROR - {e}")
                results[key][n] = {'error': str(e)}

    # Save results
    output_file = Path(__file__).parent / "alergia_timing.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    benchmark_alergia_timing()
