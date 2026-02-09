#!/usr/bin/env python3
"""Quick benchmark to collect Alergia training times for Blackjack."""

import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from blackjack.baselines import alergia_baseline


def benchmark_alergia_timing():
    base_dir = Path(__file__).parent / "results" / "traces"
    n_values = [4, 8, 12, 16, 20]
    strategies = ["threshold", "conservative", "basic"]

    results = {}

    for strategy in strategies:
        results[strategy] = {}
        print(f"\n=== {strategy} ===")
        strategy_dir = base_dir / strategy

        if not strategy_dir.exists():
            print(f"  Directory not found: {strategy_dir}")
            continue

        for n in n_values:
            trace_dir = strategy_dir / f"n_{n}"
            pos_dir = trace_dir / "pos"

            if not pos_dir.exists() or not list(pos_dir.glob("*.jsonl")):
                print(f"  n={n:4d}: no traces")
                continue

            try:
                start = time.time()
                model, meta = alergia_baseline.train(trace_dir, strategy=strategy, eps=2.0)
                train_time = time.time() - start

                results[strategy][n] = {
                    'train_time': round(train_time, 4),
                    'num_states': meta.get('num_states'),
                    'num_traces': meta.get('num_traces')
                }
                print(f"  n={n:4d}: {train_time:.4f}s ({meta.get('num_states', '?')} states)")
            except Exception as e:
                print(f"  n={n:4d}: ERROR - {e}")
                results[strategy][n] = {'error': str(e)}

    # Save results
    output_file = Path(__file__).parent / "alergia_timing.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\n" + "="*50)
    print("ALERGIA TRAINING TIME (seconds)")
    print("="*50)
    print(f"{'Strategy':<15} | " + " | ".join(f"{n:>6}" for n in n_values))
    print("-"*50)

    for strategy in strategies:
        row = []
        for n in n_values:
            if n in results.get(strategy, {}) and 'train_time' in results[strategy][n]:
                row.append(f"{results[strategy][n]['train_time']:>6.4f}")
            else:
                row.append("     -")
        print(f"{strategy:<15} | " + " | ".join(row))


if __name__ == "__main__":
    benchmark_alergia_timing()
