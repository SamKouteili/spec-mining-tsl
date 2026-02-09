#!/usr/bin/env python3
"""
Run the TSL_f mining pipeline on all generated trace sets.
Collects results into a summary JSON.

Supported games:
- frozen_lake: Navigate grid to goal, avoid holes
- taxi: Pick up passenger, deliver to destination
- cliff_walking: Navigate to goal, avoid cliff
- blackjack: Card game, don't bust
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime

# Configuration
TRACE_COUNTS = [5, 10, 15, 20, 25]
NUM_SETS = 5
CONDITIONS = ["fixed", "random_pos", "random_size"]
MAX_SIZE = 40

# Supported games
GAMES = ["frozen_lake", "taxi", "cliff_walking", "blackjack"]

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
EVAL_DIR = PROJECT_DIR / "eval"
SRC_DIR = PROJECT_DIR / "src"


def run_pipeline(trace_dir: Path, max_size: int = MAX_SIZE) -> dict:
    """Run pipeline on a trace directory, return results."""
    result = {
        "success": False,
        "liveness": "",
        "safety": "",
        "spec": "",
        "error": ""
    }

    # Check if traces exist
    pos_dir = trace_dir / "pos"
    neg_dir = trace_dir / "neg"
    if not pos_dir.exists() or not neg_dir.exists():
        result["error"] = "Missing pos/neg directories"
        return result

    # Run pipeline
    cmd = [
        "./mine.sh",
        str(trace_dir),
        "--mode", "safety-liveness",
        "--collect-all",
        "--max-size", str(max_size),
        "--self-inputs-only",
        "--prune"
    ]

    try:
        proc = subprocess.run(
            cmd, cwd=str(SRC_DIR),
            capture_output=True, text=True, timeout=300
        )

        # Read output files
        out_dir = trace_dir / "out"

        if (out_dir / "liveness.tsl").exists():
            result["liveness"] = (out_dir / "liveness.tsl").read_text().strip()

        if (out_dir / "safety.tsl").exists():
            result["safety"] = (out_dir / "safety.tsl").read_text().strip()

        if (out_dir / "spec.tsl").exists():
            result["spec"] = (out_dir / "spec.tsl").read_text().strip()

        result["success"] = bool(result["liveness"] or result["safety"])

        if not result["success"]:
            result["error"] = "No specs found"

    except subprocess.TimeoutExpired:
        result["error"] = "Pipeline timeout"
    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run TSL_f pipeline on generated traces")
    parser.add_argument("game", choices=GAMES,
                        help="Game to run pipeline for")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS,
                        default=CONDITIONS,
                        help="Conditions to process (default: all)")
    parser.add_argument("--counts", nargs="+", type=int,
                        default=TRACE_COUNTS,
                        help="Trace counts to process (default: 5 10 15 20 25)")
    parser.add_argument("--sets", type=int, default=NUM_SETS,
                        help="Number of sets per count (default: 5)")
    parser.add_argument("--max-size", type=int, default=MAX_SIZE,
                        help="Maximum formula size for Bolt (default: 10)")
    args = parser.parse_args()

    max_size = args.max_size

    print("=" * 60)
    print(f"TSL_f Pipeline Execution - {args.game}")
    print("=" * 60)
    print(f"Game: {args.game}")
    print(f"Conditions: {args.conditions}")
    print(f"Trace counts: {args.counts}")
    print(f"Sets per count: {args.sets}")
    print(f"Max formula size: {max_size}")
    print()

    game_dir = EVAL_DIR / args.game
    all_results = []

    for condition in args.conditions:
        print(f"\n=== Condition: {condition} ===")

        for n in args.counts:
            for set_id in range(1, args.sets + 1):
                trace_dir = game_dir / condition / f"n_{n}" / f"set_{set_id}"

                print(f"  Processing {condition}/n_{n}/set_{set_id}...", end=" ", flush=True)

                if not trace_dir.exists():
                    print("SKIP (not found)")
                    continue

                result = run_pipeline(trace_dir, max_size)

                if result["success"]:
                    print("OK")
                else:
                    print(f"FAIL: {result['error']}")

                # Store result
                all_results.append({
                    "game": args.game,
                    "condition": condition,
                    "n": n,
                    "set": set_id,
                    **result
                })

        # Write per-condition results
        cond_dir = game_dir / condition
        if cond_dir.exists():
            results_file = cond_dir / "results.json"
            condition_results = [r for r in all_results if r["condition"] == condition]
            with open(results_file, "w") as f:
                json.dump(condition_results, f, indent=2)
            print(f"  Results saved to: {results_file}")

    # Write combined results for this game
    combined_file = game_dir / "all_results.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to: {combined_file}")

    # Generate summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for condition in args.conditions:
        cond_results = [r for r in all_results if r["condition"] == condition]
        if not cond_results:
            continue

        success_count = sum(1 for r in cond_results if r["success"])
        total = len(cond_results)
        print(f"\n{condition}:")
        print(f"  Success rate: {success_count}/{total} ({100*success_count/total:.1f}%)")

        # Show unique specs found
        unique_liveness = set(r["liveness"] for r in cond_results if r["liveness"])
        unique_safety = set(r["safety"] for r in cond_results if r["safety"])
        print(f"  Unique liveness specs: {len(unique_liveness)}")
        print(f"  Unique safety specs: {len(unique_safety)}")

        if unique_liveness:
            print(f"  Liveness examples:")
            for spec in list(unique_liveness)[:2]:
                print(f"    - {spec[:70]}...")

        if unique_safety:
            print(f"  Safety examples:")
            for spec in list(unique_safety)[:2]:
                print(f"    - {spec[:70]}...")

    print("\n" + "=" * 60)
    print("Pipeline execution complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
