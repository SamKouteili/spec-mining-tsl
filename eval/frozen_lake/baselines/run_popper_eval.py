#!/usr/bin/env python3
"""
Full Popper Evaluation Script

Runs Popper baseline evaluation across multiple training trace sizes,
both with and without movement functions as background knowledge.

Usage:
    python run_popper_eval.py --max-n 100 --num-tests 100
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from run_full_eval import generate_test_configs

# Import from same directory
try:
    from . import popper_baseline
except ImportError:
    import popper_baseline


def generate_traces(n: int, output_dir: Path, train_mode: str = "var_config") -> bool:
    """
    Generate n positive + n negative traces using tfrozen_lake_game.py.

    Args:
        n: Number of positive (and negative) traces to generate
        output_dir: Directory to store traces
        train_mode: 'var_config' or 'var_size'

    Returns:
        True if generation succeeded
    """
    game_script = Path(__file__).parent.parent.parent.parent / "games" / "tfrozen_lake_game.py"

    if not game_script.exists():
        print(f"  ERROR: Game script not found: {game_script}")
        return False

    # Build command
    cmd = [
        sys.executable, str(game_script),
        "--gen", str(n),
        "--output", str(output_dir)
    ]

    # Add flags based on train_mode
    if train_mode in ["var_config", "var_size"]:
        cmd.append("--random-placements")
    if train_mode == "var_size":
        cmd.append("--random-size")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  ERROR: Trace generation failed: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  ERROR: Trace generation timed out")
        return False


def run_evaluation(
    n_values: List[int],
    num_tests: int,
    test_config: str,
    train_mode: str,
    timeout_per_action: int = 60,
    output_dir: Path = None
) -> Dict:
    """
    Run full evaluation across multiple n values.

    Returns dict with results for each n value and with/without functions.
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="popper_eval_"))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate test configs once (same for all runs)
    print(f"Generating {num_tests} {test_config} test configs (seed=42)...")
    test_configs = generate_test_configs(num_tests, test_config, seed=42)
    print(f"  Generated {len(test_configs)} configs")

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_values": n_values,
            "num_tests": num_tests,
            "test_config": test_config,
            "train_mode": train_mode,
            "timeout_per_action": timeout_per_action
        },
        "results": {}
    }

    for n in n_values:
        print(f"\n{'='*60}")
        print(f"  n = {n}")
        print(f"{'='*60}")

        trace_dir = output_dir / f"n_{n}"

        # Generate traces if needed
        if not (trace_dir / "pos").exists() or len(list((trace_dir / "pos").glob("*.jsonl"))) < n:
            print(f"  Generating {n} pos + {n} neg traces...")
            trace_dir.mkdir(parents=True, exist_ok=True)
            if not generate_traces(n, trace_dir, train_mode):
                results["results"][n] = {"error": "Trace generation failed"}
                continue
        else:
            print(f"  Using existing traces from {trace_dir}")

        # Count actual traces
        pos_count = len(list((trace_dir / "pos").glob("*.jsonl")))
        neg_count = len(list((trace_dir / "neg").glob("*.jsonl"))) if (trace_dir / "neg").exists() else 0
        print(f"  Found {pos_count} pos, {neg_count} neg traces")

        results["results"][n] = {}

        # Run WITHOUT functions
        print(f"\n  [1/2] Popper WITHOUT functions...")
        start = time.time()
        try:
            result_no_func = popper_baseline.train_and_evaluate(
                trace_dir, test_configs,
                timeout_per_action=timeout_per_action,
                with_functions=False
            )
            result_no_func["wall_time"] = time.time() - start
            win_rate = result_no_func["successes"] / result_no_func["total"] * 100
            print(f"    Win rate: {result_no_func['successes']}/{result_no_func['total']} ({win_rate:.1f}%)")
            print(f"    Rules learned: {result_no_func.get('num_rules', 0)}")
            print(f"    Train time: {result_no_func.get('train_time', 0):.2f}s")
        except Exception as e:
            result_no_func = {"error": str(e)}
            print(f"    ERROR: {e}")

        results["results"][n]["without_functions"] = result_no_func

        # Run WITH functions
        print(f"\n  [2/2] Popper WITH functions...")
        start = time.time()
        try:
            result_with_func = popper_baseline.train_and_evaluate(
                trace_dir, test_configs,
                timeout_per_action=timeout_per_action,
                with_functions=True
            )
            result_with_func["wall_time"] = time.time() - start
            win_rate = result_with_func["successes"] / result_with_func["total"] * 100
            print(f"    Win rate: {result_with_func['successes']}/{result_with_func['total']} ({win_rate:.1f}%)")
            print(f"    Rules learned: {result_with_func.get('num_rules', 0)}")
            print(f"    Train time: {result_with_func.get('train_time', 0):.2f}s")
        except Exception as e:
            result_with_func = {"error": str(e)}
            print(f"    ERROR: {e}")

        results["results"][n]["with_functions"] = result_with_func

    return results


def print_summary_table(results: Dict):
    """Print a summary table of results."""
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'n':>5} | {'Without Functions':^25} | {'With Functions':^25}")
    print(f"{'':>5} | {'Win Rate':>10} {'Rules':>6} {'Time':>7} | {'Win Rate':>10} {'Rules':>6} {'Time':>7}")
    print("-"*80)

    for n, data in sorted(results["results"].items(), key=lambda x: int(x[0])):
        n = int(n)

        # Without functions
        no_func = data.get("without_functions", {})
        if "error" in no_func:
            no_func_str = "ERROR"
            no_func_rules = "-"
            no_func_time = "-"
        else:
            win_rate = no_func.get("successes", 0) / no_func.get("total", 1) * 100
            no_func_str = f"{win_rate:.1f}%"
            no_func_rules = str(no_func.get("num_rules", 0))
            no_func_time = f"{no_func.get('train_time', 0):.1f}s"

        # With functions
        with_func = data.get("with_functions", {})
        if "error" in with_func:
            with_func_str = "ERROR"
            with_func_rules = "-"
            with_func_time = "-"
        else:
            win_rate = with_func.get("successes", 0) / with_func.get("total", 1) * 100
            with_func_str = f"{win_rate:.1f}%"
            with_func_rules = str(with_func.get("num_rules", 0))
            with_func_time = f"{with_func.get('train_time', 0):.1f}s"

        print(f"{n:>5} | {no_func_str:>10} {no_func_rules:>6} {no_func_time:>7} | {with_func_str:>10} {with_func_rules:>6} {with_func_time:>7}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Run full Popper evaluation')
    parser.add_argument('--max-n', type=int, default=100,
                        help='Maximum number of training traces')
    parser.add_argument('--n-values', type=str, default=None,
                        help='Comma-separated list of n values (overrides --max-n)')
    parser.add_argument('--num-tests', type=int, default=100,
                        help='Number of test configurations')
    parser.add_argument('--test-config', type=str, default='var_config',
                        choices=['var_config', 'var_size'],
                        help='Test configuration type')
    parser.add_argument('--train-mode', type=str, default='var_config',
                        choices=['fixed', 'var_config', 'var_size'],
                        help='Training trace configuration type')
    parser.add_argument('--timeout', type=int, default=60,
                        help='Timeout per action in seconds')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for traces and results')
    parser.add_argument('--results-file', type=str, default=None,
                        help='File to save results JSON')
    args = parser.parse_args()

    # Parse n values
    if args.n_values:
        n_values = [int(x.strip()) for x in args.n_values.split(',')]
    else:
        # Default progression up to max_n
        all_n = [5, 10, 15, 20, 25, 50, 75, 100]
        n_values = [n for n in all_n if n <= args.max_n]

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent / f"popper_eval_{timestamp}"

    print("="*80)
    print("POPPER FULL EVALUATION")
    print("="*80)
    print(f"  n values: {n_values}")
    print(f"  Test configs: {args.num_tests} ({args.test_config})")
    print(f"  Train mode: {args.train_mode}")
    print(f"  Timeout per action: {args.timeout}s")
    print(f"  Output directory: {output_dir}")
    print("="*80)

    # Run evaluation
    results = run_evaluation(
        n_values=n_values,
        num_tests=args.num_tests,
        test_config=args.test_config,
        train_mode=args.train_mode,
        timeout_per_action=args.timeout,
        output_dir=output_dir
    )

    # Print summary
    print_summary_table(results)

    # Save results
    results_file = args.results_file or (output_dir / "results.json")
    # Remove details to keep file size manageable
    results_compact = {
        "metadata": results["metadata"],
        "results": {}
    }
    for n, data in results["results"].items():
        results_compact["results"][n] = {}
        for key in ["without_functions", "with_functions"]:
            if key in data:
                entry = data[key].copy()
                entry.pop("details", None)  # Remove per-config details
                results_compact["results"][n][key] = entry

    with open(results_file, "w") as f:
        json.dump(results_compact, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
