#!/usr/bin/env python3
"""
Evaluation script for TSL specification mining.
Runs the mining pipeline on multiple benchmarks with varying trace counts.
"""

import subprocess
import os
import re
import csv
from pathlib import Path
from typing import Dict, List

# Configuration
EVAL_DIR = Path("eval2")
SRC_DIR = Path("src")
PIPELINE_SCRIPT = SRC_DIR / "pipeline.sh"
TRACER_SCRIPT = SRC_DIR / "tracer.py"
TRACE_COUNTS = [4, 6, 8, 10, 12]
TRACE_LENGTH = 6
EVAL_TRACES_POS = 50
EVAL_TRACES_NEG = 50
EVAL_TRACE_LENGTH = 6
RESULTS_CSV = EVAL_DIR / "samples.csv"


def get_benchmarks() -> List[str]:
    """Get all benchmark directories in eval/."""
    benchmarks = []
    for item in EVAL_DIR.iterdir():
        if item.is_dir():
            # Check if there's a .tsl file with the same name
            tsl_file = item / f"{item.name}.tsl"
            if tsl_file.exists():
                benchmarks.append(item.name)
    return sorted(benchmarks)


def run_pipeline(tsl_path: Path, out_path: Path, num_samples: int, length: int) -> bool:
    """Run the mining pipeline."""
    try:
        # Ensure pipeline.sh is executable
        os.chmod(PIPELINE_SCRIPT, 0o755)

        # Convert paths to absolute since we're running from src/
        tsl_abs = tsl_path.resolve()
        out_abs = out_path.resolve()

        result = subprocess.run(
            ["bash", "pipeline.sh", str(tsl_abs), str(out_abs),
             str(num_samples), str(length)],
            cwd=SRC_DIR,
            capture_output=True,
            text=True,
            timeout=3600  # 1 h timeout
        )

        if result.returncode != 0:
            print(f"  ⚠️  Pipeline failed with return code {result.returncode}")
            print(f"  stderr: {result.stderr}")
            return False

        return True
    except subprocess.TimeoutExpired:
        print(f"  ⚠️  Pipeline timed out after 2 hours")
        return False
    except Exception as e:
        print(f"  ⚠️  Pipeline error: {e}")
        return False


def check_ltl_exists(ltl_path: Path) -> bool:
    """Check if LTL file was created by the pipeline."""
    if not ltl_path.exists():
        print(f"  ⚠️  LTL file not found: {ltl_path}")
        return False

    # Check if file is not empty
    if ltl_path.stat().st_size == 0:
        print(f"  ⚠️  LTL file is empty: {ltl_path}")
        return False

    return True


def run_evaluation(tsl_path: Path, tslm_path: Path, log_path: Path) -> float:
    """Run tracer.py in eval mode and extract accuracy."""
    try:
        # Convert paths to absolute since we're running from src/
        tsl_abs = tsl_path.resolve()
        tslm_abs = tslm_path.resolve()

        result = subprocess.run(
            ["python", "tracer.py", "eval",
             "--tsl", str(tsl_abs),
             "--tslm", str(tslm_abs),
             "-p", str(EVAL_TRACES_POS),
             "-n", str(EVAL_TRACES_NEG),
             "-l", str(EVAL_TRACE_LENGTH),
             "--timeout", "300"],
            cwd=SRC_DIR,
            capture_output=True,
            text=True,
            timeout=900  # 10 minute timeout
        )

        # Write full output to log file
        with open(log_path, 'w') as f:
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)
            f.write(f"\n=== RETURN CODE: {result.returncode} ===\n")

        # Extract accuracy from the last line
        # Looking for: "Total accuracy: 0.75"
        output = result.stdout
        match = re.search(r"Total accuracy:\s+([\d.]+)", output)

        if match:
            accuracy = float(match.group(1))
            return accuracy
        else:
            print(f"  ⚠️  Could not parse accuracy from output")
            return -1.0

    except subprocess.TimeoutExpired:
        print(f"  ⚠️  Evaluation timed out")
        with open(log_path, 'w') as f:
            f.write("TIMEOUT during evaluation\n")
        return -1.0
    except Exception as e:
        print(f"  ⚠️  Evaluation error: {e}")
        with open(log_path, 'w') as f:
            f.write(f"ERROR: {e}\n")
        return -1.0


def main():
    """Main evaluation loop."""
    print("=" * 60)
    print("TSL Specification Mining Evaluation")
    print("=" * 60)
    print(f"Trace counts: {TRACE_COUNTS}")
    print(f"Trace length: {TRACE_LENGTH}")
    print(f"Eval traces: {EVAL_TRACES_POS} positive, {EVAL_TRACES_NEG} negative")
    print("=" * 60)

    # Get all benchmarks
    benchmarks = get_benchmarks()
    print(f"\nFound {len(benchmarks)} benchmarks: {', '.join(benchmarks)}\n")

    # Results storage: {benchmark: {num_samples: accuracy}}
    results: Dict[str, Dict[int, float]] = {}

    # Process each benchmark
    for benchmark in benchmarks:
        print(f"\n{'='*60}")
        print(f"Processing: {benchmark}")
        print(f"{'='*60}")

        benchmark_dir = EVAL_DIR / benchmark
        tsl_path = benchmark_dir / f"{benchmark}.tsl"

        results[benchmark] = {}

        # Process each trace count
        for num_samples in TRACE_COUNTS:
            config_name = f"{benchmark}n{num_samples}l{TRACE_LENGTH}"
            config_dir = benchmark_dir / config_name

            print(f"\n[{config_name}]")

            # Create config directory
            config_dir.mkdir(exist_ok=True)

            # Define paths
            out_tsl = config_dir / f"{config_name}.tsl"
            out_ltl = config_dir / f"{config_name}.tsl.ltl"
            eval_log = config_dir / "eval_debug.log"

            # Check if already processed
            if out_ltl.exists() and eval_log.exists():
                # Try to extract accuracy from existing log
                try:
                    with open(eval_log, 'r') as f:
                        log_content = f.read()
                    match = re.search(r"Total accuracy:\s+([\d.]+)", log_content)
                    if match:
                        accuracy = float(match.group(1))
                        results[benchmark][num_samples] = accuracy
                        print(f"  ⏭️  Already processed - Accuracy: {accuracy:.2f}")
                        continue
                except:
                    pass  # If we can't read the log, re-run the evaluation

            # Step 1: Run pipeline (this creates both .tsl and .tsl.ltl)
            skip_pipeline = out_ltl.exists()
            if skip_pipeline:
                print(f"  1/2 Skipping mining pipeline (output exists)...")
                success = True
            else:
                print(f"  1/2 Running mining pipeline...")
                success = run_pipeline(tsl_path, out_tsl, num_samples, TRACE_LENGTH)

            if not success:
                print(f"  ❌ Pipeline failed for {config_name}")
                results[benchmark][num_samples] = -1.0
                continue

            # Check if LTL file was created
            if not check_ltl_exists(out_ltl):
                print(f"  ❌ LTL file not created for {config_name}")
                results[benchmark][num_samples] = -1.0
                continue

            # Step 2: Evaluate
            print(f"  2/2 Running evaluation...")
            accuracy = run_evaluation(tsl_path, out_ltl, eval_log)

            results[benchmark][num_samples] = accuracy

            if accuracy >= 0:
                print(f"  ✓ Accuracy: {accuracy:.2f}")
            else:
                print(f"  ❌ Evaluation failed")

    # Write results to CSV
    print(f"\n{'='*60}")
    print("Writing results to CSV...")
    print(f"{'='*60}\n")

    with open(RESULTS_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Header
        header = ['num_samples'] + [str(n) for n in TRACE_COUNTS]
        writer.writerow(header)

        # Data rows
        for benchmark in sorted(results.keys()):
            row = [benchmark]
            for num_samples in TRACE_COUNTS:
                accuracy = results[benchmark].get(num_samples, -1.0)
                if accuracy >= 0:
                    row.append(f"{accuracy:.2f}")
                else:
                    row.append("FAILED")
            writer.writerow(row)

    print(f"✓ Results written to: {RESULTS_CSV}")

    # Print summary table
    print(f"\n{'='*60}")
    print("Summary Results")
    print(f"{'='*60}\n")

    # Print header
    print(f"{'Benchmark':<20}", end='')
    for n in TRACE_COUNTS:
        print(f"{n:>8}", end='')
    print()
    print('-' * 60)

    # Print data
    for benchmark in sorted(results.keys()):
        print(f"{benchmark:<20}", end='')
        for num_samples in TRACE_COUNTS:
            accuracy = results[benchmark].get(num_samples, -1.0)
            if accuracy >= 0:
                print(f"{accuracy:>8.2f}", end='')
            else:
                print(f"{'FAIL':>8}", end='')
        print()

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
