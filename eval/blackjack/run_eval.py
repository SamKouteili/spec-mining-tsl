#!/usr/bin/env python3
"""
Blackjack Evaluation Script

Evaluates learning methods (TSLf, Alergia, BC, DT) across all strategies
(threshold, conservative, basic) at different training sizes.

Usage:
    python run_eval.py --n-values 4 8 12 16 20 --num-win-tests 100 --num-strategy-tests 50

Output: LaTeX table with format:
    Method    Strategy       4    8   12   16   20 ...
    TSLf      Threshold     85%(42%)  ...
              Conservative  ...
              Basic         ...
    Alergia   ...

Where: strategy_followed%(win% in brackets)
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
GAMES_DIR = PROJECT_ROOT / "games"
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "games"))

STRATEGIES = ["threshold", "conservative", "basic"]
METHODS = ["tslf", "alergia", "bc", "dt"]
DEFAULT_N_VALUES = [4, 8, 12, 16, 20]
TRAIN_SEED = 12345
EVAL_SEED = 42


# ============== Result Dataclasses ==============

@dataclass
class MethodResult:
    """Result for one method on one strategy at one N value."""
    method: str
    strategy: str
    n: int
    win_rate: float
    strategy_adherence: float
    games_won: int = 0
    games_tested_win: int = 0
    games_followed: int = 0
    games_tested_strategy: int = 0
    train_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class EvalResults:
    """All evaluation results."""
    results: List[MethodResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    def get_result(self, method: str, strategy: str, n: int) -> Optional[MethodResult]:
        for r in self.results:
            if r.method == method and r.strategy == strategy and r.n == n:
                return r
        return None


# ============== Trace Generation ==============

def generate_traces(strategy: str, n: int, output_dir: Path) -> bool:
    """Generate n positive and n negative traces for a strategy."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pos").mkdir(exist_ok=True)
    (output_dir / "neg").mkdir(exist_ok=True)

    game_script = GAMES_DIR / "blackjack_game.py"

    cmd = [
        sys.executable, str(game_script),
        "--gen", str(n),
        "--strategy", strategy,
        "--output", str(output_dir),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  Error generating traces: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  Trace generation timed out")
        return False


# ============== Model Training ==============

def train_bc(trace_dir: Path, strategy: str) -> Tuple[Any, Dict]:
    """Train BC model."""
    from baselines.bc_baseline import train_and_evaluate
    return train_and_evaluate(trace_dir, strategy)


def train_dt(trace_dir: Path, strategy: str) -> Tuple[Any, Dict]:
    """Train DT model."""
    from baselines.dt_baseline import train_and_evaluate
    return train_and_evaluate(trace_dir, strategy)


def train_alergia(trace_dir: Path, strategy: str) -> Tuple[Any, Dict]:
    """Train Alergia model."""
    from baselines.alergia_baseline import train_and_evaluate
    return train_and_evaluate(trace_dir, strategy)


def train_tslf(trace_dir: Path, strategy: str) -> Tuple[Any, Dict]:
    """Train TSLf model (run mining pipeline)."""
    from tslf_wrapper import train_and_evaluate
    return train_and_evaluate(trace_dir, strategy)


TRAIN_FUNCTIONS = {
    "bc": train_bc,
    "dt": train_dt,
    "alergia": train_alergia,
    "tslf": train_tslf,
}


# ============== Model Evaluation ==============

def evaluate_model(model: Any, method: str, strategy: str,
                   num_win_tests: int, num_strategy_tests: int) -> Dict:
    """
    Evaluate a trained model.

    Returns dict with win_rate, strategy_adherence, games_won, etc.
    """
    from test_harness import (
        BlackjackSimulator,
        make_bc_action_fn,
        make_dt_action_fn,
        make_alergia_action_fn,
        make_oracle_action_fn,
    )

    # Create action function based on method
    if model is None:
        # Failed to train - use random as fallback
        from test_harness import make_random_action_fn
        action_fn = make_random_action_fn(strategy)
        error = "Model training failed, using random"
    else:
        error = None
        if method == "bc":
            action_fn = make_bc_action_fn(model, strategy)
        elif method == "dt":
            action_fn = make_dt_action_fn(model, strategy)
        elif method == "alergia":
            action_fn = make_alergia_action_fn(model, strategy)
        elif method == "tslf":
            # TSLf returns action function directly
            if callable(model):
                action_fn = model
            else:
                from test_harness import make_random_action_fn
                action_fn = make_random_action_fn(strategy)
                error = "TSLf failed to produce action function"
        else:
            raise ValueError(f"Unknown method: {method}")

    # Create simulator
    simulator = BlackjackSimulator(strategy, seed=EVAL_SEED)

    # Evaluate win rate
    win_result = simulator.evaluate(action_fn, num_win_tests, seed=EVAL_SEED)

    # Evaluate strategy adherence (separate run for consistency)
    strategy_result = simulator.evaluate(action_fn, num_strategy_tests, seed=EVAL_SEED + 1000)

    return {
        "win_rate": win_result.win_rate,
        "games_won": win_result.games_won,
        "games_tested_win": win_result.games_played,
        "strategy_adherence": strategy_result.strategy_adherence,
        "games_followed": strategy_result.games_strategy_followed,
        "games_tested_strategy": strategy_result.games_played,
        "decision_accuracy": strategy_result.decision_accuracy,
        "error": error,
    }


# ============== Main Evaluation Loop ==============

def run_evaluation(n_values: List[int], num_win_tests: int, num_strategy_tests: int,
                   methods: List[str], strategies: List[str],
                   output_dir: Path) -> EvalResults:
    """Run full evaluation."""
    results = EvalResults()
    total_runs = len(methods) * len(strategies) * len(n_values)
    current_run = 0

    print("=" * 70)
    print("  BLACKJACK EVALUATION")
    print("=" * 70)
    print(f"  Methods: {methods}")
    print(f"  Strategies: {strategies}")
    print(f"  N values: {n_values}")
    print(f"  Win tests per config: {num_win_tests}")
    print(f"  Strategy tests per config: {num_strategy_tests}")
    print(f"  Total runs: {total_runs}")
    print("=" * 70)

    for strategy in strategies:
        print(f"\n{'=' * 60}")
        print(f"  STRATEGY: {strategy.upper()}")
        print("=" * 60)

        for n in n_values:
            print(f"\n  --- N = {n} ---")

            # Generate traces
            trace_dir = output_dir / strategy / f"n_{n}"
            print(f"  Generating {n} traces...")
            if not generate_traces(strategy, n, trace_dir):
                print(f"  FAILED to generate traces")
                for method in methods:
                    results.results.append(MethodResult(
                        method=method,
                        strategy=strategy,
                        n=n,
                        win_rate=0.0,
                        strategy_adherence=0.0,
                        error="Trace generation failed"
                    ))
                continue

            for method in methods:
                current_run += 1
                print(f"\n  [{current_run}/{total_runs}] {method.upper()} on {strategy} n={n}")

                try:
                    # Train
                    train_fn = TRAIN_FUNCTIONS[method]
                    start_time = time.time()
                    model, train_meta = train_fn(trace_dir, strategy)
                    train_time = time.time() - start_time
                    print(f"    Training: {train_time:.1f}s")

                    # Evaluate
                    eval_result = evaluate_model(
                        model, method, strategy,
                        num_win_tests, num_strategy_tests
                    )

                    result = MethodResult(
                        method=method,
                        strategy=strategy,
                        n=n,
                        win_rate=eval_result["win_rate"],
                        strategy_adherence=eval_result["strategy_adherence"],
                        games_won=eval_result["games_won"],
                        games_tested_win=eval_result["games_tested_win"],
                        games_followed=eval_result["games_followed"],
                        games_tested_strategy=eval_result["games_tested_strategy"],
                        train_time=train_time,
                        error=eval_result.get("error"),
                        metadata=train_meta
                    )

                    print(f"    Win rate: {result.win_rate:.1%} ({result.games_won}/{result.games_tested_win})")
                    print(f"    Strategy: {result.strategy_adherence:.1%} ({result.games_followed}/{result.games_tested_strategy})")

                except Exception as e:
                    print(f"    ERROR: {e}")
                    result = MethodResult(
                        method=method,
                        strategy=strategy,
                        n=n,
                        win_rate=0.0,
                        strategy_adherence=0.0,
                        error=str(e)
                    )

                results.results.append(result)

    return results


# ============== Output Formatting ==============

def generate_latex_table(results: EvalResults, n_values: List[int]) -> str:
    """Generate LaTeX table from results."""
    # Header
    n_cols = " & ".join([f"n={n}" for n in n_values])
    latex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Blackjack Evaluation Results: Strategy Adherence (Win Rate)}",
        r"\begin{tabular}{ll" + "c" * len(n_values) + "}",
        r"\toprule",
        f"Method & Strategy & {n_cols} \\\\",
        r"\midrule",
    ]

    for method in METHODS:
        method_rows = []
        for i, strategy in enumerate(STRATEGIES):
            cells = []
            for n in n_values:
                r = results.get_result(method, strategy, n)
                if r and r.error is None:
                    cell = f"{r.strategy_adherence:.0%}({r.win_rate:.0%})"
                elif r:
                    cell = "ERR"
                else:
                    cell = "-"
                cells.append(cell)

            if i == 0:
                row = f"{method.upper()} & {strategy.capitalize()} & " + " & ".join(cells) + r" \\"
            else:
                row = f" & {strategy.capitalize()} & " + " & ".join(cells) + r" \\"
            method_rows.append(row)

        latex.extend(method_rows)
        latex.append(r"\midrule")

    # Remove last midrule and add bottomrule
    latex[-1] = r"\bottomrule"
    latex.extend([
        r"\end{tabular}",
        r"\label{tab:blackjack_eval}",
        r"\end{table}",
    ])

    return "\n".join(latex)


def generate_ascii_table(results: EvalResults, n_values: List[int]) -> str:
    """Generate ASCII table for console output."""
    # Calculate column widths
    method_width = max(len(m) for m in METHODS) + 1
    strategy_width = max(len(s) for s in STRATEGIES) + 1
    n_width = 12  # "XX%(YY%)" format

    # Header
    header = f"{'Method':<{method_width}} {'Strategy':<{strategy_width}}"
    for n in n_values:
        header += f" {'n=' + str(n):^{n_width}}"

    lines = [
        "=" * len(header),
        "BLACKJACK EVALUATION RESULTS",
        "Format: Strategy%(Win%)",
        "=" * len(header),
        header,
        "-" * len(header),
    ]

    for method in METHODS:
        for i, strategy in enumerate(STRATEGIES):
            if i == 0:
                row = f"{method.upper():<{method_width}}"
            else:
                row = " " * method_width

            row += f" {strategy.capitalize():<{strategy_width}}"

            for n in n_values:
                r = results.get_result(method, strategy, n)
                if r and r.error is None:
                    cell = f"{r.strategy_adherence*100:.0f}%({r.win_rate*100:.0f}%)"
                elif r:
                    cell = "ERR"
                else:
                    cell = "-"
                row += f" {cell:^{n_width}}"

            lines.append(row)
        lines.append("-" * len(header))

    return "\n".join(lines)


def save_results(results: EvalResults, output_dir: Path):
    """Save results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = results.timestamp

    # Save JSON
    json_file = output_dir / f"results_{timestamp}.json"
    json_data = {
        "timestamp": timestamp,
        "results": [
            {
                "method": r.method,
                "strategy": r.strategy,
                "n": r.n,
                "win_rate": r.win_rate,
                "strategy_adherence": r.strategy_adherence,
                "games_won": r.games_won,
                "games_tested_win": r.games_tested_win,
                "games_followed": r.games_followed,
                "games_tested_strategy": r.games_tested_strategy,
                "train_time": r.train_time,
                "error": r.error,
            }
            for r in results.results
        ]
    }
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Results saved to: {json_file}")

    return json_file


# ============== CLI ==============

def main():
    parser = argparse.ArgumentParser(description="Blackjack Evaluation")
    parser.add_argument("--n-values", type=int, nargs="+", default=DEFAULT_N_VALUES,
                        help="Training sizes to evaluate")
    parser.add_argument("--num-win-tests", type=int, default=100,
                        help="Number of games for win rate evaluation")
    parser.add_argument("--num-strategy-tests", type=int, default=50,
                        help="Number of games for strategy adherence evaluation")
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        choices=METHODS, help="Methods to evaluate")
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES,
                        choices=STRATEGIES, help="Strategies to evaluate")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results",
                        help="Output directory")
    parser.add_argument("--skip-tslf", action="store_true",
                        help="Skip TSLf (experimental, may not work well for blackjack)")

    args = parser.parse_args()

    methods = args.methods
    if args.skip_tslf and "tslf" in methods:
        methods = [m for m in methods if m != "tslf"]

    results = run_evaluation(
        n_values=args.n_values,
        num_win_tests=args.num_win_tests,
        num_strategy_tests=args.num_strategy_tests,
        methods=methods,
        strategies=args.strategies,
        output_dir=args.output_dir / "traces"
    )

    # Print results
    print("\n" + generate_ascii_table(results, args.n_values))

    # Generate and save LaTeX table
    latex = generate_latex_table(results, args.n_values)
    latex_file = args.output_dir / f"table_{results.timestamp}.tex"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latex_file.write_text(latex)
    print(f"\nLaTeX table saved to: {latex_file}")

    # Save full results
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
