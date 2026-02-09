#!/usr/bin/env python3
"""
Run Alergia baseline evaluation for Blackjack.

Uses existing traces from eval/blackjack/results/traces/.
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
BLACKJACK_EVAL_DIR = PROJECT_ROOT / "eval" / "blackjack"

sys.path.insert(0, str(BLACKJACK_EVAL_DIR))

from baselines.alergia_baseline import train_and_evaluate as train_alergia
from test_harness import (
    BlackjackSimulator,
    make_alergia_action_fn,
)

STRATEGIES = ["threshold", "conservative", "basic"]
N_VALUES = [4, 8, 12, 16, 20]
EVAL_SEED = 42
NUM_WIN_TESTS = 100
NUM_STRATEGY_TESTS = 50


def evaluate_alergia(model, strategy: str) -> dict:
    """Evaluate Alergia model."""
    if model is None:
        return {
            "win_rate": 0.0,
            "strategy_adherence": 0.0,
            "games_won": 0,
            "games_tested_win": 0,
            "games_followed": 0,
            "games_tested_strategy": 0,
            "error": "Model training failed"
        }

    action_fn = make_alergia_action_fn(model, strategy)
    simulator = BlackjackSimulator(strategy, seed=EVAL_SEED)

    # Evaluate win rate
    win_result = simulator.evaluate(action_fn, NUM_WIN_TESTS, seed=EVAL_SEED)

    # Evaluate strategy adherence
    strategy_result = simulator.evaluate(action_fn, NUM_STRATEGY_TESTS, seed=EVAL_SEED + 1000)

    return {
        "win_rate": win_result.win_rate,
        "games_won": win_result.games_won,
        "games_tested_win": win_result.games_played,
        "strategy_adherence": strategy_result.strategy_adherence,
        "games_followed": strategy_result.games_strategy_followed,
        "games_tested_strategy": strategy_result.games_played,
        "error": None
    }


def main():
    traces_dir = BLACKJACK_EVAL_DIR / "results" / "traces"
    output_file = SCRIPT_DIR / "alergia_results.json"

    results = []

    print("=" * 60)
    print("ALERGIA BASELINE EVALUATION - BLACKJACK")
    print("=" * 60)

    for strategy in STRATEGIES:
        print(f"\n--- Strategy: {strategy.upper()} ---")

        for n in N_VALUES:
            trace_dir = traces_dir / strategy / f"n_{n}"

            if not trace_dir.exists():
                print(f"  n={n}: trace dir not found, skipping")
                results.append({
                    "method": "alergia",
                    "strategy": strategy,
                    "n": n,
                    "error": "Traces not found"
                })
                continue

            try:
                # Train Alergia
                model, train_meta = train_alergia(trace_dir, strategy)

                # Evaluate
                eval_result = evaluate_alergia(model, strategy)

                result = {
                    "method": "alergia",
                    "strategy": strategy,
                    "n": n,
                    "win_rate": eval_result["win_rate"],
                    "strategy_adherence": eval_result["strategy_adherence"],
                    "games_won": eval_result["games_won"],
                    "games_tested_win": eval_result["games_tested_win"],
                    "games_followed": eval_result["games_followed"],
                    "games_tested_strategy": eval_result["games_tested_strategy"],
                    "train_time": train_meta.get("train_time", 0),
                    "num_states": train_meta.get("num_states"),
                    "error": eval_result.get("error")
                }

                adh = result["strategy_adherence"]
                win = result["win_rate"]
                states = result.get("num_states", "?")
                print(f"  n={n}: {adh:.0%} adherence, {win:.0%} win rate, {states} states")

            except Exception as e:
                print(f"  n={n}: ERROR - {e}")
                result = {
                    "method": "alergia",
                    "strategy": strategy,
                    "n": n,
                    "win_rate": 0.0,
                    "strategy_adherence": 0.0,
                    "error": str(e)
                }

            results.append(result)

    # Save results
    with open(output_file, 'w') as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Alergia Results - Strategy%(Win%)")
    print("=" * 70)
    print(f"{'Strategy':<14} | " + " | ".join(f"n={n:>2}" for n in N_VALUES))
    print("-" * 70)

    for strategy in STRATEGIES:
        row = f"{strategy.capitalize():<14} |"
        for n in N_VALUES:
            for r in results:
                if r["strategy"] == strategy and r["n"] == n:
                    if r.get("error"):
                        row += "   ERR   |"
                    else:
                        adh = r["strategy_adherence"] * 100
                        win = r["win_rate"] * 100
                        row += f" {adh:>2.0f}%({win:>2.0f}%) |"
                    break
        print(row)


if __name__ == "__main__":
    main()
