#!/usr/bin/env python3
"""
Generate trace sets for TSL_f evaluation.

Generates traces for three conditions:
- fixed: Default board with fixed positions
- random_pos: Random goal/obstacle placements
- random_size: Random board dimensions

For each: n=5,10,15,20,25 traces x 5 independent sets

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
import shutil
from pathlib import Path

# Configuration
TRACE_COUNTS = [5, 10, 15, 20, 25]
NUM_SETS = 5
CONDITIONS = ["fixed", "random_pos", "random_size"]

# Game configurations: game_name -> (script_name, log_subdir)
GAMES = {
    "frozen_lake": ("tfrozen_lake_game.py", "tfrozen_lake"),
    "taxi": ("ttaxi_game.py", "ttaxi"),
    "cliff_walking": ("cliff_walking_game.py", "cliff_walking"),
    "blackjack": ("tblackjack_game.py", "tblackjack"),
}

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
GAMES_DIR = PROJECT_DIR / "games"
EVAL_DIR = PROJECT_DIR / "eval"


def generate_traces(game: str, condition: str, n: int, set_id: int):
    """Generate traces for a specific game/condition/n/set combination."""
    script_name, log_subdir = GAMES[game]

    out_dir = EVAL_DIR / game / condition / f"n_{n}" / f"set_{set_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already generated
    pos_dir = out_dir / "pos"
    neg_dir = out_dir / "neg"
    if pos_dir.exists() and neg_dir.exists():
        pos_count = len(list(pos_dir.glob("*.jsonl")))
        neg_count = len(list(neg_dir.glob("*.jsonl")))
        if pos_count >= n and neg_count >= n:
            print(f"  {condition} n={n} set={set_id}: already exists ({pos_count} pos, {neg_count} neg)")
            return True

    # Build command
    cmd = ["python", script_name, "--gen", str(n)]
    if condition == "random_pos":
        cmd.append("--random-placements")
    elif condition == "random_size":
        cmd.append("--random-size")

    # Run trace generation
    try:
        result = subprocess.run(
            cmd, cwd=str(GAMES_DIR),
            capture_output=True, text=True, timeout=120
        )
    except subprocess.TimeoutExpired:
        print(f"  {condition} n={n} set={set_id}: TIMEOUT")
        return False

    # Find the generated logs directory
    logs_base = GAMES_DIR / "Logs" / log_subdir
    if not logs_base.exists():
        print(f"  {condition} n={n} set={set_id}: No logs directory created")
        print(f"    stdout: {result.stdout[:200] if result.stdout else 'empty'}")
        print(f"    stderr: {result.stderr[:200] if result.stderr else 'empty'}")
        return False

    # Get the most recent log directory
    log_dirs = sorted(logs_base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not log_dirs:
        print(f"  {condition} n={n} set={set_id}: No log directories found")
        return False

    latest_log = log_dirs[0]

    # Move pos and neg directories
    src_pos = latest_log / "pos"
    src_neg = latest_log / "neg"

    if src_pos.exists() and src_neg.exists():
        # Remove existing if present
        if pos_dir.exists():
            shutil.rmtree(pos_dir)
        if neg_dir.exists():
            shutil.rmtree(neg_dir)

        # Move
        shutil.move(str(src_pos), str(pos_dir))
        shutil.move(str(src_neg), str(neg_dir))

        # Cleanup empty log dir
        try:
            latest_log.rmdir()
        except:
            pass

        pos_count = len(list(pos_dir.glob("*.jsonl")))
        neg_count = len(list(neg_dir.glob("*.jsonl")))
        print(f"  {condition} n={n} set={set_id}: {pos_count} pos, {neg_count} neg")
        return True
    else:
        print(f"  {condition} n={n} set={set_id}: Missing pos/neg in {latest_log}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate trace sets for TSL_f evaluation")
    parser.add_argument("game", choices=list(GAMES.keys()),
                        help="Game to generate traces for")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS,
                        default=CONDITIONS,
                        help="Conditions to generate (default: all)")
    parser.add_argument("--counts", nargs="+", type=int,
                        default=TRACE_COUNTS,
                        help="Trace counts to generate (default: 5 10 15 20 25)")
    parser.add_argument("--sets", type=int, default=NUM_SETS,
                        help="Number of independent sets per count (default: 5)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"TSL_f Trace Generation - {args.game}")
    print("=" * 60)
    print(f"Game: {args.game}")
    print(f"Conditions: {args.conditions}")
    print(f"Trace counts: {args.counts}")
    print(f"Sets per count: {args.sets}")
    print()

    for condition in args.conditions:
        print(f"\n=== Condition: {condition} ===")
        for n in args.counts:
            for set_id in range(1, args.sets + 1):
                generate_traces(args.game, condition, n, set_id)

    print("\n" + "=" * 60)
    print("Trace generation complete!")
    print(f"Output directory: {EVAL_DIR / args.game}")
    print("=" * 60)


if __name__ == "__main__":
    main()
