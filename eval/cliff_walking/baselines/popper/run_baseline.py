#!/usr/bin/env python3
"""
Complete Popper ILP Baseline for CliffWalking.

This script:
1. Generates training data (or uses existing traces)
2. Creates Popper examples for each action predicate
3. Runs Popper to learn rules
4. Evaluates the learned policy

Usage:
    python run_baseline.py --gen 10
    python run_baseline.py --traces path/to/traces
    python run_baseline.py --eval  # Just evaluate previously learned rules
"""

import argparse
import subprocess
import sys
import re
from pathlib import Path


def generate_traces(num_traces, output_dir, random_height=False):
    """Generate CliffWalking traces."""
    print(f"\n=== Generating {num_traces} traces ===")

    games_dir = Path(__file__).parent.parent.parent.parent.parent / 'games'
    game_script = games_dir / 'cliff_walking_game.py'

    cmd = [
        sys.executable, str(game_script),
        '--gen', str(num_traces),
        '--output', str(output_dir)
    ]

    if random_height:
        cmd.append('--random-height')

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(games_dir))

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False

    print(result.stdout)
    return True


def generate_examples(output_dir):
    """Generate Popper examples for each action predicate."""
    print(f"\n=== Generating Popper examples ===")

    script_dir = Path(__file__).parent
    gen_script = script_dir / 'generate_examples.py'

    cmd = [sys.executable, str(gen_script), '--output', str(output_dir)]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False

    print(result.stdout)
    return True


def run_popper(action_dir, timeout=60):
    """Run Popper on a single action predicate."""
    cmd = ['popper-ilp', str(action_dir)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        # Parse the solution
        if 'SOLUTION' in result.stdout:
            # Extract rules from output
            lines = result.stdout.split('\n')
            rules = []
            in_solution = False
            for line in lines:
                if 'SOLUTION' in line:
                    in_solution = True
                    continue
                if in_solution and ':-' in line:
                    rules.append(line.strip())
                if in_solution and '***' in line and rules:
                    break

            return rules
        else:
            return None

    except subprocess.TimeoutExpired:
        return None


def learn_policy(work_dir, timeout=60):
    """Learn policy rules for all actions."""
    print(f"\n=== Learning policy with Popper ===")

    work_dir = Path(work_dir)
    learned_rules = {}

    for action in ['up', 'down', 'right']:
        action_dir = work_dir / f'should_{action}'
        if not action_dir.exists():
            print(f"Warning: {action_dir} not found")
            continue

        print(f"\nLearning should_{action}...")
        rules = run_popper(action_dir, timeout)

        if rules:
            learned_rules[action] = rules
            print(f"  Learned {len(rules)} rule(s):")
            for rule in rules:
                print(f"    {rule}")
        else:
            print(f"  No solution found")

    return learned_rules


def parse_rule(rule_str):
    """Parse a Popper rule into a usable format."""
    # Example: "should_up(V0):- cliff_danger(V0)."
    match = re.match(r'should_(\w+)\(V0\):-\s*(.+)\.', rule_str)
    if match:
        action = match.group(1)
        body = match.group(2)
        # Parse body predicates
        preds = [p.strip().replace('(V0)', '') for p in body.split(',')]
        return action, preds
    return None, None


def evaluate_policy(learned_rules, num_tests=100):
    """Evaluate the learned policy on CliffWalking."""
    print(f"\n=== Evaluating learned policy ===")

    wins = 0
    losses = 0

    for test_num in range(num_tests):
        # Simple CliffWalking simulation
        x, y = 0, 0  # Start position
        goalx, goaly = 11, 0
        cliff_x_min, cliff_x_max = 1, 10
        cliff_height = 1

        max_steps = 50
        for step in range(max_steps):
            # Check win/lose
            if x == goalx and y == goaly:
                wins += 1
                break

            if cliff_x_min <= x <= cliff_x_max and y < cliff_height:
                losses += 1
                break

            # Compute predicates
            preds = {
                'at_goal': x == goalx and y == goaly,
                'at_goal_x': x == goalx,
                'not_at_goal_x': x != goalx,
                'left_of_goal': x < goalx,
                'above_goal_y': y > goaly,
                'above_cliff': y >= cliff_height,
                'cliff_danger': cliff_x_min <= x <= cliff_x_max and y <= cliff_height,
                'safe': not (cliff_x_min <= x <= cliff_x_max and y <= cliff_height),
                'at_start': x == 0 and y == 0,
                'not_at_start': not (x == 0 and y == 0),
            }

            # Determine action using learned rules
            action = None

            for act, rules in learned_rules.items():
                for rule in rules:
                    _, body_preds = parse_rule(rule)
                    if body_preds:
                        # Check if all body predicates are true
                        if all(preds.get(p, False) for p in body_preds):
                            action = act
                            break
                if action:
                    break

            # Execute action
            if action == 'up':
                y = min(3, y + 1)
            elif action == 'down':
                y = max(0, y - 1)
            elif action == 'right':
                x = min(11, x + 1)
            elif action == 'left':
                x = max(0, x - 1)
            else:
                # No action determined, stay in place (will timeout)
                pass
        else:
            # Ran out of steps
            losses += 1

    win_rate = wins / num_tests * 100
    print(f"Results: {wins}/{num_tests} wins ({win_rate:.1f}%)")
    return win_rate


def main():
    parser = argparse.ArgumentParser(description='Popper ILP Baseline for CliffWalking')
    parser.add_argument('--gen', type=int, help='Generate N traces')
    parser.add_argument('--traces', type=str, help='Path to existing traces')
    parser.add_argument('--random-height', action='store_true',
                        help='Use random cliff heights')
    parser.add_argument('--timeout', type=int, default=60,
                        help='Popper timeout per action (default: 60s)')
    parser.add_argument('--eval', action='store_true',
                        help='Only evaluate previously learned rules')
    parser.add_argument('--num-tests', type=int, default=100,
                        help='Number of test episodes (default: 100)')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    work_dir = script_dir / 'work'

    if not args.eval:
        # Generate traces if requested
        if args.gen:
            trace_dir = script_dir / 'traces'
            if not generate_traces(args.gen, trace_dir, args.random_height):
                return 1

        # Generate examples
        if not generate_examples(work_dir):
            return 1

    # Learn policy
    learned_rules = learn_policy(work_dir, args.timeout)

    if not learned_rules:
        print("\nNo rules learned. Cannot evaluate.")
        return 1

    # Print summary
    print("\n=== Learned Policy Summary ===")
    for action, rules in learned_rules.items():
        print(f"\nshould_{action}:")
        for rule in rules:
            print(f"  {rule}")

    # Evaluate
    win_rate = evaluate_policy(learned_rules, args.num_tests)

    print(f"\n=== Final Result ===")
    print(f"Win Rate: {win_rate:.1f}%")

    return 0


if __name__ == '__main__':
    sys.exit(main())
