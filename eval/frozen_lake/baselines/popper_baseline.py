#!/usr/bin/env python3
"""
Popper ILP Baseline for FrozenLake

Inductive Logic Programming baseline using Popper.
Uses FAIR background knowledge - equivalent to what TSL_f receives:
- Position extraction (raw numeric values)
- Grid bounds (0 to max_coord)
- Comparison operators

Popper must discover movement patterns (+1/-1), just like TSL_f discovers them via SyGuS.

Usage:
    # Standalone
    python popper_baseline.py --traces path/to/traces --num-tests 100

    # Or via train_and_evaluate() for integration with run_full_eval.py
"""

import json
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import from renamed file - handle both module and standalone execution
try:
    from . import popper_convert
except ImportError:
    import popper_convert


# ============== Popper Execution ==============

def run_popper(problem_dir: Path, timeout: int = 300) -> Tuple[Optional[str], float]:
    """Run Popper on the problem directory using the library API."""
    start = time.time()
    try:
        from popper.util import Settings
        from popper.loop import learn_solution

        settings = Settings(
            kbpath=str(problem_dir),
            timeout=timeout,
            quiet=True,
            noisy=False,
        )

        program, best_score, stats = learn_solution(settings)
        elapsed = time.time() - start

        if program:
            lines = []
            for head, body in program:
                head_pred = head.predicate
                head_args = ','.join(f'V{a}' for a in head.arguments)
                head_str = f"{head_pred}({head_args})"

                body_parts = []
                for lit in body:
                    lit_pred = lit.predicate
                    lit_args = ','.join(f'V{a}' for a in lit.arguments)
                    body_parts.append(f"{lit_pred}({lit_args})")

                if body_parts:
                    lines.append(f"{head_str}:- {', '.join(body_parts)}.")
                else:
                    lines.append(f"{head_str}.")

            return '\n'.join(lines), elapsed
        else:
            return None, elapsed

    except Exception as e:
        print(f"    Popper error: {e}")
        return None, time.time() - start


def parse_popper_rules(program: str) -> List[Dict]:
    """Parse Popper output into structured rules.

    Handles flat format: go_right(V0,V1,V2,V3):- my_lt(V0,V2).
    Where V0=PX, V1=PY, V2=GX, V3=GY
    """
    rules = []
    # Match: go_action(args):- body.
    pattern = r'go_(\w+)\(([^)]+)\)\s*:-\s*(.+?)\.'

    for match in re.finditer(pattern, program, re.MULTILINE):
        action = match.group(1)
        head_args = [a.strip() for a in match.group(2).split(',')]
        body = match.group(3)

        conditions = [c.strip() for c in body.split(',') if c.strip()]

        rules.append({
            'action': action,
            'head_args': head_args,
            'conditions': conditions,
            'raw': match.group(0)
        })

    return rules


# ============== Policy Execution ==============

class PopperPolicy:
    """Execute Popper-learned rules as a policy.

    For flat format with holes, rules look like:
    go_right(V0,V1,V2,V3,V4,V5,V6,V7,V8,V9):- my_lt(V0,V2).

    Where V0=PX, V1=PY, V2=GX, V3=GY, V4=H0X, V5=H0Y, V6=H1X, V7=H1Y, V8=H2X, V9=H2Y
    """

    def __init__(self, rules: List[Dict], max_coord: int = 4):
        self.rules = rules
        self.max_coord = max_coord

    def evaluate_rule(self, rule: Dict, state: Dict) -> bool:
        """Evaluate if a rule matches the given state."""
        px, py = state['player']
        gx, gy = state['goal']

        # Get hole positions (default to -1 if not present)
        h0 = state.get('hole0', [-1, -1])
        h1 = state.get('hole1', [-1, -1])
        h2 = state.get('hole2', [-1, -1])
        h0x, h0y = h0[0], h0[1]
        h1x, h1y = h1[0], h1[1]
        h2x, h2y = h2[0], h2[1]
        holes = [(h0x, h0y), (h1x, h1y), (h2x, h2y)]

        # Map head variables to actual values
        head_args = rule.get('head_args', [])

        if len(head_args) == 1:
            # Stateful format: go_action(State) - evaluate using state directly
            return self._evaluate_stateful(rule, state, px, py, gx, gy, holes)
        elif len(head_args) >= 10:
            # Full format with holes: go_action(PX, PY, GX, GY, H0X, H0Y, H1X, H1Y, H2X, H2Y)
            var_map = {
                head_args[0]: px,
                head_args[1]: py,
                head_args[2]: gx,
                head_args[3]: gy,
                head_args[4]: h0x,
                head_args[5]: h0y,
                head_args[6]: h1x,
                head_args[7]: h1y,
                head_args[8]: h2x,
                head_args[9]: h2y,
            }
        elif len(head_args) >= 4:
            # Legacy format: go_action(PX, PY, GX, GY)
            var_map = {
                head_args[0]: px,
                head_args[1]: py,
                head_args[2]: gx,
                head_args[3]: gy,
            }
        else:
            return self._evaluate_old_format(rule, state)

        # Evaluate all conditions
        for cond in rule['conditions']:
            if not self._eval_condition(cond, var_map):
                return False
        return True

    def _evaluate_stateful(self, rule: Dict, state: Dict, px: int, py: int,
                           gx: int, gy: int, holes: List[Tuple[int, int]]) -> bool:
        """Evaluate stateful rule where head is go_action(State).

        Stateful rules use predicates like goal_same_y(V0), goal_right(V0), etc.
        The state V0 is implicit and we evaluate predicates directly on the state.
        """
        # For stateful rules, we need to evaluate predicates that take the state
        # and possibly extract values from it
        var_map = {}  # Will store extracted values like V1, V2, etc.

        for cond in rule['conditions']:
            if not self._eval_stateful_condition(cond, px, py, gx, gy, holes, var_map):
                return False
        return True

    def _eval_stateful_condition(self, cond: str, px: int, py: int, gx: int, gy: int,
                                  holes: List[Tuple[int, int]], var_map: Dict) -> bool:
        """Evaluate a stateful condition."""
        match = re.match(r'(\w+)\(([^)]*)\)', cond)
        if not match:
            return False

        pred = match.group(1)
        args = [a.strip() for a in match.group(2).split(',') if a.strip()]

        # Derived predicates that take just the state (V0)
        if pred == 'goal_same_y':
            return py == gy
        elif pred == 'goal_same_x':
            return px == gx
        elif pred == 'goal_right':
            return px < gx
        elif pred == 'goal_left':
            return px > gx
        elif pred == 'goal_down':
            return py < gy
        elif pred == 'goal_up':
            return py > gy
        elif pred == 'would_hit_right':
            return any(px + 1 == hx and py == hy for hx, hy in holes if hx >= 0)
        elif pred == 'would_hit_left':
            return any(px - 1 == hx and py == hy for hx, hy in holes if hx >= 0)
        elif pred == 'would_hit_down':
            return any(px == hx and py + 1 == hy for hx, hy in holes if hx >= 0)
        elif pred == 'would_hit_up':
            return any(px == hx and py - 1 == hy for hx, hy in holes if hx >= 0)

        # SAFE direction predicates (goal direction AND not blocked)
        elif pred == 'safe_right':
            goal_right = px < gx
            would_hit = any(px + 1 == hx and py == hy for hx, hy in holes if hx >= 0)
            return goal_right and not would_hit
        elif pred == 'safe_left':
            goal_left = px > gx
            would_hit = any(px - 1 == hx and py == hy for hx, hy in holes if hx >= 0)
            return goal_left and not would_hit
        elif pred == 'safe_down':
            goal_down = py < gy
            would_hit = any(px == hx and py + 1 == hy for hx, hy in holes if hx >= 0)
            return goal_down and not would_hit
        elif pred == 'safe_up':
            goal_up = py > gy
            would_hit = any(px == hx and py - 1 == hy for hx, hy in holes if hx >= 0)
            return goal_up and not would_hit

        # Not blocked predicates
        elif pred == 'not_blocked_right':
            return not any(px + 1 == hx and py == hy for hx, hy in holes if hx >= 0)
        elif pred == 'not_blocked_left':
            return not any(px - 1 == hx and py == hy for hx, hy in holes if hx >= 0)
        elif pred == 'not_blocked_down':
            return not any(px == hx and py + 1 == hy for hx, hy in holes if hx >= 0)
        elif pred == 'not_blocked_up':
            return not any(px == hx and py - 1 == hy for hx, hy in holes if hx >= 0)

        # Predicates that extract values: player_x(S, X) binds X to px
        if pred == 'player_x' and len(args) == 2:
            var_map[args[1]] = px
            return True
        elif pred == 'player_y' and len(args) == 2:
            var_map[args[1]] = py
            return True
        elif pred == 'goal_x' and len(args) == 2:
            var_map[args[1]] = gx
            return True
        elif pred == 'goal_y' and len(args) == 2:
            var_map[args[1]] = gy
            return True

        # is_hole(S, X, Y) - check if (X, Y) is a hole
        if pred == 'is_hole' and len(args) == 3:
            x_arg, y_arg = args[1], args[2]
            x_val = var_map.get(x_arg)
            y_val = var_map.get(y_arg)
            if x_val is None or y_val is None:
                # If values not bound yet, we can't evaluate
                # Try to bind if the hole exists
                for hx, hy in holes:
                    if hx >= 0:
                        if x_val is None and y_val is not None and hy == y_val:
                            var_map[x_arg] = hx
                            return True
                        elif y_val is None and x_val is not None and hx == x_val:
                            var_map[y_arg] = hy
                            return True
                        elif x_val is None and y_val is None:
                            var_map[x_arg] = hx
                            var_map[y_arg] = hy
                            return True
                return False
            return any(hx == x_val and hy == y_val for hx, hy in holes if hx >= 0)

        # Comparison predicates using var_map
        vals = []
        for arg in args:
            if arg in var_map:
                vals.append(var_map[arg])
            elif arg.lstrip('-').isdigit():
                vals.append(int(arg))
            else:
                return False

        if pred == 'my_lt' and len(vals) == 2:
            return vals[0] < vals[1]
        elif pred == 'my_gt' and len(vals) == 2:
            return vals[0] > vals[1]
        elif pred == 'my_eq' and len(vals) == 2:
            return vals[0] == vals[1]
        elif pred == 'inc' and len(vals) == 2:
            return vals[1] == vals[0] + 1
        elif pred == 'dec' and len(vals) == 2:
            return vals[1] == vals[0] - 1

        return False

    def _eval_condition(self, cond: str, var_map: Dict) -> bool:
        """Evaluate a single condition given variable bindings."""
        match = re.match(r'(\w+)\(([^)]*)\)', cond)
        if not match:
            return False

        pred = match.group(1)
        args = [a.strip() for a in match.group(2).split(',')]

        # Get actual values for the arguments
        vals = []
        for arg in args:
            if arg in var_map:
                vals.append(var_map[arg])
            elif arg.lstrip('-').isdigit():
                vals.append(int(arg))
            else:
                return False

        if pred == 'my_lt' and len(vals) == 2:
            return vals[0] < vals[1]
        elif pred == 'my_gt' and len(vals) == 2:
            return vals[0] > vals[1]
        elif pred == 'my_eq' and len(vals) == 2:
            return vals[0] == vals[1]
        elif pred == 'at_min' and len(vals) == 1:
            return vals[0] == 0
        elif pred == 'at_max' and len(vals) == 1:
            return vals[0] == self.max_coord
        elif pred == 'in_bounds' and len(vals) == 1:
            return 0 <= vals[0] <= self.max_coord
        elif pred == 'inc' and len(vals) == 2:
            return vals[1] == vals[0] + 1
        elif pred == 'dec' and len(vals) == 2:
            return vals[1] == vals[0] - 1
        elif pred == 'id' and len(vals) == 2:
            return vals[0] == vals[1]

        return False

    def _evaluate_old_format(self, rule: Dict, state: Dict) -> bool:
        """Fallback for old state-based format."""
        conditions = rule['conditions']
        px, py = state['player']
        gx, gy = state['goal']

        has_lt = any('my_lt' in c for c in conditions)
        has_gt = any('my_gt' in c for c in conditions)

        action = rule['action']
        if action == 'right' and has_lt:
            return px < gx
        elif action == 'left' and has_gt:
            return px > gx
        elif action == 'down' and has_lt:
            return py < gy
        elif action == 'up' and has_gt:
            return py > gy
        return False

    def get_action(self, state: Dict) -> Optional[str]:
        """Get action for state. Returns first matching rule's action."""
        for rule in self.rules:
            if self.evaluate_rule(rule, state):
                return rule['action']
        return None


# ============== Training ==============

def train_popper(trace_dir: Path, timeout_per_action: int = 120,
                 with_functions: bool = False, use_stateful: bool = False,
                 minimal_bias: bool = False) -> Tuple[Optional[PopperPolicy], Dict]:
    """
    Train Popper on traces from a directory.

    Args:
        trace_dir: Directory containing pos/ and neg/ subdirs with .jsonl traces
        timeout_per_action: Timeout in seconds for learning each action
        with_functions: If True, provide movement functions (+1/-1) as background knowledge.
                       If False, Popper must discover movement patterns from comparisons only.
        use_stateful: If True, use stateful representation where goal/hole positions are
                     encoded as ground facts in background knowledge (1 head var instead of 10).
                     This includes hole information and is much more tractable.
        minimal_bias: If True (and use_stateful=True), only allow safe_X predicates in bias.
                     This forces Popper to learn clean rules like go_right(S) :- safe_right(S).

    Returns:
        (policy, metadata) where policy is None if training failed
    """
    out_dir = Path(tempfile.mkdtemp(prefix='popper_'))

    # Load traces
    pos_traces, neg_traces = popper_convert.load_traces(trace_dir)
    if not pos_traces:
        return None, {"error": "No positive traces found"}

    # Extract examples - use stateful or flat format
    if use_stateful:
        pos_examples, neg_examples = popper_convert.extract_examples_stateful(pos_traces, neg_traces)
    else:
        pos_examples, neg_examples = popper_convert.extract_examples_flat(pos_traces, neg_traces)

    # Learn rules for each action
    all_rules = []
    total_elapsed = 0
    action_results = {}

    for action in ['right', 'left', 'up', 'down']:
        action_dir = out_dir / f'action_{action}'
        action_dir.mkdir(parents=True, exist_ok=True)

        if use_stateful:
            with open(action_dir / 'exs.pl', 'w') as f:
                f.write(popper_convert.generate_exs_pl_stateful(pos_examples, neg_examples, action))
            with open(action_dir / 'bk.pl', 'w') as f:
                f.write(popper_convert.generate_bk_pl_stateful(pos_examples, neg_examples))
            with open(action_dir / 'bias.pl', 'w') as f:
                f.write(popper_convert.generate_bias_pl_stateful(action, minimal=minimal_bias))
        else:
            with open(action_dir / 'exs.pl', 'w') as f:
                f.write(popper_convert.generate_exs_pl_flat(pos_examples, neg_examples, action))
            with open(action_dir / 'bk.pl', 'w') as f:
                f.write(popper_convert.generate_bk_pl_flat(with_functions=with_functions))
            with open(action_dir / 'bias.pl', 'w') as f:
                f.write(popper_convert.generate_bias_pl_flat(action, with_functions=with_functions))

        program, elapsed = run_popper(action_dir, timeout=timeout_per_action)
        total_elapsed += elapsed

        if program:
            rules = parse_popper_rules(program)
            all_rules.extend(rules)
            action_results[action] = {
                'success': True,
                'num_rules': len(rules),
                'rules': [r['raw'] for r in rules],
                'time': elapsed
            }
        else:
            action_results[action] = {
                'success': False,
                'num_rules': 0,
                'rules': [],
                'time': elapsed
            }

    metadata = {
        'num_pos_traces': len(pos_traces),
        'num_neg_traces': len(neg_traces),
        'num_pos_examples': len(pos_examples),
        'num_neg_examples': len(neg_examples),
        'num_rules': len(all_rules),
        'train_time': total_elapsed,
        'action_results': action_results,
        'rules': [r['raw'] for r in all_rules]
    }

    if all_rules:
        return PopperPolicy(all_rules), metadata
    else:
        return None, metadata


# ============== Evaluation ==============

def evaluate_on_configs(policy: PopperPolicy, test_configs: List[Dict],
                        max_steps: int = 100) -> Dict:
    """
    Evaluate policy on test configurations (same format as run_full_eval.py).

    Args:
        policy: Trained PopperPolicy
        test_configs: List of config dicts with keys:
            - grid_size: int
            - start_pos: {x, y}
            - goal: {x, y}
            - holes: [{x, y}, ...]

    Returns:
        Dict with successes, total, steps, details
    """
    import random

    successes = 0
    steps_list = []
    details = []

    for config in test_configs:
        # Handle both config formats
        size = config.get('grid_size') or config.get('size', 4)

        # Handle start position
        start = config.get('start_pos')
        if start is None:
            start = {'x': 0, 'y': 0}  # Default start
        elif isinstance(start, list):
            start = {'x': start[0], 'y': start[1]}

        # Handle goal position
        goal = config['goal']
        if isinstance(goal, list):
            goal = {'x': goal[0], 'y': goal[1]}

        # Handle holes
        holes_raw = config['holes']
        holes = []
        for h in holes_raw:
            if isinstance(h, list):
                holes.append((h[0], h[1]))
            else:
                holes.append((h['x'], h['y']))

        # Initialize state
        state = {
            'player': [start['x'], start['y']],
            'goal': [goal['x'], goal['y']],
        }
        for i, (hx, hy) in enumerate(holes):
            state[f'hole{i}'] = [hx, hy]

        trajectory = [(start['x'], start['y'])]
        result = None
        steps = 0

        for step in range(max_steps):
            px, py = state['player']
            gx, gy = state['goal']

            # Check win
            if px == gx and py == gy:
                result = 'PASS'
                successes += 1
                steps_list.append(step)
                break

            # Check holes
            on_hole = False
            for hx, hy in holes:
                if px == hx and py == hy:
                    on_hole = True
                    break

            if on_hole:
                result = 'FAIL'
                break

            # Get action from policy
            action = policy.get_action(state)
            if action is None:
                # Random fallback if no rule matches
                action = random.choice(['up', 'down', 'left', 'right'])

            # Execute action
            if action == 'right':
                state['player'][0] = min(px + 1, size - 1)
            elif action == 'left':
                state['player'][0] = max(px - 1, 0)
            elif action == 'down':
                state['player'][1] = min(py + 1, size - 1)
            elif action == 'up':
                state['player'][1] = max(py - 1, 0)

            trajectory.append(tuple(state['player']))
            steps += 1

        if result is None:
            result = 'TIMEOUT'

        details.append({
            'config': config.get('name', 'unknown'),
            'result': result,
            'steps': steps,
            'trajectory': trajectory,
            'reason': 'reached_goal' if result == 'PASS' else ('hit_hole' if result == 'FAIL' else 'max_steps')
        })

    return {
        'successes': successes,
        'total': len(test_configs),
        'steps': steps_list,
        'avg_steps': sum(steps_list) / len(steps_list) if steps_list else None,
        'details': details
    }


def train_and_evaluate(trace_dir: Path, test_configs: List[Dict],
                       timeout_per_action: int = 120,
                       with_functions: bool = False,
                       use_stateful: bool = False,
                       minimal_bias: bool = False) -> Dict:
    """
    Train Popper on traces and evaluate on test configs.

    This is the main interface for integration with run_full_eval.py.

    Args:
        trace_dir: Directory containing pos/ and neg/ subdirs with .jsonl traces
        test_configs: List of test configuration dicts
        timeout_per_action: Timeout in seconds for learning each action
        with_functions: If True, provide movement functions (+1/-1) as background knowledge
        use_stateful: If True, use stateful representation (state atoms with ground facts
                     for goals/holes - much more tractable, includes safety info)
        minimal_bias: If True (and use_stateful=True), only allow safe_X predicates

    Returns:
        Dict with successes, total, steps, train_time, etc.
    """
    # Train
    policy, train_metadata = train_popper(trace_dir, timeout_per_action,
                                          with_functions=with_functions,
                                          use_stateful=use_stateful,
                                          minimal_bias=minimal_bias)

    if policy is None:
        return {
            'error': train_metadata.get('error', 'Training failed - no rules learned'),
            'successes': 0,
            'total': len(test_configs),
            'train_time': train_metadata.get('train_time', 0),
            'num_rules': 0,
            'with_functions': with_functions,
            'use_stateful': use_stateful
        }

    # Evaluate
    eval_result = evaluate_on_configs(policy, test_configs)

    successes = eval_result['successes']
    total = eval_result['total']
    win_rate = successes / total if total > 0 else 0

    return {
        'successes': successes,
        'total': total,
        'win_rate': win_rate,
        'steps': eval_result['steps'],
        'avg_steps': eval_result['avg_steps'],
        'train_time': train_metadata['train_time'],
        'num_rules': train_metadata['num_rules'],
        'total_rules': train_metadata['num_rules'],
        'rules': train_metadata.get('action_results', {}),
        'details': eval_result['details'],
        'with_functions': with_functions,
        'use_stateful': use_stateful
    }


# ============== CLI ==============

def main():
    import argparse

    # Add parent directory for imports when running standalone
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    parser = argparse.ArgumentParser(description='Run Popper ILP baseline')
    parser.add_argument('--traces', type=str, required=True,
                        help='Path to traces directory')
    parser.add_argument('--num-tests', type=int, default=100,
                        help='Number of test configurations')
    parser.add_argument('--timeout', type=int, default=120,
                        help='Timeout per action in seconds')
    parser.add_argument('--test-config', type=str, default='var_config',
                        choices=['var_config', 'var_size'],
                        help='Test configuration type')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for test config generation')
    parser.add_argument('--with-functions', action='store_true',
                        help='Provide movement functions (+1/-1) as background knowledge')
    args = parser.parse_args()

    # Import test config generator from run_full_eval
    from run_full_eval import generate_test_configs

    traces_dir = Path(args.traces)

    print("=" * 60)
    print("POPPER ILP BASELINE")
    print(f"  with_functions: {args.with_functions}")
    print("=" * 60)

    # Generate test configs using the same function as run_full_eval.py
    print(f"\n[1] Generating {args.num_tests} {args.test_config} test configs (seed={args.seed})...")
    test_configs = generate_test_configs(args.num_tests, args.test_config, seed=args.seed)
    print(f"    Generated {len(test_configs)} configs")

    # Train and evaluate
    print(f"\n[2] Training on traces from {traces_dir}...")
    result = train_and_evaluate(traces_dir, test_configs, timeout_per_action=args.timeout,
                                with_functions=args.with_functions)

    # Print results
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")

    if 'error' in result:
        print(f"  Error: {result['error']}")
    else:
        win_rate = result['successes'] / result['total'] if result['total'] > 0 else 0
        print(f"  Win rate:      {result['successes']}/{result['total']} ({win_rate*100:.1f}%)")
        print(f"  Learning time: {result['train_time']:.1f}s")
        print(f"  Rules learned: {result['num_rules']}")
        if result['rules']:
            print(f"\n  Learned rules:")
            for rule in result['rules']:
                print(f"    {rule}")


if __name__ == '__main__':
    main()
