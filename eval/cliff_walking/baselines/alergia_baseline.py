#!/usr/bin/env python3
"""
Alergia Baseline for CliffWalking

Learns a Stochastic Mealy Machine (SMM) from demonstration traces using Alergia.
The SMM maps state observations to action probability distributions.

Unlike RPNI, Alergia naturally handles non-deterministic data where the same
observation can lead to different actions across traces.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from aalpy.learning_algs import run_Alergia
from aalpy.base import Automaton


# ============== State Encoding ==============

def encode_observation(x: int, y: int, goal_x: int, goal_y: int,
                       cliff_x_min: int, cliff_x_max: int, cliff_height: int,
                       width: int = None, height: int = None) -> str:
    """
    Encode game state as a discrete observation symbol.

    Uses relative position to goal and danger indicators for adjacent cells.
    """
    # Relative direction to goal (sign only for generalization)
    dx = np.sign(goal_x - x)  # -1, 0, 1
    dy = np.sign(goal_y - y)  # -1, 0, 1

    # Check if at goal
    at_goal = (x == goal_x and y == goal_y)

    if at_goal:
        return "GOAL"

    # Check adjacent cells for cliff
    # Cliff occupies: cliff_x_min <= x <= cliff_x_max AND y < cliff_height
    def is_cliff(cx, cy):
        return cliff_x_min <= cx <= cliff_x_max and cy < cliff_height

    danger_left = is_cliff(x - 1, y)
    danger_right = is_cliff(x + 1, y)
    danger_up = is_cliff(x, y + 1)  # y increases upward in logged coords
    danger_down = is_cliff(x, y - 1)  # y decreases downward

    # Direction encoding
    dir_x = {-1: 'W', 0: '_', 1: 'E'}[dx]  # West/East
    dir_y = {-1: 'S', 0: '_', 1: 'N'}[dy]  # South/North (y=0 is bottom, so N is up)

    # Danger encoding (which adjacent cells have cliff)
    dangers = []
    if danger_left: dangers.append('l')
    if danger_right: dangers.append('r')
    if danger_up: dangers.append('u')
    if danger_down: dangers.append('d')
    danger_str = ''.join(dangers) if dangers else '0'

    return f"{dir_x}{dir_y}_{danger_str}"


def decode_action(curr_x: int, curr_y: int, next_x: int, next_y: int,
                  var_moves: bool = False) -> str:
    """Infer action from position change."""
    delta_x = next_x - curr_x
    delta_y = next_y - curr_y

    if var_moves:
        # Variant movements (logged coords, y=0 is bottom)
        if delta_y == 2 and delta_x == 0:
            return "UP"
        elif delta_y == -1 and delta_x == 0:
            return "DOWN"
        elif delta_x == -1 and delta_y == 0:
            return "LEFT"
        elif delta_x == curr_x + 1 and delta_y == 0:
            return "RIGHT"  # x_new = (x * 2) + 1
        else:
            return "STAY"
    else:
        # Standard movements (logged coords, y=0 is bottom)
        if delta_x == 1 and delta_y == 0:
            return "RIGHT"
        elif delta_x == -1 and delta_y == 0:
            return "LEFT"
        elif delta_x == 0 and delta_y == 1:
            return "UP"
        elif delta_x == 0 and delta_y == -1:
            return "DOWN"
        else:
            return "STAY"


def action_to_delta(action: str, curr_x: int, curr_y: int, width: int, height: int,
                    var_moves: bool = False) -> Tuple[int, int]:
    """Convert action string to new position."""
    if var_moves:
        if action == "LEFT":
            new_x, new_y = curr_x - 1, curr_y
        elif action == "RIGHT":
            new_x, new_y = (curr_x * 2) + 1, curr_y
        elif action == "UP":
            new_x, new_y = curr_x, curr_y + 2
        elif action == "DOWN":
            new_x, new_y = curr_x, curr_y - 1
        else:  # STAY
            return curr_x, curr_y
        # Var moves: stay in place if out of bounds
        if 0 <= new_x < width and 0 <= new_y < height:
            return new_x, new_y
        else:
            return curr_x, curr_y
    else:
        # Standard movements with clamping
        if action == "LEFT":
            return max(0, curr_x - 1), curr_y
        elif action == "RIGHT":
            return min(width - 1, curr_x + 1), curr_y
        elif action == "UP":
            return curr_x, min(height - 1, curr_y + 1)
        elif action == "DOWN":
            return curr_x, max(0, curr_y - 1)
        else:  # STAY
            return curr_x, curr_y


# ============== Trace Loading & Conversion ==============

def load_traces_from_dir(trace_dir: Path, include_neg: bool = False) -> Tuple[List[List[dict]], List[List[dict]]]:
    """
    Load traces from directory.

    Returns:
        pos_traces: List of positive traces (each trace is list of state dicts)
        neg_traces: List of negative traces
    """
    pos_traces = []
    neg_traces = []

    pos_dir = trace_dir / "pos"
    neg_dir = trace_dir / "neg"

    if pos_dir.exists():
        for trace_file in sorted(pos_dir.glob("*.jsonl")):
            states = []
            with open(trace_file) as f:
                for line in f:
                    states.append(json.loads(line))
            if states:
                pos_traces.append(states)

    if include_neg and neg_dir.exists():
        for trace_file in sorted(neg_dir.glob("*.jsonl")):
            states = []
            with open(trace_file) as f:
                for line in f:
                    states.append(json.loads(line))
            if states:
                neg_traces.append(states)

    return pos_traces, neg_traces


def trace_to_smm_sequence(trace: List[dict], var_moves: bool = False) -> List[Tuple[str, str]]:
    """
    Convert a trace to SMM format: [(obs1, act1), (obs2, act2), ...]

    For Alergia SMM learning: list of (input, output) tuples.
    """
    sequence = []

    for i in range(len(trace) - 1):
        curr = trace[i]
        next_state = trace[i + 1]

        # Extract positions
        x = curr["x"]
        y = curr["y"]
        goal_x = curr["goalX"]
        goal_y = curr["goalY"]
        cliff_x_min = curr["cliffXMin"]
        cliff_x_max = curr["cliffXMax"]
        cliff_height = curr["cliffHeight"]

        # Encode observation
        obs = encode_observation(x, y, goal_x, goal_y, cliff_x_min, cliff_x_max, cliff_height)

        # Decode action
        action = decode_action(x, y, next_state["x"], next_state["y"], var_moves)

        sequence.append((obs, action))

    return sequence


def convert_to_alergia_format(traces: List[List[dict]], var_moves: bool = False) -> List[List[Tuple[str, str]]]:
    """
    Convert game traces to Alergia SMM format.

    Alergia SMM expects: [[(I, O), (I, O), ...], [(I, O), ...], ...]
    Where I = input (observation), O = output (action)
    """
    result = []
    for trace in traces:
        seq = trace_to_smm_sequence(trace, var_moves)
        if seq:
            result.append(seq)
    return result


# ============== Training ==============

def train(trace_dir: Path, eps: float = 0.05, var_moves: bool = False,
          print_info: bool = False) -> Tuple[Optional[Automaton], dict]:
    """
    Train a Stochastic Mealy Machine from traces using Alergia.

    Args:
        trace_dir: Directory containing pos/ subfolder with .jsonl traces
        eps: Epsilon for Hoeffding compatibility test (lower = more merging)
        var_moves: Whether traces use variant movement functions
        print_info: Print learning progress

    Returns:
        model: Learned SMM (or None if learning fails)
        metadata: Training metadata
    """
    # Load positive traces only (we learn from demonstrations)
    pos_traces, _ = load_traces_from_dir(trace_dir, include_neg=False)

    if not pos_traces:
        raise ValueError(f"No positive traces found in {trace_dir}/pos/")

    # Convert traces to SMM format
    smm_data = convert_to_alergia_format(pos_traces, var_moves)

    if not smm_data:
        raise ValueError("No valid sequences extracted from traces")

    # Collect unique inputs and outputs for metadata
    all_inputs = set()
    all_outputs = set()
    for seq in smm_data:
        for obs, action in seq:
            all_inputs.add(obs)
            all_outputs.add(action)

    metadata = {
        "num_traces": len(pos_traces),
        "num_sequences": len(smm_data),
        "unique_observations": sorted(all_inputs),
        "unique_actions": sorted(all_outputs),
        "avg_trace_length": np.mean([len(seq) for seq in smm_data]),
        "eps": eps,
        "var_moves": var_moves
    }

    # Run Alergia to learn SMM
    try:
        model = run_Alergia(
            data=smm_data,
            automaton_type='smm',
            eps=eps,
            print_info=print_info
        )

        if model is not None:
            metadata["num_states"] = len(model.states)

        return model, metadata

    except Exception as e:
        metadata["error"] = str(e)
        return None, metadata


# ============== Evaluation ==============

def is_cliff(x: int, y: int, cliff_x_min: int, cliff_x_max: int,
             cliff_height: int, height: int) -> bool:
    """
    Check if position is on the cliff.

    cliff_height is 1-indexed: cliff occupies y < cliff_height
    """
    return cliff_x_min <= x <= cliff_x_max and y < cliff_height


def evaluate(model: Automaton, test_configs: List[dict],
             max_steps: int = 100, use_sampling: bool = False,
             var_moves: bool = False) -> Dict:
    """
    Evaluate SMM on test configurations.

    At each step:
    1. Encode current state as observation
    2. Query SMM for action (most likely or sampled)
    3. Execute action
    4. Repeat until goal, cliff, or timeout

    Args:
        model: Trained Stochastic Mealy Machine
        test_configs: List of config dicts with width, height, goalX, goalY, cliff params
        max_steps: Maximum steps per episode
        use_sampling: If True, sample from action distribution; else use most likely
        var_moves: Whether to use variant movement functions

    Returns:
        Dict with successes, total, success_rate, details
    """
    successes = 0
    steps_list = []
    details = []

    for config in test_configs:
        width = config["width"]
        height = config["height"]
        goal_x = config["goalX"]
        goal_y = config["goalY"]
        cliff_x_min = config["cliffXMin"]
        cliff_x_max = config["cliffXMax"]
        cliff_height = config["cliffHeight"]

        # Reset SMM to initial state
        model.reset_to_initial()

        # Run episode - start at (0, 0) which is bottom-left in logged coords
        x, y = 0, 0
        trajectory = [(x, y)]
        result_reason = "timeout"
        final_step = max_steps

        for step in range(max_steps):
            # Check terminal conditions
            if x == goal_x and y == goal_y:
                result_reason = "goal"
                final_step = step
                successes += 1
                steps_list.append(step)
                break

            if is_cliff(x, y, cliff_x_min, cliff_x_max, cliff_height, height):
                result_reason = "cliff"
                final_step = step
                break

            # Encode observation
            obs = encode_observation(x, y, goal_x, goal_y,
                                    cliff_x_min, cliff_x_max, cliff_height)

            # Query SMM for action
            action = get_smm_action(model, obs, use_sampling)

            if action is None:
                # No transition for this observation - pick random action
                action = np.random.choice(["LEFT", "RIGHT", "UP", "DOWN"])

            # Execute action
            new_x, new_y = action_to_delta(action, x, y, width, height, var_moves)
            x, y = new_x, new_y
            trajectory.append((x, y))

        details.append({
            "config": config.get("name", "unknown"),
            "result": "PASS" if result_reason == "goal" else "FAIL",
            "reason": result_reason,
            "steps": final_step,
            "trajectory": trajectory[:15]
        })

    return {
        "successes": successes,
        "total": len(test_configs),
        "success_rate": successes / len(test_configs) if test_configs else 0,
        "steps": steps_list,
        "avg_steps": np.mean(steps_list) if steps_list else None,
        "details": details
    }


def get_smm_action(model: Automaton, observation: str, use_sampling: bool = False) -> Optional[str]:
    """
    Get action from SMM given an observation.

    For SMM, transitions are stored as:
    state.transitions = {input: [(target_state, output, prob), ...]}
    """
    try:
        current_state = model.current_state

        # Get transitions for this observation
        if observation not in current_state.transitions:
            return None

        transitions = current_state.transitions[observation]
        if not transitions:
            return None

        if use_sampling:
            # Sample from distribution
            probs = [t[2] for t in transitions]
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
                idx = np.random.choice(len(transitions), p=probs)
                target_state, action, _ = transitions[idx]
                model.current_state = target_state
                return action
        else:
            # Take most likely
            best = max(transitions, key=lambda t: t[2])
            target_state, action, _ = best
            model.current_state = target_state
            return action

    except Exception:
        return None


# ============== Convenience Function ==============

def train_and_evaluate(trace_dir: Path, test_configs: List[dict],
                       eps: float = 2.0, use_sampling: bool = False,
                       var_moves: bool = False) -> Dict:
    """
    Train SMM and evaluate on test configs.

    Returns combined results dict.
    """
    import time

    start = time.time()
    model, meta = train(trace_dir, eps=eps, var_moves=var_moves)
    train_time = time.time() - start

    if model is None:
        return {
            "method": "alergia",
            "train_time": train_time,
            "error": meta.get("error", "Alergia learning failed"),
            "successes": 0,
            "total": len(test_configs),
            "success_rate": 0.0,
            **meta
        }

    eval_result = evaluate(model, test_configs, use_sampling=use_sampling, var_moves=var_moves)

    return {
        "method": "alergia",
        "train_time": train_time,
        **meta,
        **eval_result
    }


# ============== CLI ==============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Alergia baseline for CliffWalking")
    parser.add_argument("trace_dir", type=Path, help="Directory with pos/ traces")
    parser.add_argument("--test-configs", type=Path, help="JSON file with test configs")
    parser.add_argument("--eps", type=float, default=2.0, help="Epsilon for Hoeffding test")
    parser.add_argument("--sample", action="store_true", help="Sample from action distribution")
    parser.add_argument("--var-moves", action="store_true", help="Use variant movement functions")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Train
    print(f"Training Alergia SMM from {args.trace_dir}...")
    model, meta = train(args.trace_dir, eps=args.eps, var_moves=args.var_moves,
                        print_info=args.verbose)

    print(f"Training metadata: {json.dumps(meta, indent=2, default=str)}")

    if model is None:
        print("Alergia learning failed!")
        exit(1)

    print(f"Learned SMM with {len(model.states)} states")

    # Evaluate if test configs provided
    if args.test_configs and args.test_configs.exists():
        with open(args.test_configs) as f:
            test_configs = json.load(f)

        print(f"\nEvaluating on {len(test_configs)} test configs...")
        result = evaluate(model, test_configs, use_sampling=args.sample, var_moves=args.var_moves)
        print(f"Success rate: {result['success_rate']:.2%} ({result['successes']}/{result['total']})")
