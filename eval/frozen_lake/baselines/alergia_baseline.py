#!/usr/bin/env python3
"""
Alergia Baseline for FrozenLake

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

def encode_observation(player: List[int], goal: List[int], holes: List[List[int]],
                       grid_size: int = 4) -> str:
    """
    Encode game state as a discrete observation symbol.

    Uses relative position to goal and danger indicators for adjacent cells.
    """
    px, py = player
    gx, gy = goal

    # Relative direction to goal (sign only for generalization)
    dx = np.sign(gx - px)  # -1, 0, 1
    dy = np.sign(gy - py)  # -1, 0, 1

    # Check if at goal
    at_goal = (px == gx and py == gy)

    # Check adjacent cells for holes
    hole_set = {tuple(h) for h in holes}

    danger_left = (px - 1, py) in hole_set
    danger_right = (px + 1, py) in hole_set
    danger_up = (px, py - 1) in hole_set
    danger_down = (px, py + 1) in hole_set

    # Encode as string
    if at_goal:
        return "GOAL"

    # Direction encoding
    dir_x = {-1: 'W', 0: '_', 1: 'E'}[dx]  # West/East
    dir_y = {-1: 'N', 0: '_', 1: 'S'}[dy]  # North/South

    # Danger encoding (which adjacent cells have holes)
    dangers = []
    if danger_left: dangers.append('l')
    if danger_right: dangers.append('r')
    if danger_up: dangers.append('u')
    if danger_down: dangers.append('d')
    danger_str = ''.join(dangers) if dangers else '0'

    return f"{dir_x}{dir_y}_{danger_str}"


def decode_action(curr_player: List[int], next_player: List[int]) -> str:
    """Infer action from player position change."""
    dx = next_player[0] - curr_player[0]
    dy = next_player[1] - curr_player[1]

    if dx == 1 and dy == 0:
        return "RIGHT"
    elif dx == -1 and dy == 0:
        return "LEFT"
    elif dx == 0 and dy == 1:
        return "DOWN"
    elif dx == 0 and dy == -1:
        return "UP"
    else:
        return "STAY"


def action_to_delta(action: str) -> Tuple[int, int]:
    """Convert action string to movement delta."""
    return {
        "LEFT": (-1, 0),
        "RIGHT": (1, 0),
        "UP": (0, -1),
        "DOWN": (0, 1),
        "STAY": (0, 0)
    }.get(action, (0, 0))


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


def trace_to_smm_sequence(trace: List[dict]) -> List[Tuple[str, str]]:
    """
    Convert a trace to SMM format: [(obs1, act1), (obs2, act2), ...]

    For Alergia SMM learning: list of (input, output) tuples.
    """
    sequence = []

    for i in range(len(trace) - 1):
        curr = trace[i]
        next_state = trace[i + 1]

        # Extract positions
        player = curr["player"]
        goal = curr["goal"]
        holes = [curr.get(f"hole{j}") for j in range(5) if curr.get(f"hole{j}") is not None]

        # Encode observation
        obs = encode_observation(player, goal, holes)

        # Decode action
        action = decode_action(player, next_state["player"])

        sequence.append((obs, action))

    return sequence


def convert_to_alergia_format(traces: List[List[dict]]) -> List[List[Tuple[str, str]]]:
    """
    Convert game traces to Alergia SMM format.

    Alergia SMM expects: [[(I, O), (I, O), ...], [(I, O), ...], ...]
    Where I = input (observation), O = output (action)
    """
    result = []
    for trace in traces:
        seq = trace_to_smm_sequence(trace)
        if seq:
            result.append(seq)
    return result


# ============== Training ==============

def train(trace_dir: Path, eps: float = 0.05,
          print_info: bool = False) -> Tuple[Optional[Automaton], dict]:
    """
    Train a Stochastic Mealy Machine from traces using Alergia.

    Args:
        trace_dir: Directory containing pos/ subfolder with .jsonl traces
        eps: Epsilon for Hoeffding compatibility test (lower = more merging)
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
    smm_data = convert_to_alergia_format(pos_traces)

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
        "eps": eps
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

def evaluate(model: Automaton, test_configs: List[dict],
             max_steps: int = 100, use_sampling: bool = False) -> Dict:
    """
    Evaluate SMM on test configurations.

    At each step:
    1. Encode current state as observation
    2. Query SMM for action (most likely or sampled)
    3. Execute action
    4. Repeat until goal, hole, or timeout

    Args:
        model: Trained Stochastic Mealy Machine
        test_configs: List of config dicts with grid_size, goal, holes
        max_steps: Maximum steps per episode
        use_sampling: If True, sample from action distribution; else use most likely

    Returns:
        Dict with successes, total, success_rate, details
    """
    successes = 0
    steps_list = []
    details = []

    for config in test_configs:
        size = config["grid_size"]
        goal = [config["goal"]["x"], config["goal"]["y"]]
        holes = [[h["x"], h["y"]] for h in config["holes"]]
        holes_set = {tuple(h) for h in holes}

        # Reset SMM to initial state
        model.reset_to_initial()

        # Run episode
        player = [0, 0]
        trajectory = [tuple(player)]
        result_reason = "timeout"
        final_step = max_steps

        for step in range(max_steps):
            # Check terminal conditions
            if tuple(player) == tuple(goal):
                result_reason = "goal"
                final_step = step
                successes += 1
                steps_list.append(step)
                break

            if tuple(player) in holes_set:
                result_reason = "hole"
                final_step = step
                break

            # Encode observation
            obs = encode_observation(player, goal, holes, size)

            # Query SMM for action
            action = get_smm_action(model, obs, use_sampling)

            if action is None:
                # No transition for this observation - pick random action
                action = np.random.choice(["LEFT", "RIGHT", "UP", "DOWN"])

            # Execute action
            delta = action_to_delta(action)
            new_x = max(0, min(size - 1, player[0] + delta[0]))
            new_y = max(0, min(size - 1, player[1] + delta[1]))
            player = [new_x, new_y]
            trajectory.append(tuple(player))

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
                       eps: float = 0.05, use_sampling: bool = False) -> Dict:
    """
    Train SMM and evaluate on test configs.

    Returns combined results dict.
    """
    import time

    start = time.time()
    model, meta = train(trace_dir, eps=eps)
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

    eval_result = evaluate(model, test_configs, use_sampling=use_sampling)

    return {
        "method": "alergia",
        "train_time": train_time,
        **meta,
        **eval_result
    }


# ============== CLI ==============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Alergia baseline for FrozenLake")
    parser.add_argument("trace_dir", type=Path, help="Directory with pos/ traces")
    parser.add_argument("--test-configs", type=Path, help="JSON file with test configs")
    parser.add_argument("--eps", type=float, default=0.05, help="Epsilon for Hoeffding test")
    parser.add_argument("--sample", action="store_true", help="Sample from action distribution")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Train
    print(f"Training Alergia SMM from {args.trace_dir}...")
    model, meta = train(args.trace_dir, eps=args.eps, print_info=args.verbose)

    print(f"Training metadata: {json.dumps(meta, indent=2)}")

    if model is None:
        print("Alergia learning failed!")
        exit(1)

    print(f"Learned SMM with {len(model.states)} states")

    # Evaluate if test configs provided
    if args.test_configs and args.test_configs.exists():
        with open(args.test_configs) as f:
            test_configs = json.load(f)

        print(f"\nEvaluating on {len(test_configs)} test configs...")
        result = evaluate(model, test_configs, use_sampling=args.sample)
        print(f"Success rate: {result['success_rate']:.2%} ({result['successes']}/{result['total']})")
