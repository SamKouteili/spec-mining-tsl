#!/usr/bin/env python3
"""
Alergia Baseline for CliffWalking with Variant Moves

This version uses an observation encoding specifically designed for non-linear
movement functions:
  - left: x_new = x - 1
  - right: x_new = (x * 2) + 1
  - up: y_new = y + 2
  - down: y_new = y - 1

The key insight is that with non-linear moves, we need to encode whether
each move would OVERSHOOT the goal, not just the direction to the goal.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from aalpy.learning_algs import run_Alergia
from aalpy.base import Automaton


# ============== State Encoding for Var Moves ==============

def encode_observation_var_moves(x: int, y: int, goal_x: int, goal_y: int,
                                  cliff_x_min: int, cliff_x_max: int, cliff_height: int,
                                  width: int, height: int) -> str:
    """
    Encode game state using zone + safe moves encoding.

    Key insight: The observation must uniquely determine the optimal action
    across ALL board configurations. We encode:
    1. Zone: above/below cliff, left/at/right of goal
    2. Which moves are safe (to avoid cliff/OOB)
    3. Whether RIGHT would overshoot (the key var_moves issue)
    """
    if x == goal_x and y == goal_y:
        return "GOAL"

    def is_cliff_pos(cx, cy):
        return cliff_x_min <= cx <= cliff_x_max and cy < cliff_height

    def is_oob(cx, cy):
        return cx < 0 or cx >= width or cy < 0 or cy >= height

    def is_safe(cx, cy):
        return not is_oob(cx, cy) and not is_cliff_pos(cx, cy)

    # Zone encoding (3 bits)
    above_cliff = y >= cliff_height
    at_goal_x = x == goal_x
    past_goal_x = x > goal_x

    # Zone name
    if past_goal_x:
        zone = "R"  # Right of goal (overshot)
    elif at_goal_x:
        zone = "X"  # At goal X
    elif above_cliff:
        zone = "A"  # Above cliff, need to go right
    else:
        zone = "B"  # Below cliff, need to go up

    # Y position relative to goal
    if y > goal_y:
        y_rel = "H"  # High (above goal)
    elif y < goal_y:
        y_rel = "L"  # Low (below goal)
    else:
        y_rel = "Y"  # At goal Y

    # Safe moves (only encode the most important ones for each zone)
    up_safe = is_safe(x, y + 2)
    down_safe = is_safe(x, y - 1)
    right_safe = is_safe((x * 2) + 1, y)
    right_overshoot = ((x * 2) + 1) > goal_x

    # Critical: in zone A, whether RIGHT is usable (safe AND doesn't overshoot)
    if zone == "A":
        right_ok = right_safe and not right_overshoot
        safe_str = "r" if right_ok else "n"  # r=right ok, n=no right
    elif zone == "B":
        safe_str = "u" if up_safe else "n"  # u=up safe
    elif zone == "X":
        safe_str = "d" if down_safe else "n"  # d=down safe
    else:  # zone R (overshot)
        safe_str = "l"  # left is always the goal

    return f"{zone}{y_rel}{safe_str}"


def decode_action_var_moves(curr_x: int, curr_y: int, next_x: int, next_y: int) -> str:
    """Infer action from position change with var_moves."""
    delta_x = next_x - curr_x
    delta_y = next_y - curr_y

    # Var moves (in logged coords, y=0 is bottom):
    # left: x_new = x - 1 (delta_x = -1)
    # right: x_new = (x * 2) + 1 (delta_x = x + 1)
    # up: y_new = y + 2 (delta_y = +2)
    # down: y_new = y - 1 (delta_y = -1)

    if delta_y == 2 and delta_x == 0:
        return "UP"
    elif delta_y == -1 and delta_x == 0:
        return "DOWN"
    elif delta_x == -1 and delta_y == 0:
        return "LEFT"
    elif delta_x == curr_x + 1 and delta_y == 0:
        return "RIGHT"  # x_new = (x*2) + 1, so delta = x + 1
    else:
        return "STAY"


def action_to_delta_var_moves(action: str, curr_x: int, curr_y: int,
                               width: int, height: int) -> Tuple[int, int]:
    """Convert action string to new position with var_moves."""
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

    # Stay in place if out of bounds
    if 0 <= new_x < width and 0 <= new_y < height:
        return new_x, new_y
    else:
        return curr_x, curr_y


# ============== Trace Loading & Conversion ==============

def load_traces_from_dir(trace_dir: Path) -> List[List[dict]]:
    """Load traces from directory."""
    traces = []
    pos_dir = trace_dir / "pos"

    if not pos_dir.exists():
        return traces

    for trace_file in sorted(pos_dir.glob("*.jsonl")):
        states = []
        with open(trace_file) as f:
            for line in f:
                states.append(json.loads(line))
        if states:
            traces.append(states)

    return traces


def trace_to_smm_sequence(trace: List[dict], width: int, height: int) -> List[Tuple[str, str]]:
    """Convert a trace to SMM format with var_moves encoding."""
    sequence = []

    for i in range(len(trace) - 1):
        curr = trace[i]
        next_state = trace[i + 1]

        x = curr["x"]
        y = curr["y"]
        goal_x = curr["goalX"]
        goal_y = curr["goalY"]
        cliff_x_min = curr["cliffXMin"]
        cliff_x_max = curr["cliffXMax"]
        cliff_height = curr["cliffHeight"]

        # Encode observation with var_moves-specific features
        obs = encode_observation_var_moves(x, y, goal_x, goal_y,
                                           cliff_x_min, cliff_x_max, cliff_height,
                                           width, height)

        # Decode action
        action = decode_action_var_moves(x, y, next_state["x"], next_state["y"])

        sequence.append((obs, action))

    return sequence


def infer_board_dimensions(traces: List[List[dict]]) -> Tuple[int, int]:
    """Infer board dimensions from traces."""
    max_x = 0
    max_y = 0
    for trace in traces:
        for state in trace:
            max_x = max(max_x, state["x"], state["goalX"])
            max_y = max(max_y, state["y"], state["goalY"])
    # Add 1 because positions are 0-indexed
    return max_x + 1, max_y + 1


def convert_to_alergia_format(traces: List[List[dict]]) -> List[List[Tuple[str, str]]]:
    """Convert game traces to Alergia SMM format."""
    if not traces:
        return []

    # Infer dimensions from traces (use max across all traces)
    width, height = 0, 0
    for trace in traces:
        for state in trace:
            width = max(width, state["x"] + 1, state["goalX"] + 1)
            height = max(height, state["y"] + 1, state["goalY"] + 1)
    # Use reasonable maximums
    width = max(width, 15)
    height = max(height, 6)

    result = []
    for trace in traces:
        seq = trace_to_smm_sequence(trace, width, height)
        if seq:
            result.append(seq)
    return result


# ============== Training ==============

def train(trace_dir: Path, eps: float = 0.05,
          print_info: bool = False) -> Tuple[Optional[Automaton], dict]:
    """Train SMM from traces using var_moves encoding."""
    traces = load_traces_from_dir(trace_dir)

    if not traces:
        raise ValueError(f"No traces found in {trace_dir}/pos/")

    smm_data = convert_to_alergia_format(traces)

    if not smm_data:
        raise ValueError("No valid sequences extracted from traces")

    all_inputs = set()
    all_outputs = set()
    for seq in smm_data:
        for obs, action in seq:
            all_inputs.add(obs)
            all_outputs.add(action)

    metadata = {
        "num_traces": len(traces),
        "num_sequences": len(smm_data),
        "unique_observations": sorted(all_inputs),
        "unique_actions": sorted(all_outputs),
        "avg_trace_length": np.mean([len(seq) for seq in smm_data]),
        "eps": eps
    }

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
             cliff_height: int) -> bool:
    """Check if position is on the cliff."""
    return cliff_x_min <= x <= cliff_x_max and y < cliff_height


def get_smm_action(model: Automaton, observation: str) -> Optional[str]:
    """Get action from SMM given an observation."""
    try:
        current_state = model.current_state

        if observation not in current_state.transitions:
            return None

        transitions = current_state.transitions[observation]
        if not transitions:
            return None

        best = max(transitions, key=lambda t: t[2])
        target_state, action, _ = best
        model.current_state = target_state
        return action

    except Exception:
        return None


def evaluate(model: Automaton, test_configs: List[dict],
             max_steps: int = 100) -> Dict:
    """Evaluate SMM on test configurations with var_moves."""
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

        model.reset_to_initial()

        x, y = 0, 0
        trajectory = [(x, y)]
        result_reason = "timeout"
        final_step = max_steps

        for step in range(max_steps):
            if x == goal_x and y == goal_y:
                result_reason = "goal"
                final_step = step
                successes += 1
                steps_list.append(step)
                break

            if is_cliff(x, y, cliff_x_min, cliff_x_max, cliff_height):
                result_reason = "cliff"
                final_step = step
                break

            obs = encode_observation_var_moves(x, y, goal_x, goal_y,
                                               cliff_x_min, cliff_x_max, cliff_height,
                                               width, height)

            action = get_smm_action(model, obs)

            if action is None:
                # Random action if no transition
                action = np.random.choice(["LEFT", "RIGHT", "UP", "DOWN"])

            new_x, new_y = action_to_delta_var_moves(action, x, y, width, height)
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


# ============== Convenience Function ==============

def train_and_evaluate(trace_dir: Path, test_configs: List[dict],
                       eps: float = 2.0) -> Dict:
    """Train SMM and evaluate on test configs."""
    import time

    start = time.time()
    model, meta = train(trace_dir, eps=eps)
    train_time = time.time() - start

    if model is None:
        return {
            "method": "alergia_var_moves",
            "train_time": train_time,
            "error": meta.get("error", "Alergia learning failed"),
            "successes": 0,
            "total": len(test_configs),
            "success_rate": 0.0,
            **meta
        }

    eval_result = evaluate(model, test_configs)

    return {
        "method": "alergia_var_moves",
        "train_time": train_time,
        **meta,
        **eval_result
    }
