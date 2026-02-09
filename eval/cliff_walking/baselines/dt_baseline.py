#!/usr/bin/env python3
"""
Decision Tree (DT) Baseline for CliffWalking

Decision tree classifier that learns action classification from demonstration traces.
Supports both standard and variant movement functions.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

from sklearn.tree import DecisionTreeClassifier

# Reuse helper functions from BC baseline
from .bc_baseline import (
    load_traces_from_dir,
    extract_state_action_pairs,
    action_to_delta,
    is_cliff,
    deduplicate_with_majority
)


# ============== Training ==============

def train(trace_dir: Path, max_depth: int = None, min_samples_leaf: int = 1,
          deduplicate: bool = False, var_moves: bool = False) -> Tuple[DecisionTreeClassifier, float, dict]:
    """
    Train DT model from traces.

    Args:
        trace_dir: Directory containing pos/ subfolder with .jsonl traces
        max_depth: Maximum tree depth (None for unlimited)
        min_samples_leaf: Minimum samples required at leaf node
        deduplicate: If True, remove conflicting examples with majority voting
        var_moves: Whether traces use variant movement functions

    Returns:
        model: Trained DecisionTreeClassifier
        accuracy: Training accuracy
        metadata: Training metadata dict
    """
    traces = load_traces_from_dir(trace_dir)
    if not traces:
        raise ValueError(f"No traces found in {trace_dir}/pos/")

    X, y = extract_state_action_pairs(traces, var_moves=var_moves)
    original_samples = len(X)

    if deduplicate:
        X, y = deduplicate_with_majority(X, y)
    if len(X) == 0:
        raise ValueError("No training samples extracted from traces")

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    model.fit(X, y)

    # Compute training accuracy
    preds = model.predict(X)
    accuracy = (preds == y).mean()

    metadata = {
        "num_traces": len(traces),
        "num_samples": len(X),
        "original_samples": original_samples,
        "deduplicated": deduplicate,
        "var_moves": var_moves,
        "tree_depth": model.get_depth(),
        "num_leaves": model.get_n_leaves(),
        "action_distribution": np.bincount(y, minlength=5).tolist()
    }

    return model, float(accuracy), metadata


# ============== Evaluation ==============

def evaluate(model: DecisionTreeClassifier, test_configs: List[dict],
             max_steps: int = 100, var_moves: bool = False) -> Dict:
    """
    Evaluate DT model on test configurations.

    Args:
        model: Trained DecisionTreeClassifier
        test_configs: List of config dicts with width, height, goalX, goalY, cliff params
        max_steps: Maximum steps per episode
        var_moves: Whether to use variant movement functions

    Returns:
        Dict with successes, total, steps list, details
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

        # Start position: logged (0, 0) which is bottom-left
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

            if is_cliff(x, y, cliff_x_min, cliff_x_max, cliff_height, height):
                result_reason = "cliff"
                final_step = step
                break

            # Build features
            features = np.array([[x, y, goal_x, goal_y, cliff_x_min, cliff_x_max, cliff_height]],
                               dtype=np.float32)

            # Predict action
            action = model.predict(features)[0]

            # Move
            new_x, new_y = action_to_delta(action, x, y, width, height, var_moves)
            x, y = new_x, new_y
            trajectory.append((x, y))

        details.append({
            "config": config.get("name", "unknown"),
            "result": "PASS" if result_reason == "goal" else "FAIL",
            "reason": result_reason,
            "steps": final_step,
            "trajectory": trajectory[:10]
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
                       var_moves: bool = False) -> Dict:
    """
    Train DT model and evaluate on test configs.

    Returns combined results dict.
    """
    import time

    start = time.time()
    model, train_acc, meta = train(trace_dir, var_moves=var_moves)
    train_time = time.time() - start

    eval_result = evaluate(model, test_configs, var_moves=var_moves)

    return {
        "method": "dt",
        "train_time": train_time,
        "train_accuracy": train_acc,
        "num_traces": meta["num_traces"],
        "num_samples": meta["num_samples"],
        "tree_depth": meta["tree_depth"],
        "num_leaves": meta["num_leaves"],
        **eval_result
    }
