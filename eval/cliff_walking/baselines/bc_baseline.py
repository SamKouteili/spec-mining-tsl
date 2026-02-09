#!/usr/bin/env python3
"""
Behavioral Cloning (BC) Baseline for CliffWalking

Neural network that learns action classification from demonstration traces.
Supports both standard and variant movement functions.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ============== Model ==============

class BCModel(nn.Module):
    """Neural network for action classification."""

    def __init__(self, input_dim: int = 7, hidden_dim: int = 128, num_actions: int = 5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x):
        return self.network(x)

    def predict_action(self, x) -> torch.Tensor:
        """Return predicted action indices."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)


# ============== Training ==============

def load_traces_from_dir(trace_dir: Path) -> List[List[dict]]:
    """Load traces from a directory containing .jsonl files."""
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


def extract_state_action_pairs(traces: List[List[dict]], var_moves: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract (state_features, action) pairs from traces.

    Features: [x, y, goalX, goalY, cliffXMin, cliffXMax, cliffHeight]
    Actions: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP, 4=STAY

    Args:
        traces: List of traces
        var_moves: Whether to use variant movement functions for action inference

    Returns:
        X: Feature array of shape (n_samples, 7)
        y: Action array of shape (n_samples,) with values 0-4
    """
    X = []
    y = []

    for trace in traces:
        for i in range(len(trace) - 1):
            curr = trace[i]
            next_state = trace[i + 1]

            # Get positions from cliff_walking format
            curr_x = curr["x"]
            curr_y = curr["y"]
            next_x = next_state["x"]
            next_y = next_state["y"]
            goal_x = curr["goalX"]
            goal_y = curr["goalY"]
            cliff_x_min = curr["cliffXMin"]
            cliff_x_max = curr["cliffXMax"]
            cliff_height = curr["cliffHeight"]

            # Compute action from delta
            delta_x = next_x - curr_x
            delta_y = next_y - curr_y
            action = delta_to_action(delta_x, delta_y, curr_x, var_moves)

            # Build features
            features = [curr_x, curr_y, goal_x, goal_y, cliff_x_min, cliff_x_max, cliff_height]

            X.append(features)
            y.append(action)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def delta_to_action(delta_x: int, delta_y: int, curr_x: int = 0, var_moves: bool = False) -> int:
    """
    Convert movement delta to action index.
    0=LEFT, 1=DOWN, 2=RIGHT, 3=UP, 4=STAY

    NOTE: Uses LOGGED coordinates where y=0 is BOTTOM.
    - UP increases y (moving toward top of screen)
    - DOWN decreases y (moving toward bottom of screen)

    For var_moves (in logged coords):
    - left: x_new = x - 1 (delta_x = -1)
    - right: x_new = (x * 2) + 1 (delta_x = x + 1)
    - up: y_new = y + 2 (delta_y = +2)
    - down: y_new = y - 1 (delta_y = -1)
    """
    if var_moves:
        if delta_y == 2 and delta_x == 0:
            return 3  # UP (y += 2 in logged coords)
        elif delta_y == -1 and delta_x == 0:
            return 1  # DOWN (y -= 1 in logged coords)
        elif delta_x == -1 and delta_y == 0:
            return 0  # LEFT (x -= 1)
        elif delta_x == curr_x + 1 and delta_y == 0:
            return 2  # RIGHT (x = (x*2) + 1, so delta = x + 1)
        else:
            return 4  # STAY
    else:
        # Standard movements (in logged coords: y=0 is bottom)
        if delta_x == -1 and delta_y == 0:
            return 0  # LEFT
        elif delta_y == -1 and delta_x == 0:
            return 1  # DOWN (y decreases)
        elif delta_x == 1 and delta_y == 0:
            return 2  # RIGHT
        elif delta_y == 1 and delta_x == 0:
            return 3  # UP (y increases)
        else:
            return 4  # STAY


def action_to_delta(action: int, curr_x: int, curr_y: int, width: int, height: int,
                    var_moves: bool = False) -> Tuple[int, int]:
    """
    Convert action index to new position.

    Returns new (x, y) position after applying the action.

    NOTE: Uses LOGGED coordinates where y=0 is BOTTOM.
    - UP increases y (moving toward top of screen)
    - DOWN decreases y (moving toward bottom of screen)
    """
    if var_moves:
        if action == 0:  # LEFT
            new_x = curr_x - 1
            new_y = curr_y
        elif action == 1:  # DOWN (y decreases in logged coords)
            new_x = curr_x
            new_y = curr_y - 1
        elif action == 2:  # RIGHT
            new_x = (curr_x * 2) + 1
            new_y = curr_y
        elif action == 3:  # UP (y increases in logged coords)
            new_x = curr_x
            new_y = curr_y + 2
        else:  # STAY
            return curr_x, curr_y

        # Var moves: stay in place if out of bounds
        if 0 <= new_x < width and 0 <= new_y < height:
            return new_x, new_y
        else:
            return curr_x, curr_y
    else:
        # Standard movements with clamping (in logged coords: y=0 is bottom)
        if action == 0:  # LEFT
            return max(0, curr_x - 1), curr_y
        elif action == 1:  # DOWN (y decreases toward 0)
            return curr_x, max(0, curr_y - 1)
        elif action == 2:  # RIGHT
            return min(width - 1, curr_x + 1), curr_y
        elif action == 3:  # UP (y increases toward height-1)
            return curr_x, min(height - 1, curr_y + 1)
        else:  # STAY
            return curr_x, curr_y


def deduplicate_with_majority(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove conflicting examples by keeping majority action for each feature vector."""
    from collections import defaultdict, Counter

    feature_to_actions = defaultdict(list)
    for features, action in zip(X, y):
        key = tuple(features)
        feature_to_actions[key].append(action)

    X_dedup = []
    y_dedup = []
    for features, actions in feature_to_actions.items():
        majority_action = Counter(actions).most_common(1)[0][0]
        X_dedup.append(list(features))
        y_dedup.append(majority_action)

    return np.array(X_dedup, dtype=np.float32), np.array(y_dedup, dtype=np.int64)


def train(trace_dir: Path, epochs: int = 500, lr: float = 0.001,
          batch_size: int = 32, deduplicate: bool = False,
          var_moves: bool = False) -> Tuple[BCModel, float, dict]:
    """
    Train BC model from traces.

    Args:
        trace_dir: Directory containing pos/ subfolder with .jsonl traces
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        deduplicate: If True, remove conflicting examples with majority voting
        var_moves: Whether traces use variant movement functions

    Returns:
        model: Trained BCModel
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

    model = BCModel(input_dim=X.shape[1])

    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    # Compute training accuracy
    model.eval()
    with torch.no_grad():
        preds = model.predict_action(X_tensor).numpy()
        accuracy = (preds == y).mean()

    metadata = {
        "num_traces": len(traces),
        "num_samples": len(X),
        "original_samples": original_samples,
        "deduplicated": deduplicate,
        "var_moves": var_moves,
        "action_distribution": np.bincount(y, minlength=5).tolist()
    }

    return model, float(accuracy), metadata


# ============== Evaluation ==============

def is_cliff(x: int, y: int, cliff_x_min: int, cliff_x_max: int,
             cliff_height: int, height: int) -> bool:
    """
    Check if position is on the cliff.

    Note: In cliff_walking, y coordinates are flipped for logging:
    logged_y = 0 is the bottom row (internal y = height - 1)

    cliff_height is 1-indexed: cliff occupies logged_y < cliff_height
    """
    # Cliff is at bottom (low y in logged coords)
    # Cliff x-range: from cliff_x_min to cliff_x_max
    return cliff_x_min <= x <= cliff_x_max and y < cliff_height


def evaluate(model: BCModel, test_configs: List[dict],
             max_steps: int = 100, var_moves: bool = False) -> Dict:
    """
    Evaluate BC model on test configurations.

    Args:
        model: Trained BCModel
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

        # Start position: bottom-left in logged coords is (0, height-1)
        # But logged coords have y=0 at bottom, so start is (0, height-1) internal -> (0, 0) logged
        # Actually looking at cliff_walking_game.py:
        # - Start: (0, height-1) in internal coords
        # - Logged: (0, (height-1) - (height-1)) = (0, 0)
        # So we start at logged (0, 0) but that maps to top-left in the grid visually
        # Wait, let me re-read...
        #
        # From game code:
        # logged_y = (self.height - 1) - self.player_y
        # Start: self.player_y = self.height - 1 (internal)
        # So logged_y = (height - 1) - (height - 1) = 0
        #
        # Goal: self.goal_y = self.height - 1 (internal, same row as start but rightmost)
        # So logged goal_y = 0
        #
        # So both start and goal are at logged y = 0 (bottom row in logged coords)

        x, y = 0, 0  # Start at logged (0, 0) which is bottom-left
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
            features = [x, y, goal_x, goal_y, cliff_x_min, cliff_x_max, cliff_height]

            # Predict action
            features_arr = np.array(features, dtype=np.float32).reshape(1, -1)
            features_tensor = torch.tensor(features_arr, dtype=torch.float32)
            action = model.predict_action(features_tensor).item()

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
                       epochs: int = 300, var_moves: bool = False) -> Dict:
    """
    Train BC model and evaluate on test configs.

    Returns combined results dict.
    """
    import time

    start = time.time()
    model, train_acc, meta = train(trace_dir, epochs=epochs, var_moves=var_moves)
    train_time = time.time() - start

    eval_result = evaluate(model, test_configs, var_moves=var_moves)

    return {
        "method": "bc",
        "train_time": train_time,
        "train_accuracy": train_acc,
        "num_traces": meta["num_traces"],
        "num_samples": meta["num_samples"],
        **eval_result
    }
