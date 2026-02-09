#!/usr/bin/env python3
"""
Behavioral Cloning (BC) Baseline for FrozenLake

Neural network that learns action classification from demonstration traces.
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

    def __init__(self, input_dim: int = 14, hidden_dim: int = 128, num_actions: int = 5):
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


def extract_state_action_pairs(traces: List[List[dict]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract (state_features, action) pairs from traces.

    Returns:
        X: Feature array of shape (n_samples, 14)
        y: Action array of shape (n_samples,) with values 0-4
    """
    X = []
    y = []

    for trace in traces:
        for i in range(len(trace) - 1):
            curr = trace[i]
            next_state = trace[i + 1]

            # Get positions - handle both formats
            if "player" in curr and isinstance(curr["player"], list):
                curr_x, curr_y = curr["player"]
                next_x, next_y = next_state["player"]
                goal_x, goal_y = curr["goal"]
                holes = []
                for h in range(5):
                    key = f"hole{h}"
                    if key in curr and isinstance(curr[key], list):
                        holes.append(tuple(curr[key]))
            else:
                curr_x = curr.get("playerX", curr.get("player_x", 0))
                curr_y = curr.get("playerY", curr.get("player_y", 0))
                next_x = next_state.get("playerX", next_state.get("player_x", 0))
                next_y = next_state.get("playerY", next_state.get("player_y", 0))
                goal_x = curr.get("goalX", curr.get("goal_x", 3))
                goal_y = curr.get("goalY", curr.get("goal_y", 3))
                holes = []
                for h in range(5):
                    hx = curr.get(f"hole{h}X", curr.get(f"hole{h}_x", -1))
                    hy = curr.get(f"hole{h}Y", curr.get(f"hole{h}_y", -1))
                    if hx >= 0:
                        holes.append((hx, hy))

            # Compute action from delta
            delta_x = next_x - curr_x
            delta_y = next_y - curr_y
            action = delta_to_action(delta_x, delta_y)

            # Build features: [player_x, player_y, goal_x, goal_y, hole0_x, hole0_y, ...]
            features = [curr_x, curr_y, goal_x, goal_y]
            for h in range(5):
                if h < len(holes):
                    features.extend([holes[h][0], holes[h][1]])
                else:
                    features.extend([0, 0])

            X.append(features)
            y.append(action)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def delta_to_action(delta_x: int, delta_y: int) -> int:
    """Convert movement delta to action index. 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP, 4=STAY"""
    if delta_x == -1 and delta_y == 0:
        return 0  # LEFT
    elif delta_x == 0 and delta_y == 1:
        return 1  # DOWN
    elif delta_x == 1 and delta_y == 0:
        return 2  # RIGHT
    elif delta_x == 0 and delta_y == -1:
        return 3  # UP
    else:
        return 4  # STAY


def action_to_delta(action: int) -> Tuple[int, int]:
    """Convert action index to movement delta."""
    deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (0, 0)}
    return deltas.get(action, (0, 0))


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
          batch_size: int = 32, deduplicate: bool = False) -> Tuple[BCModel, float, dict]:
    """
    Train BC model from traces.

    Args:
        trace_dir: Directory containing pos/ subfolder with .jsonl traces
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        deduplicate: If True, remove conflicting examples with majority voting

    Returns:
        model: Trained BCModel
        accuracy: Training accuracy
        metadata: Training metadata dict
    """
    traces = load_traces_from_dir(trace_dir)
    if not traces:
        raise ValueError(f"No traces found in {trace_dir}/pos/")

    X, y = extract_state_action_pairs(traces)
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
        "action_distribution": np.bincount(y, minlength=5).tolist()
    }

    return model, float(accuracy), metadata


# ============== Evaluation ==============

def evaluate(model: BCModel, test_configs: List[dict],
             max_steps: int = 100) -> Dict:
    """
    Evaluate BC model on test configurations.

    Args:
        model: Trained BCModel
        test_configs: List of config dicts with grid_size, goal, holes
        max_steps: Maximum steps per episode

    Returns:
        Dict with successes, total, steps list, details
    """
    successes = 0
    steps_list = []
    details = []

    for config in test_configs:
        size = config["grid_size"]
        goal = (config["goal"]["x"], config["goal"]["y"])
        holes = [(h["x"], h["y"]) for h in config["holes"]]
        holes_set = set(holes)

        # Run episode
        x, y = 0, 0
        trajectory = [(x, y)]
        result_reason = "timeout"
        final_step = max_steps

        for step in range(max_steps):
            if (x, y) == goal:
                result_reason = "goal"
                final_step = step
                successes += 1
                steps_list.append(step)
                break

            if (x, y) in holes_set:
                result_reason = "hole"
                final_step = step
                break

            # Build features
            features = [x, y, goal[0], goal[1]]
            for hx, hy in holes:
                features.extend([hx, hy])
            while len(features) < 14:
                features.extend([0, 0])

            # Predict action
            features_arr = np.array(features, dtype=np.float32).reshape(1, -1)
            features_tensor = torch.tensor(features_arr, dtype=torch.float32)
            action = model.predict_action(features_tensor).item()

            # Move
            delta = action_to_delta(action)
            new_x = max(0, min(size - 1, x + delta[0]))
            new_y = max(0, min(size - 1, y + delta[1]))
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
                       epochs: int = 300) -> Dict:
    """
    Train BC model and evaluate on test configs.

    Returns combined results dict.
    """
    import time

    start = time.time()
    model, train_acc, meta = train(trace_dir, epochs=epochs)
    train_time = time.time() - start

    eval_result = evaluate(model, test_configs)

    return {
        "method": "bc",
        "train_time": train_time,
        "train_accuracy": train_acc,
        "num_traces": meta["num_traces"],
        "num_samples": meta["num_samples"],
        **eval_result
    }
