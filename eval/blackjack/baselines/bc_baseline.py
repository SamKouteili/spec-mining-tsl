#!/usr/bin/env python3
"""
Behavioral Cloning (BC) Baseline for Blackjack

Neural network that learns hit/stand action classification from demonstration traces.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ============== Model ==============

class BCModel(nn.Module):
    """Neural network for hit/stand action classification."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, num_actions: int = 2):
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
        """Return predicted action indices (0=hit, 1=stand)."""
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


def get_feature_dim(strategy: str) -> int:
    """Get feature dimension based on strategy."""
    if strategy == "threshold":
        return 2  # count, standThreshold
    elif strategy == "conservative":
        return 4  # count, standThreshold, standVsWeakMin, isWeakDealer
    else:  # basic
        return 4  # count, standThreshold, standVsWeakMin, isWeakDealer


def extract_features(state: dict, strategy: str) -> List[float]:
    """Extract feature vector from state based on strategy."""
    features = [state["count"]]

    if strategy == "threshold":
        features.append(state["standThreshold"])
    elif strategy == "conservative":
        features.append(state["standThreshold"])
        features.append(state.get("standVsWeakMin", 12))  # Conservative: 12
        features.append(1.0 if state.get("isWeakDealer", False) else 0.0)
    else:  # basic
        features.append(state["standThreshold"])
        features.append(state.get("standVsWeakMin", 13))  # Basic: 13
        features.append(1.0 if state.get("isWeakDealer", False) else 0.0)

    return features


def extract_state_action_pairs(traces: List[List[dict]], strategy: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract (state_features, action) pairs from traces.

    Actions: 0=HIT, 1=STAND
    Action is inferred from stood transition: if stood changes True, action was STAND
    Otherwise action was HIT (even if busted)

    Args:
        traces: List of traces
        strategy: Strategy name for feature extraction

    Returns:
        X: Feature array
        y: Action array with values 0-1
    """
    X = []
    y = []

    for trace in traces:
        for i in range(len(trace) - 1):
            curr = trace[i]
            next_state = trace[i + 1]

            # Skip if already stood (no more actions)
            if curr["stood"]:
                continue

            # Determine action: if stood becomes True, action was STAND
            if next_state["stood"]:
                action = 1  # STAND
            else:
                action = 0  # HIT

            features = extract_features(curr, strategy)
            X.append(features)
            y.append(action)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def train(trace_dir: Path, strategy: str, epochs: int = 500, lr: float = 0.001,
          batch_size: int = 32) -> Tuple[BCModel, float, dict]:
    """
    Train BC model from traces.

    Args:
        trace_dir: Directory containing pos/ subfolder with .jsonl traces
        strategy: Strategy name (threshold, conservative, basic)
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size

    Returns:
        model: Trained BCModel
        accuracy: Training accuracy
        metadata: Training metadata dict
    """
    traces = load_traces_from_dir(trace_dir)
    if not traces:
        raise ValueError(f"No traces found in {trace_dir}/pos/")

    X, y = extract_state_action_pairs(traces, strategy)
    if len(X) == 0:
        raise ValueError("No training samples extracted from traces")

    input_dim = get_feature_dim(strategy)
    model = BCModel(input_dim=input_dim)

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
        "strategy": strategy,
        "action_distribution": np.bincount(y, minlength=2).tolist()
    }

    return model, float(accuracy), metadata


# ============== Convenience Function ==============

def train_and_evaluate(trace_dir: Path, strategy: str, epochs: int = 300) -> Tuple[BCModel, dict]:
    """
    Train BC model from traces.

    Returns:
        model: Trained model
        metadata: Training metadata
    """
    import time

    start = time.time()
    model, train_acc, meta = train(trace_dir, strategy, epochs=epochs)
    train_time = time.time() - start

    return model, {
        "method": "bc",
        "train_time": train_time,
        "train_accuracy": train_acc,
        **meta
    }
