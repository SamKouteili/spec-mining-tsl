#!/usr/bin/env python3
"""
Decision Tree (DT) Baseline for Blackjack

Decision tree classifier that learns hit/stand action classification from demonstration traces.
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
    get_feature_dim,
    extract_features,
)


# ============== Training ==============

def train(trace_dir: Path, strategy: str, max_depth: int = None,
          min_samples_leaf: int = 1) -> Tuple[DecisionTreeClassifier, float, dict]:
    """
    Train DT model from traces.

    Args:
        trace_dir: Directory containing pos/ subfolder with .jsonl traces
        strategy: Strategy name (threshold, conservative, basic)
        max_depth: Maximum tree depth (None for unlimited)
        min_samples_leaf: Minimum samples required at leaf node

    Returns:
        model: Trained DecisionTreeClassifier
        accuracy: Training accuracy
        metadata: Training metadata dict
    """
    traces = load_traces_from_dir(trace_dir)
    if not traces:
        raise ValueError(f"No traces found in {trace_dir}/pos/")

    X, y = extract_state_action_pairs(traces, strategy)
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
        "strategy": strategy,
        "tree_depth": model.get_depth(),
        "num_leaves": model.get_n_leaves(),
        "action_distribution": np.bincount(y, minlength=2).tolist()
    }

    return model, float(accuracy), metadata


# ============== Convenience Function ==============

def train_and_evaluate(trace_dir: Path, strategy: str) -> Tuple[DecisionTreeClassifier, dict]:
    """
    Train DT model from traces.

    Returns:
        model: Trained model
        metadata: Training metadata
    """
    import time

    start = time.time()
    model, train_acc, meta = train(trace_dir, strategy)
    train_time = time.time() - start

    return model, {
        "method": "dt",
        "train_time": train_time,
        "train_accuracy": train_acc,
        **meta
    }
