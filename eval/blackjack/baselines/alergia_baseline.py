#!/usr/bin/env python3
"""
Alergia Baseline for Blackjack

Learns a Stochastic Mealy Machine (SMM) from demonstration traces using Alergia.
The SMM maps state observations to action probability distributions.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from aalpy.learning_algs import run_Alergia
from aalpy.base import Automaton


# ============== State Encoding ==============

def encode_observation(count: int, strategy: str,
                       stand_threshold: int = 17,
                       stand_vs_weak_min: int = 13,
                       is_weak_dealer: bool = False) -> str:
    """
    Encode game state as a discrete observation symbol.

    Uses relative position to thresholds for generalization.
    """
    # Dealer strength from derived boolean
    dealer_str = "weak" if is_weak_dealer else "strong"

    if strategy == "threshold":
        # Threshold strategy only cares about count vs threshold
        rel_stand = "above" if count >= stand_threshold else "below"
        return f"c{rel_stand}"

    elif strategy == "conservative":
        # Track if count is in the stand_vs_weak_min to threshold range
        if count >= stand_threshold:
            count_zone = "high"
        elif count >= stand_vs_weak_min:  # 12 for conservative
            count_zone = "mid"
        else:
            count_zone = "low"
        return f"c{count_zone}_d{dealer_str}"

    else:  # basic
        # Track count relative to thresholds
        if count >= stand_threshold:
            count_zone = "high"
        elif count >= stand_vs_weak_min:
            count_zone = "mid"
        elif count > 11:
            count_zone = "edge"  # 12 only
        else:
            count_zone = "low"
        return f"c{count_zone}_d{dealer_str}"


def decode_action(curr_stood: bool, next_stood: bool) -> str:
    """Infer action from stood transition."""
    if curr_stood:
        return "DONE"  # No action - game already ended
    elif next_stood:
        return "STAND"
    else:
        return "HIT"


# ============== Trace Loading & Conversion ==============

def load_traces_from_dir(trace_dir: Path) -> List[List[dict]]:
    """Load traces from directory."""
    traces = []
    pos_dir = trace_dir / "pos"

    if pos_dir.exists():
        for trace_file in sorted(pos_dir.glob("*.jsonl")):
            states = []
            with open(trace_file) as f:
                for line in f:
                    states.append(json.loads(line))
            if states:
                traces.append(states)

    return traces


def trace_to_smm_sequence(trace: List[dict], strategy: str) -> List[Tuple[str, str]]:
    """
    Convert a trace to SMM format: [(obs1, act1), (obs2, act2), ...]
    """
    sequence = []

    # Get constants from first state
    first = trace[0]
    stand_threshold = first.get("standThreshold", 17)
    stand_vs_weak_min = first.get("standVsWeakMin", 13)

    for i in range(len(trace) - 1):
        curr = trace[i]
        next_state = trace[i + 1]

        # Skip if already stood
        if curr["stood"]:
            continue

        # Get derived boolean (if present, for conservative/basic strategies)
        is_weak_dealer = curr.get("isWeakDealer", False)

        # Encode observation
        obs = encode_observation(
            curr["count"],
            strategy,
            stand_threshold,
            stand_vs_weak_min,
            is_weak_dealer
        )

        # Decode action
        action = decode_action(curr["stood"], next_state["stood"])

        sequence.append((obs, action))

    return sequence


def convert_to_alergia_format(traces: List[List[dict]], strategy: str) -> List[List[Tuple[str, str]]]:
    """Convert game traces to Alergia SMM format."""
    result = []
    for trace in traces:
        seq = trace_to_smm_sequence(trace, strategy)
        if seq:
            result.append(seq)
    return result


# ============== Training ==============

def train(trace_dir: Path, strategy: str, eps: float = 0.05,
          print_info: bool = False) -> Tuple[Optional[Automaton], dict]:
    """
    Train a Stochastic Mealy Machine from traces using Alergia.

    Args:
        trace_dir: Directory containing pos/ subfolder with .jsonl traces
        strategy: Strategy name (threshold, conservative, basic)
        eps: Epsilon for Hoeffding compatibility test
        print_info: Print learning progress

    Returns:
        model: Learned SMM (or None if learning fails)
        metadata: Training metadata
    """
    traces = load_traces_from_dir(trace_dir)

    if not traces:
        raise ValueError(f"No traces found in {trace_dir}/pos/")

    # Convert traces to SMM format
    smm_data = convert_to_alergia_format(traces, strategy)

    if not smm_data:
        raise ValueError("No valid sequences extracted from traces")

    # Collect unique inputs and outputs
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
        "eps": eps,
        "strategy": strategy
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


# ============== Convenience Function ==============

def train_and_evaluate(trace_dir: Path, strategy: str, eps: float = 2.0) -> Tuple[Optional[Automaton], dict]:
    """
    Train SMM from traces.

    Returns:
        model: Trained model (or None)
        metadata: Training metadata
    """
    import time

    start = time.time()
    model, meta = train(trace_dir, strategy, eps=eps)
    train_time = time.time() - start

    return model, {
        "method": "alergia",
        "train_time": train_time,
        **meta
    }


def get_smm_action(model: Automaton, observation: str, use_sampling: bool = False) -> Optional[str]:
    """
    Get action from SMM given an observation.
    """
    try:
        current_state = model.current_state

        if observation not in current_state.transitions:
            return None

        transitions = current_state.transitions[observation]
        if not transitions:
            return None

        if use_sampling:
            probs = [t[2] for t in transitions]
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
                idx = np.random.choice(len(transitions), p=probs)
                target_state, action, _ = transitions[idx]
                model.current_state = target_state
                return action
        else:
            best = max(transitions, key=lambda t: t[2])
            target_state, action, _ = best
            model.current_state = target_state
            return action

    except Exception:
        return None
