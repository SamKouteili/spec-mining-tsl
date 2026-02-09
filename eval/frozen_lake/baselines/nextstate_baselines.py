#!/usr/bin/env python3
"""
Behavioral Cloning Baseline - Next State Prediction

Instead of predicting discrete actions, BC predicts the next state directly.
This is fairer comparison to TSL_f which discovers functions from data.

If BC predicts an invalid next state, the episode fails.
"""

import json
import numpy as np
import random
from pathlib import Path
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.tree import DecisionTreeRegressor


# ============== Environment ==============

class FrozenLakeEnv:
    """Simple FrozenLake environment."""

    def __init__(self, size=4, random_placements=True):
        self.size = size
        self.random_placements = random_placements
        self._init_board()
        self.reset()

    def _init_board(self):
        if self.random_placements:
            while True:
                self.goal = (random.randint(0, self.size - 1),
                            random.randint(0, self.size - 1))
                if self.goal != (0, 0):
                    break
            self.holes = []
            forbidden = [(0, 0), self.goal]
            while len(self.holes) < 3:
                hole = (random.randint(0, self.size - 1),
                       random.randint(0, self.size - 1))
                if hole not in forbidden and hole not in self.holes:
                    self.holes.append(hole)
        else:
            self.goal = (self.size - 1, self.size - 1)
            self.holes = [(1, 1), (3, 1), (3, 2)]

    def reset(self):
        if self.random_placements:
            self._init_board()
        self.player = (0, 0)
        self.done = False
        self.won = False
        return self._get_state()

    def _get_state(self):
        state = {
            "player": list(self.player),
            "goal": list(self.goal),
        }
        for i, hole in enumerate(self.holes):
            state[f"hole{i}"] = list(hole)
        return state

    def is_valid_move(self, new_pos):
        """Check if moving to new_pos is a valid single-step move."""
        new_x, new_y = new_pos
        cur_x, cur_y = self.player

        # Must be within bounds
        if not (0 <= new_x < self.size and 0 <= new_y < self.size):
            return False

        # Must be adjacent (Manhattan distance <= 1)
        dx = abs(new_x - cur_x)
        dy = abs(new_y - cur_y)

        # Valid moves: stay in place, or move exactly 1 step in x OR y (not both)
        if dx + dy > 1:
            return False

        return True

    def step_to_position(self, new_pos):
        """Move directly to a position (if valid)."""
        if self.done:
            return self._get_state(), 0, True, {"reason": "already_done"}

        new_x, new_y = new_pos

        # Check validity
        if not self.is_valid_move(new_pos):
            self.done = True
            self.won = False
            return self._get_state(), -1.0, True, {"reason": "invalid_move"}

        self.player = (new_x, new_y)

        if self.player == self.goal:
            self.done = True
            self.won = True
            return self._get_state(), 1.0, True, {"reason": "goal"}
        elif self.player in self.holes:
            self.done = True
            self.won = False
            return self._get_state(), -1.0, True, {"reason": "hole"}

        return self._get_state(), 0, False, {}


# ============== Trace Generation ==============

def bfs_path(env):
    """Find path from start to goal using BFS."""
    start = (0, 0)
    goal = env.goal

    if start == goal:
        return []

    queue = deque([(start, [])])
    visited = {start}

    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    while queue:
        (x, y), path = queue.popleft()
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if not (0 <= nx < env.size and 0 <= ny < env.size):
                continue
            if (nx, ny) in env.holes or (nx, ny) in visited:
                continue

            new_path = path + [(nx, ny)]

            if (nx, ny) == goal:
                return new_path

            visited.add((nx, ny))
            queue.append(((nx, ny), new_path))

    return None


def generate_positive_trace(env):
    """Generate a successful trace using BFS."""
    env.reset()
    path = bfs_path(env)

    if path is None:
        return None

    trace = [env._get_state()]
    for next_pos in path:
        state, _, done, info = env.step_to_position(next_pos)
        trace.append(state)
        if done and info.get("reason") != "goal":
            return None

    if env.won:
        return trace
    return None


def generate_traces(n_traces, random_placements=True):
    """Generate n positive traces."""
    traces = []
    env = FrozenLakeEnv(size=4, random_placements=random_placements)

    attempts = 0
    max_attempts = n_traces * 100

    while len(traces) < n_traces and attempts < max_attempts:
        attempts += 1
        trace = generate_positive_trace(env)
        if trace is not None:
            traces.append(trace)

    return traces


# ============== Feature Extraction ==============

def state_to_features(state, board_size=4):
    """
    Convert state to feature vector using RELATIVE features for generalization.

    Features (14 total):
    - Direction to goal (normalized): dx, dy
    - Manhattan distance to goal (normalized)
    - For each hole: direction from player (dx, dy) and whether adjacent
    """
    player = state["player"]
    goal = state["goal"]

    # Handle variable board sizes
    norm = max(board_size - 1, 1)

    holes = []
    for i in range(3):
        key = f"hole{i}"
        if key in state:
            holes.append(state[key])
        else:
            holes.append([board_size, board_size])  # Far away if missing

    # Direction to goal (relative, normalized)
    dx_goal = (goal[0] - player[0]) / norm
    dy_goal = (goal[1] - player[1]) / norm
    dist_goal = (abs(goal[0] - player[0]) + abs(goal[1] - player[1])) / (2 * norm)

    features = [
        # Current position (still needed for boundary awareness)
        player[0] / norm,
        player[1] / norm,
        # Goal direction and distance (relative)
        dx_goal,
        dy_goal,
        dist_goal,
    ]

    # For each hole: relative direction and adjacency indicator
    for hole in holes:
        dx_hole = (hole[0] - player[0]) / norm
        dy_hole = (hole[1] - player[1]) / norm
        # Is this hole adjacent? (Manhattan distance <= 1)
        is_adjacent = 1.0 if (abs(hole[0] - player[0]) + abs(hole[1] - player[1])) <= 1 else 0.0
        features.extend([dx_hole, dy_hole, is_adjacent])

    return features  # 5 + 3*3 = 14 features


def delta_to_action(delta_x, delta_y):
    """Convert movement delta to action index.
    Actions: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP, 4=STAY
    """
    if delta_x == -1 and delta_y == 0:
        return 0  # LEFT
    elif delta_x == 0 and delta_y == 1:
        return 1  # DOWN
    elif delta_x == 1 and delta_y == 0:
        return 2  # RIGHT
    elif delta_x == 0 and delta_y == -1:
        return 3  # UP
    else:
        return 4  # STAY (or invalid)


def action_to_delta(action):
    """Convert action index to movement delta."""
    if action == 0:
        return (-1, 0)  # LEFT
    elif action == 1:
        return (0, 1)   # DOWN
    elif action == 2:
        return (1, 0)   # RIGHT
    elif action == 3:
        return (0, -1)  # UP
    else:
        return (0, 0)   # STAY


def extract_training_data(traces, predict_delta=True, as_actions=False):
    """
    Extract (state, target) pairs from traces.

    If as_actions=True: target is action class (0-4)
    If predict_delta=True: target is movement delta (-1, 0, or 1 for each axis)
    If predict_delta=False: target is normalized next position (legacy)
    """
    states = []
    targets = []

    for trace in traces:
        # Infer board size from goal position (max coordinate + 1 or from other info)
        if trace:
            first_state = trace[0]
            # Estimate board size as max coordinate we see + 1
            all_coords = [first_state["player"][0], first_state["player"][1],
                         first_state["goal"][0], first_state["goal"][1]]
            for i in range(3):
                key = f"hole{i}"
                if key in first_state:
                    all_coords.extend([first_state[key][0], first_state[key][1]])
            board_size = max(all_coords) + 1
            board_size = max(board_size, 4)  # At least 4x4
        else:
            board_size = 4

        for i in range(len(trace) - 1):
            current_state = trace[i]
            next_state = trace[i + 1]

            features = state_to_features(current_state, board_size)
            current_pos = current_state["player"]
            next_pos = next_state["player"]

            delta_x = next_pos[0] - current_pos[0]
            delta_y = next_pos[1] - current_pos[1]

            if as_actions:
                # Classification: action class
                target = delta_to_action(delta_x, delta_y)
            elif predict_delta:
                # Regression: movement delta
                target = [delta_x, delta_y]
            else:
                # Legacy: normalized absolute position
                norm = board_size - 1
                target = [next_pos[0] / norm, next_pos[1] / norm]

            states.append(features)
            targets.append(target)

    if as_actions:
        return np.array(states, dtype=np.float32), np.array(targets, dtype=np.int64)
    return np.array(states, dtype=np.float32), np.array(targets, dtype=np.float32)


# ============== Model ==============

class BCNextStateModel(nn.Module):
    """BC model that predicts movement delta (regression)."""

    def __init__(self, input_dim=14, hidden_dim=64, output_dim=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Output in [-1, 1] range for deltas
        )

    def forward(self, x):
        return self.network(x)

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)


class BCActionModel(nn.Module):
    """BC model that predicts action class (classification)."""

    def __init__(self, input_dim=14, hidden_dim=128, num_actions=5):
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

    def predict_action(self, x):
        """Return predicted action index."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def predict(self, x):
        """Return delta for compatibility with evaluate_model."""
        actions = self.predict_action(x).numpy()
        deltas = np.array([action_to_delta(a) for a in actions], dtype=np.float32)
        return deltas


def train_bc(states, deltas, epochs=300, lr=0.001, batch_size=32):
    """Train BC model to predict movement delta."""
    model = BCNextStateModel(input_dim=states.shape[1])

    X = torch.tensor(states)
    y = torch.tensor(deltas)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

    # Compute training accuracy (how often prediction rounds to correct delta)
    model.eval()
    with torch.no_grad():
        preds = model.predict(X).numpy()
        # Round to nearest integer (-1, 0, or 1)
        preds_rounded = np.round(preds).astype(int)
        targets_rounded = np.round(y.numpy()).astype(int)
        accuracy = (preds_rounded == targets_rounded).all(axis=1).mean()

    return model, accuracy


def train_dt(states, deltas):
    """Train Decision Tree regressor to predict movement delta."""
    model = DecisionTreeRegressor(random_state=42, max_depth=10)
    model.fit(states, deltas)

    # Compute training accuracy
    preds = model.predict(states)
    preds_rounded = np.round(preds).astype(int)
    targets_rounded = np.round(deltas).astype(int)
    accuracy = (preds_rounded == targets_rounded).all(axis=1).mean()

    return model, accuracy


def train_bc_classifier(states, actions, epochs=300, lr=0.001, batch_size=32):
    """Train BC model as action classifier."""
    model = BCActionModel(input_dim=states.shape[1])

    X = torch.tensor(states)
    y = torch.tensor(actions, dtype=torch.long)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    # Compute training accuracy
    model.eval()
    with torch.no_grad():
        preds = model.predict_action(X).numpy()
        accuracy = (preds == actions).mean()

    return model, accuracy


def train_dt_classifier(states, actions):
    """Train Decision Tree classifier for actions."""
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=42, max_depth=15)
    model.fit(states, actions)

    # Compute training accuracy
    preds = model.predict(states)
    accuracy = (preds == actions).mean()

    return model, accuracy


class DTClassifierWrapper:
    """Wrapper for DT classifier to match BC interface."""
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        actions = self.model.predict(x)
        deltas = np.array([action_to_delta(a) for a in actions], dtype=np.float32)
        return deltas


class DTWrapper:
    """Wrapper to give DT same interface as BC model."""
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        return self.model.predict(x)


# ============== Evaluation ==============

def evaluate_model(model, model_type, num_episodes=100, random_placements=True):
    """Evaluate model on FrozenLake."""
    wins = 0
    invalid_moves = 0
    holes_hit = 0
    timeouts = 0

    for _ in range(num_episodes):
        env = FrozenLakeEnv(size=4, random_placements=random_placements)
        state = env.reset()

        for step in range(50):
            features = state_to_features(state)

            # Predict movement delta
            if model_type == "bc":
                features_tensor = torch.tensor([features], dtype=torch.float32)
                pred = model.predict(features_tensor)
                if hasattr(pred, 'numpy'):
                    pred = pred.numpy()
                pred = pred[0]
            else:  # dt
                pred = model.predict([features])[0]

            # Round delta to nearest integer (-1, 0, or 1)
            delta_x = int(round(pred[0]))
            delta_y = int(round(pred[1]))

            # Clamp deltas to valid range
            delta_x = max(-1, min(1, delta_x))
            delta_y = max(-1, min(1, delta_y))

            # Apply delta to current position
            current_x, current_y = state["player"]
            pred_x = current_x + delta_x
            pred_y = current_y + delta_y

            state, _, done, info = env.step_to_position((pred_x, pred_y))

            if done:
                reason = info.get("reason")
                if reason == "goal":
                    wins += 1
                elif reason == "invalid_move":
                    invalid_moves += 1
                elif reason == "hole":
                    holes_hit += 1
                break
        else:
            timeouts += 1

    return {
        "win_rate": wins / num_episodes,
        "invalid_rate": invalid_moves / num_episodes,
        "hole_rate": holes_hit / num_episodes,
        "timeout_rate": timeouts / num_episodes,
    }


# ============== Main Experiment ==============

def run_scaling_experiment():
    n_values = [10, 25, 50, 100, 200, 500, 1000]
    num_runs = 3

    bc_results = []
    dt_results = []

    print("=" * 70)
    print("NEXT-STATE PREDICTION - SCALING EXPERIMENT")
    print("=" * 70)
    print("\nBoth BC and DT predict next state directly (regression).")
    print("Invalid moves → episode fails.")
    print(f"\nTesting n = {n_values}")
    print()

    for n in n_values:
        print(f"\n--- n = {n} ---")

        bc_random_wins = []
        bc_fixed_wins = []
        bc_invalid_rates = []
        dt_random_wins = []
        dt_fixed_wins = []
        dt_invalid_rates = []
        sample_counts = []

        for run in range(num_runs):
            # Generate training traces
            traces = generate_traces(n, random_placements=True)
            states, next_positions = extract_training_data(traces)
            sample_counts.append(len(states))

            # Train and evaluate BC
            bc_model, bc_train_acc = train_bc(states, next_positions)
            bc_random_result = evaluate_model(bc_model, "bc", random_placements=True)
            bc_fixed_result = evaluate_model(bc_model, "bc", random_placements=False)

            bc_random_wins.append(bc_random_result["win_rate"])
            bc_fixed_wins.append(bc_fixed_result["win_rate"])
            bc_invalid_rates.append(bc_random_result["invalid_rate"])

            # Train and evaluate DT
            dt_model, dt_train_acc = train_dt(states, next_positions)
            dt_random_result = evaluate_model(dt_model, "dt", random_placements=True)
            dt_fixed_result = evaluate_model(dt_model, "dt", random_placements=False)

            dt_random_wins.append(dt_random_result["win_rate"])
            dt_fixed_wins.append(dt_fixed_result["win_rate"])
            dt_invalid_rates.append(dt_random_result["invalid_rate"])

            print(f"  Run {run+1}: samples={len(states)}, "
                  f"BC random={bc_random_result['win_rate']:.0%} (inv={bc_random_result['invalid_rate']:.0%}), "
                  f"DT random={dt_random_result['win_rate']:.0%} (inv={dt_random_result['invalid_rate']:.0%})")

        bc_results.append({
            "n": n,
            "mean_samples": float(np.mean(sample_counts)),
            "random_mean": float(np.mean(bc_random_wins)),
            "random_std": float(np.std(bc_random_wins)),
            "fixed_mean": float(np.mean(bc_fixed_wins)),
            "fixed_std": float(np.std(bc_fixed_wins)),
            "invalid_mean": float(np.mean(bc_invalid_rates)),
        })

        dt_results.append({
            "n": n,
            "mean_samples": float(np.mean(sample_counts)),
            "random_mean": float(np.mean(dt_random_wins)),
            "random_std": float(np.std(dt_random_wins)),
            "fixed_mean": float(np.mean(dt_fixed_wins)),
            "fixed_std": float(np.std(dt_fixed_wins)),
            "invalid_mean": float(np.mean(dt_invalid_rates)),
        })

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: BEHAVIORAL CLONING (Neural Net)")
    print("=" * 70)
    print(f"{'n':>6} | {'Samples':>8} | {'Random Win':>14} | {'Fixed Win':>14} | {'Invalid':>8}")
    print("-" * 65)

    for r in bc_results:
        print(f"{r['n']:>6} | {r['mean_samples']:>8.0f} | "
              f"{r['random_mean']:>5.1%} +/- {r['random_std']:.1%} | "
              f"{r['fixed_mean']:>5.1%} +/- {r['fixed_std']:.1%} | "
              f"{r['invalid_mean']:>7.1%}")

    print("\n" + "=" * 70)
    print("SUMMARY: DECISION TREE (Regressor)")
    print("=" * 70)
    print(f"{'n':>6} | {'Samples':>8} | {'Random Win':>14} | {'Fixed Win':>14} | {'Invalid':>8}")
    print("-" * 65)

    for r in dt_results:
        print(f"{r['n']:>6} | {r['mean_samples']:>8.0f} | "
              f"{r['random_mean']:>5.1%} +/- {r['random_std']:.1%} | "
              f"{r['fixed_mean']:>5.1%} +/- {r['fixed_std']:.1%} | "
              f"{r['invalid_mean']:>7.1%}")

    # Save results
    output_path = Path(__file__).parent / "nextstate_results.json"
    with open(output_path, "w") as f:
        json.dump({"bc": bc_results, "dt": dt_results}, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return bc_results, dt_results


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    run_scaling_experiment()
