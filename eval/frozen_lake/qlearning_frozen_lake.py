#!/usr/bin/env python3
"""
Q-Learning implementation for FrozenLake evaluation.

Compares Q-learning trained on:
1. Fixed board configurations -> tested on variable configs
2. Variable board configurations -> tested on variable configs
"""

import numpy as np
import random
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path


class FrozenLakeEnv:
    """FrozenLake environment matching the game's mechanics."""

    def __init__(self, size=4, random_size=False, random_placements=False):
        self.random_size = random_size
        self.random_placements = random_placements
        self.base_size = size
        self.reset()

    def _init_board(self):
        """Initialize board configuration."""
        if self.random_size:
            self.size = random.randint(3, 5)
        else:
            self.size = self.base_size

        if self.random_placements:
            # Random goal (not at start)
            while True:
                self.goal = (random.randint(0, self.size - 1),
                            random.randint(0, self.size - 1))
                if self.goal != (0, 0):
                    break

            # Random holes (not at start or goal)
            self.holes = set()
            forbidden = {(0, 0), self.goal}
            while len(self.holes) < 3:
                h = (random.randint(0, self.size - 1),
                     random.randint(0, self.size - 1))
                if h not in forbidden and h not in self.holes:
                    self.holes.add(h)
        else:
            # Fixed positions
            self.goal = (self.size - 1, self.size - 1)
            if self.size == 4:
                self.holes = {(1, 1), (3, 1), (3, 2)}
            else:
                self.holes = {(1, 1), (self.size - 1, 1), (self.size - 1, self.size - 2)}

    def reset(self):
        """Reset environment for new episode."""
        self._init_board()
        self.player = (0, 0)
        self.done = False
        self.won = False
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        """Get state representation for Q-table.

        Always uses relative positions for consistent state space across
        fixed and variable configurations. This enables fair comparison
        of generalization capability.
        """
        px, py = self.player
        gx, gy = self.goal

        # Relative position to goal
        rel_goal = (gx - px, gy - py)

        # Relative positions to holes (sorted for consistency)
        rel_holes = tuple(sorted((hx - px, hy - py) for hx, hy in self.holes))

        # Include board size for size-varying environments
        return (self.size, rel_goal, rel_holes)

    def step(self, action):
        """Execute action. Actions: 0=up, 1=down, 2=left, 3=right."""
        if self.done:
            return self._get_state(), 0, True, {}

        px, py = self.player

        if action == 0:  # up
            py = max(0, py - 1)
        elif action == 1:  # down
            py = min(self.size - 1, py + 1)
        elif action == 2:  # left
            px = max(0, px - 1)
        elif action == 3:  # right
            px = min(self.size - 1, px + 1)

        self.player = (px, py)
        self.steps += 1

        # Check termination
        reward = 0
        if self.player == self.goal:
            self.done = True
            self.won = True
            reward = 1.0
        elif self.player in self.holes:
            self.done = True
            self.won = False
            reward = -1.0
        elif self.steps >= 100:  # Max steps
            self.done = True
            self.won = False
            reward = -0.1
        else:
            reward = -0.01  # Small step penalty to encourage efficiency

        return self._get_state(), reward, self.done, {'won': self.won}


class QLearningAgent:
    """Q-Learning agent with epsilon-greedy exploration."""

    def __init__(self, n_actions=4, alpha=0.1, gamma=0.99, epsilon=0.1, epsilon_decay=0.995, epsilon_min=0.01):
        self.n_actions = n_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def get_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        """Update Q-value using TD learning."""
        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state][action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_qlearning(env, agent, n_episodes, verbose=False):
    """Train Q-learning agent for n_episodes."""
    wins = 0
    episode_rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.get_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state

            if done:
                if info.get('won', False):
                    wins += 1
                break

        episode_rewards.append(total_reward)
        agent.decay_epsilon()

        if verbose and (episode + 1) % 100 == 0:
            recent_wins = sum(1 for r in episode_rewards[-100:] if r > 0)
            print(f"Episode {episode + 1}/{n_episodes}, Win rate (last 100): {recent_wins}%")

    return wins / n_episodes, episode_rewards


def evaluate_agent(agent, env, n_episodes=100):
    """Evaluate trained agent on test environment."""
    wins = 0

    for _ in range(n_episodes):
        state = env.reset()

        while True:
            action = agent.get_action(state, training=False)
            state, _, done, info = env.step(action)

            if done:
                if info.get('won', False):
                    wins += 1
                break

    return wins / n_episodes


def run_experiment(train_mode, test_mode, episode_counts, n_test_episodes=100, n_runs=5):
    """Run Q-learning experiment with given training/test modes.

    Args:
        train_mode: 'fixed' or 'variable'
        test_mode: 'fixed', 'var_config', or 'var_size'
        episode_counts: List of training episode counts to test
        n_test_episodes: Number of test episodes per evaluation
        n_runs: Number of independent runs for averaging

    Returns:
        Dictionary of results
    """
    results = {}

    for n_episodes in episode_counts:
        run_accuracies = []

        for run in range(n_runs):
            # Create training environment
            if train_mode == 'fixed':
                train_env = FrozenLakeEnv(size=4, random_size=False, random_placements=False)
            elif train_mode == 'var_config':
                train_env = FrozenLakeEnv(size=4, random_size=False, random_placements=True)
            else:  # var_size
                train_env = FrozenLakeEnv(size=4, random_size=True, random_placements=True)

            # Create test environment
            if test_mode == 'fixed':
                test_env = FrozenLakeEnv(size=4, random_size=False, random_placements=False)
            elif test_mode == 'var_config':
                test_env = FrozenLakeEnv(size=4, random_size=False, random_placements=True)
            else:  # var_size
                test_env = FrozenLakeEnv(size=4, random_size=True, random_placements=True)

            # Train agent
            agent = QLearningAgent(
                alpha=0.1,
                gamma=0.99,
                epsilon=1.0,  # Start with full exploration
                epsilon_decay=0.995,
                epsilon_min=0.01
            )

            train_qlearning(train_env, agent, n_episodes, verbose=False)

            # Evaluate
            accuracy = evaluate_agent(agent, test_env, n_test_episodes)
            run_accuracies.append(accuracy)

        avg_accuracy = np.mean(run_accuracies)
        std_accuracy = np.std(run_accuracies)
        results[n_episodes] = {
            'accuracy': avg_accuracy,
            'std': std_accuracy,
            'runs': run_accuracies
        }

        print(f"Episodes: {n_episodes:5d} | Accuracy: {avg_accuracy*100:5.1f}% (+/- {std_accuracy*100:.1f}%)")

    return results


def print_results_table(all_results):
    """Print results as a formatted table."""
    print("\n" + "=" * 80)
    print("Q-LEARNING RESULTS FOR FROZEN LAKE")
    print("=" * 80)

    # Get all episode counts
    episode_counts = sorted(list(all_results.values())[0].keys())

    # Header
    print(f"\n{'Training → Test':<30} | " + " | ".join(f"{n:>7}" for n in episode_counts))
    print("-" * 80)

    for exp_name, results in all_results.items():
        row = f"{exp_name:<30} |"
        for n in episode_counts:
            acc = results[n]['accuracy'] * 100
            std = results[n]['std'] * 100
            row += f" {acc:5.1f}% |"
        print(row)

    print("-" * 80)
    print("Values shown: Mean accuracy % over 5 runs")


def generate_latex_table(all_results, output_path=None):
    """Generate LaTeX table of results."""
    episode_counts = sorted(list(all_results.values())[0].keys())

    latex = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Q-Learning Results on FrozenLake (Win Rate \%)}",
        r"\begin{tabular}{l" + "c" * len(episode_counts) + "}",
        r"\toprule",
        r"Training $\rightarrow$ Test & " + " & ".join(str(n) for n in episode_counts) + r" \\",
        r"\midrule"
    ]

    for exp_name, results in all_results.items():
        row = exp_name.replace("→", r"$\rightarrow$") + " & "
        cells = []
        for n in episode_counts:
            acc = results[n]['accuracy'] * 100
            std = results[n]['std'] * 100
            cells.append(f"{acc:.1f} $\\pm$ {std:.1f}")
        row += " & ".join(cells) + r" \\"
        latex.append(row)

    latex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:qlearning_results}",
        r"\end{table}"
    ])

    latex_str = "\n".join(latex)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex_str)
        print(f"\nLaTeX table saved to: {output_path}")

    return latex_str


def main():
    parser = argparse.ArgumentParser(description="Q-Learning for FrozenLake")
    parser.add_argument('--episodes', type=int, nargs='+',
                        default=[100, 500, 1000, 2000, 5000, 10000, 20000, 50000],
                        help='List of episode counts to test')
    parser.add_argument('--n-runs', type=int, default=5,
                        help='Number of runs for averaging')
    parser.add_argument('--n-test', type=int, default=100,
                        help='Number of test episodes per evaluation')
    parser.add_argument('--output-dir', type=str, default='eval/frozen_lake/qlearning',
                        help='Output directory for results')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Experiment 0: Fixed training -> Fixed testing (baseline sanity check)
    print("\n" + "=" * 60)
    print("Experiment 0: Fixed Train → Fixed Test (Baseline)")
    print("=" * 60)
    all_results["Fixed → Fixed"] = run_experiment(
        train_mode='fixed',
        test_mode='fixed',
        episode_counts=args.episodes,
        n_test_episodes=args.n_test,
        n_runs=args.n_runs
    )

    # Experiment 1: Fixed training -> Variable config testing
    print("\n" + "=" * 60)
    print("Experiment 1: Fixed Train → Variable Config Test")
    print("=" * 60)
    all_results["Fixed → Var Config"] = run_experiment(
        train_mode='fixed',
        test_mode='var_config',
        episode_counts=args.episodes,
        n_test_episodes=args.n_test,
        n_runs=args.n_runs
    )

    # Experiment 2: Variable config training -> Variable config testing
    print("\n" + "=" * 60)
    print("Experiment 2: Variable Config Train → Variable Config Test")
    print("=" * 60)
    all_results["Var Config → Var Config"] = run_experiment(
        train_mode='var_config',
        test_mode='var_config',
        episode_counts=args.episodes,
        n_test_episodes=args.n_test,
        n_runs=args.n_runs
    )

    # Experiment 3: Fixed training -> Variable size testing
    print("\n" + "=" * 60)
    print("Experiment 3: Fixed Train → Variable Size Test")
    print("=" * 60)
    all_results["Fixed → Var Size"] = run_experiment(
        train_mode='fixed',
        test_mode='var_size',
        episode_counts=args.episodes,
        n_test_episodes=args.n_test,
        n_runs=args.n_runs
    )

    # Experiment 4: Variable size training -> Variable size testing
    print("\n" + "=" * 60)
    print("Experiment 4: Variable Size Train → Variable Size Test")
    print("=" * 60)
    all_results["Var Size → Var Size"] = run_experiment(
        train_mode='var_size',
        test_mode='var_size',
        episode_counts=args.episodes,
        n_test_episodes=args.n_test,
        n_runs=args.n_runs
    )

    # Print results
    print_results_table(all_results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON results
    json_path = output_dir / f"results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nJSON results saved to: {json_path}")

    # LaTeX table
    latex_path = output_dir / f"table_{timestamp}.tex"
    generate_latex_table(all_results, latex_path)


if __name__ == "__main__":
    main()
