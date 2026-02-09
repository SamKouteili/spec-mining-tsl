#!/usr/bin/env python3
"""
Interactive Frozen Lake Game (Tuple Version)
Navigate the grid with arrow keys to reach the goal while avoiding holes.
Each game session logs timesteps as jsonl entries with tuple-based coordinates.

This version uses tuples for positions:
  - player: [x, y]
  - goal: [x, y]
  - hole0, hole1, hole2: [x, y]

This enables tuple-based specification mining with equality predicates.
"""

import os
import json
import sys
import tty
import termios
import random
import argparse
import yaml
import time
from datetime import datetime
from pathlib import Path
from collections import deque


class TupleFrozenLakeGame:
    def __init__(self, size=4, session_dir=None, random_size=False, random_placements=False):
        """Initialize a Frozen Lake game grid."""
        self.random_size = random_size
        self.random_placements = random_placements

        # Determine board size
        if random_size:
            self.size = random.randint(3, 5)
        else:
            self.size = size

        # Initialize board and positions
        self._init_board()

        # Game state
        self.game_over = False
        self.won = False
        self.trace = []

        # Use provided session directory or create new one
        if session_dir is None:
            self.session_dir = self._create_session_dir()
        else:
            self.session_dir = session_dir

        # Log initial state (including static positions)
        self._log_state()

    def _init_board(self):
        """Initialize the board grid and positions."""
        # Starting position (always top-left)
        self.player_x = 0
        self.player_y = 0

        if self.random_placements:
            # Random goal and hole placements
            # Goal is randomly placed (not at start)
            while True:
                self.goal_x = random.randint(0, self.size - 1)
                self.goal_y = random.randint(0, self.size - 1)
                if (self.goal_x, self.goal_y) != (0, 0):  # Not at start
                    break

            # Place 3 holes randomly (not at start or goal)
            self.holes = []
            forbidden = [(0, 0), (self.goal_x, self.goal_y)]
            while len(self.holes) < 3:
                hx = random.randint(0, self.size - 1)
                hy = random.randint(0, self.size - 1)
                if (hx, hy) not in forbidden and (hx, hy) not in self.holes:
                    self.holes.append((hx, hy))
        else:
            # Fixed positions (scaled to board size if needed)
            # Goal is always bottom-right
            self.goal_x = self.size - 1
            self.goal_y = self.size - 1
            # Always exactly 3 holes for consistency with random placements
            if self.size == 4:
                self.holes = [(1, 1), (3, 1), (3, 2)]
            else:
                # For other board sizes, place 3 holes in reasonable positions
                self.holes = [
                    (1, 1),
                    (self.size - 1, 1),
                    (self.size - 1, self.size - 2)
                ]

        # Build grid based on positions
        self.grid = []
        for y in range(self.size):
            row = []
            for x in range(self.size):
                if x == 0 and y == 0:
                    row.append('S')
                elif (x, y) == (self.goal_x, self.goal_y):
                    row.append('G')
                elif (x, y) in self.holes:
                    row.append('H')
                else:
                    row.append('F')
            self.grid.append(row)

    def _create_session_dir(self):
        """Create a new directory for this game session."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path(f"Logs/tfrozen_lake/{timestamp}")
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "pos").mkdir(exist_ok=True)
        (session_dir / "neg").mkdir(exist_ok=True)
        return session_dir

    def _log_state(self):
        """Log the current state as a jsonl entry with tuple-based coordinates."""
        state = {
            "player": [self.player_x, self.player_y],
            "goal": [self.goal_x, self.goal_y],
        }

        # Add hole positions as constant tuples
        for i, (hx, hy) in enumerate(self.holes):
            state[f"hole{i}"] = [hx, hy]

        self.trace.append(state)

    def display(self):
        """Display the current game state."""
        os.system('clear' if os.name == 'posix' else 'cls')

        # ANSI color codes
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RESET = '\033[0m'

        print("=" * 40)
        print("FROZEN LAKE GAME")
        print("=" * 40)
        print(f"\nBoard Size: {self.size}x{self.size}")
        print("Controls: Arrow Keys (↑↓←→) | q: Quit")
        print("Legend: P=Player S=Start F=Frozen H=Hole G=Goal\n")

        # Display grid
        for y in range(self.size):
            row = ""
            for x in range(self.size):
                if x == self.player_x and y == self.player_y:
                    cell = f"{YELLOW} P {RESET}"
                else:
                    tile = self.grid[y][x]
                    if tile == 'H':
                        cell = f"{RED} H {RESET}"
                    elif tile == 'G':
                        cell = f"{GREEN} G {RESET}"
                    else:
                        cell = f" {tile} "
                row += cell
            print(row)

        print(f"\nPosition: ({self.player_x}, {self.player_y})")
        print(f"Moves: {len(self.trace) - 1}")  # -1 because first entry is initial state

        if self.game_over:
            if self.won:
                print(f"\n{GREEN}🎉 YOU WON! You reached the goal!{RESET}")
            else:
                print(f"\n{RED}❌ GAME OVER! You fell in a hole or quit.{RESET}")

    def move(self, direction):
        """Move the player in the specified direction."""
        if self.game_over:
            return

        new_x, new_y = self.player_x, self.player_y

        if direction == 'up':
            new_y = max(0, self.player_y - 1)
        elif direction == 'down':
            new_y = min(self.size - 1, self.player_y + 1)
        elif direction == 'left':
            new_x = max(0, self.player_x - 1)
        elif direction == 'right':
            new_x = min(self.size - 1, self.player_x + 1)

        # Update position
        self.player_x = new_x
        self.player_y = new_y

        # Log the new state
        self._log_state()

        # Check win/lose conditions
        if (self.player_x, self.player_y) == (self.goal_x, self.goal_y):
            self.game_over = True
            self.won = True
        elif (self.player_x, self.player_y) in self.holes:
            self.game_over = True
            self.won = False

    def save_trace(self):
        """Save the trace to the appropriate directory (pos or neg)."""
        subdir = "pos" if self.won else "neg"
        prefix = "pos_trace_" if self.won else "neg_trace_"

        # Count existing files to generate unique filename
        trace_dir = self.session_dir / subdir
        existing_files = list(trace_dir.glob(f"{prefix}*.jsonl"))
        trace_num = len(existing_files) + 1

        trace_file = trace_dir / f"{prefix}{trace_num}.jsonl"

        with open(trace_file, 'w') as f:
            for entry in self.trace:
                f.write(json.dumps(entry) + '\n')

        print(f"\nTrace saved to: {trace_file}")
        return trace_file

    def get_key(self):
        """Get a single keypress from the user."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)

            # Handle arrow keys (escape sequences)
            if ch == '\x1b':
                ch2 = sys.stdin.read(1)
                if ch2 == '[':
                    ch3 = sys.stdin.read(1)
                    if ch3 == 'A':
                        return 'up'
                    elif ch3 == 'B':
                        return 'down'
                    elif ch3 == 'C':
                        return 'right'
                    elif ch3 == 'D':
                        return 'left'
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def reset(self):
        """Reset the game state for a new game instance."""
        # Re-randomize board size if flag is set
        if self.random_size:
            self.size = random.randint(3, 5)

        # Re-initialize board (will randomize placements if flag is set)
        self._init_board()

        # Reset game state
        self.game_over = False
        self.won = False
        self.trace = []
        self._log_state()

    def _bfs_path(self, add_detours=False):
        """Find path from start to goal using BFS, avoiding holes.

        Args:
            add_detours: If True, add random exploration moves to create variety
        """
        start = (0, 0)
        goal = (self.goal_x, self.goal_y)

        if start == goal:
            return []

        queue = deque([(start, [])])
        visited = {start}

        while queue:
            (x, y), path = queue.popleft()

            # Randomize direction order for variety
            directions = [('up', (0, -1)), ('down', (0, 1)),
                         ('left', (-1, 0)), ('right', (1, 0))]
            random.shuffle(directions)

            # Try all four directions
            for direction, (dx, dy) in directions:
                nx, ny = x + dx, y + dy

                # Check bounds
                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    continue

                # Skip holes and visited
                if (nx, ny) in self.holes or (nx, ny) in visited:
                    continue

                new_path = path + [direction]

                # Found goal
                if (nx, ny) == goal:
                    # Add random detours if requested
                    if add_detours and random.random() < 0.5:
                        return self._add_detours(new_path)
                    return new_path

                visited.add((nx, ny))
                queue.append(((nx, ny), new_path))

        return None  # No path exists

    def _add_detours(self, base_path, num_detours=None):
        """Add random exploration moves to a path while still reaching goal.

        Inserts detours at random points that explore nearby cells but return.
        """
        if num_detours is None:
            num_detours = random.randint(1, 3)

        path = base_path.copy()

        for _ in range(num_detours):
            # Pick random position to insert detour
            insert_pos = random.randint(0, len(path))

            # Generate a short random walk and return sequence
            detour_length = random.randint(1, 2)
            detour_moves = []

            for _ in range(detour_length):
                move = random.choice(['up', 'down', 'left', 'right'])
                detour_moves.append(move)
                # Add reverse move to return
                reverse = {'up': 'down', 'down': 'up',
                          'left': 'right', 'right': 'left'}
                detour_moves.append(reverse[move])

            # Insert detour at random position
            path = path[:insert_pos] + detour_moves + path[insert_pos:]

        return path

    def _bfs_explore_random(self, max_steps=100):
        """BFS with random direction selection for exploration."""
        directions = ['up', 'down', 'left', 'right']
        path = []
        x, y = 0, 0

        for _ in range(max_steps):
            direction = random.choice(directions)

            if direction == 'up':
                ny = max(0, y - 1)
                nx = x
            elif direction == 'down':
                ny = min(self.size - 1, y + 1)
                nx = x
            elif direction == 'left':
                nx = max(0, x - 1)
                ny = y
            else:
                nx = min(self.size - 1, x + 1)
                ny = y

            if (nx, ny) != (x, y):
                path.append(direction)
                x, y = nx, ny

                if (x, y) in self.holes:
                    return path

                if (x, y) == (self.goal_x, self.goal_y):
                    if path:
                        path.pop()
                        x, y = 0, 0
                        for d in path:
                            if d == 'up':
                                y = max(0, y - 1)
                            elif d == 'down':
                                y = min(self.size - 1, y + 1)
                            elif d == 'left':
                                x = max(0, x - 1)
                            else:
                                x = min(self.size - 1, x + 1)

        return None

    def _trace_to_key(self, trace):
        """Convert trace to hashable key for uniqueness checking."""
        def make_hashable(v):
            if isinstance(v, list):
                return tuple(v)
            return v
        return tuple(tuple((k, make_hashable(v)) for k, v in sorted(state.items())) for state in trace)

    def _execute_path(self, path):
        """Execute a path (list of directions) and return the trace."""
        for direction in path:
            self.move(direction)
        return self.trace.copy()

    def generate_traces(self, num_traces):
        """Generate num_traces unique traces via BFS exploration."""
        pos_traces = set()
        neg_traces = set()

        attempts = 0
        max_attempts = num_traces * 100

        print(f"\nBFS exploration: generating {num_traces * 2} traces...")

        while (len(pos_traces) < num_traces or len(neg_traces) < num_traces) and attempts < max_attempts:
            attempts += 1

            self.reset()

            if len(pos_traces) < num_traces:
                path = self._bfs_path(add_detours=True)
                if path is not None:
                    self.reset()
                    trace = self._execute_path(path)

                    if self.won:
                        trace_key = self._trace_to_key(trace)
                        if trace_key not in pos_traces:
                            pos_traces.add(trace_key)
                            self._save_trace_to_file(trace, is_positive=True,
                                                      trace_num=len(pos_traces))
                            print(f"  BFS trace {len(pos_traces)}/{num_traces} (goal)")

            if len(neg_traces) < num_traces:
                self.reset()
                path = self._bfs_explore_random()
                if path is not None:
                    self.player_x, self.player_y = 0, 0
                    self.game_over = False
                    self.won = False
                    self.trace = []
                    self._log_state()

                    trace = self._execute_path(path)

                    final_pos = (self.player_x, self.player_y)
                    at_hole = final_pos in self.holes
                    at_goal = final_pos == (self.goal_x, self.goal_y)

                    if not at_hole or at_goal:
                        continue

                    trace_key = self._trace_to_key(trace)

                    if trace_key not in neg_traces:
                        neg_traces.add(trace_key)
                        self._save_trace_to_file(trace, is_positive=False,
                                                  trace_num=len(neg_traces))
                        print(f"  BFS trace {len(neg_traces)}/{num_traces} (exploration)")

        if len(pos_traces) < num_traces:
            print(f"\nWarning: Only found {len(pos_traces)} unique goal paths")
        if len(neg_traces) < num_traces:
            print(f"\nWarning: Only found {len(neg_traces)} unique exploration paths")

        print(f"\nBFS exploration complete!")
        print(f"  Goal paths: {len(pos_traces)}")
        print(f"  Exploration paths: {len(neg_traces)}")

    def _save_trace_to_file(self, trace, is_positive, trace_num):
        """Save a trace to file."""
        subdir = "pos" if is_positive else "neg"
        prefix = "pos_trace_" if is_positive else "neg_trace_"

        trace_dir = self.session_dir / subdir
        trace_file = trace_dir / f"{prefix}{trace_num}.jsonl"

        with open(trace_file, 'w') as f:
            for entry in trace:
                f.write(json.dumps(entry) + '\n')

    def get_config(self, name: str = None) -> dict:
        """
        Export the current board configuration as a dictionary.

        Args:
            name: Optional name for this configuration

        Returns:
            Configuration dictionary compatible with specification_validator
        """
        if name is None:
            name = f"{self.size}x{self.size}_config"

        holes_list = [{"x": hx, "y": hy} for hx, hy in self.holes]

        return {
            "name": name,
            "grid_size": self.size,
            "start_pos": {"x": 0, "y": 0},
            "goal": {"x": self.goal_x, "y": self.goal_y},
            "holes": holes_list
        }

    def play(self):
        """Main game loop for a single game instance."""
        self.display()

        quit_session = False
        quit_mid_game = False
        while not self.game_over:
            key = self.get_key()

            if key == 'q':
                quit_session = True
                quit_mid_game = True
                break
            elif key in ['up', 'down', 'left', 'right']:
                self.move(key)
                self.display()

        # Final display (only if game ended naturally)
        if not quit_mid_game:
            self.display()

            # Save the trace only if game ended naturally (win or loss, not quit)
            self.save_trace()

        return quit_session


def generate_random_configs(
    num_configs: int,
    random_size: bool = False,
    random_placements: bool = True,
    base_size: int = 4,
    seed: int = None
) -> list:
    """
    Generate multiple random board configurations.

    Args:
        num_configs: Number of configurations to generate
        random_size: Whether to randomize board size
        random_placements: Whether to randomize goal/hole placements
        base_size: Base board size if not randomizing
        seed: Random seed for reproducibility

    Returns:
        List of configuration dictionaries
    """
    if seed is not None:
        random.seed(seed)

    configs = []
    seen_configs = set()

    attempts = 0
    max_attempts = num_configs * 50

    while len(configs) < num_configs and attempts < max_attempts:
        attempts += 1

        # Create a temporary game instance to generate a random board
        game = TupleFrozenLakeGame(
            size=base_size,
            session_dir=Path("/tmp"),  # Dummy, won't be used
            random_size=random_size,
            random_placements=random_placements
        )

        # Create a hashable key for uniqueness checking
        config_key = (
            game.size,
            game.goal_x, game.goal_y,
            tuple(sorted(game.holes))
        )

        if config_key not in seen_configs:
            seen_configs.add(config_key)
            config = game.get_config(name=f"config_{len(configs) + 1}")
            configs.append(config)

    return configs


def generate_config_yaml(
    objective: str,
    num_configs: int,
    output_path: Path,
    random_size: bool = False,
    random_placements: bool = True,
    timeout_steps: int = 1000,
    synthesis_timeout: int = 10
) -> dict:
    """
    Generate a complete YAML configuration file for specification_validator.

    Args:
        objective: The TSL objective specification
        num_configs: Number of configurations to generate
        output_path: Path to write the YAML file
        random_size: Whether to randomize board sizes
        random_placements: Whether to randomize placements
        timeout_steps: Max steps for game execution
        synthesis_timeout: Synthesis timeout in minutes

    Returns:
        The generated configuration dictionary
    """
    configs = generate_random_configs(
        num_configs=num_configs,
        random_size=random_size,
        random_placements=random_placements
    )

    yaml_config = {
        "name": "ice_lake",
        "synthesis": {
            "command": "issy",
            "args": ["--tslmt", "--synt", "--pruning", "1"],
            "timeout_minutes": synthesis_timeout
        },
        "debug": True,
        "run_configuration": [
            {
                "objective": objective,
                "timeout": timeout_steps,
                "configurations": configs
            }
        ]
    }

    # Write YAML file
    with open(output_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)

    return yaml_config


def wait_for_key():
    """Wait for any keypress."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def replay_trajectory(board_config: dict, trajectory: list, delay: float = 0.5):
    """
    Replay a trajectory on a frozen lake board with animation.

    Args:
        board_config: Dictionary with grid_size, goal, holes
        trajectory: List of (x, y) tuples representing positions
        delay: Delay between moves in seconds
    """
    # Extract board config
    size = board_config.get("grid_size", 4)

    goal = board_config.get("goal", {"x": size - 1, "y": size - 1})
    if isinstance(goal, dict):
        goal_x, goal_y = goal["x"], goal["y"]
    else:
        goal_x, goal_y = goal[0], goal[1]

    holes_raw = board_config.get("holes", [])
    holes = []
    for h in holes_raw:
        if isinstance(h, dict):
            holes.append((h["x"], h["y"]))
        else:
            holes.append((h[0], h[1]))

    # Build the grid
    grid = []
    for y in range(size):
        row = []
        for x in range(size):
            if x == 0 and y == 0:
                row.append('S')
            elif (x, y) == (goal_x, goal_y):
                row.append('G')
            elif (x, y) in holes:
                row.append('H')
            else:
                row.append('F')
        grid.append(row)

    # ANSI color codes
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    RESET = '\033[0m'

    def display_board(player_x: int, player_y: int, step: int, total_steps: int, message: str = ""):
        """Display the board with current player position."""
        os.system('clear' if os.name == 'posix' else 'cls')

        print("=" * 50)
        print(f"{CYAN}SYNTHESIZED CONTROLLER REPLAY{RESET}")
        print("=" * 50)
        print(f"\nBoard Size: {size}x{size}")
        print(f"Step: {step}/{total_steps}")
        print("Legend: P=Player S=Start F=Frozen H=Hole G=Goal\n")

        # Display grid
        for y in range(size):
            row_str = ""
            for x in range(size):
                if x == player_x and y == player_y:
                    cell = f"{YELLOW} P {RESET}"
                else:
                    tile = grid[y][x]
                    if tile == 'H':
                        cell = f"{RED} H {RESET}"
                    elif tile == 'G':
                        cell = f"{GREEN} G {RESET}"
                    else:
                        cell = f" {tile} "
                row_str += cell
            print(row_str)

        print(f"\nPosition: ({player_x}, {player_y})")
        if message:
            print(f"\n{message}")

    print(f"\n{CYAN}Starting replay of {len(trajectory)} positions...{RESET}")
    time.sleep(1)

    # Replay each position
    for i, (x, y) in enumerate(trajectory):
        # Check for win/lose
        message = ""
        if (x, y) == (goal_x, goal_y):
            message = f"{GREEN}GOAL REACHED! Controller succeeded.{RESET}"
        elif (x, y) in holes:
            message = f"{RED}FELL IN HOLE! Controller failed.{RESET}"

        display_board(x, y, i + 1, len(trajectory), message)

        if message:
            # End state - pause longer
            time.sleep(2)
            break
        else:
            time.sleep(delay)

    print("\n" + "=" * 50)
    print("Replay complete!")
    print("=" * 50)


def main():
    """Run the game session with multiple game instances."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Interactive Frozen Lake Game (Tuple Version)")
    parser.add_argument('--random-size', action='store_true',
                        help='Randomize board size (3x3 to 5x5) for each game instance')
    parser.add_argument('--random-placements', action='store_true',
                        help='Randomize goal and hole positions for each game instance')
    parser.add_argument('--gen', type=int, metavar='N',
                        help='Generate N positive and N negative traces automatically (non-interactive mode)')
    parser.add_argument('--output', type=str, metavar='DIR',
                        help='Output directory for generated traces (default: Logs/tfrozen_lake/<timestamp>)')
    parser.add_argument('--config', type=int, metavar='N',
                        help='Generate N board configurations as JSON (combine with --random-size/--random-placements)')
    parser.add_argument('--objective', type=str, metavar='SPEC',
                        help='TSL objective specification for --config mode (outputs YAML if provided)')
    parser.add_argument('--config-output', type=str, metavar='FILE',
                        help='Output file for --config mode (default: stdout for JSON, configs/generated_config.yaml for YAML)')
    parser.add_argument('--timeout-steps', type=int, default=1000,
                        help='Max steps for game execution in --config mode (default: 1000)')
    parser.add_argument('--synthesis-timeout', type=int, default=10,
                        help='Synthesis timeout in minutes for --config mode (default: 10)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None, uses system time)')
    parser.add_argument('--save-config', type=str, metavar='FILE',
                        help='Save board configuration to JSON file (for demo mode)')
    parser.add_argument('--replay', type=str, metavar='TRAJECTORY',
                        help='Replay a trajectory (format: "x1,y1;x2,y2;..." positions)')
    parser.add_argument('--replay-config', type=str, metavar='FILE',
                        help='JSON file with board config for replay mode')
    parser.add_argument('--replay-delay', type=float, default=0.5,
                        help='Delay between moves in replay mode (default: 0.5 seconds)')
    args = parser.parse_args()

    # Set random seed if provided (important for train/test separation)
    if args.seed is not None:
        random.seed(args.seed)
        print(f"  Random seed: {args.seed}")

    # Config generation mode
    if args.config is not None:
        # Use the explicit flags directly
        random_size = args.random_size
        random_placements = args.random_placements

        if args.objective:
            # YAML mode with objective (legacy behavior)
            print("=" * 50, file=sys.stderr)
            print("FROZEN LAKE - CONFIG GENERATION MODE (YAML)", file=sys.stderr)
            print("=" * 50, file=sys.stderr)

            output_path = Path(args.config_output) if args.config_output else Path("configs/generated_config.yaml")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"  Generating {args.config} random configurations...", file=sys.stderr)
            print(f"  Random sizes: {random_size}", file=sys.stderr)
            print(f"  Random placements: {random_placements}", file=sys.stderr)
            print(f"  Objective: {args.objective[:60]}...", file=sys.stderr)
            print(f"  Output: {output_path}", file=sys.stderr)

            config = generate_config_yaml(
                objective=args.objective,
                num_configs=args.config,
                output_path=output_path,
                random_size=random_size,
                random_placements=random_placements,
                timeout_steps=args.timeout_steps,
                synthesis_timeout=args.synthesis_timeout
            )

            print(f"\nGenerated {len(config['run_configuration'][0]['configurations'])} configurations", file=sys.stderr)
            print(f"Config saved to: {output_path}", file=sys.stderr)
        else:
            # JSON mode (consistent with other games)
            configs = generate_random_configs(
                num_configs=args.config,
                random_size=random_size,
                random_placements=random_placements
            )

            output = json.dumps(configs, indent=2)

            if args.config_output:
                with open(args.config_output, 'w') as f:
                    f.write(output)
                print(f"Generated {len(configs)} configurations to: {args.config_output}", file=sys.stderr)
            else:
                print(output)

        return

    # Replay mode
    if args.replay is not None:
        if not args.replay_config:
            print("Error: --replay-config is required in replay mode")
            sys.exit(1)

        # Load board config
        with open(args.replay_config) as f:
            board_config = json.load(f)

        # Parse trajectory
        trajectory = []
        for pos_str in args.replay.split(";"):
            if pos_str.strip():
                x, y = pos_str.strip().split(",")
                trajectory.append((int(x), int(y)))

        if not trajectory:
            print("Error: No positions in trajectory")
            sys.exit(1)

        # Run replay
        replay_trajectory(board_config, trajectory, delay=args.replay_delay)
        return

    # Create session directory
    if args.output:
        session_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path(f"Logs/tfrozen_lake/{timestamp}")
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "pos").mkdir(exist_ok=True)
    (session_dir / "neg").mkdir(exist_ok=True)

    # Check if in generation mode
    if args.gen is not None:
        # Automatic trace generation mode
        print("=" * 40)
        print("FROZEN LAKE (TUPLE) - TRACE GENERATION MODE")
        print("=" * 40)
        if args.random_size:
            print("  • Random board sizes enabled (3x3 to 5x5)")
        if args.random_placements:
            print("  • Random placements enabled")
        print(f"  • Target: {args.gen} positive + {args.gen} negative traces")
        print(f"  • Output: {session_dir}")

        # Create game instance for generation
        game = TupleFrozenLakeGame(session_dir=session_dir,
                                    random_size=args.random_size,
                                    random_placements=args.random_placements)

        # Generate traces
        game.generate_traces(args.gen)

        print(f"\nAll traces saved to: {session_dir}")

    else:
        # Interactive mode
        print("Welcome to Frozen Lake (Tuple Version)!")
        if args.random_size:
            print("  • Random board sizes enabled (3x3 to 5x5)")
        if args.random_placements:
            print("  • Random placements enabled")
        print("\nStarting session...")
        print("(Press any key to continue)")

        wait_for_key()

        # Track the first game's config for --save-config
        first_game_config = None

        # Play multiple game instances in the same session
        while True:
            game = TupleFrozenLakeGame(session_dir=session_dir,
                                        random_size=args.random_size,
                                        random_placements=args.random_placements)

            # Save the first game's config
            if first_game_config is None:
                first_game_config = game.get_config()

            quit_session = game.play()

            if quit_session:
                print("\n" + "=" * 40)
                print("Session ended. Thanks for playing!")
                break
            else:
                # Game ended naturally (win or loss), prompt for next game
                print("\n" + "=" * 40)
                print("Press any key to play again, or Ctrl+C to exit...")
                wait_for_key()

        # Save board config if requested
        if args.save_config and first_game_config:
            with open(args.save_config, 'w') as f:
                json.dump(first_game_config, f, indent=2)
            print(f"Board config saved to: {args.save_config}")


if __name__ == "__main__":
    main()
