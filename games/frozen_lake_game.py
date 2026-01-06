#!/usr/bin/env python3
"""
Interactive Frozen Lake Game
Navigate the grid with arrow keys to reach the goal while avoiding holes.
Each game session logs timesteps as jsonl entries.
"""

import os
import json
import sys
import tty
import termios
import random
import argparse
from datetime import datetime
from pathlib import Path
from collections import deque


class FrozenLakeGame:
    def __init__(self, size=4, session_dir=None, random_size=False, random_placements=False):
        """Initialize a Frozen Lake game grid."""
        self.random_size = random_size
        self.random_placements = random_placements

        # Determine board size
        if random_size:
            self.size = random.randint(3, 6)
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
            if self.size == 4:
                self.goal_x = 3
                self.goal_y = 3
                self.holes = [(1, 1), (3, 1), (3, 2), (0, 3)]
            else:
                # For non-4x4 boards with fixed placements, use bottom-right goal
                self.goal_x = self.size - 1
                self.goal_y = self.size - 1
                # Place 3 holes in reasonable positions
                self.holes = [
                    (1, 1) if self.size > 1 else (0, 1),
                    (self.size - 1, 1) if self.size > 1 else (1, 0),
                    (self.size - 1, self.size - 2) if self.size > 2 else (1, 1)
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
        session_dir = Path(f"Logs/frozen_lake/{timestamp}")
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "pos").mkdir(exist_ok=True)
        (session_dir / "neg").mkdir(exist_ok=True)
        return session_dir

    def _log_state(self):
        """Log the current state as a jsonl entry."""
        state = {
            "playerX": self.player_x,
            "playerY": self.player_y,
            "goalX": self.goal_x,
            "goalY": self.goal_y,
        }

        # Add hole positions as constants
        for i, (hx, hy) in enumerate(self.holes):
            state[f"hole{i}X"] = hx
            state[f"hole{i}Y"] = hy

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
            self.size = random.randint(3, 6)

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

    def _random_walk_to_hole(self, max_steps=100):
        """Generate random walk that ends at a hole (negative trace)."""
        directions = ['up', 'down', 'left', 'right']
        path = []
        x, y = 0, 0

        for _ in range(max_steps):
            # Random direction
            direction = random.choice(directions)

            # Calculate new position
            if direction == 'up':
                ny = max(0, y - 1)
                nx = x
            elif direction == 'down':
                ny = min(self.size - 1, y + 1)
                nx = x
            elif direction == 'left':
                nx = max(0, x - 1)
                ny = y
            else:  # right
                nx = min(self.size - 1, x + 1)
                ny = y

            # Only add move if it actually changes position
            if (nx, ny) != (x, y):
                path.append(direction)
                x, y = nx, ny

                # Check if we hit a hole (success for negative trace)
                if (x, y) in self.holes:
                    return path

                # Don't let it reach the goal
                if (x, y) == (self.goal_x, self.goal_y):
                    # Backtrack - remove last move
                    if path:
                        path.pop()
                        # Recalculate position
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

        return None  # Failed to find hole in max_steps

    def _trace_to_key(self, trace):
        """Convert trace to hashable key for uniqueness checking."""
        return tuple(tuple(sorted(state.items())) for state in trace)

    def _execute_path(self, path):
        """Execute a path (list of directions) and return the trace."""
        for direction in path:
            self.move(direction)
        return self.trace.copy()

    def generate_traces(self, num_traces):
        """Generate num_traces unique positive and negative traces."""
        pos_traces = set()
        neg_traces = set()

        attempts = 0
        max_attempts = num_traces * 100  # Prevent infinite loop

        print(f"\nGenerating {num_traces} positive and {num_traces} negative traces...")

        while (len(pos_traces) < num_traces or len(neg_traces) < num_traces) and attempts < max_attempts:
            attempts += 1

            # Reset board for new instance
            self.reset()

            # Try to generate positive trace
            if len(pos_traces) < num_traces:
                # Use detours to create variety (50% chance)
                path = self._bfs_path(add_detours=True)
                if path is not None:
                    self.reset()  # Reset to execute the path
                    trace = self._execute_path(path)

                    # Validate this is actually a positive trace (detour didn't hit hole)
                    if self.won:
                        trace_key = self._trace_to_key(trace)
                        if trace_key not in pos_traces:
                            pos_traces.add(trace_key)
                            # Save the trace
                            self._save_trace_to_file(trace, is_positive=True,
                                                      trace_num=len(pos_traces))
                            print(f"  Generated positive trace {len(pos_traces)}/{num_traces}")
                    # else: detour hit a hole, discard this trace

            # Try to generate negative trace (new board instance)
            if len(neg_traces) < num_traces:
                self.reset()
                path = self._random_walk_to_hole()
                if path is not None:
                    self.reset()  # Reset to execute the path
                    trace = self._execute_path(path)
                    trace_key = self._trace_to_key(trace)

                    if trace_key not in neg_traces:
                        neg_traces.add(trace_key)
                        # Save the trace
                        self._save_trace_to_file(trace, is_positive=False,
                                                  trace_num=len(neg_traces))
                        print(f"  Generated negative trace {len(neg_traces)}/{num_traces}")

        if len(pos_traces) < num_traces:
            print(f"\nWarning: Only generated {len(pos_traces)} unique positive traces")
        if len(neg_traces) < num_traces:
            print(f"\nWarning: Only generated {len(neg_traces)} unique negative traces")

        print(f"\nTrace generation complete!")
        print(f"  Positive traces: {len(pos_traces)}")
        print(f"  Negative traces: {len(neg_traces)}")

    def _save_trace_to_file(self, trace, is_positive, trace_num):
        """Save a trace to file."""
        subdir = "pos" if is_positive else "neg"
        prefix = "pos_trace_" if is_positive else "neg_trace_"

        trace_dir = self.session_dir / subdir
        trace_file = trace_dir / f"{prefix}{trace_num}.jsonl"

        with open(trace_file, 'w') as f:
            for entry in trace:
                f.write(json.dumps(entry) + '\n')

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


def wait_for_key():
    """Wait for any keypress."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main():
    """Run the game session with multiple game instances."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Interactive Frozen Lake Game")
    parser.add_argument('--random-size', action='store_true',
                        help='Randomize board size (3x3 to 6x6) for each game instance')
    parser.add_argument('--random-placements', action='store_true',
                        help='Randomize goal and hole positions for each game instance')
    parser.add_argument('--gen', type=int, metavar='N',
                        help='Generate N positive and N negative traces automatically (non-interactive mode)')
    args = parser.parse_args()

    # Create session directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(f"Logs/frozen_lake/{timestamp}")
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "pos").mkdir(exist_ok=True)
    (session_dir / "neg").mkdir(exist_ok=True)

    # Check if in generation mode
    if args.gen is not None:
        # Automatic trace generation mode
        print("=" * 40)
        print("FROZEN LAKE - TRACE GENERATION MODE")
        print("=" * 40)
        if args.random_size:
            print("  • Random board sizes enabled (3x3 to 6x6)")
        if args.random_placements:
            print("  • Random placements enabled")
        print(f"  • Target: {args.gen} positive + {args.gen} negative traces")
        print(f"  • Output: {session_dir}")

        # Create game instance for generation
        game = FrozenLakeGame(session_dir=session_dir,
                               random_size=args.random_size,
                               random_placements=args.random_placements)

        # Generate traces
        game.generate_traces(args.gen)

        print(f"\nAll traces saved to: {session_dir}")

    else:
        # Interactive mode
        print("Welcome to Frozen Lake!")
        if args.random_size:
            print("  • Random board sizes enabled (3x3 to 6x6)")
        if args.random_placements:
            print("  • Random placements enabled")
        print("\nStarting session...")
        print("(Press any key to continue)")

        wait_for_key()

        # Play multiple game instances in the same session
        while True:
            game = FrozenLakeGame(session_dir=session_dir,
                                   random_size=args.random_size,
                                   random_placements=args.random_placements)
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


if __name__ == "__main__":
    main()
