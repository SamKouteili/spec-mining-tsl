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
from datetime import datetime
from pathlib import Path


class FrozenLakeGame:
    def __init__(self, size=4, session_dir=None):
        """Initialize a static Frozen Lake game grid."""
        self.size = size

        # Define a static 4x4 map
        # S = Start, F = Frozen, H = Hole, G = Goal
        self.grid = [
            ['S', 'F', 'F', 'F'],
            ['F', 'H', 'F', 'H'],
            ['F', 'F', 'F', 'H'],
            ['H', 'F', 'F', 'G']
        ]

        # Starting position
        self.player_x = 0
        self.player_y = 0

        # Goal position
        self.goal_x = 3
        self.goal_y = 3

        # Hole positions
        self.holes = [(1, 1), (3, 1), (3, 2), (0, 3)]

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
        print("\nControls: Arrow Keys (↑↓←→) | q: Quit\n")
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
        self.player_x = 0
        self.player_y = 0
        self.game_over = False
        self.won = False
        self.trace = []
        self._log_state()

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
    print("Welcome to Frozen Lake!")
    print("\nStarting session...")
    print("(Press any key to continue)")

    wait_for_key()

    # Create session directory once at the start
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(f"Logs/frozen_lake/{timestamp}")
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "pos").mkdir(exist_ok=True)
    (session_dir / "neg").mkdir(exist_ok=True)

    # Play multiple game instances in the same session
    while True:
        game = FrozenLakeGame(session_dir=session_dir)
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
