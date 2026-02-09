#!/usr/bin/env python3
"""
Interactive Cliff Walking Game
Navigate the grid from start to goal while avoiding the cliff.
Each game session logs timesteps as jsonl entries.

State variables:
  - x, y: Player position (y=0 is bottom, y=height-1 is top)
  - goalX, goalY: Goal position
  - cliffXMin, cliffXMax: Cliff x-range (inclusive)
  - cliffHeight: 1-indexed cliff height (cliff occupies y < cliffHeight)

The cliff is a horizontal strip that the player must avoid.
Coordinate system: y=0 is BOTTOM (matches spec_generator convention).

Standard CliffWalking: 4 rows x 12 columns
  - Start: bottom-left (0, 0) in logged coords
  - Goal: bottom-right (cols-1, 0) in logged coords
  - Cliff: horizontal strip at y=0 between start and goal
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


class CliffWalkingGame:
    def __init__(self, width=12, height=4, session_dir=None,
                 random_size=False, random_height=False, var_moves=False):
        """Initialize a Cliff Walking game grid."""
        self.random_size = random_size
        self.random_height = random_height
        self.var_moves = var_moves

        # Determine board dimensions
        if random_size:
            # Width 3-12, height fixed at 4 for var_config
            self.width = random.randint(3, 12)
            self.height = 4  # Fixed height to allow cliff heights 1-3
        else:
            self.width = width
            self.height = height

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

        # Log initial state
        self._log_state()

    def _init_board(self):
        """Initialize the board grid and positions."""
        # Starting position (bottom-left corner)
        self.player_x = 0
        self.player_y = self.height - 1

        # Goal position (bottom-right corner)
        self.goal_x = self.width - 1
        self.goal_y = self.height - 1

        # Cliff region - rectangular area between start and goal
        # Default cliff height is 1 (just bottom row)
        if self.random_height:
            # Random cliff height: 1 to min(3, height-1)
            # Must leave at least one safe row above cliff for navigation
            # Max cliff height is 3 even if board is taller
            max_cliff_height = min(3, self.height - 1)
            self.cliff_height = random.randint(1, max(1, max_cliff_height))
        else:
            self.cliff_height = 1

        # Cliff region - horizontal strip between start and goal
        # Cliff x-range: always full width from 1 to width-2
        self.cliff_x_min = 1
        self.cliff_x_max = self.width - 2

        # Cliff y-range (internal coords, y=0 is top)
        # For logging, we flip so y=0 is bottom
        self.cliff_y_min_internal = self.height - self.cliff_height  # top of cliff (internal)
        self.cliff_y_max_internal = self.height - 1  # bottom of cliff (internal)

        # Build grid based on positions
        self.grid = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if x == 0 and y == self.height - 1:
                    row.append('S')  # Start
                elif x == self.goal_x and y == self.goal_y:
                    row.append('G')  # Goal
                elif self._is_cliff(x, y):
                    row.append('C')  # Cliff
                else:
                    row.append('.')  # Safe
            self.grid.append(row)

    def _is_cliff(self, x, y):
        """Check if a position is part of the cliff (internal coordinates)."""
        return (self.cliff_x_min <= x <= self.cliff_x_max and
                self.cliff_y_min_internal <= y <= self.cliff_y_max_internal)

    def _create_session_dir(self):
        """Create a new directory for this game session."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path(f"Logs/cliff_walking/{timestamp}")
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "pos").mkdir(exist_ok=True)
        (session_dir / "neg").mkdir(exist_ok=True)
        return session_dir

    def _log_state(self):
        """Log the current state as a jsonl entry.

        Coordinates are flipped so y=0 is bottom (matches spec_generator).
        """
        # Flip Y: logged_y = (height - 1) - internal_y
        logged_y = (self.height - 1) - self.player_y
        logged_goal_y = (self.height - 1) - self.goal_y

        state = {
            "x": self.player_x,
            "y": logged_y,
            "goalX": self.goal_x,
            "goalY": logged_goal_y,
            "cliffXMin": self.cliff_x_min,
            "cliffXMax": self.cliff_x_max,
            # cliffHeight is 1-indexed: cliff occupies y < cliffHeight
            # e.g., cliffHeight=1 means y=0 is cliff, cliffHeight=2 means y∈[0,1] is cliff
            "cliffHeight": self.cliff_height,
        }
        self.trace.append(state)

    def display(self):
        """Display the current game state."""
        os.system('clear' if os.name == 'posix' else 'cls')

        # ANSI color codes
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        RESET = '\033[0m'

        print("=" * 50)
        print("CLIFF WALKING GAME")
        print("=" * 50)
        print(f"\nBoard Size: {self.width}x{self.height}")
        cliff_height = self.cliff_y_max_internal - self.cliff_y_min_internal + 1
        print(f"Cliff Height: {cliff_height}")
        print("Controls: Arrow Keys or WASD | q: Quit")
        print("Legend: P=Player S=Start G=Goal C=Cliff .=Safe\n")

        # Display grid
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                if x == self.player_x and y == self.player_y:
                    cell = f"{YELLOW} P {RESET}"
                else:
                    tile = self.grid[y][x]
                    if tile == 'C':
                        cell = f"{RED} C {RESET}"
                    elif tile == 'G':
                        cell = f"{GREEN} G {RESET}"
                    elif tile == 'S':
                        cell = f"{BLUE} S {RESET}"
                    else:
                        cell = f" {tile} "
                row += cell
            print(row)

        # Show logged coordinates (y=0 is bottom)
        logged_y = (self.height - 1) - self.player_y
        logged_goal_y = (self.height - 1) - self.goal_y
        print(f"\nPosition: ({self.player_x}, {logged_y})")
        print(f"Goal: ({self.goal_x}, {logged_goal_y})")
        print(f"Moves: {len(self.trace) - 1}")

        if self.game_over:
            if self.won:
                print(f"\n{GREEN}YOU WON! You reached the goal!{RESET}")
            else:
                print(f"\n{RED}GAME OVER! You fell off the cliff!{RESET}")

    def move(self, direction):
        """Move the player in the specified direction.

        With --var-moves enabled, uses non-standard movement functions:
        - Left:  x_new = x - 1
        - Right: x_new = (x * 2) + 1
        - Up:    y_new = y - 2 (internal coords, so +2 in logged coords)
        - Down:  y_new = y + 1 (same as standard)
        """
        if self.game_over:
            return

        new_x, new_y = self._compute_next_pos(self.player_x, self.player_y, direction)

        # Update position
        self.player_x = new_x
        self.player_y = new_y

        # Log the new state
        self._log_state()

        # Check win/lose conditions
        if (self.player_x, self.player_y) == (self.goal_x, self.goal_y):
            self.game_over = True
            self.won = True
        elif self._is_cliff(self.player_x, self.player_y):
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

            # WASD support
            if ch.lower() == 'w':
                return 'up'
            elif ch.lower() == 's':
                return 'down'
            elif ch.lower() == 'a':
                return 'left'
            elif ch.lower() == 'd':
                return 'right'

            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def reset(self):
        """Reset the game state for a new game instance."""
        # Re-randomize board if flags are set
        if self.random_size:
            self.width = random.randint(3, 12)
            self.height = 4  # Fixed height to allow cliff heights 1-3

        # Re-initialize board (will randomize cliff height if flag is set)
        self._init_board()

        # Reset game state
        self.game_over = False
        self.won = False
        self.trace = []
        self._log_state()

    def _soft_reset(self):
        """Reset player position and trace WITHOUT re-randomizing board."""
        self.player_x = 0
        self.player_y = self.height - 1
        self.game_over = False
        self.won = False
        self.trace = []
        self._log_state()

    def _compute_next_pos(self, x, y, direction):
        """Compute next position given current position and direction.

        Handles both standard and variant movement functions.
        Standard: Returns (new_x, new_y) clamped to board bounds.
        Variant: Returns current position if move would go out of bounds (no clamping).
        """
        if self.var_moves:
            # Variant movement - no clamping, move fails if out of bounds
            if direction == 'up':
                ny = y - 2
                nx = x
            elif direction == 'down':
                ny = y + 1
                nx = x
            elif direction == 'left':
                nx = x - 1
                ny = y
            elif direction == 'right':
                nx = (x * 2) + 1
                ny = y
            else:
                nx, ny = x, y
            # If out of bounds, stay in place
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                return x, y
            return nx, ny
        else:
            # Standard movement - clamp to bounds
            if direction == 'up':
                ny = max(0, y - 1)
                nx = x
            elif direction == 'down':
                ny = min(self.height - 1, y + 1)
                nx = x
            elif direction == 'left':
                nx = max(0, x - 1)
                ny = y
            elif direction == 'right':
                nx = min(self.width - 1, x + 1)
                ny = y
            else:
                nx, ny = x, y
            return nx, ny

    def _bfs_path(self, add_detours=False):
        """Find path from start to goal using BFS, avoiding cliff.

        Args:
            add_detours: If True, add random exploration moves to create variety
        """
        start = (0, self.height - 1)
        goal = (self.goal_x, self.goal_y)

        if start == goal:
            return []

        queue = deque([(start, [])])
        visited = {start}

        while queue:
            (x, y), path = queue.popleft()

            # Randomize direction order for variety
            directions = ['up', 'down', 'left', 'right']
            random.shuffle(directions)

            # Try all four directions
            for direction in directions:
                nx, ny = self._compute_next_pos(x, y, direction)

                # Check bounds
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue

                # Skip cliff and visited
                if self._is_cliff(nx, ny) or (nx, ny) in visited:
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
        For var_moves, uses asymmetric return patterns since movements aren't symmetric.
        """
        if num_detours is None:
            num_detours = random.randint(1, 3)

        path = base_path.copy()

        for _ in range(num_detours):
            # Pick random position to insert detour
            insert_pos = random.randint(0, len(path))

            if self.var_moves:
                # Var_moves has asymmetric movements, use specific return patterns
                # Vertical: up (y-=2), down (y+=1)
                #   Pattern: up, down, down (y: -2, +1, +1 = 0)
                #   Pattern: down, down, up (y: +1, +1, -2 = 0)
                # Horizontal detours are tricky since left=x*2+1 doesn't have easy inverse
                detour_patterns = [
                    ['up', 'down', 'down'],
                    ['down', 'down', 'up'],
                    ['up', 'down', 'down', 'up', 'down', 'down'],  # Double vertical detour
                ]
                detour_moves = random.choice(detour_patterns)
            else:
                # Standard symmetric movements
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

    def _bfs_to_adjacent_goal(self):
        """BFS to cell adjacent to goal (same column, different row)."""
        start = (0, self.height - 1)
        target_y = self.goal_y - 1
        if target_y < 0:
            return None

        target = (self.goal_x, target_y)

        queue = deque([(start, [])])
        visited = {start}

        while queue:
            (x, y), path = queue.popleft()

            directions = ['up', 'down', 'left', 'right']
            random.shuffle(directions)

            for direction in directions:
                nx, ny = self._compute_next_pos(x, y, direction)

                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if self._is_cliff(nx, ny) or (nx, ny) in visited:
                    continue
                if (nx, ny) == (self.goal_x, self.goal_y):
                    continue

                new_path = path + [direction]

                if (nx, ny) == target:
                    return new_path

                visited.add((nx, ny))
                queue.append(((nx, ny), new_path))

        return None

    def _bfs_explore_boundary(self):
        """BFS exploration along boundary regions of the grid.

        Explores the board via BFS, collecting reachable cells near boundaries.
        Returns path to a randomly selected boundary cell with continuation.
        """
        start = (0, self.height - 1)
        pre_cliff_y = self.cliff_y_min_internal - 1

        if pre_cliff_y < 0:
            return None

        queue = deque([(start, [])])
        visited = {start}
        boundary_cells = []

        while queue:
            (x, y), path = queue.popleft()

            if y == pre_cliff_y:
                boundary_cells.append((x, y, path))

            for direction in ['up', 'down', 'left', 'right']:
                nx, ny = self._compute_next_pos(x, y, direction)

                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if self._is_cliff(nx, ny) or (nx, ny) in visited:
                    continue
                if (nx, ny) == (self.goal_x, self.goal_y):
                    continue

                visited.add((nx, ny))
                queue.append(((nx, ny), path + [direction]))

        if not boundary_cells:
            return None

        goalx_cell = None
        for cell in boundary_cells:
            if cell[0] == self.goal_x:
                goalx_cell = cell
                break

        if goalx_cell and random.random() < 0.4:
            x, y, path_to_cell = goalx_cell
            return path_to_cell + ['left', 'down']
        else:
            x, y, path_to_cell = random.choice(boundary_cells)
            return path_to_cell + ['down']

    def _bfs_to_boundary(self, target_x=None, explore_far=True):
        """BFS to boundary region with optional extended exploration."""
        start = (0, self.height - 1)
        pre_cliff_y = self.cliff_y_min_internal - 1
        if pre_cliff_y < 0:
            return None

        if target_x is None:
            target_x = random.randint(self.cliff_x_min, self.cliff_x_max)

        if explore_far:
            target = (self.goal_x, pre_cliff_y)
        else:
            target = (target_x, pre_cliff_y)

        queue = deque([(start, [])])
        visited = {start}
        path_to_target = None

        while queue:
            (x, y), path = queue.popleft()

            directions = ['up', 'down', 'left', 'right']
            random.shuffle(directions)

            for direction in directions:
                nx, ny = self._compute_next_pos(x, y, direction)

                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if self._is_cliff(nx, ny) or (nx, ny) in visited:
                    continue
                if (nx, ny) == (self.goal_x, self.goal_y):
                    continue

                new_path = path + [direction]

                if (nx, ny) == target:
                    path_to_target = new_path
                    break

                visited.add((nx, ny))
                queue.append(((nx, ny), new_path))

            if path_to_target:
                break

        if path_to_target is None:
            return None

        path = path_to_target.copy()

        if explore_far and not self.var_moves:
            steps_left = self.goal_x - target_x
            for _ in range(steps_left):
                path.append('left')
        elif explore_far and self.var_moves:
            backtrack_path = self._bfs_backtrack((self.goal_x, pre_cliff_y),
                                                  (target_x, pre_cliff_y))
            if backtrack_path:
                path.extend(backtrack_path)

        path.append('down')
        return path

    def _bfs_backtrack(self, start, goal):
        """BFS from start to goal for backtracking with var_moves."""
        if start == goal:
            return []

        queue = deque([(start, [])])
        visited = {start}

        while queue:
            (x, y), path = queue.popleft()

            for direction in ['up', 'down', 'left', 'right']:
                nx, ny = self._compute_next_pos(x, y, direction)

                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if self._is_cliff(nx, ny) or (nx, ny) in visited:
                    continue

                new_path = path + [direction]

                if (nx, ny) == goal:
                    return new_path

                visited.add((nx, ny))
                queue.append(((nx, ny), new_path))

        return None

    def _trace_to_key(self, trace):
        """Convert trace to hashable key for uniqueness checking."""
        return tuple(tuple((k, v) for k, v in sorted(state.items())) for state in trace)

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

        base_path = self._bfs_path(add_detours=False)
        if base_path is None:
            print("ERROR: No valid path found!")
            return

        print(f"  Base path length: {len(base_path)}")

        while (len(pos_traces) < num_traces or len(neg_traces) < num_traces) and attempts < max_attempts:
            attempts += 1

            if len(pos_traces) < num_traces:
                self.reset()
                roll = random.random()

                if roll < 0.4:
                    path = self._bfs_path(add_detours=False)
                    trace_type = "direct"
                elif roll < 0.7:
                    path = self._bfs_path(add_detours=True)
                    trace_type = "detour"
                else:
                    path = self._bfs_to_adjacent_goal()
                    if path is not None:
                        self._soft_reset()
                        self._execute_path(path)
                        current = (self.player_x, self.player_y)
                        goal = (self.goal_x, self.goal_y)
                        continuation = self._bfs_backtrack(current, goal)
                        if continuation:
                            path = path + continuation
                        else:
                            path = None
                    trace_type = "extended"

                if path is not None:
                    self._soft_reset()
                    trace = self._execute_path(path)

                    if self.won:
                        trace_key = self._trace_to_key(trace)
                        if trace_key not in pos_traces:
                            pos_traces.add(trace_key)
                            self._save_trace_to_file(trace, is_positive=True,
                                                      trace_num=len(pos_traces))
                            print(f"  BFS trace {len(pos_traces)}/{num_traces} ({trace_type})")

            if len(neg_traces) < num_traces:
                self.reset()
                path = self._bfs_explore_boundary()

                if path is not None:
                    self._soft_reset()
                    trace = self._execute_path(path)

                    at_cliff = self._is_cliff(self.player_x, self.player_y)
                    if at_cliff:
                        trace_key = self._trace_to_key(trace)
                        if trace_key not in neg_traces:
                            neg_traces.add(trace_key)
                            self._save_trace_to_file(trace, is_positive=False,
                                                      trace_num=len(neg_traces))
                            print(f"  BFS trace {len(neg_traces)}/{num_traces} (boundary)")

        if len(pos_traces) < num_traces:
            print(f"\nWarning: Only found {len(pos_traces)} unique goal paths")
        if len(neg_traces) < num_traces:
            print(f"\nWarning: Only found {len(neg_traces)} unique boundary paths")

        print(f"\nBFS exploration complete!")
        print(f"  Goal paths: {len(pos_traces)}")
        print(f"  Boundary paths: {len(neg_traces)}")

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
            Configuration dictionary compatible with spec_generator
        """
        if name is None:
            name = f"{self.width}x{self.height}_config"

        return {
            "name": name,
            "width": self.width,
            "height": self.height,
            "grid_size": self.width,  # For spec_generator compatibility
            "grid_rows": self.height,
            "start_pos": {"x": 0, "y": 0},
            "goal_pos": {"x": self.goal_x, "y": 0},
            "cliff_min": self.cliff_x_min,
            "cliff_max": self.cliff_x_max,
            "cliff_height": self.cliff_height,
            "var_moves": self.var_moves,
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


def bfs_reachable(width: int, height: int, start: tuple, goal: tuple,
                  cliff_x_min: int, cliff_x_max: int, cliff_height: int,
                  var_moves: bool = False) -> bool:
    """
    Check if goal is reachable from start using BFS, avoiding cliff.

    Args:
        width: Grid width
        height: Grid height
        start: Starting position (x, y) in logged coords (y=0 is bottom)
        goal: Goal position (x, y) in logged coords
        cliff_x_min: Cliff x-range minimum
        cliff_x_max: Cliff x-range maximum
        cliff_height: Cliff height (cliff occupies y < cliff_height)
        var_moves: Whether to use variant movement functions

    Returns:
        True if goal is reachable from start
    """
    def is_cliff(x, y):
        return cliff_x_min <= x <= cliff_x_max and y < cliff_height

    def get_neighbors(x, y):
        """Get valid neighbors based on movement type."""
        neighbors = []
        if var_moves:
            # Variant movements
            moves = [
                (x - 1, y),           # LEFT: x -= 1
                (x, y - 1),           # DOWN: y -= 1 (in logged coords)
                ((x * 2) + 1, y),     # RIGHT: x = (x * 2) + 1
                (x, y + 2),           # UP: y += 2 (in logged coords)
            ]
            for nx, ny in moves:
                if 0 <= nx < width and 0 <= ny < height:
                    neighbors.append((nx, ny))
        else:
            # Standard movements
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    neighbors.append((nx, ny))
        return neighbors

    if start == goal:
        return True
    if is_cliff(*start) or is_cliff(*goal):
        return False

    visited = {start}
    queue = deque([start])

    while queue:
        x, y = queue.popleft()

        for nx, ny in get_neighbors(x, y):
            if (nx, ny) == goal:
                return True
            if (nx, ny) not in visited and not is_cliff(nx, ny):
                visited.add((nx, ny))
                queue.append((nx, ny))

    return False


def generate_random_configs(
    num_configs: int,
    random_size: bool = False,
    random_height: bool = False,
    var_moves: bool = False,
    seed: int = None
) -> list:
    """
    Generate random board configurations with guaranteed reachability.

    Args:
        num_configs: Number of configurations to generate
        random_size: Whether to randomize board width (3-12)
        random_height: Whether to randomize cliff height (1-3)
        var_moves: Whether to use variant movement functions
        seed: Random seed for reproducibility

    Returns:
        List of configuration dictionaries compatible with spec_generator
    """
    if seed is not None:
        random.seed(seed)

    configs = []
    seen = set()
    board_height = 4  # Always 4 to allow cliff heights 1-3

    # If neither random flag is set, generate fixed configs
    if not random_size and not random_height:
        # Fixed configuration: 12x4 board, cliff height 1
        width = 12
        cliff_height = 1
        goal_x = width - 1
        cliff_x_min = 1
        cliff_x_max = width - 2

        for i in range(num_configs):
            configs.append({
                "name": f"config_{i + 1}",
                "width": width,
                "height": board_height,
                "grid_size": width,
                "grid_rows": board_height,
                "start_pos": {"x": 0, "y": 0},
                "goal_pos": {"x": goal_x, "y": 0},
                "cliff_min": cliff_x_min,
                "cliff_max": cliff_x_max,
                "cliff_height": cliff_height,
                "var_moves": var_moves,
            })
        return configs

    # Generate varied configs based on flags
    all_configs = []

    # Determine ranges based on flags
    width_range = range(3, 13) if random_size else [12]
    height_range = range(1, 4) if random_height else [1]

    for width in width_range:
        for cliff_height in height_range:
            goal_x = width - 1
            cliff_x_min = 1
            cliff_x_max = width - 2

            # For width 3, cliff_x_min = 1, cliff_x_max = 1
            if cliff_x_max < cliff_x_min:
                continue

            start = (0, 0)
            goal = (goal_x, 0)

            # Check reachability
            if not bfs_reachable(width, board_height, start, goal,
                                  cliff_x_min, cliff_x_max, cliff_height,
                                  var_moves=var_moves):
                continue

            all_configs.append({
                "name": f"config_{len(all_configs) + 1}",
                "width": width,
                "height": board_height,
                "grid_size": width,
                "grid_rows": board_height,
                "start_pos": {"x": 0, "y": 0},
                "goal_pos": {"x": goal_x, "y": 0},
                "cliff_min": cliff_x_min,
                "cliff_max": cliff_x_max,
                "cliff_height": cliff_height,
                "var_moves": var_moves,
            })

    # Return requested number of configs (or all if fewer available)
    if num_configs >= len(all_configs):
        return all_configs
    else:
        return random.sample(all_configs, num_configs)


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
    parser = argparse.ArgumentParser(description="Interactive Cliff Walking Game")
    parser.add_argument('--random-size', action='store_true',
                        help='Randomize board width (3-12), height fixed at 4 for each game instance')
    parser.add_argument('--random-height', '--random-placements', action='store_true',
                        dest='random_height',
                        help='Randomize cliff height for each game instance')
    parser.add_argument('--gen', type=int, metavar='N',
                        help='Generate N positive and N negative traces automatically (non-interactive mode)')
    parser.add_argument('--output', type=str, metavar='DIR',
                        help='Output directory for generated traces (default: Logs/cliff_walking/<timestamp>)')
    parser.add_argument('--width', type=int, default=12,
                        help='Board width (default: 12)')
    parser.add_argument('--height', type=int, default=4,
                        help='Board height (default: 4)')
    parser.add_argument('--var-moves', action='store_true',
                        help='Use variant movement functions: left=x-1, right=(x*2)+1, up=y+2, down=y-1')
    parser.add_argument('--config', type=int, metavar='N',
                        help='Generate N board configurations as JSON (combine with --random-size/--random-height/--var-moves)')
    parser.add_argument('--config-output', type=str, metavar='FILE',
                        help='Output JSON file for --config mode (default: stdout)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Config generation mode
    if args.config is not None:
        configs = generate_random_configs(
            num_configs=args.config,
            random_size=args.random_size,
            random_height=args.random_height,
            var_moves=args.var_moves,
            seed=args.seed
        )

        output = json.dumps(configs, indent=2)

        if args.config_output:
            with open(args.config_output, 'w') as f:
                f.write(output)
            print(f"Generated {len(configs)} configurations to: {args.config_output}", file=sys.stderr)
        else:
            print(output)
        return

    # Create session directory
    if args.output:
        session_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path(f"Logs/cliff_walking/{timestamp}")
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "pos").mkdir(exist_ok=True)
    (session_dir / "neg").mkdir(exist_ok=True)

    # Check if in generation mode
    if args.gen is not None:
        # Automatic trace generation mode
        print("=" * 50)
        print("CLIFF WALKING - TRACE GENERATION MODE")
        print("=" * 50)
        if args.random_size:
            print("  - Random board sizes enabled (width 3-12, height 4)")
        else:
            print(f"  - Board size: {args.width}x{args.height}")
        if args.random_height:
            print("  - Random cliff height enabled")
        if args.var_moves:
            print("  - Variant moves enabled: left=x-1, right=(x*2)+1, up=y+2, down=y-1")
        print(f"  - Target: {args.gen} positive + {args.gen} negative traces")
        print(f"  - Output: {session_dir}")

        # Create game instance for generation
        game = CliffWalkingGame(width=args.width, height=args.height,
                                 session_dir=session_dir,
                                 random_size=args.random_size,
                                 random_height=args.random_height,
                                 var_moves=args.var_moves)

        # Generate traces
        game.generate_traces(args.gen)

        print(f"\nAll traces saved to: {session_dir}")

    else:
        # Interactive mode
        print("Welcome to Cliff Walking!")
        if args.random_size:
            print("  - Random board sizes enabled (width 3-12, height 4)")
        else:
            print(f"  - Board size: {args.width}x{args.height}")
        if args.random_height:
            print("  - Random cliff height enabled")
        if args.var_moves:
            print("  - Variant moves enabled: left=x-1, right=(x*2)+1, up=y+2, down=y-1")
        print("\nStarting session...")
        print("(Press any key to continue)")

        wait_for_key()

        # Play multiple game instances in the same session
        while True:
            game = CliffWalkingGame(width=args.width, height=args.height,
                                     session_dir=session_dir,
                                     random_size=args.random_size,
                                     random_height=args.random_height,
                                     var_moves=args.var_moves)
            quit_session = game.play()

            if quit_session:
                print("\n" + "=" * 50)
                print("Session ended. Thanks for playing!")
                break
            else:
                # Game ended naturally (win or loss), prompt for next game
                print("\n" + "=" * 50)
                print("Press any key to play again, or Ctrl+C to exit...")
                wait_for_key()


if __name__ == "__main__":
    main()
