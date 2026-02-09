#!/usr/bin/env python3
"""
Interactive Taxi Game (Tuple Version)
Navigate the grid to pick up a passenger and deliver them to the destination.
Each game session logs timesteps as jsonl entries with tuple-based coordinates.

This version uses tuples for positions:
  - taxi: [x, y]
  - passenger: [x, y] (moves with taxi when in taxi)
  - destination: [x, y]

Standard Taxi: 5x5 grid with 4 designated locations
  - R (Red): (0, 0)
  - G (Green): (4, 0)
  - Y (Yellow): (0, 4)
  - B (Blue): (3, 4)

Game mechanics:
  - Automatic pickup when taxi reaches passenger
  - Automatic dropoff when taxi reaches destination with passenger (WIN)
  - LOSE if taxi with passenger visits wrong colored location

Actions: Arrow keys to move
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


class TupleTaxiGame:
    # Standard designated locations (name -> (x, y)) for 5x5 board
    STANDARD_LOCATIONS = {
        'R': (0, 0),
        'G': (4, 0),
        'Y': (0, 4),
        'B': (3, 4)
    }

    # Standard walls for 5x5 board: list of ((x1,y1), (x2,y2)) blocked edges
    # These match the original Taxi game layout
    STANDARD_WALLS = [
        ((0, 0), (1, 0)),  # Wall between R and adjacent cell
        ((0, 1), (1, 1)),  # Wall continues down
        ((1, 3), (2, 3)),  # Wall in middle-bottom area
        ((1, 4), (2, 4)),  # Wall continues
        ((2, 0), (3, 0)),  # Wall between middle and G area
        ((2, 1), (3, 1)),  # Wall continues down
    ]

    def __init__(self, size=5, session_dir=None, random_size=False,
                 random_pos=False, random_config=False, include_passenger_state=False):
        """Initialize a Taxi game grid.

        Args:
            size: Board size (default 5x5)
            session_dir: Directory to save traces
            random_size: Randomize board size (4-7)
            random_pos: Randomize physical locations of R/G/B/Y on grid
                        (but pickup=Red, dropoff=Green always)
            random_config: Randomize which colors are pickup/dropoff
                          (e.g., pickup=Blue, dropoff=Yellow)
            include_passenger_state: If True, include passengerInTaxi in traces.
                                     Default False - omit for cleaner temporal mining.
        """
        self.random_size = random_size
        self.random_pos = random_pos
        self.random_config = random_config
        self.include_passenger_state = include_passenger_state

        # Determine board size
        if random_size:
            self.size = random.randint(4, 7)
        else:
            self.size = size

        # Initialize board and positions
        self._init_board()

        # Game state
        self.game_over = False
        self.won = False
        self.passenger_in_taxi = False
        self.trace = []

        # Use provided session directory or create new one
        if session_dir is None:
            self.session_dir = self._create_session_dir()
        else:
            self.session_dir = session_dir

        # Log initial state
        self._log_state()

    def _init_board(self):
        """Initialize the board grid, locations, walls, and positions."""
        # Generate locations based on flags
        if self.random_pos or self.size != 5:
            self._generate_random_locations()
        else:
            self.locations = dict(self.STANDARD_LOCATIONS)

        # Generate walls
        if self.random_pos or self.size != 5:
            self._generate_random_walls()
        else:
            self.walls = set()
            for (x1, y1), (x2, y2) in self.STANDARD_WALLS:
                self.walls.add(((x1, y1), (x2, y2)))
                self.walls.add(((x2, y2), (x1, y1)))  # Bidirectional

        # Determine destination and passenger location
        if self.random_config:
            # Random pickup and dropoff from all 4 colors
            colors = ['R', 'G', 'Y', 'B']
            self.passenger_loc_name, self.destination_name = random.sample(colors, 2)
        else:
            # Fixed: pickup always at Red, dropoff always at Green
            self.passenger_loc_name = 'R'
            self.destination_name = 'G'

        self.dest_x, self.dest_y = self.locations[self.destination_name]
        self.passenger_x, self.passenger_y = self.locations[self.passenger_loc_name]

        # Starting taxi position - must not be at ANY colored location (for clean safety specs)
        self._place_taxi_avoiding_all_colored_locations()

    def _place_taxi_avoiding_all_colored_locations(self):
        """Place taxi at a position that is not at ANY colored location (R/G/Y/B).

        This ensures positive traces never start at a wrong-colored position,
        which would break the safety biconditional pattern.
        """
        # Forbid ALL colored locations - taxi should start at a "neutral" cell
        forbidden = set()
        for name, (lx, ly) in self.locations.items():
            forbidden.add((lx, ly))

        if self.random_pos or self.random_size or self.random_config:
            # Random start position avoiding all colored locations
            attempts = 0
            while attempts < 100:
                self.taxi_x = random.randint(0, self.size - 1)
                self.taxi_y = random.randint(0, self.size - 1)
                if (self.taxi_x, self.taxi_y) not in forbidden:
                    break
                attempts += 1
            else:
                # Fallback: couldn't find non-colored cell (very small board)
                # Pick any cell not at passenger/destination
                fallback_forbidden = {
                    (self.passenger_x, self.passenger_y),
                    (self.dest_x, self.dest_y)
                }
                for y in range(self.size):
                    for x in range(self.size):
                        if (x, y) not in fallback_forbidden:
                            self.taxi_x, self.taxi_y = x, y
                            return
        else:
            # Fixed start: find first valid position not at any colored location
            # Standard layout: R=(0,0), G=(4,0), Y=(0,4), B=(3,4)
            placed = False
            for y in range(self.size):
                for x in range(self.size):
                    if (x, y) not in forbidden:
                        self.taxi_x, self.taxi_y = x, y
                        placed = True
                        break
                if placed:
                    break

    def _generate_random_locations(self):
        """Generate 4 random designated locations for the grid."""
        self.locations = {}
        used_positions = set()

        loc_names = ['R', 'G', 'Y', 'B']
        for name in loc_names:
            while True:
                x = random.randint(0, self.size - 1)
                y = random.randint(0, self.size - 1)
                if (x, y) not in used_positions:
                    self.locations[name] = (x, y)
                    used_positions.add((x, y))
                    break

    def _generate_random_walls(self):
        """Generate 3 random walls for the grid."""
        self.walls = set()
        num_walls = 3

        # Collect all possible wall positions (horizontal and vertical edges)
        possible_walls = []

        # Horizontal walls (between vertically adjacent cells)
        for x in range(self.size):
            for y in range(self.size - 1):
                possible_walls.append(((x, y), (x, y + 1)))

        # Vertical walls (between horizontally adjacent cells)
        for x in range(self.size - 1):
            for y in range(self.size):
                possible_walls.append(((x, y), (x + 1, y)))

        # Randomly select walls, ensuring the grid remains connected
        random.shuffle(possible_walls)
        walls_added = 0

        for wall in possible_walls:
            if walls_added >= num_walls:
                break

            # Temporarily add wall and check connectivity
            (x1, y1), (x2, y2) = wall
            self.walls.add(((x1, y1), (x2, y2)))
            self.walls.add(((x2, y2), (x1, y1)))

            if self._is_grid_connected():
                walls_added += 1
            else:
                # Remove wall if it disconnects the grid
                self.walls.discard(((x1, y1), (x2, y2)))
                self.walls.discard(((x2, y2), (x1, y1)))

    def _is_grid_connected(self):
        """Check if all cells are reachable from (0,0) with current walls."""
        visited = set()
        queue = deque([(0, 0)])
        visited.add((0, 0))

        while queue:
            x, y = queue.popleft()

            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy

                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    continue
                if (nx, ny) in visited:
                    continue
                if self._is_wall_between((x, y), (nx, ny)):
                    continue

                visited.add((nx, ny))
                queue.append((nx, ny))

        return len(visited) == self.size * self.size

    def _is_wall_between(self, pos1, pos2):
        """Check if there's a wall between two adjacent positions."""
        return (pos1, pos2) in self.walls

    def _create_session_dir(self):
        """Create a new directory for this game session."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path(f"Logs/ttaxi/{timestamp}")
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "pos").mkdir(exist_ok=True)
        (session_dir / "neg").mkdir(exist_ok=True)
        return session_dir

    def _log_state(self):
        """Log the current state as a jsonl entry with tuple-based coordinates."""
        state = {
            "player": [self.taxi_x, self.taxi_y],
            "destination": [self.dest_x, self.dest_y],
        }

        # Passenger position: constant pickup location (does NOT move with taxi)
        # This reflects that "passenger" is where they wait to be picked up
        state["passenger"] = [self.passenger_x, self.passenger_y]

        # Only add passengerInTaxi if flag is set (default: exclude for cleaner mining)
        if self.include_passenger_state:
            state["passengerInTaxi"] = self.passenger_in_taxi

        # Add all designated locations as constants
        for name, (lx, ly) in self.locations.items():
            state[f"loc{name}"] = [lx, ly]

        self.trace.append(state)

    def display(self):
        """Display the current game state."""
        os.system('clear' if os.name == 'posix' else 'cls')

        # ANSI color codes
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        CYAN = '\033[96m'
        RESET = '\033[0m'
        GRAY = '\033[90m'

        loc_colors = {'R': RED, 'G': GREEN, 'Y': YELLOW, 'B': BLUE}

        print("=" * 50)
        print("TAXI GAME")
        print("=" * 50)
        print(f"\nBoard Size: {self.size}x{self.size}")
        print("Controls: Arrow Keys (move) | q: Quit")
        print(f"Pickup: {self.passenger_loc_name} | Destination: {self.destination_name}")
        print(f"Passenger in taxi: {'Yes' if self.passenger_in_taxi else 'No'}\n")

        # Display grid with walls
        for y in range(self.size):
            # Top wall row for this cell row
            wall_row = ""
            for x in range(self.size):
                wall_row += "+"
                # Check for wall above this cell
                if y > 0 and self._is_wall_between((x, y-1), (x, y)):
                    wall_row += f"{GRAY}--{RESET}"
                else:
                    wall_row += "  "
            wall_row += "+"
            print(wall_row)

            # Cell row
            row = ""
            for x in range(self.size):
                # Left wall
                if x > 0 and self._is_wall_between((x-1, y), (x, y)):
                    row += f"{GRAY}|{RESET}"
                else:
                    row += " "

                # Cell content
                is_taxi = (x == self.taxi_x and y == self.taxi_y)
                loc_here = None
                for name, (lx, ly) in self.locations.items():
                    if (x, y) == (lx, ly):
                        loc_here = name
                        break
                is_passenger = (not self.passenger_in_taxi and
                               x == self.passenger_x and y == self.passenger_y)
                is_dest = (x == self.dest_x and y == self.dest_y)

                if is_taxi:
                    if self.passenger_in_taxi:
                        cell = f"{CYAN}T*{RESET}"  # Taxi with passenger
                    else:
                        cell = f"{CYAN}T {RESET}"  # Empty taxi
                elif loc_here:
                    color = loc_colors.get(loc_here, RESET)
                    if is_passenger and is_dest:
                        cell = f"{color}{loc_here}!{RESET}"  # Passenger + dest
                    elif is_passenger:
                        cell = f"{color}{loc_here}P{RESET}"  # Passenger here
                    elif is_dest:
                        cell = f"{color}{loc_here}D{RESET}"  # Destination
                    else:
                        cell = f"{color}{loc_here} {RESET}"
                else:
                    cell = ". "
                row += cell

            # Right edge
            row += " "
            print(row)

        # Bottom wall row
        wall_row = ""
        for x in range(self.size):
            wall_row += "+  "
        wall_row += "+"
        print(wall_row)

        print(f"\nTaxi: ({self.taxi_x}, {self.taxi_y})")
        print(f"Moves: {len(self.trace) - 1}")

        if self.game_over:
            if self.won:
                print(f"\n{GREEN}Passenger delivered successfully!{RESET}")
            else:
                print(f"\n{RED}Wrong destination! Game over.{RESET}")

    def move(self, direction):
        """Move the taxi in the specified direction."""
        if self.game_over:
            return

        new_x, new_y = self.taxi_x, self.taxi_y

        if direction == 'up':
            new_y = self.taxi_y - 1
        elif direction == 'down':
            new_y = self.taxi_y + 1
        elif direction == 'left':
            new_x = self.taxi_x - 1
        elif direction == 'right':
            new_x = self.taxi_x + 1

        # Check bounds
        if not (0 <= new_x < self.size and 0 <= new_y < self.size):
            return  # Can't move out of bounds

        # Check for wall
        if self._is_wall_between((self.taxi_x, self.taxi_y), (new_x, new_y)):
            return  # Can't move through wall

        # Execute move
        self.taxi_x = new_x
        self.taxi_y = new_y

        # Check for automatic pickup (before checking win/lose)
        just_picked_up = False
        if (not self.passenger_in_taxi and
            self.taxi_x == self.passenger_x and
            self.taxi_y == self.passenger_y):
            self.passenger_in_taxi = True
            just_picked_up = True

        # Check for win/lose conditions when carrying passenger
        # (but not on the same move as pickup - they're still at pickup location)
        if self.passenger_in_taxi and not just_picked_up:
            # Check if at a colored location
            for name, (lx, ly) in self.locations.items():
                if (self.taxi_x, self.taxi_y) == (lx, ly):
                    if name == self.destination_name:
                        # WIN - correct destination
                        self.game_over = True
                        self.won = True
                    else:
                        # LOSE - wrong colored location
                        self.game_over = True
                        self.won = False
                    break

        self._log_state()

    def save_trace(self):
        """Save the trace to the appropriate directory (pos or neg)."""
        subdir = "pos" if self.won else "neg"
        prefix = "pos_trace_" if self.won else "neg_trace_"

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
        if self.random_size:
            self.size = random.randint(4, 7)

        self._init_board()
        self._reset_game_state()

    def _reset_game_state(self):
        """Reset only the game state without changing board layout."""
        self.game_over = False
        self.won = False
        self.passenger_in_taxi = False
        self.trace = []
        self._log_state()

    def _bfs_path(self, start, goal, avoid_positions=None):
        """Find path from start to goal using BFS, respecting walls."""
        if avoid_positions is None:
            avoid_positions = set()

        if start == goal:
            return []

        queue = deque([(start, [])])
        visited = {start}

        while queue:
            (x, y), path = queue.popleft()

            directions = [('up', (0, -1)), ('down', (0, 1)),
                         ('left', (-1, 0)), ('right', (1, 0))]
            random.shuffle(directions)

            for direction, (dx, dy) in directions:
                nx, ny = x + dx, y + dy

                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    continue

                if (nx, ny) in visited:
                    continue

                if self._is_wall_between((x, y), (nx, ny)):
                    continue

                if (nx, ny) in avoid_positions:
                    continue

                new_path = path + [direction]

                if (nx, ny) == goal:
                    return new_path

                visited.add((nx, ny))
                queue.append(((nx, ny), new_path))

        return None

    def _add_detours(self, base_path, start_pos, avoid_positions=None):
        """Add random exploration moves to a path, respecting walls."""
        if avoid_positions is None:
            avoid_positions = set()

        if len(base_path) == 0:
            return base_path

        num_detours = random.randint(1, 2)
        path = base_path.copy()

        for _ in range(num_detours):
            if len(path) == 0:
                break
            insert_pos = random.randint(0, len(path))

            # Simulate position at insert point
            x, y = start_pos
            for i in range(insert_pos):
                direction = path[i]
                dx, dy = {'up': (0, -1), 'down': (0, 1),
                         'left': (-1, 0), 'right': (1, 0)}[direction]
                x, y = x + dx, y + dy

            # Try to add a valid detour
            directions = ['up', 'down', 'left', 'right']
            random.shuffle(directions)

            for move in directions:
                dx, dy = {'up': (0, -1), 'down': (0, 1),
                         'left': (-1, 0), 'right': (1, 0)}[move]
                nx, ny = x + dx, y + dy

                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    continue
                if self._is_wall_between((x, y), (nx, ny)):
                    continue
                if (nx, ny) in avoid_positions:
                    continue

                # Valid detour - add move and reverse
                reverse = {'up': 'down', 'down': 'up',
                          'left': 'right', 'right': 'left'}

                # Check reverse is also valid
                if self._is_wall_between((nx, ny), (x, y)):
                    continue

                path = path[:insert_pos] + [move, reverse[move]] + path[insert_pos:]
                break

        return path

    def _get_wrong_colored_positions(self, exclude_passenger=False):
        """Get colored positions that are 'wrong' (not destination, optionally not passenger).

        Args:
            exclude_passenger: If True, also exclude passenger pickup location.
                               Use True when pathing TO passenger.
        """
        wrong_positions = set()
        for name, (lx, ly) in self.locations.items():
            # Skip destination
            if name == self.destination_name:
                continue
            # Skip passenger pickup location if requested
            if exclude_passenger and name == self.passenger_loc_name:
                continue
            wrong_positions.add((lx, ly))
        return wrong_positions

    def _bfs_optimal_route(self):
        """BFS to find optimal route through waypoints."""
        start = (self.taxi_x, self.taxi_y)
        passenger_pos = (self.passenger_x, self.passenger_y)
        dest_pos = (self.dest_x, self.dest_y)

        wrong_before_pickup = self._get_wrong_colored_positions(exclude_passenger=True)
        wrong_after_pickup = self._get_wrong_colored_positions(exclude_passenger=False)

        path_to_passenger = self._bfs_path(start, passenger_pos,
                                            avoid_positions=wrong_before_pickup)
        if path_to_passenger is None:
            return None

        path_to_dest = self._bfs_path(passenger_pos, dest_pos,
                                       avoid_positions=wrong_after_pickup)
        if path_to_dest is None:
            return None

        if random.random() < 0.5:
            path_to_passenger = self._add_detours(path_to_passenger, start,
                                                   avoid_positions=wrong_before_pickup)
        if random.random() < 0.5:
            path_to_dest = self._add_detours(path_to_dest, passenger_pos,
                                              avoid_positions=wrong_after_pickup)

        return path_to_passenger + path_to_dest

    def _bfs_explore_alternate(self):
        """BFS exploration with alternate routing through waypoints."""
        start = (self.taxi_x, self.taxi_y)
        passenger_pos = (self.passenger_x, self.passenger_y)
        dest_pos = (self.dest_x, self.dest_y)

        wrong_positions = self._get_wrong_colored_positions()

        if not wrong_positions:
            return None

        target_alternate = random.choice(list(wrong_positions))

        path_to_dest = self._bfs_path(start, dest_pos, avoid_positions={passenger_pos})
        if path_to_dest is None:
            return None

        path_to_passenger = self._bfs_path(dest_pos, passenger_pos)
        if path_to_passenger is None:
            return None

        path_to_alternate = self._bfs_path(passenger_pos, target_alternate)
        if path_to_alternate is None:
            return None

        return path_to_dest + path_to_passenger + path_to_alternate

    def _execute_moves(self, moves):
        """Execute a sequence of moves."""
        for direction in moves:
            if self.game_over:
                break
            self.move(direction)
        return self.trace.copy()

    def _trace_to_key(self, trace):
        """Convert trace to hashable key for uniqueness checking."""
        def make_hashable(v):
            if isinstance(v, list):
                return tuple(v)
            return v
        return tuple(tuple((k, make_hashable(v))
                          for k, v in sorted(state.items()))
                    for state in trace)

    def generate_traces(self, num_traces):
        """Generate num_traces unique traces via BFS exploration."""
        pos_traces = set()
        neg_traces = set()

        attempts = 0
        max_attempts = num_traces * 200

        print(f"\nBFS exploration: generating {num_traces * 2} traces...")

        while ((len(pos_traces) < num_traces or len(neg_traces) < num_traces)
               and attempts < max_attempts):
            attempts += 1

            if len(pos_traces) < num_traces:
                self.reset()
                moves = self._bfs_optimal_route()
                if moves is not None:
                    self._reset_game_state()
                    trace = self._execute_moves(moves)

                    if self.won:
                        trace_key = self._trace_to_key(trace)
                        if trace_key not in pos_traces:
                            pos_traces.add(trace_key)
                            self._save_trace_to_file(trace, is_positive=True,
                                                      trace_num=len(pos_traces))
                            print(f"  BFS trace {len(pos_traces)}/{num_traces} (optimal)")

            if len(neg_traces) < num_traces:
                self.reset()
                moves = self._bfs_explore_alternate()
                if moves is not None:
                    self._reset_game_state()
                    trace = self._execute_moves(moves)

                    is_valid = (not self.won and self.game_over)

                    if is_valid:
                        trace_key = self._trace_to_key(trace)
                        if trace_key not in neg_traces:
                            neg_traces.add(trace_key)
                            self._save_trace_to_file(trace, is_positive=False,
                                                      trace_num=len(neg_traces))
                            print(f"  BFS trace {len(neg_traces)}/{num_traces} (alternate)")

        if len(pos_traces) < num_traces:
            print(f"\nWarning: Only found {len(pos_traces)} unique optimal routes")
        if len(neg_traces) < num_traces:
            print(f"\nWarning: Only found {len(neg_traces)} unique alternate routes")

        print(f"\nBFS exploration complete!")
        print(f"  Optimal routes: {len(pos_traces)}")
        print(f"  Alternate routes: {len(neg_traces)}")

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
            name = f"{self.size}x{self.size}_taxi_config"

        # Convert walls to serializable format
        walls_list = []
        seen_walls = set()
        for (x1, y1), (x2, y2) in self.walls:
            # Only add one direction of each wall
            wall_key = tuple(sorted([(x1, y1), (x2, y2)]))
            if wall_key not in seen_walls:
                seen_walls.add(wall_key)
                walls_list.append({"from": {"x": x1, "y": y1}, "to": {"x": x2, "y": y2}})

        return {
            "name": name,
            "grid_size": self.size,
            "start_pos": {"x": self.taxi_x, "y": self.taxi_y},
            "colored_cells": {
                "red": {"x": self.locations['R'][0], "y": self.locations['R'][1]},
                "green": {"x": self.locations['G'][0], "y": self.locations['G'][1]},
                "blue": {"x": self.locations['B'][0], "y": self.locations['B'][1]},
                "yellow": {"x": self.locations['Y'][0], "y": self.locations['Y'][1]},
            },
            "pickup_color": self.passenger_loc_name.lower() if hasattr(self, 'passenger_loc_name') else "red",
            "dropoff_color": self.destination_name.lower() if hasattr(self, 'destination_name') else "green",
            "barriers": walls_list,
            # Also include raw location data for convenience
            "locations": {k: {"x": v[0], "y": v[1]} for k, v in self.locations.items()},
            "passenger": {"x": self.passenger_x, "y": self.passenger_y},
            "destination": {"x": self.dest_x, "y": self.dest_y},
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

        if not quit_mid_game:
            self.display()
            self.save_trace()

        return quit_session


def bfs_reachable(size: int, start: tuple, passenger: tuple, destination: tuple,
                  locations: dict, walls: set) -> bool:
    """
    Check if taxi can pick up passenger and reach destination without hitting wrong locations.

    Args:
        size: Grid size
        start: Starting position (x, y)
        passenger: Passenger position (x, y)
        destination: Destination position (x, y)
        locations: Dict mapping location names to (x, y) tuples
        walls: Set of wall tuples ((x1,y1), (x2,y2))

    Returns:
        True if the path is valid
    """
    def bfs_path(from_pos, to_pos, avoid_positions=None):
        """BFS to check if path exists."""
        if avoid_positions is None:
            avoid_positions = set()
        if from_pos == to_pos:
            return True
        if from_pos in avoid_positions or to_pos in avoid_positions:
            return False

        visited = {from_pos}
        queue = deque([from_pos])

        while queue:
            x, y = queue.popleft()

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy

                if not (0 <= nx < size and 0 <= ny < size):
                    continue
                if (nx, ny) in visited:
                    continue
                if ((x, y), (nx, ny)) in walls:
                    continue
                if (nx, ny) in avoid_positions:
                    continue

                if (nx, ny) == to_pos:
                    return True

                visited.add((nx, ny))
                queue.append((nx, ny))

        return False

    # Find wrong colored positions (not destination, not passenger for first leg)
    wrong_after_pickup = set()
    for name, pos in locations.items():
        if pos != destination:
            wrong_after_pickup.add(pos)

    wrong_before_pickup = wrong_after_pickup - {passenger}

    # Check: can reach passenger from start avoiding wrong locations?
    if not bfs_path(start, passenger, wrong_before_pickup):
        return False

    # Check: can reach destination from passenger avoiding wrong locations?
    if not bfs_path(passenger, destination, wrong_after_pickup - {passenger}):
        return False

    return True


def generate_random_configs(
    num_configs: int,
    random_pos: bool = False,
    random_config: bool = False,
    seed: int = None
) -> list:
    """
    Generate random board configurations with guaranteed reachability.

    Args:
        num_configs: Number of configurations to generate
        random_pos: Whether to randomize positions of R/G/B/Y
        random_config: Whether to randomize which colors are pickup/dropoff
        seed: Random seed for reproducibility

    Returns:
        List of configuration dictionaries compatible with spec_generator
    """
    if seed is not None:
        random.seed(seed)

    configs = []
    seen = set()
    size = 5  # Standard 5x5 board

    # Standard locations
    standard_locations = {
        'R': (0, 0),
        'G': (4, 0),
        'Y': (0, 4),
        'B': (3, 4)
    }

    # Standard walls
    standard_walls = set()
    wall_list = [
        ((0, 0), (1, 0)), ((0, 1), (1, 1)),
        ((1, 3), (2, 3)), ((1, 4), (2, 4)),
        ((2, 0), (3, 0)), ((2, 1), (3, 1)),
    ]
    for (x1, y1), (x2, y2) in wall_list:
        standard_walls.add(((x1, y1), (x2, y2)))
        standard_walls.add(((x2, y2), (x1, y1)))

    # If neither random flag is set, generate fixed configs
    if not random_pos and not random_config:
        # Fixed configuration: standard board, pickup=R, dropoff=G
        locations = standard_locations
        passenger = locations['R']
        destination = locations['G']

        # Find valid start position
        for y in range(size):
            for x in range(size):
                if (x, y) not in [passenger, destination]:
                    start = (x, y)
                    break
            else:
                continue
            break

        for i in range(num_configs):
            # Convert walls to serializable format
            walls_list = []
            seen_walls_local = set()
            for (x1, y1), (x2, y2) in standard_walls:
                wall_key = tuple(sorted([(x1, y1), (x2, y2)]))
                if wall_key not in seen_walls_local:
                    seen_walls_local.add(wall_key)
                    walls_list.append({"from": {"x": x1, "y": y1}, "to": {"x": x2, "y": y2}})

            configs.append({
                "name": f"config_{i + 1}",
                "grid_size": size,
                "start_pos": {"x": start[0], "y": start[1]},
                "colored_cells": {
                    "red": {"x": locations['R'][0], "y": locations['R'][1]},
                    "green": {"x": locations['G'][0], "y": locations['G'][1]},
                    "blue": {"x": locations['B'][0], "y": locations['B'][1]},
                    "yellow": {"x": locations['Y'][0], "y": locations['Y'][1]},
                },
                "pickup_color": "red",
                "dropoff_color": "green",
                "barriers": walls_list,
                "locations": {k: {"x": v[0], "y": v[1]} for k, v in locations.items()},
                "passenger": {"x": passenger[0], "y": passenger[1]},
                "destination": {"x": destination[0], "y": destination[1]},
            })
        return configs

    # Variable configs
    attempts = 0
    max_attempts = num_configs * 100

    while len(configs) < num_configs and attempts < max_attempts:
        attempts += 1

        # Generate random positions if requested
        if random_pos:
            locations = {}
            used_positions = set()

            for name in ['R', 'G', 'Y', 'B']:
                while True:
                    x = random.randint(0, size - 1)
                    y = random.randint(0, size - 1)
                    if (x, y) not in used_positions:
                        locations[name] = (x, y)
                        used_positions.add((x, y))
                        break
        else:
            locations = standard_locations

        # Determine pickup/dropoff
        if random_config:
            # Random pickup and dropoff colors
            colors = ['R', 'G', 'Y', 'B']
            passenger_name, dest_name = random.sample(colors, 2)
            passenger = locations[passenger_name]
            destination = locations[dest_name]
        else:
            # Fixed: pickup=R, dropoff=G
            passenger = locations['R']
            destination = locations['G']
            passenger_name = 'R'
            dest_name = 'G'

        # Generate random walls (3 walls)
        walls = set()
        possible_walls = []

        # Horizontal walls
        for x in range(size):
            for y in range(size - 1):
                possible_walls.append(((x, y), (x, y + 1)))

        # Vertical walls
        for x in range(size - 1):
            for y in range(size):
                possible_walls.append(((x, y), (x + 1, y)))

        random.shuffle(possible_walls)
        walls_added = 0

        for wall in possible_walls:
            if walls_added >= 3:
                break
            (x1, y1), (x2, y2) = wall
            walls.add(((x1, y1), (x2, y2)))
            walls.add(((x2, y2), (x1, y1)))
            walls_added += 1

        # Find valid start position
        forbidden = {passenger, destination}
        start = None
        for y in range(size):
            for x in range(size):
                if (x, y) not in forbidden and (x, y) not in [locations[n] for n in locations]:
                    start = (x, y)
                    break
            if start:
                break

        # Fallback: just avoid passenger and destination
        if start is None:
            for y in range(size):
                for x in range(size):
                    if (x, y) not in forbidden:
                        start = (x, y)
                        break
                if start:
                    break

        if start is None:
            continue

        # Check reachability
        if not bfs_reachable(size, start, passenger, destination, locations, walls):
            continue

        # Check uniqueness
        config_key = (
            tuple(sorted((k, v) for k, v in locations.items())),
            passenger_name, dest_name,
            tuple(sorted(walls))
        )
        if config_key in seen:
            continue
        seen.add(config_key)

        # Convert walls to serializable format
        walls_list = []
        seen_walls_local = set()
        for (x1, y1), (x2, y2) in walls:
            wall_key = tuple(sorted([(x1, y1), (x2, y2)]))
            if wall_key not in seen_walls_local:
                seen_walls_local.add(wall_key)
                walls_list.append({"from": {"x": x1, "y": y1}, "to": {"x": x2, "y": y2}})

        color_map = {'R': 'red', 'G': 'green', 'B': 'blue', 'Y': 'yellow'}

        configs.append({
            "name": f"config_{len(configs) + 1}",
            "grid_size": size,
            "start_pos": {"x": start[0], "y": start[1]},
            "colored_cells": {
                "red": {"x": locations['R'][0], "y": locations['R'][1]},
                "green": {"x": locations['G'][0], "y": locations['G'][1]},
                "blue": {"x": locations['B'][0], "y": locations['B'][1]},
                "yellow": {"x": locations['Y'][0], "y": locations['Y'][1]},
            },
            "pickup_color": color_map.get(passenger_name, "red"),
            "dropoff_color": color_map.get(dest_name, "green"),
            "barriers": walls_list,
            "locations": {k: {"x": v[0], "y": v[1]} for k, v in locations.items()},
            "passenger": {"x": passenger[0], "y": passenger[1]},
            "destination": {"x": destination[0], "y": destination[1]},
        })

    return configs


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
    parser = argparse.ArgumentParser(description="Interactive Taxi Game (Tuple Version)")
    parser.add_argument('--random-size', action='store_true',
                        help='Randomize board size (4-7) for each game instance')
    parser.add_argument('--random-pos', action='store_true',
                        help='Randomize physical positions of R/G/B/Y on grid (pickup=Red, dropoff=Green)')
    parser.add_argument('--random-config', action='store_true',
                        help='Randomize which colors are pickup/dropoff (e.g., pickup=Blue, dropoff=Yellow)')
    parser.add_argument('--gen', type=int, metavar='N',
                        help='Generate N positive and N negative traces automatically')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for traces (default: Logs/ttaxi/{timestamp})')
    parser.add_argument('--include-passenger-state', action='store_true',
                        help='Include passengerInTaxi boolean in traces (default: exclude for cleaner temporal mining)')
    parser.add_argument('--config', type=int, metavar='N',
                        help='Generate N board configurations as JSON (combine with --random-pos/--random-config)')
    parser.add_argument('--config-output', type=str, metavar='FILE',
                        help='Output JSON file for --config mode (default: stdout)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Config generation mode
    if args.config is not None:
        configs = generate_random_configs(
            num_configs=args.config,
            random_pos=args.random_pos,
            random_config=args.random_config,
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

    if args.output:
        session_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path(f"Logs/ttaxi/{timestamp}")
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "pos").mkdir(exist_ok=True)
    (session_dir / "neg").mkdir(exist_ok=True)

    if args.gen is not None:
        print("=" * 50)
        print("TAXI (TUPLE) - TRACE GENERATION MODE")
        print("=" * 50)
        if args.random_size:
            print("  * Random board sizes enabled (4-7)")
        if args.random_pos:
            print("  * Random positions enabled (pickup=Red, dropoff=Green)")
        if args.random_config:
            print("  * Random config enabled (pickup/dropoff colors vary)")
        if args.include_passenger_state:
            print("  * Including passengerInTaxi in traces")
        else:
            print("  * Excluding passengerInTaxi from traces (for temporal mining)")
        print(f"  * Target: {args.gen} positive + {args.gen} negative traces")
        print(f"  * Output: {session_dir}")

        game = TupleTaxiGame(session_dir=session_dir,
                             random_size=args.random_size,
                             random_pos=args.random_pos,
                             random_config=args.random_config,
                             include_passenger_state=args.include_passenger_state)

        game.generate_traces(args.gen)
        print(f"\nAll traces saved to: {session_dir}")

    else:
        print("Welcome to Taxi (Tuple Version)!")
        print("\nGame mechanics:")
        print("  - Taxi automatically picks up passenger when reaching them")
        print("  - Taxi automatically drops off at destination (WIN)")
        print("  - Going to wrong colored location with passenger (LOSE)")
        if args.random_size:
            print("  * Random board sizes enabled (4-7)")
        if args.random_pos:
            print("  * Random positions enabled (pickup=Red, dropoff=Green)")
        if args.random_config:
            print("  * Random config enabled (pickup/dropoff colors vary)")
        print("\nControls:")
        print("  Arrow keys - Move taxi")
        print("  q - Quit")
        print("\nStarting session...")
        print("(Press any key to continue)")

        wait_for_key()

        while True:
            game = TupleTaxiGame(session_dir=session_dir,
                                 random_size=args.random_size,
                                 random_pos=args.random_pos,
                                 random_config=args.random_config,
                                 include_passenger_state=args.include_passenger_state)
            quit_session = game.play()

            if quit_session:
                print("\n" + "=" * 50)
                print("Session ended. Thanks for playing!")
                break
            else:
                print("\n" + "=" * 50)
                print("Press any key to play again, or Ctrl+C to exit...")
                wait_for_key()


if __name__ == "__main__":
    main()
