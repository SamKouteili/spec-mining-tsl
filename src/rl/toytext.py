"""
ToyText game-specific implementations for the RL synthesis loop.

This module contains implementations specific to the ToyText games:
- FrozenLake (ice_lake)
- Taxi
- CliffWalking
- Blackjack

Keep this separate from synt.py to allow plugging in other games later.
"""

import sys
from pathlib import Path
from typing import Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


#################################################################
################## CONFIG GENERATION ############################
#################################################################

def generate_configs(
    game: str,
    num_configs: int = 1,
    seed: Optional[int] = None,
    **kwargs
) -> list[dict[str, Any]]:
    """
    Generate game configurations programmatically.

    This is the main API for generating configs. It wraps the game-specific
    config generators with a unified interface.

    Args:
        game: Game type (ice_lake, frozen_lake, taxi, cliff_walking, blackjack)
        num_configs: Number of configurations to generate
        seed: Random seed for reproducibility
        **kwargs: Game-specific options (see below)

    Game-specific kwargs:
        ice_lake/frozen_lake:
            - random_size: bool = False (randomize board size 3-5)
            - random_placements: bool = True (randomize goal/hole positions)
            - base_size: int = 4 (base board size)

        taxi:
            - random_pos: bool = False (randomize R/G/B/Y positions)
            - random_config: bool = False (randomize pickup/dropoff colors)

        cliff_walking:
            - random_size: bool = False (randomize board width 3-12)
            - random_height: bool = False (randomize cliff height 1-3)
            - var_moves: bool = False (use variant movement functions)

        blackjack:
            - strategies: list = None (list of strategies to use)

    Returns:
        List of configuration dictionaries

    Example:
        # Generate 5 random frozen lake configs
        configs = generate_configs(
            game="ice_lake",
            num_configs=5,
            random_placements=True,
            seed=42
        )

        # Generate a fixed taxi config
        configs = generate_configs(game="taxi", num_configs=1)
    """
    game = game.lower()

    if game in ("ice_lake", "frozen_lake"):
        from games.tfrozen_lake_game import generate_random_configs
        return generate_random_configs(
            num_configs=num_configs,
            random_size=kwargs.get("random_size", False),
            random_placements=kwargs.get("random_placements", False),
            base_size=kwargs.get("base_size", 4),
            seed=seed,
        )

    elif game == "taxi":
        from games.ttaxi_game import generate_random_configs
        return generate_random_configs(
            num_configs=num_configs,
            random_pos=kwargs.get("random_pos", False),
            random_config=kwargs.get("random_config", False),
            seed=seed,
        )

    elif game == "cliff_walking":
        from games.cliff_walking_game import generate_random_configs
        return generate_random_configs(
            num_configs=num_configs,
            random_size=kwargs.get("random_size", False),
            random_height=kwargs.get("random_height", False),
            var_moves=kwargs.get("var_moves", False),
            seed=seed,
        )

    elif game == "blackjack":
        from games.blackjack_game import generate_random_configs
        return generate_random_configs(
            num_configs=num_configs,
            strategies=kwargs.get("strategies", None),
            seed=seed,
        )

    else:
        raise ValueError(f"Unknown game: {game}. Supported: ice_lake, taxi, cliff_walking, blackjack")


def generate_config(
    game: str,
    seed: Optional[int] = None,
    varied: bool = False,
    **kwargs
) -> dict[str, Any]:
    """
    Generate a single game configuration.

    Args:
        game: Game type
        seed: Random seed for reproducibility
        varied: If True, randomize board layout; if False, use fixed layout
        **kwargs: Game-specific options (override varied defaults)

    Returns:
        Single configuration dictionary
    """
    game = game.lower()

    # Apply varied defaults based on game type
    if varied:
        if game in ("ice_lake", "frozen_lake"):
            kwargs.setdefault("random_placements", True)
        elif game == "taxi":
            kwargs.setdefault("random_pos", True)
            kwargs.setdefault("random_config", True)
        elif game == "cliff_walking":
            kwargs.setdefault("random_height", True)
    else:
        # Fixed layout: explicitly disable randomization
        if game in ("ice_lake", "frozen_lake"):
            kwargs.setdefault("random_placements", False)
        elif game == "taxi":
            kwargs.setdefault("random_pos", False)
            kwargs.setdefault("random_config", False)
        elif game == "cliff_walking":
            kwargs.setdefault("random_height", False)

    configs = generate_configs(game=game, num_configs=1, seed=seed, **kwargs)
    return configs[0]


#################################################################
################## RANDOM CONTROLLER GENERATORS #################
#################################################################

def generate_random_controller_ice_lake(params: dict[str, Any]) -> str:
    """Generate a random walk controller for ice_lake/frozen_lake."""
    grid_size = params.get("grid_size", 4)
    start_x = params.get("start_pos", {}).get("x", 0)
    start_y = params.get("start_pos", {}).get("y", 0)

    return f'''
#include <stdlib.h>
#include <time.h>

int x = {start_x};
int y = {start_y};

void read_inputs(void);

int main() {{
    srand(time(NULL));
    x = {start_x};
    y = {start_y};

    while (1) {{
        read_inputs();

        // Random move: 0=up, 1=down, 2=left, 3=right
        int move = rand() % 4;
        int new_x = x, new_y = y;

        switch (move) {{
            case 0: new_y = y - 1; break;  // up
            case 1: new_y = y + 1; break;  // down
            case 2: new_x = x - 1; break;  // left
            case 3: new_x = x + 1; break;  // right
        }}

        // Bounds check
        if (new_x >= 0 && new_x < {grid_size} && new_y >= 0 && new_y < {grid_size}) {{
            x = new_x;
            y = new_y;
        }}
    }}
    return 0;
}}
'''


def generate_random_controller_taxi(params: dict[str, Any]) -> str:
    """Generate a random walk controller for taxi."""
    grid_size = params.get("grid_size", 5)
    start_x = params.get("start_pos", {}).get("x", 2)
    start_y = params.get("start_pos", {}).get("y", 2)

    return f'''
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

int x = {start_x};
int y = {start_y};
bool passengerInTaxi = false;

void read_inputs(void);

int main() {{
    srand(time(NULL));
    x = {start_x};
    y = {start_y};
    passengerInTaxi = false;

    while (1) {{
        read_inputs();

        // Random move: 0=up, 1=down, 2=left, 3=right
        int move = rand() % 4;
        int new_x = x, new_y = y;

        switch (move) {{
            case 0: new_y = y - 1; break;
            case 1: new_y = y + 1; break;
            case 2: new_x = x - 1; break;
            case 3: new_x = x + 1; break;
        }}

        // Bounds check
        if (new_x >= 0 && new_x < {grid_size} && new_y >= 0 && new_y < {grid_size}) {{
            x = new_x;
            y = new_y;
        }}

        // Passenger pickup is handled by game harness
    }}
    return 0;
}}
'''


def generate_random_controller_cliff_walking(params: dict[str, Any]) -> str:
    """Generate a random walk controller for cliff_walking."""
    grid_cols = params.get("grid_size", 12)
    grid_rows = params.get("grid_rows", 4)
    start_x = params.get("start_pos", {}).get("x", 0)
    start_y = params.get("start_pos", {}).get("y", 0)

    return f'''
#include <stdlib.h>
#include <time.h>

int x = {start_x};
int y = {start_y};

void read_inputs(void);

int main() {{
    srand(time(NULL));
    x = {start_x};
    y = {start_y};

    while (1) {{
        read_inputs();

        // Random move: 0=up, 1=down, 2=left, 3=right
        int move = rand() % 4;
        int new_x = x, new_y = y;

        switch (move) {{
            case 0: new_y = y - 1; break;
            case 1: new_y = y + 1; break;
            case 2: new_x = x - 1; break;
            case 3: new_x = x + 1; break;
        }}

        // Bounds check
        if (new_x >= 0 && new_x < {grid_cols} && new_y >= 0 && new_y < {grid_rows}) {{
            x = new_x;
            y = new_y;
        }}
    }}
    return 0;
}}
'''


# Registry of random controller generators by game name
RANDOM_CONTROLLER_GENERATORS = {
    "ice_lake": generate_random_controller_ice_lake,
    "frozen_lake": generate_random_controller_ice_lake,
    "taxi": generate_random_controller_taxi,
    "cliff_walking": generate_random_controller_cliff_walking,
}


def get_random_controller_generator(game: str):
    """
    Get the random controller generator for a game.

    Args:
        game: Game type

    Returns:
        Generator function, or None if not supported
    """
    return RANDOM_CONTROLLER_GENERATORS.get(game.lower())


def generate_random_controller(game: str, params: dict[str, Any]) -> str:
    """
    Generate a random controller for the specified game.

    Args:
        game: Game type
        params: Game configuration parameters

    Returns:
        C code for a random controller

    Raises:
        ValueError: If game is not supported
    """
    generator = get_random_controller_generator(game)
    if generator is None:
        raise ValueError(f"No random controller generator for game: {game}")
    return generator(params)


#################################################################
################## CONFIG CONVERSION ############################
#################################################################

def config_to_params(config: dict[str, Any], game: str) -> dict[str, Any]:
    """
    Convert a game config dict to the params format expected by SynthesisAPI.

    Args:
        config: Configuration from game --config command or generate_config()
        game: Game type

    Returns:
        Parameters dict for synthesis
    """
    game = game.lower()

    if game in ("ice_lake", "frozen_lake"):
        return {
            "grid_size": config.get("grid_size", config.get("size", 4)),
            "goal": config.get("goal", {"x": 3, "y": 3}),
            "holes": config.get("holes", []),
            "start_pos": config.get("start_pos", {"x": 0, "y": 0}),
        }

    elif game == "cliff_walking":
        goal_pos = config.get("goal_pos", config.get("goal", {"x": 11, "y": 0}))
        return {
            "grid_size": config.get("grid_size", config.get("width", 12)),
            "grid_rows": config.get("grid_rows", config.get("height", 4)),
            "cliff_height": config.get("cliff_height", 1),
            "cliff_min": config.get("cliff_min", 1),
            "cliff_max": config.get("cliff_max", 10),
            "start_pos": config.get("start_pos", {"x": 0, "y": 0}),
            "goal_pos": goal_pos,
            "variable_updates": config.get("variable_updates"),
        }

    elif game == "taxi":
        return {
            "grid_size": config.get("grid_size", 5),
            "start_pos": config.get("start_pos", {"x": 2, "y": 2}),
            "colored_cells": config.get("colored_cells", {
                "red": {"x": 0, "y": 0},
                "green": {"x": 4, "y": 0},
                "blue": {"x": 3, "y": 4},
                "yellow": {"x": 0, "y": 4},
            }),
            "pickup_color": config.get("pickup_color", "red"),
            "dropoff_color": config.get("dropoff_color", "green"),
            "barriers": config.get("barriers", []),
        }

    elif game == "blackjack":
        return {
            "strategy": config.get("strategy", "threshold"),
            "dealer_stand_threshold": config.get("dealer_stand_threshold", 17),
            "bust_threshold": config.get("bust_threshold", 21),
        }

    else:
        # Unknown game, return config as-is
        return config


#################################################################
################## TRACE PARSING ################################
#################################################################

import re


def parse_trace_from_output(output: str, game: str) -> list[dict[str, Any]]:
    """
    Parse game output into a list of state dictionaries (JSONL format).

    Args:
        output: Raw game stdout
        game: Game type (ice_lake, taxi, cliff_walking, blackjack)

    Returns:
        List of state dicts, one per timestep
    """
    trace = []
    game = game.lower()

    if game in ("ice_lake", "frozen_lake"):
        # Format: Step N: Position (X,Y)
        # Use player tuple for proper mining predicate discovery
        pattern = r'Step (\d+): Position \((\d+),(\d+)\)'
        for match in re.finditer(pattern, output):
            x = int(match.group(2))
            y = int(match.group(3))
            trace.append({"player": [x, y]})

    elif game == "taxi":
        # Format: Step N: Position (X,Y), passengerInTaxi=0/1
        pattern = r'Step (\d+): Position \((\d+),(\d+)\), passengerInTaxi=(\d+)'
        for match in re.finditer(pattern, output):
            x = int(match.group(2))
            y = int(match.group(3))
            passenger = bool(int(match.group(4)))
            trace.append({"x": x, "y": y, "passengerInTaxi": passenger})

    elif game == "cliff_walking":
        # Format: Step N: Position (X,Y)
        # Use player tuple for proper mining predicate discovery
        pattern = r'Step (\d+): Position \((\d+),(\d+)\)'
        for match in re.finditer(pattern, output):
            x = int(match.group(2))
            y = int(match.group(3))
            trace.append({"player": [x, y]})

    elif game == "blackjack":
        # Blackjack has more complex output, parse game outcomes
        # For now, return empty - blackjack traces need different handling
        pass

    return trace


#################################################################
################## TOYTEXT GAME REGISTRY ########################
#################################################################

SUPPORTED_GAMES = ["ice_lake", "frozen_lake", "taxi", "cliff_walking", "blackjack"]


def is_supported_game(game: str) -> bool:
    """Check if a game is supported by the ToyText module."""
    return game.lower() in SUPPORTED_GAMES


#################################################################
################## CLI ##########################################
#################################################################

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="ToyText game utilities - generate configs and run random walks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a config and run a random walk
  python src/rl/toytext.py ice_lake --seed 42

  # Generate multiple random walks
  python src/rl/toytext.py ice_lake --num-runs 5 --seed 42

  # Save config to file
  python src/rl/toytext.py taxi --save-config config.json

  # Use random placements for ice_lake
  python src/rl/toytext.py ice_lake --random-placements --seed 123
"""
    )
    parser.add_argument("game", choices=SUPPORTED_GAMES,
                        help="Game type")
    parser.add_argument("--seed", "-s", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--num-runs", "-n", type=int, default=1,
                        help="Number of random walks to run")
    parser.add_argument("--timeout", "-t", type=int, default=100,
                        help="Step timeout for random walks")
    parser.add_argument("--save-config", type=Path, default=None,
                        help="Save generated config to JSON file")
    parser.add_argument("--random-placements", action="store_true",
                        help="Use random placements (ice_lake)")
    parser.add_argument("--random-size", action="store_true",
                        help="Use random board size")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Debug mode")

    args = parser.parse_args()

    # Generate config
    config = generate_config(
        game=args.game,
        seed=args.seed,
        random_placements=args.random_placements,
        random_size=args.random_size,
    )
    params = config_to_params(config, args.game)

    print(f"=== {args.game.upper()} Config ===")
    print(json.dumps(config, indent=2))

    if args.save_config:
        with open(args.save_config, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\nConfig saved to: {args.save_config}")

    # Run random walks
    print(f"\n=== Running {args.num_runs} random walk(s) ===")

    # Import SynthesisAPI here to avoid circular imports at module level
    from src.rl.synt import SynthesisAPI

    api = SynthesisAPI(game=args.game, debug=args.debug)

    pos_count = 0
    neg_count = 0

    for i in range(args.num_runs):
        result = api.initial_random_run(
            params=params,
            timeout_steps=args.timeout,
        )

        if result.is_positive:
            pos_count += 1
            status = "POSITIVE (goal)"
        else:
            neg_count += 1
            status = f"NEGATIVE ({result.failure_reason})"

        print(f"Run {i+1}: {status} in {result.steps} steps")

        # Print trace positions
        if result.trace:
            positions = []
            for entry in result.trace:
                x = entry.get("x", entry.get("playerX"))
                y = entry.get("y", entry.get("playerY"))
                if x is not None and y is not None:
                    positions.append(f"({x},{y})")
            if positions:
                print(f"  Trace: {' -> '.join(positions)}")

    print(f"\nSummary: {pos_count} positive, {neg_count} negative")
