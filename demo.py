#!/usr/bin/env python3
"""
Demo script for TSL_f specification mining and controller synthesis.

This script demonstrates the complete pipeline:
1. Interactive game play to generate traces
2. Mining a TSL_f specification from traces
3. Synthesizing THREE controllers simultaneously (1 fixed, 2 varied configs)
4. Replaying all controller trajectories side by side

Usage:
    python demo.py frozen_lake
    python demo.py frozen_lake --random-placements
    python demo.py frozen_lake --gen-traces 10
"""

import argparse
import subprocess
import json
import yaml
import sys
import os
import re
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional


sys.path.insert(0, str(Path(__file__).parent))
from games.tfrozen_lake_game import generate_random_configs as fl_generate_random_configs

# Global debug flag
DEBUG = False


def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    print()
    print(char * 60)
    print(f"  {text}")
    print(char * 60)
    print()


def print_debug(text: str):
    """Print only if debug mode is enabled."""
    if DEBUG:
        print(text)


# ============== Board Configuration Generation ==============

def generate_varied_configs(num_configs: int, seed: int = 42) -> list[dict]:
    """
    Generate random test configurations with guaranteed reachability.
    Uses var_config mode: fixed 4x4 size, random goal/hole placements.

    Uses the frozen lake game's config generation function.

    Args:
        num_configs: Number of configurations to generate
        seed: Random seed for reproducibility

    Returns:
        List of configuration dictionaries
    """
    configs = fl_generate_random_configs(
        num_configs=num_configs,
        random_size=False,
        random_placements=True,
        base_size=4,
        seed=seed
    )

    # Rename configs for demo purposes
    for i, config in enumerate(configs):
        config["name"] = f"varied_{i + 1}"

    return configs


# ============== Game and Trace Generation ==============

def run_interactive_game(game_name: str, random_placements: bool = False) -> tuple[Path, dict]:
    """
    Run the game in interactive mode and return the session directory and board config.

    Returns:
        Tuple of (session_dir, board_config)
    """
    if game_name != "frozen_lake":
        raise ValueError(f"Unsupported game: {game_name}. Only frozen_lake is currently supported.")

    print_header("STEP 1: Play the Game")
    print("Play the Frozen Lake game to generate training traces.")
    print("Navigate using arrow keys. Press 'q' to quit when done.")
    print()
    print("IMPORTANT: Generate BOTH positive (winning) AND negative (losing) traces!")
    print("The mining algorithm learns from the contrast between success and failure.")
    print()
    input("Press Enter to start playing...")

    # Create a session directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(f"games/Logs/tfrozen_lake/{timestamp}")
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "pos").mkdir(exist_ok=True)
    (session_dir / "neg").mkdir(exist_ok=True)

    # Build the game command
    game_script = Path("games/tfrozen_lake_game.py")
    cmd = [sys.executable, str(game_script), "--output", str(session_dir)]
    if random_placements:
        cmd.append("--random-placements")

    # Save config to a file that the game will write
    config_file = session_dir / "board_config.json"
    cmd.extend(["--save-config", str(config_file)])

    # Run the game
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        pass  # User quit, that's expected
    except KeyboardInterrupt:
        print("\n\nGame interrupted.")

    # Load the board config
    if not config_file.exists():
        print("Error: No board configuration found. Did you play at least one game?")
        sys.exit(1)

    with open(config_file) as f:
        board_config = json.load(f)

    # Check that we have traces
    pos_traces = list((session_dir / "pos").glob("*.jsonl"))
    neg_traces = list((session_dir / "neg").glob("*.jsonl"))

    print(f"\nTraces generated: {len(pos_traces)} positive, {len(neg_traces)} negative")

    if len(pos_traces) == 0 and len(neg_traces) == 0:
        print("\nError: No traces were generated. Please play at least one game.")
        sys.exit(1)

    if len(pos_traces) == 0:
        print("\nWarning: No positive (winning) traces. Mining may not work well.")
    if len(neg_traces) == 0:
        print("\nWarning: No negative (losing) traces. Mining may not work well.")

    return session_dir, board_config


def run_auto_generation(game_name: str, num_traces: int, random_placements: bool = False) -> tuple[Path, dict]:
    """
    Automatically generate traces using the game's --gen flag.

    Returns:
        Tuple of (session_dir, board_config)
    """
    if game_name != "frozen_lake":
        raise ValueError(f"Unsupported game: {game_name}. Only frozen_lake is currently supported.")

    print_header("STEP 1: Generate Traces")
    print(f"Automatically generating {num_traces} positive and {num_traces} negative traces...")
    print()

    # Create a session directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(f"games/Logs/tfrozen_lake/{timestamp}")

    # Build the game command
    game_script = Path("games/tfrozen_lake_game.py")
    cmd = [
        sys.executable, str(game_script),
        "--gen", str(num_traces),
        "--output", str(session_dir)
    ]
    if random_placements:
        cmd.append("--random-placements")

    # Run the generation
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print_debug(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Trace generation failed:")
        print(e.stdout)
        print(e.stderr)
        sys.exit(1)

    # Load the board config from the first trace file
    pos_traces = list((session_dir / "pos").glob("*.jsonl"))
    neg_traces = list((session_dir / "neg").glob("*.jsonl"))

    print(f"Generated: {len(pos_traces)} positive, {len(neg_traces)} negative traces")

    # Extract board config from the first trace
    trace_file = pos_traces[0] if pos_traces else neg_traces[0]
    with open(trace_file) as f:
        first_state = json.loads(f.readline())

    # Reconstruct board config from trace
    board_config = {
        "grid_size": 4,  # Default, will be overridden if we can detect it
        "start_pos": {"x": 0, "y": 0},
        "goal": {"x": first_state["goal"][0], "y": first_state["goal"][1]},
        "holes": []
    }

    # Extract holes
    for key in first_state:
        if key.startswith("hole"):
            board_config["holes"].append({
                "x": first_state[key][0],
                "y": first_state[key][1]
            })

    # Try to infer grid size from goal position
    goal_x = first_state["goal"][0]
    goal_y = first_state["goal"][1]
    board_config["grid_size"] = max(goal_x, goal_y) + 1

    # Save config for later use
    config_file = session_dir / "board_config.json"
    with open(config_file, "w") as f:
        json.dump(board_config, f, indent=2)

    return session_dir, board_config


# ============== Mining ==============

def run_mining(session_dir: Path) -> str:
    """
    Run the specification mining pipeline on the generated traces.

    Returns:
        The mined specification string
    """
    print_header("STEP 2: Mine Specification")
    print("Mining a data-aware TSL_f specification from traces...")
    print()

    # Run the mining pipeline
    mine_script = Path("src/mine.sh")
    cmd = [
        "bash", str(mine_script), str(session_dir),
        "--mode", "safety-liveness",
        "--self-inputs-only",
        "--prune",
        "--game", "frozen_lake",
        "--collect-all",
        "--max-size", "7"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print_debug(result.stdout)
    except subprocess.CalledProcessError as e:
        print_debug("Mining output:")
        print_debug(e.stdout)
        print("Mining failed. Check that you have enough varied traces.")
        if DEBUG:
            print("Mining errors:")
            print(e.stderr)
        sys.exit(1)

    # Read the mined spec
    spec_file = session_dir / "out" / "spec.tsl"
    if not spec_file.exists():
        spec_file = session_dir / "out" / "spec_transformed.tsl"

    if not spec_file.exists():
        print("Error: No specification was mined.")
        sys.exit(1)

    spec = spec_file.read_text().strip()

    print(f"Mined specification: {spec}")
    print()

    return spec


# ============== Synthesis ==============

def create_synthesis_config(board_config: dict, spec: str, output_path: Path, config_name: str = "demo_config") -> Path:
    """
    Create a YAML configuration file for the synthesis pipeline.

    Returns:
        Path to the created config file
    """
    # Convert board config to synthesis format
    holes_list = []
    if "holes" in board_config:
        for hole in board_config["holes"]:
            if isinstance(hole, dict):
                holes_list.append({"x": hole["x"], "y": hole["y"]})
            elif isinstance(hole, (list, tuple)):
                holes_list.append({"x": hole[0], "y": hole[1]})

    goal = board_config.get("goal", {"x": 3, "y": 3})
    if isinstance(goal, (list, tuple)):
        goal = {"x": goal[0], "y": goal[1]}

    start_pos = board_config.get("start_pos", {"x": 0, "y": 0})
    if isinstance(start_pos, (list, tuple)):
        start_pos = {"x": start_pos[0], "y": start_pos[1]}

    config = {
        "name": "ice_lake",
        "variable_updates": {
            "x": "[x <- x] || [x <- add x i1()] || [x <- sub x i1()]",
            "y": "[y <- y] || [y <- add y i1()] || [y <- sub y i1()]"
        },
        "synthesis": {
            "command": "issy",
            "args": ["--tslmt", "--synt", "--pruning", "1", "--accel", "no"],
            "timeout_minutes": 30
        },
        # Always enable debug to capture trajectory output
        "debug": True,
        "run_configuration": [
            {
                "name": config_name,
                "grid_size": board_config.get("grid_size", 4),
                "start_pos": start_pos,
                "goal": goal,
                "holes": holes_list,
                "objectives": [
                    {
                        "objective": spec,
                        "timeout": 1000
                    }
                ]
            }
        ]
    }

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return output_path


@dataclass
class SynthesisResult:
    """Result from a synthesis run."""
    config_name: str
    board_config: dict
    trajectory: list[tuple[int, int]]
    success: bool
    error: Optional[str] = None


def run_single_synthesis(config_path: Path, config_name: str, board_config: dict) -> SynthesisResult:
    """
    Run a single synthesis and return the result.
    This function is designed to be run in a thread.
    """
    trajectory = []
    success = False
    error = None

    try:
        pipeline_script = Path("games/synt/run_pipeline.py")
        cmd = [sys.executable, str(pipeline_script), "ice_lake", str(config_path)]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        if process.stdout is None:
            return SynthesisResult(config_name, board_config, [], False, "Failed to capture output")

        for line in process.stdout:
            print_debug(f"[{config_name}] {line.rstrip()}")

            # Extract position from "Step N: Position (X,Y)" lines
            match = re.search(r"Step \d+: Position \((\d+),(\d+)\)", line)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                trajectory.append((x, y))

            # Check for success
            if "SUCCESS" in line or "Goal reached" in line:
                success = True

            # Check for failure
            if "FAIL" in line:
                success = False

        process.wait()

    except Exception as e:
        error = str(e)

    return SynthesisResult(config_name, board_config, trajectory, success, error)


def run_parallel_synthesis(configs: list[tuple[Path, str, dict]]) -> list[SynthesisResult]:
    """
    Run multiple synthesis jobs in parallel.

    Args:
        configs: List of (config_path, config_name, board_config) tuples

    Returns:
        List of SynthesisResult objects
    """
    print_header("STEP 3: Synthesize Controllers")
    print("Synthesizing controllers from the mined specification...")
    print("(This may take a few minutes)")
    print()

    results = []

    with ThreadPoolExecutor(max_workers=len(configs)) as executor:
        futures = {
            executor.submit(run_single_synthesis, path, name, board): (name, board)
            for path, name, board in configs
        }

        # Track completion
        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            results.append(result)

            status = "SUCCESS" if result.success else "FAILED"
            steps = len(result.trajectory)
            print(f"  [{completed}/{len(configs)}] {result.config_name}: {status} ({steps} steps)")

    print()
    return results


# ============== Replay ==============

def replay_trajectory(board_config: dict, trajectory: list[tuple[int, int]], title: str = ""):
    """
    Replay a single trajectory on the game board.
    """
    if not trajectory:
        print(f"\n{title}: No trajectory to replay.")
        return

    if title:
        print(f"\n--- {title} ---")

    # Run the game in replay mode
    game_script = Path("games/tfrozen_lake_game.py")

    # Convert trajectory to the format expected by the game
    trajectory_str = ";".join(f"{x},{y}" for x, y in trajectory)

    # Create a temporary config file for the board
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(board_config, f)
        config_file = f.name

    cmd = [
        sys.executable, str(game_script),
        "--replay", trajectory_str,
        "--replay-config", config_file
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        pass
    finally:
        os.unlink(config_file)


def replay_all_trajectories(results: list[SynthesisResult]):
    """
    Replay all synthesis results sequentially with automatic transitions.
    """
    import time

    successful_results = [r for r in results if r.trajectory]

    if not successful_results:
        print("\nNo trajectories to replay.")
        return

    print_header("STEP 4: Replay Controllers")
    print("Watch the synthesized controllers play the game!")
    print()
    input("Press Enter to start replay...")

    for i, result in enumerate(successful_results, 1):
        replay_trajectory(
            result.board_config,
            result.trajectory,
            title=f"Controller {i}/{len(successful_results)}: {result.config_name}"
        )

        # 2 second delay between replays (but not after the last one)
        if i < len(successful_results):
            time.sleep(2)


# ============== Main ==============

def main():
    global DEBUG

    parser = argparse.ArgumentParser(
        description="TSL_f Specification Mining Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py frozen_lake                    # Interactive play
  python demo.py frozen_lake --gen-traces 10   # Auto-generate 10+10 traces
  python demo.py frozen_lake --random-placements

This demo will:
  1. Generate traces (play the game or auto-generate)
  2. Mine a temporal specification from traces
  3. Synthesize controllers on different game configurations
  4. Watch the controllers play the game
"""
    )

    parser.add_argument(
        "game",
        choices=["frozen_lake"],
        help="Game to run the demo with (currently only frozen_lake is supported)"
    )

    parser.add_argument(
        "--random-placements",
        action="store_true",
        help="Use random goal and hole placements (default: fixed layout)"
    )

    parser.add_argument(
        "--gen-traces",
        type=int,
        metavar="N",
        help="Auto-generate N positive and N negative traces (skip interactive play)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show detailed output from mining and synthesis"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for varied configuration generation (default: 42)"
    )

    args = parser.parse_args()

    DEBUG = args.debug

    print_header("TSL_f SPECIFICATION MINING DEMO", "=")
    print("This demo showcases the complete specification mining pipeline:")
    print("  1. Generate traces (play the game or auto-generate)")
    print("  2. Mine a temporal specification from traces")
    print("  3. Synthesize controllers on different game configurations")
    print("  4. Watch the controllers play the game")

    # Step 1: Generate traces
    if args.gen_traces:
        session_dir, training_board_config = run_auto_generation(
            args.game,
            args.gen_traces,
            random_placements=args.random_placements
        )
    else:
        session_dir, training_board_config = run_interactive_game(
            args.game,
            random_placements=args.random_placements
        )

    # Step 2: Run mining
    spec = run_mining(session_dir)

    # Step 3: Prepare 3 synthesis configurations
    # - 1 fixed (the training board)
    # - 2 varied (randomly generated)

    out_dir = session_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate varied configurations
    varied_configs = generate_varied_configs(num_configs=2, seed=args.seed)

    # Prepare all synthesis configs
    synthesis_configs = []

    # Fixed config (training board)
    fixed_config_path = out_dir / "synthesis_config_fixed.yaml"
    create_synthesis_config(training_board_config, spec, fixed_config_path, "fixed")
    synthesis_configs.append((fixed_config_path, "fixed", training_board_config))

    # Varied configs
    for i, varied_board in enumerate(varied_configs, 1):
        config_path = out_dir / f"synthesis_config_varied_{i}.yaml"
        create_synthesis_config(varied_board, spec, config_path, f"varied_{i}")
        synthesis_configs.append((config_path, f"varied_{i}", varied_board))

    print_debug(f"Created {len(synthesis_configs)} synthesis configs")

    # Step 3: Run parallel synthesis
    results = run_parallel_synthesis(synthesis_configs)

    # Step 4: Replay all trajectories
    replay_all_trajectories(results)

    # Summary
    print_header("DEMO COMPLETE")
    print(f"Mined specification: {spec}")
    print()
    print("Controller Results:")
    for result in results:
        status = "SUCCESS" if result.success else "FAILED"
        steps = len(result.trajectory) if result.trajectory else 0
        goal = result.board_config.get("goal", {})
        goal_str = f"({goal.get('x', '?')},{goal.get('y', '?')})"
        print(f"  {result.config_name}: {status} - {steps} steps - goal at {goal_str}")

    success_count = sum(1 for r in results if r.success)
    print()
    print(f"Overall: {success_count}/{len(results)} controllers reached the goal")

    print_debug(f"\nSession directory: {session_dir}")


if __name__ == "__main__":
    main()
