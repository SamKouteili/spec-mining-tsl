"""
Modal-based specification evaluator.

Provides parallel synthesis evaluation using Modal's cloud infrastructure.
This is a drop-in replacement for local sequential evaluation.

Usage:
    from modal_eval.modal_evaluator import evaluate_spec_on_configs_modal

    result = evaluate_spec_on_configs_modal(
        spec_dir=Path("path/to/spec"),
        test_configs=[{...}, {...}],
        game="frozen_lake",
        max_concurrent=50
    )
"""

import json
import re
import time
import yaml
from datetime import datetime
from pathlib import Path
import sys

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
sys.path.insert(0, str(PROJECT_ROOT / "games" / "synt"))  # For spec_transformer

# Import local fallback queue for processing timeouts
from modal_eval.local_fallback import run_local_fallback


def get_timestamp() -> str:
    """Get current timestamp string for log file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_config_yaml(
    objective: str,
    config: dict,
    config_name: str,
    game_name: str = "ice_lake",
    variable_updates: dict | None = None,
    timeout_minutes: int = 30,  # 30 min is safe - all successful syntheses complete under 10 min
    timeout_steps: int = 1000
) -> str:
    """Generate a config YAML for a single configuration."""
    # Handle taxi game config format conversion
    if game_name == "taxi":
        # Taxi configs use 'size' and 'taxi_start' instead of 'grid_size' and 'start_pos'
        grid_size = config.get("grid_size", config.get("size", 5))

        # Convert start position
        start_pos = config.get("start_pos")
        if start_pos is None:
            taxi_start = config.get("taxi_start", (0, 0))
            if isinstance(taxi_start, (list, tuple)):
                start_pos = {"x": taxi_start[0], "y": taxi_start[1]}
            else:
                start_pos = taxi_start

        run_config = {
            "name": config_name,
            "grid_size": grid_size,
            "start_pos": start_pos,
            "objectives": [{
                "objective": objective,
                "timeout": timeout_steps
            }]
        }

        # Convert locations to colored_cells format
        locations = config.get("locations", {})
        colored_cells = {}
        color_map = {'R': 'red', 'G': 'green', 'Y': 'yellow', 'B': 'blue'}
        for short_name, pos in locations.items():
            full_name = color_map.get(short_name, short_name.lower())
            if isinstance(pos, (list, tuple)):
                colored_cells[full_name] = {"x": pos[0], "y": pos[1]}
            else:
                colored_cells[full_name] = pos
        run_config["colored_cells"] = colored_cells

        # Map pickup/dropoff colors
        passenger_name = config.get("passenger_name", "R")
        dest_name = config.get("dest_name", "G")
        run_config["pickup_color"] = color_map.get(passenger_name, passenger_name.lower())
        run_config["dropoff_color"] = color_map.get(dest_name, dest_name.lower())

        # Convert walls to barriers format
        walls = config.get("walls", set())
        barriers = []
        seen = set()
        if isinstance(walls, set):
            for wall in walls:
                if isinstance(wall, tuple) and len(wall) == 2:
                    (x1, y1), (x2, y2) = wall
                    key = tuple(sorted([(x1, y1), (x2, y2)]))
                    if key not in seen:
                        seen.add(key)
                        barriers.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
        run_config["barriers"] = barriers
    else:
        # Build the run configuration for other game types
        run_config = {
            "name": config_name,
            "grid_size": config["grid_size"],
            "start_pos": config["start_pos"],
            "objectives": [{
                "objective": objective,
                "timeout": timeout_steps
            }]
        }

        # Add game-specific fields
        if game_name == "cliff_walking":
            # Cliff walking uses goal_pos, grid_rows, and cliff dimensions
            run_config["goal_pos"] = config.get("goal", config.get("goal_pos", {"x": 11, "y": 0}))
            run_config["grid_rows"] = config.get("grid_rows", config["grid_size"])
            run_config["cliff_min"] = config.get("cliff_min", 1)
            run_config["cliff_max"] = config.get("cliff_max", config["grid_size"] - 2)
            run_config["cliff_height"] = config.get("cliff_height", 1)
        else:
            # Ice lake / frozen lake uses goal and holes
            run_config["goal"] = config["goal"]
            run_config["holes"] = config.get("holes", [])

    yaml_config = {
        "name": game_name,
        "synthesis": {
            "command": "issy-optimized",  # Use optimized binary (14% faster than static)
            "args": [
                "--tslmt", "--synt",
                "--pruning", "1",
                "--accel", "no",  # No acceleration is 12% faster for finite-state specs
                # "--accel-attr", "polycomp-ext",
                # "--accel-difficulty", "hard"
            ],
            "timeout_minutes": timeout_minutes
        },
        "debug": True,
        "run_configuration": [run_config]
    }

    # Add variable_updates if provided
    if variable_updates:
        yaml_config["variable_updates"] = variable_updates

    return yaml.dump(yaml_config, default_flow_style=False)


def evaluate_spec_on_configs_modal(
    spec_dir: Path,
    test_configs: list[dict],
    log_dir: Path | None = None,
    game: str = "frozen_lake",
    max_concurrent: int = 100,
    timeout_minutes: int = 45,  # 30 min is safe - all successful syntheses complete under 10 min
    timeout_steps: int = 1000,
    n_value: int | None = None,  # n value for organizing timeout configs by training size
    run_local_fallback_on_timeouts: bool = False,  # Whether to run timeout configs locally
    local_fallback_workers: int = 2,  # Number of concurrent local workers
    local_fallback_timeout: int = 20  # Local timeout in minutes
) -> dict:
    """
    Evaluate mined spec on test configurations using Modal for parallel synthesis.

    Args:
        spec_dir: Path to directory containing mined specs (liveness.tsl, safety.tsl)
        test_configs: List of test configuration dicts
        log_dir: Optional directory for logging
        game: Game type (frozen_lake, taxi, etc.)
        max_concurrent: Maximum concurrent Modal tasks
        timeout_minutes: Synthesis timeout per config (default 30 min)
        timeout_steps: Game execution timeout
        n_value: Training n value for organizing timeout configs (e.g., 5, 10, 15...)
        run_local_fallback_on_timeouts: If True, run timeout configs locally after Modal
        local_fallback_workers: Number of concurrent local workers (default 2)
        local_fallback_timeout: Local timeout in minutes (default 20)

    Returns:
        dict with successes, total, timeouts, avg_steps, details
        - Timeouts are NOT counted as successes or failures
        - success_rate = successes / (total - timeouts)
        - If local fallback is enabled, results are merged into the main counts
    """
    from spec_transformer import load_and_transform_specs

    # Import Modal app here to avoid import issues at module load time
    from modal_eval.synthesis_app import app, synthesize_and_validate

    # Load and transform specs
    specs = load_and_transform_specs(spec_dir, game=game)
    objective = specs.get('final')
    variable_updates = specs.get('variable_updates', {})

    if not objective:
        return {"error": "No valid spec found", "successes": 0, "total": len(test_configs)}

    # Map game name
    game_name = "ice_lake" if game == "frozen_lake" else game

    print(f"  Evaluating spec on {len(test_configs)} configs using Modal...")
    print(f"  Objective: {objective}")
    print(f"  Max concurrent: {max_concurrent}")

    start_time = time.time()

    # Prepare all config YAMLs
    tasks = []
    for i, config in enumerate(test_configs):
        task_id = f"config_{i+1}"
        config_yaml = generate_config_yaml(
            objective=objective,
            config=config,
            config_name=task_id,
            game_name=game_name,
            variable_updates=variable_updates,
            timeout_minutes=timeout_minutes,
            timeout_steps=timeout_steps
        )
        tasks.append({
            "task_id": task_id,
            "config_yaml": config_yaml,
            "game_name": game_name,
            "config": config
        })

    # DEBUG: Print all generated configs and exit before using Modal credits
    print("\n" + "="*80)
    print("DEBUG: Generated config YAMLs (exiting before Modal call)")
    print("="*80)
    for i, task in enumerate(tasks):
        print(f"\n--- Config {i+1}: {task['task_id']} (game: {task['game_name']}) ---")
        print(task['config_yaml'])
    print("="*80)
    print(f"Total configs: {len(tasks)}")
    print("DEBUG EXIT: Remove this block to actually run Modal")
    print("="*80 + "\n")
    # sys.exit(0)

    # Run Modal tasks in parallel using starmap
    # Modal handles parallelization internally
    results = {}

    # Prepare arguments for starmap
    config_yamls = [t["config_yaml"] for t in tasks]
    game_names = [t["game_name"] for t in tasks]
    task_ids = [t["task_id"] for t in tasks]

    # Use Modal's starmap for parallel execution
    # The app.run() context runs the Modal app and allows calling .starmap()
    modal_run = app.run()
    with modal_run:
        # starmap takes iterable of (arg1, arg2, arg3) tuples
        modal_results = list(
            synthesize_and_validate.starmap(
                zip(config_yamls, game_names, task_ids)
            )
        )

        for i, modal_result in enumerate(modal_results):
            task_id = tasks[i]["task_id"]
            results[task_id] = {
                "task": tasks[i],
                "result": modal_result
            }
            # Print result as it comes in with failure type distinction
            result_entry = modal_result.get("results", [{}])[0] if modal_result.get("results") else {}
            failure_type = result_entry.get("failure_type", "unknown")
            steps = result_entry.get("steps")
            synth_time = modal_result.get("synthesis_time", 0)

            # Format status with failure type
            if failure_type == "success":
                status = "PASS"
            elif failure_type == "unrealizable":
                status = "UNREALIZABLE"  # Spec has no winning strategy
            elif failure_type == "synthesis_timeout":
                status = "SYNTH_TIMEOUT"
            elif failure_type == "execution_fail":
                status = "EXEC_FAIL"  # Controller ran but didn't reach goal
            else:
                status = "SYNTH_ERROR"

            steps_str = f" ({steps} steps)" if steps else ""
            error = modal_result.get("error_message", "") or result_entry.get("error", "")
            error_str = f" - {error[:60]}" if error else ""
            print(f"    [{i+1}/{len(tasks)}] {task_id}: {status}{steps_str} [{synth_time:.1f}s]{error_str}")

            # Print path/trajectory if synthesis succeeded (pass or fail execution)
            game_output = result_entry.get("game_output", "")
            if game_output and failure_type in ("success", "execution_fail"):
                # Extract coordinates from game output
                # Format: "Step N: Position (x,y)" -> extract just "(x,y)"
                positions = re.findall(r'Position \((\d+),(\d+)\)', game_output)
                if positions:
                    # Format as compact path: (0,0)→(1,0)→(1,1)...
                    path_coords = [f"({x},{y})" for x, y in positions]
                    path_str = "→".join(path_coords[:20])  # Show first 20 positions
                    print(f"         Path: {path_str}")
                    if len(positions) > 20:
                        print(f"         ... ({len(positions) - 20} more positions)")
                # Also show final outcome line (SUCCESS/FAIL)
                outcome_lines = [l.strip() for l in game_output.split('\n') if 'SUCCESS' in l or 'FAIL' in l]
                if outcome_lines:
                    print(f"         Outcome: {outcome_lines[-1]}")

    elapsed = time.time() - start_time
    print(f"  Modal evaluation complete in {elapsed:.1f}s")

    # Aggregate results - track timeouts separately
    successes = 0
    timeouts = 0
    timeout_configs = []  # Store configs that timed out for later analysis
    steps_list = []
    details = []

    for task_id, data in results.items():
        result = data["result"]
        config = data["task"]["config"]
        result_entry = result.get("results", [{}])[0] if result.get("results") else {}
        failure_type = result_entry.get("failure_type", "unknown")

        # Track timeouts separately - they don't count as success or failure
        if failure_type == "synthesis_timeout":
            timeouts += 1
            timeout_configs.append({
                "task_id": task_id,
                "config": config,
                "synthesis_time": result.get("synthesis_time"),
                "error": result.get("error_message") or result_entry.get("error")
            })
        elif result.get("success") and result.get("passed", 0) > 0:
            successes += 1
            # Extract steps from results
            for r in result.get("results", []):
                if r.get("passed") and r.get("steps"):
                    steps_list.append(r["steps"])

        details.append({
            "config": config.get("name", task_id),
            "success": result.get("success", False),
            "failure_type": failure_type,
            "passed": result.get("passed", 0),
            "total": result.get("total", 1),
            "steps": result_entry.get("steps"),
            "error": result.get("error_message") or result_entry.get("error"),
            "synthesis_time": result.get("synthesis_time"),
            "game_output": result_entry.get("game_output"),  # Trajectory log
        })

    # Calculate effective total (excluding timeouts)
    effective_total = len(test_configs) - timeouts

    # Log results with timestamps
    timestamp = get_timestamp()
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

        # Main evaluation log with timestamp
        log_file = log_dir / f"modal_eval_{timestamp}.log"
        with open(log_file, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"Modal Evaluation - {timestamp}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Spec dir: {spec_dir}\n")
            f.write(f"Objective: {objective}\n")
            f.write(f"N value: {n_value if n_value else 'N/A'}\n")
            f.write(f"\n")
            f.write(f"Results: {successes}/{effective_total} (excluding {timeouts} timeouts)\n")
            f.write(f"  - Successes: {successes}\n")
            f.write(f"  - Failures: {effective_total - successes}\n")
            f.write(f"  - Timeouts: {timeouts} (not counted)\n")
            f.write(f"Total time: {elapsed:.1f}s\n")
            f.write(f"{'='*60}\n\n")

            for d in details:
                ft = d.get("failure_type", "unknown")
                if ft == "success":
                    status = "PASS"
                elif ft == "unrealizable":
                    status = "UNREALIZABLE"
                elif ft == "synthesis_timeout":
                    status = "SYNTH_TIMEOUT"
                elif ft == "execution_fail":
                    status = "EXEC_FAIL"
                else:
                    status = "SYNTH_ERROR"
                f.write(f"  {d['config']}: {status}")
                if d.get("steps"):
                    f.write(f" ({d['steps']} steps)")
                if d.get("synthesis_time"):
                    f.write(f" [{d['synthesis_time']:.1f}s]")
                if d.get("error"):
                    f.write(f" - {d['error'][:100]}")
                f.write("\n")

        # Also append to cumulative log for easy viewing
        with open(log_dir / "modal_eval.log", 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"[{timestamp}] Results: {successes}/{effective_total} (timeouts: {timeouts})\n")
            f.write(f"Spec dir: {spec_dir}\n")
            f.write(f"{'='*60}\n")

        # Detailed JSON results with timestamp
        detailed_file = log_dir / f"modal_eval_detailed_{timestamp}.json"
        detailed_output = {
            "timestamp": timestamp,
            "spec_dir": str(spec_dir),
            "objective": objective,
            "variable_updates": variable_updates,
            "n_value": n_value,
            "total_configs": len(test_configs),
            "successes": successes,
            "timeouts": timeouts,
            "effective_total": effective_total,
            "elapsed_time": elapsed,
            "results": []
        }
        for task_id, data in results.items():
            detailed_output["results"].append({
                "task_id": task_id,
                "config": data["task"]["config"],
                "result": data["result"]
            })
        with open(detailed_file, 'w') as f:
            json.dump(detailed_output, f, indent=2)

        # Save timeout configs to separate file for analysis
        if timeout_configs:
            n_folder = f"n_{n_value}" if n_value else "n_unknown"
            timeout_dir = log_dir / "timeouts" / n_folder
            timeout_dir.mkdir(parents=True, exist_ok=True)

            timeout_file = timeout_dir / f"timeout_configs_{timestamp}.json"
            with open(timeout_file, 'w') as f:
                json.dump({
                    "timestamp": timestamp,
                    "spec_dir": str(spec_dir),
                    "objective": objective,
                    "n_value": n_value,
                    "timeout_count": len(timeout_configs),
                    "configs": timeout_configs
                }, f, indent=2)

            print(f"  Saved {len(timeout_configs)} timeout configs to: {timeout_file}")

        # Trajectories log with timestamp
        trajectories_file = log_dir / f"trajectories_{timestamp}.log"
        with open(trajectories_file, 'w') as f:
            f.write(f"Trajectories for {spec_dir}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Objective: {objective}\n")
            f.write("=" * 60 + "\n\n")
            for d in details:
                f.write(f"Config: {d['config']}\n")
                ft = d.get("failure_type", "unknown")
                if ft == "success":
                    f.write("Status: PASS\n")
                elif ft == "synthesis_timeout":
                    f.write("Status: TIMEOUT (excluded from results)\n")
                else:
                    f.write(f"Status: FAIL ({ft})\n")
                f.write(f"Steps: {d.get('steps', 'N/A')}\n")
                if d.get('game_output'):
                    f.write("Trajectory:\n")
                    f.write(d['game_output'])
                else:
                    f.write("Trajectory: N/A (synthesis failed or timed out)\n")
                f.write("\n" + "-" * 40 + "\n\n")

    # Print summary with raw numbers (not percentages)
    print(f"\n  Summary: {successes}/{effective_total} passed (excluding {timeouts} timeouts)")

    # Run local fallback on timeout configs if enabled
    local_fallback_results = None
    if run_local_fallback_on_timeouts and timeout_configs:
        print(f"\n  Starting local fallback for {len(timeout_configs)} timeout configs...")

        local_result = run_local_fallback(
            timeout_configs=timeout_configs,
            spec_dir=spec_dir,
            objective=objective,
            game=game,
            n_value=n_value,
            log_dir=log_dir,
            max_workers=local_fallback_workers,
            timeout_minutes=local_fallback_timeout
        )

        local_fallback_results = {
            "successes": local_result.successes,
            "failures": local_result.failures,
            "still_pending": local_result.still_pending,
            "cancelled": local_result.cancelled,
            "results": [
                {
                    "task_id": r.task_id,
                    "config": r.config,
                    "success": r.success,
                    "steps": r.steps,
                    "synthesis_time": r.synthesis_time,
                    "error": r.error
                }
                for r in local_result.results
            ]
        }

        # Merge local results into main results
        # Local successes reduce the timeout count and add to successes
        local_successes = local_result.successes
        local_completed = local_result.successes + local_result.failures

        successes += local_successes
        timeouts -= local_completed  # Remove completed from timeout count
        effective_total = len(test_configs) - timeouts

        # Add steps from local results
        for r in local_result.results:
            if r.success and r.steps:
                steps_list.append(r.steps)

        # Print updated summary
        if local_result.cancelled:
            print(f"\n  [LOCAL FALLBACK] Cancelled - partial results merged")
        print(f"\n  Updated Summary: {successes}/{effective_total} passed (excluding {timeouts} remaining timeouts)")

    return {
        "successes": successes,
        "total": len(test_configs),
        "timeouts": timeouts,
        "effective_total": effective_total,  # total - timeouts
        # Success rate only considers non-timeout configs
        "success_rate": successes / effective_total if effective_total > 0 else 0,
        "steps": steps_list,
        "avg_steps": sum(steps_list) / len(steps_list) if steps_list else None,
        "total_time": elapsed,
        "details": details,
        "timeout_configs": timeout_configs,
        "local_fallback_results": local_fallback_results
    }


# Test function
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Modal evaluator")
    parser.add_argument("--spec-dir", type=str, required=True, help="Path to spec directory")
    parser.add_argument("--num-configs", type=int, default=5, help="Number of test configs")
    parser.add_argument("--game", type=str, default="frozen_lake", help="Game type")

    args = parser.parse_args()

    # Generate simple test configs
    import random
    random.seed(42)

    test_configs = []
    for i in range(args.num_configs):
        size = 3  # Small for testing
        goal_x = random.randint(1, size - 1)
        goal_y = random.randint(1, size - 1)
        test_configs.append({
            "name": f"test_config_{i+1}",
            "grid_size": size,
            "start_pos": {"x": 0, "y": 0},
            "goal": {"x": goal_x, "y": goal_y},
            "holes": []
        })

    result = evaluate_spec_on_configs_modal(
        spec_dir=Path(args.spec_dir),
        test_configs=test_configs,
        game=args.game,
        max_concurrent=10
    )

    print(f"\nResult: {result['successes']}/{result['effective_total']} passed (excluding {result['timeouts']} timeouts)")
