#!/usr/bin/env python3
"""
Convenience wrapper for the Spec Validator Pipeline.

Usage:
    python run_pipeline.py ice_lake                       # uses configs/ice_lake.yaml
    python run_pipeline.py taxi --debug                   # uses configs/taxi.yaml with debug
    python run_pipeline.py taxi path/to/custom.yaml       # uses custom config for taxi game
    python run_pipeline.py --help

This runs all objectives and configurations defined in the config file.
"""

import argparse
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipeline import Pipeline, load_config


def main():
    parser = argparse.ArgumentParser(
        description="Spec Validator Pipeline - Multi-Objective Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py ice_lake                       # uses configs/ice_lake.yaml
  python run_pipeline.py taxi --debug                   # uses configs/taxi.yaml
  python run_pipeline.py taxi path/to/custom.yaml       # uses custom config for taxi
  python run_pipeline.py cliff_walking ./my_config.yaml # uses custom config for cliff_walking

The pipeline will:
  1. Read objectives and configurations from the config file
  2. For each objective, test all its configurations:
     - Generate TSLMT spec with configuration parameters
     - Synthesize a controller
     - Embed controller into game file
     - Run validation
  3. Calculate scores (passed/total) per objective
""",
    )

    parser.add_argument(
        "game_name",
        type=str,
        help="Game type (ice_lake, taxi, cliff_walking, blackjack)",
    )

    parser.add_argument(
        "config_file",
        type=str,
        nargs="?",
        default=None,
        help="Optional path to custom config file (defaults to configs/{game_name}.yaml)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )

    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available configurations and exit",
    )

    args = parser.parse_args()

    # List configs if requested
    if args.list_configs:
        configs_dir = project_root / "configs"
        if configs_dir.exists():
            print("Available configurations:")
            for f in sorted(configs_dir.glob("*.yaml")):
                print(f"  - {f.stem}")
        else:
            print("No configs directory found")
        return 0

    # Determine config file path
    if args.config_file:
        # User specified a custom config file
        config_path = Path(args.config_file)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}", file=sys.stderr)
            return 1
    else:
        # Use default config based on game name
        config_path = project_root / "configs" / f"{args.game_name}.yaml"
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}", file=sys.stderr)
            print(f"\nAvailable configs:", file=sys.stderr)
            configs_dir = project_root / "configs"
            if configs_dir.exists():
                for f in sorted(configs_dir.glob("*.yaml")):
                    print(f"  - {f.stem}", file=sys.stderr)
            return 1

    # Load config
    try:
        config = load_config(config_path, root_dir=project_root)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    # Override the game name from command line (in case custom config has different name)
    config.name = args.game_name

    # Run pipeline
    pipeline = Pipeline(config)
    result = pipeline.run(debug_override=args.debug if args.debug else None)

    # Return exit code based on overall success
    if result.error_message:
        print(f"\nPipeline error: {result.error_message}", file=sys.stderr)
        return 1

    # Calculate overall success rate
    total_passed = sum(cfg.passed for cfg in result.configurations)
    total_objectives = sum(cfg.total for cfg in result.configurations)

    if total_objectives == 0:
        print("\nNo objectives were run", file=sys.stderr)
        return 1

    # Return 0 if at least one configuration passed
    if total_passed > 0:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
