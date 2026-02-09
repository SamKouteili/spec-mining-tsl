#!/usr/bin/env python3
"""
Synthesis Evaluation Script

This script takes mined TSL specifications, transforms them to Issy format,
generates random board configurations, and evaluates the synthesized controllers.

Usage:
    python synthesis_eval.py <spec_dir> --num-configs 100 --output results.json

Where spec_dir contains:
    - liveness.tsl
    - safety.tsl
    - (or spec.tsl for combined)
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "games"))

from spec_transformer import load_and_transform_specs


@dataclass
class ConfigResult:
    """Result from a single configuration."""
    config_name: str
    grid_size: int
    goal: dict
    holes: list
    success: bool
    steps: Optional[int] = None
    synthesis_time: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class SynthesisEvalResult:
    """Complete evaluation result."""
    spec_dir: str
    objective: str
    num_configs: int
    num_successes: int
    success_rate: float
    avg_steps: Optional[float]
    avg_synthesis_time: Optional[float]
    total_time: float
    configs: list
    liveness_original: Optional[str] = None
    safety_original: Optional[str] = None
    timestamp: str = ""


def run_specification_validator(
    config_yaml_path: Path,
    validator_dir: Path,
    debug: bool = False
) -> dict:
    """
    Run the specification_validator pipeline.

    Args:
        config_yaml_path: Path to the generated config YAML
        validator_dir: Path to the specification_validator repo
        debug: Whether to print debug output

    Returns:
        Dictionary with results parsed from the pipeline output
    """
    # Run the pipeline
    cmd = [
        sys.executable,  # Use current Python
        str(validator_dir / "run_pipeline.py"),
        "ice_lake",  # Config name
    ]
    if debug:
        cmd.append("--debug")

    # We need to copy our config to the validator's configs directory
    validator_config_path = validator_dir / "configs" / "ice_lake.yaml"

    # Backup existing config if present
    backup_path = None
    if validator_config_path.exists():
        backup_path = validator_config_path.with_suffix('.yaml.bak')
        validator_config_path.rename(backup_path)

    try:
        # Copy our config
        import shutil
        shutil.copy(config_yaml_path, validator_config_path)

        # Run pipeline
        result = subprocess.run(
            cmd,
            cwd=validator_dir,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        output = result.stdout + result.stderr

        if debug:
            print("Pipeline output:")
            print(output)

        # Parse results from output
        return parse_pipeline_output(output)

    finally:
        # Restore backup
        if backup_path and backup_path.exists():
            if validator_config_path.exists():
                validator_config_path.unlink()
            backup_path.rename(validator_config_path)


def parse_pipeline_output(output: str) -> dict:
    """
    Parse the pipeline output to extract results.

    Returns:
        Dictionary with config results
    """
    results = {
        "configs": [],
        "total_passed": 0,
        "total_configs": 0
    }

    lines = output.split('\n')

    current_config = None
    for line in lines:
        line = line.strip()

        # Look for config start
        if "--- Config" in line and ":" in line:
            # Extract config name
            # Format: "--- Config 1/3: config_1 ---"
            parts = line.split(":")
            if len(parts) >= 2:
                config_name = parts[1].strip().rstrip(" ---")
                current_config = {"name": config_name}

        # Look for PASS/FAIL
        if current_config:
            if line.startswith("PASS:"):
                # Extract steps: "PASS: Goal reached in 7 steps"
                import re
                match = re.search(r'(\d+)\s*steps', line)
                steps = int(match.group(1)) if match else None
                current_config["success"] = True
                current_config["steps"] = steps
                results["configs"].append(current_config)
                results["total_passed"] += 1
                results["total_configs"] += 1
                current_config = None

            elif line.startswith("FAIL:"):
                current_config["success"] = False
                current_config["error"] = line.replace("FAIL:", "").strip()
                results["configs"].append(current_config)
                results["total_configs"] += 1
                current_config = None

            elif line.startswith("SKIP:"):
                current_config["success"] = False
                current_config["error"] = line.replace("SKIP:", "").strip()
                results["configs"].append(current_config)
                results["total_configs"] += 1
                current_config = None

        # Look for synthesis time
        if "Synthesis complete" in line:
            import re
            match = re.search(r'\((\d+\.?\d*)s\)', line)
            if match and current_config:
                current_config["synthesis_time"] = float(match.group(1))

        # Look for TOTAL line
        if "TOTAL:" in line:
            import re
            match = re.search(r'(\d+)/(\d+)', line)
            if match:
                results["total_passed"] = int(match.group(1))
                results["total_configs"] = int(match.group(2))

    return results


def evaluate_specs(
    spec_dir: Path,
    num_configs: int,
    validator_dir: Path,
    output_path: Optional[Path] = None,
    random_size: bool = False,
    random_placements: bool = True,
    timeout_steps: int = 1000,
    synthesis_timeout: int = 10,
    debug: bool = False,
    game: str = "frozen_lake"
) -> SynthesisEvalResult:
    """
    Main evaluation function.

    Args:
        spec_dir: Directory containing mined specs
        num_configs: Number of random configs to test
        validator_dir: Path to specification_validator repo
        output_path: Path to write results JSON
        random_size: Whether to randomize board sizes
        random_placements: Whether to randomize placements
        timeout_steps: Max steps for game execution
        synthesis_timeout: Synthesis timeout in minutes
        debug: Whether to print debug output

    Returns:
        SynthesisEvalResult with all results
    """
    start_time = time.time()

    print("=" * 60)
    print("Synthesis Evaluation")
    print("=" * 60)
    print(f"Spec directory: {spec_dir}")
    print(f"Number of configs: {num_configs}")
    print(f"Validator directory: {validator_dir}")
    print()

    # Step 1: Load and transform specs
    print("[1/4] Loading and transforming specs...")
    specs = load_and_transform_specs(spec_dir, game=game)

    if not specs.get('final'):
        raise ValueError(f"No valid specs found in {spec_dir}")

    objective = specs['final']
    print(f"  Original liveness: {specs.get('liveness_original', 'N/A')}")
    print(f"  Original safety: {specs.get('safety_original', 'N/A')}")
    print(f"  Transformed objective: {objective[:80]}...")
    print()

    # Step 2: Generate random configs
    print("[2/4] Generating random configurations...")

    # Import the config generator from games
    games_dir = Path(__file__).parent.parent / "games"
    sys.path.insert(0, str(games_dir))
    from tfrozen_lake_game import generate_config_yaml

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_path = Path(f.name)

    generate_config_yaml(
        objective=objective,
        num_configs=num_configs,
        output_path=config_path,
        random_size=random_size,
        random_placements=random_placements,
        timeout_steps=timeout_steps,
        synthesis_timeout=synthesis_timeout
    )
    print(f"  Generated {num_configs} configurations")
    print(f"  Config written to: {config_path}")
    print()

    # Step 3: Run pipeline
    print("[3/4] Running specification_validator pipeline...")
    pipeline_results = run_specification_validator(
        config_path,
        validator_dir,
        debug=debug
    )
    print(f"  Pipeline complete")
    print()

    # Step 4: Collate results
    print("[4/4] Collating results...")

    # Load the config to get full details
    import yaml
    with open(config_path) as f:
        config_yaml = yaml.safe_load(f)

    configs_details = config_yaml['run_configuration'][0]['configurations']

    # Build result list
    config_results = []
    successful_steps = []
    synthesis_times = []

    for i, cfg in enumerate(configs_details):
        pipeline_cfg = pipeline_results['configs'][i] if i < len(pipeline_results['configs']) else {}

        result = ConfigResult(
            config_name=cfg['name'],
            grid_size=cfg['grid_size'],
            goal=cfg['goal'],
            holes=cfg['holes'],
            success=pipeline_cfg.get('success', False),
            steps=pipeline_cfg.get('steps'),
            synthesis_time=pipeline_cfg.get('synthesis_time'),
            error_message=pipeline_cfg.get('error')
        )
        config_results.append(result)

        if result.success and result.steps:
            successful_steps.append(result.steps)
        if result.synthesis_time:
            synthesis_times.append(result.synthesis_time)

    num_successes = sum(1 for r in config_results if r.success)
    success_rate = num_successes / num_configs if num_configs > 0 else 0
    avg_steps = sum(successful_steps) / len(successful_steps) if successful_steps else None
    avg_synth_time = sum(synthesis_times) / len(synthesis_times) if synthesis_times else None

    total_time = time.time() - start_time

    result = SynthesisEvalResult(
        spec_dir=str(spec_dir),
        objective=objective,
        num_configs=num_configs,
        num_successes=num_successes,
        success_rate=success_rate,
        avg_steps=avg_steps,
        avg_synthesis_time=avg_synth_time,
        total_time=total_time,
        configs=[asdict(c) for c in config_results],
        liveness_original=specs.get('liveness_original'),
        safety_original=specs.get('safety_original'),
        timestamp=datetime.now().isoformat()
    )

    # Print summary
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Success rate: {num_successes}/{num_configs} ({success_rate*100:.1f}%)")
    if avg_steps:
        print(f"Average steps (successful): {avg_steps:.1f}")
    if avg_synth_time:
        print(f"Average synthesis time: {avg_synth_time:.1f}s")
    print(f"Total evaluation time: {total_time:.1f}s")
    print()

    # Cleanup
    config_path.unlink()

    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        print(f"Results saved to: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate mined TSL specs via synthesis on random boards'
    )
    parser.add_argument(
        'spec_dir',
        type=str,
        help='Directory containing mined specs (liveness.tsl, safety.tsl)'
    )
    parser.add_argument(
        '--num-configs', '-n',
        type=int,
        default=100,
        help='Number of random configurations to test (default: 100)'
    )
    parser.add_argument(
        '--validator-dir',
        type=str,
        default='../specification_validator',
        help='Path to specification_validator repo (default: ../specification_validator)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--random-size',
        action='store_true',
        help='Randomize board sizes (4-6)'
    )
    parser.add_argument(
        '--timeout-steps',
        type=int,
        default=1000,
        help='Max steps for game execution (default: 1000)'
    )
    parser.add_argument(
        '--synthesis-timeout',
        type=int,
        default=10,
        help='Synthesis timeout in minutes (default: 10)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output'
    )
    parser.add_argument(
        '--game', '-g',
        type=str,
        default='frozen_lake',
        choices=['frozen_lake', 'taxi', 'cliff_walking', 'generic'],
        help='Game for spec transformations (default: frozen_lake)'
    )
    args = parser.parse_args()

    spec_dir = Path(args.spec_dir)
    if not spec_dir.exists():
        print(f"Error: Spec directory not found: {spec_dir}", file=sys.stderr)
        sys.exit(1)

    validator_dir = Path(args.validator_dir).resolve()
    if not validator_dir.exists():
        print(f"Error: Validator directory not found: {validator_dir}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else None

    result = evaluate_specs(
        spec_dir=spec_dir,
        num_configs=args.num_configs,
        validator_dir=validator_dir,
        output_path=output_path,
        random_size=args.random_size,
        random_placements=True,  # Always randomize placements
        timeout_steps=args.timeout_steps,
        synthesis_timeout=args.synthesis_timeout,
        debug=args.debug,
        game=args.game
    )

    # Exit with error if success rate is 0
    if result.num_successes == 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
