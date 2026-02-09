"""
Synthesis API for RL loop.

Provides a clean interface to the games/synt pipeline for:
1. Taking mined safety/liveness specs
2. Synthesizing controllers via Issy
3. Running controllers and extracting traces in JSONL format

Usage:
    from src.rl.synt import SynthesisAPI, load_objective_from_dir

    api = SynthesisAPI(game="ice_lake")

    # Option 1: Load objective from directory with safety.tsl/liveness.tsl
    objective = load_objective_from_dir("path/to/out/", game="ice_lake")
    result = api.synthesize_and_run(
        objective=objective,
        params={"grid_size": 4, "goal": {"x": 3, "y": 3}, "holes": [{"x": 1, "y": 1}]}
    )

    # Option 2: Pass objective string directly (already transformed)
    result = api.synthesize_and_run(
        objective="F (((eq x goalx) && (eq y goaly)))",
        params={...}
    )

    # Option 3: Transform specs manually
    objective = api.transform_specs(safety="(X[!] foo) U (END)", liveness="F bar")
    result = api.synthesize_and_run(objective=objective, params={...})

    if result.success:
        trace = result.trace  # List of dicts in JSONL format
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional
import tempfile
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from games.synt.spec_transformer import transform_spec
from games.synt.pipeline.spec_generator import generate_spec
from games.synt.pipeline.synthesizer import synthesize, SynthesisResult
from games.synt.pipeline.embedder import embed_from_template
from games.synt.pipeline.runner import build_game, run_game_once, RunResult


#################################################################
################## SPEC LOADING HELPERS #########################
#################################################################

def load_objective_from_dir(
    spec_dir: Path,
    game: str = "ice_lake",
    verbose: bool = False,
) -> str:
    """
    Load and transform specs from a directory containing safety.tsl/liveness.tsl.

    This is a standalone helper function that reads spec files from a directory,
    transforms them to Issy-compatible format, and combines them into a single
    objective string.

    Args:
        spec_dir: Directory containing safety.tsl and/or liveness.tsl
        game: Game type for game-specific transformations
        verbose: Print transformation steps

    Returns:
        Combined and transformed objective string

    Raises:
        ValueError: If no spec files found in directory
    """
    spec_dir = Path(spec_dir)
    safety_file = spec_dir / "safety.tsl"
    liveness_file = spec_dir / "liveness.tsl"

    parts = []

    if liveness_file.exists():
        liveness = liveness_file.read_text().strip()
        if liveness:
            transformed = transform_spec(liveness, is_safety=False, game=game, verbose=verbose)
            parts.append(transformed)

    if safety_file.exists():
        safety = safety_file.read_text().strip()
        if safety:
            transformed = transform_spec(safety, is_safety=True, game=game, verbose=verbose)
            parts.append(transformed)

    if not parts:
        raise ValueError(f"No spec files found in {spec_dir}")

    # Combine with &&
    if len(parts) == 1:
        return parts[0]
    return " && ".join(f"({p})" for p in parts)


def combine_specs(
    safety: Optional[str] = None,
    liveness: Optional[str] = None,
    game: str = "ice_lake",
    verbose: bool = False,
) -> str:
    """
    Transform and combine safety/liveness spec strings into a single objective.

    Args:
        safety: Raw safety spec string (with U END pattern)
        liveness: Raw liveness spec string (F-rooted)
        game: Game type for game-specific transformations
        verbose: Print transformation steps

    Returns:
        Combined and transformed objective string

    Raises:
        ValueError: If neither safety nor liveness provided
    """
    parts = []

    if liveness:
        transformed = transform_spec(liveness, is_safety=False, game=game, verbose=verbose)
        parts.append(transformed)

    if safety:
        transformed = transform_spec(safety, is_safety=True, game=game, verbose=verbose)
        parts.append(transformed)

    if not parts:
        raise ValueError("At least one of safety or liveness must be provided")

    if len(parts) == 1:
        return parts[0]
    return " && ".join(f"({p})" for p in parts)


@dataclass
class TraceEntry:
    """A single timestep in a trace."""
    step: int
    state: dict[str, Any]


@dataclass
class SynthResult:
    """Result of synthesis + execution."""
    success: bool
    trace: list[dict[str, Any]] = field(default_factory=list)
    steps: Optional[int] = None
    synthesis_time: Optional[float] = None
    game_output: str = ""
    error_message: Optional[str] = None
    controller_code: str = ""
    spec_used: str = ""


@dataclass
class RandomRunResult:
    """Result of a random walk execution."""
    is_positive: bool  # True if goal reached, False if failed (hole, timeout, etc.)
    trace: list[dict[str, Any]] = field(default_factory=list)
    steps: Optional[int] = None
    game_output: str = ""
    failure_reason: Optional[str] = None  # None if positive, otherwise explains why negative


def _get_trace_parser(game: str):
    """
    Get the trace parser function for a game.

    Tries to import from the appropriate game module (e.g., toytext).
    Returns None if not found.
    """
    try:
        from src.rl.toytext import parse_trace_from_output
        return lambda output: parse_trace_from_output(output, game)
    except ImportError:
        return None


class SynthesisAPI:
    """
    API for synthesizing and running controllers from mined specs.

    This provides a high-level interface to the synthesis pipeline,
    handling spec transformation, synthesis, embedding, and execution.
    """

    def __init__(
        self,
        game: str = "ice_lake",
        synthesis_command: str = "issy",
        synthesis_args: Optional[list[str]] = None,
        synthesis_timeout_minutes: Optional[float] = None,
        debug: bool = False,
    ):
        """
        Initialize the synthesis API.

        Args:
            game: Game type (ice_lake, taxi, cliff_walking, blackjack)
            synthesis_command: Path/name of synthesis tool
            synthesis_args: Arguments for synthesis tool
            synthesis_timeout_minutes: Timeout for synthesis (None = no timeout)
            debug: Enable debug output
        """
        self.game = game
        self.synthesis_command = synthesis_command
        self.synthesis_args = synthesis_args or [
            "--tslmt", "--solve", "--synt",
            "--pruning", "1",
            "--accel", "no"
        ]
        self.synthesis_timeout = synthesis_timeout_minutes
        self.debug = debug

    def transform_specs(
        self,
        safety: Optional[str] = None,
        liveness: Optional[str] = None,
    ) -> str:
        """
        Transform mined specs to Issy-compatible format and combine them.

        This is a convenience method that uses the standalone combine_specs function.

        Args:
            safety: Safety spec string (with U END pattern)
            liveness: Liveness spec string (F-rooted)

        Returns:
            Combined and transformed objective string for synthesis
        """
        return combine_specs(
            safety=safety,
            liveness=liveness,
            game=self.game,
            verbose=self.debug,
        )

    def generate_tslmt_spec(self, params: dict[str, Any], objective: str) -> str:
        """
        Generate a complete TSLMT specification for synthesis.

        Args:
            params: Game-specific configuration parameters
            objective: The TSL guarantee objective (already transformed)

        Returns:
            Complete TSLMT specification string
        """
        return generate_spec(self.game, params, objective)

    def synthesize(self, spec: str) -> SynthesisResult:
        """
        Synthesize a controller from a TSLMT specification.

        Args:
            spec: Complete TSLMT specification string

        Returns:
            SynthesisResult with controller code or error info
        """
        # Write spec to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tslmt', delete=False) as f:
            f.write(spec)
            spec_path = Path(f.name)

        try:
            result = synthesize(
                spec_path,
                command=self.synthesis_command,
                args=self.synthesis_args,
                debug=self.debug,
                timeout_minutes=self.synthesis_timeout,
            )
            return result
        finally:
            spec_path.unlink(missing_ok=True)

    def run_controller(
        self,
        params: dict[str, Any],
        synthesis_output: str,
        timeout_steps: int = 1000,
        trace_parser: Optional[callable] = None,
    ) -> tuple[RunResult, list[dict[str, Any]]]:
        """
        Embed controller into game and run it.

        Args:
            params: Game-specific configuration parameters
            synthesis_output: Raw synthesis output containing controller code
            timeout_steps: Maximum steps before timeout
            trace_parser: Optional function to parse trace from output.
                Signature: (output: str) -> list[dict]. If None, uses default
                from game module (e.g., toytext).

        Returns:
            Tuple of (RunResult, trace as list of state dicts)
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            game_file = tmp_dir / "game.c"

            # Inject max_steps into params for game template
            game_params = {**params, "max_steps": timeout_steps}

            # Generate complete game file from template + controller
            embed_from_template(
                game_name=self.game,
                params=game_params,
                synthesis_output=synthesis_output,
                output_path=game_file,
            )

            # Build
            if not build_game(tmp_dir, debug=self.debug):
                return RunResult(
                    success=False,
                    error_message="Build failed"
                ), []

            # Run
            run_result = run_game_once(
                tmp_dir,
                timeout_steps=timeout_steps,
                config_params=params,
                debug=self.debug,
            )

            # Parse trace from output
            parser = trace_parser or _get_trace_parser(self.game)
            trace = parser(run_result.output) if parser else []

            return run_result, trace

    def synthesize_and_run(
        self,
        objective: str,
        params: dict[str, Any],
        timeout_steps: int = 50,
    ) -> SynthResult:
        """
        Full pipeline: generate spec -> synthesize -> run -> extract trace.

        Args:
            objective: The TSL objective string (already transformed to Issy format).
                       Use load_objective_from_dir() or combine_specs() to create this.
            params: Game-specific configuration parameters (grid_size, goal, holes, etc.)
            timeout_steps: Maximum steps for game execution

        Returns:
            SynthResult with trace and metadata
        """
        try:
            # Step 1: Generate TSLMT spec
            if self.debug:
                print("[1/3] Generating TSLMT spec...")
            tslmt_spec = self.generate_tslmt_spec(params, objective)

            # Step 2: Synthesize
            if self.debug:
                print("[2/3] Synthesizing controller...")
            synth_result = self.synthesize(tslmt_spec)

            if not synth_result.success:
                return SynthResult(
                    success=False,
                    error_message=f"Synthesis failed: {synth_result.error_message}",
                    synthesis_time=synth_result.duration,
                    spec_used=objective,
                )

            # Step 3: Run controller
            if self.debug:
                print("[3/3] Running controller...")
            run_result, trace = self.run_controller(
                params,
                synth_result.controller_code,
                timeout_steps=timeout_steps,
            )

            return SynthResult(
                success=run_result.success,
                trace=trace,
                steps=run_result.steps,
                synthesis_time=synth_result.duration,
                game_output=run_result.output,
                error_message=run_result.error_message if not run_result.success else None,
                controller_code=synth_result.controller_code,
                spec_used=objective,
            )

        except Exception as e:
            return SynthResult(
                success=False,
                error_message=str(e),
            )

    def synthesize_from_directory(
        self,
        spec_dir: Path,
        params: dict[str, Any],
        timeout_steps: int = 1000,
    ) -> SynthResult:
        """
        Synthesize and run from a directory containing liveness.tsl/safety.tsl.

        Convenience method that loads specs from a directory and runs synthesis.

        Args:
            spec_dir: Directory containing spec files
            params: Game-specific configuration parameters
            timeout_steps: Maximum steps for game execution

        Returns:
            SynthResult with trace and metadata
        """
        objective = load_objective_from_dir(spec_dir, game=self.game, verbose=self.debug)
        return self.synthesize_and_run(
            objective=objective,
            params=params,
            timeout_steps=timeout_steps,
        )

    def initial_random_run(
        self,
        params: dict[str, Any],
        timeout_steps: int = 1000,
        start_state: Optional[dict[str, Any]] = None,
        random_controller_generator: Optional[callable] = None,
    ) -> RandomRunResult:
        """
        Perform a random walk of the game and return the trace with outcome.

        This generates a random controller that makes random valid moves,
        runs it on the game, and returns whether the trace is positive
        (reached goal) or negative (failed due to hole, cliff, timeout, etc.).

        Args:
            params: Game-specific configuration parameters (grid_size, goal, holes, etc.)
            timeout_steps: Maximum steps before timeout (negative trace)
            start_state: Optional state dict to start from (e.g., {"player": [2, 3]}
                or {"x": 2, "y": 3}). Overrides params["start_pos"]. Useful for
                continuing a timed-out trace from its final position.
            random_controller_generator: Optional custom generator function.
                If None, uses the default from toytext module (for ToyText games).
                Signature: (params: dict) -> str (C code)

        Returns:
            RandomRunResult with:
                - is_positive: True if goal reached, False otherwise
                - trace: List of state dicts (JSONL format)
                - steps: Number of steps taken
                - game_output: Raw game output
                - failure_reason: If negative, explains why (hole, cliff, timeout, etc.)
        """
        # Convert start_state to start_pos format if provided
        if start_state is not None:
            if "player" in start_state:
                start_pos = {"x": start_state["player"][0], "y": start_state["player"][1]}
            else:
                start_pos = {"x": start_state.get("x", 0), "y": start_state.get("y", 0)}
            params = {**params, "start_pos": start_pos}
        # Get the random controller generator
        if random_controller_generator is not None:
            generator = random_controller_generator
        else:
            # Try to import from toytext for ToyText games
            try:
                from src.rl.toytext import get_random_controller_generator
                generator = get_random_controller_generator(self.game)
            except ImportError:
                generator = None

        if generator is None:
            return RandomRunResult(
                is_positive=False,
                failure_reason=f"No random controller generator for game: {self.game}. "
                               f"Provide one via random_controller_generator parameter."
            )

        # Generate random controller code
        controller_code = generator(params)

        if self.debug:
            print(f"[random_run] Generated random controller for {self.game}")

        # Run the controller using existing infrastructure
        # We need to create a "synthesis output" format that embed_from_template expects
        # The random controller IS the full code, so we wrap it appropriately
        synthesis_output = controller_code

        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                game_file = tmp_dir / "game.c"

                # Import game template generator
                from games.synt.pipeline.game_templates import generate_game

                # Inject max_steps into params for game template
                game_params = {**params, "max_steps": timeout_steps}

                # Generate game harness
                game_harness = generate_game(self.game, game_params)

                # Combine harness + random controller
                complete_code = game_harness + "\n/* RANDOM CONTROLLER */\n" + controller_code

                game_file.write_text(complete_code)

                # Build
                if not build_game(tmp_dir, debug=self.debug):
                    return RandomRunResult(
                        is_positive=False,
                        failure_reason="Build failed"
                    )

                # Run
                run_result = run_game_once(
                    tmp_dir,
                    timeout_steps=timeout_steps,
                    config_params=params,
                    debug=self.debug,
                )

                # Parse trace
                parser = _get_trace_parser(self.game)
                trace = parser(run_result.output) if parser else []

                # Determine if positive or negative
                is_positive = run_result.success
                failure_reason = None if is_positive else run_result.error_message

                return RandomRunResult(
                    is_positive=is_positive,
                    trace=trace,
                    steps=run_result.steps,
                    game_output=run_result.output,
                    failure_reason=failure_reason,
                )

        except Exception as e:
            return RandomRunResult(
                is_positive=False,
                failure_reason=str(e),
            )


def write_trace_jsonl(trace: list[dict[str, Any]], output_path: Path) -> None:
    """Write trace to a JSONL file."""
    with open(output_path, 'w') as f:
        for entry in trace:
            f.write(json.dumps(entry) + '\n')


def load_config_file(config_path: Path, config_index: int = 0) -> dict[str, Any]:
    """
    Load configuration from a JSON file generated by game --config commands.

    Args:
        config_path: Path to JSON config file (single config or array of configs)
        config_index: If file contains array, which index to use (default: 0)

    Returns:
        Configuration dictionary
    """
    with open(config_path) as f:
        data = json.load(f)

    # Handle array of configs
    if isinstance(data, list):
        if config_index >= len(data):
            raise ValueError(f"Config index {config_index} out of range (file has {len(data)} configs)")
        return data[config_index]

    return data


# CLI for quick testing
if __name__ == "__main__":
    import argparse
    from src.rl.toytext import config_to_params

    parser = argparse.ArgumentParser(
        description="Synthesis API CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a config file from a game, then use it
  python games/tfrozen_lake_game.py --config 1 --config-output my_config.json
  python src/rl/synt.py spec_dir/ --config my_config.json

  # Using a config file with multiple configs (pick index 2)
  python games/tfrozen_lake_game.py --config 5 --random-placements --config-output configs.json
  python src/rl/synt.py spec_dir/ --config configs.json --config-index 2

  # Different games
  python games/cliff_walking_game.py --config 1 --config-output cw.json
  python src/rl/synt.py spec_dir/ --game cliff_walking --config cw.json
"""
    )
    parser.add_argument("spec_dir", type=Path, help="Directory with safety.tsl/liveness.tsl")
    parser.add_argument("--game", "-g", default="ice_lake",
                        choices=["ice_lake", "frozen_lake", "cliff_walking", "taxi", "blackjack"],
                        help="Game type (default: ice_lake)")
    parser.add_argument("--config", "-c", type=Path, required=True,
                        help="JSON config file from game --config command")
    parser.add_argument("--config-index", type=int, default=0,
                        help="If config file contains array, which index to use (default: 0)")
    parser.add_argument("--timeout", type=int, default=1000, help="Step timeout")
    parser.add_argument("--output", "-o", type=Path, help="Output JSONL file for trace")
    parser.add_argument("--debug", "-d", action="store_true", help="Debug mode")

    args = parser.parse_args()

    # Load config from file
    if args.debug:
        print(f"Loading config from: {args.config}")

    config = load_config_file(args.config, args.config_index)
    params = config_to_params(config, args.game)

    if args.debug:
        print(f"Config name: {config.get('name', 'unnamed')}")
        print(f"Params: {json.dumps(params, indent=2)}")

    api = SynthesisAPI(game=args.game, debug=args.debug)
    result = api.synthesize_from_directory(
        spec_dir=args.spec_dir,
        params=params,
        timeout_steps=args.timeout,
    )

    print(f"\nSuccess: {result.success}")
    if result.error_message:
        print(f"Error: {result.error_message}")
    if result.steps:
        print(f"Steps: {result.steps}")
    if result.synthesis_time:
        print(f"Synthesis time: {result.synthesis_time:.1f}s")

    print(f"\nTrace ({len(result.trace)} entries):")
    for entry in result.trace[:10]:
        print(f"  {entry}")
    if len(result.trace) > 10:
        print(f"  ... ({len(result.trace) - 10} more)")

    if args.output and result.trace:
        write_trace_jsonl(result.trace, args.output)
        print(f"\nTrace written to: {args.output}")
