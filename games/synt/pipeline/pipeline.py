"""
Pipeline orchestrator for the Spec Validator Pipeline.

Coordinates all pipeline stages: spec generation, synthesis, embedding, and execution.

Flow:
1. For each configuration:
   - For each objective under that configuration:
     a. Generate TSLMT spec with configuration parameters and objective
     b. Synthesize a controller
     c. Embed controller into game file
     d. Run game and validate (goal reached within timeout, no hazard violations)
     e. Record pass/fail
   - Calculate score for this configuration
2. Output summary with scores per configuration
"""

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import tempfile
from typing import Optional

from .config import PipelineConfig, ObjectiveSpec, GameConfiguration, validate_config
from .spec_generator import generate_spec
from .synthesizer import synthesize, SynthesisResult
from .embedder import embed_from_template
from .runner import run_game_once, build_game, RunResult
from .logger import PipelineLogger, DualOutput


@dataclass
class ObjectiveRunResult:
    """Result of running a single objective."""
    objective: str
    success: bool
    steps: Optional[int] = None
    error_message: Optional[str] = None
    synthesis_time: Optional[float] = None
    game_output: Optional[str] = None  # Full game output including trajectory


@dataclass
class ConfigurationResult:
    """Result of testing a configuration across all its objectives."""
    config_name: str
    objectives: list[ObjectiveRunResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for o in self.objectives if o.success)

    @property
    def total(self) -> int:
        return len(self.objectives)

    @property
    def score(self) -> str:
        if self.total == 0:
            return "0/0 (0%)"
        pct = (self.passed / self.total) * 100
        return f"{self.passed}/{self.total} ({pct:.1f}%)"


@dataclass
class PipelineResult:
    """Complete result of a pipeline run."""
    config_name: str
    configurations: list[ConfigurationResult] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None
    log_path: Optional[Path] = None


class Pipeline:
    """
    Main pipeline orchestrator.

    Usage:
        config = load_config("configs/ice_lake.yaml")
        pipeline = Pipeline(config)
        result = pipeline.run()
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, debug_override: Optional[bool] = None) -> PipelineResult:
        """
        Run the complete pipeline with all objectives and configurations.

        Args:
            debug_override: Override the debug setting from config

        Returns:
            PipelineResult with all objective results and scores
        """
        debug = debug_override if debug_override is not None else self.config.debug

        # Validate config
        errors = validate_config(self.config)
        if errors:
            return PipelineResult(
                config_name=self.config.name,
                success=False,
                error_message=f"Configuration errors: {'; '.join(errors)}",
            )

        # Set up logging
        logs_dir = self.config.root_dir / "logs"
        logger = PipelineLogger(self.config.name, logs_dir)

        result = PipelineResult(config_name=self.config.name)

        # Use DualOutput to capture all print statements to both console and log
        with DualOutput(logger):
            print("=" * 60)
            print(f"  Spec Validator Pipeline - {self.config.name}")
            print("=" * 60)
            print(f"\nUsing template-based generation for specs and games")
            print(f"Configurations: {len(self.config.configurations)}")

            # Process each configuration
            for cfg_idx, config in enumerate(self.config.configurations):
                cfg_result = self._run_config(config, cfg_idx + 1, debug)
                result.configurations.append(cfg_result)

            # Print summary
            self._print_summary(result)

        # Save log file
        log_path = logger.save()
        result.log_path = log_path
        print(f"\nLog saved to: {log_path}")

        return result

    def _run_config(
        self,
        config: GameConfiguration,
        cfg_num: int,
        debug: bool,
    ) -> ConfigurationResult:
        """Run all objectives for a single configuration."""
        print(f"\n{'=' * 60}")
        print(f"  Configuration {cfg_num}: {config.name}")
        print(f"  Objectives: {len(config.objectives)}")
        print("=" * 60)

        if debug:
            print(f"  Parameters: {config.params}")

        cfg_result = ConfigurationResult(config_name=config.name)

        for obj_idx, objective in enumerate(config.objectives):
            obj_result = self._run_objective(
                config,
                objective,
                obj_idx + 1,
                len(config.objectives),
                debug,
            )
            cfg_result.objectives.append(obj_result)

        print(f"\n  Configuration Score: {cfg_result.score}")
        return cfg_result

    def _run_objective(
        self,
        config: GameConfiguration,
        objective: ObjectiveSpec,
        obj_num: int,
        total_objs: int,
        debug: bool,
    ) -> ObjectiveRunResult:
        """Run a single objective: generate spec, synthesize, embed, run."""
        print(f"\n  --- Objective {obj_num}/{total_objs}: {objective.objective[:50]}{'...' if len(objective.objective) > 50 else ''} ---")
        print(f"  Timeout: {objective.timeout} steps")

        try:
            # Step 1: Generate spec
            print("  [1/4] Generating spec...")
            spec = generate_spec(self.config.name, config.params, objective.objective)

            if debug:
                print("  Generated spec preview:")
                for line in spec.split('\n')[:10]:
                    print(f"    {line}")
                print("    ...")

            # Save debug copy of spec to spec_output folder
            spec_output_dir = self.config.root_dir / "spec_output"
            spec_output_dir.mkdir(exist_ok=True)
            debug_name = config.name.replace(" ", "_").replace("/", "_")
            obj_suffix = f"_obj{obj_num}"
            spec_debug_file = spec_output_dir / f"{self.config.name}_{debug_name}{obj_suffix}_debug.tslmt"
            spec_debug_file.write_text(spec)
            print(f"  Spec saved to: {spec_debug_file}")

            # Write spec to temp file
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.tslmt', delete=False
            ) as f:
                f.write(spec)
                spec_path = Path(f.name)

            # Step 2: Synthesize
            timeout_str = f" (timeout: {self.config.synthesis.timeout_minutes}min)" if self.config.synthesis.timeout_minutes else ""
            print(f"  [2/4] Synthesizing controller...{timeout_str}")
            synthesis_result = synthesize(
                spec_path,
                command=self.config.synthesis.command,
                args=self.config.synthesis.args,
                debug=debug,
                timeout_minutes=self.config.synthesis.timeout_minutes,
            )

            # Clean up spec file
            spec_path.unlink()

            if not synthesis_result.success:
                if synthesis_result.timed_out:
                    print(f"  SKIP: Synthesis timed out after {self.config.synthesis.timeout_minutes} minutes")
                else:
                    print(f"  FAIL: Synthesis failed - {synthesis_result.error_message}")
                return ObjectiveRunResult(
                    objective=objective.objective,
                    success=False,
                    error_message=f"Synthesis failed: {synthesis_result.error_message}",
                    synthesis_time=synthesis_result.duration,
                )

            print(f"  Synthesis complete ({synthesis_result.duration:.1f}s)")

            # Step 3: Generate game from template and embed controller
            print("  [3/4] Generating game and embedding controller...")

            # Create temp directory for game build
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                game_file = tmp_dir / "game.c"

                # Generate complete game file from template + controller
                embed_from_template(
                    game_name=self.config.name,
                    params=config.params,
                    synthesis_output=synthesis_result.controller_code,
                    output_path=game_file,
                )

                # Save a debug copy to games folder
                debug_name = config.name.replace(" ", "_").replace("/", "_")
                obj_suffix = f"_obj{obj_num}"
                games_dir = self.config.root_dir / "games"
                games_dir.mkdir(exist_ok=True)
                debug_file = games_dir / f"{self.config.name}_{debug_name}{obj_suffix}_debug.c"
                shutil.copy(game_file, debug_file)
                print(f"  Debug copy saved to: {debug_file}")

                # Step 4: Build and run
                print("  [4/4] Building and running...")

                # Build game
                if not build_game(tmp_dir, debug=debug):
                    print("  FAIL: Build failed")
                    return ObjectiveRunResult(
                        objective=objective.objective,
                        success=False,
                        error_message="Build failed",
                        synthesis_time=synthesis_result.duration,
                    )

                # Run game
                run_result = run_game_once(
                    tmp_dir,
                    timeout_steps=objective.timeout,
                    config_params=config.params,
                    debug=debug,
                )

                if run_result.success:
                    print(f"  PASS: Goal reached in {run_result.steps} steps")
                else:
                    print(f"  FAIL: {run_result.error_message}")

                return ObjectiveRunResult(
                    objective=objective.objective,
                    success=run_result.success,
                    steps=run_result.steps,
                    error_message=run_result.error_message if not run_result.success else None,
                    synthesis_time=synthesis_result.duration,
                    game_output=run_result.output,  # Full trajectory log
                )

        except Exception as e:
            print(f"  FAIL: {str(e)}")
            return ObjectiveRunResult(
                objective=objective.objective,
                success=False,
                error_message=str(e),
            )

    def _print_summary(self, result: PipelineResult) -> None:
        """Print final summary of all configurations."""
        print("\n" + "=" * 60)
        print("  SUMMARY")
        print("=" * 60)

        for cfg_result in result.configurations:
            print(f"\n  {cfg_result.config_name}")
            for obj in cfg_result.objectives:
                status = "PASS" if obj.success else "FAIL"
                steps_str = f" ({obj.steps} steps)" if obj.steps else ""
                err_str = f" - {obj.error_message}" if obj.error_message else ""
                obj_preview = obj.objective[:40] + "..." if len(obj.objective) > 40 else obj.objective
                print(f"    {obj_preview}: {status}{steps_str}{err_str}")
            print(f"  Score: {cfg_result.score}")

        print("\n" + "=" * 60)
        total_passed = sum(cfg.passed for cfg in result.configurations)
        total_objs = sum(cfg.total for cfg in result.configurations)
        if total_objs > 0:
            total_pct = (total_passed / total_objs) * 100
            print(f"  TOTAL: {total_passed}/{total_objs} ({total_pct:.1f}%)")
        print("=" * 60)
