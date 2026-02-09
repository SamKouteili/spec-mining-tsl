"""
Local fallback queue for processing Modal timeout configs.

Runs timed-out configs locally with configurable concurrency and timeout.
Supports graceful shutdown - outputs partial results on Ctrl+C.

Usage:
    from modal_eval.local_fallback import LocalFallbackQueue

    queue = LocalFallbackQueue(max_workers=2, timeout_minutes=20)
    queue.add_timeout_configs(timeout_configs, spec_dir, objective, game)

    # Run and get results (blocks until done or cancelled)
    results = queue.run()
"""

import json
import signal
import subprocess
import sys
import tempfile
import threading
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
VALIDATOR_DIR = PROJECT_ROOT / "games" / "synt"


@dataclass
class LocalRunResult:
    """Result from a single local synthesis run."""
    task_id: str
    config: dict
    success: bool
    steps: Optional[int] = None
    synthesis_time: Optional[float] = None
    error: Optional[str] = None
    game_output: Optional[str] = None


@dataclass
class LocalFallbackResult:
    """Aggregated results from local fallback queue."""
    successes: int = 0
    failures: int = 0
    still_pending: int = 0
    results: list = field(default_factory=list)
    cancelled: bool = False
    n_value: int = None  # Track which n-value this is for


# Callback type for result updates
ResultCallback = callable  # (n_value: int, result: LocalRunResult) -> None


class LocalFallbackQueue:
    """
    Queue for processing Modal timeout configs locally.

    Features:
    - Configurable concurrency (default 2 workers)
    - 20 minute timeout per config
    - Graceful shutdown on Ctrl+C
    - Outputs partial results if cancelled
    """

    def __init__(
        self,
        max_workers: int = 2,
        timeout_minutes: int = 20,
        log_dir: Optional[Path] = None,
        on_result_callback: Optional[callable] = None  # Called when each result completes
    ):
        self.max_workers = max_workers
        self.timeout_minutes = timeout_minutes
        self.log_dir = log_dir
        self.on_result_callback = on_result_callback

        self.configs_to_process = []
        self.spec_dir = None
        self.objective = None
        self.game = None
        self.n_value = None

        # Results tracking
        self.results = LocalFallbackResult()
        self.results_lock = threading.Lock()

        # Cancellation handling
        self.cancelled = False
        self.executor: Optional[ThreadPoolExecutor] = None
        self.futures: list[Future] = []

        # Background execution
        self._background_thread: Optional[threading.Thread] = None
        self._is_running = False

    def add_timeout_configs(
        self,
        timeout_configs: list[dict],
        spec_dir: Path,
        objective: str,
        game: str = "frozen_lake",
        n_value: int = None
    ):
        """Add timeout configs from Modal to the local queue."""
        self.configs_to_process = timeout_configs
        self.spec_dir = spec_dir
        self.objective = objective
        self.game = game
        self.n_value = n_value
        self.results.still_pending = len(timeout_configs)

    def _create_config_yaml(self, config: dict, task_id: str) -> str:
        """Create a YAML config for a single run."""
        game_name = "ice_lake" if self.game == "frozen_lake" else self.game

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
                "name": task_id,
                "grid_size": grid_size,
                "start_pos": start_pos,
                "objectives": [{
                    "objective": self.objective,
                    "timeout": 1000
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
            # Build base run configuration for other game types
            run_config = {
                "name": task_id,
                "grid_size": config["grid_size"],
                "start_pos": config["start_pos"],
                "objectives": [{
                    "objective": self.objective,
                    "timeout": 1000
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
                "command": "issy",  # Use local issy binary
                "args": [
                    "--tslmt", "--synt",
                    "--pruning", "1",
                    "--accel", "no",
                ],
                "timeout_minutes": self.timeout_minutes
            },
            "debug": False,
            "run_configuration": [run_config]
        }

        return yaml.dump(yaml_config, default_flow_style=False)

    def _run_single_config(self, timeout_config: dict) -> LocalRunResult:
        """Run a single config locally using run_pipeline.py."""
        task_id = timeout_config.get("task_id", "unknown")
        config = timeout_config.get("config", {})

        if self.cancelled:
            return LocalRunResult(
                task_id=task_id,
                config=config,
                success=False,
                error="Cancelled before starting"
            )

        # Create temp config file
        config_yaml = self._create_config_yaml(config, task_id)

        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.yaml', delete=False
            ) as f:
                f.write(config_yaml)
                config_path = Path(f.name)

            game_name = "ice_lake" if self.game == "frozen_lake" else self.game

            # Run synthesis locally
            import time
            start_time = time.time()

            cmd = [
                sys.executable,
                str(VALIDATOR_DIR / "run_pipeline.py"),
                game_name,
                str(config_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_minutes * 60 + 30,  # Add buffer
                cwd=VALIDATOR_DIR
            )

            synthesis_time = time.time() - start_time

            # Parse output
            output = result.stdout + result.stderr
            success = "PASS:" in output

            steps = None
            if success:
                import re
                # Match "PASS (X steps)" or "PASS: Goal reached in X steps" - not "Timeout: 1000 steps"
                match = re.search(r'PASS[:\s].*?(\d+)\s*steps', output)
                if match:
                    steps = int(match.group(1))

            # Clean up
            config_path.unlink()

            return LocalRunResult(
                task_id=task_id,
                config=config,
                success=success,
                steps=steps,
                synthesis_time=synthesis_time,
                game_output=output if success else None,
                error=None if success else f"Synthesis failed: {output[-500:]}"
            )

        except subprocess.TimeoutExpired:
            return LocalRunResult(
                task_id=task_id,
                config=config,
                success=False,
                error=f"Local timeout after {self.timeout_minutes} minutes"
            )
        except Exception as e:
            return LocalRunResult(
                task_id=task_id,
                config=config,
                success=False,
                error=str(e)
            )

    def _handle_result(self, result: LocalRunResult):
        """Thread-safe result handling."""
        with self.results_lock:
            self.results.results.append(result)
            self.results.still_pending -= 1

            if result.success:
                self.results.successes += 1
            else:
                self.results.failures += 1

            # Print progress
            status = "PASS" if result.success else "FAIL"
            steps_str = f" ({result.steps} steps)" if result.steps else ""
            time_str = f" [{result.synthesis_time:.1f}s]" if result.synthesis_time else ""
            print(f"  [LOCAL n={self.n_value}] {result.task_id}: {status}{steps_str}{time_str}")

            # Call callback if provided
            if self.on_result_callback:
                try:
                    self.on_result_callback(self.n_value, result)
                except Exception as e:
                    print(f"  [LOCAL] Callback error: {e}")

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\n  [LOCAL] Cancellation requested - finishing current jobs...")
        self.cancelled = True
        self.results.cancelled = True

        # Don't raise exception - let current jobs finish
        # The run() method will check self.cancelled and stop submitting new work

    def run(self) -> LocalFallbackResult:
        """
        Run all queued configs locally.

        Returns LocalFallbackResult with:
        - successes: number of successful runs
        - failures: number of failed runs
        - still_pending: configs that weren't started (if cancelled)
        - results: list of LocalRunResult
        - cancelled: True if interrupted by Ctrl+C
        """
        if not self.configs_to_process:
            return self.results

        print(f"\n  [LOCAL FALLBACK] Processing {len(self.configs_to_process)} timeout configs")
        print(f"  [LOCAL FALLBACK] Workers: {self.max_workers}, Timeout: {self.timeout_minutes}min each")
        print(f"  [LOCAL FALLBACK] Press Ctrl+C to cancel (partial results will be saved)\n")

        # Set up signal handler for graceful shutdown
        original_handler = signal.signal(signal.SIGINT, self._signal_handler)

        try:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

            # Submit jobs
            for timeout_config in self.configs_to_process:
                if self.cancelled:
                    break

                future = self.executor.submit(self._run_single_config, timeout_config)
                self.futures.append(future)

            # Collect results as they complete
            for future in as_completed(self.futures):
                if self.cancelled and not future.done():
                    continue

                try:
                    result = future.result(timeout=1)
                    self._handle_result(result)
                except Exception as e:
                    print(f"  [LOCAL] Error: {e}")

            # Shutdown executor
            self.executor.shutdown(wait=True, cancel_futures=self.cancelled)

        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_handler)

        # Log results
        if self.log_dir:
            self._save_results()

        self._is_running = False
        return self.results

    def run_in_background(self) -> None:
        """
        Start processing in background thread.
        Use get_current_results() to check progress.
        Use cancel() to stop early.
        """
        if self._is_running:
            return

        self._is_running = True
        self.results.n_value = self.n_value

        def _run_wrapper():
            try:
                self._run_internal()
            except Exception as e:
                print(f"  [LOCAL n={self.n_value}] Background error: {e}")
            finally:
                self._is_running = False
                if self.log_dir:
                    self._save_results()

        self._background_thread = threading.Thread(target=_run_wrapper, daemon=True)
        self._background_thread.start()
        print(f"  [LOCAL FALLBACK n={self.n_value}] Started in background ({len(self.configs_to_process)} configs, {self.max_workers} workers)")

    def _run_internal(self) -> None:
        """Internal run logic without signal handling (for background use)."""
        if not self.configs_to_process:
            return

        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Submit jobs
        for timeout_config in self.configs_to_process:
            if self.cancelled:
                break
            future = self.executor.submit(self._run_single_config, timeout_config)
            self.futures.append(future)

        # Collect results as they complete
        for future in as_completed(self.futures):
            if self.cancelled and not future.done():
                continue
            try:
                result = future.result(timeout=1)
                self._handle_result(result)
            except Exception as e:
                print(f"  [LOCAL n={self.n_value}] Error: {e}")

        # Shutdown executor
        self.executor.shutdown(wait=True, cancel_futures=self.cancelled)

    def is_running(self) -> bool:
        """Check if background processing is still running."""
        return self._is_running

    def get_current_results(self) -> LocalFallbackResult:
        """Get current results (thread-safe snapshot)."""
        with self.results_lock:
            return LocalFallbackResult(
                successes=self.results.successes,
                failures=self.results.failures,
                still_pending=self.results.still_pending,
                results=list(self.results.results),
                cancelled=self.results.cancelled,
                n_value=self.n_value
            )

    def cancel(self) -> None:
        """Cancel background processing."""
        self.cancelled = True
        self.results.cancelled = True

    def wait(self, timeout: float = None) -> LocalFallbackResult:
        """Wait for background processing to complete."""
        if self._background_thread:
            self._background_thread.join(timeout=timeout)
        return self.results

    def _save_results(self):
        """Save local fallback results to log file."""
        if not self.log_dir:
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        log_file = self.log_dir / f"local_fallback_{timestamp}.json"

        output = {
            "timestamp": timestamp,
            "n_value": self.n_value,
            "spec_dir": str(self.spec_dir) if self.spec_dir else None,
            "objective": self.objective,
            "total_queued": len(self.configs_to_process),
            "successes": self.results.successes,
            "failures": self.results.failures,
            "still_pending": self.results.still_pending,
            "cancelled": self.results.cancelled,
            "results": [
                {
                    "task_id": r.task_id,
                    "config": r.config,
                    "success": r.success,
                    "steps": r.steps,
                    "synthesis_time": r.synthesis_time,
                    "error": r.error
                }
                for r in self.results.results
            ]
        }

        with open(log_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n  [LOCAL FALLBACK] Results saved to: {log_file}")


class BackgroundFallbackManager:
    """
    Manages multiple background local fallback queues.

    Tracks results across all n-values and provides live updates.
    """

    def __init__(
        self,
        max_workers: int = 2,
        timeout_minutes: int = 20,
        log_dir: Optional[Path] = None,
        on_update_callback: Optional[callable] = None  # Called when any result completes
    ):
        self.max_workers = max_workers
        self.timeout_minutes = timeout_minutes
        self.log_dir = log_dir
        self.on_update_callback = on_update_callback

        self.queues: dict[int, LocalFallbackQueue] = {}  # n_value -> queue
        self.results: dict[int, LocalFallbackResult] = {}  # n_value -> results
        self.results_lock = threading.Lock()

    def add_timeout_configs(
        self,
        n_value: int,
        timeout_configs: list[dict],
        spec_dir: Path,
        objective: str,
        game: str = "frozen_lake"
    ) -> None:
        """Add timeout configs for a specific n-value and start processing in background."""
        if not timeout_configs:
            return

        queue = LocalFallbackQueue(
            max_workers=self.max_workers,
            timeout_minutes=self.timeout_minutes,
            log_dir=self.log_dir,
            on_result_callback=self._on_result
        )

        queue.add_timeout_configs(
            timeout_configs=timeout_configs,
            spec_dir=spec_dir,
            objective=objective,
            game=game,
            n_value=n_value
        )

        self.queues[n_value] = queue
        self.results[n_value] = LocalFallbackResult(
            still_pending=len(timeout_configs),
            n_value=n_value
        )

        # Start in background
        queue.run_in_background()

    def _on_result(self, n_value: int, result: LocalRunResult) -> None:
        """Called when any individual result completes."""
        with self.results_lock:
            if n_value in self.results:
                r = self.results[n_value]
                r.results.append(result)
                r.still_pending -= 1
                if result.success:
                    r.successes += 1
                else:
                    r.failures += 1

        # Notify external callback
        if self.on_update_callback:
            try:
                self.on_update_callback(n_value, result, self.get_all_results())
            except Exception as e:
                print(f"  [FALLBACK MANAGER] Callback error: {e}")

    def get_all_results(self) -> dict[int, LocalFallbackResult]:
        """Get current results for all n-values (thread-safe)."""
        with self.results_lock:
            return {
                n: LocalFallbackResult(
                    successes=r.successes,
                    failures=r.failures,
                    still_pending=r.still_pending,
                    results=list(r.results),
                    cancelled=r.cancelled,
                    n_value=r.n_value
                )
                for n, r in self.results.items()
            }

    def get_result(self, n_value: int) -> Optional[LocalFallbackResult]:
        """Get results for a specific n-value."""
        with self.results_lock:
            if n_value in self.results:
                r = self.results[n_value]
                return LocalFallbackResult(
                    successes=r.successes,
                    failures=r.failures,
                    still_pending=r.still_pending,
                    results=list(r.results),
                    cancelled=r.cancelled,
                    n_value=r.n_value
                )
            return None

    def any_running(self) -> bool:
        """Check if any queues are still running."""
        return any(q.is_running() for q in self.queues.values())

    def cancel_all(self) -> None:
        """Cancel all background processing."""
        for queue in self.queues.values():
            queue.cancel()

    def wait_all(self, timeout: float = None) -> dict[int, LocalFallbackResult]:
        """Wait for all background processing to complete."""
        for queue in self.queues.values():
            queue.wait(timeout=timeout)
        return self.get_all_results()

    def get_pending_count(self) -> int:
        """Get total pending configs across all n-values."""
        with self.results_lock:
            return sum(r.still_pending for r in self.results.values())


def run_local_fallback(
    timeout_configs: list[dict],
    spec_dir: Path,
    objective: str,
    game: str = "frozen_lake",
    n_value: int = None,
    log_dir: Path = None,
    max_workers: int = 2,
    timeout_minutes: int = 20
) -> LocalFallbackResult:
    """
    Convenience function to run local fallback on timeout configs.

    Returns LocalFallbackResult that can be merged with Modal results.
    """
    queue = LocalFallbackQueue(
        max_workers=max_workers,
        timeout_minutes=timeout_minutes,
        log_dir=log_dir
    )

    queue.add_timeout_configs(
        timeout_configs=timeout_configs,
        spec_dir=spec_dir,
        objective=objective,
        game=game,
        n_value=n_value
    )

    return queue.run()
