"""
Synthesizer module for the Spec Validator Pipeline.

Handles interaction with the Issy synthesis tool.
"""

from dataclasses import dataclass
from pathlib import Path
import subprocess
import time
import threading
from typing import Optional


@dataclass
class SynthesisResult:
    """Result of a synthesis operation."""
    success: bool
    controller_code: str
    output: str
    duration: float
    error_message: Optional[str] = None
    timed_out: bool = False


def synthesize(
    spec_path: Path,
    command: str = "issy",
    args: Optional[list[str]] = None,
    debug: bool = False,
    timeout_minutes: Optional[float] = None,
) -> SynthesisResult:
    """
    Synthesize a controller from a TSLMT specification.

    Args:
        spec_path: Path to the .tslmt specification file
        command: The synthesis command (default: "issy")
        args: Additional arguments for the synthesis tool
        debug: If True, print synthesis output live
        timeout_minutes: Maximum time in minutes for synthesis (None = no timeout)

    Returns:
        SynthesisResult with the controller code or error information
    """
    if args is None:
        args = [
            "--tslmt", "--solve", "--synt", "--info",
            "--pruning", "1",
            "--accel-attr", "geom-ext",
            "--accel-difficulty", "easy"
        ]

    full_command = [command] + args + [str(spec_path)]

    # Convert timeout to seconds
    timeout_seconds = timeout_minutes * 60 if timeout_minutes is not None else None

    # DEBUG
    timeout_str = f" (timeout: {timeout_minutes}min)" if timeout_minutes else ""
    print(f"Running synthesis command{timeout_str}: {' '.join(full_command)}")

    start_time = time.time()

    try:
        process = subprocess.Popen(
            full_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_lines = []
        timed_out = False

        def read_output():
            """Read output from the process."""
            for line in iter(process.stdout.readline, ''):
                if debug:
                    print(line.rstrip())
                output_lines.append(line)
            process.stdout.close()

        # Start a thread to read output
        reader_thread = threading.Thread(target=read_output)
        reader_thread.start()

        # Wait for process with optional timeout
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            print(f"  Synthesis timeout ({timeout_minutes} minutes) - killing process...")
            process.kill()
            process.wait()  # Ensure process is fully terminated

        # Wait for reader thread to finish
        reader_thread.join(timeout=5)

        duration = time.time() - start_time
        output = ''.join(output_lines)

        if timed_out:
            return SynthesisResult(
                success=False,
                controller_code="",
                output=output,
                duration=duration,
                error_message=f"Synthesis timed out after {timeout_minutes} minutes",
                timed_out=True,
            )

        if process.returncode != 0:
            # Include last 10 lines of output in error message
            error_lines = output.strip().split('\n')[-10:]
            error_detail = '\n'.join(error_lines) if error_lines else "(no output)"
            # Print the full output to logs for debugging
            print(f"Synthesis FAILED (code {process.returncode}). Output:")
            print(output if output else "(no output)")
            return SynthesisResult(
                success=False,
                controller_code="",
                output=output,
                duration=duration,
                error_message=f"Synthesis failed with return code {process.returncode}:\n{error_detail}",
            )

        return SynthesisResult(
            success=True,
            controller_code=output,
            output=output,
            duration=duration,
        )

    except FileNotFoundError:
        return SynthesisResult(
            success=False,
            controller_code="",
            output="",
            duration=time.time() - start_time,
            error_message=f"Synthesis command not found: {command}",
        )
    except Exception as e:
        return SynthesisResult(
            success=False,
            controller_code="",
            output="",
            duration=time.time() - start_time,
            error_message=str(e),
        )