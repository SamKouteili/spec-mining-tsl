"""
Runner module for the Spec Validator Pipeline.

Handles game compilation and execution.
"""

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
from typing import Optional, Any


# Simple Makefile for game compilation (no cJSON dependency)
MAKEFILE_CONTENT = """
CC := gcc
CFLAGS := -std=c11 -O0 -g -Wall
TARGET := game_binary
SRC := game.c

all: $(TARGET)

$(TARGET): $(SRC)
\t$(CC) $(CFLAGS) -o $@ $^

clean:
\trm -f $(TARGET)
"""


@dataclass
class RunResult:
    """Result of a single game run."""
    success: bool
    steps: Optional[int] = None
    return_code: int = 0
    output: str = ""
    error_message: Optional[str] = None


def build_game(tmp_dir: Path, debug: bool = False) -> bool:
    """
    Build the game in a temporary directory.

    Args:
        tmp_dir: Directory containing the game source
        debug: If True, print build output

    Returns:
        True if build succeeded
    """
    # Write Makefile
    (tmp_dir / "Makefile").write_text(MAKEFILE_CONTENT)

    process = subprocess.Popen(
        ["make"],
        cwd=tmp_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines = []
    for line in process.stdout:
        output_lines.append(line)
        if debug:
            print(f"    {line}", end="")

    process.stdout.close()
    process.wait()

    if process.returncode != 0:
        print(f"    Build failed with code {process.returncode}")
        print("    Output:", "".join(output_lines[-20:]))  # Last 20 lines

    return process.returncode == 0


def run_game_once(
    tmp_dir: Path,
    timeout_steps: int = 1000,
    config_params: Optional[dict[str, Any]] = None,
    debug: bool = False,
    timeout_seconds: int = 60,
) -> RunResult:
    """
    Run the game binary once and check for success.

    The game is expected to:
    - Exit with code 0 on success (goal reached)
    - Exit with non-zero code on failure (timeout, hazard, etc.)
    - Print "in X steps" or "X steps" somewhere in output

    Args:
        tmp_dir: Directory containing the compiled game binary
        timeout_steps: Maximum steps before considering it a timeout
        config_params: Game configuration parameters (for validation)
        debug: If True, print game output live
        timeout_seconds: Wall-clock timeout in seconds

    Returns:
        RunResult with success status and step count
    """
    cmd = ["./game_binary"]

    try:
        process = subprocess.Popen(
            cmd,
            cwd=tmp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_lines = []
        for line in process.stdout:
            output_lines.append(line)
            if debug:
                print(f"    {line}", end="")

        process.stdout.close()
        process.wait(timeout=timeout_seconds)

        output = "".join(output_lines)
        return_code = process.returncode

        # Parse steps from output
        steps = parse_steps_from_output(output)

        # Check for success
        if return_code == 0:
            # Goal reached
            return RunResult(
                success=True,
                steps=steps,
                return_code=return_code,
                output=output,
            )
        else:
            # Determine failure reason
            error_msg = determine_failure_reason(output, return_code, steps, timeout_steps)
            return RunResult(
                success=False,
                steps=steps,
                return_code=return_code,
                output=output,
                error_message=error_msg,
            )

    except subprocess.TimeoutExpired:
        process.kill()
        return RunResult(
            success=False,
            return_code=-1,
            error_message=f"Wall-clock timeout ({timeout_seconds}s) exceeded",
        )
    except Exception as e:
        return RunResult(
            success=False,
            return_code=-1,
            error_message=str(e),
        )


def parse_steps_from_output(output: str) -> Optional[int]:
    """Extract the number of steps from game output."""
    # Try various patterns
    patterns = [
        r"in (\d+) steps",
        r"(\d+) steps",
        r"Step[s]?: (\d+)",
        r"Total steps: (\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


def determine_failure_reason(
    output: str,
    return_code: int,
    steps: Optional[int],
    timeout_steps: int,
) -> str:
    """Determine the reason for a game failure."""
    output_lower = output.lower()

    # Check for specific failure messages
    if "hole" in output_lower or "fell" in output_lower:
        return "Fell into hole"
    if "cliff" in output_lower:
        return "Fell off cliff"
    if "barrier" in output_lower or "obstacle" in output_lower:
        return "Hit barrier"
    if "timeout" in output_lower or "step limit" in output_lower:
        return f"Step timeout ({timeout_steps} steps)"
    if steps and steps >= timeout_steps:
        return f"Step timeout ({steps} >= {timeout_steps})"
    if "bounds" in output_lower or "out of" in output_lower:
        return "Out of bounds"

    # Generic failure
    if return_code != 0:
        return f"Exit code {return_code}"

    return "Unknown failure"
