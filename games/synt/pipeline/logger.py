"""
Logger module for the Spec Validator Pipeline.

Provides a logger that captures output to both console and a log file.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional
import sys


class PipelineLogger:
    """
    Logger that writes to both console and a log file.

    Usage:
        logger = PipelineLogger("ice_lake", logs_dir)
        logger.log("Starting pipeline...")
        # ... pipeline runs ...
        logger.save()  # Writes to logs/ice_lake_2024-01-15_14-30-45.log
    """

    def __init__(self, game_name: str, logs_dir: Path):
        """
        Initialize the logger.

        Args:
            game_name: Name of the game (used in log filename)
            logs_dir: Directory to save log files
        """
        self.game_name = game_name
        self.logs_dir = logs_dir
        self.lines: list[str] = []
        self.start_time = datetime.now()

        # Ensure logs directory exists
        self.logs_dir.mkdir(exist_ok=True)

        # Generate log filename with timestamp
        timestamp = self.start_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.log_filename = f"{game_name}_{timestamp}.log"
        self.log_path = self.logs_dir / self.log_filename

    def log(self, message: str = "", end: str = "\n") -> None:
        """
        Log a message to both console and the internal buffer.

        Args:
            message: The message to log
            end: Line ending (default: newline)
        """
        # Print to console
        print(message, end=end)

        # Store in buffer
        self.lines.append(message + end)

    def log_section(self, title: str, char: str = "=", width: int = 60) -> None:
        """Log a section header."""
        self.log(char * width)
        self.log(f"  {title}")
        self.log(char * width)

    def save(self) -> Path:
        """
        Save the log buffer to file.

        Returns:
            Path to the saved log file
        """
        # Add footer with summary
        end_time = datetime.now()
        duration = end_time - self.start_time

        self.lines.append("\n")
        self.lines.append("=" * 60 + "\n")
        self.lines.append(f"  Log saved: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.lines.append(f"  Total duration: {duration.total_seconds():.1f}s\n")
        self.lines.append("=" * 60 + "\n")

        # Write to file
        self.log_path.write_text("".join(self.lines))

        return self.log_path

    def get_log_path(self) -> Path:
        """Get the path where the log will be saved."""
        return self.log_path


class DualOutput:
    """
    Context manager that captures stdout to both console and a logger.

    Usage:
        logger = PipelineLogger("ice_lake", logs_dir)
        with DualOutput(logger):
            print("This goes to both console and logger")
    """

    def __init__(self, logger: PipelineLogger):
        self.logger = logger
        self.original_stdout = None

    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        return False

    def write(self, message: str) -> None:
        """Write to both original stdout and logger buffer."""
        if self.original_stdout:
            self.original_stdout.write(message)
        if message:  # Don't add empty strings
            self.logger.lines.append(message)

    def flush(self) -> None:
        """Flush the original stdout."""
        if self.original_stdout:
            self.original_stdout.flush()
