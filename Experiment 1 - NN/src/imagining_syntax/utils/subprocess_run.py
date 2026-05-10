"""Shared CLI utilities for the experiment runners."""
import subprocess
import sys


def run_command(cmd, description, *, show_output=True):
    """Run a shell command and handle errors.

    Args:
        cmd: The shell command string.
        description: Human-readable label for logs.
        show_output: If True, print captured stdout after the subprocess exits.
            On failure, stdout/stderr are always printed regardless of this flag.
            Non-empty stderr is printed on success regardless of this flag, so
            errors and warnings remain loud even when normal output is suppressed.

    Returns:
        The command's stdout as a string. Exits with status 1 on failure.
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:  # pragma: no cover - subprocess error reporting; suite never injects a failing sub-runner
        print(f"ERROR: {description} failed!")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)

    if show_output:
        print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")

    return result.stdout
