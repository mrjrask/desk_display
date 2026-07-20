#!/usr/bin/env python3
"""Run the full desk_display test suite.

This aggregates the regular pytest suite plus standalone diagnostic test scripts
under ``scripts/`` whose names match ``test_*.py``. New individual scripts added
with that naming convention are picked up automatically.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
THIS_FILE = Path(__file__).resolve()

STAGED_RUFF_CLEANUP_RULES = ("B", "C4", "PIE", "RUF", "SIM", "UP", "PLC", "PLE", "PLW")
STAGED_RUFF_CLEANUP_IGNORES = ("B008", "PLW0603")


@dataclass(frozen=True)
class TestCommand:
    """A command that participates in the aggregate test run."""

    name: str
    command: tuple[str, ...]


def _discover_standalone_scripts() -> list[Path]:
    """Return standalone test scripts, excluding this aggregate runner."""

    scripts = []
    for path in sorted(SCRIPTS_DIR.glob("test_*.py")):
        if path.resolve() == THIS_FILE:
            continue
        scripts.append(path)
    return scripts


def _build_commands(pytest_args: Sequence[str]) -> list[TestCommand]:
    commands = [
        TestCommand(
            name="pytest suite",
            command=(sys.executable, "-m", "pytest", *pytest_args),
        )
    ]

    for script in _discover_standalone_scripts():
        commands.append(
            TestCommand(
                name=f"standalone script: {script.relative_to(REPO_ROOT)}",
                command=(sys.executable, str(script.relative_to(REPO_ROOT))),
            )
        )

    return commands


def _build_lint_cleanup_command() -> TestCommand:
    """Return the report-only Ruff command for staged lint cleanup.

    Run isolated from pyproject so per-file ignores for staged legacy modules
    do not hide the violations this cleanup report is intended to surface. Vendored
    sensor libraries stay excluded because they are not part of the cleanup migration.
    """

    return TestCommand(
        name="staged Ruff cleanup report",
        command=(
            sys.executable,
            "-m",
            "ruff",
            "check",
            ".",
            "--isolated",
            "--target-version",
            "py311",
            "--line-length",
            "100",
            "--exclude",
            "vendor",
            "--select",
            ",".join(STAGED_RUFF_CLEANUP_RULES),
            "--ignore",
            ",".join(STAGED_RUFF_CLEANUP_IGNORES),
            "--exit-zero",
            "--statistics",
        ),
    )


def _print_command(command: TestCommand) -> None:
    shellish = " ".join(command.command)
    print(f"\n=== {command.name} ===", flush=True)
    print(f"$ {shellish}", flush=True)


def _run_command(command: TestCommand) -> int:
    _print_command(command)
    completed = subprocess.run(
        command.command, cwd=REPO_ROOT, env=os.environ.copy(), check=False
    )
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Run remaining test commands even after one fails.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the commands that would run without executing them.",
    )
    parser.add_argument(
        "--lint-cleanup",
        action="store_true",
        help=(
            "Include a report-only Ruff cleanup pass for the staged lint families; "
            "the command runs isolated from pyproject per-file ignores and uses "
            "--exit-zero so existing findings do not fail the suite."
        ),
    )
    parser.epilog = (
        "Any arguments not recognized by this runner are passed through to pytest. "
        "Use '--' before pytest arguments only when you need to pass a pytest "
        "argument that has the same name as an aggregate-runner option."
    )
    args, pytest_args = parser.parse_known_args(argv)

    pytest_args = list(pytest_args)
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]

    commands = _build_commands(pytest_args)
    if args.lint_cleanup:
        commands.append(_build_lint_cleanup_command())

    if args.list:
        for command in commands:
            print(" ".join(command.command))
        return 0

    failures: list[tuple[TestCommand, int]] = []
    for command in commands:
        returncode = _run_command(command)
        if returncode != 0:
            failures.append((command, returncode))
            if not args.continue_on_error:
                break

    if failures:
        print("\nFailed test command(s):", file=sys.stderr)
        for command, returncode in failures:
            print(f"- {command.name}: exit {returncode}", file=sys.stderr)
        return 1

    print("\nAll test commands passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
