#!/usr/bin/env python3
"""Reject new Ruff per-file suppressions while allowing baseline reductions."""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

tomllib = importlib.import_module("tomllib") if sys.version_info >= (3, 11) else None

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE = {
    "utils.py": {"C420", "PLC0415", "RUF002", "SIM102", "SIM105", "UP006", "UP015", "UP035"},
    "data_fetch.py": {
        "PIE808",
        "RUF001",
        "SIM102",
        "SIM105",
        "UP006",
        "UP015",
        "UP017",
        "UP028",
        "UP031",
        "UP034",
        "UP035",
    },
    "main.py": {
        "C420",
        "PLC0415",
        "PLW0602",
        "PLW2901",
        "RUF001",
        "SIM102",
        "SIM103",
        "SIM105",
        "UP006",
        "UP017",
        "UP035",
    },
    "vendor/**/*.py": {
        "B007",
        "B904",
        "PLC0415",
        "RUF100",
        "SIM105",
        "SIM118",
        "UP024",
        "UP030",
        "UP032",
        "UP034",
    },
    "scripts/waveshare_oled_status.py": {"BLE001"},
    "scripts/test_api_connections.py": {"BLE001"},
}


def find_baseline_drift(configured: dict[str, list[str]]) -> list[str]:
    """Return additions or removals that are not reflected on both sides."""

    drift = []
    configured_sets = {module: set(rules) for module, rules in configured.items()}
    for module in BASELINE.keys() | configured_sets.keys():
        approved = BASELINE.get(module, set())
        current = configured_sets.get(module, set())
        drift.extend(f"{module}: added {rule}" for rule in current - approved)
        drift.extend(
            f"{module}: baseline still contains removed {rule}" for rule in approved - current
        )
    return sorted(drift)


def _parse_per_file_ignores(config_text: str) -> dict[str, list[str]]:
    """Parse the simple string-list entries in Ruff's per-file-ignore table.

    This dependency-free fallback keeps the check runnable on Python 3.10,
    where the standard-library ``tomllib`` module is not available.
    """

    section_name = "[tool.ruff.lint.per-file-ignores]"
    in_section = False
    configured = {}
    for raw_line in config_text.splitlines():
        line = raw_line.strip()
        if line == section_name:
            in_section = True
            continue
        if in_section and line.startswith("["):
            break
        if not in_section or not line or line.startswith("#"):
            continue

        key_text, separator, rules_text = line.partition("=")
        if not separator:
            raise ValueError(f"Invalid per-file-ignore entry: {raw_line}")
        module = ast.literal_eval(key_text.strip())
        rules = ast.literal_eval(rules_text.strip())
        if (
            not isinstance(module, str)
            or not isinstance(rules, list)
            or not all(isinstance(rule, str) for rule in rules)
        ):
            raise ValueError(f"Invalid per-file-ignore entry: {raw_line}")
        configured[module] = rules
    return configured


def load_per_file_ignores(config_path: Path) -> dict[str, list[str]]:
    """Load Ruff per-file ignores with a Python 3.10-compatible fallback."""

    if tomllib is None:
        return _parse_per_file_ignores(config_path.read_text(encoding="utf-8"))
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)
    return config["tool"]["ruff"]["lint"].get("per-file-ignores", {})


def main() -> int:
    configured = load_per_file_ignores(REPO_ROOT / "pyproject.toml")
    drift = find_baseline_drift(configured)
    if drift:
        print("Ruff suppression config and baseline differ:", file=sys.stderr)
        for item in drift:
            print(f"- {item}", file=sys.stderr)
        return 1
    print("Ruff suppression config matches its baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
