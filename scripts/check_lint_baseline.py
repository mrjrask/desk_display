#!/usr/bin/env python3
"""Reject new Ruff per-file suppressions while allowing baseline reductions."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

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


def find_baseline_growth(configured: dict[str, list[str]]) -> list[str]:
    """Return suppressions that are not in the approved baseline."""

    growth = []
    for module, rules in configured.items():
        approved = BASELINE.get(module, set())
        growth.extend(f"{module}: {rule}" for rule in rules if rule not in approved)
    return sorted(growth)


def main() -> int:
    with (REPO_ROOT / "pyproject.toml").open("rb") as config_file:
        config = tomllib.load(config_file)
    configured = config["tool"]["ruff"]["lint"].get("per-file-ignores", {})
    growth = find_baseline_growth(configured)
    if growth:
        print("Ruff suppression baseline grew:", file=sys.stderr)
        for item in growth:
            print(f"- {item}", file=sys.stderr)
        return 1
    print("Ruff suppression baseline did not grow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
