#!/usr/bin/env python3
"""Report staged Ruff findings grouped by module and rule without failing."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("main.py", "data_fetch.py", "utils.py")
RULES = ("B", "C4", "PIE", "RUF", "SIM", "UP", "PLC", "PLE", "PLW")
CHECK_RULES = ("E", "F", "I", *RULES)
IGNORES = ("B008", "PLW0603", "UP045")


def _group_findings(findings: list[dict[str, object]]) -> dict[str, Counter[str]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for finding in findings:
        module = Path(str(finding["filename"])).name
        grouped[module][str(finding["code"])] += 1
    return grouped


def main() -> int:
    command = (
        sys.executable,
        "-m",
        "ruff",
        "check",
        *TARGETS,
        "--isolated",
        "--target-version",
        "py311",
        "--line-length",
        "100",
        "--select",
        ",".join(CHECK_RULES),
        "--ignore",
        ",".join(IGNORES),
        "--output-format",
        "json",
        "--exit-zero",
    )
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        return completed.returncode

    findings = [
        finding
        for finding in json.loads(completed.stdout)
        if str(finding["code"]).startswith(RULES)
    ]
    grouped = _group_findings(findings)
    for module in TARGETS:
        print(f"{module}:")
        counts = grouped.get(module)
        if not counts:
            print("  clean")
            continue
        for code, count in sorted(counts.items()):
            print(f"  {code}: {count}")
    print(f"Total: {sum(sum(counts.values()) for counts in grouped.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
