#!/usr/bin/env python3
"""Report tracked image assets that exceed the repository size policy."""
from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = PROJECT_ROOT / "images"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_MAX_DIMENSION = 128
LEAGUE_LOGO_MAX_DIMENSION = 160
DEFAULT_MAX_BYTES = 75_000
LEAGUE_LOGO_MAX_BYTES = 100_000
WEATHER_ICON_MAX_DIMENSION = 128
NIXIE_DIGIT_MAX_DIMENSION = 256

LEAGUE_LOGO_NAMES = {
    "afc", "ahl", "al", "mlb", "nba", "nfc", "nfl", "nhl", "nl", "oly", "sb", "scp", "wc",
}
POLICY_DIRS = {"ahl", "nba", "nhl", "mlb", "nfl", "oly"}


@dataclass(frozen=True)
class Policy:
    asset_class: str
    max_dimension: int
    max_bytes: int


def tracked_images() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "images"],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return [PROJECT_ROOT / line for line in result.stdout.splitlines() if Path(line).suffix.lower() in SUPPORTED_EXTENSIONS]


def policy_for(path: Path) -> Policy | None:
    rel = path.relative_to(IMAGE_ROOT)
    parts = rel.parts
    if len(parts) >= 2 and parts[0] in POLICY_DIRS:
        if path.stem.lower() in LEAGUE_LOGO_NAMES:
            return Policy("league logo", LEAGUE_LOGO_MAX_DIMENSION, LEAGUE_LOGO_MAX_BYTES)
        return Policy("team logo/flag", DEFAULT_MAX_DIMENSION, DEFAULT_MAX_BYTES)
    if len(parts) >= 2 and parts[0] == "weather":
        return Policy("weather icon", WEATHER_ICON_MAX_DIMENSION, DEFAULT_MAX_BYTES)
    if len(parts) >= 2 and parts[0] == "nixie":
        return Policy("nixie digit", NIXIE_DIGIT_MAX_DIMENSION, DEFAULT_MAX_BYTES)
    return None


def violations(paths: list[Path]) -> list[str]:
    messages: list[str] = []
    for path in paths:
        policy = policy_for(path)
        if policy is None:
            continue
        with Image.open(path) as image:
            width, height = image.size
        size = path.stat().st_size
        if max(width, height) > policy.max_dimension or size > policy.max_bytes:
            rel = path.relative_to(PROJECT_ROOT)
            messages.append(
                f"{rel}: {width}x{height}, {size} bytes exceeds {policy.asset_class} "
                f"limit ({policy.max_dimension}px max edge, {policy.max_bytes} bytes)"
            )
    return messages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Exit with status 1 when images exceed policy limits; default is report-only.",
    )
    args = parser.parse_args()
    failures = violations(tracked_images())
    if failures:
        print("Image asset policy violations:")
        for failure in failures:
            print(f"- {failure}")
        return 1 if args.fail_on_violation else 0
    print("All tracked image assets satisfy the size policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
