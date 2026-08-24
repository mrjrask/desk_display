#!/usr/bin/env python3
"""Report tracked image assets that exceed the repository size policy."""
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    try:
        from scripts._venv_bootstrap import reexec_with_project_venv
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _venv_bootstrap import reexec_with_project_venv
    reexec_with_project_venv()

import argparse
import json
import os
import subprocess
from dataclasses import dataclass

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
POLICY_DIRS = {"ahl", "nba", "nhl", "mlb", "nfl", "oly", "air"}


@dataclass(frozen=True)
class Policy:
    asset_class: str
    max_dimension: int
    max_bytes: int


def image_paths(lines: list[str]) -> list[Path]:
    return [
        PROJECT_ROOT / line
        for line in lines
        if line.startswith("images/")
        and Path(line).suffix.lower() in SUPPORTED_EXTENSIONS
        and (PROJECT_ROOT / line).exists()
    ]


def tracked_images() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "images"],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return image_paths(result.stdout.splitlines())


def push_before_sha() -> str | None:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        return None
    with Path(event_path).open() as event_file:
        before = json.load(event_file).get("before")
    if before and before != "0" * 40:
        return before
    return None


def changed_images() -> list[Path]:
    base_ref = os.environ.get("GITHUB_BASE_REF")
    before_sha = push_before_sha()
    if base_ref:
        diff_args = [f"origin/{base_ref}...HEAD"]
    elif before_sha:
        diff_args = [before_sha, "HEAD"]
    else:
        diff_args = ["HEAD^", "HEAD"]
    command = [
        "git",
        "diff",
        "--name-only",
        "--diff-filter=ACMR",
        *diff_args,
        "--",
        "images",
    ]
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return image_paths(result.stdout.splitlines())


def policy_for(path: Path) -> Policy | None:
    rel = path.relative_to(IMAGE_ROOT)
    parts = rel.parts
    if len(parts) >= 2 and parts[0] in POLICY_DIRS:
        if path.stem.lower() in LEAGUE_LOGO_NAMES:
            return Policy("league logo", LEAGUE_LOGO_MAX_DIMENSION, LEAGUE_LOGO_MAX_BYTES)
        return Policy("team logo/flag", DEFAULT_MAX_DIMENSION, DEFAULT_MAX_BYTES)
    if len(parts) >= 2 and parts[0] in {"weather", "WeatherKit"}:
        return Policy("weather icon", WEATHER_ICON_MAX_DIMENSION, DEFAULT_MAX_BYTES)
    if len(parts) >= 2 and parts[0] in {
        "nixie",
        "nixie-digits",
        "nixie_digits",
        "nixie_clock",
    }:
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
    parser.add_argument(
        "--changed-only",
        action="store_true",
        help="Only check image assets changed by this push or pull request.",
    )
    args = parser.parse_args()
    paths = changed_images() if args.changed_only else tracked_images()
    failures = violations(paths)
    if failures:
        print("Image asset policy violations:")
        for failure in failures:
            print(f"- {failure}")
        return 1 if args.fail_on_violation else 0
    print("All tracked image assets satisfy the size policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
