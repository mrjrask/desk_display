#!/usr/bin/env python3
"""Interactively update the screen rotation config to a default profile.

Prompts for which default screen set to load (small or large), or lets you
cancel without making any changes. This is the interactive counterpart to
load_default_screen_config.py, which takes the profile as a CLI argument
instead of prompting for it.

Usage:
    python3 scripts/update_screen_config.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from load_default_screen_config import (  # noqa: E402
    DEFAULT_SCREEN_PROFILES,
    _load_profile_config,
)
from import_screen_rotation_config import LOCAL_CONFIG_PATH, _save_config  # noqa: E402
from schedule import build_scheduler  # noqa: E402

PROMPT = (
    "\nWhich default screen configuration should be loaded?\n"
    "  1) small\n"
    "  2) large\n"
    "  c) cancel (no changes)\n"
    "> "
)

CHOICES = {
    "1": "small",
    "2": "large",
    "small": "small",
    "large": "large",
}

CANCEL_CHOICES = {"c", "cancel", "q", "quit", ""}


def prompt_profile() -> str | None:
    while True:
        try:
            answer = input(PROMPT).strip().lower()
        except EOFError:
            return None

        if answer in CANCEL_CHOICES:
            return None
        if answer in CHOICES:
            return CHOICES[answer]

        print(
            f"Unrecognized choice {answer!r}. "
            f"Enter one of: {', '.join(sorted(DEFAULT_SCREEN_PROFILES))}, or 'c' to cancel."
        )


def main() -> int:
    profile = prompt_profile()
    if profile is None:
        print("Cancelled. No changes made.")
        return 0

    try:
        config = _load_profile_config(profile)
        build_scheduler(config)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _save_config(config)
    print(f"Loaded {profile} screen rotation defaults into {LOCAL_CONFIG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
