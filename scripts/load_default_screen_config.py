#!/usr/bin/env python3
"""Load the repo's small or large screen rotation defaults without the web UI.

This mirrors what the Screen Config page does when you pick a profile from
the "Load selected defaults" dropdown and then click Save: it takes the
config (screens/playlists/sequence) from default_screens_<profile>.json and
writes it to the active local screen rotation config. Style (background
colors) and layouts (quad tiles) are left untouched, exactly like the web UI.

Usage:
    python3 scripts/load_default_screen_config.py small
    python3 scripts/load_default_screen_config.py large
    python3 scripts/load_default_screen_config.py small --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from import_screen_rotation_config import (  # noqa: E402
    LOCAL_CONFIG_PATH,
    _normalize_legacy_scoreboard_ids,
    _save_config,
    _validate_config_payload,
)
from schedule import build_scheduler  # noqa: E402

DEFAULT_SCREEN_PROFILES = {
    "large": REPO_ROOT / "default_screens_large.json",
    "small": REPO_ROOT / "default_screens_small.json",
}


def _load_profile_config(profile: str) -> Dict[str, Any]:
    path = DEFAULT_SCREEN_PROFILES.get(profile)
    if path is None:
        raise ValueError(
            f"Unknown default profile '{profile}'. Choose one of: "
            + ", ".join(sorted(DEFAULT_SCREEN_PROFILES))
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")

    config_payload = payload.get("config") if isinstance(payload.get("config"), dict) else payload
    config = _validate_config_payload(config_payload)
    config, _ = _normalize_legacy_scoreboard_ids(config)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profile", choices=sorted(DEFAULT_SCREEN_PROFILES), help="Which default set to load")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the resulting config without writing it",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        config = _load_profile_config(args.profile)
        build_scheduler(config)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(json.dumps(config, indent=2))
        return 0

    _save_config(config)
    print(f"Loaded {args.profile} screen rotation defaults into {LOCAL_CONFIG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
