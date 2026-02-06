#!/usr/bin/env python3
"""Print the active screen rotation configuration JSON to stdout.

Resolution mirrors the config UI behavior:
1. Use ``SCREENS_CONFIG_LOCAL_PATH`` when it exists.
2. Otherwise fall back to ``SCREENS_CONFIG_PATH``.
3. Defaults are ``screens_config.local.json`` and ``screens_config.json`` at the
   project root.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = Path(
    os.environ.get("SCREENS_CONFIG_PATH", str(REPO_ROOT / "screens_config.json"))
).expanduser()
LOCAL_CONFIG_PATH = Path(
    os.environ.get("SCREENS_CONFIG_LOCAL_PATH", str(REPO_ROOT / "screens_config.local.json"))
).expanduser()


def _load_config(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Configuration file is not valid JSON: {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Configuration at {path} must be a JSON object")
    screens = payload.get("screens")
    if not isinstance(screens, dict):
        raise ValueError(f"Configuration at {path} must include a 'screens' object")
    return payload


def resolve_active_config_path() -> Path:
    return LOCAL_CONFIG_PATH if LOCAL_CONFIG_PATH.exists() else DEFAULT_CONFIG_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit minified JSON instead of pretty-printed JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    active_path = resolve_active_config_path()

    try:
        payload = _load_config(active_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.compact:
        print(json.dumps(payload, separators=(",", ":"), sort_keys=False))
    else:
        print(json.dumps(payload, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
