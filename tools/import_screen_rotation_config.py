#!/usr/bin/env python3
"""Import a screen rotation JSON payload from file/stdin.

Accepted input mirrors the Screen Config UI import format:
- full payload: {"config": {...}, "style": {...}, "layouts": {...}}
- bare config object: {"screens": {...}, ...}
- entry-list config: {"screens": [{"id": ..., "frequency": ...}, ...], ...}

By default, writes to the active local config/style/layout files.
Use --dry-run to validate and preview normalized payloads without writing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config_ui import (
    _build_config,
    _build_layouts,
    _build_style_config,
    _build_screen_entries,
    _load_active_layouts_config,
    _load_active_style_config,
    _normalize_import_config_payload,
    _normalize_layouts_config,
    _normalize_legacy_scoreboard_ids,
    _save_config,
    _save_layouts_config,
    _save_style_config,
    _validate_style_payload,
)
from schedule import build_scheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        default="-",
        help="Path to JSON payload (default: stdin)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print normalized payload without persisting",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit minified JSON output for dry-run result",
    )
    return parser.parse_args()


def _read_payload(source: str) -> Dict[str, Any]:
    if source == "-":
        if sys.stdin.isatty():
            print(
                "Paste JSON payload, then press Ctrl-D (Linux/macOS) or Ctrl-Z then Enter (Windows).",
                file=sys.stderr,
            )
        raw = sys.stdin.read()
    else:
        raw = Path(source).read_text(encoding="utf-8")

    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    if not raw:
        raise ValueError("No input received. Provide JSON via file or stdin.")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Import payload must be a JSON object")
    return payload


def _resolve_import(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    config_payload = payload.get("config", payload)
    derived_style_payload: Optional[Dict[str, Any]] = None
    derived_layouts_payload: Optional[Dict[str, Any]] = None

    if isinstance(config_payload, dict) and isinstance(config_payload.get("screens"), list):
        entries = config_payload.get("screens", [])
        config = _build_config(entries)
        for key in ("playlists", "sequence"):
            value = config_payload.get(key)
            if value is not None:
                config[key] = value
        config, _ = _normalize_legacy_scoreboard_ids(config)
        derived_style_payload = _build_style_config(entries, _load_active_style_config())

        quad_pages_payload = payload.get("quad_pages")
        quad_enabled_payload = payload.get("quad_enabled", False)
        if isinstance(quad_pages_payload, list):
            derived_layouts_payload = _build_layouts({"quad_enabled": quad_enabled_payload, "quad_pages": quad_pages_payload})
    else:
        if not isinstance(config_payload, dict):
            raise ValueError("Configuration must be a JSON object")
        config = _normalize_import_config_payload(config_payload)
        config, _ = _normalize_legacy_scoreboard_ids(config)

    # Validate scheduler viability before writing.
    build_scheduler(config)

    style_payload = payload.get("style")
    if derived_style_payload is not None:
        style_config = derived_style_payload
    elif style_payload is not None:
        style_config = _validate_style_payload(style_payload)
    else:
        style_config = _load_active_style_config()

    layouts_payload = payload.get("layouts")
    if derived_layouts_payload is not None:
        layouts_config = derived_layouts_payload
    elif layouts_payload is not None:
        layouts_config = _normalize_layouts_config(layouts_payload)
    else:
        layouts_config = _load_active_layouts_config()

    return config, style_config, layouts_config


def main() -> int:
    args = parse_args()

    try:
        payload = _read_payload(args.input)
        config, style_config, layouts_config = _resolve_import(payload)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        preview = {
            "status": "ok",
            "config": config,
            "style": style_config,
            "layouts": layouts_config,
            "screens": _build_screen_entries(config, style_config),
        }
        if args.compact:
            print(json.dumps(preview, separators=(",", ":"), sort_keys=False))
        else:
            print(json.dumps(preview, indent=2, sort_keys=False))
        return 0

    _save_config(config)
    _save_style_config(style_config)
    _save_layouts_config(layouts_config)
    print("Imported screen rotation configuration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
