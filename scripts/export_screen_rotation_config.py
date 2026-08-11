#!/usr/bin/env python3
"""Export the same JSON payload as the Screen Config page export button.

Output shape mirrors the browser's "Export configuration" action:
{
  "config": {
    "screens": ...,
    "playlists": ...,
    "sequence": ...
  },
  "style": {
    "screens": {
      "<screen_id>": {"background": "#RRGGBB"}
    }
  }
}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config_ui import _build_screen_entries, _load_active_config, _load_active_style_config


def _parse_alt_screen(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_playlist_assignments(config: Dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, str]]:
    config_playlists = config.get("playlists")
    if not isinstance(config_playlists, dict):
        return [], {}

    sequence = config.get("sequence")
    ordered_ids: List[str] = []
    if isinstance(sequence, list):
        for item in sequence:
            if isinstance(item, dict):
                playlist_id = item.get("playlist")
                if isinstance(playlist_id, str) and playlist_id and playlist_id not in ordered_ids:
                    ordered_ids.append(playlist_id)

    for playlist_id in config_playlists.keys():
        if isinstance(playlist_id, str) and playlist_id and playlist_id not in ordered_ids:
            ordered_ids.append(playlist_id)

    playlists: list[dict[str, str]] = []
    assignments: dict[str, str] = {}
    for playlist_id in ordered_ids:
        playlist = config_playlists.get(playlist_id)
        if not isinstance(playlist, dict):
            continue
        label = playlist.get("label")
        name = label.strip() if isinstance(label, str) and label.strip() else playlist_id
        playlists.append({"id": playlist_id, "name": name})

        steps = playlist.get("steps")
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            screen_id = step.get("screen")
            if isinstance(screen_id, str) and screen_id and screen_id not in assignments:
                assignments[screen_id] = playlist_id

    return playlists, assignments


def _build_config_payload(screens: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    screens_payload: Dict[str, Any] = {}
    for screen in screens:
        screen_id = str(screen.get("id", "")).strip()
        if not screen_id:
            continue

        frequency_raw = screen.get("frequency", 0)
        try:
            frequency = int(frequency_raw)
        except (TypeError, ValueError):
            frequency = 0

        extra_seconds_raw = screen.get("extra_seconds", 0)
        try:
            extra_seconds = int(extra_seconds_raw)
        except (TypeError, ValueError):
            extra_seconds = 0
        hide_after_at = str(screen.get("hide_after_at", "")).strip()
        hide_after_enabled = bool(screen.get("hide_after_enabled", False)) and bool(hide_after_at)

        alt_screens = _parse_alt_screen(str(screen.get("alt_screen", "")).strip())
        base_spec: Dict[str, Any] = {"frequency": frequency}
        if extra_seconds > 0:
            base_spec["extra_seconds"] = extra_seconds
        if hide_after_enabled:
            base_spec["hide_after_enabled"] = True
            base_spec["hide_after_at"] = hide_after_at

        if alt_screens:
            alt_frequency_raw = screen.get("alt_frequency")
            try:
                alt_frequency = int(alt_frequency_raw)
            except (TypeError, ValueError):
                alt_frequency = 1
            base_spec["alt"] = {
                "screen": alt_screens[0] if len(alt_screens) == 1 else alt_screens,
                "frequency": alt_frequency,
            }
            screens_payload[screen_id] = base_spec
        elif len(base_spec) == 1:
            screens_payload[screen_id] = frequency
        else:
            screens_payload[screen_id] = base_spec

    playlists, assignments = _build_playlist_assignments(config)
    playlists_payload: Dict[str, Any] = {}
    for playlist in playlists:
        pid = playlist["id"]
        steps = [
            {"screen": screen["id"]}
            for screen in screens
            if str(screen.get("id", "")).strip() and assignments.get(screen["id"]) == pid
        ]
        playlists_payload[pid] = {
            "label": playlist["name"],
            "steps": steps,
        }

    return {
        "screens": screens_payload,
        "playlists": playlists_payload,
        "sequence": [{"playlist": playlist["id"]} for playlist in playlists],
    }


def _build_style_payload(screens: List[Dict[str, Any]]) -> Dict[str, Any]:
    screens_payload: Dict[str, Dict[str, str]] = {}
    for screen in screens:
        screen_id = str(screen.get("id", "")).strip()
        if not screen_id:
            continue
        background = str(screen.get("background", "")).strip()
        if not background:
            continue
        screens_payload[screen_id] = {"background": background}
    return {"screens": screens_payload}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="store_true", help="Emit minified JSON output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = _load_active_config()
        style_config = _load_active_style_config()
        screens = _build_screen_entries(config, style_config)

        payload = {
            "config": _build_config_payload(screens, config),
            "style": _build_style_payload(screens),
        }
    except Exception as exc:  # surface friendly CLI error
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.compact:
        print(json.dumps(payload, separators=(",", ":"), sort_keys=False))
    else:
        print(json.dumps(payload, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
