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
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paths import (
    resolve_layouts_config_path,
    resolve_screens_config_paths,
    resolve_style_config_path,
)
from schedule import build_scheduler
from screens_catalog import SCREEN_IDS, canonical_screen_id

_screens_config_paths = resolve_screens_config_paths()
LOCAL_CONFIG_PATH = str(_screens_config_paths.local_override_path)
STYLE_CONFIG_PATH = str(resolve_style_config_path())
LAYOUTS_CONFIG_PATH = str(resolve_layouts_config_path())
HIDDEN_CONFIG_SCREEN_IDS = {
    "cubs next 2",
    "sox next 2",
    "cubs last 2",
    "sox last 2",
}


def _coerce_frequency(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _merge_screen_specs(existing: Any, incoming: Any) -> Any:
    if existing is None:
        return incoming
    existing_freq = _coerce_frequency(existing.get("frequency", 0)) if isinstance(existing, dict) else _coerce_frequency(existing)
    incoming_freq = _coerce_frequency(incoming.get("frequency", 0)) if isinstance(incoming, dict) else _coerce_frequency(incoming)
    if existing_freq is None or incoming_freq is None:
        return incoming
    return incoming if incoming_freq > existing_freq else existing


def _canonicalize_screen_reference(value: Any) -> Any:
    if isinstance(value, str):
        return canonical_screen_id(value)
    if isinstance(value, list):
        canonical: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            mapped = canonical_screen_id(item)
            if mapped not in canonical:
                canonical.append(mapped)
        return canonical
    return value


def _normalize_legacy_scoreboard_ids(config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if not isinstance(config, dict):
        return config, False

    changed = False
    normalized = dict(config)
    screens = normalized.get("screens")
    if isinstance(screens, dict):
        cleaned_screens: dict[str, Any] = {}
        for raw_screen_id, raw_spec in screens.items():
            if not isinstance(raw_screen_id, str):
                continue
            screen_id = canonical_screen_id(raw_screen_id)
            if screen_id != raw_screen_id:
                changed = True

            spec = raw_spec
            if isinstance(raw_spec, dict):
                spec_copy = dict(raw_spec)
                alt = spec_copy.get("alt")
                if isinstance(alt, dict):
                    alt_copy = dict(alt)
                    canonical_alt = _canonicalize_screen_reference(alt_copy.get("screen"))
                    if canonical_alt != alt_copy.get("screen"):
                        alt_copy["screen"] = canonical_alt
                        changed = True
                    spec_copy["alt"] = alt_copy
                spec = spec_copy

            existing = cleaned_screens.get(screen_id)
            if existing is None:
                cleaned_screens[screen_id] = spec
                continue

            try:
                existing_freq = int(existing.get("frequency", 0)) if isinstance(existing, dict) else int(existing)
                new_freq = int(spec.get("frequency", 0)) if isinstance(spec, dict) else int(spec)
            except Exception:
                existing_freq = None
                new_freq = None
            if existing_freq is not None and new_freq is not None and new_freq > existing_freq:
                changed = True
                cleaned_screens[screen_id] = spec
        normalized["screens"] = cleaned_screens

    playlists = normalized.get("playlists")
    if isinstance(playlists, dict):
        cleaned_playlists: dict[str, Any] = {}
        for playlist_id, playlist in playlists.items():
            if not isinstance(playlist, dict):
                cleaned_playlists[playlist_id] = playlist
                continue
            playlist_copy = dict(playlist)
            steps = playlist_copy.get("steps")
            if isinstance(steps, list):
                cleaned_steps: list[Any] = []
                for step in steps:
                    if not isinstance(step, dict):
                        cleaned_steps.append(step)
                        continue
                    step_copy = dict(step)
                    step_screen = step_copy.get("screen")
                    if isinstance(step_screen, str):
                        mapped = canonical_screen_id(step_screen)
                        if mapped != step_screen:
                            changed = True
                            step_copy["screen"] = mapped
                    cleaned_steps.append(step_copy)
                playlist_copy["steps"] = cleaned_steps
            cleaned_playlists[playlist_id] = playlist_copy
        normalized["playlists"] = cleaned_playlists

    return normalized, changed


def _validate_config_payload(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object")
    screens = data.get("screens")
    if not isinstance(screens, dict):
        raise ValueError("Configuration must include a 'screens' mapping")
    return data


def _normalize_import_config_payload(data: dict[str, Any]) -> dict[str, Any]:
    normalized = _validate_config_payload(data)
    screens = normalized.get("screens", {})
    normalized_screens: dict[str, Any] = {}
    for screen_id, raw in screens.items():
        canonical_id = canonical_screen_id(screen_id) if isinstance(screen_id, str) else screen_id
        if not isinstance(canonical_id, str):
            continue
        if isinstance(raw, dict):
            frequency = raw.get("frequency", 0)
            try:
                frequency_int = int(frequency)
            except (TypeError, ValueError):
                frequency_int = frequency
            extra_seconds_raw = raw.get("extra_seconds", 0)
            try:
                extra_seconds = int(extra_seconds_raw)
            except (TypeError, ValueError):
                extra_seconds = extra_seconds_raw
            hide_after_enabled = bool(raw.get("hide_after_enabled", False))
            hide_after_at_raw = raw.get("hide_after_at")
            hide_after_at = str(hide_after_at_raw).strip() if hide_after_at_raw is not None else ""

            alt_payload: Optional[dict[str, Any]] = None
            alt = raw.get("alt")
            if isinstance(alt, dict):
                alt_payload = dict(alt)
                alt_payload["screen"] = _canonicalize_screen_reference(alt_payload.get("screen"))
                alt_frequency = alt_payload.get("frequency")
                if alt_frequency is not None:
                    try:
                        alt_payload["frequency"] = int(alt_frequency)
                    except (TypeError, ValueError):
                        pass

            normalized_spec: dict[str, Any] = {"frequency": frequency_int}
            if isinstance(extra_seconds, int) and extra_seconds > 0:
                normalized_spec["extra_seconds"] = extra_seconds
            if alt_payload is not None:
                normalized_spec["alt"] = alt_payload
            if hide_after_enabled and hide_after_at:
                normalized_spec["hide_after_enabled"] = True
                normalized_spec["hide_after_at"] = hide_after_at
            normalized_screens[canonical_id] = _merge_screen_specs(normalized_screens.get(canonical_id), normalized_spec)
            continue
        try:
            normalized_raw: Any = int(raw)
        except (TypeError, ValueError):
            normalized_raw = raw
        normalized_screens[canonical_id] = _merge_screen_specs(normalized_screens.get(canonical_id), normalized_raw)

    result = dict(normalized)
    result["screens"] = normalized_screens
    cleaned, _ = _normalize_legacy_scoreboard_ids(result)
    return cleaned


def _default_layouts_config() -> dict[str, Any]:
    return {"screens": {"quad": {"enabled": False, "scroll_speed": 1.0, "pages": [{"tiles": ["date", "weather1", "weather hourly", "inside"]}]}}}


def _normalize_quad_scroll_speed(value: Any) -> float:
    try:
        speed = float(value)
    except (TypeError, ValueError):
        speed = 1.0
    return min(3.0, max(0.25, speed))


def _normalize_quad_page(raw_page: Any, defaults: list[str]) -> dict[str, Any]:
    tiles_source = raw_page.get("tiles") if isinstance(raw_page, dict) else raw_page
    if not isinstance(tiles_source, list):
        tiles_source = []
    tiles: list[str] = []
    for item in tiles_source:
        if not isinstance(item, str):
            continue
        tile = item.strip()
        if not tile or tile == "quad":
            continue
        tiles.append(tile)
        if len(tiles) >= 4:
            break
    while len(tiles) < 4:
        tiles.append(defaults[len(tiles)])
    return {"tiles": tiles}


def _normalize_layouts_config(data: Any) -> dict[str, Any]:
    result = _default_layouts_config()
    defaults = result["screens"]["quad"]["pages"][0]["tiles"]
    if not isinstance(data, dict):
        return result
    screens = data.get("screens")
    if not isinstance(screens, dict):
        return result
    quad = screens.get("quad")
    if not isinstance(quad, dict):
        return result
    result["screens"]["quad"]["enabled"] = bool(quad.get("enabled", False))
    result["screens"]["quad"]["scroll_speed"] = _normalize_quad_scroll_speed(quad.get("scroll_speed", 1.0))
    pages: list[dict[str, Any]] = []
    raw_pages = quad.get("pages")
    if isinstance(raw_pages, list):
        pages = [_normalize_quad_page(raw_page, defaults) for raw_page in raw_pages]
    if not pages and isinstance(quad.get("tiles"), list):
        pages = [_normalize_quad_page({"tiles": quad.get("tiles")}, defaults)]
    if not pages:
        pages = [result["screens"]["quad"]["pages"][0]]
    result["screens"]["quad"]["pages"] = pages
    return result


def _load_active_layouts_config() -> dict[str, Any]:
    try:
        with open(LAYOUTS_CONFIG_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return _default_layouts_config()
    return _normalize_layouts_config(data)


def _save_layouts_config(config: dict[str, Any]) -> None:
    tmp_path = f"{LAYOUTS_CONFIG_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")
    os.replace(tmp_path, LAYOUTS_CONFIG_PATH)


def _build_layouts(entries: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(entries, dict):
        raise ValueError("Invalid layouts payload")
    quad_enabled = bool(entries.get("quad_enabled", False))
    quad_scroll_speed = _normalize_quad_scroll_speed(entries.get("quad_scroll_speed", 1.0))
    raw_pages = entries.get("quad_pages")
    if not isinstance(raw_pages, list):
        legacy_tiles = entries.get("quad_tiles")
        if isinstance(legacy_tiles, list):
            raw_pages = [{"tiles": legacy_tiles}]
        else:
            raise ValueError("quad_pages must be a list")
    defaults = _default_layouts_config()["screens"]["quad"]["pages"][0]["tiles"]
    pages: list[dict[str, Any]] = []
    for raw_page in raw_pages:
        if isinstance(raw_page, dict):
            pages.append(_normalize_quad_page(raw_page, defaults))
    if not pages:
        pages = [{"tiles": defaults.copy()}]
    return {"screens": {"quad": {"enabled": quad_enabled, "scroll_speed": quad_scroll_speed, "pages": pages}}}


def _load_active_style_config() -> dict[str, Any]:
    try:
        with open(STYLE_CONFIG_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {"screens": {}}
    if not isinstance(data, dict) or not isinstance(data.get("screens"), dict):
        raise ValueError("Style configuration must include a 'screens' mapping")
    return data


def _normalise_hex_color(value: str) -> Optional[str]:
    cleaned = value.strip()
    if not cleaned:
        return None
    if not all(char in "0123456789abcdefABCDEF" for char in cleaned.lstrip("#")):
        return None
    if len(cleaned.lstrip("#")) != 6:
        return None
    return cleaned.upper() if cleaned.startswith("#") else f"#{cleaned.upper()}"


def _rgb_to_hex(rgb: Iterable[int]) -> str:
    channels = list(rgb)
    if len(channels) != 3:
        raise ValueError("RGB color must have exactly 3 channels")
    return "#{0:02X}{1:02X}{2:02X}".format(*channels)


def _default_background_for_screen(screen_id: str) -> tuple[int, int, int]:
    lowered = screen_id.lower()
    if any(token in lowered for token in ("scoreboard", "standings", "overview", "stand1", "stand2", "stand3")):
        return (125, 125, 125)
    return (0, 0, 0)


def _validate_style_payload(data: Any) -> dict[str, Any]:
    if data is None:
        return {"screens": {}}
    if not isinstance(data, dict):
        raise ValueError("Style configuration must be a JSON object")
    screens = data.get("screens")
    if screens is None:
        return {"screens": {}}
    if not isinstance(screens, dict):
        raise ValueError("Style configuration must include a 'screens' mapping")
    invalid_screens: list[str] = []
    normalised_screens: dict[str, dict[str, str]] = {}
    for screen_id, spec in screens.items():
        if not isinstance(screen_id, str):
            continue
        background: Optional[str] = None
        if isinstance(spec, str):
            background = spec
        elif isinstance(spec, dict):
            raw_background = spec.get("background")
            if isinstance(raw_background, str):
                background = raw_background
        if background is None:
            continue
        normalised = _normalise_hex_color(background)
        if not normalised:
            invalid_screens.append(screen_id)
            continue
        normalised_screens[screen_id] = {"background": normalised}
    if invalid_screens:
        raise ValueError("Invalid background color for: " + ", ".join(sorted(set(invalid_screens))))
    return {"screens": normalised_screens}


def _build_screen_entries(config: dict[str, Any], style_config: dict[str, Any]) -> list[dict[str, Any]]:
    screens = config.get("screens", {})
    if not isinstance(screens, dict):
        return []
    style_screens = style_config.get("screens", {})
    if not isinstance(style_screens, dict):
        style_screens = {}
    ordered_screen_ids: list[str] = [screen_id for screen_id in screens.keys() if screen_id not in HIDDEN_CONFIG_SCREEN_IDS]
    for screen_id in SCREEN_IDS:
        if screen_id not in HIDDEN_CONFIG_SCREEN_IDS and screen_id not in ordered_screen_ids:
            ordered_screen_ids.append(screen_id)
    entries: list[dict[str, Any]] = []
    for screen_id in ordered_screen_ids:
        raw = screens.get(screen_id, 0)
        entry: dict[str, Any] = {
            "id": screen_id,
            "frequency": 0,
            "extra_seconds": 0,
            "alt_screen": "",
            "alt_frequency": "",
            "hide_after_at": "",
            "hide_after_enabled": False,
            "background": _rgb_to_hex(_default_background_for_screen(screen_id)),
        }
        if isinstance(raw, dict):
            entry["frequency"] = raw.get("frequency", 0)
            entry["extra_seconds"] = raw.get("extra_seconds", 0)
            entry["hide_after_at"] = raw.get("hide_after_at", "") if isinstance(raw.get("hide_after_at"), str) else ""
            entry["hide_after_enabled"] = bool(raw.get("hide_after_enabled", False))
            alt = raw.get("alt") if isinstance(raw.get("alt"), dict) else None
            if alt:
                alt_screen = alt.get("screen")
                entry["alt_screen"] = ", ".join(alt_screen) if isinstance(alt_screen, list) else str(alt_screen or "")
                entry["alt_frequency"] = alt.get("frequency", "")
        else:
            entry["frequency"] = raw
        style_entry = style_screens.get(screen_id)
        if isinstance(style_entry, dict):
            background = style_entry.get("background")
            if isinstance(background, str):
                normalised = _normalise_hex_color(background)
                if normalised:
                    entry["background"] = normalised
        entries.append(entry)
    return entries


def _build_config(entries: list[dict[str, Any]]) -> dict[str, Any]:
    screens: dict[str, Any] = {}
    for entry in entries:
        screen_id = canonical_screen_id(str(entry.get("id", "")).strip())
        if not screen_id or screen_id in HIDDEN_CONFIG_SCREEN_IDS:
            continue
        frequency = int(entry.get("frequency", 0))
        extra_seconds = int(entry.get("extra_seconds", 0))
        if extra_seconds < 0:
            raise ValueError(f"Additional seconds for '{screen_id}' cannot be negative")
        spec: dict[str, Any] = {"frequency": frequency}
        if extra_seconds > 0:
            spec["extra_seconds"] = extra_seconds
        hide_after_at = str(entry.get("hide_after_at", "")).strip()
        hide_after_enabled = bool(entry.get("hide_after_enabled", False))
        if hide_after_enabled and hide_after_at:
            spec["hide_after_enabled"] = True
            spec["hide_after_at"] = hide_after_at
        alt_screen_raw = str(entry.get("alt_screen", "")).strip()
        if alt_screen_raw:
            alt_screens = [canonical_screen_id(item.strip()) for item in alt_screen_raw.split(",") if item.strip()]
            alt_frequency = int(entry.get("alt_frequency") or 1)
            spec["alt"] = {"screen": alt_screens[0] if len(alt_screens) == 1 else alt_screens, "frequency": alt_frequency}
        screens[screen_id] = spec if any(k != "frequency" for k in spec) else frequency
    cleaned, _ = _normalize_legacy_scoreboard_ids({"screens": screens})
    return cleaned


def _build_style_config(entries: list[dict[str, Any]], style_config: dict[str, Any]) -> dict[str, Any]:
    screens: dict[str, Any] = {}
    existing = style_config.get("screens")
    if isinstance(existing, dict):
        for screen_id, spec in existing.items():
            if isinstance(screen_id, str) and isinstance(spec, dict):
                screens[screen_id] = dict(spec)
    invalid_screens: list[str] = []
    for entry in entries:
        screen_id = str(entry.get("id", "")).strip()
        if not screen_id:
            continue
        background_value = str(entry.get("background", "")).strip()
        if not background_value:
            continue
        normalised = _normalise_hex_color(background_value)
        if not normalised:
            invalid_screens.append(screen_id)
            continue
        default_hex = _rgb_to_hex(_default_background_for_screen(screen_id))
        if normalised != default_hex:
            screens.setdefault(screen_id, {})["background"] = normalised
    if invalid_screens:
        raise ValueError("Invalid background color for: " + ", ".join(sorted(set(invalid_screens))))
    return {"screens": screens}


def _save_config(config: dict[str, Any]) -> None:
    tmp_path = f"{LOCAL_CONFIG_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")
    os.replace(tmp_path, LOCAL_CONFIG_PATH)


def _save_style_config(config: dict[str, Any]) -> None:
    tmp_path = f"{STYLE_CONFIG_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")
    os.replace(tmp_path, STYLE_CONFIG_PATH)


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


def _read_payload(source: str) -> dict[str, Any]:
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


def _resolve_import(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_payload = payload.get("config", payload)
    derived_style_payload: Optional[dict[str, Any]] = None
    derived_layouts_payload: Optional[dict[str, Any]] = None

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
