#!/usr/bin/env python3
"""Web UI for editing the screen rotation configuration."""
from __future__ import annotations

import json
import logging
import os
import socket
import time
from urllib.parse import urlsplit, urlunsplit
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
    g,
)

from paths import resolve_storage_paths
from schedule import build_scheduler
from screens_catalog import SCREEN_IDS, canonical_screen_id

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.environ.get(
    "SCREENS_CONFIG_PATH", os.path.join(SCRIPT_DIR, "screens_config.json")
)
LOCAL_CONFIG_PATH = os.environ.get(
    "SCREENS_CONFIG_LOCAL_PATH", os.path.join(SCRIPT_DIR, "screens_config.local.json")
)
STYLE_CONFIG_PATH = os.environ.get(
    "SCREENS_STYLE_PATH", os.path.join(SCRIPT_DIR, "screens_style.json")
)
LAYOUTS_CONFIG_PATH = os.environ.get(
    "SCREENS_LAYOUTS_PATH", os.path.join(SCRIPT_DIR, "screens_layouts.json")
)

SCREEN_CONFIG_HOST = os.environ.get("SCREEN_CONFIG_HOST", "0.0.0.0")
SCREEN_CONFIG_PORT = int(os.environ.get("SCREEN_CONFIG_PORT", "5002"))
SCREEN_UI_USERNAME = os.environ.get("SCREEN_UI_USERNAME", "")
SCREEN_UI_PASSWORD = os.environ.get("SCREEN_UI_PASSWORD", "")
SCREEN_AUTH_ENABLED = os.environ.get("SCREEN_AUTH_ENABLED", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ALLOWED_SCREEN_EXTS = (".png", ".jpg", ".jpeg")
app = Flask(__name__)
app.secret_key = SCREEN_UI_PASSWORD or "desk-display-config-ui"
WEB_LOGGER = logging.getLogger("desk_display.web")
HIDDEN_CONFIG_SCREEN_IDS = {
    "cubs next 2",
    "sox next 2",
    "cubs last 2",
    "sox last 2",
}


def _canonicalize_screen_reference(value: Any) -> Any:
    if isinstance(value, str):
        return canonical_screen_id(value)
    if isinstance(value, list):
        canonical: List[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            mapped = canonical_screen_id(item)
            if mapped not in canonical:
                canonical.append(mapped)
        return canonical
    return value


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


def _normalize_legacy_scoreboard_ids(config: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
    if not isinstance(config, dict):
        return config, False

    changed = False
    normalized = dict(config)
    screens = normalized.get("screens")
    if isinstance(screens, dict):
        cleaned_screens: Dict[str, Any] = {}
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
        cleaned_playlists: Dict[str, Any] = {}
        for playlist_id, playlist in playlists.items():
            if not isinstance(playlist, dict):
                cleaned_playlists[playlist_id] = playlist
                continue
            playlist_copy = dict(playlist)
            steps = playlist_copy.get("steps")
            if isinstance(steps, list):
                cleaned_steps: List[Any] = []
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


@app.before_request
def _capture_request_start_time() -> None:
    g.request_started_at = time.perf_counter()


@app.after_request
def _log_web_request(response: Any) -> Any:
    started = getattr(g, "request_started_at", None)
    elapsed_ms: Optional[int] = None
    if isinstance(started, float):
        elapsed_ms = int((time.perf_counter() - started) * 1000)

    duration = f" {elapsed_ms}ms" if elapsed_ms is not None else ""
    WEB_LOGGER.info(
        "🌐 %s %s -> %s%s ip=%s",
        request.method,
        request.full_path if request.query_string else request.path,
        response.status_code,
        duration,
        request.remote_addr or "unknown",
    )
    return response


def _is_auth_enabled() -> bool:
    return bool(SCREEN_UI_PASSWORD) or SCREEN_AUTH_ENABLED


def _is_authenticated() -> bool:
    return bool(session.get("screen_ui_authenticated"))


@app.before_request
def _require_authentication() -> Optional[Any]:
    if not _is_auth_enabled():
        return None

    if request.endpoint in {"login", "logout", "static"}:
        return None

    if _is_authenticated():
        return None

    if request.path.startswith("/api/"):
        return jsonify({"error": "Authentication required"}), 401

    return redirect(url_for("login", next=request.full_path if request.query_string else request.path))


def _load_config(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {"screens": {}}
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object")
    screens = data.get("screens")
    if not isinstance(screens, dict):
        raise ValueError("Configuration must include a 'screens' mapping")
    normalized, _ = _normalize_legacy_scoreboard_ids(data)
    return normalized


def _validate_config_payload(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object")
    screens = data.get("screens")
    if not isinstance(screens, dict):
        raise ValueError("Configuration must include a 'screens' mapping")
    return data


def _normalize_import_config_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize imported config values so scheduler validation is predictable."""

    normalized = _validate_config_payload(data)
    screens = normalized.get("screens", {})
    normalized_screens: Dict[str, Any] = {}

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

            alt_payload: Optional[Dict[str, Any]] = None
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

            normalized_spec: Dict[str, Any] = {"frequency": frequency_int}
            if alt_payload is not None:
                normalized_spec["alt"] = alt_payload
            normalized_screens[canonical_id] = _merge_screen_specs(
                normalized_screens.get(canonical_id),
                normalized_spec,
            )
            continue

        try:
            normalized_raw: Any = int(raw)
        except (TypeError, ValueError):
            normalized_raw = raw
        normalized_screens[canonical_id] = _merge_screen_specs(
            normalized_screens.get(canonical_id),
            normalized_raw,
        )

    result = dict(normalized)
    result["screens"] = normalized_screens
    cleaned, _ = _normalize_legacy_scoreboard_ids(result)
    return cleaned




def _default_layouts_config() -> Dict[str, Any]:
    return {
        "screens": {
            "quad": {
                "enabled": False,
                "scroll_speed": 1.0,
                "pages": [
                    {"tiles": ["date", "weather1", "weather hourly", "inside"]},
                ],
            }
        }
    }


def _normalize_quad_scroll_speed(value: Any) -> float:
    try:
        speed = float(value)
    except (TypeError, ValueError):
        speed = 1.0
    return min(3.0, max(0.25, speed))


def _normalize_quad_page(raw_page: Any, defaults: List[str]) -> Dict[str, Any]:
    tiles_source = raw_page.get("tiles") if isinstance(raw_page, dict) else raw_page
    if not isinstance(tiles_source, list):
        tiles_source = []

    tiles: List[str] = []
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


def _normalize_layouts_config(data: Any) -> Dict[str, Any]:
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

    pages: List[Dict[str, Any]] = []
    raw_pages = quad.get("pages")
    if isinstance(raw_pages, list):
        pages = [_normalize_quad_page(raw_page, defaults) for raw_page in raw_pages]

    if not pages and isinstance(quad.get("tiles"), list):
        pages = [_normalize_quad_page({"tiles": quad.get("tiles")}, defaults)]

    if not pages:
        pages = [result["screens"]["quad"]["pages"][0]]

    result["screens"]["quad"]["pages"] = pages
    return result


def _load_layouts_config(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return _default_layouts_config()
    return _normalize_layouts_config(data)


def _load_active_layouts_config() -> Dict[str, Any]:
    return _load_layouts_config(LAYOUTS_CONFIG_PATH)


def _save_layouts_config(config: Dict[str, Any]) -> None:
    tmp_path = f"{LAYOUTS_CONFIG_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")
    os.replace(tmp_path, LAYOUTS_CONFIG_PATH)


def _build_layouts(entries: Dict[str, Any]) -> Dict[str, Any]:
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
    pages: List[Dict[str, Any]] = []

    for raw_page in raw_pages:
        if not isinstance(raw_page, dict):
            continue
        page = _normalize_quad_page(raw_page, defaults)
        pages.append(page)

    if not pages:
        pages = [{"tiles": defaults.copy()}]

    return {"screens": {"quad": {"enabled": quad_enabled, "scroll_speed": quad_scroll_speed, "pages": pages}}}


def _load_style_config(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {"screens": {}}
    if not isinstance(data, dict):
        raise ValueError("Style configuration must be a JSON object")
    screens = data.get("screens")
    if not isinstance(screens, dict):
        raise ValueError("Style configuration must include a 'screens' mapping")
    return data


def _load_active_config() -> Dict[str, Any]:
    if os.path.exists(LOCAL_CONFIG_PATH):
        config = _load_config(LOCAL_CONFIG_PATH)
        cleaned, changed = _normalize_legacy_scoreboard_ids(config)
        if changed:
            _save_config(cleaned)
        return cleaned
    config = _load_config(DEFAULT_CONFIG_PATH)
    cleaned, changed = _normalize_legacy_scoreboard_ids(config)
    if changed:
        _save_config(cleaned)
    return cleaned


def _load_active_style_config() -> Dict[str, Any]:
    return _load_style_config(STYLE_CONFIG_PATH)


def _validate_style_payload(data: Any) -> Dict[str, Any]:
    if data is None:
        return {"screens": {}}
    if not isinstance(data, dict):
        raise ValueError("Style configuration must be a JSON object")
    screens = data.get("screens")
    if screens is None:
        return {"screens": {}}
    if not isinstance(screens, dict):
        raise ValueError("Style configuration must include a 'screens' mapping")

    invalid_screens: List[str] = []
    normalised_screens: Dict[str, Dict[str, str]] = {}
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
        raise ValueError(
            "Invalid background color for: " + ", ".join(sorted(set(invalid_screens)))
        )

    return {"screens": normalised_screens}


def _parse_alt_screen(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    parts = [item.strip() for item in value.split(",")]
    parts = [item for item in parts if item]
    return parts or None


def _serialize_alt_screen(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value if item)
    if isinstance(value, str):
        return value
    return ""


def _build_playlist_assignments(config: Dict[str, Any]) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    config_playlists = config.get("playlists")
    if not isinstance(config_playlists, dict):
        return [], {}

    sequence = config.get("sequence")
    ordered_ids: List[str] = []
    if isinstance(sequence, list):
        for item in sequence:
            if not isinstance(item, dict):
                continue
            playlist_id = item.get("playlist")
            if isinstance(playlist_id, str) and playlist_id and playlist_id not in ordered_ids:
                ordered_ids.append(playlist_id)

    for playlist_id in config_playlists.keys():
        if isinstance(playlist_id, str) and playlist_id and playlist_id not in ordered_ids:
            ordered_ids.append(playlist_id)

    playlists: List[Dict[str, str]] = []
    assignments: Dict[str, str] = {}
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


def _sanitize_filename_prefix(name: str) -> str:
    """Return a filesystem-friendly filename prefix."""

    safe = name.strip().replace("/", "-").replace("\\", "-")
    safe = safe.replace(" ", "_")
    safe = "".join(ch for ch in safe if ch.isalnum() or ch in ("_", "-"))
    return safe or "screen"


def _sanitize_directory_name(name: str) -> str:
    """Return a filesystem-friendly directory name while keeping spaces."""

    safe = name.strip().replace("/", "-").replace("\\", "-")
    safe = "".join(ch for ch in safe if ch.isalnum() or ch in (" ", "-", "_"))
    return safe or "Screens"


def _current_screenshot_dir() -> Path:
    storage_paths = resolve_storage_paths()
    return storage_paths.current_screenshot_dir


def _format_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def _format_elapsed_since(timestamp: float) -> str:
    total_seconds = max(0, int(datetime.now().timestamp() - timestamp))
    days, remainder = divmod(total_seconds, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    return f"{days}d {hours}h {minutes}m {seconds}s ago"


def _is_screenshot_stale(timestamp: float, *, max_age_seconds: int = 2 * 60 * 60) -> bool:
    return datetime.now().timestamp() - timestamp > max_age_seconds


def _build_screenshot_entries() -> List[Dict[str, Any]]:
    storage_paths = resolve_storage_paths()
    screenshot_dir = storage_paths.screenshot_dir
    current_dir = storage_paths.current_screenshot_dir
    config = _load_active_config()
    screens_config = config.get("screens", {})
    ordered_screen_ids: List[str] = []
    if isinstance(screens_config, dict):
        ordered_screen_ids.extend(list(screens_config.keys()))
    for screen_id in SCREEN_IDS:
        if screen_id not in ordered_screen_ids:
            ordered_screen_ids.append(screen_id)
    entries: List[Dict[str, Any]] = []
    for screen_id in ordered_screen_ids:
        prefix = _sanitize_filename_prefix(screen_id)
        entry: Dict[str, Any] = {
            "id": screen_id,
            "path": None,
            "timestamp": None,
            "elapsed": None,
            "version": None,
            "is_stale": False,
        }
        candidates: List[Path] = [
            current_dir / f"{prefix}{ext}" for ext in ALLOWED_SCREEN_EXTS
        ]
        screen_dir = screenshot_dir / _sanitize_directory_name(screen_id)
        if screen_dir.is_dir():
            candidates.extend(
                path
                for path in screen_dir.glob(f"{prefix}_*")
                if path.is_file() and path.suffix.lower() in ALLOWED_SCREEN_EXTS
            )

        path: Optional[Path] = None
        existing_candidates = [candidate for candidate in candidates if candidate.exists()]
        if existing_candidates:
            path = max(existing_candidates, key=lambda item: item.stat().st_mtime)

        if path and path.exists():
            entry["path"] = path.relative_to(screenshot_dir).as_posix()
            try:
                modified_time = path.stat().st_mtime
                entry["timestamp"] = _format_timestamp(modified_time)
                entry["elapsed"] = _format_elapsed_since(modified_time)
                entry["version"] = int(modified_time)
                entry["is_stale"] = _is_screenshot_stale(modified_time)
            except OSError:
                entry["timestamp"] = None
                entry["elapsed"] = None
                entry["version"] = None
                entry["is_stale"] = False
        entries.append(entry)
    return entries




def _load_display_status() -> Dict[str, Any]:
    storage_paths = resolve_storage_paths()
    status_path = storage_paths.current_screenshot_dir / "display_status.json"

    status: Dict[str, Any] = {
        "screen_id": None,
        "rendered_at": None,
        "elapsed": None,
        "loop_iteration": None,
        "frame_id": None,
        "is_stale": True,
    }

    if not status_path.exists():
        return status

    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return status

    if not isinstance(payload, dict):
        return status

    screen_id = payload.get("screen_id")
    if isinstance(screen_id, str) and screen_id.strip():
        status["screen_id"] = screen_id

    rendered_at_raw = payload.get("rendered_at")
    if isinstance(rendered_at_raw, str) and rendered_at_raw.strip():
        try:
            rendered_dt = datetime.fromisoformat(rendered_at_raw)
        except ValueError:
            rendered_dt = None

        if rendered_dt is not None:
            rendered_ts = rendered_dt.timestamp()
            status["rendered_at"] = _format_timestamp(rendered_ts)
            status["elapsed"] = _format_elapsed_since(rendered_ts)
            status["is_stale"] = _is_screenshot_stale(rendered_ts, max_age_seconds=10 * 60)

    loop_iteration = payload.get("loop_iteration")
    if isinstance(loop_iteration, int):
        status["loop_iteration"] = loop_iteration

    frame_id = payload.get("frame_id")
    if isinstance(frame_id, int):
        status["frame_id"] = frame_id

    return status

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


def _coerce_env_color_component(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return max(0, min(255, value))


def _get_scoreboard_background_color() -> Tuple[int, int, int]:
    return (
        _coerce_env_color_component("SCOREBOARD_BACKGROUND_R", 125),
        _coerce_env_color_component("SCOREBOARD_BACKGROUND_G", 125),
        _coerce_env_color_component("SCOREBOARD_BACKGROUND_B", 125),
    )


def _default_background_for_screen(screen_id: str) -> Tuple[int, int, int]:
    lowered = screen_id.lower()
    if any(token in lowered for token in ("scoreboard", "standings", "overview", "stand1", "stand2", "stand3")):
        return _get_scoreboard_background_color()
    return (0, 0, 0)


def _build_screen_entries(
    config: Dict[str, Any],
    style_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    screens = config.get("screens", {})
    if not isinstance(screens, dict):
        return []
    style_screens = style_config.get("screens", {})
    if not isinstance(style_screens, dict):
        style_screens = {}

    ordered_screen_ids: List[str] = []
    ordered_screen_ids.extend(
        [screen_id for screen_id in screens.keys() if screen_id not in HIDDEN_CONFIG_SCREEN_IDS]
    )
    for screen_id in SCREEN_IDS:
        if screen_id in HIDDEN_CONFIG_SCREEN_IDS:
            continue
        if screen_id not in ordered_screen_ids:
            ordered_screen_ids.append(screen_id)

    entries: List[Dict[str, Any]] = []
    for screen_id in ordered_screen_ids:
        raw = screens.get(screen_id, 0)
        entry: Dict[str, Any] = {
            "id": screen_id,
            "frequency": 0,
            "alt_screen": "",
            "alt_frequency": "",
            "background": _rgb_to_hex(_default_background_for_screen(screen_id)),
        }
        if isinstance(raw, dict):
            entry["frequency"] = raw.get("frequency", 0)
            alt = raw.get("alt") if isinstance(raw.get("alt"), dict) else None
            if alt:
                entry["alt_screen"] = _serialize_alt_screen(alt.get("screen"))
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


def _build_config(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    screens: Dict[str, Any] = {}
    for entry in entries:
        screen_id = canonical_screen_id(str(entry.get("id", "")).strip())
        if not screen_id:
            continue
        if screen_id in HIDDEN_CONFIG_SCREEN_IDS:
            continue
        frequency = int(entry.get("frequency", 0))
        alt_screen_raw = entry.get("alt_screen")
        alt_frequency = entry.get("alt_frequency")
        alt_screen = _parse_alt_screen(str(alt_screen_raw).strip()) if alt_screen_raw is not None else None
        if alt_screen:
            alt_screen = [canonical_screen_id(item) for item in alt_screen]
            alt_frequency_int = int(alt_frequency) if alt_frequency not in ("", None) else 1
            alt_payload: Dict[str, Any] = {"screen": alt_screen[0] if len(alt_screen) == 1 else alt_screen}
            alt_payload["frequency"] = alt_frequency_int
            screens[screen_id] = {"frequency": frequency, "alt": alt_payload}
        else:
            screens[screen_id] = frequency
    cleaned, _ = _normalize_legacy_scoreboard_ids({"screens": screens})
    return cleaned


def _build_style_config(
    entries: List[Dict[str, Any]],
    style_config: Dict[str, Any],
) -> Dict[str, Any]:
    screens: Dict[str, Any] = {}
    existing = style_config.get("screens")
    if isinstance(existing, dict):
        for screen_id, spec in existing.items():
            if isinstance(screen_id, str) and isinstance(spec, dict):
                screens[screen_id] = dict(spec)

    invalid_screens: List[str] = []
    for entry in entries:
        screen_id = str(entry.get("id", "")).strip()
        if not screen_id:
            continue
        background_raw = entry.get("background")
        background_value = str(background_raw).strip() if background_raw is not None else ""
        if not background_value:
            if screen_id in screens:
                screens[screen_id].pop("background", None)
                if not screens[screen_id]:
                    screens.pop(screen_id, None)
            continue
        normalised = _normalise_hex_color(background_value)
        if not normalised:
            invalid_screens.append(screen_id)
            continue
        default_hex = _rgb_to_hex(_default_background_for_screen(screen_id))
        if normalised == default_hex:
            if screen_id in screens:
                screens[screen_id].pop("background", None)
                if not screens[screen_id]:
                    screens.pop(screen_id, None)
            continue
        screens.setdefault(screen_id, {})["background"] = normalised

    if invalid_screens:
        raise ValueError(
            "Invalid background color for: " + ", ".join(sorted(set(invalid_screens)))
        )
    return {"screens": screens}


def _save_config(config: Dict[str, Any]) -> None:
    tmp_path = f"{LOCAL_CONFIG_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")
    os.replace(tmp_path, LOCAL_CONFIG_PATH)


def _save_style_config(config: Dict[str, Any]) -> None:
    tmp_path = f"{STYLE_CONFIG_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")
    os.replace(tmp_path, STYLE_CONFIG_PATH)


def run_config_ui(host: str = SCREEN_CONFIG_HOST, port: int = SCREEN_CONFIG_PORT) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S",
        )
    from waitress import serve

    serve(app, host=host, port=port, threads=8)


@app.context_processor
def inject_machine_hostname() -> Dict[str, str]:
    return {"machine_hostname": socket.gethostname()}


def _safe_next_target(target: Optional[str]) -> str:
    fallback = url_for("screen_config")
    if not target:
        return fallback

    candidate = target.strip()
    if not candidate:
        return fallback

    if candidate.startswith("/") and not candidate.startswith("//"):
        return candidate

    parsed = urlsplit(candidate)
    if parsed.scheme and parsed.netloc and parsed.netloc == request.host:
        path = parsed.path or "/"
        return urlunsplit(("", "", path, parsed.query, ""))

    return fallback


@app.route("/login", methods=["GET", "POST"])
def login() -> Any:
    if not _is_auth_enabled():
        return redirect(url_for("screen_config"))

    error: Optional[str] = None
    next_url = _safe_next_target(request.args.get("next") or request.form.get("next"))

    if request.method == "POST":
        supplied_username = request.form.get("username", "").strip()
        supplied = request.form.get("password", "")
        username_matches = not SCREEN_UI_USERNAME or supplied_username == SCREEN_UI_USERNAME
        if SCREEN_UI_PASSWORD and username_matches and supplied == SCREEN_UI_PASSWORD:
            session["screen_ui_authenticated"] = True
            return redirect(next_url)
        error = "Incorrect username or password"

    return render_template(
        "login.html",
        error=error,
        next_url=next_url,
        login_username=SCREEN_UI_USERNAME,
    )


@app.post("/logout")
def logout() -> Any:
    session.pop("screen_ui_authenticated", None)
    return redirect(url_for("login"))


@app.route("/", methods=["GET"])
def screen_config() -> str:
    config = _load_active_config()
    style_config = _load_active_style_config()
    layouts_config = _load_active_layouts_config()
    entries = _build_screen_entries(config, style_config)
    quad_config = layouts_config.get("screens", {}).get("quad", {})
    quad_enabled = bool(quad_config.get("enabled", False))
    quad_scroll_speed = _normalize_quad_scroll_speed(quad_config.get("scroll_speed", 1.0))
    quad_pages = quad_config.get("pages", [])
    playlists, playlist_assignments = _build_playlist_assignments(config)
    return render_template(
        "screen_config.html",
        screens=entries,
        screen_ids=sorted(SCREEN_IDS),
        quad_enabled=quad_enabled,
        quad_scroll_speed=quad_scroll_speed,
        quad_pages=quad_pages,
        config_path=DEFAULT_CONFIG_PATH,
        playlists=playlists,
        playlist_assignments=playlist_assignments,
    )


@app.get("/screenshots")
def screen_screenshots() -> str:
    entries = _build_screenshot_entries()
    return render_template("screenshots.html", screens=entries, display_status=_load_display_status())


@app.get("/api/screenshots")
def get_screenshots() -> Any:
    return jsonify({"screens": _build_screenshot_entries(), "display_status": _load_display_status()})


@app.get("/screenshots/current/<path:filename>")
def screenshot_current(filename: str) -> Any:
    if not filename.lower().endswith(ALLOWED_SCREEN_EXTS):
        abort(404)
    current_dir = _current_screenshot_dir()
    return send_from_directory(str(current_dir), filename)


@app.get("/screenshots/file/<path:relative_path>")
def screenshot_file(relative_path: str) -> Any:
    if not relative_path.lower().endswith(ALLOWED_SCREEN_EXTS):
        abort(404)
    storage_paths = resolve_storage_paths()
    screenshot_dir = storage_paths.screenshot_dir.resolve()
    target = (screenshot_dir / relative_path).resolve()
    if screenshot_dir != target and screenshot_dir not in target.parents:
        abort(404)
    if not target.exists() or not target.is_file():
        abort(404)
    return send_from_directory(str(screenshot_dir), relative_path)


@app.get("/api/screens")
def get_screens() -> Any:
    config = _load_active_config()
    style_config = _load_active_style_config()
    layouts_config = _load_active_layouts_config()
    return jsonify(
        {
            "screens": _build_screen_entries(config, style_config),
            "screen_ids": sorted(SCREEN_IDS),
            "quad_enabled": bool(layouts_config.get("screens", {}).get("quad", {}).get("enabled", False)),
            "quad_scroll_speed": _normalize_quad_scroll_speed(layouts_config.get("screens", {}).get("quad", {}).get("scroll_speed", 1.0)),
            "quad_pages": layouts_config.get("screens", {}).get("quad", {}).get("pages", []),
        }
    )


@app.get("/api/screens/defaults")
def get_default_screens() -> Any:
    config = _load_config(DEFAULT_CONFIG_PATH)
    style_config = _load_style_config(STYLE_CONFIG_PATH)
    layouts_config = _load_layouts_config(LAYOUTS_CONFIG_PATH)
    return jsonify({
        "screens": _build_screen_entries(config, style_config),
        "quad_enabled": bool(layouts_config.get("screens", {}).get("quad", {}).get("enabled", False)),
        "quad_scroll_speed": _normalize_quad_scroll_speed(layouts_config.get("screens", {}).get("quad", {}).get("scroll_speed", 1.0)),
        "quad_pages": layouts_config.get("screens", {}).get("quad", {}).get("pages", []),
    })


@app.get("/api/screens/export")
def export_screens() -> Any:
    config = _load_active_config()
    payload = json.dumps(config, indent=2)
    return (
        payload,
        200,
        {
            "Content-Type": "application/json",
            "Content-Disposition": "attachment; filename=screens_config.export.json",
        },
    )


@app.post("/api/screens")
def save_screens() -> Any:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid payload"}), 400
    entries = payload.get("screens")
    if not isinstance(entries, list):
        return jsonify({"error": "Screens list required"}), 400

    try:
        config = _build_config(entries)
        for key, expected_type in (("playlists", dict), ("sequence", list)):
            value = payload.get(key)
            if isinstance(value, expected_type):
                config[key] = value
        config, _ = _normalize_legacy_scoreboard_ids(config)
        style_config = _load_active_style_config()
        style_config = _build_style_config(entries, style_config)
        layouts = _build_layouts(payload)
        build_scheduler(config)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    _save_config(config)
    _save_style_config(style_config)
    _save_layouts_config(layouts)
    return jsonify(
        {
            "status": "ok",
            "screens": _build_screen_entries(config, style_config),
            "quad_enabled": bool(layouts.get("screens", {}).get("quad", {}).get("enabled", False)),
            "quad_scroll_speed": _normalize_quad_scroll_speed(layouts.get("screens", {}).get("quad", {}).get("scroll_speed", 1.0)),
            "quad_pages": layouts.get("screens", {}).get("quad", {}).get("pages", []),
        }
    )


@app.post("/api/screens/import")
def import_screens() -> Any:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid payload"}), 400

    config_payload = payload.get("config", payload)
    derived_style_payload: Optional[Dict[str, Any]] = None
    derived_layouts_payload: Optional[Dict[str, Any]] = None
    try:
        if isinstance(config_payload, dict) and isinstance(config_payload.get("screens"), list):
            entries = config_payload.get("screens", [])
            config = _build_config(entries)
            for key in ("playlists", "sequence"):
                value = config_payload.get(key)
                if value is not None:
                    config[key] = value
            config, _ = _normalize_legacy_scoreboard_ids(config)
            derived_style_payload = _build_style_config(entries, _load_active_style_config())
            quad_pages_payload = payload.get("quad_pages") if isinstance(payload, dict) else None
            quad_enabled_payload = payload.get("quad_enabled") if isinstance(payload, dict) else False
            if isinstance(quad_pages_payload, list):
                derived_layouts_payload = _build_layouts({"quad_enabled": quad_enabled_payload, "quad_pages": quad_pages_payload})
        else:
            config = _normalize_import_config_payload(config_payload)
            config, _ = _normalize_legacy_scoreboard_ids(config)
        build_scheduler(config)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    style_payload = payload.get("style")
    if derived_style_payload is not None:
        style_config = derived_style_payload
    elif style_payload is not None:
        try:
            style_config = _validate_style_payload(style_payload)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
    else:
        style_config = _load_active_style_config()

    layouts_payload = payload.get("layouts")
    if derived_layouts_payload is not None:
        layouts_config = derived_layouts_payload
    elif layouts_payload is not None:
        layouts_config = _normalize_layouts_config(layouts_payload)
    else:
        layouts_config = _load_active_layouts_config()

    _save_config(config)
    _save_style_config(style_config)
    _save_layouts_config(layouts_config)
    entries = _build_screen_entries(config, style_config)
    return jsonify(
        {
            "status": "ok",
            "screens": entries,
            "quad_enabled": bool(layouts_config.get("screens", {}).get("quad", {}).get("enabled", False)),
            "quad_scroll_speed": _normalize_quad_scroll_speed(layouts_config.get("screens", {}).get("quad", {}).get("scroll_speed", 1.0)),
            "quad_pages": layouts_config.get("screens", {}).get("quad", {}).get("pages", []),
        }
    )




if __name__ == "__main__":
    run_config_ui()
