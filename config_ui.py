#!/usr/bin/env python3
"""Web UI for editing the screen rotation configuration."""
from __future__ import annotations

import contextlib
import json
import logging
import os
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

from flask import (
    Flask,
    abort,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)

import config
from paths import (
    resolve_layouts_config_path,
    resolve_screens_config_paths,
    resolve_storage_paths,
    resolve_style_config_path,
)
from schedule import build_scheduler
from screens_catalog import SCREEN_IDS, canonical_screen_id

# Config path precedence/fallback rules are centralized in paths.py.
_screens_config_paths = resolve_screens_config_paths()
DEFAULT_CONFIG_PATH = str(_screens_config_paths.default_path)
LOCAL_CONFIG_PATH = str(_screens_config_paths.local_override_path)
DEFAULT_SCREENS_PATH = os.environ.get(
    "DEFAULT_SCREENS_PATH", str(Path(__file__).resolve().parent / "default_screens_large.json")
)
DEFAULT_SCREEN_BUNDLES = {
    "large": {
        "label": "Large Default Configuration",
        "path": os.environ.get("DEFAULT_SCREENS_LARGE_PATH", DEFAULT_SCREENS_PATH),
    },
    "small": {
        "label": "Small Default Configuration",
        "path": os.environ.get(
            "DEFAULT_SCREENS_SMALL_PATH",
            str(Path(__file__).resolve().parent / "default_screens_small.json"),
        ),
    },
}
DEFAULT_SCREEN_PROFILE = "large"
STYLE_CONFIG_PATH = str(resolve_style_config_path())
LAYOUTS_CONFIG_PATH = str(resolve_layouts_config_path())

SCREEN_CONFIG_HOST = os.environ.get("SCREEN_CONFIG_HOST", "0.0.0.0")
SCREEN_CONFIG_PORT = int(os.environ.get("SCREEN_CONFIG_PORT", "5002"))
SCREEN_UI_USERNAME = os.environ.get("SCREEN_UI_USERNAME", "")
SCREEN_UI_PASSWORD = os.environ.get("SCREEN_UI_PASSWORD", "")
ALLOWED_SCREEN_EXTS = (".png", ".jpg", ".jpeg")
# The Waveshare OLED/LCD HAT is a separate pair of small hardware displays
# driven by scripts/waveshare_oled_status.py, not a rotation "screen" from
# screens_config.json. They're appended to the screenshots page (rather than
# the main screen loop) only when that helper has actually saved a frame, so
# desks without the OLED HAT don't show two permanently-empty cards.
OLED_SCREEN_IDS = ("oled left", "oled right")
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
        canonical: list[str] = []
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


def _normalize_scroll_speed(value: Any) -> float:
    try:
        speed = float(value)
    except (TypeError, ValueError):
        speed = 1.0
    return round(min(3.0, max(0.25, speed)), 2)


def _normalize_scroll_smoothness(value: Any) -> float:
    try:
        smoothness = float(value)
    except (TypeError, ValueError):
        smoothness = 1.0
    return round(min(2.0, max(0.5, smoothness)), 2)


def _normalize_scroll_settings(value: Any) -> dict[str, float]:
    settings = value if isinstance(value, dict) else {}
    return {
        "speed": _normalize_scroll_speed(settings.get("speed", 1.0)),
        "smoothness": _normalize_scroll_smoothness(settings.get("smoothness", 1.0)),
    }


def _normalize_screen_scroll_override(value: Any) -> Optional[dict[str, float]]:
    """Normalize a per-screen ``scroll`` override, or None when unset.

    Only speed is exposed per-screen (smoothness stays a global-only knob);
    a screen with no override falls back to the global scroll speed.
    """

    if not isinstance(value, dict) or "speed" not in value:
        return None
    return {"speed": _normalize_scroll_speed(value.get("speed"))}


def _merge_screen_specs(existing: Any, incoming: Any) -> Any:
    if existing is None:
        return incoming

    existing_freq = _coerce_frequency(existing.get("frequency", 0)) if isinstance(existing, dict) else _coerce_frequency(existing)
    incoming_freq = _coerce_frequency(incoming.get("frequency", 0)) if isinstance(incoming, dict) else _coerce_frequency(incoming)

    if existing_freq is None or incoming_freq is None:
        return incoming
    return incoming if incoming_freq > existing_freq else existing


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
    screen_auth_enabled = os.environ.get("SCREEN_AUTH_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return bool(os.environ.get("SCREEN_UI_PASSWORD", "")) or screen_auth_enabled


def _get_auth_username() -> str:
    return os.environ.get("SCREEN_UI_USERNAME", SCREEN_UI_USERNAME)


def _get_auth_password() -> str:
    return os.environ.get("SCREEN_UI_PASSWORD", SCREEN_UI_PASSWORD)


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


def _load_config(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {"screens": {}}
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object")
    screens = data.get("screens")
    if not isinstance(screens, dict):
        raise ValueError("Configuration must include a 'screens' mapping")
    normalized, _ = _normalize_legacy_scoreboard_ids(data)
    normalized["scroll"] = _normalize_scroll_settings(normalized.get("scroll"))
    return normalized


def _validate_config_payload(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object")
    screens = data.get("screens")
    if not isinstance(screens, dict):
        raise ValueError("Configuration must include a 'screens' mapping")
    data["scroll"] = _normalize_scroll_settings(data.get("scroll"))
    return data


def _normalize_import_config_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize imported config values so scheduler validation is predictable."""

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
            hide_after_at = (
                str(hide_after_at_raw).strip()
                if hide_after_at_raw is not None
                else ""
            )
            scroll_override = _normalize_screen_scroll_override(raw.get("scroll"))

            alt_payload: Optional[dict[str, Any]] = None
            alt = raw.get("alt")
            if isinstance(alt, dict):
                alt_payload = dict(alt)
                alt_payload["screen"] = _canonicalize_screen_reference(alt_payload.get("screen"))
                alt_frequency = alt_payload.get("frequency")
                if alt_frequency is not None:
                    with contextlib.suppress(TypeError, ValueError):
                        alt_payload["frequency"] = int(alt_frequency)

            normalized_spec: dict[str, Any] = {"frequency": frequency_int}
            if isinstance(extra_seconds, int) and extra_seconds > 0:
                normalized_spec["extra_seconds"] = extra_seconds
            if alt_payload is not None:
                normalized_spec["alt"] = alt_payload
            if hide_after_enabled and hide_after_at:
                normalized_spec["hide_after_enabled"] = True
                normalized_spec["hide_after_at"] = hide_after_at
            if scroll_override is not None:
                normalized_spec["scroll"] = scroll_override
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
    result["scroll"] = _normalize_scroll_settings(result.get("scroll"))
    cleaned, _ = _normalize_legacy_scoreboard_ids(result)
    return cleaned




def _default_layouts_config() -> dict[str, Any]:
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


def _load_layouts_config(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return _default_layouts_config()
    return _normalize_layouts_config(data)


def _load_active_layouts_config() -> dict[str, Any]:
    return _load_layouts_config(LAYOUTS_CONFIG_PATH)


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
        if not isinstance(raw_page, dict):
            continue
        page = _normalize_quad_page(raw_page, defaults)
        pages.append(page)

    if not pages:
        pages = [{"tiles": defaults.copy()}]

    return {"screens": {"quad": {"enabled": quad_enabled, "scroll_speed": quad_scroll_speed, "pages": pages}}}


def _load_style_config(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {"screens": {}}
    if not isinstance(data, dict):
        raise ValueError("Style configuration must be a JSON object")
    screens = data.get("screens")
    if not isinstance(screens, dict):
        raise ValueError("Style configuration must include a 'screens' mapping")
    return data


def _resolve_default_screens_path(profile: Optional[str] = None) -> str:
    profile_id = (profile or DEFAULT_SCREEN_PROFILE).strip().lower()
    if profile_id == "large":
        return DEFAULT_SCREENS_PATH
    if profile_id in DEFAULT_SCREEN_BUNDLES:
        return DEFAULT_SCREEN_BUNDLES[profile_id]["path"]
    raise ValueError("Unknown default configuration. Choose large or small.")


def _load_default_screens_bundle(profile: Optional[str] = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the repo-backed defaults used by the web UI Load Defaults button."""

    default_screens_path = _resolve_default_screens_path(profile)
    try:
        with open(default_screens_path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        payload = _load_config(DEFAULT_CONFIG_PATH)

    if not isinstance(payload, dict):
        raise ValueError("Default screens file must be a JSON object")

    config_payload = payload.get("config") if isinstance(payload.get("config"), dict) else payload
    config = _validate_config_payload(config_payload)
    config, _ = _normalize_legacy_scoreboard_ids(config)

    style_config = _load_style_config(STYLE_CONFIG_PATH)

    layouts_payload = payload.get("layouts")
    if layouts_payload is not None:
        layouts_config = _normalize_layouts_config(layouts_payload)
    else:
        layouts_config = _load_layouts_config(LAYOUTS_CONFIG_PATH)

    return config, style_config, layouts_config

def _load_active_config() -> dict[str, Any]:
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


def _load_active_style_config() -> dict[str, Any]:
    return _load_style_config(STYLE_CONFIG_PATH)


def _parse_alt_screen(value: Optional[str]) -> Optional[list[str]]:
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


def _build_playlist_assignments(config: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, str]]:
    config_playlists = config.get("playlists")
    if not isinstance(config_playlists, dict):
        return [], {}

    sequence = config.get("sequence")
    ordered_ids: list[str] = []
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


def _apply_playlist_grouping(
    ordered_screen_ids: list[str],
    playlists: list[dict[str, str]],
    playlist_assignments: dict[str, str],
) -> list[str]:
    """Reorder screen ids to match the Config page's rendered grouping.

    The Config page's JS groups rows by playlist on every render: the
    "Ungrouped" screens first, then each playlist's screens in the
    playlist sequence order, with each group's screens kept in their
    relative ``ordered_screen_ids`` order. This mirrors that exact
    algorithm (see ``renderScreens`` in screen_config.html) so the
    Screenshots page's arrangement matches what's actually displayed on
    the Config page.
    """

    group_ids = [""] + [playlist["id"] for playlist in playlists]
    grouped: list[str] = []
    for group_id in group_ids:
        grouped.extend(
            screen_id
            for screen_id in ordered_screen_ids
            if playlist_assignments.get(screen_id, "") == group_id
        )
    return grouped


def _build_screenshot_entry(
    screen_id: str,
    screenshot_dir: Path,
    current_dir: Path,
    *,
    search_history: bool = True,
) -> dict[str, Any]:
    prefix = _sanitize_filename_prefix(screen_id)
    entry: dict[str, Any] = {
        "id": screen_id,
        "path": None,
        "timestamp": None,
        "elapsed": None,
        "version": None,
        "is_stale": False,
    }
    candidates: list[Path] = [
        current_dir / f"{prefix}{ext}" for ext in ALLOWED_SCREEN_EXTS
    ]
    if search_history:
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
    return entry


def _build_screenshot_entries() -> list[dict[str, Any]]:
    storage_paths = resolve_storage_paths()
    screenshot_dir = storage_paths.screenshot_dir
    current_dir = storage_paths.current_screenshot_dir
    config = _load_active_config()
    screens_config = config.get("screens", {})
    ordered_screen_ids = _ordered_screen_ids(screens_config)
    playlists, playlist_assignments = _build_playlist_assignments(config)
    ordered_screen_ids = _apply_playlist_grouping(ordered_screen_ids, playlists, playlist_assignments)
    entries = [
        _build_screenshot_entry(screen_id, screenshot_dir, current_dir)
        for screen_id in ordered_screen_ids
    ]

    # The OLED HAT displays aren't part of screens_config's rotation, so they
    # aren't in ordered_screen_ids. Only their latest "current" frame exists
    # (no rotating history), and the card stays hidden by the "Hide screens
    # without screenshots" toggle until the OLED helper has actually saved one.
    entries.extend(
        _build_screenshot_entry(screen_id, screenshot_dir, current_dir, search_history=False)
        for screen_id in OLED_SCREEN_IDS
    )
    return entries


def _load_display_status() -> dict[str, Any]:
    storage_paths = resolve_storage_paths()
    status_path = storage_paths.current_screenshot_dir / "display_status.json"

    status: dict[str, Any] = {
        "screen_id": None,
        "rendered_at": None,
        "elapsed": None,
        "loop_iteration": None,
        "frame_id": None,
        "screen_play_counts": {},
        "display": {
            "profile_id": config.get_display_profile_id(),
            "width": config.WIDTH,
            "height": config.HEIGHT,
        },
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

    display_payload = payload.get("display")
    if isinstance(display_payload, dict):
        profile_id = display_payload.get("profile_id")
        width = display_payload.get("width")
        height = display_payload.get("height")
        if isinstance(profile_id, str) and profile_id.strip():
            status["display"]["profile_id"] = profile_id.strip()
        if isinstance(width, int) and width > 0:
            status["display"]["width"] = width
        if isinstance(height, int) and height > 0:
            status["display"]["height"] = height

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

    raw_play_counts = payload.get("screen_play_counts")
    if isinstance(raw_play_counts, dict):
        parsed_counts: dict[str, int] = {}
        for screen_name, count in raw_play_counts.items():
            if isinstance(screen_name, str) and isinstance(count, int):
                parsed_counts[screen_name] = count
        status["screen_play_counts"] = parsed_counts

    return status


def _load_service_status(unit_name: str = "desk_display.service") -> dict[str, Any]:
    status: dict[str, Any] = {
        "unit": unit_name,
        "active_state": "unknown",
        "sub_state": "unknown",
        "unit_file_state": "unknown",
        "is_active": False,
        "summary": "Unknown",
        "error": None,
    }

    try:
        result = subprocess.run(
            [
                "systemctl",
                "show",
                "--no-pager",
                "--property=ActiveState,SubState,UnitFileState",
                unit_name,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        status["error"] = str(exc)
        status["summary"] = "Unavailable"
        return status

    if result.returncode != 0:
        stderr = result.stderr.strip()
        status["error"] = stderr or f"systemctl exited with {result.returncode}"
        status["summary"] = "Unavailable"
        return status

    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key == "ActiveState":
            status["active_state"] = value or "unknown"
        elif key == "SubState":
            status["sub_state"] = value or "unknown"
        elif key == "UnitFileState":
            status["unit_file_state"] = value or "unknown"

    status["is_active"] = status["active_state"] == "active"
    active = status["active_state"]
    sub = status["sub_state"]
    enabled = status["unit_file_state"]
    status["summary"] = f"{active} ({sub}), {enabled}"
    return status

def _ordered_screen_ids(
    screens_config: Any,
    *,
    exclude: frozenset[str] = frozenset(),
) -> list[str]:
    """Return screen ids in the same order they're arranged on the Config page.

    Screens explicitly ordered in ``screens_config`` (a dict, so insertion
    order reflects the drag-and-drop arrangement saved from the Config page)
    come first, followed by any catalog screens not yet present in the
    config, in catalog order. Shared by both the Config page and the
    Screenshots page so their arrangements can't drift apart.
    """

    ordered_screen_ids: list[str] = []
    if isinstance(screens_config, dict):
        ordered_screen_ids.extend(
            screen_id for screen_id in screens_config.keys() if screen_id not in exclude
        )
    for screen_id in SCREEN_IDS:
        if screen_id in exclude:
            continue
        if screen_id not in ordered_screen_ids:
            ordered_screen_ids.append(screen_id)
    return ordered_screen_ids


def _build_screen_entries(
    config: dict[str, Any],
    style_config: dict[str, Any],
) -> list[dict[str, Any]]:
    screens = config.get("screens", {})
    if not isinstance(screens, dict):
        return []

    ordered_screen_ids = _ordered_screen_ids(screens, exclude=HIDDEN_CONFIG_SCREEN_IDS)

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
            "scroll_speed_enabled": False,
            "scroll_speed": 1.0,
        }
        if isinstance(raw, dict):
            entry["frequency"] = raw.get("frequency", 0)
            entry["extra_seconds"] = raw.get("extra_seconds", 0)
            scroll_override = _normalize_screen_scroll_override(raw.get("scroll"))
            if scroll_override is not None:
                entry["scroll_speed_enabled"] = True
                entry["scroll_speed"] = scroll_override["speed"]
            hide_after_at = raw.get("hide_after_at")
            if isinstance(hide_after_at, str):
                try:
                    parsed_hide_after = datetime.fromisoformat(hide_after_at.strip())
                    entry["hide_after_at"] = parsed_hide_after.strftime("%Y-%m-%dT%H:%M")
                except ValueError:
                    entry["hide_after_at"] = hide_after_at
            else:
                entry["hide_after_at"] = ""
            entry["hide_after_enabled"] = bool(raw.get("hide_after_enabled", False))
            alt = raw.get("alt") if isinstance(raw.get("alt"), dict) else None
            if alt:
                entry["alt_screen"] = _serialize_alt_screen(alt.get("screen"))
                entry["alt_frequency"] = alt.get("frequency", "")
        else:
            entry["frequency"] = raw
        entries.append(entry)

    return entries


def _build_selectable_screen_ids(entries: list[dict[str, Any]]) -> list[str]:
    ordered_screen_ids: list[str] = []

    for entry in entries:
        screen_id = entry.get("id")
        if not isinstance(screen_id, str):
            continue
        if screen_id in HIDDEN_CONFIG_SCREEN_IDS:
            continue
        if screen_id not in ordered_screen_ids:
            ordered_screen_ids.append(screen_id)

    for screen_id in SCREEN_IDS:
        if screen_id in HIDDEN_CONFIG_SCREEN_IDS:
            continue
        if screen_id not in ordered_screen_ids:
            ordered_screen_ids.append(screen_id)

    return ordered_screen_ids


def _build_config(entries: list[dict[str, Any]]) -> dict[str, Any]:
    screens: dict[str, Any] = {}
    for entry in entries:
        screen_id = canonical_screen_id(str(entry.get("id", "")).strip())
        if not screen_id:
            continue
        if screen_id in HIDDEN_CONFIG_SCREEN_IDS:
            continue
        frequency = int(entry.get("frequency", 0))
        extra_seconds = int(entry.get("extra_seconds", 0))
        if extra_seconds < 0:
            raise ValueError(f"Additional seconds for '{screen_id}' cannot be negative")
        alt_screen_raw = entry.get("alt_screen")
        alt_frequency = entry.get("alt_frequency")
        hide_after_at_raw = entry.get("hide_after_at")
        hide_after_at = str(hide_after_at_raw).strip() if hide_after_at_raw is not None else ""
        hide_after_enabled = bool(entry.get("hide_after_enabled", False))
        if hide_after_enabled and not hide_after_at:
            raise ValueError(f"Hide-after date/time for '{screen_id}' must be provided when enabled")
        scroll_speed_enabled = bool(entry.get("scroll_speed_enabled", False))
        scroll_override = (
            {"speed": _normalize_scroll_speed(entry.get("scroll_speed", 1.0))}
            if scroll_speed_enabled
            else None
        )
        alt_screen = _parse_alt_screen(str(alt_screen_raw).strip()) if alt_screen_raw is not None else None

        spec: dict[str, Any] = {"frequency": frequency}
        if alt_screen:
            alt_screen = [canonical_screen_id(item) for item in alt_screen]
            alt_frequency_int = int(alt_frequency) if alt_frequency not in ("", None) else 1
            spec["alt"] = {
                "screen": alt_screen[0] if len(alt_screen) == 1 else alt_screen,
                "frequency": alt_frequency_int,
            }
        if extra_seconds > 0:
            spec["extra_seconds"] = extra_seconds
        if hide_after_enabled and hide_after_at:
            spec["hide_after_enabled"] = True
            spec["hide_after_at"] = hide_after_at
        if scroll_override is not None:
            spec["scroll"] = scroll_override

        screens[screen_id] = frequency if spec.keys() == {"frequency"} else spec
    cleaned, _ = _normalize_legacy_scoreboard_ids({"screens": screens})
    return cleaned


def _save_config(config: dict[str, Any]) -> None:
    tmp_path = f"{LOCAL_CONFIG_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")
    os.replace(tmp_path, LOCAL_CONFIG_PATH)


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
def inject_machine_hostname() -> dict[str, str]:
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
        auth_username = _get_auth_username()
        auth_password = _get_auth_password()
        username_matches = not auth_username or supplied_username == auth_username
        if auth_password and username_matches and supplied == auth_password:
            session["screen_ui_authenticated"] = True
            return redirect(next_url)
        error = "Incorrect username or password"

    return render_template(
        "login.html",
        error=error,
        next_url=next_url,
        login_username=_get_auth_username(),
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
    selectable_screen_ids = _build_selectable_screen_ids(entries)
    quad_config = layouts_config.get("screens", {}).get("quad", {})
    quad_enabled = bool(quad_config.get("enabled", False))
    quad_scroll_speed = _normalize_quad_scroll_speed(quad_config.get("scroll_speed", 1.0))
    quad_pages = quad_config.get("pages", [])
    playlists, playlist_assignments = _build_playlist_assignments(config)
    scroll_settings = _normalize_scroll_settings(config.get("scroll"))
    return render_template(
        "screen_config.html",
        screens=entries,
        scroll_settings=scroll_settings,
        screen_ids=selectable_screen_ids,
        quad_enabled=quad_enabled,
        quad_scroll_speed=quad_scroll_speed,
        quad_pages=quad_pages,
        config_path=DEFAULT_CONFIG_PATH,
        playlists=playlists,
        playlist_assignments=playlist_assignments,
        service_status=_load_service_status(),
    )


@app.get("/screenshots")
def screen_screenshots() -> str:
    entries = _build_screenshot_entries()
    return render_template(
        "screenshots.html",
        screens=entries,
        display_status=_load_display_status(),
        service_status=_load_service_status(),
    )


@app.get("/api/screenshots")
def get_screenshots() -> Any:
    return jsonify(
        {
            "screens": _build_screenshot_entries(),
            "display_status": _load_display_status(),
            "service_status": _load_service_status(),
        }
    )


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
    entries = _build_screen_entries(config, style_config)
    return jsonify(
        {
            "screens": entries,
            "screen_ids": _build_selectable_screen_ids(entries),
            "scroll": _normalize_scroll_settings(config.get("scroll")),
            "quad_enabled": bool(layouts_config.get("screens", {}).get("quad", {}).get("enabled", False)),
            "quad_scroll_speed": _normalize_quad_scroll_speed(layouts_config.get("screens", {}).get("quad", {}).get("scroll_speed", 1.0)),
            "quad_pages": layouts_config.get("screens", {}).get("quad", {}).get("pages", []),
        }
    )


@app.get("/api/screens/defaults")
def get_default_screens() -> Any:
    profile = request.args.get("profile", DEFAULT_SCREEN_PROFILE)
    try:
        config, style_config, layouts_config = _load_default_screens_bundle(profile)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    playlists, playlist_assignments = _build_playlist_assignments(config)
    return jsonify({
        "config": config,
        "default_profiles": DEFAULT_SCREEN_BUNDLES,
        "selected_default_profile": (profile or DEFAULT_SCREEN_PROFILE).strip().lower(),
        "screens": _build_screen_entries(config, style_config),
        "playlists": playlists,
        "scroll": _normalize_scroll_settings(config.get("scroll")),
        "playlist_assignments": playlist_assignments,
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
        # The UI no longer exposes global scroll speed/smoothness controls
        # (per-screen speed overrides replace them), so a save that doesn't
        # include "scroll" should leave whatever is already on disk alone
        # instead of resetting it to the 1.0/1.0 defaults.
        current_scroll = _load_active_config().get("scroll")
        config["scroll"] = _normalize_scroll_settings(payload.get("scroll", current_scroll))
        config, _ = _normalize_legacy_scoreboard_ids(config)
        if any(key in payload for key in ("quad_enabled", "quad_pages", "quad_tiles", "quad_scroll_speed")):
            layouts = _build_layouts(payload)
        else:
            layouts = _load_active_layouts_config()
        build_scheduler(config)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    _save_config(config)
    _save_layouts_config(layouts)
    return jsonify(
        {
            "status": "ok",
            "screens": _build_screen_entries(config, _load_active_style_config()),
            "scroll": _normalize_scroll_settings(config.get("scroll")),
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
    derived_layouts_payload: Optional[dict[str, Any]] = None
    try:
        if isinstance(config_payload, dict) and isinstance(config_payload.get("screens"), list):
            entries = config_payload.get("screens", [])
            config = _build_config(entries)
            for key in ("playlists", "sequence"):
                value = config_payload.get(key)
                if value is not None:
                    config[key] = value
            current_scroll = _load_active_config().get("scroll")
            config["scroll"] = _normalize_scroll_settings(config_payload.get("scroll", current_scroll))
            config, _ = _normalize_legacy_scoreboard_ids(config)
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

    layouts_payload = payload.get("layouts")
    if derived_layouts_payload is not None:
        layouts_config = derived_layouts_payload
    elif layouts_payload is not None:
        layouts_config = _normalize_layouts_config(layouts_payload)
    else:
        layouts_config = _load_active_layouts_config()

    _save_config(config)
    _save_layouts_config(layouts_config)
    entries = _build_screen_entries(config, _load_active_style_config())
    return jsonify(
        {
            "status": "ok",
            "screens": entries,
            "scroll": _normalize_scroll_settings(config.get("scroll")),
            "quad_enabled": bool(layouts_config.get("screens", {}).get("quad", {}).get("enabled", False)),
            "quad_scroll_speed": _normalize_quad_scroll_speed(layouts_config.get("screens", {}).get("quad", {}).get("scroll_speed", 1.0)),
            "quad_pages": layouts_config.get("screens", {}).get("quad", {}).get("pages", []),
        }
    )




if __name__ == "__main__":
    run_config_ui()
