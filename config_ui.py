#!/usr/bin/env python3
"""Web UI for editing the screen rotation configuration."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from paths import resolve_storage_paths
from schedule import build_scheduler
from screens_catalog import SCREEN_IDS

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

SCREEN_CONFIG_HOST = os.environ.get("SCREEN_CONFIG_HOST", "0.0.0.0")
SCREEN_CONFIG_PORT = int(os.environ.get("SCREEN_CONFIG_PORT", "5002"))
ALLOWED_SCREEN_EXTS = (".png", ".jpg", ".jpeg")

app = Flask(__name__)


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
    return data


def _validate_config_payload(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object")
    screens = data.get("screens")
    if not isinstance(screens, dict):
        raise ValueError("Configuration must include a 'screens' mapping")
    return data


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
        return _load_config(LOCAL_CONFIG_PATH)
    return _load_config(DEFAULT_CONFIG_PATH)


def _load_active_style_config() -> Dict[str, Any]:
    return _load_style_config(STYLE_CONFIG_PATH)


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


def _sanitize_filename_prefix(name: str) -> str:
    """Return a filesystem-friendly filename prefix."""

    safe = name.strip().replace("/", "-").replace("\\", "-")
    safe = safe.replace(" ", "_")
    safe = "".join(ch for ch in safe if ch.isalnum() or ch in ("_", "-"))
    return safe or "screen"


def _current_screenshot_dir() -> Path:
    storage_paths = resolve_storage_paths()
    return storage_paths.current_screenshot_dir


def _format_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def _build_screenshot_entries() -> List[Dict[str, Optional[str]]]:
    current_dir = _current_screenshot_dir()
    config = _load_active_config()
    screens_config = config.get("screens", {})
    ordered_screen_ids: List[str] = []
    if isinstance(screens_config, dict):
        ordered_screen_ids.extend(list(screens_config.keys()))
    for screen_id in SCREEN_IDS:
        if screen_id not in ordered_screen_ids:
            ordered_screen_ids.append(screen_id)
    entries: List[Dict[str, Optional[str]]] = []
    for screen_id in ordered_screen_ids:
        prefix = _sanitize_filename_prefix(screen_id)
        filename = f"{prefix}.png"
        path = current_dir / filename
        entry: Dict[str, Optional[str]] = {
            "id": screen_id,
            "filename": None,
            "timestamp": None,
        }
        if path.exists():
            entry["filename"] = filename
            try:
                entry["timestamp"] = _format_timestamp(path.stat().st_mtime)
            except OSError:
                entry["timestamp"] = None
        entries.append(entry)
    return entries


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
    if lowered == "travel map":
        return (18, 18, 18)
    if any(token in lowered for token in ("scoreboard", "standings", "overview", "stand1", "stand2")):
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

    ordered_screen_ids = list(SCREEN_IDS)
    for screen_id in screens:
        if screen_id not in SCREEN_IDS:
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
        screen_id = str(entry.get("id", "")).strip()
        if not screen_id:
            continue
        frequency = int(entry.get("frequency", 0))
        alt_screen_raw = entry.get("alt_screen")
        alt_frequency = entry.get("alt_frequency")
        alt_screen = _parse_alt_screen(str(alt_screen_raw).strip()) if alt_screen_raw is not None else None
        if alt_screen:
            alt_frequency_int = int(alt_frequency) if alt_frequency not in ("", None) else 1
            alt_payload: Dict[str, Any] = {"screen": alt_screen[0] if len(alt_screen) == 1 else alt_screen}
            alt_payload["frequency"] = alt_frequency_int
            screens[screen_id] = {"frequency": frequency, "alt": alt_payload}
        else:
            screens[screen_id] = frequency
    return {"screens": screens}


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
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


@app.route("/", methods=["GET"])
def screen_config() -> str:
    config = _load_active_config()
    style_config = _load_active_style_config()
    entries = _build_screen_entries(config, style_config)
    return render_template(
        "screen_config.html",
        screens=entries,
        screen_ids=sorted(SCREEN_IDS),
        config_path=DEFAULT_CONFIG_PATH,
    )


@app.get("/screenshots")
def screen_screenshots() -> str:
    entries = _build_screenshot_entries()
    return render_template("screenshots.html", screens=entries)


@app.get("/screenshots/current/<path:filename>")
def screenshot_current(filename: str) -> Any:
    if not filename.lower().endswith(ALLOWED_SCREEN_EXTS):
        abort(404)
    current_dir = _current_screenshot_dir()
    return send_from_directory(str(current_dir), filename)


@app.get("/api/screens")
def get_screens() -> Any:
    config = _load_active_config()
    style_config = _load_active_style_config()
    return jsonify(
        {
            "screens": _build_screen_entries(config, style_config),
            "screen_ids": sorted(SCREEN_IDS),
        }
    )


@app.get("/api/screens/defaults")
def get_default_screens() -> Any:
    config = _load_config(DEFAULT_CONFIG_PATH)
    style_config = _load_style_config(STYLE_CONFIG_PATH)
    return jsonify({"screens": _build_screen_entries(config, style_config)})


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
        style_config = _load_active_style_config()
        style_config = _build_style_config(entries, style_config)
        build_scheduler(config)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    _save_config(config)
    _save_style_config(style_config)
    return jsonify({"status": "ok"})


@app.post("/api/screens/import")
def import_screens() -> Any:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid payload"}), 400

    config_payload = payload.get("config", payload)
    try:
        config = _validate_config_payload(config_payload)
        build_scheduler(config)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    _save_config(config)
    style_config = _load_active_style_config()
    entries = _build_screen_entries(config, style_config)
    return jsonify({"status": "ok", "screens": entries})


if __name__ == "__main__":
    run_config_ui()
