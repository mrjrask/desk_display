#!/usr/bin/env python3
"""Standalone web server hosting Feed pages fed by screenshots uploaded from
other desk_display Pis.

This is intentionally separate from ``config_ui.py``: that app serves a
``/feed`` page built from screens *this* machine renders locally. This app
instead serves one ``/feed/<source>`` page per remote Pi, built entirely from
screenshots pushed to it over HTTP by ``scripts/screenshot_uploader.py``
running on those other Pis. It has no dependency on the rendering stack
(``main.py``, ``screens/``, GPIO drivers, ...), so it can be installed by
itself (see ``Installers/install_feed_server.sh``) on a Pi that only needs to
aggregate and display screenshots, such as a Feed host with no attached
desk_display panel of its own.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import socket
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from flask import (
    Flask,
    abort,
    g,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from PIL import Image, UnidentifiedImageError

FEED_SERVER_HOST = os.environ.get("FEED_SERVER_HOST", "0.0.0.0")
FEED_SERVER_PORT = int(os.environ.get("FEED_SERVER_PORT", "5003"))
FEED_UPLOAD_TOKEN = os.environ.get("FEED_UPLOAD_TOKEN", "").strip()
FEED_MAX_UPLOAD_BYTES = int(os.environ.get("FEED_MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
FEED_STALE_SECONDS = int(os.environ.get("FEED_STALE_SECONDS", str(2 * 60)))
# A source's screenshot page (unlike the source list, which just flags a
# source as stale) drops a screen entirely once it hasn't been re-uploaded
# within this window, so a Pi that's stopped pushing frames doesn't linger.
FEED_SCREEN_STALE_SECONDS = int(os.environ.get("FEED_SCREEN_STALE_SECONDS", str(20 * 60)))
FEED_HEARTBEAT_STALE_SECONDS = int(os.environ.get("FEED_HEARTBEAT_STALE_SECONDS", str(10 * 60)))
ALLOWED_SCREEN_EXTS = (".png", ".jpg", ".jpeg")

_PROJECT_ROOT = Path(__file__).resolve().parent
FEED_STORAGE_DIR = Path(
    os.environ.get("FEED_STORAGE_DIR", str(_PROJECT_ROOT / "feed_uploads"))
).expanduser()
DEFAULT_SCREENS_LARGE_PATH = _PROJECT_ROOT / "default_screens_large.json"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = FEED_MAX_UPLOAD_BYTES
WEB_LOGGER = logging.getLogger("desk_display.feed_server")


def _sanitize_id(value: str) -> str:
    """Return a filesystem-safe identifier for a source name or screen id."""

    safe = value.strip().replace("/", "-").replace("\\", "-")
    safe = safe.replace(" ", "_")
    safe = "".join(ch for ch in safe if ch.isalnum() or ch in ("_", "-"))
    return safe or "unknown"


def _load_large_screen_order() -> list[str]:
    """Return sanitized screen ids in the order large-screen defaults play them.

    Flattens ``default_screens_large.json``'s ``sequence`` (playlist order)
    and each playlist's ``steps`` (screen order within it) into a single
    list, matching the exact id every screen's uploaded filename uses
    (``_sanitize_id`` mirrors the sanitizing ``main.py`` applies when it
    names screenshot files). Falls back to an empty list -- callers then
    fall back to alphabetical order -- if the defaults file is missing or
    malformed, e.g. on a Feed-server-only checkout without it.
    """

    try:
        payload = json.loads(DEFAULT_SCREENS_LARGE_PATH.read_text(encoding="utf-8"))
        config = payload["config"]
        playlists = config["playlists"]
        sequence = config["sequence"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return []

    order: list[str] = []
    for item in sequence:
        playlist = playlists.get(item.get("playlist")) if isinstance(item, dict) else None
        steps = playlist.get("steps") if isinstance(playlist, dict) else None
        if not isinstance(steps, list):
            continue
        for step in steps:
            screen = step.get("screen") if isinstance(step, dict) else None
            if isinstance(screen, str) and screen:
                order.append(_sanitize_id(screen))
    return order


LARGE_SCREEN_ORDER = _load_large_screen_order()


def _source_dir(source_id: str) -> Path:
    return FEED_STORAGE_DIR / source_id


def _source_current_dir(source_id: str) -> Path:
    return _source_dir(source_id) / "current"


def _format_timestamp(timestamp: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def _format_elapsed_since(timestamp: float) -> str:
    total_seconds = max(0, int(time.time() - timestamp))
    days, remainder = divmod(total_seconds, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    return f"{days}d {hours}h {minutes}m {seconds}s ago"


def _is_stale(timestamp: float) -> bool:
    return time.time() - timestamp > FEED_STALE_SECONDS


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


@app.context_processor
def inject_machine_hostname() -> dict[str, str]:
    return {"machine_hostname": socket.gethostname()}


def _build_source_screen_entries(source_id: str) -> list[dict[str, Any]]:
    current_dir = _source_current_dir(source_id)
    if not current_dir.is_dir():
        return []

    by_id: dict[str, dict[str, Any]] = {}
    for path in current_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in ALLOWED_SCREEN_EXTS:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        # Keep only the newest file per screen id, so a screen never shows
        # more than one screenshot even if a stale extra file lingers.
        existing = by_id.get(path.stem)
        if existing is not None and existing["version"] >= int(mtime):
            continue
        by_id[path.stem] = {
            "id": path.stem,
            "filename": path.name,
            "version": int(mtime),
        }

    # Large-screen-defaults order first, then any uploaded screen id that
    # defaults file doesn't know about (e.g. a newer screen), alphabetically.
    ordered_ids = [screen_id for screen_id in LARGE_SCREEN_ORDER if screen_id in by_id]
    remaining_ids = sorted(set(by_id) - set(ordered_ids))
    return [by_id[screen_id] for screen_id in ordered_ids + remaining_ids]


def _filter_fresh_screen_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop screen entries whose upload is older than ``FEED_SCREEN_STALE_SECONDS``."""

    now = time.time()
    return [
        entry
        for entry in entries
        if now - entry.get("version", 0) <= FEED_SCREEN_STALE_SECONDS
    ]


def _status_path(source_id: str) -> Path:
    return _source_current_dir(source_id) / "display_status.json"


def _load_source_display_status(source_id: str) -> dict[str, Any]:
    """Read the heartbeat a source Pi last uploaded for its display_status.json."""

    status: dict[str, Any] = {
        "screen_id": None,
        "rendered_at": None,
        "elapsed": None,
        "loop_iteration": None,
        "frame_id": None,
        "screen_play_counts": {},
        "display": None,
        "is_stale": True,
    }

    status_path = _status_path(source_id)
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
        display_info: dict[str, Any] = {}
        if isinstance(profile_id, str) and profile_id.strip():
            display_info["profile_id"] = profile_id.strip()
        if isinstance(width, int) and width > 0:
            display_info["width"] = width
        if isinstance(height, int) and height > 0:
            display_info["height"] = height
        if display_info:
            status["display"] = display_info

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
            status["is_stale"] = (time.time() - rendered_ts) > FEED_HEARTBEAT_STALE_SECONDS

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


def _list_sources() -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    if not FEED_STORAGE_DIR.is_dir():
        return sources
    for entry in sorted(FEED_STORAGE_DIR.iterdir()):
        if not entry.is_dir():
            continue
        current_dir = entry / "current"
        screen_count = 0
        last_mtime: Optional[float] = None
        if current_dir.is_dir():
            for path in current_dir.iterdir():
                if not path.is_file() or path.suffix.lower() not in ALLOWED_SCREEN_EXTS:
                    continue
                screen_count += 1
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                if last_mtime is None or mtime > last_mtime:
                    last_mtime = mtime
        sources.append(
            {
                "name": entry.name,
                "screen_count": screen_count,
                "timestamp": _format_timestamp(last_mtime) if last_mtime else None,
                "elapsed": _format_elapsed_since(last_mtime) if last_mtime else None,
                "is_stale": _is_stale(last_mtime) if last_mtime else True,
                "display_status": _load_source_display_status(entry.name),
            }
        )
    return sources


@app.get("/")
def feed_index() -> str:
    return render_template("feed_index.html", sources=_list_sources())


@app.get("/feed/<source>")
def feed_source_page(source: str) -> str:
    source_id = _sanitize_id(source)
    return render_template(
        "feed_source.html",
        source=source_id,
        screens=_filter_fresh_screen_entries(_build_source_screen_entries(source_id)),
        display_status=_load_source_display_status(source_id),
    )


@app.get("/api/feed/<source>/screenshots")
def feed_source_screenshots(source: str) -> Any:
    source_id = _sanitize_id(source)
    return jsonify(
        {
            "screens": _filter_fresh_screen_entries(_build_source_screen_entries(source_id)),
            "display_status": _load_source_display_status(source_id),
        }
    )


@app.get("/feed/<source>/file/<path:filename>")
def feed_screenshot_file(source: str, filename: str) -> Any:
    if not filename.lower().endswith(ALLOWED_SCREEN_EXTS):
        abort(404)
    source_id = _sanitize_id(source)
    current_dir = _source_current_dir(source_id).resolve()
    target = (current_dir / filename).resolve()
    if current_dir != target and current_dir not in target.parents:
        abort(404)
    if not target.exists() or not target.is_file():
        abort(404)
    return send_from_directory(str(current_dir), filename)


@app.post("/api/feed/<source>/upload")
def upload_screenshot(source: str) -> Any:
    if not FEED_UPLOAD_TOKEN:
        return jsonify({"error": "Feed server has no FEED_UPLOAD_TOKEN configured"}), 503

    auth_header = request.headers.get("Authorization", "")
    provided = auth_header[7:] if auth_header.startswith("Bearer ") else ""
    if not provided or not hmac.compare_digest(provided, FEED_UPLOAD_TOKEN):
        return jsonify({"error": "Unauthorized"}), 401

    screen_id_raw = request.form.get("screen_id", "").strip()
    if not screen_id_raw:
        return jsonify({"error": "screen_id is required"}), 400

    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "file is required"}), 400

    try:
        image = Image.open(uploaded.stream)
        image.verify()
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "Uploaded file is not a valid image"}), 400

    uploaded.stream.seek(0)
    try:
        image = Image.open(uploaded.stream)
        image.load()
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "Uploaded file is not a valid image"}), 400

    source_id = _sanitize_id(source)
    screen_id = _sanitize_id(screen_id_raw)
    target_dir = _source_current_dir(source_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    for existing in target_dir.glob(f"{screen_id}.*"):
        if existing.suffix.lower() in ALLOWED_SCREEN_EXTS:
            existing.unlink(missing_ok=True)

    target_path = target_dir / f"{screen_id}.png"
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{screen_id}.", suffix=".png.tmp", dir=target_dir
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, "wb") as tmp_file:
            image.save(tmp_file, format="PNG")
        os.replace(tmp_path, target_path)
    except OSError as exc:
        WEB_LOGGER.warning("Failed to persist upload for %s/%s: %s", source_id, screen_id, exc)
        tmp_path.unlink(missing_ok=True)
        return jsonify({"error": "Failed to store image"}), 500

    return jsonify({"status": "ok", "source": source_id, "screen_id": screen_id})


@app.post("/api/feed/<source>/status")
def upload_display_status(source: str) -> Any:
    if not FEED_UPLOAD_TOKEN:
        return jsonify({"error": "Feed server has no FEED_UPLOAD_TOKEN configured"}), 503

    auth_header = request.headers.get("Authorization", "")
    provided = auth_header[7:] if auth_header.startswith("Bearer ") else ""
    if not provided or not hmac.compare_digest(provided, FEED_UPLOAD_TOKEN):
        return jsonify({"error": "Unauthorized"}), 401

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON object body is required"}), 400

    source_id = _sanitize_id(source)
    target_dir = _source_current_dir(source_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    target_path = target_dir / "display_status.json"
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=".display_status.", suffix=".json.tmp", dir=target_dir
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_file:
            json.dump(payload, tmp_file)
        os.replace(tmp_path, target_path)
    except OSError as exc:
        WEB_LOGGER.warning("Failed to persist display status for %s: %s", source_id, exc)
        tmp_path.unlink(missing_ok=True)
        return jsonify({"error": "Failed to store status"}), 500

    return jsonify({"status": "ok", "source": source_id})


def run_feed_server(host: str = FEED_SERVER_HOST, port: int = FEED_SERVER_PORT) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S",
        )
    if not FEED_UPLOAD_TOKEN:
        WEB_LOGGER.warning(
            "FEED_UPLOAD_TOKEN is not set; uploads will be rejected until it is configured."
        )
    from waitress import serve

    serve(app, host=host, port=port, threads=8)


if __name__ == "__main__":
    run_feed_server()
