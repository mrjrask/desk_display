#!/usr/bin/env python3
"""Continuously upload this Pi's current screenshots to a remote Feed server.

Companion to ``feed_server.py``: run this on any desk_display Pi (e.g. a
HyperPixel or HyperPixel Square install) that should mirror its live
screenshots onto a centralized Feed host instead of (or in addition to)
serving its own local ``/feed`` page. It watches the same
``<screenshot_dir>/current`` folder that ``main.py`` writes on every render
(see ``paths.resolve_storage_paths``) and POSTs any screen whose file has
changed since the last successful upload to the Feed server's ingest API.
"""
from __future__ import annotations

import logging
import os
import signal
import socket
import sys
from pathlib import Path
from threading import Event
from typing import Iterator

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import resolve_storage_paths  # noqa: E402

LOGGER = logging.getLogger("desk_display.screenshot_uploader")
_STOP_EVENT = Event()

ALLOWED_SCREEN_EXTS = (".png", ".jpg", ".jpeg")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


FEED_UPLOAD_URL = os.environ.get("FEED_UPLOAD_URL", "").strip().rstrip("/")
FEED_UPLOAD_TOKEN = os.environ.get("FEED_UPLOAD_TOKEN", "").strip()
FEED_SOURCE_NAME = os.environ.get("FEED_SOURCE_NAME", "").strip() or socket.gethostname()
FEED_UPLOAD_INTERVAL_SECONDS = max(1.0, _env_float("FEED_UPLOAD_INTERVAL_SECONDS", 5.0))
FEED_UPLOAD_TIMEOUT_SECONDS = max(1.0, _env_float("FEED_UPLOAD_TIMEOUT_SECONDS", 10.0))


def _request_stop(signum: int, _frame: object) -> None:
    LOGGER.info("Received signal %s; stopping screenshot uploader loop.", signum)
    _STOP_EVENT.set()


def _iter_current_screenshots(current_dir: Path) -> Iterator[Path]:
    if not current_dir.is_dir():
        return
    for entry in sorted(current_dir.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in ALLOWED_SCREEN_EXTS:
            continue
        yield entry


def _content_type_for(path: Path) -> str:
    if path.suffix.lower() in (".jpg", ".jpeg"):
        return "image/jpeg"
    return "image/png"


def _upload_file(session: requests.Session, path: Path) -> bool:
    screen_id = path.stem
    url = f"{FEED_UPLOAD_URL}/api/feed/{FEED_SOURCE_NAME}/upload"
    headers = {"Authorization": f"Bearer {FEED_UPLOAD_TOKEN}"} if FEED_UPLOAD_TOKEN else {}
    try:
        with open(path, "rb") as fh:
            response = session.post(
                url,
                headers=headers,
                data={"screen_id": screen_id},
                files={"file": (path.name, fh, _content_type_for(path))},
                timeout=FEED_UPLOAD_TIMEOUT_SECONDS,
            )
    except requests.RequestException as exc:
        LOGGER.warning("Feed upload for '%s' failed: %s", screen_id, exc)
        return False

    if response.status_code != 200:
        LOGGER.warning(
            "Feed upload for '%s' failed: %s %s",
            screen_id,
            response.status_code,
            response.text[:200],
        )
        return False

    LOGGER.debug("Uploaded '%s' to %s", screen_id, url)
    return True


def run_loop() -> int:
    logging.basicConfig(
        level=os.getenv("FEED_UPLOAD_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not FEED_UPLOAD_URL:
        LOGGER.error(
            "FEED_UPLOAD_URL is not set; nothing to upload to. "
            "Set it in .env, e.g. FEED_UPLOAD_URL=http://192.168.1.200:5003"
        )
        return 1

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    storage_paths = resolve_storage_paths()
    current_dir = storage_paths.current_screenshot_dir

    LOGGER.info(
        "Uploading screenshots from %s to %s as source '%s' every %.1fs",
        current_dir,
        FEED_UPLOAD_URL,
        FEED_SOURCE_NAME,
        FEED_UPLOAD_INTERVAL_SECONDS,
    )

    last_uploaded_mtimes: dict[str, float] = {}
    session = requests.Session()

    while not _STOP_EVENT.is_set():
        for path in _iter_current_screenshots(current_dir):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            key = path.name
            if last_uploaded_mtimes.get(key) == mtime:
                continue
            if _upload_file(session, path):
                last_uploaded_mtimes[key] = mtime
        _STOP_EVENT.wait(FEED_UPLOAD_INTERVAL_SECONDS)

    return 0


if __name__ == "__main__":
    raise SystemExit(run_loop())
