"""The config UI's Live page: every display's latest screenshots, kept current.

The browser version of ``scripts/collect_client_screenshots.py``.  The page
polls ``/api/live`` every few seconds and swaps in new images as they arrive,
grouped by screen (or by display), with each screenshot's capture age.

Where the screenshots come from, as in the script:

* Each online (or briefly late, "stale") display's own config UI answers
  ``/api/screenshots`` at the address it last reached the server from, on the
  same port as this config UI (``SCREEN_CONFIG_PORT``, 5002 by default).  The server fetches the
  list and proxies each image (``/api/live/<id>/image``), so the browser needs
  no route to the display.  Lists are reused for a few seconds and images are
  cached by capture time, so a page left open costs one small request per
  display per poll.  A display that asks for a password is sent this config
  UI's own SCREEN_UI_USERNAME / SCREEN_UI_PASSWORD.
* The server's own panel (combined mode, address 127.0.0.1) is read from this
  process directly; its images are this config UI's ``/screenshots/file/``.
* A display the server cannot reach (on another network) shows the
  screenshots it uploaded to the server (DESK_DISPLAY_CLIENT_UPLOAD_SCREENSHOTS),
  which also fill in screens its own config UI did not return.  Uploads arrive
  every few minutes, so they are older than direct ones.
"""
from __future__ import annotations

import os
import threading
import time
import urllib.error
import urllib.parse
from collections import OrderedDict
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from flask import Blueprint, Response, abort, jsonify, render_template, request

from remote_display.screenshot_uploads import ScreenshotInbox, UploadRejected, parse_time
from scripts.collect_client_screenshots import (
    ACTIVE_STATES,
    AuthRequired,
    ConfigUI,
    _is_loopback,
    client_base_url,
)

DEFAULT_CLIENT_PORT = 5002
# How often the page polls, how long one display's list is reused, how long a
# display that did not answer is left alone, and how long one request waits.
POLL_SECONDS = 5
LIST_TTL_SECONDS = 4.0
FAILURE_TTL_SECONDS = 20.0
FETCH_TIMEOUT_SECONDS = 4.0
IMAGE_CACHE_BYTES = 48 * 1024 * 1024
IMAGE_TYPES = ("image/png", "image/jpeg", "image/gif")


@dataclass
class _Listing:
    fetched_at: float
    entries: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


class _ImageCache:
    """Recently proxied images, keyed by display, path and capture time."""

    def __init__(self, max_bytes: int = IMAGE_CACHE_BYTES) -> None:
        self._max = max_bytes
        self._items: OrderedDict[tuple[str, str, str], tuple[bytes, str]] = OrderedDict()
        self._size = 0
        self._lock = threading.Lock()

    def get(self, key: tuple[str, str, str]) -> tuple[bytes, str] | None:
        with self._lock:
            item = self._items.get(key)
            if item is not None:
                self._items.move_to_end(key)
            return item

    def put(self, key: tuple[str, str, str], data: bytes, content_type: str) -> None:
        if len(data) > self._max:
            return
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._size -= len(old[0])
            self._items[key] = (data, content_type)
            self._size += len(data)
            while self._size > self._max:
                _, (dropped, _) = self._items.popitem(last=False)
                self._size -= len(dropped)


def _reason(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}"
    if isinstance(exc, urllib.error.URLError):
        return str(exc.reason)
    return str(exc) or exc.__class__.__name__


def _label(row: Mapping[str, Any]) -> str:
    client_id = str(row.get("client_id") or "")
    name = row.get("friendly_name") or client_id
    return name if name == client_id else f"{name} ({client_id})"


def _size(dimensions: Any) -> tuple[int | None, int | None]:
    width, sep, height = str(dimensions or "").partition("x")
    try:
        return (int(width), int(height)) if sep else (None, None)
    except ValueError:
        return None, None


def _is_local(row: Mapping[str, Any]) -> bool:
    """The server's own panel (combined mode) reaches the server over loopback."""

    address = str(row.get("address") or "").strip()
    return bool(address) and _is_loopback(address)


def register(
    app,
    *,
    client_rows: Callable[[], list[dict[str, Any]]],
    local_entries: Callable[[], list[dict[str, Any]]],
    env: Mapping[str, str] | None = None,
    clock: Callable[[], float] | None = None,
    make_ui: Callable[[str], ConfigUI] | None = None,
) -> Blueprint:
    """Add the Live page and its API to *app*.

    *client_rows* lists the displays (the Clients page's rows); *local_entries*
    lists this machine's own screenshots (the Screenshots page's entries), for
    the server's own panel.
    """

    source_env = os.environ if env is None else env
    now = clock or time.time
    blueprint = Blueprint("live", __name__)
    listings: dict[str, _Listing] = {}
    uis: dict[str, ConfigUI] = {}
    images = _ImageCache()
    lock = threading.Lock()

    def client_port() -> int:
        try:
            return int(source_env.get("SCREEN_CONFIG_PORT") or DEFAULT_CLIENT_PORT)
        except ValueError:
            return DEFAULT_CLIENT_PORT

    def ui_for(base_url: str) -> ConfigUI:
        with lock:
            ui = uis.get(base_url)
            if ui is None:
                if make_ui is not None:
                    ui = make_ui(base_url)
                else:
                    password = source_env.get("SCREEN_UI_PASSWORD") or None
                    ui = ConfigUI(base_url, lambda: password, source_env.get("SCREEN_UI_USERNAME") or "",
                                  FETCH_TIMEOUT_SECONDS)
                uis[base_url] = ui
            return ui

    def base_url(row: Mapping[str, Any]) -> str:
        return client_base_url(dict(row), "http://localhost", {}, client_port())

    def fetch_listing(row: Mapping[str, Any]) -> _Listing:
        url = base_url(row)
        try:
            payload = ui_for(url).get_json("/api/screenshots")
        except AuthRequired as exc:
            return _Listing(now(), error=str(exc))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return _Listing(now(), error=f"could not reach {url} ({_reason(exc)})")
        entries = [e for e in payload.get("screens") or [] if isinstance(e, dict) and e.get("id")]
        return _Listing(now(), entries=entries)

    def listing_for(row: Mapping[str, Any]) -> _Listing:
        client_id = str(row["client_id"])
        with lock:
            cached = listings.get(client_id)
        if cached is not None:
            ttl = FAILURE_TTL_SECONDS if cached.error else LIST_TTL_SECONDS
            if 0 <= now() - cached.fetched_at < ttl:
                return cached
        listing = fetch_listing(row)
        with lock:
            listings[client_id] = listing
        return listing

    def uploaded_screens(client_id: str) -> list[dict[str, Any]]:
        inbox: ScreenshotInbox | None = app.extensions.get("desk_display_uploaded_screenshots")
        if inbox is None:
            return []
        try:
            return inbox.screens(client_id)
        except (UploadRejected, OSError):
            return []

    def shots_from(entries: list[dict[str, Any]], client_id: str, *, local: bool) -> tuple[list[str], dict[str, dict]]:
        order: list[str] = []
        shots: dict[str, dict[str, Any]] = {}
        quoted = urllib.parse.quote(client_id, safe="")
        for entry in entries:
            screen = str(entry["id"])
            order.append(screen)
            path = entry.get("path")
            if not path:
                continue
            version = entry.get("version")
            query = urllib.parse.urlencode({"path": path, "v": version or ""})
            shots[screen] = {
                "screen": screen,
                "url": (f"/screenshots/file/{urllib.parse.quote(path)}?v={version or ''}" if local
                        else f"/api/live/{quoted}/image?{query}"),
                "captured_at": float(version) if isinstance(version, (int, float)) else None,
                "uploaded": False,
            }
        return order, shots

    def display(row: Mapping[str, Any]) -> dict[str, Any]:
        client_id = str(row["client_id"])
        width, height = _size(row.get("dimensions"))
        result: dict[str, Any] = {
            "client_id": client_id,
            "label": _label(row),
            "display_profile": row.get("display_profile"),
            "dimensions": row.get("dimensions"),
            "width": width,
            "height": height,
            "state": row.get("state"),
            "current_screen": row.get("current_screen"),
            "error": None,
            "note": None,
            "fetched_at": None,
        }
        order: list[str] = []
        shots: dict[str, dict[str, Any]] = {}
        if _is_local(row):
            try:
                order, shots = shots_from(local_entries(), client_id, local=True)
                result["fetched_at"] = now()
            except Exception as exc:  # noqa: BLE001 - one display's failure stays on its tile
                result["error"] = f"could not read this machine's screenshots ({_reason(exc)})"
        else:
            listing = listing_for(row)
            result["fetched_at"] = listing.fetched_at
            if listing.error:
                result["error"] = listing.error
            else:
                order, shots = shots_from(listing.entries, client_id, local=False)
        quoted = urllib.parse.quote(client_id, safe="")
        added = 0
        for entry in uploaded_screens(client_id):
            screen = entry.get("screen")
            name = entry.get("file")
            if not screen or not name or screen in shots:
                continue
            captured = parse_time(entry.get("captured_at"))
            if screen not in order:
                order.append(screen)
            shots[screen] = {
                "screen": screen,
                "url": f"/api/clients/{quoted}/uploaded-screenshots/{urllib.parse.quote(name)}"
                       f"?v={int(captured or 0)}",
                "captured_at": captured,
                "uploaded": True,
            }
            added += 1
        if added and result["error"]:
            result["note"] = f"{result['error']}; showing the screenshots it uploaded to the server"
            result["error"] = None
        result["screen_order"] = [s for s in order if s in shots]
        result["screens"] = shots
        return result

    @blueprint.get("/live")
    def live_page():
        return render_template("live.html", poll_seconds=POLL_SECONDS)

    @blueprint.get("/api/live")
    def live_api():
        rows = [r for r in client_rows() if r.get("client_id")]
        include_all = request.args.get("all") in {"1", "true", "yes"}
        active = [r for r in rows if include_all or r.get("state") in ACTIVE_STATES]
        skipped = [{"client_id": r["client_id"], "label": _label(r), "state": r.get("state")}
                   for r in rows if r not in active]
        if active:
            with ThreadPoolExecutor(max_workers=min(8, len(active))) as pool:
                displays = list(pool.map(display, active))
        else:
            displays = []
        order: dict[str, None] = {}
        for item in displays:
            for screen in item["screen_order"]:
                order.setdefault(screen, None)
        return jsonify({
            "now": now(),
            "poll_seconds": POLL_SECONDS,
            "screen_order": list(order),
            "displays": displays,
            "skipped": skipped,
        })

    @blueprint.get("/api/live/<client_id>/image")
    def live_image(client_id: str):
        path = request.args.get("path") or ""
        version = request.args.get("v") or ""
        with lock:
            listing = listings.get(client_id)
        # Only images the display itself listed, so this is no general proxy.
        if listing is None or not any(e.get("path") == path for e in listing.entries):
            abort(404)
        key = (client_id, path, version)
        cached = images.get(key)
        if cached is None:
            row = next((r for r in client_rows() if r.get("client_id") == client_id), None)
            if row is None:
                abort(404)
            try:
                data, content_type = ui_for(base_url(row)).get("/screenshots/file/" + urllib.parse.quote(path))
            except (urllib.error.URLError, OSError, AuthRequired) as exc:
                return jsonify({"error": "unreachable", "message": _reason(exc)}), 502
            content_type = content_type.split(";", 1)[0].strip()
            if content_type not in IMAGE_TYPES:
                return jsonify({"error": "not_an_image", "message": f"the display answered {content_type}"}), 502
            images.put(key, data, content_type)
            cached = (data, content_type)
        response = Response(cached[0], mimetype=cached[1])
        # The URL names the capture time, so a browser may keep it.
        response.headers["Cache-Control"] = "private, max-age=600" if version else "no-store"
        return response

    app.register_blueprint(blueprint)
    return blueprint


__all__ = ["register"]
