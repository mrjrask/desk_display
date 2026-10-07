"""CPU temperatures of the LAN's Raspberry Pis from MMM-RemoteTempMonitor.

The ``remote temp monitor`` screen shows the same table as the
MMM-RemoteTempMonitor MagicMirror module
(https://github.com/mrjrask/MMM-RemoteTempMonitor).  That module's node
helper collects every device's UDP broadcasts and republishes them as one
JSON snapshot on its aggregate port::

    http://<MagicMirror host>:9877/temps

``{"type": "temperature_snapshot", "count", "updatedAt", "devices": [...]}``,
each device ``{"deviceId", "hostname", "celsius", "fahrenheit", "pi_model",
"pi_ram", "platform", "cpu_arch", "lastSeen", "ip", ...}`` (plus aliases
such as ``temp_c`` and ``temperature: {"celsius", "fahrenheit"}``).  The
module itself drops a device it has not heard from for 30 seconds, so the
snapshot is already the list the MagicMirror shows.

This module fetches and normalises that snapshot and applies the module's
sort order and colour thresholds; nothing here draws, and the drawing code
(:mod:`screens.draw_remote_temps`) makes no network requests.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections.abc import Mapping
from typing import Any, Optional

SCREEN_ID = "remote temp monitor"
FEED = "remote_temps"
DEFAULT_HOST = "192.168.1.201"
DEFAULT_PORT = 9877
ENDPOINT_PATH = "/temps"
REFRESH_SECONDS = 60
# Data older than this (since the last good fetch) is shown as cached.
STALE_AFTER_SECONDS = 3 * REFRESH_SECONDS
REQUEST_TIMEOUT_SECONDS = 5.0

# MMM-RemoteTempMonitor's default tempThresholds (°C) and the classes its
# getTempColorClass() picks from them.  Like the module, a reading is
# "normal" (green) below the warm threshold.
THRESHOLDS = {"warm": 60.0, "hot": 70.0, "very_hot": 80.0, "critical": 85.0}
NORMAL = "normal"
WARM = "warm"
HOT = "hot"
VERY_HOT = "very_hot"
CRITICAL = "critical"
LEVELS = (NORMAL, WARM, HOT, VERY_HOT, CRITICAL)


class RemoteTempsError(RuntimeError):
    """The snapshot could not be fetched or was not usable."""


# ── Endpoint ────────────────────────────────────────────────────────────────


def endpoint_url(env: Optional[Mapping[str, str]] = None) -> str:
    """``http://REMOTE_TEMP_MONITOR_HOST:REMOTE_TEMP_MONITOR_PORT/temps``."""

    env = os.environ if env is None else env
    host = (env.get("REMOTE_TEMP_MONITOR_HOST") or "").strip() or DEFAULT_HOST
    port_text = (env.get("REMOTE_TEMP_MONITOR_PORT") or "").strip()
    try:
        port = int(port_text) if port_text else DEFAULT_PORT
    except ValueError:
        logging.warning("REMOTE_TEMP_MONITOR_PORT=%r is not a port; using %d", port_text, DEFAULT_PORT)
        port = DEFAULT_PORT
    if not 0 < port < 65536:
        port = DEFAULT_PORT
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"  # a bare IPv6 address
    return f"http://{host}:{port}{ENDPOINT_PATH}"


# ── Parsing ─────────────────────────────────────────────────────────────────


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any) -> Optional[str]:
    text = str(value).strip() if value is not None else ""
    return text or None


def _device(raw: Any) -> Optional[dict[str, Any]]:
    if not isinstance(raw, Mapping):
        return None
    hostname = _text(raw.get("hostname")) or _text(raw.get("name"))
    nested = raw.get("temperature") if isinstance(raw.get("temperature"), Mapping) else {}
    celsius = next((value for value in (_number(raw.get("celsius")), _number(raw.get("temp_c")),
                                        _number(raw.get("temperature_c")), _number(nested.get("celsius")))
                    if value is not None), None)
    if hostname is None or celsius is None or not -50 <= celsius <= 150:
        return None
    fahrenheit = next((value for value in (_number(raw.get("fahrenheit")), _number(raw.get("temp_f")),
                                           _number(raw.get("temperature_f")), _number(nested.get("fahrenheit")))
                       if value is not None), None)
    if fahrenheit is None:
        fahrenheit = celsius * 9 / 5 + 32
    return {
        "id": _text(raw.get("deviceId")) or _text(raw.get("device_id")) or _text(raw.get("id")) or hostname,
        "hostname": hostname,
        "celsius": celsius,
        "fahrenheit": fahrenheit,
        "pi_model": _text(raw.get("pi_model")),
        "pi_ram": _text(raw.get("pi_ram")),
        "ip": _text(raw.get("ip")),
    }


def parse_snapshot(raw: Any, *, fetched_at: Optional[float] = None) -> dict[str, Any]:
    """Normalise an aggregate ``/temps`` snapshot into the ``remote_temps`` feed payload.

    Raises :class:`RemoteTempsError` for a payload that is not a snapshot.
    An empty device list is a good snapshot: the module has not heard from
    any device yet ("No temperature monitors found").
    """

    if not isinstance(raw, Mapping):
        raise RemoteTempsError("temperature snapshot is not a JSON object")
    devices = next((raw.get(key) for key in ("devices", "temps", "temperatures")
                    if isinstance(raw.get(key), list)), None)
    if devices is None:
        raise RemoteTempsError("temperature snapshot has no device list")
    parsed = [device for device in (_device(item) for item in devices) if device is not None]
    return {
        "devices": parsed,
        "updated_at": _text(raw.get("updatedAt")),
        "fetched_at": time.time() if fetched_at is None else float(fetched_at),
    }


def valid_payload(value: Any) -> bool:
    """Whether *value* is a payload :func:`parse_snapshot` produced."""

    return isinstance(value, Mapping) and isinstance(value.get("devices"), list)


# ── Fetching with a shared cache ────────────────────────────────────────────

_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {}


def _download(url: Optional[str] = None) -> Any:
    from services.http_client import get_session, http_get

    response = http_get(url or endpoint_url(), timeout=REQUEST_TIMEOUT_SECONDS, session=get_session("remote_temps"))
    response.raise_for_status()
    return response.json()


def fetch_snapshot(*, force: bool = False, download=None, clock=time.monotonic) -> dict[str, Any]:
    """The temperatures payload, from one shared cache refreshed every minute.

    Raises :class:`RemoteTempsError` when the refresh fails or the snapshot
    is unusable; the last good payload stays cached either way.
    """

    with _CACHE_LOCK:
        cached, fetched = _CACHE.get("payload"), _CACHE.get("fetched_monotonic")
    if not force and cached is not None and fetched is not None and clock() - fetched < REFRESH_SECONDS:
        return cached
    try:
        raw = (download or _download)()
    except Exception as exc:  # any network or JSON failure
        raise RemoteTempsError(f"temperature snapshot request failed: {exc}") from exc
    try:
        payload = parse_snapshot(raw)
    except RemoteTempsError as exc:
        logging.warning("Rejected remote temperature snapshot: %s", exc)
        raise
    with _CACHE_LOCK:
        _CACHE["payload"] = payload
        _CACHE["fetched_monotonic"] = clock()
    return payload


def get_snapshot(*, force: bool = False, download=None, clock=time.monotonic) -> Optional[dict[str, Any]]:
    """Like :func:`fetch_snapshot`, but a failure returns the last good payload (or None)."""

    try:
        return fetch_snapshot(force=force, download=download, clock=clock)
    except RemoteTempsError as exc:
        logging.warning("Remote temperature refresh failed; showing cached data if any: %s", exc)
        with _CACHE_LOCK:
            return _CACHE.get("payload")


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


# ── Presentation ────────────────────────────────────────────────────────────


def level_of(celsius: float) -> str:
    """The module's colour class for a reading (``getTempColorClass``)."""

    if celsius >= THRESHOLDS["critical"]:
        return CRITICAL
    if celsius >= THRESHOLDS["very_hot"]:
        return VERY_HOT
    if celsius >= THRESHOLDS["hot"]:
        return HOT
    if celsius >= THRESHOLDS["warm"]:
        return WARM
    return NORMAL


def display_name(device: Mapping[str, Any]) -> str:
    """``hostname (model | RAM)``, as the module labels a row."""

    name = str(device.get("hostname") or "")
    model, ram = device.get("pi_model"), device.get("pi_ram")
    if model and ram:
        return f"{name} ({model} | {ram})"
    if model or ram:
        return f"{name} ({model or ram})"
    return name


def select(payload: Any, *, now: Optional[float] = None) -> Optional[dict[str, Any]]:
    """The rows to draw, hottest first (the module's default sort), plus freshness; None without data."""

    if not valid_payload(payload):
        return None
    now = time.time() if now is None else now
    rows = []
    for device in sorted(payload["devices"], key=lambda item: -float(item["celsius"])):
        rows.append({
            "name": display_name(device),
            "hostname": device["hostname"],
            "celsius": float(device["celsius"]),
            "fahrenheit": float(device["fahrenheit"]),
            "level": level_of(float(device["celsius"])),
        })
    fetched_at = payload.get("fetched_at")
    since_fetch = max(0.0, now - float(fetched_at)) if isinstance(fetched_at, (int, float)) else 0.0
    return {
        "rows": rows,
        "age_minutes": int(since_fetch // 60),
        "stale": since_fetch > STALE_AFTER_SECONDS,
    }


__all__ = [
    "CRITICAL", "DEFAULT_HOST", "DEFAULT_PORT", "ENDPOINT_PATH", "FEED", "HOT", "LEVELS", "NORMAL",
    "REFRESH_SECONDS", "RemoteTempsError", "SCREEN_ID", "STALE_AFTER_SECONDS", "THRESHOLDS", "VERY_HOT",
    "WARM", "clear_cache", "display_name", "endpoint_url", "fetch_snapshot", "get_snapshot", "level_of",
    "parse_snapshot", "select", "valid_payload",
]
