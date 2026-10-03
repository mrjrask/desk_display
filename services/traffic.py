"""Chicagoland freeway traffic from Travel Midwest's Quick Traffic report.

The ``traffic`` screen shows four fixed Edens/Kennedy segments: inbound on
every display except hyper, which shows the matching outbound four.  This
module fetches the whole report once, picks the eight segments by their
exact Travel Midwest descriptions and returns a small payload both
directions share; :func:`select` then picks one direction's four rows.

The report is ``[{"ageInMinutes": "3", "oldest": "..."}, [group, ...]]``,
each group ``{"caption", "path", "rows": [segment, ...]}`` and each segment
``{"description", "shortDescription", "ids", "id", "travelTime", "speed",
"over"}``.  A reversible-lane segment that is closed in that direction
reports a travel time of zero or less; it is kept as N/A, never hidden.

Nothing here draws, and the drawing code (:mod:`screens.draw_traffic`) makes
no network requests.
"""
from __future__ import annotations

import logging
import os
import socket
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

REPORT_URL = "https://travelmidwest.com/lmiga/chicagoQuickTraffic.json"
SCREEN_ID = "traffic"
FEED = "traffic"
REFRESH_SECONDS = 5 * 60
# Data older than this (since the last good fetch) is shown as cached.
STALE_AFTER_SECONDS = 2 * REFRESH_SECONDS
REQUEST_TIMEOUT_SECONDS = 15.0

INBOUND = "inbound"
OUTBOUND = "outbound"
DIRECTIONS = (INBOUND, OUTBOUND)
# The render scope the server gives an outbound display's traffic screen.
OUTBOUND_SCOPE = "traffic-outbound"

NORMAL = "normal"
ELEVATED = "elevated"
HEAVY = "heavy"
UNAVAILABLE = "unavailable"


class TrafficFeedError(RuntimeError):
    """The report could not be fetched or was not usable."""


@dataclass(frozen=True)
class Segment:
    key: str
    description: str
    road: str
    label: str
    direction: str
    reversible: bool = False


SEGMENTS: tuple[Segment, ...] = (
    Segment("edens_lakecook_jane_byrne",
            "Inbound Edens from Lake Cook to I-290/Jane Byrne Interchange (via Kennedy)",
            "Edens", "Lake Cook → Downtown", INBOUND),
    Segment("edens_lakecook_montrose", "Inbound Edens from Lake Cook to Montrose",
            "Edens", "Lake Cook → Montrose", INBOUND),
    Segment("kennedy_montrose_jane_byrne", "Inbound Kennedy from Montrose to I-290/Jane Byrne Interchange",
            "Kennedy", "Montrose → Downtown", INBOUND),
    Segment("kennedy_reversible_inbound", "Inbound Kennedy Reversibles from Montrose to Ohio",
            "Kennedy", "Montrose → Ohio", INBOUND, reversible=True),
    Segment("kennedy_jane_byrne_montrose", "Outbound Kennedy from I-290/Jane Byrne Interchange to Montrose",
            "Kennedy", "Downtown → Montrose", OUTBOUND),
    Segment("kennedy_reversible_outbound", "Outbound Kennedy Reversibles from Ohio to Montrose",
            "Kennedy", "Ohio → Montrose", OUTBOUND, reversible=True),
    Segment("edens_jane_byrne_lakecook",
            "Outbound Edens from I-290/Jane Byrne Interchange (via Kennedy) to Lake Cook",
            "Edens", "Downtown → Lake Cook", OUTBOUND),
    Segment("edens_montrose_lakecook", "Outbound Edens from Montrose to Lake Cook",
            "Edens", "Montrose → Lake Cook", OUTBOUND),
)
SEGMENTS_BY_KEY = {segment.key: segment for segment in SEGMENTS}
SEGMENTS_BY_DESCRIPTION = {segment.description: segment for segment in SEGMENTS}
# Rows each direction shows, in screen order: inbound is Edens then Kennedy,
# outbound is Kennedy then Edens (reversible lane first within each road).
DIRECTION_KEYS: Mapping[str, tuple[str, ...]] = {
    INBOUND: ("edens_lakecook_montrose", "edens_lakecook_jane_byrne",
              "kennedy_reversible_inbound", "kennedy_montrose_jane_byrne"),
    OUTBOUND: ("kennedy_reversible_outbound", "kennedy_jane_byrne_montrose",
               "edens_jane_byrne_lakecook", "edens_montrose_lakecook"),
}
ROAD_ROUTES = {"Edens": "I-94", "Kennedy": "I-90/94"}
# Detail data's congestion levels (optional; the Quick Traffic report has none).
_ELEVATED_CONGESTION = {"light", "medium", "moderate"}
_HEAVY_CONGESTION = {"heavy", "severe"}


# ── Parsing ─────────────────────────────────────────────────────────────────


def _minutes(value: Any) -> Optional[int]:
    """A positive whole number, or None (N/A: missing, closed, or zero)."""

    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number <= 0:  # NaN, closed or no reading
        return None
    return int(round(number))


def _segment_payload(segment: Segment, row: Mapping[str, Any]) -> dict[str, Any]:
    ids = row.get("ids")
    ids = [str(item) for item in ids] if isinstance(ids, list) else []
    primary = row.get("id")
    return {
        "key": segment.key,
        "description": segment.description,
        "short_description": str(row.get("shortDescription") or "") or None,
        "travel_time": _minutes(row.get("travelTime")),
        "speed": _minutes(row.get("speed")),
        "over": row.get("over") is True,
        "ids": ids,
        "id": str(primary) if primary not in (None, "") else None,
        "congestion": None,
    }


def parse_report(raw: Any, *, fetched_at: Optional[float] = None) -> dict[str, Any]:
    """Normalise a Quick Traffic report into the ``traffic`` feed payload.

    Raises :class:`TrafficFeedError` for a payload that is not a report or
    holds none of the eight segments.  Segments missing from an otherwise
    good report are kept as N/A rows.
    """

    if not isinstance(raw, list) or len(raw) < 2 or not isinstance(raw[0], Mapping):
        raise TrafficFeedError("Travel Midwest report is not [metadata, groups]")
    meta, groups = raw[0], raw[1]
    if not isinstance(groups, list):
        raise TrafficFeedError("Travel Midwest report has no segment groups")
    found: dict[str, dict[str, Any]] = {}
    for group in groups:
        rows = group.get("rows") if isinstance(group, Mapping) else None
        for row in rows if isinstance(rows, list) else ():
            if not isinstance(row, Mapping):
                continue
            segment = SEGMENTS_BY_DESCRIPTION.get(str(row.get("description") or "").strip())
            if segment is not None and segment.key not in found:
                found[segment.key] = _segment_payload(segment, row)
    if not found:
        raise TrafficFeedError("Travel Midwest report has none of the traffic screen's segments")
    age = meta.get("ageInMinutes")
    try:
        age_minutes = max(0, int(float(age))) if age not in (None, "") else None
    except (TypeError, ValueError):
        age_minutes = None
    segments = {}
    for segment in SEGMENTS:
        segments[segment.key] = found.get(segment.key) or {
            "key": segment.key, "description": segment.description, "short_description": None,
            "travel_time": None, "speed": None, "over": False, "ids": [], "id": None, "congestion": None,
        }
    return {
        "segments": segments,
        "missing": [segment.key for segment in SEGMENTS if segment.key not in found],
        "age_minutes": age_minutes,
        "oldest": str(meta.get("oldest") or "") or None,
        "fetched_at": time.time() if fetched_at is None else float(fetched_at),
    }


def valid_payload(value: Any) -> bool:
    """Whether *value* is a payload :func:`parse_report` produced."""

    return (isinstance(value, Mapping) and isinstance(value.get("segments"), Mapping)
            and bool(value["segments"]))


# ── Fetching with a shared cache ────────────────────────────────────────────

_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {}


def _download(url: str = REPORT_URL) -> Any:
    from services.http_client import get_session, http_get

    response = http_get(url, timeout=REQUEST_TIMEOUT_SECONDS, session=get_session("traffic"))
    response.raise_for_status()
    return response.json()


def fetch_report(*, force: bool = False, download=None, clock=time.monotonic) -> dict[str, Any]:
    """The traffic payload, from one shared cache refreshed every five minutes.

    Raises :class:`TrafficFeedError` when the refresh fails or the report
    is unusable; the last good payload stays cached either way.  See
    :func:`get_report` for a caller that wants the cached copy instead.
    """

    with _CACHE_LOCK:
        cached, fetched = _CACHE.get("payload"), _CACHE.get("fetched_monotonic")
    if not force and cached is not None and fetched is not None and clock() - fetched < REFRESH_SECONDS:
        return cached
    try:
        raw = (download or _download)()
    except Exception as exc:  # noqa: BLE001 - any network or JSON failure
        raise TrafficFeedError(f"Travel Midwest request failed: {exc}") from exc
    try:
        payload = parse_report(raw)
    except TrafficFeedError as exc:
        logging.warning("Rejected Travel Midwest traffic report: %s", exc)
        raise
    with _CACHE_LOCK:
        _CACHE["payload"] = payload
        _CACHE["fetched_monotonic"] = clock()
    return payload


def get_report(*, force: bool = False, download=None, clock=time.monotonic) -> Optional[dict[str, Any]]:
    """Like :func:`fetch_report`, but a failure returns the last good payload (or None)."""

    try:
        return fetch_report(force=force, download=download, clock=clock)
    except TrafficFeedError as exc:
        logging.warning("Traffic refresh failed; showing cached data if any: %s", exc)
        with _CACHE_LOCK:
            return _CACHE.get("payload")


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


# ── Direction ───────────────────────────────────────────────────────────────


def _named(client_id: str, names: Sequence[str]) -> bool:
    lowered = client_id.strip().lower()
    return any(lowered == name or lowered.startswith(f"{name}-") for name in names)


def outbound_displays(env: Optional[Mapping[str, str]] = None) -> tuple[str, ...]:
    """Display IDs that show outbound traffic (``TRAFFIC_OUTBOUND_DISPLAYS``, default hyper)."""

    raw = (os.environ if env is None else env).get("TRAFFIC_OUTBOUND_DISPLAYS")
    if raw is None:
        raw = "hyper"
    return tuple(part.strip().lower() for part in raw.split(",") if part.strip())


def direction_for_display(client_id: str, env: Optional[Mapping[str, str]] = None) -> str:
    """Outbound for hyper (``hyper`` or ``hyper-…``), inbound for every other display."""

    return OUTBOUND if _named(client_id, outbound_displays(env)) else INBOUND


def direction_for_scope(scope: Optional[str]) -> str:
    """The direction a server render key's scope stands for."""

    return OUTBOUND if scope == OUTBOUND_SCOPE else INBOUND


def screen_scopes(client_id: str, screens, env: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """``{"traffic": OUTBOUND_SCOPE}`` for an outbound display that plays the screen."""

    if SCREEN_ID in set(screens) and direction_for_display(client_id, env) == OUTBOUND:
        return {SCREEN_ID: OUTBOUND_SCOPE}
    return {}


def local_direction(env: Optional[Mapping[str, str]] = None, hostname: Optional[str] = None) -> str:
    """A standalone display's direction: ``TRAFFIC_DIRECTION``, else by host name."""

    env = os.environ if env is None else env
    chosen = (env.get("TRAFFIC_DIRECTION") or "auto").strip().lower()
    if chosen in DIRECTIONS:
        return chosen
    return direction_for_display(hostname if hostname is not None else socket.gethostname(), env)


# ── Selection and presentation state ────────────────────────────────────────


def status_of(row: Mapping[str, Any]) -> str:
    """Normal, elevated, heavy or unavailable, led by Travel Midwest's ``over`` flag."""

    if row.get("travel_time") is None:
        return UNAVAILABLE
    congestion = str(row.get("congestion") or "").strip().lower()
    if row.get("over") or congestion in _HEAVY_CONGESTION:
        return HEAVY
    if congestion in _ELEVATED_CONGESTION:
        return ELEVATED
    return NORMAL


def select(payload: Any, direction: str, *, now: Optional[float] = None) -> Optional[dict[str, Any]]:
    """The four rows *direction* shows, grouped by road, plus freshness; None without data."""

    if not valid_payload(payload):
        return None
    direction = direction if direction in DIRECTIONS else INBOUND
    now = time.time() if now is None else now
    groups: list[dict[str, Any]] = []
    for key in DIRECTION_KEYS[direction]:
        segment = SEGMENTS_BY_KEY[key]
        row = dict(payload["segments"].get(key) or {})
        entry = {
            "key": key,
            "label": segment.label,
            "reversible": segment.reversible,
            "travel_time": row.get("travel_time"),
            "speed": row.get("speed"),
            "over": bool(row.get("over")),
            "status": status_of(row),
        }
        if not groups or groups[-1]["road"] != segment.road:
            groups.append({"road": segment.road, "route": ROAD_ROUTES.get(segment.road, ""), "rows": []})
        groups[-1]["rows"].append(entry)
    fetched_at = payload.get("fetched_at")
    since_fetch = max(0.0, now - float(fetched_at)) if isinstance(fetched_at, (int, float)) else 0.0
    report_age = payload.get("age_minutes")
    age = None if report_age is None else int(report_age) + int(since_fetch // 60)
    return {
        "direction": direction,
        "groups": groups,
        "age_minutes": age,
        "stale": since_fetch > STALE_AFTER_SECONDS,
    }


__all__ = [
    "DIRECTIONS", "DIRECTION_KEYS", "ELEVATED", "FEED", "HEAVY", "INBOUND", "NORMAL", "OUTBOUND",
    "OUTBOUND_SCOPE", "REFRESH_SECONDS", "REPORT_URL", "SCREEN_ID", "SEGMENTS", "STALE_AFTER_SECONDS",
    "Segment", "TrafficFeedError", "UNAVAILABLE", "clear_cache", "direction_for_display",
    "direction_for_scope", "fetch_report", "get_report", "local_direction", "outbound_displays",
    "parse_report", "screen_scopes", "select", "status_of", "valid_payload",
]
