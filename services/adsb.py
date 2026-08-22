"""ADS-B receiver polling, local SQLite storage, and daily stat rollups.

This module is intentionally split from the display code. The standalone
collector (``scripts/adsb_collector.py``) is the only thing that writes to
the SQLite database; the "adsb stats" screen (``screens/draw_adsb_stats.py``)
only reads it. A receiver outage or slow poll therefore never blocks or
slows down screen rendering, and the display process never makes a network
call to a receiver.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import math
import re
import sqlite3
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

from paths import resolve_cache_file_path
from services import aircraft_type_db
from services.http_client import http_get

EARTH_RADIUS_NM = 3440.065
EARTH_RADIUS_MI = 3958.8

# dump1090-fa's web alias has varied across FlightAware PiAware image
# versions ("skyaware" is the current branding; "dump1090-fa" is the older
# alias, still present on many installs) and plain dump1090 serves straight
# from the web root. Try each in order and remember whichever one answers
# for a given host so steady-state polling only makes one request.
_AIRCRAFT_JSON_PATHS: tuple[str, ...] = (
    "/dump1090-fa/data/aircraft.json",
    "/skyaware/data/aircraft.json",
    "/data/aircraft.json",
)
_STATS_JSON_PATHS: tuple[str, ...] = (
    "/dump1090-fa/data/stats.json",
    "/skyaware/data/stats.json",
    "/data/stats.json",
)
_working_path_index: dict[str, int] = {}

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sightings (
    day TEXT NOT NULL,
    hex TEXT NOT NULL,
    device TEXT NOT NULL,
    callsign TEXT,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    max_distance_nm REAL,
    max_distance_at REAL,
    max_altitude_ft INTEGER,
    PRIMARY KEY (day, hex, device)
);
CREATE INDEX IF NOT EXISTS idx_sightings_day ON sightings(day);

CREATE TABLE IF NOT EXISTS device_status (
    device TEXT PRIMARY KEY,
    last_poll_at REAL,
    online INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    tracked_hexes TEXT,
    messages_total INTEGER,
    tracked_types TEXT,
    tracked_callsigns TEXT
);

CREATE TABLE IF NOT EXISTS message_baseline (
    day TEXT NOT NULL,
    device TEXT NOT NULL,
    baseline INTEGER NOT NULL,
    PRIMARY KEY (day, device)
);

CREATE TABLE IF NOT EXISTS all_time_furthest (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    hex TEXT,
    callsign TEXT,
    device TEXT,
    distance_nm REAL,
    seen_at REAL
);
"""


@dataclass(frozen=True)
class AdsbDevice:
    """A configured dump1090-fa receiver endpoint."""

    host: str
    label: str


@dataclass(frozen=True)
class AircraftSighting:
    """One aircraft observed in a single poll of a receiver's aircraft.json."""

    hex: str
    callsign: Optional[str]
    distance_nm: Optional[float]
    altitude_ft: Optional[int]
    aircraft_type: Optional[str] = None


@dataclass(frozen=True)
class PollResult:
    """Outcome of polling one receiver for one cycle."""

    device: AdsbDevice
    ok: bool
    error: Optional[str]
    sightings: tuple[AircraftSighting, ...] = ()
    messages_total: Optional[int] = None


@dataclass(frozen=True)
class FurthestCatch:
    hex: str
    callsign: Optional[str]
    device: str
    distance_nm: float
    seen_at: float


@dataclass(frozen=True)
class DailyStats:
    """Rolled-up stats for a single local day, computed from raw sightings."""

    day: str
    total_combined: int
    total_by_device: dict[str, int]
    furthest: Optional[FurthestCatch]
    busiest_hour_combined: Optional[tuple[int, int]]
    busiest_hour_by_device: dict[str, tuple[int, int]]
    hourly_counts_combined: dict[int, int]
    highest_altitude_ft: Optional[int]
    messages_today_by_device: dict[str, int]
    currently_tracked_combined: int
    currently_tracked_by_device: dict[str, int]
    device_online: dict[str, bool]
    all_time_furthest: Optional[FurthestCatch]
    device_errors: dict[str, str] = field(default_factory=dict)
    currently_tracked_by_model: dict[str, int] = field(default_factory=dict)
    currently_tracked_by_airline: dict[str, int] = field(default_factory=dict)


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float, *, unit: str = "nm"
) -> float:
    """Great-circle distance between two lat/lon points in nautical miles or miles."""

    radius = EARTH_RADIUS_MI if unit == "mi" else EARTH_RADIUS_NM
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))


_AIRLINE_CALLSIGN_RE = re.compile(r"^[A-Z]{2,3}(?=\d)")


def _airline_code(callsign: Optional[str]) -> str:
    """Best-effort ICAO airline code from a flight callsign (e.g. "UAL123"
    -> "UAL"). Falls back to "Other" for tail numbers (e.g. "N12345") and
    anything else that doesn't look like a scheduled-flight callsign."""

    if not callsign:
        return "Other"
    match = _AIRLINE_CALLSIGN_RE.match(callsign.strip().upper())
    return match.group(0) if match else "Other"


def _parse_altitude(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, str):
        if raw.strip().lower() == "ground":
            return 0
        try:
            return int(float(raw))
        except ValueError:
            return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _parse_aircraft(
    raw_list: Any,
    *,
    home_lat: Optional[float],
    home_lon: Optional[float],
    unit: str,
) -> list[AircraftSighting]:
    sightings: list[AircraftSighting] = []
    if not isinstance(raw_list, list):
        return sightings

    for entry in raw_list:
        if not isinstance(entry, dict):
            continue
        hex_id = entry.get("hex")
        if not hex_id:
            continue

        callsign_raw = entry.get("flight")
        callsign = callsign_raw.strip() if isinstance(callsign_raw, str) else None
        callsign = callsign or None

        altitude = _parse_altitude(entry.get("alt_baro", entry.get("alt_geom")))

        type_raw = entry.get("t")
        aircraft_type = type_raw.strip() if isinstance(type_raw, str) else None
        aircraft_type = aircraft_type or None

        distance = None
        lat, lon = entry.get("lat"), entry.get("lon")
        if (
            home_lat is not None
            and home_lon is not None
            and isinstance(lat, (int, float))
            and isinstance(lon, (int, float))
        ):
            distance = haversine_distance(home_lat, home_lon, lat, lon, unit=unit)

        sightings.append(
            AircraftSighting(
                hex=str(hex_id).strip().lower(),
                callsign=callsign,
                distance_nm=distance,
                altitude_ft=altitude,
                aircraft_type=aircraft_type,
            )
        )
    return sightings


def _fill_missing_types(sightings: list[AircraftSighting]) -> list[AircraftSighting]:
    """Fall back to the local aircraft type database for sightings whose
    aircraft.json entry didn't carry a ``"t"`` field (see
    ``services/aircraft_type_db.py`` -- most receivers never populate it)."""

    filled = []
    for sighting in sightings:
        if sighting.aircraft_type is None:
            looked_up = aircraft_type_db.lookup(sighting.hex)
            if looked_up is not None:
                sighting = replace(sighting, aircraft_type=looked_up)
        filled.append(sighting)
    return filled


def _extract_messages_total(stats_payload: Any) -> Optional[int]:
    if not isinstance(stats_payload, dict):
        return None
    total_section = stats_payload.get("total")
    messages = total_section.get("messages") if isinstance(total_section, dict) else None
    if messages is None:
        messages = stats_payload.get("messages")
    try:
        return int(messages) if messages is not None else None
    except (TypeError, ValueError):
        return None


def _short_error(exc: BaseException) -> str:
    """Condense a (often very verbose, nested) requests/urllib3 exception.

    The OS-level reason (``Connection refused``, ``timed out``, ``Name or
    service not known``, ...) is normally at the very end of the chained
    ``str()`` output, so keep the tail rather than the front.
    """

    text = str(exc)
    if len(text) > 120:
        text = f"…{text[-117:]}"
    return f"{exc.__class__.__name__}: {text}"


def _get_json(path: str, url: str, *, timeout: float) -> tuple[Any, Optional[str]]:
    """Fetch JSON from ``url``, returning (payload, error_detail keyed by ``path``)."""

    try:
        response = http_get(url, timeout=timeout)
    except Exception as exc:  # connection refused, DNS failure, timeout, etc.
        return None, f"{path}: {_short_error(exc)}"
    if response.status_code != 200:
        return None, f"{path}: HTTP {response.status_code}"
    try:
        return response.json(), None
    except ValueError as exc:
        return None, f"{path}: invalid JSON ({exc})"


def _fetch_aircraft_json(
    base: str, host_key: str, *, timeout: float
) -> tuple[Any, Optional[int], list[str]]:
    """Try each known aircraft.json path, preferring the last one that worked.

    Returns (payload, path_index_used, errors_from_failed_attempts).
    """

    cached_index = _working_path_index.get(host_key)
    order = list(range(len(_AIRCRAFT_JSON_PATHS)))
    if cached_index is not None:
        order.remove(cached_index)
        order.insert(0, cached_index)

    errors: list[str] = []
    for index in order:
        path = _AIRCRAFT_JSON_PATHS[index]
        payload, error = _get_json(path, f"{base}{path}", timeout=timeout)
        if isinstance(payload, dict):
            return payload, index, errors
        if error:
            errors.append(error)
    return None, None, errors


def poll_device(
    device: AdsbDevice,
    *,
    home_lat: Optional[float],
    home_lon: Optional[float],
    unit: str = "nm",
    timeout: float = 5.0,
) -> PollResult:
    """Poll one receiver's aircraft.json (and best-effort stats.json).

    dump1090-fa's web path has varied across PiAware image versions
    (``/dump1090-fa/data/...`` vs. ``/skyaware/data/...`` vs. plain
    ``/data/...``), so every known variant is tried; whichever one answers
    is cached per host so later polls make a single request.
    """

    base = device.host.strip().rstrip("/")
    if not base.startswith(("http://", "https://")):
        base = f"http://{base}"

    aircraft_payload, path_index, errors = _fetch_aircraft_json(base, device.host, timeout=timeout)
    if not isinstance(aircraft_payload, dict):
        detail = errors[-1] if errors else "no response"
        if len(errors) > 1:
            detail += f" (tried {len(errors)} known paths)"
        return PollResult(device=device, ok=False, error=f"aircraft.json unreachable: {detail}")
    _working_path_index[device.host] = path_index

    sightings = tuple(
        _fill_missing_types(
            _parse_aircraft(
                aircraft_payload.get("aircraft"), home_lat=home_lat, home_lon=home_lon, unit=unit
            )
        )
    )
    stats_path = _STATS_JSON_PATHS[path_index]
    stats_payload, _stats_error = _get_json(stats_path, f"{base}{stats_path}", timeout=timeout)
    return PollResult(
        device=device,
        ok=True,
        error=None,
        sightings=sightings,
        messages_total=_extract_messages_total(stats_payload),
    )


class AdsbStore:
    """SQLite-backed storage for raw sightings and derived daily stats."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(resolve_cache_file_path("ADSB_DB_PATH", "adsb_stats.db"))
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        with contextlib.suppress(sqlite3.Error):
            self._conn.execute("PRAGMA journal_mode=WAL")
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA_SQL)
            self._ensure_column("device_status", "tracked_types", "TEXT")
            self._ensure_column("device_status", "tracked_callsigns", "TEXT")

    def _ensure_column(self, table: str, column: str, coltype: str) -> None:
        """Add *column* to *table* if it's missing (for DBs created by an
        older schema version — ``CREATE TABLE IF NOT EXISTS`` doesn't alter
        existing tables)."""

        existing = {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def record_poll(self, result: PollResult, *, now: float, day: str) -> None:
        """Persist one poll cycle's outcome for a single device."""

        device_label = result.device.label
        with self._lock, self._conn:
            if not result.ok:
                self._conn.execute(
                    """
                    INSERT INTO device_status (device, last_poll_at, online, error)
                    VALUES (?, ?, 0, ?)
                    ON CONFLICT(device) DO UPDATE SET
                        last_poll_at = excluded.last_poll_at,
                        online = 0,
                        error = excluded.error
                    """,
                    (device_label, now, result.error),
                )
                return

            for sighting in result.sightings:
                distance_at = now if sighting.distance_nm is not None else None
                self._conn.execute(
                    """
                    INSERT INTO sightings (
                        day, hex, device, callsign, first_seen, last_seen,
                        max_distance_nm, max_distance_at, max_altitude_ft
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(day, hex, device) DO UPDATE SET
                        last_seen = excluded.last_seen,
                        callsign = COALESCE(excluded.callsign, sightings.callsign),
                        max_distance_nm = CASE
                            WHEN excluded.max_distance_nm IS NOT NULL AND (
                                sightings.max_distance_nm IS NULL
                                OR excluded.max_distance_nm > sightings.max_distance_nm
                            ) THEN excluded.max_distance_nm
                            ELSE sightings.max_distance_nm
                        END,
                        max_distance_at = CASE
                            WHEN excluded.max_distance_nm IS NOT NULL AND (
                                sightings.max_distance_nm IS NULL
                                OR excluded.max_distance_nm > sightings.max_distance_nm
                            ) THEN excluded.max_distance_at
                            ELSE sightings.max_distance_at
                        END,
                        max_altitude_ft = CASE
                            WHEN excluded.max_altitude_ft IS NOT NULL AND (
                                sightings.max_altitude_ft IS NULL
                                OR excluded.max_altitude_ft > sightings.max_altitude_ft
                            ) THEN excluded.max_altitude_ft
                            ELSE sightings.max_altitude_ft
                        END
                    """,
                    (
                        day,
                        sighting.hex,
                        device_label,
                        sighting.callsign,
                        now,
                        now,
                        sighting.distance_nm,
                        distance_at,
                        sighting.altitude_ft,
                    ),
                )

            furthest_this_poll = max(
                (s for s in result.sightings if s.distance_nm is not None),
                key=lambda s: s.distance_nm,
                default=None,
            )
            if furthest_this_poll is not None:
                # sighting.callsign is only what this single poll saw; the
                # sightings row just upserted above may already carry a
                # callsign learned from an earlier or later poll today
                # (COALESCE'd in). Prefer that so an all-time record doesn't
                # permanently lock in a blank flight number just because the
                # farthest-distance poll happened to arrive before the
                # aircraft's identification message.
                resolved_row = self._conn.execute(
                    "SELECT callsign FROM sightings WHERE day = ? AND hex = ? AND device = ?",
                    (day, furthest_this_poll.hex, device_label),
                ).fetchone()
                resolved_callsign = (
                    resolved_row[0] if resolved_row and resolved_row[0] else furthest_this_poll.callsign
                )
                self._maybe_update_all_time(
                    furthest_this_poll, device_label, now, callsign=resolved_callsign
                )

            tracked_hexes = ",".join(sorted({s.hex for s in result.sightings}))
            tracked_types = ",".join(
                sorted(f"{s.hex}:{s.aircraft_type}" for s in result.sightings if s.aircraft_type)
            )
            tracked_callsigns = ",".join(
                sorted(f"{s.hex}:{s.callsign}" for s in result.sightings if s.callsign)
            )
            self._conn.execute(
                """
                INSERT INTO device_status (
                    device, last_poll_at, online, error, tracked_hexes, messages_total,
                    tracked_types, tracked_callsigns
                ) VALUES (?, ?, 1, NULL, ?, ?, ?, ?)
                ON CONFLICT(device) DO UPDATE SET
                    last_poll_at = excluded.last_poll_at,
                    online = 1,
                    error = NULL,
                    tracked_hexes = excluded.tracked_hexes,
                    messages_total = COALESCE(excluded.messages_total, device_status.messages_total),
                    tracked_types = excluded.tracked_types,
                    tracked_callsigns = excluded.tracked_callsigns
                """,
                (
                    device_label,
                    now,
                    tracked_hexes,
                    result.messages_total,
                    tracked_types,
                    tracked_callsigns,
                ),
            )

            if result.messages_total is not None:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO message_baseline (day, device, baseline)
                    VALUES (?, ?, ?)
                    """,
                    (day, device_label, result.messages_total),
                )

    def _maybe_update_all_time(
        self,
        sighting: AircraftSighting,
        device_label: str,
        now: float,
        *,
        callsign: Optional[str] = None,
    ) -> None:
        row = self._conn.execute(
            "SELECT distance_nm FROM all_time_furthest WHERE id = 1"
        ).fetchone()
        current_best = row[0] if row else None
        if current_best is not None and sighting.distance_nm <= current_best:
            return
        self._conn.execute(
            """
            INSERT INTO all_time_furthest (id, hex, callsign, device, distance_nm, seen_at)
            VALUES (1, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                hex = excluded.hex,
                callsign = excluded.callsign,
                device = excluded.device,
                distance_nm = excluded.distance_nm,
                seen_at = excluded.seen_at
            """,
            (sighting.hex, callsign if callsign is not None else sighting.callsign, device_label, sighting.distance_nm, now),
        )

    def prune(self, retention_days: int, *, now_day: dt.date) -> int:
        """Delete raw sightings/baselines older than ``retention_days``."""

        cutoff = (now_day - dt.timedelta(days=retention_days)).isoformat()
        with self._lock, self._conn:
            cursor = self._conn.execute("DELETE FROM sightings WHERE day < ?", (cutoff,))
            self._conn.execute("DELETE FROM message_baseline WHERE day < ?", (cutoff,))
            return cursor.rowcount

    def compute_daily_stats(self, *, day: str, tz: dt.tzinfo) -> DailyStats:
        """Roll raw sightings up into the metrics the display screen shows."""

        with self._lock:
            sighting_rows = self._conn.execute(
                """
                SELECT hex, device, callsign, first_seen, max_distance_nm,
                       max_distance_at, max_altitude_ft
                FROM sightings WHERE day = ?
                """,
                (day,),
            ).fetchall()
            device_rows = self._conn.execute(
                """
                SELECT device, online, error, tracked_hexes, messages_total, tracked_types,
                       tracked_callsigns
                FROM device_status
                """
            ).fetchall()
            baseline_rows = self._conn.execute(
                "SELECT device, baseline FROM message_baseline WHERE day = ?", (day,)
            ).fetchall()
            all_time_row = self._conn.execute(
                """
                SELECT hex, callsign, device, distance_nm, seen_at
                FROM all_time_furthest WHERE id = 1
                """
            ).fetchone()

        total_by_device: dict[str, int] = {}
        first_seen_by_hex: dict[str, float] = {}
        hours_by_device: dict[str, dict[int, set[str]]] = {}
        furthest: Optional[FurthestCatch] = None
        highest_altitude: Optional[int] = None

        for row in sighting_rows:
            (
                hex_id,
                device,
                callsign,
                first_seen,
                max_distance_nm,
                max_distance_at,
                max_altitude_ft,
            ) = row
            total_by_device[device] = total_by_device.get(device, 0) + 1
            existing_first = first_seen_by_hex.get(hex_id)
            if existing_first is None or first_seen < existing_first:
                first_seen_by_hex[hex_id] = first_seen

            hour = dt.datetime.fromtimestamp(first_seen, tz=tz).hour
            hours_by_device.setdefault(device, {}).setdefault(hour, set()).add(hex_id)

            if max_distance_nm is not None and (
                furthest is None or max_distance_nm > furthest.distance_nm
            ):
                furthest = FurthestCatch(
                    hex=hex_id,
                    callsign=callsign,
                    device=device,
                    distance_nm=max_distance_nm,
                    seen_at=max_distance_at or first_seen,
                )

            if max_altitude_ft is not None and (
                highest_altitude is None or max_altitude_ft > highest_altitude
            ):
                highest_altitude = max_altitude_ft

        busiest_by_device: dict[str, tuple[int, int]] = {}
        for device, hour_map in hours_by_device.items():
            best_hour, hexes = max(hour_map.items(), key=lambda item: len(item[1]))
            busiest_by_device[device] = (best_hour, len(hexes))

        combined_hour_map: dict[int, set[str]] = {}
        for hex_id, first_seen in first_seen_by_hex.items():
            hour = dt.datetime.fromtimestamp(first_seen, tz=tz).hour
            combined_hour_map.setdefault(hour, set()).add(hex_id)
        busiest_combined = None
        if combined_hour_map:
            best_hour, hexes = max(combined_hour_map.items(), key=lambda item: len(item[1]))
            busiest_combined = (best_hour, len(hexes))
        hourly_counts_combined = {
            hour: len(hexes) for hour, hexes in combined_hour_map.items()
        }

        device_online: dict[str, bool] = {}
        device_errors: dict[str, str] = {}
        currently_tracked_by_device: dict[str, int] = {}
        all_current_hexes: set[str] = set()
        messages_total_by_device: dict[str, int] = {}
        type_by_hex: dict[str, str] = {}
        callsign_by_hex: dict[str, str] = {}
        for (
            device,
            online,
            error,
            tracked_hexes,
            messages_total,
            tracked_types,
            tracked_callsigns,
        ) in device_rows:
            device_online[device] = bool(online)
            if error:
                device_errors[device] = error
            hexes = {h for h in (tracked_hexes or "").split(",") if h}
            currently_tracked_by_device[device] = len(hexes)
            all_current_hexes |= hexes
            if messages_total is not None:
                messages_total_by_device[device] = messages_total
            for pair in (tracked_types or "").split(","):
                hex_id, sep, aircraft_type = pair.partition(":")
                if sep and hex_id and aircraft_type:
                    type_by_hex[hex_id] = aircraft_type
            for pair in (tracked_callsigns or "").split(","):
                hex_id, sep, callsign = pair.partition(":")
                if sep and hex_id and callsign:
                    callsign_by_hex[hex_id] = callsign

        currently_tracked_by_model: dict[str, int] = {}
        currently_tracked_by_airline: dict[str, int] = {}
        for hex_id in all_current_hexes:
            model = type_by_hex.get(hex_id, "Unknown")
            currently_tracked_by_model[model] = currently_tracked_by_model.get(model, 0) + 1
            airline = _airline_code(callsign_by_hex.get(hex_id))
            currently_tracked_by_airline[airline] = currently_tracked_by_airline.get(airline, 0) + 1

        baseline_by_device = dict(baseline_rows)
        messages_today_by_device: dict[str, int] = {}
        for device, total in messages_total_by_device.items():
            baseline = baseline_by_device.get(device)
            if baseline is not None:
                messages_today_by_device[device] = max(0, total - baseline)

        all_time_furthest = None
        if all_time_row and all_time_row[3] is not None:
            hex_id, callsign, device, distance_nm, seen_at = all_time_row
            all_time_furthest = FurthestCatch(
                hex=hex_id,
                callsign=callsign,
                device=device,
                distance_nm=distance_nm,
                seen_at=seen_at,
            )

        return DailyStats(
            day=day,
            total_combined=len(first_seen_by_hex),
            total_by_device=total_by_device,
            furthest=furthest,
            busiest_hour_combined=busiest_combined,
            busiest_hour_by_device=busiest_by_device,
            hourly_counts_combined=hourly_counts_combined,
            highest_altitude_ft=highest_altitude,
            messages_today_by_device=messages_today_by_device,
            currently_tracked_combined=len(all_current_hexes),
            currently_tracked_by_device=currently_tracked_by_device,
            device_online=device_online,
            all_time_furthest=all_time_furthest,
            device_errors=device_errors,
            currently_tracked_by_model=currently_tracked_by_model,
            currently_tracked_by_airline=currently_tracked_by_airline,
        )


def today_key(tz: dt.tzinfo, *, now: Optional[float] = None) -> str:
    """Return the local calendar-day key (YYYY-MM-DD) used to bucket sightings."""

    moment = dt.datetime.fromtimestamp(now, tz=tz) if now is not None else dt.datetime.now(tz=tz)
    return moment.date().isoformat()
