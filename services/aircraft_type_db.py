"""Local hex -> ICAO aircraft type lookup, mirroring tar1090's own database.

dump1090-fa/readsb's ``aircraft.json`` only carries a per-aircraft ``"t"``
(type) field when the receiver operator has separately configured readsb
with ``--db-file`` pointing at an aircraft database -- most installs never
do this. tar1090's web map looks "complete" anyway because it resolves
aircraft type client-side, in the browser, from its own bundled copy of the
same database (https://github.com/wiedehopf/tar1090-db), entirely
independent of what aircraft.json contains. That's why the ADS-B stats
screen previously showed "Unknown" for nearly everything even though the
receiver's own map does not.

This module downloads that same database (the flat, semicolon-delimited
CSV published on the repo's ``csv`` branch) into a small local SQLite
lookup table and refreshes it periodically, so ``services.adsb`` can fill in
a type for aircraft.json entries that omit ``"t"``.
"""

from __future__ import annotations

import contextlib
import gzip
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

import config
from paths import resolve_cache_file_path
from services.http_client import http_get

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS aircraft_types (
    hex TEXT PRIMARY KEY,
    type TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None
_conn_path: Optional[str] = None


def _db_path() -> str:
    return str(resolve_cache_file_path("ADSB_TYPE_DB_PATH", "aircraft_types.db"))


def _get_conn() -> Optional[sqlite3.Connection]:
    """Return a cached connection to the lookup DB, or ``None`` if it doesn't
    exist yet (nothing has been downloaded/built)."""

    global _conn, _conn_path
    db_path = _db_path()
    if not Path(db_path).exists():
        return None
    with _lock:
        if _conn is not None and _conn_path == db_path:
            return _conn
        if _conn is not None:
            with contextlib.suppress(sqlite3.Error):
                _conn.close()
        _conn = sqlite3.connect(db_path, check_same_thread=False)
        _conn_path = db_path
        return _conn


def lookup(hex_id: str) -> Optional[str]:
    """Look up the ICAO type designator (e.g. ``"B738"``) for *hex_id*.

    Purely local/read-only -- never touches the network. Returns ``None`` if
    the lookup DB hasn't been built yet, or the hex isn't in it.
    """

    if not config.ADSB_TYPE_DB_ENABLED or not hex_id:
        return None
    conn = _get_conn()
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT type FROM aircraft_types WHERE hex = ?", (hex_id.strip().lower(),)
        ).fetchone()
    except sqlite3.Error:
        logger.exception("ADS-B type DB: lookup failed")
        return None
    return row[0] if row else None


def _parse_csv_rows(raw_csv: bytes):
    """Yield ``(hex_lower, type)`` for each usable row of the tar1090-db CSV.

    Format is flat, semicolon-delimited, one aircraft per line, no header:
    ``icao24;registration;type;dbFlags;description;...``. Rows with no type
    (e.g. "Miscode" placeholder entries) are skipped.
    """

    text = raw_csv.decode("utf-8", errors="replace")
    for line in text.splitlines():
        if not line:
            continue
        fields = line.split(";")
        if len(fields) < 3:
            continue
        hex_id = fields[0].strip().lower()
        aircraft_type = fields[2].strip()
        if hex_id and aircraft_type:
            yield hex_id, aircraft_type


def _is_fresh(conn: sqlite3.Connection, *, max_age_days: int) -> bool:
    row = conn.execute("SELECT value FROM meta WHERE key = 'updated_at'").fetchone()
    if row is None:
        return False
    try:
        updated_at = float(row[0])
    except (TypeError, ValueError):
        return False
    return (time.time() - updated_at) < max_age_days * 86400


def refresh(*, force: bool = False, timeout: float = 60.0) -> bool:
    """Download and rebuild the local type lookup DB if it's missing/stale.

    Best-effort: network or parse failures are logged and swallowed so a
    receiver's offline database refresh never breaks polling. Returns
    ``True`` if the DB was (re)built.
    """

    if not config.ADSB_TYPE_DB_ENABLED:
        return False

    db_path = _db_path()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        conn.executescript(_SCHEMA_SQL)
        if not force and _is_fresh(conn, max_age_days=config.ADSB_TYPE_DB_REFRESH_DAYS):
            return False

        try:
            response = http_get(config.ADSB_TYPE_DB_URL, timeout=timeout)
            response.raise_for_status()
            raw_csv = gzip.decompress(response.content)
        except Exception:
            logger.warning("ADS-B type DB: refresh failed", exc_info=True)
            return False

        rows = list(_parse_csv_rows(raw_csv))
        if not rows:
            logger.warning("ADS-B type DB: downloaded database had no usable rows")
            return False

        with conn:
            conn.execute("DELETE FROM aircraft_types")
            conn.executemany(
                "INSERT OR REPLACE INTO aircraft_types (hex, type) VALUES (?, ?)", rows
            )
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('updated_at', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(time.time()),),
            )
        logger.info("ADS-B type DB: refreshed with %d aircraft types", len(rows))
        return True
    finally:
        conn.close()
        # Force _get_conn() to reopen against the freshly rebuilt file.
        global _conn, _conn_path
        with _lock:
            if _conn is not None:
                with contextlib.suppress(sqlite3.Error):
                    _conn.close()
            _conn = None
            _conn_path = None
