"""The config UI's Stats page: CPU by purpose, data transferred, storage.

``/stats``      the page (refreshes itself every 10 seconds).
``/api/stats``  the data it shows.

On a render server the numbers come from the sampler the server runs
(:mod:`remote_display.resource_stats`), which also breaks the server's own CPU
down by thread and counts each display's traffic.  Each display's own CPU,
memory, storage and data counters arrive in its heartbeat and are read here
from the client registry snapshot.  When no server is publishing (a
standalone or client install, or the server is stopped) this process samples
the machine itself: processes and storage, without the server's thread
breakdown or per-display traffic.
"""
from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, render_template

from remote_display.playlist_store import PlaylistStore, age_seconds, registry_snapshot_path, store_path
from remote_display.registry import read_snapshot
from remote_display.resource_stats import (
    PROJECT_ROOT,
    STALE_AFTER_SECONDS,
    StatsSampler,
    read_stats,
    stats_path,
)

# The page polls every 10 s; a local sample is reused for this long.
LOCAL_SAMPLE_MIN_INTERVAL_SECONDS = 5.0
RESOURCE_FIELDS = (
    "process_cpu_percent", "process_rss_bytes", "uptime_seconds", "system_cpu_percent", "cpu_count",
    "load_1m", "memory_total_bytes", "memory_available_bytes", "temperature_c", "disk_total_bytes",
    "disk_free_bytes", "cache_bytes", "cache_limit_bytes", "bytes_received", "bytes_sent",
    "net_rx_bytes", "net_tx_bytes",
)


def _local_storage(env: Mapping[str, str], panel_cache: Path | None) -> dict[str, tuple[Path, int | None]]:
    storage: dict[str, tuple[Path, int | None]] = {
        "Caches (cache/)": (PROJECT_ROOT / "cache", None),
        "Image caches (images/cache/)": (PROJECT_ROOT / "images" / "cache", None),
    }
    if panel_cache is not None:
        try:
            limit = int(env.get("DESK_DISPLAY_CLIENT_CACHE_MAX_MB") or 256) << 20
        except ValueError:
            limit = 256 << 20
        storage["Display client cache"] = (panel_cache, limit)
    return storage


def client_stats_rows(snapshot: Mapping[str, Any], names: Mapping[str, Any],
                      traffic: Mapping[str, Any], now: float) -> list[dict[str, Any]]:
    """One row per display: what it reported about itself, and its traffic with the server."""

    clients = snapshot.get("clients") or {}
    heartbeat = float(snapshot.get("heartbeat_interval_seconds") or 100)
    totals = traffic.get("totals") or {}
    session = traffic.get("session") or {}
    rates = traffic.get("rates") or {}
    rows = []
    for client_id in sorted(set(clients) | {c for c in totals if not c.startswith("_")}):
        entry = clients.get(client_id) or {}
        resources = entry.get("resources") or {}
        telemetry = entry.get("telemetry") or {}
        seen = age_seconds(entry.get("last_seen"), now)
        if entry.get("disabled"):
            state = "disabled"
        elif seen is None:
            state = "unknown"
        elif seen > heartbeat * 1.5:
            state = "stale"
        else:
            state = "online"
        rows.append({
            "client_id": client_id,
            "friendly_name": (names.get(client_id) or {}).get("friendly_name"),
            "display_profile": (entry.get("capabilities") or {}).get("display_profile"),
            "address": entry.get("address"),
            "state": state,
            "last_heartbeat_age_seconds": seen,
            # None from displays whose software predates resource reports.
            "resources": {key: resources.get(key) for key in RESOURCE_FIELDS} if resources else None,
            "last_sync_download_bytes": telemetry.get("download_bytes"),
            "heartbeat_rtt_ms": telemetry.get("heartbeat_rtt_ms"),
            "traffic_total": totals.get(client_id),
            "traffic_session": session.get(client_id),
            "traffic_rate": rates.get(client_id),
        })
    return rows


def register(app, *, env: Mapping[str, str] | None = None, clock: Callable[[], float] | None = None,
             local_sampler: StatsSampler | None = None,
             panel_cache_dir: Callable[[], Path | None] | None = None) -> Blueprint:
    """Add the Stats page and its API to *app*.

    *panel_cache_dir* names this device's display client cache (client and
    combined installs), shown against its limit when the page samples locally.
    """

    source_env = os.environ if env is None else env
    now = clock or time.time
    blueprint = Blueprint("stats", __name__)
    local = {"sampler": local_sampler, "document": None, "at": None}
    lock = threading.Lock()

    def _local_document() -> dict[str, Any]:
        with lock:
            if local["sampler"] is None:
                panel_cache = panel_cache_dir() if panel_cache_dir is not None else None
                local["sampler"] = StatsSampler(role="local", own_threads=False,
                                                storage=_local_storage(source_env, panel_cache))
            current = time.monotonic()
            if local["document"] is None or current - local["at"] >= LOCAL_SAMPLE_MIN_INTERVAL_SECONDS:
                local["document"] = local["sampler"].sample()
                local["at"] = current
            return local["document"]

    @blueprint.get("/stats")
    def stats_page():
        return render_template("stats.html")

    @blueprint.get("/api/stats")
    def stats_api():
        current = now()
        published = read_stats(stats_path(source_env))
        fresh = published is not None and 0 <= current - float(published.get("generated_at") or 0) <= STALE_AFTER_SECONDS
        stats = published if fresh else _local_document()
        snapshot = read_snapshot(registry_snapshot_path(source_env))
        try:
            names = PlaylistStore(store_path(source_env)).snapshot()["clients"]
        except Exception:  # noqa: BLE001 - names are decoration; the stats still show
            names = {}
        return jsonify({
            "source": "server" if fresh else "local",
            # A server stopped publishing (stopped, or still starting).
            "server_stale_seconds": None if published is None or fresh
            else round(current - float(published.get("generated_at") or 0)),
            "now": current,
            "stats": stats,
            "clients": client_stats_rows(snapshot, names, stats.get("traffic") or {}, current),
        })

    app.register_blueprint(blueprint)
    return blueprint


__all__ = ["client_stats_rows", "register"]
