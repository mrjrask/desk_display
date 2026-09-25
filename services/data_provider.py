"""Canonical cached read APIs for weather and sports payloads.

Thread-safety: ``DataProvider`` may be shared by multiple threads. Cache
lookups/updates and stale fallback decisions are guarded by an instance-level
re-entrant lock so concurrent readers do not race on ``_cache``.
"""
from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

import data_fetch
from config import CENTRAL_TIME
from services.sports.mlb import fetch_scoreboard as fetch_mlb_scoreboard
from services.sports.nba import fetch_scoreboard as fetch_nba_scoreboard
from services.sports.ncaam import fetch_scoreboard as fetch_ncaam_scoreboard
from services.sports.nfl import (
    WeeklyResult,
    fetch_next_scoreboard as fetch_nfl_next_scoreboard,
    fetch_week_scoreboard_result as fetch_nfl_week_scoreboard_result,
)
from services.sports.nhl import fetch_scoreboard as fetch_nhl_scoreboard
from services.sports.world_cup import fetch_scoreboard as fetch_world_cup_scoreboard


@dataclass
class _Entry:
    value: Any
    fetched_at: float


@dataclass
class _Flight:
    lock: threading.Lock
    generation: int = 0
    result: Any = None
    error: Optional[Exception] = None


def _payload_source_label(key: str, value: Any) -> Optional[str]:
    """Return a human-readable upstream source label for payload logging."""

    if key == "weather" and isinstance(value, dict):
        source = value.get("source") or value.get("provider")
        if isinstance(source, str) and source.strip():
            return source.strip()
    return None


class DataProvider:
    """TTL cache with stale fallback for API-backed payloads."""

    def __init__(self) -> None:
        self._cache: dict[str, _Entry] = {}
        self._cache_lock = threading.RLock()
        self._flights: dict[str, _Flight] = {}

    def read(
        self,
        key: str,
        fetcher: Callable[[], Any],
        *,
        ttl_seconds: int = 300,
        force: bool = False,
    ) -> Any:
        """Read an arbitrary named source through the shared single-flight cache.

        This is the low-level extension point used by :class:`DataCoordinator`.
        A source name must be stable and globally identify the request; callers
        should include request-shaping parameters in it when necessary.
        """

        if not key or not isinstance(key, str):
            raise ValueError("Data source key must be a non-empty string")
        return self._read_cached(key, fetcher, 0 if force else max(0, ttl_seconds))

    def _read_cached(
        self,
        key: str,
        fetcher: Callable[[], Any],
        ttl_seconds: int,
    ) -> Any:
        now = time.monotonic()
        with self._cache_lock:
            cached = self._cache.get(key)
            flight = self._flights.setdefault(key, _Flight(lock=threading.Lock()))
            observed_generation = flight.generation
        if cached and now - cached.fetched_at < ttl_seconds:
            source = _payload_source_label(key, cached.value)
            if source:
                logging.info(
                    "Using cached %s payload from %s (age: %.0fs, TTL: %ds)",
                    key,
                    source,
                    now - cached.fetched_at,
                    ttl_seconds,
                )
            return cached.value

        with flight.lock:
            now = time.monotonic()
            with self._cache_lock:
                cached = self._cache.get(key)
            if cached and now - cached.fetched_at < ttl_seconds:
                return cached.value
            if flight.generation != observed_generation:
                if flight.error is not None:
                    raise flight.error
                return flight.result

            try:
                value = fetcher()
                if value is None:
                    with self._cache_lock:
                        stale = self._cache.get(key)
                    if stale is not None:
                        source = _payload_source_label(key, stale.value)
                        if source:
                            logging.warning(
                                "Using stale %s payload from %s after empty fetch result",
                                key,
                                source,
                            )
                        else:
                            logging.warning("Using stale %s payload after empty fetch result", key)
                        result = stale.value
                    else:
                        result = None
                else:
                    fetched_at = time.monotonic()
                    with self._cache_lock:
                        self._cache[key] = _Entry(value=value, fetched_at=fetched_at)
                    source = _payload_source_label(key, value)
                    if source:
                        logging.info("Fetched %s payload from %s", key, source)
                    result = value
            except Exception as exc:
                with self._cache_lock:
                    stale = self._cache.get(key)
                if stale is not None:
                    source = _payload_source_label(key, stale.value)
                    if source:
                        logging.warning(
                            "Using stale %s payload from %s after fetch failure: %s",
                            key,
                            source,
                            exc,
                        )
                    else:
                        logging.warning("Using stale %s payload after fetch failure: %s", key, exc)
                    result = stale.value
                else:
                    flight.result = None
                    flight.error = exc
                    flight.generation += 1
                    raise

            flight.result = result
            flight.error = None
            flight.generation += 1
            return result

    def read_weather(self, *, ttl_seconds: int = 300) -> Any:
        return self._read_cached(
            "weather",
            lambda: data_fetch.fetch_weather(force_refresh=True),
            ttl_seconds,
        )

    def read_sports_payloads(
        self,
        *,
        ttl_seconds: int = 120,
        leagues: Optional[set[str]] = None,
        force_refresh_leagues: Optional[set[str]] = None,
    ) -> dict[str, Any]:
        supported_leagues = {"nfl", "mlb", "nba", "ncaam", "nhl", "world_cup"}
        selected_leagues = frozenset(
            league
            for league in (leagues or supported_leagues)
            if league in supported_leagues
        )

        def _fetch_payloads() -> dict[str, Any]:
            now = dt.datetime.now(CENTRAL_TIME)
            today = now.date()

            def _fetch_nfl() -> Any:
                if "nfl" in (force_refresh_leagues or set()):
                    weekly = fetch_nfl_week_scoreboard_result(
                        now=now,
                        force_refresh=True,
                    )
                else:
                    weekly = fetch_nfl_week_scoreboard_result(now=now)
                if weekly.games:
                    return weekly
                return WeeklyResult(games=fetch_nfl_next_scoreboard(start_date=today))

            all_tasks: dict[str, Callable[[], Any]] = {
                "nfl": _fetch_nfl,
                "mlb": lambda: fetch_mlb_scoreboard(now=now),
                "nba": lambda: fetch_nba_scoreboard(now=now),
                "ncaam": lambda: fetch_ncaam_scoreboard(now=now),
                "nhl": lambda: fetch_nhl_scoreboard(now=now),
                "world_cup": lambda: fetch_world_cup_scoreboard(now=now),
            }

            tasks = {league: fetcher for league, fetcher in all_tasks.items() if league in selected_leagues}

            scoreboards: dict[str, Any] = {league: [] for league in all_tasks}
            scoreboard_metadata: dict[str, dict[str, bool]] = {"nfl": {"stale": False}}
            if not tasks:
                return {
                    "scoreboards": scoreboards,
                    "scoreboard_metadata": scoreboard_metadata,
                }

            with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
                futures = {
                    league: pool.submit(fetcher)
                    for league, fetcher in tasks.items()
                }

                for league, future in futures.items():
                    try:
                        result = future.result()
                        if league == "nfl" and isinstance(result, WeeklyResult):
                            scoreboards[league] = result.games
                            scoreboard_metadata[league] = {"stale": result.stale}
                        else:
                            scoreboards[league] = result or []
                    except Exception as exc:
                        logging.error("Failed to fetch %s scoreboard payload: %s", league, exc)

            return {
                "scoreboards": {
                    "nfl": scoreboards["nfl"],
                    "mlb": scoreboards["mlb"],
                    "nba": scoreboards["nba"],
                    "ncaam": scoreboards["ncaam"],
                    "nhl": scoreboards["nhl"],
                    "world_cup": scoreboards["world_cup"],
                },
                "scoreboard_metadata": scoreboard_metadata,
            }

        cache_key = "sports_payloads:" + ",".join(sorted(selected_leagues))
        force_refresh = bool(selected_leagues.intersection(force_refresh_leagues or set()))
        return self._read_cached(
            cache_key,
            _fetch_payloads,
            0 if force_refresh else ttl_seconds,
        )


provider = DataProvider()
