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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import data_fetch
from config import CENTRAL_TIME
from services.sports.mlb import fetch_scoreboard as fetch_mlb_scoreboard
from services.sports.nba import fetch_scoreboard as fetch_nba_scoreboard
from services.sports.ncaam import fetch_scoreboard as fetch_ncaam_scoreboard
from services.sports.nfl import (
    fetch_next_scoreboard as fetch_nfl_next_scoreboard,
    fetch_week_scoreboard as fetch_nfl_week_scoreboard,
)
from services.sports.nhl import fetch_scoreboard as fetch_nhl_scoreboard


@dataclass
class _Entry:
    value: Any
    fetched_at: float


class DataProvider:
    """TTL cache with stale fallback for API-backed payloads."""

    def __init__(self) -> None:
        self._cache: Dict[str, _Entry] = {}
        self._cache_lock = threading.RLock()

    def _read_cached(
        self,
        key: str,
        fetcher: Callable[[], Any],
        ttl_seconds: int,
    ) -> Any:
        now = time.monotonic()
        with self._cache_lock:
            cached = self._cache.get(key)
        if cached and now - cached.fetched_at < ttl_seconds:
            return cached.value

        try:
            value = fetcher()
            if value is None:
                with self._cache_lock:
                    stale = self._cache.get(key)
                if stale is not None:
                    logging.warning("Using stale %s payload after empty fetch result", key)
                    return stale.value
                return None
            with self._cache_lock:
                self._cache[key] = _Entry(value=value, fetched_at=now)
            return value
        except Exception as exc:
            with self._cache_lock:
                stale = self._cache.get(key)
            if stale is not None:
                logging.warning("Using stale %s payload after fetch failure: %s", key, exc)
                return stale.value
            raise

    def read_weather(self, *, ttl_seconds: int = 300) -> Any:
        return self._read_cached(
            "weather",
            lambda: data_fetch.fetch_weather(force_refresh=True),
            ttl_seconds,
        )

    def read_sports_payloads(self, *, ttl_seconds: int = 120) -> Dict[str, Any]:
        def _fetch_payloads() -> Dict[str, Any]:
            now = dt.datetime.now(CENTRAL_TIME)
            today = now.date()

            def _fetch_nfl() -> Any:
                return fetch_nfl_week_scoreboard(now=now) or fetch_nfl_next_scoreboard(start_date=today)

            tasks: Dict[str, Callable[[], Any]] = {
                "nfl": _fetch_nfl,
                "mlb": lambda: fetch_mlb_scoreboard(now=now),
                "nba": lambda: fetch_nba_scoreboard(now=now),
                "ncaam": lambda: fetch_ncaam_scoreboard(now=now),
                "nhl": lambda: fetch_nhl_scoreboard(now=now),
            }

            scoreboards: Dict[str, Any] = {league: [] for league in tasks}
            with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
                futures = {
                    league: pool.submit(fetcher)
                    for league, fetcher in tasks.items()
                }

                for league, future in futures.items():
                    try:
                        result = future.result()
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
                },
            }

        return self._read_cached("sports_payloads", _fetch_payloads, ttl_seconds)


provider = DataProvider()
