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
from screens.mlb_scoreboard import _fetch_games_for_date as _fetch_mlb_games_for_date, _scoreboard_date as _mlb_scoreboard_date
from screens.nba_scoreboard import _fetch_games_for_date as _fetch_nba_games_for_date, _scoreboard_date as _nba_scoreboard_date
from screens.ncaam_scoreboard import _fetch_games_for_date as _fetch_ncaam_games_for_date, _scoreboard_date as _ncaam_scoreboard_date
from screens.nfl_scoreboard import (
    _fetch_games_for_week as _fetch_nfl_games_for_week,
    _fetch_next_games as _fetch_nfl_next_games,
)
from screens.nhl_scoreboard import _fetch_games_for_date as _fetch_nhl_games_for_date, _scoreboard_date as _nhl_scoreboard_date


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
                return _fetch_nfl_games_for_week(now) or _fetch_nfl_next_games(today)

            tasks: Dict[str, Callable[[], Any]] = {
                "nfl": _fetch_nfl,
                "mlb": lambda: _fetch_mlb_games_for_date(_mlb_scoreboard_date(now)),
                "nba": lambda: _fetch_nba_games_for_date(_nba_scoreboard_date(now)),
                "ncaam": lambda: _fetch_ncaam_games_for_date(_ncaam_scoreboard_date(now)),
                "nhl": lambda: _fetch_nhl_games_for_date(_nhl_scoreboard_date(now)),
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
