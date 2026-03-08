"""Canonical cached read APIs for weather and sports payloads."""
from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import data_fetch
from config import CENTRAL_TIME
from screens.mlb_scoreboard import _fetch_games_for_date as _fetch_mlb_games_for_date, _scoreboard_date as _mlb_scoreboard_date
from screens.nba_scoreboard import _fetch_games_for_date as _fetch_nba_games_for_date, _scoreboard_date as _nba_scoreboard_date
from screens.nfl_scoreboard import (
    _fetch_games_for_week as _fetch_nfl_games_for_week,
    _fetch_next_games as _fetch_nfl_next_games,
)
from screens.nhl_scoreboard import _fetch_games_for_date as _fetch_nhl_games_for_date, _scoreboard_date as _nhl_scoreboard_date
from screens.wbc_scoreboard import _fetch_games_for_date as _fetch_wbc_games_for_date, _scoreboard_date as _wbc_scoreboard_date


@dataclass
class _Entry:
    value: Any
    fetched_at: float


class DataProvider:
    """TTL cache with stale fallback for API-backed payloads."""

    def __init__(self) -> None:
        self._cache: Dict[str, _Entry] = {}

    def _read_cached(
        self,
        key: str,
        fetcher: Callable[[], Any],
        ttl_seconds: int,
    ) -> Any:
        now = time.monotonic()
        cached = self._cache.get(key)
        if cached and now - cached.fetched_at < ttl_seconds:
            return cached.value

        try:
            value = fetcher()
            if value is None:
                if cached is not None:
                    logging.warning("Using stale %s payload after empty fetch result", key)
                    return cached.value
                return None
            self._cache[key] = _Entry(value=value, fetched_at=now)
            return value
        except Exception as exc:
            if cached is not None:
                logging.warning("Using stale %s payload after fetch failure: %s", key, exc)
                return cached.value
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
            return {
                "scoreboards": {
                    "nfl": _fetch_nfl_games_for_week(now) or _fetch_nfl_next_games(now.date()),
                    "mlb": _fetch_mlb_games_for_date(_mlb_scoreboard_date(now)),
                    "wbc": _fetch_wbc_games_for_date(_wbc_scoreboard_date(now)),
                    "nba": _fetch_nba_games_for_date(_nba_scoreboard_date(now)),
                    "nhl": _fetch_nhl_games_for_date(_nhl_scoreboard_date(now)),
                },
            }

        return self._read_cached("sports_payloads", _fetch_payloads, ttl_seconds)


provider = DataProvider()
