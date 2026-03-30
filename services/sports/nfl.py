"""NFL scoreboard fetch service."""

from __future__ import annotations

import datetime as dt

from config import CENTRAL_TIME
from screens.nfl_scoreboard import _fetch_games_for_week, _fetch_next_games


def fetch_week_scoreboard(*, now: dt.datetime | None = None) -> list[dict]:
    games = _fetch_games_for_week(now)
    return games if isinstance(games, list) else []


def fetch_next_scoreboard(*, start_date: dt.date, max_days: int = 370) -> list[dict]:
    games = _fetch_next_games(start_date, max_days=max_days)
    return games if isinstance(games, list) else []


def fetch_scoreboard(*, now: dt.datetime | None = None) -> list[dict]:
    current_now = now or dt.datetime.now(CENTRAL_TIME)
    games = fetch_week_scoreboard(now=current_now)
    if games:
        return games
    return fetch_next_scoreboard(start_date=current_now.date())
