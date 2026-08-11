"""NCAAM scoreboard fetch service."""

from __future__ import annotations

import datetime as dt

from config import CENTRAL_TIME
from screens.ncaam_scoreboard import _fetch_games_for_date, _scoreboard_date
from services.sports.scoreboard_window import (
    before_scoreboard_update,
    compose_pre_update_scoreboard,
)


def scoreboard_date(now: dt.datetime | None = None) -> dt.date:
    return _scoreboard_date(now)


def fetch_scoreboard(
    *,
    day: dt.date | None = None,
    now: dt.datetime | None = None,
    mode: str | None = None,
) -> list[dict]:
    current_now = now or dt.datetime.now(CENTRAL_TIME)
    selected_mode = (mode or "").strip().lower()
    target_day = day or scoreboard_date(current_now)
    if day is None and before_scoreboard_update(now=current_now, scoreboard_day=target_day):
        games = compose_pre_update_scoreboard(
            now=current_now,
            scoreboard_day=target_day,
            fetch_games_for_date=lambda selected_day: _fetch_games_for_date(selected_day, mode=mode),
        )
        if selected_mode != "tournament" or games:
            return games
    else:
        games = _fetch_games_for_date(target_day, mode=mode)
    if not isinstance(games, list):
        games = []

    if selected_mode != "tournament" or games:
        return games

    # March Madness has off-days; advance to the next day with games.
    for day_offset in range(1, 8):
        next_day = target_day + dt.timedelta(days=day_offset)
        next_games = _fetch_games_for_date(next_day, mode=mode)
        if isinstance(next_games, list) and next_games:
            return next_games

    return games
