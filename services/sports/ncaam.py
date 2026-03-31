"""NCAAM scoreboard fetch service."""

from __future__ import annotations

import datetime as dt

from screens.ncaam_scoreboard import _fetch_games_for_date, _scoreboard_date


def scoreboard_date(now: dt.datetime | None = None) -> dt.date:
    return _scoreboard_date(now)


def fetch_scoreboard(
    *,
    day: dt.date | None = None,
    now: dt.datetime | None = None,
    mode: str | None = None,
) -> list[dict]:
    selected_mode = (mode or "").strip().lower()
    target_day = day or scoreboard_date(now)
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
