"""NBA scoreboard fetch service."""

from __future__ import annotations

import datetime as dt

from screens.nba_scoreboard import _fetch_games_for_date, _scoreboard_date


def scoreboard_date(now: dt.datetime | None = None) -> dt.date:
    return _scoreboard_date(now)


def fetch_scoreboard(*, day: dt.date | None = None, now: dt.datetime | None = None) -> list[dict]:
    target_day = day or scoreboard_date(now)
    games = _fetch_games_for_date(target_day)
    return games if isinstance(games, list) else []
