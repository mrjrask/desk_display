"""MLB scoreboard fetch service."""

from __future__ import annotations

import datetime as dt

from config import CENTRAL_TIME
from services.sports.scoreboard_window import before_scoreboard_update, compose_pre_update_scoreboard
from screens.mlb_scoreboard import _fetch_games_for_date, _scoreboard_date


def scoreboard_date(now: dt.datetime | None = None) -> dt.date:
    return _scoreboard_date(now)


def fetch_scoreboard(*, day: dt.date | None = None, now: dt.datetime | None = None) -> list[dict]:
    current_now = now or dt.datetime.now(CENTRAL_TIME)
    target_day = day or scoreboard_date(current_now)
    if day is None and before_scoreboard_update(now=current_now, scoreboard_day=target_day):
        return compose_pre_update_scoreboard(
            now=current_now,
            scoreboard_day=target_day,
            fetch_games_for_date=_fetch_games_for_date,
        )
    games = _fetch_games_for_date(target_day)
    return games if isinstance(games, list) else []
