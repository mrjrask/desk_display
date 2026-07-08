"""FIFA World Cup scoreboard fetch service."""

from __future__ import annotations

import datetime as dt

from config import CENTRAL_TIME
from services.sports.scoreboard_window import before_scoreboard_update, compose_pre_update_scoreboard
from screens.world_cup_scoreboard import (
    _fetch_games_for_date,
    _round_dates,
    _round_for_date,
    _scoreboard_date,
    _with_round_metadata,
)


def scoreboard_date(now: dt.datetime | None = None) -> dt.date:
    return _scoreboard_date(now)


def fetch_scoreboard(*, day: dt.date | None = None, now: dt.datetime | None = None) -> list[dict]:
    current_now = now or dt.datetime.now(CENTRAL_TIME)
    selected_day = day or scoreboard_date(current_now)
    round_name = None if day is not None else _round_for_date(selected_day)
    if round_name:
        games = []
        seen: set[str] = set()
        for round_day in _round_dates(round_name):
            for game in _fetch_games_for_date(round_day):
                game_id = str(game.get("id") or f"{game.get('date')}-{len(games)}")
                if game_id in seen:
                    continue
                seen.add(game_id)
                games.append(game)
        games.sort(key=lambda game: str(game.get("date") or ""))
        return _with_round_metadata(games, round_name)
    if day is None and before_scoreboard_update(now=current_now, scoreboard_day=selected_day):
        return compose_pre_update_scoreboard(
            now=current_now,
            scoreboard_day=selected_day,
            fetch_games_for_date=_fetch_games_for_date,
        )
    games = _fetch_games_for_date(selected_day)
    return games if isinstance(games, list) else []
