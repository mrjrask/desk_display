"""Helpers for pre-update scoreboard payload composition."""

from __future__ import annotations

import datetime as dt
from typing import Callable


_FINAL_TOKENS = ("final", "completed", "complete")
_FINAL_EXACT_STATES = {"post", "4"}
_SCHEDULED_TOKENS = ("scheduled", "preview", "pregame", "pre", "future")
_CANCELLED_TOKENS = ("postponed", "canceled", "cancelled", "suspend")


def before_scoreboard_update(*, now: dt.datetime, scoreboard_day: dt.date) -> bool:
    """Return whether the service is still showing the prior scoreboard day."""

    return scoreboard_day < now.date()


def _status_values(game: dict) -> list[str]:
    values: list[str] = []

    def add(value: object) -> None:
        if value is not None:
            text = str(value).strip().lower()
            if text:
                values.append(text)

    status = game.get("status") if isinstance(game, dict) else None
    if isinstance(status, dict):
        for key in ("abstractGameState", "detailedState", "statusCode"):
            add(status.get(key))
        type_info = status.get("type")
        if isinstance(type_info, dict):
            for key in ("state", "name", "description", "shortDetail", "detail"):
                add(type_info.get(key))
            if type_info.get("completed") is True:
                values.append("completed")
        if status.get("completed") is True:
            values.append("completed")
    for key in ("gameState", "gameScheduleState", "status", "state", "statusType"):
        value = game.get(key) if isinstance(game, dict) else None
        if isinstance(value, dict):
            continue
        add(value)
    return values


def is_final_game(game: dict) -> bool:
    values = _status_values(game)
    if any(any(token in value for token in _CANCELLED_TOKENS) for value in values):
        return False
    return (
        any(any(token in value for token in _FINAL_TOKENS) for value in values)
        or any(value in _FINAL_EXACT_STATES for value in values)
    )


def is_scheduled_game(game: dict) -> bool:
    values = _status_values(game)
    if any(any(token in value for token in _CANCELLED_TOKENS) for value in values):
        return False
    if any(any(token == value or token in value for token in _SCHEDULED_TOKENS) for value in values):
        return True
    return any(value in {"1", "s"} for value in values)


def compose_pre_update_scoreboard(
    *,
    now: dt.datetime,
    scoreboard_day: dt.date,
    fetch_games_for_date: Callable[[dt.date], list[dict]],
) -> list[dict]:
    """Before the update cutoff, return yesterday finals followed by today's schedule."""

    prior_games = fetch_games_for_date(scoreboard_day)
    games = [game for game in (prior_games or []) if is_final_game(game)]
    if before_scoreboard_update(now=now, scoreboard_day=scoreboard_day):
        today_games = fetch_games_for_date(now.date())
        games.extend(game for game in (today_games or []) if is_scheduled_game(game))
    return games
