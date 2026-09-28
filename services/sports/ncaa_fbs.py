"""NCAA FBS Top 25 scoreboard fetch service."""

from __future__ import annotations

import datetime as dt
import logging
import threading

from config import CENTRAL_TIME
from screens.ncaa_fbs_scoreboard import _scoreboard_date, fetch_games_for_range

# The last week that loaded, reused when every ESPN host fails so a transient
# block does not blank the board.
_LAST_GOOD: dict[str, object] = {}
_LAST_GOOD_LOCK = threading.Lock()


def scoreboard_date(now: dt.datetime | None = None) -> dt.date:
    return _scoreboard_date(now)


def week_start_for_date(day: dt.date) -> dt.date:
    """Return the Monday starting the display week that contains *day*.

    Weeks run Monday through Sunday, matching ESPN's college football
    calendar. The morning cutoff in ``scoreboard_date`` keeps the weekend's
    scores up until Monday late morning.
    """

    return day - dt.timedelta(days=day.weekday())


def week_dates(day: dt.date) -> list[dt.date]:
    start = week_start_for_date(day)
    return [start + dt.timedelta(days=offset) for offset in range(7)]


def _game_sort_key(indexed: tuple[int, dict]) -> tuple[str, int]:
    index, game = indexed
    return (str(game.get("date") or ""), index)


def fetch_scoreboard(
    *, day: dt.date | None = None, now: dt.datetime | None = None
) -> list[dict]:
    """Return every Top 25 game in the Monday-Sunday week, oldest first.

    The whole week is one request (ESPN refuses clients that hammer it). If
    every ESPN host fails, the last games loaded for the same week are kept.
    """

    current_now = now or dt.datetime.now(CENTRAL_TIME)
    target_day = day or scoreboard_date(current_now)
    days = week_dates(target_day)
    try:
        fetched = fetch_games_for_range(days[0], days[-1])
    except Exception as exc:
        with _LAST_GOOD_LOCK:
            stale = _LAST_GOOD.get("games") if _LAST_GOOD.get("week") == days[0] else None
        logging.error(
            "Failed to fetch NCAA FBS scoreboard for week of %s: %s%s",
            days[0],
            exc,
            "; keeping last good games" if stale else "",
        )
        return list(stale) if isinstance(stale, list) else []

    games: list[dict] = []
    seen: set[object] = set()
    for game in fetched:
        game_id = game.get("id")
        if game_id is not None:
            if game_id in seen:
                continue
            seen.add(game_id)
        games.append(game)
    games = [game for _, game in sorted(enumerate(games), key=_game_sort_key)]
    with _LAST_GOOD_LOCK:
        _LAST_GOOD.update(week=days[0], games=list(games))
    return games
