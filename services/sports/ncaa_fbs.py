"""NCAA FBS Top 25 scoreboard fetch service."""

from __future__ import annotations

import datetime as dt
import threading

from config import CENTRAL_TIME
from screens.ncaa_fbs_scoreboard import _fetch_games_for_date, _scoreboard_date

# Days whose Top 25 games are all final never change again, so the weekly
# fetch reuses them instead of re-requesting the whole week on every refresh.
_FINAL_DAY_CACHE: dict[dt.date, list[dict]] = {}
_FINAL_DAY_CACHE_LOCK = threading.Lock()


def scoreboard_date(now: dt.datetime | None = None) -> dt.date:
    return _scoreboard_date(now)


def week_start_for_date(day: dt.date) -> dt.date:
    """Return the Wednesday starting the display week that contains *day*.

    Like the NFL scoreboard, a week runs Wednesday through Tuesday, so on a
    Monday or Tuesday the board still shows the weekend just played.
    """

    return day - dt.timedelta(days=(day.weekday() - 2) % 7)


def week_dates(day: dt.date) -> list[dt.date]:
    start = week_start_for_date(day)
    return [start + dt.timedelta(days=offset) for offset in range(7)]


def _all_final(games: list[dict]) -> bool:
    return bool(games) and all(
        str(((game.get("status") or {}).get("type") or {}).get("state") or "").lower() == "post"
        or bool(((game.get("status") or {}).get("type") or {}).get("completed"))
        for game in games
    )


def _games_for_day(day: dt.date, today: dt.date) -> list[dict]:
    with _FINAL_DAY_CACHE_LOCK:
        cached = _FINAL_DAY_CACHE.get(day)
    if cached is not None:
        return list(cached)
    games = _fetch_games_for_date(day)
    games = games if isinstance(games, list) else []
    if day < today and _all_final(games):
        with _FINAL_DAY_CACHE_LOCK:
            _FINAL_DAY_CACHE[day] = list(games)
            for stale_day in [d for d in _FINAL_DAY_CACHE if d < day - dt.timedelta(days=14)]:
                del _FINAL_DAY_CACHE[stale_day]
    return games


def _game_sort_key(indexed: tuple[int, dict]) -> tuple[str, int]:
    index, game = indexed
    return (str(game.get("date") or ""), index)


def fetch_scoreboard(
    *, day: dt.date | None = None, now: dt.datetime | None = None
) -> list[dict]:
    """Return every Top 25 game in the Wednesday-Tuesday week, oldest first."""

    current_now = now or dt.datetime.now(CENTRAL_TIME)
    target_day = day or scoreboard_date(current_now)
    games: list[dict] = []
    seen: set[object] = set()
    for week_day in week_dates(target_day):
        for game in _games_for_day(week_day, current_now.date()):
            game_id = game.get("id")
            if game_id is not None:
                if game_id in seen:
                    continue
                seen.add(game_id)
            games.append(game)
    return [game for _, game in sorted(enumerate(games), key=_game_sort_key)]
