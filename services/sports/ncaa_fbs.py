"""NCAA FBS Top 25 scoreboard fetch service."""

from __future__ import annotations

import datetime as dt
import logging
import threading

from config import CENTRAL_TIME
from screens.ncaa_fbs_scoreboard import (
    _extract_rank,
    _scoreboard_date,
    fetch_games_for_day,
)

# The last week that loaded, reused when every ESPN host fails so a transient
# block does not blank the board.
_LAST_GOOD: dict[str, object] = {}
_LAST_GOOD_LOCK = threading.Lock()

# Days whose Top 25 games are all final never change again, so refreshes
# reuse them instead of re-requesting every day of the week.
_FINAL_DAY_CACHE: dict[dt.date, list[dict]] = {}


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


def _all_final(games: list[dict]) -> bool:
    return bool(games) and all(
        str(((game.get("status") or {}).get("type") or {}).get("state") or "").lower() == "post"
        or bool(((game.get("status") or {}).get("type") or {}).get("completed"))
        for game in games
    )


def _games_for_day(day: dt.date, today: dt.date) -> list[dict]:
    with _LAST_GOOD_LOCK:
        cached = _FINAL_DAY_CACHE.get(day)
    if cached is not None:
        return list(cached)
    games = fetch_games_for_day(day)
    if day < today and _all_final(games):
        with _LAST_GOOD_LOCK:
            _FINAL_DAY_CACHE[day] = list(games)
            for stale_day in [d for d in _FINAL_DAY_CACHE if d < day - dt.timedelta(days=14)]:
                del _FINAL_DAY_CACHE[stale_day]
    return games


_UNRANKED = 999


def _team_ranks(game: dict) -> list[int]:
    teams = game.get("teams") if isinstance(game.get("teams"), dict) else {}
    ranks = []
    for side in ("away", "home"):
        team = teams.get(side)
        rank = _extract_rank(team) if isinstance(team, dict) else None
        ranks.append(rank if rank is not None else _UNRANKED)
    return sorted(ranks)


def _game_sort_key(indexed: tuple[int, dict]) -> tuple[int, int, str, int]:
    """Best-ranked team first, then the other team's rank, then kickoff."""

    index, game = indexed
    best, other = _team_ranks(game)
    return (best, other, str(game.get("date") or ""), index)


def fetch_scoreboard(
    *, day: dt.date | None = None, now: dt.datetime | None = None
) -> list[dict]:
    """Return every Top 25 game in the Monday-Sunday week, best-ranked first.

    ESPN is asked one day at a time (the working host rejects ranged
    dates). If any day fails on every ESPN host, the last games loaded for the same week are kept rather than showing a partial or empty week.
    """

    current_now = now or dt.datetime.now(CENTRAL_TIME)
    target_day = day or scoreboard_date(current_now)
    days = week_dates(target_day)
    fetched: list[dict] = []
    try:
        for week_day in days:
            fetched.extend(_games_for_day(week_day, current_now.date()))
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
