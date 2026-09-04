"""NFL scoreboard fetch service."""

from __future__ import annotations

import csv
import datetime as dt
import io
import logging
import time
from collections.abc import Mapping, MutableMapping
from typing import Any

from config import CENTRAL_TIME
from screens.nfl_scoreboard import _fetch_games_for_week, _fetch_next_games

REQUEST_TIMEOUT = 10
FETCH_CACHE_TTL_SECONDS = 60
_SITE_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
)
_CDN_SCOREBOARD_URL = "https://cdn.espn.com/core/nfl/scoreboard"
_NFLVERSE_SCHEDULE_URL = (
    "https://github.com/nflverse/nfldata/releases/download/schedules/games.csv"
)


def _date_parameter(start: dt.date, end: dt.date) -> str:
    first = start.strftime("%Y%m%d")
    return first if start == end else f"{first}-{end.strftime('%Y%m%d')}"


def _events_from_payload(payload: Any) -> list[dict]:
    """Find the ESPN event list in either Site API or CDN response wrappers."""

    if not isinstance(payload, Mapping):
        raise ValueError("NFL scoreboard response was not an object")
    events = payload.get("events")
    if isinstance(events, list):
        return [event for event in events if isinstance(event, dict)]
    for key in ("content", "sbData", "scoreboard"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            try:
                return _events_from_payload(nested)
            except ValueError:
                pass
    raise ValueError("NFL scoreboard response did not contain events")


def _games_from_events(events: list[dict]) -> list[dict]:
    games: list[dict] = []
    for event in events:
        competitions = event.get("competitions") or []
        if not competitions or not isinstance(competitions[0], Mapping):
            continue
        game = dict(competitions[0])
        game["_event_date"] = event.get("date")
        game["_event_name"] = event.get("name")
        game["_event_short_name"] = event.get("shortName")
        games.append(game)
    return games


def _fetch_espn(url: str, dates: str, *, session: Any) -> list[dict]:
    separator = "&" if "?" in url else "?"
    response = session.get(
        f"{url}{separator}limit=100&dates={dates}",
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return _games_from_events(_events_from_payload(response.json()))


def _nflverse_status(row: Mapping[str, str]) -> dict:
    away_score = (row.get("away_score") or "").strip()
    home_score = (row.get("home_score") or "").strip()
    completed = bool(away_score and home_score)
    return {
        "type": {
            "state": "post" if completed else "pre",
            "completed": completed,
            "description": "Final" if completed else "Scheduled",
            "shortDetail": "Final" if completed else (row.get("gametime") or "Scheduled"),
        }
    }


def _fetch_nflverse(start: dt.date, end: dt.date, *, session: Any) -> list[dict]:
    response = session.get(_NFLVERSE_SCHEDULE_URL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    games: list[dict] = []
    for row in csv.DictReader(io.StringIO(response.text)):
        try:
            game_day = dt.date.fromisoformat(row.get("gameday") or "")
        except ValueError:
            continue
        if not start <= game_day <= end:
            continue
        game_time = (row.get("gametime") or "00:00").strip()
        event_date = f"{game_day.isoformat()}T{game_time}:00Z"
        away = (row.get("away_team") or "").strip()
        home = (row.get("home_team") or "").strip()
        game_id = row.get("game_id") or row.get("old_game_id")
        games.append(
            {
                "id": game_id,
                "competitors": [
                    {
                        "homeAway": "away",
                        "team": {"abbreviation": away},
                        "score": row.get("away_score"),
                    },
                    {
                        "homeAway": "home",
                        "team": {"abbreviation": home},
                        "score": row.get("home_score"),
                    },
                ],
                "status": _nflverse_status(row),
                "_event_date": event_date,
                "_event_name": f"{away} at {home}",
                "_event_short_name": f"{away} @ {home}",
            }
        )
    return games


def fetch_range(
    start: dt.date,
    end: dt.date,
    *,
    session: Any,
    cache: MutableMapping[tuple[object, ...], tuple[float, list[dict]]],
) -> list[dict]:
    """Fetch an inclusive NFL range with ESPN CDN and nflverse fallbacks."""

    if end < start:
        return []
    cache_key = (start, end, "nfl_scoreboard_range")
    now = time.monotonic()
    cached = cache.get(cache_key)
    if cached and now - cached[0] < FETCH_CACHE_TTL_SECONDS:
        return cached[1]

    dates = _date_parameter(start, end)
    providers = (
        ("ESPN Site", lambda: _fetch_espn(_SITE_SCOREBOARD_URL, dates, session=session)),
        (
            "ESPN CDN",
            lambda: _fetch_espn(
                _CDN_SCOREBOARD_URL + "?xhr=1", dates, session=session
            ),
        ),
        ("nflverse", lambda: _fetch_nflverse(start, end, session=session)),
    )
    for provider_name, fetch in providers:
        try:
            games = fetch()
        except Exception as exc:
            logging.warning("Failed to fetch NFL scoreboard from %s: %s", provider_name, exc)
            continue
        cache[cache_key] = (now, games)
        return games
    return []


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
