"""Upstream feed catalog and helpers shared by the display loop and the render server.

``main.py`` (standalone) and :mod:`services.server_feeds` (headless render
server) both use these tables and helpers, so the two refresh the same feeds
for the same screens on the same schedule.
"""
from __future__ import annotations

import contextlib
import datetime
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from config import CENTRAL_TIME, SCHEDULE_UPDATE_INTERVAL

AIR_QUALITY_HISTORY_WINDOW_SECONDS = 6 * 60 * 60
# Samples closer together than this replace the previous sample.
AIR_QUALITY_SAMPLE_SPACING_SECONDS = 10 * 60
AirQualitySample = tuple[float, Optional[int], Optional[int], Optional[int]]


def default_cache() -> dict[str, Any]:
    """The empty shape screens expect before any feed has been refreshed."""

    def blank(*keys: str, **extra: Any) -> dict[str, Any]:
        return {**dict.fromkeys(keys), **extra}

    baseball = (
        "stand", "last", "last_alt", "live", "next", "next_alt",
        "current_series", "next_series", "next_home_series", "next_home",
    )
    return {
        "bears": blank("stand"),
        "weather": None,
        "air_quality": None,
        "hawks": blank("stand", "last", "live", "live_feed", "next", "next_home"),
        "wolves": blank("last", "live", "next", "next_home"),
        "bulls": blank("stand", "last", "live", "next", "next_home"),
        "cubs": blank(*baseball, schedule_covers_today=False),
        "sox": blank(*baseball, schedule_covers_today=False),
        "scoreboards": blank("nfl", "mlb", "nba", "ncaam", "nhl"),
        "scoreboard_metadata": {"nfl": {"stale": False}},
    }


def load_air_quality_history(history_path: str, now: float) -> list[AirQualitySample]:
    """Return valid recent AQI samples persisted by an earlier process."""

    path = Path(os.path.expandvars(history_path)).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        raw_history = payload.get("history")
        if not isinstance(raw_history, list):
            raise ValueError("history must be a list")
        history = []
        for entry in raw_history:
            if not isinstance(entry, (list, tuple)) or len(entry) != 4:
                continue
            try:
                stamp = float(entry[0])
                components = tuple(None if value is None else int(value) for value in entry[1:])
            except (TypeError, ValueError):
                continue
            if 0 <= now - stamp <= AIR_QUALITY_HISTORY_WINDOW_SECONDS:
                history.append((stamp, *components))
        return sorted(history, key=lambda entry: entry[0])
    except FileNotFoundError:
        return []
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logging.warning("Unable to load air-quality chart history from %s: %s", path, exc)
        return []


def save_air_quality_history(history_path: str, history: list[AirQualitySample]) -> None:
    """Atomically persist AQI chart samples for the next process startup."""

    path = Path(os.path.expandvars(history_path)).expanduser()
    temp_name: Optional[str] = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f"{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temp_name = handle.name
            json.dump({"history": history}, handle)
        os.replace(temp_name, path)
    except OSError as exc:
        logging.warning("Unable to save air-quality chart history to %s: %s", path, exc)
        if temp_name:
            with contextlib.suppress(OSError):
                os.remove(temp_name)


def append_air_quality_sample(
    history: list[AirQualitySample], report: Any, now: float
) -> list[AirQualitySample]:
    """Add *report*'s AQI components to *history* and drop samples outside the window."""

    history = list(history)
    sample = (now, report.us_aqi_pm2_5, report.us_aqi_pm10, report.us_aqi_ozone)
    if history and now - history[-1][0] < AIR_QUALITY_SAMPLE_SPACING_SECONDS:
        history[-1] = sample
    else:
        history.append(sample)
    return [entry for entry in history if 0 <= now - entry[0] <= AIR_QUALITY_HISTORY_WINDOW_SECONDS]


FEED_DEPENDENCIES: dict[str, set[str]] = {
    "weather": {
        "weather1",
        "weather2",
        "weather hourly",
        "weather daily",
        "weather radar",
        "weather alert",
        "weather logo",
        "weather quad",
    },
    "air_quality": {"air quality", "weather quad"},
    "bears": {"bears stand1", "bears stand2"},
    "hawks": {"hawks stand1", "hawks last", "hawks live", "hawks next", "hawks next home", "hawks logo"},
    "wolves": {"wolves last", "wolves live", "wolves next", "wolves next home", "wolves logo"},
    "bulls": {"bulls stand1", "bulls last", "bulls live", "bulls next", "bulls next home", "bulls logo"},
    "cubs": {
        "cubs stand1",
        "cubs stand2",
        "cubs last",
        "cubs live",
        "cubs no game",
        "cubs next",
        "cubs current series",
        "cubs next series",
        "cubs next home series",
        "cubs next home",
        "cubs logo",
    },
    "sox": {
        "sox stand1",
        "sox stand2",
        "sox last",
        "sox live",
        "sox no game",
        "sox next",
        "sox current series",
        "sox next series",
        "sox next home series",
        "sox next home",
        "sox logo",
    },
    "scoreboards": {
        "NFL Scoreboard",
        "NFL Scoreboard v2",
        "NHL Scoreboard",
        "NHL Scoreboard v2",
        "MLB Scoreboard",
        "MLB Scoreboard v2",
        "NBA Scoreboard",
        "NBA Scoreboard v2",
        "NCAAM Scoreboard",
        "World Cup Scoreboard",
    },
}

FEED_REFRESH_INTERVALS: dict[str, int] = {
    "weather": SCHEDULE_UPDATE_INTERVAL,
    "air_quality": SCHEDULE_UPDATE_INTERVAL,
    "hawks": SCHEDULE_UPDATE_INTERVAL,
    "bulls": SCHEDULE_UPDATE_INTERVAL,
    "wolves": SCHEDULE_UPDATE_INTERVAL,
    "bears": 1800,
    "cubs": 1800,
    "sox": 1800,
    # Game schedules change rarely. Live-window screen loads bypass this daily
    # refresh interval below so scores are still fetched immediately.
    "scoreboards": 24 * 60 * 60,
}

SCOREBOARD_SCREEN_IDS = {
    "NFL Scoreboard",
    "NFL Scoreboard v2",
    "NHL Scoreboard",
    "NHL Scoreboard v2",
    "MLB Scoreboard",
    "MLB Scoreboard v2",
    "NBA Scoreboard",
    "NBA Scoreboard v2",
    "NCAAM Scoreboard",
    "World Cup Scoreboard",
}

SCOREBOARD_SCREEN_TO_LEAGUES: dict[str, set[str]] = {
    "NFL Scoreboard": {"nfl"},
    "NFL Scoreboard v2": {"nfl"},
    "NHL Scoreboard": {"nhl"},
    "NHL Scoreboard v2": {"nhl"},
    "MLB Scoreboard": {"mlb"},
    "MLB Scoreboard v2": {"mlb"},
    "NBA Scoreboard": {"nba"},
    "NBA Scoreboard v2": {"nba"},
    "NCAAM Scoreboard": {"ncaam"},
    "World Cup Scoreboard": {"world_cup"},
}

LIVE_TEAM_SCREEN_TO_FEED: dict[str, str] = {
    "cubs live": "cubs",
    "sox live": "sox",
}

STARTUP_CRITICAL_FEEDS: tuple[str, ...] = ("weather", "scoreboards", "air_quality")


def feeds_for_screen(screen_id: str) -> set[str]:
    """Feeds whose data *screen_id* renders."""

    return {feed for feed, screens in FEED_DEPENDENCIES.items() if screen_id in screens}


def scoreboard_leagues_for_screens(screen_ids: set[str]) -> set[str]:
    leagues: set[str] = set()
    for screen_id in screen_ids:
        leagues.update(SCOREBOARD_SCREEN_TO_LEAGUES.get(screen_id, set()))
    return leagues


def scoreboard_date_for_league(
    league: str, now: Optional[datetime.datetime] = None
) -> datetime.date:
    """Return the date currently selected by a league's scoreboard provider."""

    current = now or datetime.datetime.now(CENTRAL_TIME)
    if current.tzinfo is None:
        current = current.replace(tzinfo=CENTRAL_TIME)

    # These providers use league-specific morning cutoffs, rather than midnight,
    # to keep late games attached to the preceding scoreboard day.
    if league in {"mlb", "nba", "nhl", "ncaam", "world_cup"}:
        from services.sports import mlb, nba, ncaam, nhl, world_cup

        providers = {
            "mlb": mlb,
            "nba": nba,
            "nhl": nhl,
            "ncaam": ncaam,
            "world_cup": world_cup,
        }
        return providers[league].scoreboard_date(current)
    return current.astimezone(CENTRAL_TIME).date()


def is_live_scoreboard_game(game: object) -> bool:
    if not isinstance(game, dict):
        return False

    status_fields: list[str] = []
    status_blob = game.get("status")
    if isinstance(status_blob, dict):
        for key in (
            "detailedState",
            "abstractGameState",
            "gameStatus",
            "gameStatusText",
            "state",
            "gameState",
            "displayClock",
        ):
            value = status_blob.get(key)
            if value:
                status_fields.append(str(value))
        type_blob = status_blob.get("type")
        if isinstance(type_blob, dict):
            for key in ("state", "description", "detail", "shortDetail"):
                value = type_blob.get(key)
                if value:
                    status_fields.append(str(value))
        coded = str(status_blob.get("codedGameState") or "").upper()
        status_code = str(status_blob.get("statusCode") or "").upper()
    else:
        coded = str(game.get("codedGameState") or "").upper()
        status_code = str(game.get("statusCode") or game.get("gameStatus") or "").upper()

    for key in (
        "gameStatusText",
        "gameStatus",
        "detailedState",
        "abstractGameState",
        "status",
        "gameState",
        "displayClock",
    ):
        value = game.get(key)
        if value:
            status_fields.append(str(value))

    status_text = " ".join(part.strip().lower() for part in status_fields if str(part).strip())

    if any(
        token in status_text
        for token in ("final", "postponed", "canceled", "cancelled", "suspend", "scheduled", "preview", "pregame")
    ):
        return False

    if any(
        token in status_text
        for token in (
            "live",
            "in progress",
            "in-progress",
            "intermission",
            "halftime",
            "quarter",
            "period",
            "ot",
            "top",
            "bottom",
        )
    ):
        return True

    if isinstance(status_blob, dict) and isinstance(status_blob.get("type"), dict):
        if str(status_blob["type"].get("state") or "").lower() == "in":
            return True

    return coded == "I" or status_code in {"I", "2", "3"}


def scoreboards_have_live_games(scoreboards: object) -> bool:
    if not isinstance(scoreboards, dict):
        return False

    for games in scoreboards.values():
        if not isinstance(games, list):
            continue
        for game in games:
            if is_live_scoreboard_game(game):
                return True

    return False


LIVE_GAME_LEAD_IN = datetime.timedelta(minutes=30)


def is_terminal_scoreboard_game(
    game: object, *, league: Optional[str] = None
) -> bool:
    """Return whether a game no longer needs score updates."""

    if not isinstance(game, dict):
        return False

    values: list[str] = []
    status = game.get("status")
    if isinstance(status, dict):
        if status.get("completed") is True:
            return True
        for key in (
            "detailedState",
            "abstractGameState",
            "gameStatus",
            "gameStatusText",
            "state",
            "gameState",
            "statusCode",
            "codedGameState",
        ):
            if status.get(key) is not None:
                values.append(str(status[key]))
        status_type = status.get("type")
        if isinstance(status_type, dict):
            if status_type.get("completed") is True:
                return True
            for key in ("state", "description", "detail", "shortDetail"):
                if status_type.get(key) is not None:
                    values.append(str(status_type[key]))
    elif status is not None:
        values.append(str(status))

    for key in (
        "gameStatusText",
        "gameStatus",
        "detailedState",
        "abstractGameState",
        "gameState",
        "gameScheduleState",
        "state",
        "statusType",
        "statusCode",
        "codedGameState",
    ):
        if game.get(key) is not None:
            values.append(str(game[key]))
    if game.get("completed") is True:
        return True

    normalized_values = {value.strip().upper() for value in values}
    terminal_codes = {
        "nba": {"3"},
        "nhl": {"4"},
        "mlb": {"F", "O"},
    }
    # With no league context, recognize every terminal encoding supported by
    # the scoreboard providers. Callers iterating scoreboards should supply the
    # league because numeric codes overlap (NHL uses NBA's final code for live).
    supported_codes = (
        terminal_codes.get(league, set())
        if league is not None
        else set().union(*terminal_codes.values())
    )
    status_text = " ".join(values).lower()
    return bool(normalized_values & supported_codes) or any(
        token in status_text
        for token in (
            "final",
            "postponed",
            "canceled",
            "cancelled",
            "forfeit",
        )
    ) or "POST" in normalized_values


def scoreboard_game_start(game: object) -> Optional[datetime.datetime]:
    """Return a scoreboard game's timezone-aware start time when available."""

    if not isinstance(game, dict):
        return None
    raw_start = (
        game.get("_start_local")
        or game.get("_event_date")
        or game.get("start_time")
        or game.get("date")
    )
    if isinstance(raw_start, datetime.datetime):
        parsed = raw_start
    elif isinstance(raw_start, str) and raw_start.strip():
        try:
            parsed = datetime.datetime.fromisoformat(raw_start.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=CENTRAL_TIME)
    return parsed


def scoreboards_in_live_window(
    scoreboards: object, *, now: Optional[datetime.datetime] = None
) -> bool:
    """Return true from shortly before kickoff until every game is terminal.

    A fixed expected-duration cutoff can strand a cached scheduled or in-progress
    score when a game is delayed or runs long.  Once kickoff has passed, continue
    refreshing that game until its provider explicitly marks it final (or another
    terminal state), regardless of which day of the week it is played.
    """

    if not isinstance(scoreboards, dict):
        return False
    current = now or datetime.datetime.now(datetime.UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=datetime.UTC)
    for league, games in scoreboards.items():
        if not isinstance(games, list):
            continue
        for game in games:
            if is_terminal_scoreboard_game(game, league=str(league)):
                continue
            if is_live_scoreboard_game(game):
                return True
            start = scoreboard_game_start(game)
            if (
                start is not None
                and start - LIVE_GAME_LEAD_IN <= current
            ):
                return True
    return False
