"""Free, no-key NFL scoreboard providers and payload normalization."""

from __future__ import annotations

import csv
import datetime as dt
import io
import logging
import time
import urllib.parse
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any
from zoneinfo import ZoneInfo

from config import CENTRAL_TIME
from services.http_client import get_session

ESPN_SITE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
ESPN_CDN_URL = "https://cdn.espn.com/core/nfl/schedule"
NFLVERSE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
NFLVERSE_TIME_ZONE = ZoneInfo("America/New_York")
REQUEST_TIMEOUT = 10
FETCH_CACHE_TTL_SECONDS = 60

# ``fetch_range`` uses the scoreboard endpoints (rather than the schedule
# endpoint used by ``normalize_espn_cdn``) and the nflverse release asset.
# Keep these provider-specific constants separate from the public URLs above.
_SITE_SCOREBOARD_URL = ESPN_SITE_URL
_CDN_SCOREBOARD_URL = "https://cdn.espn.com/core/nfl/scoreboard"
_NFLVERSE_SCHEDULE_URL = (
    "https://github.com/nflverse/nfldata/releases/download/schedules/games.csv"
)
_NFLVERSE_SCHEDULE_CACHE_KEY = ("nflverse", "complete_schedule")

_SESSION = get_session("nfl")
_RANGE_CACHE: dict[tuple[dt.date, dt.date], tuple[float, list[dict[str, Any]]]] = {}


class InvalidProviderPayload(ValueError):
    """Raised when a provider responds, but not with its documented container."""


@dataclass
class WeeklyResult:
    """Outcome of assembling an NFL week from independently fetched dates."""

    games: list[dict[str, Any]] = field(default_factory=list)
    successful_dates: int = 0
    failed_dates: int = 0
    bye_teams: list[str] | None = None
    stale: bool = False


def _event_identity(event: dict[str, Any]) -> str | None:
    value = event.get("id") or event.get("uid")
    return str(value) if value is not None else None


def _event_start(event: dict[str, Any]) -> float:
    value = event.get("date")
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return float("inf")


def fetch_week_dates(
    dates: Iterable[dt.date], *, session: Any = None
) -> WeeklyResult:
    """Fetch each ESPN scoreboard date once and report aggregate completeness."""

    session = session or _SESSION
    unique: dict[str, dict[str, Any]] = {}
    successful_dates = failed_dates = 0
    bye_teams: list[str] | None = None
    for day in dates:
        try:
            payload = _request_json(session, ESPN_SITE_URL, dates=f"{day:%Y%m%d}")
            events = payload.get("events")
            if not isinstance(events, list):
                raise InvalidProviderPayload("ESPN Site payload has no events list")
            if not all(isinstance(event, dict) for event in events):
                raise InvalidProviderPayload("ESPN Site events contained a non-object")
            successful_dates += 1
            if bye_teams is None and isinstance(payload.get("byeTeams"), list):
                bye_teams = [str(team) for team in payload["byeTeams"]]
            for event in events:
                identity = _event_identity(event)
                if identity is not None:
                    unique.setdefault(identity, event)
        except Exception as exc:
            failed_dates += 1
            logging.warning("NFL date %s failed: %s", day.isoformat(), exc)

    events = sorted(unique.values(), key=lambda event: (_event_start(event), _event_identity(event) or ""))
    return WeeklyResult(
        games=[game for event in events if (game := _event_to_game(event)) is not None],
        successful_dates=successful_dates,
        failed_dates=failed_dates,
        bye_teams=bye_teams,
    )


def fetch_espn_week(
    start: dt.date, end: dt.date, *, session: Any = None
) -> WeeklyResult:
    """Fetch ESPN's explicit whole-week feed without accepting an invalid shape."""

    session = session or _SESSION
    try:
        payload = _request_json(
            session, ESPN_SITE_URL, limit=100, dates=f"{start:%Y%m%d}-{end:%Y%m%d}"
        )
        games = _games_in_range(normalize_espn_site(payload), start, end)
    except Exception as exc:
        logging.warning("NFL whole-week fallback failed: %s", exc)
        return WeeklyResult(failed_dates=1)
    return WeeklyResult(games=games, successful_dates=1)


def _request_json(session: Any, url: str, **params: Any) -> dict[str, Any]:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise InvalidProviderPayload(f"{url} returned a non-object payload")
    return payload


def _event_to_game(event: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a Site API/CDN event into the single renderer contract."""
    competitions = event.get("competitions")
    if (
        not isinstance(competitions, list)
        or not competitions
        or not isinstance(competitions[0], dict)
    ):
        return None
    competition = competitions[0]
    competitors = competition.get("competitors")
    status = competition.get("status") or event.get("status")
    if not isinstance(competitors, list) or not isinstance(status, dict):
        return None
    event_id = event.get("id") or competition.get("id")
    start = event.get("date") or competition.get("date")
    if event_id is None or not isinstance(start, str):
        return None

    normalized_competitors: list[dict[str, Any]] = []
    scores: dict[str, Any] = {}
    for raw in competitors:
        if not isinstance(raw, dict) or raw.get("homeAway") not in {"home", "away"}:
            continue
        side = dict(raw)
        score = side.get("score")
        if isinstance(score, dict):
            score = score.get("displayValue") or score.get("value")
        side["score"] = score
        normalized_competitors.append(side)
        scores[str(side["homeAway"])] = score
    if len(normalized_competitors) < 2:
        return None
    return {
        "event_id": str(event_id),
        "id": str(event_id),
        "start_time": start,
        "date": start,
        "event_name": event.get("name") or event.get("shortName") or "NFL game",
        "name": event.get("name") or event.get("shortName") or "NFL game",
        "shortName": event.get("shortName") or event.get("name") or "NFL game",
        "competitors": normalized_competitors,
        "scores": scores,
        "status": status,
        "_event_date": start,
        "_event_name": event.get("name"),
        "_event_short_name": event.get("shortName"),
    }


def normalize_espn_site(payload: dict[str, Any]) -> list[dict[str, Any]]:
    events = payload.get("events")
    if not isinstance(events, list):
        raise InvalidProviderPayload("ESPN Site payload has no events list")
    return _normalize_events(events)


def normalize_espn_cdn(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize CDN schedule payloads (and its occasional events variant)."""
    if isinstance(payload.get("events"), list):
        return _normalize_events(payload["events"])
    schedule = (payload.get("content") or {}).get("schedule")
    if not isinstance(schedule, dict):
        raise InvalidProviderPayload("ESPN CDN payload has no schedule")
    events: list[dict[str, Any]] = []
    for day in schedule.values():
        if isinstance(day, dict):
            # The CDN currently calls these ``games`` while older/alternate
            # schedule responses call them ``events``. Support both shapes.
            day_games = day.get("games")
            if not isinstance(day_games, list):
                day_games = day.get("events")
            if isinstance(day_games, list):
                events.extend(day_games)
    return _normalize_events(events)


def _normalize_events(events: Iterable[Any]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for event in events:
        game = _event_to_game(event) if isinstance(event, dict) else None
        if game is not None:
            unique.setdefault(game["id"], game)
    return list(unique.values())


def _games_in_range(
    games: Iterable[dict[str, Any]], start: dt.date, end: dt.date
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for game in games:
        value = game.get("date")
        try:
            event_day = (
                dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                .astimezone(CENTRAL_TIME)
                .date()
            )
        except (TypeError, ValueError):
            continue
        if start <= event_day <= end:
            filtered.append(game)
    return filtered


def _nflverse_games(session: Any, start: dt.date, end: dt.date) -> list[dict[str, Any]]:
    response = session.get(NFLVERSE_URL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    rows = csv.DictReader(io.StringIO(response.text))
    required_columns = {"game_id", "gameday", "away_team", "home_team", "result"}
    if not rows.fieldnames or not required_columns.issubset(rows.fieldnames):
        raise InvalidProviderPayload("nflverse games CSV has unexpected columns")
    games: list[dict[str, Any]] = []
    for row in rows:
        try:
            day = dt.date.fromisoformat(row["gameday"])
            local_start = dt.datetime.combine(
                day,
                dt.time.fromisoformat(row.get("gametime") or "00:00"),
                tzinfo=NFLVERSE_TIME_ZONE,
            )
        except (TypeError, ValueError):
            continue
        if not start <= day <= end:
            continue
        # nflverse is intentionally not a live provider. Its result column is
        # the completion signal; ignore any transient scores until it is set.
        completed = bool((row.get("result") or "").strip())
        away_score = (row.get("away_score") or None) if completed else None
        home_score = (row.get("home_score") or None) if completed else None
        start_time = local_start.astimezone(dt.UTC).isoformat().replace("+00:00", "Z")
        event = {
            "id": row["game_id"],
            "date": start_time,
            "name": f"{row['away_team']} at {row['home_team']}",
            "shortName": f"{row['away_team']} @ {row['home_team']}",
            "competitions": [{
                "id": row["game_id"],
                "competitors": [
                    {
                        "homeAway": "away",
                        "team": {"abbreviation": row["away_team"]},
                        "score": away_score,
                    },
                    {
                        "homeAway": "home",
                        "team": {"abbreviation": row["home_team"]},
                        "score": home_score,
                    },
                ],
                "status": {"type": {
                    "state": "post" if completed else "pre",
                    "completed": completed,
                    "description": "Final" if completed else "Scheduled",
                    "shortDetail": "Final" if completed else "Scheduled",
                }},
            }],
        }
        game = _event_to_game(event)
        if game:
            games.append(game)
    return list({game["id"]: game for game in games}.values())


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
    payload = response.json()
    if url.startswith(_CDN_SCOREBOARD_URL):
        return normalize_espn_cdn(payload)
    return _games_from_events(_events_from_payload(payload))


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


def _fetch_nflverse_schedule(
    *,
    session: Any,
    cache: MutableMapping[tuple[object, ...], tuple[float, list[dict]]],
) -> list[dict]:
    """Download and parse the complete nflverse schedule, reusing it across ranges."""

    now = time.monotonic()
    cached = cache.get(_NFLVERSE_SCHEDULE_CACHE_KEY)
    if cached and now - cached[0] < FETCH_CACHE_TTL_SECONDS:
        # Use an idle TTL so a long discovery scan cannot expire the schedule
        # while it is still making progress through consecutive weekly ranges.
        cache[_NFLVERSE_SCHEDULE_CACHE_KEY] = (now, cached[1])
        return cached[1]

    response = session.get(_NFLVERSE_SCHEDULE_URL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    games: list[dict] = []
    for row in csv.DictReader(io.StringIO(response.text)):
        try:
            game_day = dt.date.fromisoformat(row.get("gameday") or "")
        except ValueError:
            continue
        game_time = (row.get("gametime") or "00:00").strip()
        try:
            local_start = dt.datetime.combine(
                game_day,
                dt.time.fromisoformat(game_time),
                tzinfo=NFLVERSE_TIME_ZONE,
            )
        except ValueError:
            continue
        event_date = (
            local_start.astimezone(dt.UTC).isoformat().replace("+00:00", "Z")
        )
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
    cache[_NFLVERSE_SCHEDULE_CACHE_KEY] = (now, games)
    return games


def _fetch_nflverse(
    start: dt.date,
    end: dt.date,
    *,
    session: Any,
    cache: MutableMapping[tuple[object, ...], tuple[float, list[dict]]],
) -> list[dict]:
    schedule = _fetch_nflverse_schedule(session=session, cache=cache)
    return [
        game
        for game in schedule
        if start.isoformat() <= str(game.get("_event_date", ""))[:10] <= end.isoformat()
    ]


def fetch_range(
    start: dt.date,
    end: dt.date,
    *,
    session: Any = None,
    cache: MutableMapping[tuple[object, ...], tuple[float, list[dict]]] | None = None,
    failed_providers: set[str] | None = None,
) -> list[dict]:
    """Fetch an inclusive NFL range with ESPN CDN and nflverse fallbacks.

    When ``failed_providers`` is supplied, fallback providers that raise are
    added to it and skipped on later calls.  Discovery scans can therefore
    share the set across range windows instead of repeatedly waiting on an
    unavailable fallback.  The primary Site provider is retried in every
    window, and empty responses remain eligible because an empty week is a
    valid response rather than a provider failure.
    """

    if end < start:
        return []
    session = session or _SESSION
    cache = _RANGE_CACHE if cache is None else cache
    cache_key = (start, end, "nfl_scoreboard_range")
    legacy_cache_key = (start, f"nfl_providers:{end.isoformat()}")
    now = time.monotonic()
    cached = cache.get(cache_key) or cache.get(legacy_cache_key)
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
        (
            "nflverse",
            lambda: _fetch_nflverse(start, end, session=session, cache=cache),
        ),
    )
    successful_response = False
    for provider_name, fetch in providers:
        if failed_providers is not None and provider_name in failed_providers:
            continue
        try:
            games = fetch()
        except Exception as exc:
            logging.warning("Failed to fetch NFL scoreboard from %s: %s", provider_name, exc)
            if failed_providers is not None and provider_name != "ESPN Site":
                failed_providers.add(provider_name)
            continue
        successful_response = True
        if games:
            cache[cache_key] = (now, games)
            return games
        logging.info("NFL scoreboard from %s was empty; trying fallback", provider_name)
    if successful_response:
        cache[cache_key] = (now, [])
    elif cached:
        logging.warning("All NFL providers failed; retaining stale cached range")
        return cached[1]
    return []


def fetch_week_scoreboard(*, now: dt.datetime | None = None) -> list[dict]:
    from screens.nfl_scoreboard import _fetch_games_for_week
    return _fetch_games_for_week(now)


def fetch_week_scoreboard_result(*, now: dt.datetime | None = None) -> WeeklyResult:
    """Return the selected display week together with its freshness metadata."""

    from screens import nfl_scoreboard

    games = nfl_scoreboard._fetch_games_for_week(now)
    result = nfl_scoreboard._LAST_WEEKLY_RESULT
    if isinstance(result, WeeklyResult) and result.games == games:
        return result
    return WeeklyResult(games=games)


def fetch_next_scoreboard(*, start_date: dt.date, max_days: int = 370) -> list[dict]:
    from screens.nfl_scoreboard import _fetch_next_games
    return _fetch_next_games(start_date, max_days=max_days)


def fetch_scoreboard(*, now: dt.datetime | None = None) -> list[dict]:
    current_now = now or dt.datetime.now(CENTRAL_TIME)
    games = fetch_week_scoreboard(now=current_now)
    return games or fetch_next_scoreboard(start_date=current_now.date())
