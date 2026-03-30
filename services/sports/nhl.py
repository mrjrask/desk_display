"""NHL scoreboard fetch + normalization service."""

from __future__ import annotations

import datetime as dt
import logging
import socket
import time
from typing import Any, Dict, Iterable, Optional

from config import CENTRAL_TIME
from services.http_client import NHL_HEADERS, get_session

REQUEST_TIMEOUT = 10
API_WEB_SCOREBOARD_URL = "https://api-web.nhle.com/v1/scoreboard/{date}"
API_WEB_SCOREBOARD_NOW_URL = "https://api-web.nhle.com/v1/scoreboard/now"
API_WEB_SCOREBOARD_PARAMS = {"site": "en_nhl"}

_SESSION = get_session()
STATSAPI_HOST = "statsapi.web.nhl.com"
API_WEB_HOST = "api-web.nhle.com"
_DNS_RETRY_INTERVAL = 600
_dns_block_until = 0.0


def scoreboard_date(now: dt.datetime | None = None) -> dt.date:
    now = now or dt.datetime.now(CENTRAL_TIME)
    cutoff = now.replace(hour=9, minute=30, second=0, microsecond=0)
    if now < cutoff:
        return (now - dt.timedelta(days=1)).date()
    return now.date()


def _timestamp_to_local(ts: str) -> Optional[dt.datetime]:
    if not ts:
        return None
    try:
        parsed = dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(CENTRAL_TIME)
    except Exception:
        return None


def _ordinal_from_number(num: Any) -> str:
    try:
        value = int(num)
    except Exception:
        if isinstance(num, str) and num.strip():
            return num.strip().upper()
        return ""

    if value <= 0:
        return ""
    if value == 1:
        return "1ST"
    if value == 2:
        return "2ND"
    if value == 3:
        return "3RD"
    return f"{value}TH"


def _normalize_team_name(team: Dict[str, Any]) -> Optional[str]:
    name = team.get("name") or team.get("teamName")
    if isinstance(name, str) and name.strip():
        return name.strip()
    if isinstance(name, dict):
        for key in ("default", "en", "name"):
            value = name.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    place = team.get("placeName")
    nickname = None
    if isinstance(place, dict):
        for key in ("default", "en"):
            value = place.get(key)
            if isinstance(value, str) and value.strip():
                nickname = value.strip()
                break
    elif isinstance(place, str) and place.strip():
        nickname = place.strip()
    if nickname:
        club = team.get("clubName") or team.get("commonName")
        if isinstance(club, dict):
            for key in ("default", "en"):
                value = club.get(key)
                if isinstance(value, str) and value.strip():
                    return f"{nickname} {value.strip()}".strip()
        if isinstance(club, str) and club.strip():
            return f"{nickname} {club.strip()}".strip()
    return None


def _map_api_web_team(team: Dict[str, Any]) -> Dict[str, Any]:
    team = team or {}
    abbr = None
    for key in ("abbrev", "triCode", "abbreviation", "teamTricode"):
        value = team.get(key)
        if isinstance(value, str) and value.strip():
            abbr = value.strip().upper()
            break
    team_id = team.get("id") or team.get("teamId")
    name = _normalize_team_name(team)
    mapped = {"team": {"id": team_id, "abbreviation": abbr, "triCode": abbr}}
    if name:
        mapped["team"]["name"] = name

    score = team.get("score")
    if score is None:
        score = team.get("goals")
    if score is not None:
        mapped["score"] = score

    sog = team.get("sog") or team.get("shotsOnGoal") or team.get("shots")
    if sog is not None:
        mapped["shotsOnGoal"] = sog

    return mapped


def _map_api_web_game(game: Dict[str, Any], day: dt.date) -> Dict[str, Any]:
    start_candidates = (
        game.get("startTimeUTC"),
        game.get("startTime"),
        game.get("gameDateTime"),
        game.get("gameDate"),
    )
    game_dt = None
    for candidate in start_candidates:
        if not candidate or not isinstance(candidate, str):
            continue
        text = candidate.strip()
        if not text:
            continue
        if not text.endswith("Z"):
            try:
                parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                continue
            parsed = parsed.astimezone(dt.timezone.utc)
            game_dt = parsed
            break
        try:
            parsed = dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
        game_dt = parsed.replace(tzinfo=dt.timezone.utc)
        break

    if game_dt is None:
        game_dt = dt.datetime.combine(day, dt.time(0, 0), tzinfo=dt.timezone.utc)

    clock = game.get("clock") or {}
    period = game.get("periodDescriptor") or {}
    outcome = game.get("gameOutcome") or {}

    time_remaining = None
    for key in ("timeRemaining", "time", "displayValue", "remaining", "label"):
        value = clock.get(key)
        if value:
            time_remaining = str(value).upper()
            break

    intermission = clock.get("inIntermission")
    period_ord = (
        period.get("ordinalNum")
        or period.get("ordinal")
        or _ordinal_from_number(period.get("number"))
        or _ordinal_from_number(period.get("period"))
    )
    if isinstance(period_ord, str):
        period_ord = period_ord.upper()

    has_shootout = False
    if isinstance(period.get("periodType"), str) and period["periodType"].strip().upper() == "SO":
        has_shootout = True
    if isinstance(outcome.get("lastPeriodType"), str) and outcome["lastPeriodType"].strip().upper() == "SO":
        has_shootout = True

    game_state = (game.get("gameState") or game.get("gameScheduleState") or "").upper()
    detailed_state = (game.get("gameStatus") or "").strip()
    abstract_state = ""
    status_code = ""
    if game_state in {"LIVE", "CRIT"}:
        abstract_state, status_code, detailed_state = "live", "3", detailed_state or "In Progress"
    elif game_state in {"FINAL", "OFF"}:
        abstract_state, status_code, detailed_state = "final", "4", detailed_state or "Final"
    elif game_state in {"FUT", "PRE", "SCHEDULED", "PREGAME"}:
        abstract_state, status_code, detailed_state = "preview", "1", detailed_state or "Scheduled"
    elif game_state in {"POSTP", "POSTPONED"}:
        abstract_state, status_code, detailed_state = "preview", "1", "Postponed"
    else:
        detailed_state = detailed_state or game_state or "Scheduled"

    linescore = {}
    if period_ord:
        linescore["currentPeriodOrdinal"] = period_ord
    if time_remaining:
        linescore["currentPeriodTimeRemaining"] = time_remaining
    if has_shootout:
        linescore["hasShootout"] = True
    if intermission is not None:
        linescore["intermissionInfo"] = {"inIntermission": bool(intermission)}

    mapped = {
        "gamePk": game.get("id") or game.get("gamePk") or game.get("gameId"),
        "gameDate": game_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": {"abstractGameState": abstract_state, "detailedState": detailed_state},
        "teams": {
            "away": _map_api_web_team(game.get("awayTeam") or game.get("away")),
            "home": _map_api_web_team(game.get("homeTeam") or game.get("home")),
        },
    }
    if status_code:
        mapped["status"]["statusCode"] = status_code
    if linescore:
        mapped["linescore"] = linescore
    return mapped


def _extract_api_web_games(data: Dict[str, Any], day: dt.date) -> list[Dict[str, Any]]:
    def _normalize_date(value: Any) -> Optional[str]:
        if not value:
            return None
        if isinstance(value, dt.date):
            return value.isoformat()
        text = str(value).strip()
        if not text:
            return None
        if "T" in text:
            text = text.split("T", 1)[0]
        return text

    day_iso = day.isoformat()
    games: list[Dict[str, Any]] = []

    def _append_from(container: Iterable[Any]):
        for item in container or []:
            if isinstance(item, dict):
                games.append(item)

    direct = data.get("games")
    if isinstance(direct, list):
        _append_from(direct)
    scoreboard = data.get("scoreboard")
    if isinstance(scoreboard, dict) and isinstance(scoreboard.get("games"), list):
        _append_from(scoreboard.get("games"))
    for key in ("gameWeek", "gamesByDate", "gamesByDay", "gamesByDateV2"):
        buckets = data.get(key)
        if not isinstance(buckets, list):
            continue
        for bucket in buckets:
            if not isinstance(bucket, dict):
                continue
            bucket_date = (
                _normalize_date(bucket.get("date"))
                or _normalize_date(bucket.get("gameDate"))
                or _normalize_date(bucket.get("day"))
            )
            if bucket_date and bucket_date != day_iso:
                continue
            bucket_games = bucket.get("games")
            if isinstance(bucket_games, list):
                _append_from(bucket_games)

    seen_ids: set[Any] = set()
    filtered: list[Dict[str, Any]] = []
    for game in games:
        game_id = game.get("id") or game.get("gamePk") or game.get("gameId")
        key = game_id or id(game)
        if key in seen_ids:
            continue
        seen_ids.add(key)
        filtered.append(game)
    return filtered


def _hydrate_games(raw_games: Iterable[dict]) -> list[dict]:
    games: list[dict] = []
    for game in raw_games or []:
        if not isinstance(game, dict):
            continue
        enriched = game.copy()
        start_local = _timestamp_to_local(enriched.get("gameDate"))
        if start_local:
            enriched["_start_local"] = start_local
            enriched["_start_sort"] = start_local.timestamp()
        else:
            enriched["_start_sort"] = float("inf")
        games.append(enriched)
    games.sort(key=lambda g: (g.get("_start_sort", float("inf")), g.get("gamePk", 0)))
    return games


def prepare_scoreboard_data(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return _hydrate_games(payload)
    if isinstance(payload, dict) and isinstance(payload.get("games"), list):
        return _hydrate_games(payload.get("games"))
    return []


def _fetch_games_api_web(day: dt.date) -> list[dict]:
    urls = [API_WEB_SCOREBOARD_URL.format(date=day.isoformat()), API_WEB_SCOREBOARD_NOW_URL]
    for url in urls:
        try:
            response = _SESSION.get(
                url,
                timeout=REQUEST_TIMEOUT,
                headers=NHL_HEADERS,
                params=API_WEB_SCOREBOARD_PARAMS,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logging.error("Failed to fetch NHL scoreboard fallback %s: %s", url, exc)
            continue

        games_payload = _extract_api_web_games(data, day)
        if not games_payload:
            continue

        mapped_games = []
        for game in games_payload:
            try:
                mapped_games.append(_map_api_web_game(game, day))
            except Exception as exc:
                logging.debug("Skipping api-web game due to error: %s", exc)
        if mapped_games:
            return _hydrate_games(mapped_games)
    return []


def _statsapi_available() -> bool:
    global _dns_block_until
    now = time.time()
    if now < _dns_block_until:
        return False
    try:
        socket.getaddrinfo(STATSAPI_HOST, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        logging.debug("NHL statsapi DNS lookup failed: %s", exc)
        _dns_block_until = now + _DNS_RETRY_INTERVAL
        return False
    except Exception as exc:
        logging.debug("Unexpected error checking NHL statsapi DNS: %s", exc)
    else:
        _dns_block_until = 0.0
        return True
    return True


def _fetch_games_for_date(day: dt.date) -> list[dict]:
    if not _statsapi_available():
        logging.debug("Using api-web NHL scoreboard endpoint for %s (statsapi DNS failure)", day)
        return _fetch_games_api_web(day)

    stats_url = (
        "https://statsapi.web.nhl.com/api/v1/schedule"
        f"?date={day.isoformat()}&expand=schedule.linescore,schedule.teams"
    )
    data: Optional[Dict[str, Any]] = None
    try:
        response = _SESSION.get(stats_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        logging.error("Failed to fetch NHL schedule: %s", exc)

    games: list[dict] = []
    if data:
        for day_info in data.get("dates", []) or []:
            games.extend(day_info.get("games", []) or [])
    if games:
        return _hydrate_games(games)

    logging.info("Falling back to api-web NHL scoreboard endpoint for %s", day)
    return _fetch_games_api_web(day)


def fetch_scoreboard(*, day: dt.date | None = None, now: dt.datetime | None = None) -> list[dict]:
    target_day = day or scoreboard_date(now)
    games = _fetch_games_for_date(target_day)
    return games if isinstance(games, list) else []
