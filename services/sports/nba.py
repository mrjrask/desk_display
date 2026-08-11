"""NBA scoreboard fetch service."""

from __future__ import annotations

import datetime
import logging
from collections.abc import Iterable
from typing import Any, Dict, Optional

from config import CENTRAL_TIME
from services.http_client import get_session
from services.sports.scoreboard_window import (
    before_scoreboard_update,
    compose_pre_update_scoreboard,
)

REQUEST_TIMEOUT = 10
_SESSION = get_session()
_NBA_HEADERS = {
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}
_NBA_SCOREBOARD_BASES: tuple[tuple[str, bool], ...] = (
    ("https://cdn.nba.com/static/json/liveData/scoreboard", True),
    ("https://nba-prod-us-east-1-media.s3.amazonaws.com/json/liveData/scoreboard", False),
)
_FORBIDDEN_CACHE_TTL = datetime.timedelta(minutes=30)
_last_forbidden: Optional[datetime.datetime] = None
_nba_cdn_fallback_notice_at_by_day: dict[datetime.date, datetime.datetime] = {}


def _scoreboard_date(now: Optional[datetime.datetime] = None) -> datetime.date:
    now = now or datetime.datetime.now(CENTRAL_TIME)
    cutoff = now.replace(hour=10, minute=10, second=0, microsecond=0)
    if now < cutoff:
        return (now - datetime.timedelta(days=1)).date()
    return now.date()


def _ordinal_from_number(num: Any, *, is_overtime: bool = False) -> str:
    try:
        value = int(num)
    except Exception:
        if isinstance(num, str) and num.strip():
            return num.strip().upper()
        return ""
    if value <= 0:
        return ""
    if is_overtime:
        if value <= 1:
            return "OT"
        return f"{value}OT"
    if value == 1:
        return "1ST"
    if value == 2:
        return "2ND"
    if value == 3:
        return "3RD"
    if value == 4:
        return "4TH"
    return f"{value}TH"


def _normalize_clock(clock: Any) -> str:
    if not clock:
        return ""
    if isinstance(clock, (int, float)):
        minutes = int(clock) // 60
        seconds = int(clock) % 60
        return f"{minutes}:{seconds:02d}"
    text = str(clock).strip()
    if not text:
        return ""
    if text.startswith("PT"):
        minutes = 0
        seconds = 0
        rem = text[2:]
        try:
            if "M" in rem:
                min_part, rem = rem.split("M", 1)
                minutes = int(float(min_part))
            if "S" in rem:
                sec_part = rem.split("S", 1)[0]
                seconds = int(float(sec_part))
        except Exception:
            return text
        return f"{minutes}:{seconds:02d}"
    return text.upper()

def _timestamp_to_local(ts: str) -> Optional[datetime.datetime]:
    if not ts:
        return None
    text = str(ts).strip()
    if not text:
        return None
    fmt_candidates = ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"]
    for fmt in fmt_candidates:
        try:
            dt = datetime.datetime.strptime(text, fmt)
        except Exception:
            continue
        else:
            dt = dt.replace(tzinfo=datetime.UTC)
            return dt.astimezone(CENTRAL_TIME)
    try:
        dt = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.UTC)
    return dt.astimezone(CENTRAL_TIME)


def _hydrate_games(raw_games: Iterable[dict]) -> list[dict]:
    games: list[dict] = []
    for game in raw_games:
        game = game or {}
        start_local = game.get("_start_local")
        if not isinstance(start_local, datetime.datetime):
            start_local = _timestamp_to_local(game.get("gameDate"))
            if not start_local:
                start_local = _timestamp_to_local(game.get("startTimeUTC"))
            if not start_local:
                start_local = _timestamp_to_local(game.get("gameTimeUTC"))
            if start_local:
                game["_start_local"] = start_local
        if isinstance(start_local, datetime.datetime):
            game["_start_sort"] = start_local.timestamp()
        else:
            game["_start_sort"] = float("inf")
        games.append(game)
    games.sort(key=lambda g: (g.get("_start_sort", float("inf")), g.get("gamePk", g.get("gameId", 0))))
    return games


def _parse_period_info(game: dict[str, Any]) -> tuple[Optional[int], str, Optional[int]]:
    period_info = game.get("period")
    period_type = ""
    final_period = None
    number: Optional[int] = None

    if isinstance(period_info, dict):
        for key in ("current", "number", "period", "sequence"):
            value = period_info.get(key)
            if value not in (None, ""):
                try:
                    number = int(value)
                    break
                except Exception:
                    pass
        period_type = str(period_info.get("type") or period_info.get("periodType") or "").upper()
    elif period_info not in (None, ""):
        try:
            number = int(period_info)
        except Exception:
            pass

    descriptor = game.get("periodDescriptor") or {}
    if isinstance(descriptor, dict):
        if number is None:
            for key in ("period", "number"):
                value = descriptor.get(key)
                if value not in (None, ""):
                    try:
                        number = int(value)
                        break
                    except Exception:
                        pass
        if not period_type:
            period_type = str(descriptor.get("type") or descriptor.get("periodType") or "").upper()
        final_period_val = descriptor.get("maxRegular") or descriptor.get("max") or descriptor.get("total")
        if final_period_val not in (None, ""):
            try:
                final_period = int(final_period_val)
            except Exception:
                pass

    if final_period is None:
        final_period = number

    return number, period_type, final_period


def _map_team(team: dict[str, Any]) -> dict[str, Any]:
    team = team or {}
    abbr = ""
    for key in ("teamTricode", "triCode", "tricode", "abbreviation", "abbr"):
        value = team.get(key)
        if isinstance(value, str) and value.strip():
            abbr = value.strip().upper()
            break
    name_parts = []
    for key in ("teamCity", "city"):
        value = team.get(key)
        if isinstance(value, str) and value.strip():
            name_parts.append(value.strip())
            break
    for key in ("teamName", "nickname", "name"):
        value = team.get(key)
        if isinstance(value, str) and value.strip():
            if name_parts:
                name_parts.append(value.strip())
            else:
                name_parts.append(value.strip())
            break
    full_name = " ".join(name_parts).strip()

    mapped: dict[str, Any] = {"team": {}}
    if abbr:
        mapped["team"]["abbreviation"] = abbr
        mapped["team"]["triCode"] = abbr
    if full_name:
        mapped["team"]["name"] = full_name
    team_id = team.get("teamId") or team.get("id")
    if team_id not in (None, ""):
        mapped["team"]["id"] = team_id

    score = team.get("score")
    if score not in (None, ""):
        mapped["score"] = score

    return mapped


def _map_game(game: dict[str, Any]) -> dict[str, Any]:
    game = game or {}
    status_code_raw = game.get("gameStatus") or game.get("statusNum")
    status_code = ""
    if status_code_raw not in (None, ""):
        try:
            status_code = str(int(status_code_raw))
        except Exception:
            status_code = str(status_code_raw)

    status_text = (game.get("gameStatusText") or game.get("statusText") or "").strip()
    if not status_text:
        status_text = {"1": "Scheduled", "2": "In Progress", "3": "Final"}.get(status_code, "")

    abstract = ""
    if status_code == "3":
        abstract = "final"
    elif status_code == "2":
        abstract = "live"
    elif status_code == "1":
        abstract = "preview"

    game_date = game.get("gameTimeUTC") or game.get("gameTime") or game.get("startTimeUTC") or game.get("gameDate")
    mapped: dict[str, Any] = {
        "gamePk": game.get("gameId") or game.get("id") or game.get("gameCode"),
        "gameDate": game_date,
        "status": {
            "statusCode": status_code,
            "detailedState": status_text,
        },
        "teams": {
            "away": _map_team(game.get("awayTeam") or game.get("away")),
            "home": _map_team(game.get("homeTeam") or game.get("home")),
        },
    }
    for source_key, target_key in (
        ("seasonType", "seasonType"),
        ("seasonStage", "seasonStage"),
        ("gameType", "gameType"),
    ):
        value = game.get(source_key)
        if value not in (None, ""):
            mapped[target_key] = value
    if abstract:
        mapped["status"]["abstractGameState"] = abstract

    period_number, period_type, final_period = _parse_period_info(game)
    clock = _normalize_clock(game.get("gameClock") or game.get("clock"))

    linescore: dict[str, Any] = {}
    if period_number is not None:
        is_ot = False
        if period_number > 4 or period_type in {"OT", "OVERTIME"}:
            is_ot = True
            ot_number = period_number - 4 if period_number > 4 else period_number
            linescore["currentPeriodOrdinal"] = _ordinal_from_number(ot_number, is_overtime=True)
        else:
            linescore["currentPeriodOrdinal"] = _ordinal_from_number(period_number)
        linescore["finalPeriod"] = period_number if status_code == "3" else final_period or period_number
    elif final_period is not None:
        linescore["finalPeriod"] = final_period

    if clock:
        linescore["currentPeriodTimeRemaining"] = clock

    if linescore:
        mapped["linescore"] = linescore

    start_local = _timestamp_to_local(mapped.get("gameDate"))
    if start_local:
        mapped["_start_local"] = start_local
        mapped["_start_sort"] = start_local.timestamp()

    return mapped


def _espn_status_code(status_type: dict[str, Any]) -> str:
    status_type = status_type or {}
    raw = status_type.get("id") or status_type.get("state") or status_type.get("name")
    code = ""
    if raw not in (None, ""):
        try:
            code = str(int(raw))
        except Exception:
            code = str(raw)

    state = str(status_type.get("state") or "").lower()
    if state.startswith("pre"):
        return "1"
    if state.startswith("in"):
        return "2"
    if state.startswith("post"):
        return "3"
    if status_type.get("completed"):
        return "3"
    return code


def _espn_status_text(status: dict[str, Any]) -> str:
    status = status or {}
    status_type = status.get("type") or {}
    for key in ("shortDetail", "detail", "description", "name"):
        value = status_type.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _espn_status_abstract(status_code: str, status: dict[str, Any]) -> str:
    status = status or {}
    status_type = status.get("type") or {}
    state = str(status_type.get("state") or "").lower()
    if status_type.get("completed") or state.startswith("post"):
        return "final"
    if state.startswith("in"):
        return "live"
    if state.startswith("pre"):
        return "preview"
    return {"3": "final", "2": "live", "1": "preview"}.get(status_code, "")


def _map_espn_competitor(comp: dict[str, Any]) -> dict[str, Any]:
    comp = comp or {}
    team = comp.get("team") or {}
    abbr = team.get("abbreviation") or comp.get("teamAbbreviation") or ""
    if isinstance(abbr, str):
        abbr = abbr.strip().upper()
    else:
        abbr = ""

    location = team.get("location") or team.get("displayName") or ""
    nickname = team.get("name") or team.get("shortDisplayName") or ""
    if not location and isinstance(team.get("displayName"), str):
        location = team["displayName"]
    if not nickname and isinstance(team.get("displayName"), str):
        parts = team["displayName"].split()
        if len(parts) > 1:
            nickname = parts[-1]
            location = " ".join(parts[:-1])
        else:
            nickname = team["displayName"]

    mapped: dict[str, Any] = {
        "teamTricode": abbr,
        "teamCity": location,
        "teamName": nickname,
        "teamId": team.get("id") or comp.get("id"),
        "score": comp.get("score"),
    }
    return mapped


def _map_espn_game(event: dict[str, Any], competition: dict[str, Any], day: datetime.date) -> Optional[dict[str, Any]]:
    competition = competition or {}
    event_date = competition.get("date") or event.get("date")
    if event_date:
        start_local = _timestamp_to_local(event_date)
        if start_local and start_local.date() != day:
            return None

    status = competition.get("status") or event.get("status") or {}
    status_code = _espn_status_code(status.get("type") or {})
    status_text = _espn_status_text(status)
    abstract = _espn_status_abstract(status_code, status)

    period_number = status.get("period")
    try:
        period_number = int(period_number)
    except Exception:
        period_number = None

    period_descriptor: dict[str, Any] = {}
    if period_number is not None:
        period_descriptor["period"] = period_number
        period_descriptor["maxRegular"] = 4
        period_descriptor["total"] = period_number
        if period_number > 4:
            period_descriptor["type"] = "OT"

    clock = status.get("displayClock") or status.get("clock")
    if clock not in (None, ""):
        clock = str(clock)
    else:
        clock = ""

    home_team: dict[str, Any] = {}
    away_team: dict[str, Any] = {}
    for competitor in competition.get("competitors") or []:
        mapped = _map_espn_competitor(competitor)
        side = (competitor.get("homeAway") or "").lower()
        if side == "home":
            home_team = mapped
        elif side == "away" or not away_team:
            away_team = mapped
        else:
            home_team = home_team or mapped

    game_id = competition.get("id") or event.get("id")
    season_info = event.get("season") or {}
    season_type = season_info.get("type")
    season_slug = season_info.get("slug")
    mapped_game: dict[str, Any] = {
        "gameId": game_id,
        "id": game_id,
        "gameCode": event.get("uid"),
        "gameDate": event_date,
        "gameTimeUTC": event_date,
        "startTimeUTC": event_date,
        "gameStatus": status_code,
        "statusNum": status_code,
        "gameStatusText": status_text,
        "statusText": status_text,
        "gameClock": clock,
        "period": {"number": period_number} if period_number is not None else None,
        "periodDescriptor": period_descriptor or None,
        "awayTeam": away_team,
        "homeTeam": home_team,
        "seasonType": season_type,
        "seasonStage": season_slug,
    }
    if abstract:
        mapped_game["status"] = {
            "statusCode": status_code,
            "detailedState": status_text,
            "abstractGameState": abstract,
        }
    if event_date:
        start_local = _timestamp_to_local(event_date)
        if start_local:
            mapped_game["_start_local"] = start_local
    return mapped_game


def _fetch_games_from_espn(day: datetime.date) -> Optional[list[dict]]:
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        f"?dates={day.strftime('%Y%m%d')}"
    )
    try:
        response = _SESSION.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        logging.error("Failed to fetch NBA scoreboard from ESPN for %s: %s", day, exc)
        return None

    raw_games: list[dict] = []
    for event in data.get("events") or []:
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        mapped = _map_espn_game(event, competitions[0], day)
        if mapped:
            raw_games.append(mapped)

    mapped_games = [_map_game(game) for game in raw_games]
    return _hydrate_games(mapped_games)


def _log_nba_cdn_fallback(day: datetime.date) -> None:
    """Log one fallback notice per day within the forbidden cache window."""

    now = datetime.datetime.now()
    last_notice_at = _nba_cdn_fallback_notice_at_by_day.get(day)
    if last_notice_at and (now - last_notice_at) < _FORBIDDEN_CACHE_TTL:
        return

    logging.info(
        "Using NBA CDN scoreboard fallback while ESPN data is unavailable (first encountered for %s)",
        day,
    )
    _nba_cdn_fallback_notice_at_by_day[day] = now


def _reset_nba_cdn_fallback_notice() -> None:
    """Drop stale fallback notices so dates can be logged again later."""

    now = datetime.datetime.now()
    stale_days = [
        day
        for day, notice_at in _nba_cdn_fallback_notice_at_by_day.items()
        if (now - notice_at) >= _FORBIDDEN_CACHE_TTL
    ]
    for day in stale_days:
        _nba_cdn_fallback_notice_at_by_day.pop(day, None)


def _fetch_games_from_nba_cdn(day: datetime.date) -> list[dict]:
    def _load_json(url: str, *, respect_forbidden_cache: bool = True) -> Optional[dict[str, Any]]:
        global _last_forbidden

        if (
            respect_forbidden_cache
            and _last_forbidden
            and (datetime.datetime.now() - _last_forbidden) < _FORBIDDEN_CACHE_TTL
        ):
            logging.debug(
                "Skipping NBA scoreboard fetch for %s due to recent 403", url
            )
            return None

        try:
            response = _SESSION.get(url, timeout=REQUEST_TIMEOUT, headers=_NBA_HEADERS)
            if response.status_code == 404:
                return None
            if response.status_code == 403:
                now = datetime.datetime.now()
                if not _last_forbidden or (now - _last_forbidden) >= _FORBIDDEN_CACHE_TTL:
                    logging.debug(
                        "NBA scoreboard returned HTTP 403 for %s; suppressing further attempts for %s",
                        url,
                        _FORBIDDEN_CACHE_TTL,
                    )
                _last_forbidden = now
                return None
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logging.error("Failed to fetch NBA scoreboard from %s: %s", url, exc)
            return None

    data: Optional[dict[str, Any]] = None
    source_base: Optional[str] = None
    today = datetime.date.today()
    for base, respect_cache in _NBA_SCOREBOARD_BASES:
        date_url = f"{base}/scoreboard_{day.strftime('%Y%m%d')}.json"
        data = _load_json(date_url, respect_forbidden_cache=respect_cache)
        if isinstance(data, dict):
            source_base = base
            break
        if day == today:
            today_url = f"{base}/todaysScoreboard.json"
            data = _load_json(today_url, respect_forbidden_cache=respect_cache)
            if isinstance(data, dict):
                source_base = base
                break

    if source_base and source_base != _NBA_SCOREBOARD_BASES[0][0]:
        logging.info(
            "NBA scoreboard fetched successfully from alternate base %s", source_base
        )

    if not isinstance(data, dict):
        return []

    _reset_nba_cdn_fallback_notice()
    games_raw: Iterable[dict] = []
    if isinstance(data.get("scoreboard"), dict):
        games_raw = data["scoreboard"].get("games") or []
    elif isinstance(data.get("games"), list):
        games_raw = data.get("games") or []

    mapped_games = [_map_game(game) for game in games_raw]
    hydrated = _hydrate_games(mapped_games)
    if hydrated:
        return hydrated

    return []


def _fetch_games_for_date(day: datetime.date) -> list[dict]:
    games = _fetch_games_from_espn(day)
    if games is not None:
        _reset_nba_cdn_fallback_notice()
        return games

    _log_nba_cdn_fallback(day)
    return _fetch_games_from_nba_cdn(day)


def fetch_games_for_date(day: datetime.date) -> list[dict]:
    """Fetch NBA games for a date from ESPN, falling back to NBA CDN data."""

    return _fetch_games_for_date(day)


def scoreboard_date(now: Optional[datetime.datetime] = None) -> datetime.date:
    return _scoreboard_date(now)


def fetch_scoreboard(*, day: datetime.date | None = None, now: datetime.datetime | None = None) -> list[dict]:
    current_now = now or datetime.datetime.now(CENTRAL_TIME)
    target_day = day or scoreboard_date(current_now)
    if day is None and before_scoreboard_update(now=current_now, scoreboard_day=target_day):
        return compose_pre_update_scoreboard(
            now=current_now,
            scoreboard_day=target_day,
            fetch_games_for_date=fetch_games_for_date,
        )
    games = fetch_games_for_date(target_day)
    return games if isinstance(games, list) else []
