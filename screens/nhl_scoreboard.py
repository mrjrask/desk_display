#!/usr/bin/env python3
"""
nhl_scoreboard.py

Render a scrolling NHL scoreboard using the same layout as the MLB version.
Maintains the previous day's games until 9:30 AM Central before switching to
the current day's slate.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import socket
import sys
import time
from typing import Any, Dict, Iterable, Optional

from PIL import Image, ImageDraw

from config import (
    WIDTH,
    HEIGHT,
    FONT_TITLE_SPORTS,
    FONT_TEAM_SPORTS,
    FONT_STATUS,
    CENTRAL_TIME,
    IMAGES_DIR,
    SCOREBOARD_SCROLL_STEP,
    SCOREBOARD_SCROLL_DELAY,
    SCOREBOARD_SCROLL_PAUSE_TOP,
    SCOREBOARD_SCROLL_PAUSE_BOTTOM,
    SCOREBOARD_BACKGROUND_COLOR,
    SCOREBOARD_IN_PROGRESS_SCORE_COLOR,
    SCOREBOARD_FINAL_WINNING_SCORE_COLOR,
    SCOREBOARD_FINAL_LOSING_SCORE_COLOR,
)
from utils import (
    ScreenImage,
    clear_display,
    clone_font,
    load_team_logo,
    log_call,
)
from services.http_client import NHL_HEADERS, get_session

# ─── Constants ────────────────────────────────────────────────────────────────
TITLE               = "NHL Scoreboard"
TITLE_GAP           = 8
BLOCK_SPACING       = 10
SCORE_ROW_H         = 56
STATUS_ROW_H        = 18
REQUEST_TIMEOUT     = 10
API_WEB_SCOREBOARD_URL = "https://api-web.nhle.com/v1/scoreboard/{date}"
API_WEB_SCOREBOARD_NOW_URL = "https://api-web.nhle.com/v1/scoreboard/now"
API_WEB_SCOREBOARD_PARAMS = {"site": "en_nhl"}

COL_WIDTHS = [70, 60, 60, 60, 70]  # total = 320
_TOTAL_COL_WIDTH = sum(COL_WIDTHS)
_COL_LEFT = max(0, (WIDTH - _TOTAL_COL_WIDTH) // 2)
COL_X = [_COL_LEFT]
for w in COL_WIDTHS:
    COL_X.append(COL_X[-1] + w)

SCORE_FONT              = clone_font(FONT_TEAM_SPORTS, 39)
STATUS_FONT             = clone_font(FONT_STATUS, 28)
CENTER_FONT             = clone_font(FONT_STATUS, 28)
TITLE_FONT              = FONT_TITLE_SPORTS
LOGO_HEIGHT             = 52
LOGO_DIR                = os.path.join(IMAGES_DIR, "nhl")
LEAGUE_LOGO_KEYS        = ("NHL", "nhl")
LEAGUE_LOGO_GAP         = 4
LEAGUE_LOGO_HEIGHT      = max(1, int(round(LOGO_HEIGHT * 1.25)))
IN_PROGRESS_SCORE_COLOR = SCOREBOARD_IN_PROGRESS_SCORE_COLOR
IN_PROGRESS_STATUS_COLOR = IN_PROGRESS_SCORE_COLOR
FINAL_WINNING_SCORE_COLOR = SCOREBOARD_FINAL_WINNING_SCORE_COLOR
FINAL_LOSING_SCORE_COLOR = SCOREBOARD_FINAL_LOSING_SCORE_COLOR
BACKGROUND_COLOR = SCOREBOARD_BACKGROUND_COLOR

_LOGO_CACHE: dict[str, Optional[Image.Image]] = {}
_LEAGUE_LOGO: Optional[Image.Image] = None
_LEAGUE_LOGO_LOADED = False

_SESSION = get_session()

STATSAPI_HOST = "statsapi.web.nhl.com"
API_WEB_HOST = "api-web.nhle.com"
_DNS_RETRY_INTERVAL = 600  # seconds
_dns_block_until = 0.0


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _scoreboard_date(now: Optional[datetime.datetime] = None) -> datetime.date:
    now = now or datetime.datetime.now(CENTRAL_TIME)
    cutoff = now.replace(hour=9, minute=30, second=0, microsecond=0)
    if now < cutoff:
        return (now - datetime.timedelta(days=1)).date()
    return now.date()


def _load_logo_cached(abbr: str) -> Optional[Image.Image]:
    key = (abbr or "").strip()
    if not key:
        return None
    cache_key = key.upper()
    if cache_key in _LOGO_CACHE:
        return _LOGO_CACHE[cache_key]

    candidates = [cache_key, cache_key.lower(), cache_key.title()]
    for candidate in candidates:
        path = os.path.join(LOGO_DIR, f"{candidate}.png")
        if os.path.exists(path):
            logo = load_team_logo(LOGO_DIR, candidate, height=LOGO_HEIGHT)
            _LOGO_CACHE[cache_key] = logo
            return logo

    _LOGO_CACHE[cache_key] = None
    return None


def _get_league_logo() -> Optional[Image.Image]:
    global _LEAGUE_LOGO, _LEAGUE_LOGO_LOADED
    if not _LEAGUE_LOGO_LOADED:
        for key in LEAGUE_LOGO_KEYS:
            logo = load_team_logo(LOGO_DIR, key, height=LEAGUE_LOGO_HEIGHT)
            if logo is not None:
                _LEAGUE_LOGO = logo
                break
        _LEAGUE_LOGO_LOADED = True
    return _LEAGUE_LOGO


def _team_logo_abbr(team: dict) -> str:
    if not isinstance(team, dict):
        return ""
    for key in ("abbreviation", "abbrev", "triCode", "shortName"):
        value = team.get(key)
        if isinstance(value, str) and value.strip():
            candidate = value.strip().upper()
            if os.path.exists(os.path.join(LOGO_DIR, f"{candidate}.png")):
                return candidate
    name = (team.get("name") or team.get("teamName") or "").strip()
    return name[:3].upper() if name else ""


def _should_display_scores(game: dict) -> bool:
    status = (game or {}).get("status", {}) or {}
    abstract = (status.get("abstractGameState") or "").lower()
    detailed = (status.get("detailedState") or "").lower()
    code = (status.get("statusCode") or "").strip()

    if abstract in {"final", "live"}:
        return True
    if code in {"3", "4"}:  # Live or Final in NHL stats API
        return True
    if "progress" in detailed or "final" in detailed:
        return True
    return False


def _is_game_in_progress(game: dict) -> bool:
    status = (game or {}).get("status", {}) or {}
    abstract = (status.get("abstractGameState") or "").lower()
    if abstract == "live":
        return True
    code = (status.get("statusCode") or "").strip()
    if code == "3":
        return True
    detailed = (status.get("detailedState") or "").lower()
    if "progress" in detailed:
        return True
    return False


def _is_game_final(game: dict) -> bool:
    status = (game or {}).get("status", {}) or {}
    abstract = (status.get("abstractGameState") or "").lower()
    detailed = (status.get("detailedState") or "").lower()
    code = (status.get("statusCode") or "").strip()

    if abstract in {"final", "completed"}:
        return True
    if code == "4":
        return True
    if "final" in detailed:
        return True
    return False


def _score_text(side: dict, *, show: bool) -> str:
    if not show:
        return "—"
    score = (side or {}).get("score")
    return "—" if score is None else str(score)


def _score_value(side: dict) -> Optional[int]:
    score = (side or {}).get("score")
    if isinstance(score, (int, float)):
        return int(score)
    if isinstance(score, str):
        cleaned = score.strip()
        if cleaned.isdigit():
            try:
                return int(cleaned)
            except Exception:
                return None
        try:
            return int(float(cleaned))
        except Exception:
            return None
    return None


def _team_result(side: dict, opponent: dict) -> Optional[str]:
    for key in ("isWinner", "winner", "won"):
        value = (side or {}).get(key)
        if isinstance(value, bool):
            return "win" if value else "loss"

    side_score = _score_value(side)
    opp_score = _score_value(opponent)
    if side_score is not None and opp_score is not None:
        if side_score > opp_score:
            return "win"
        if side_score < opp_score:
            return "loss"
    return None


def _final_results(away: dict, home: dict) -> dict:
    away_result = _team_result(away, home)
    home_result = _team_result(home, away)

    if away_result == "win":
        home_result = "loss"
    elif away_result == "loss":
        home_result = "win"
    elif home_result == "win":
        away_result = "loss"
    elif home_result == "loss":
        away_result = "win"

    return {"away": away_result, "home": home_result}
def _score_fill(team_key: str, *, in_progress: bool, final: bool, results: dict) -> tuple[int, int, int]:
    if in_progress:
        return IN_PROGRESS_SCORE_COLOR
    if final:
        result = results.get(team_key)
        if result == "loss":
            return FINAL_LOSING_SCORE_COLOR
        if result == "win":
            return FINAL_WINNING_SCORE_COLOR
    return (255, 255, 255)


def _normalize_period_for_display(period_ord: str) -> str:
    if not isinstance(period_ord, str):
        return ""
    text = period_ord.strip().upper()
    if not text:
        return ""
    if text in {"OT", "SO"}:
        return text
    if text.endswith("TH"):
        try:
            value = int(text[:-2])
        except ValueError:
            return text
        if value >= 4:
            overtime_number = value - 3
            return "OT" if overtime_number == 1 else f"{overtime_number}OT"
    return text


def _format_status(game: dict) -> str:
    status = (game or {}).get("status", {}) or {}
    linescore = (game or {}).get("linescore", {}) or {}
    detailed = (status.get("detailedState") or "").strip()
    detailed_lower = detailed.lower()
    abstract = (status.get("abstractGameState") or "").lower()

    if "postponed" in detailed_lower:
        return "Postponed"
    if "suspended" in detailed_lower:
        return detailed or "Suspended"

    if abstract in {"final", "completed"} or "final" in detailed_lower or status.get("statusCode") == "4":
        period_ord = _normalize_period_for_display(linescore.get("currentPeriodOrdinal"))
        if linescore.get("hasShootout"):
            return "Final/SO"
        if period_ord and period_ord not in {"1ST", "2ND", "3RD"}:
            return f"Final/{period_ord}"
        return "Final"

    if abstract == "live" or status.get("statusCode") == "3" or "progress" in detailed_lower:
        intermission = linescore.get("intermissionInfo") or {}
        in_intermission = intermission.get("inIntermission")
        period_ord = _normalize_period_for_display(linescore.get("currentPeriodOrdinal"))
        time_remaining = (linescore.get("currentPeriodTimeRemaining") or "").upper()
        if in_intermission:
            if period_ord:
                return f"{period_ord} INT"
            return "INT"
        if time_remaining and time_remaining != "END":
            return f"{period_ord} {time_remaining}".strip()
        if time_remaining == "END" and period_ord:
            return f"End {period_ord}".strip()
        return detailed or "In Progress"

    start_local = game.get("_start_local")
    if isinstance(start_local, datetime.datetime):
        return start_local.strftime("%I:%M %p").lstrip("0")

    return detailed or "TBD"


def _center_text(draw: ImageDraw.ImageDraw, text: str, font, x: int, width: int,
                 y: int, height: int, *, fill=(255, 255, 255)):
    if not text:
        return
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        tw, th = r - l, b - t
        tx = x + (width - tw) // 2 - l
        ty = y + (height - th) // 2 - t
    except Exception:
        tw, th = draw.textsize(text, font=font)
        tx = x + (width - tw) // 2
        ty = y + (height - th) // 2
    draw.text((tx, ty), text, font=font, fill=fill)


def _draw_game_block(canvas: Image.Image, draw: ImageDraw.ImageDraw, game: dict, top: int):
    teams = (game or {}).get("teams", {})
    away = teams.get("away", {})
    home = teams.get("home", {})

    show_scores = _should_display_scores(game)
    away_text = _score_text(away, show=show_scores)
    home_text = _score_text(home, show=show_scores)
    in_progress = _is_game_in_progress(game)
    final = _is_game_final(game)
    results = _final_results(away, home) if final else {"away": None, "home": None}

    score_top = top
    for idx, text in ((0, away_text), (2, "@"), (4, home_text)):
        font = SCORE_FONT if idx != 2 else CENTER_FONT
        if idx == 0:
            fill = _score_fill("away", in_progress=in_progress, final=final, results=results)
        elif idx == 4:
            fill = _score_fill("home", in_progress=in_progress, final=final, results=results)
        else:
            fill = (255, 255, 255)
        _center_text(draw, text, font, COL_X[idx], COL_WIDTHS[idx], score_top, SCORE_ROW_H, fill=fill)

    for idx, team_side, team_key in ((1, away, "away"), (3, home, "home")):
        team_obj = (team_side or {}).get("team", {})
        abbr = _team_logo_abbr(team_obj)
        logo = _load_logo_cached(abbr)
        if not logo:
            continue
        x0 = COL_X[idx] + (COL_WIDTHS[idx] - logo.width) // 2
        y0 = score_top + (SCORE_ROW_H - logo.height) // 2
        canvas.paste(logo, (x0, y0), logo)

    status_top = score_top + SCORE_ROW_H
    status_text = _format_status(game)
    status_fill = IN_PROGRESS_STATUS_COLOR if in_progress else (255, 255, 255)
    _center_text(draw, status_text, STATUS_FONT, COL_X[2], COL_WIDTHS[2], status_top, STATUS_ROW_H, fill=status_fill)


def _compose_canvas(games: list[dict]) -> Image.Image:
    if not games:
        return Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    block_height = SCORE_ROW_H + STATUS_ROW_H
    total_height = block_height * len(games)
    if len(games) > 1:
        total_height += BLOCK_SPACING * (len(games) - 1)
    canvas = Image.new("RGB", (WIDTH, total_height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(canvas)

    y = 0
    for idx, game in enumerate(games):
        _draw_game_block(canvas, draw, game, y)
        y += SCORE_ROW_H + STATUS_ROW_H
        if idx < len(games) - 1:
            sep_y = y + BLOCK_SPACING // 2
            draw.line((10, sep_y, WIDTH - 10, sep_y), fill=(45, 45, 45))
            y += BLOCK_SPACING
    return canvas


def _timestamp_to_local(ts: str) -> Optional[datetime.datetime]:
    if not ts:
        return None
    try:
        dt = datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
        dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(CENTRAL_TIME)
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
    mapped = {
        "team": {
            "id": team_id,
            "abbreviation": abbr,
            "triCode": abbr,
        },
    }
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


def _map_api_web_game(game: Dict[str, Any], day: datetime.date) -> Dict[str, Any]:
    start_candidates = (
        game.get("startTimeUTC"),
        game.get("startTime"),
        game.get("gameDateTime"),
        game.get("gameDate"),
    )
    game_dt = None
    for candidate in start_candidates:
        if not candidate:
            continue
        if isinstance(candidate, str):
            text = candidate.strip()
            if not text:
                continue
            if text.endswith("Z"):
                fmt = text
            else:
                try:
                    parsed = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
                except ValueError:
                    continue
                else:
                    parsed = parsed.astimezone(datetime.timezone.utc)
                    fmt = parsed.strftime("%Y-%m-%dT%H:%M:%SZ")
                    game_dt = parsed
                    break
            try:
                parsed = datetime.datetime.strptime(fmt, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                continue
            parsed = parsed.replace(tzinfo=datetime.timezone.utc)
            game_dt = parsed
            break
    if game_dt is None:
        game_dt = datetime.datetime.combine(day, datetime.time(0, 0), tzinfo=datetime.timezone.utc)

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
        abstract_state = "live"
        status_code = "3"
        detailed_state = detailed_state or "In Progress"
    elif game_state in {"FINAL", "OFF"}:
        abstract_state = "final"
        status_code = "4"
        detailed_state = detailed_state or "Final"
    elif game_state in {"FUT", "PRE", "SCHEDULED", "PREGAME"}:
        abstract_state = "preview"
        status_code = "1"
        detailed_state = detailed_state or "Scheduled"
    elif game_state in {"POSTP", "POSTPONED"}:
        abstract_state = "preview"
        status_code = "1"
        detailed_state = "Postponed"
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
        "status": {
            "abstractGameState": abstract_state,
            "detailedState": detailed_state,
        },
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


def _extract_api_web_games(data: Dict[str, Any], day: datetime.date) -> list[Dict[str, Any]]:
    """Return game payloads for the requested day from the api-web response."""

    def _normalize_date(value: Any) -> Optional[str]:
        if not value:
            return None
        if isinstance(value, datetime.date):
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

    # Direct games list at the top level.
    direct = data.get("games")
    if isinstance(direct, list):
        _append_from(direct)

    # Some responses include a nested scoreboard object with games.
    scoreboard = data.get("scoreboard")
    if isinstance(scoreboard, dict) and isinstance(scoreboard.get("games"), list):
        _append_from(scoreboard.get("games"))

    # Weekly buckets show a range of dates; pick the ones matching the target day.
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

    # De-duplicate while preserving order.
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


def _fetch_games_api_web(day: datetime.date) -> list[dict]:
    urls = [
        API_WEB_SCOREBOARD_URL.format(date=day.isoformat()),
        API_WEB_SCOREBOARD_NOW_URL,
    ]

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
            if not isinstance(game, dict):
                continue
            try:
                mapped_games.append(_map_api_web_game(game, day))
            except Exception as exc:  # defensive parsing
                logging.debug("Skipping api-web game due to error: %s", exc)
        if mapped_games:
            return _hydrate_games(mapped_games)

    return []


def _hydrate_games(raw_games: Iterable[dict]) -> list[dict]:
    games: list[dict] = []
    for game in raw_games:
        game = game or {}
        start_local = _timestamp_to_local(game.get("gameDate"))
        if start_local:
            game["_start_local"] = start_local
            game["_start_sort"] = start_local.timestamp()
        else:
            game["_start_sort"] = float("inf")
        games.append(game)
    games.sort(key=lambda g: (g.get("_start_sort", float("inf")), g.get("gamePk", 0)))
    return games


def _statsapi_available() -> bool:
    """Return True when the statsapi host resolves or a retry window has elapsed."""

    global _dns_block_until

    now = time.time()
    if now < _dns_block_until:
        return False

    try:
        socket.getaddrinfo(
            STATSAPI_HOST,
            443,
            proto=socket.IPPROTO_TCP,
        )
    except socket.gaierror as exc:
        logging.warning("NHL statsapi DNS lookup failed: %s", exc)
        _dns_block_until = now + _DNS_RETRY_INTERVAL
        return False
    except Exception as exc:  # defensive guard against unexpected errors
        logging.debug("Unexpected error checking NHL statsapi DNS: %s", exc)
    else:
        _dns_block_until = 0.0
        return True

    return True


def _format_family(family: int) -> str:
    if family == socket.AF_INET:
        return "AF_INET"
    if family == socket.AF_INET6:
        return "AF_INET6"
    return str(family)


def _format_socktype(socktype: int) -> str:
    if socktype == socket.SOCK_STREAM:
        return "SOCK_STREAM"
    if socktype == socket.SOCK_DGRAM:
        return "SOCK_DGRAM"
    return str(socktype)


def _resolve_host(host: str, *, port: int = 443) -> dict:
    """Return detailed resolution information for a host."""

    result: dict[str, Any] = {
        "host": host,
        "port": port,
    }
    start = time.perf_counter()
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        result.update(
            {
                "status": "error",
                "error": exc.strerror or str(exc),
                "errno": exc.errno,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive
        result.update(
            {
                "status": "error",
                "error": str(exc),
            }
        )
    else:
        formatted = []
        for family, socktype, proto, canonname, sockaddr in infos:
            address = sockaddr[0] if sockaddr else None
            formatted.append(
                {
                    "family": _format_family(family),
                    "socktype": _format_socktype(socktype),
                    "proto": proto,
                    "canonname": canonname,
                    "address": address,
                }
            )
        result.update(
            {
                "status": "ok",
                "addresses": formatted,
            }
        )
    result["duration_ms"] = round((time.perf_counter() - start) * 1000, 2)
    return result


def _read_resolv_conf() -> dict:
    path = "/etc/resolv.conf"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            contents = fh.read()
    except OSError as exc:
        return {
            "path": path,
            "status": "error",
            "error": str(exc),
        }
    return {
        "path": path,
        "status": "ok",
        "contents": contents,
    }


def dns_diagnostics() -> dict:
    """Collect DNS/network diagnostics for NHL endpoints."""

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report: dict[str, Any] = {
        "generated_at": now,
        "hosts": [
            _resolve_host(STATSAPI_HOST),
            _resolve_host(API_WEB_HOST),
        ],
        "resolv_conf": _read_resolv_conf(),
        "http_checks": [],
    }

    urls = [
        (
            f"https://{STATSAPI_HOST}/api/v1/schedule?date={datetime.date.today().isoformat()}",
            "statsapi_schedule",
        ),
        (API_WEB_SCOREBOARD_NOW_URL, "api_web_scoreboard_now"),
    ]

    for url, name in urls:
        check: dict[str, Any] = {"url": url, "name": name}
        start = time.perf_counter()
        try:
            response = _SESSION.get(url, timeout=min(REQUEST_TIMEOUT, 5))
            response.raise_for_status()
        except Exception as exc:
            check.update({"status": "error", "error": str(exc)})
        else:
            check.update({"status": "ok", "http_status": response.status_code})
        check["duration_ms"] = round((time.perf_counter() - start) * 1000, 2)
        report["http_checks"].append(check)

    env_overrides = {}
    for key in ("RES_OPTIONS", "LOCALDOMAIN", "HOSTALIASES"):
        if key in os.environ:
            env_overrides[key] = os.environ[key]
    if env_overrides:
        report["env"] = env_overrides

    return report


def _fetch_games_for_date(day: datetime.date) -> list[dict]:
    if not _statsapi_available():
        logging.info("Using api-web NHL scoreboard endpoint for %s (statsapi DNS failure)", day)
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


def _render_scoreboard(games: list[dict]) -> Image.Image:
    canvas = _compose_canvas(games)

    dummy = Image.new("RGB", (WIDTH, 10), BACKGROUND_COLOR)
    dd = ImageDraw.Draw(dummy)
    try:
        l, t, r, b = dd.textbbox((0, 0), TITLE, font=TITLE_FONT)
        title_h = b - t
    except Exception:
        _, title_h = dd.textsize(TITLE, font=TITLE_FONT)

    league_logo = _get_league_logo()
    logo_height = league_logo.height if league_logo else 0
    logo_gap = LEAGUE_LOGO_GAP if league_logo else 0

    content_top = logo_height + logo_gap + title_h + TITLE_GAP
    img_height = max(HEIGHT, content_top + canvas.height)
    img = Image.new("RGB", (WIDTH, img_height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    if league_logo:
        logo_x = (WIDTH - league_logo.width) // 2
        img.paste(league_logo, (logo_x, 0), league_logo)
    title_top = logo_height + logo_gap

    try:
        l, t, r, b = draw.textbbox((0, 0), TITLE, font=TITLE_FONT)
        tw, th = r - l, b - t
        tx = (WIDTH - tw) // 2 - l
        ty = title_top - t
    except Exception:
        tw, th = draw.textsize(TITLE, font=TITLE_FONT)
        tx = (WIDTH - tw) // 2
        ty = title_top
    draw.text((tx, ty), TITLE, font=TITLE_FONT, fill=(255, 255, 255))

    img.paste(canvas, (0, content_top))
    return img


def _scroll_display(display, full_img: Image.Image):
    if full_img.height <= HEIGHT:
        display.image(full_img)
        return

    max_offset = full_img.height - HEIGHT
    frame = full_img.crop((0, 0, WIDTH, HEIGHT))
    display.image(frame)
    time.sleep(SCOREBOARD_SCROLL_PAUSE_TOP)

    for offset in range(
        SCOREBOARD_SCROLL_STEP, max_offset + 1, SCOREBOARD_SCROLL_STEP
    ):
        frame = full_img.crop((0, offset, WIDTH, offset + HEIGHT))
        display.image(frame)
        time.sleep(SCOREBOARD_SCROLL_DELAY)

    time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)


# ─── Public API ───────────────────────────────────────────────────────────────
@log_call
def draw_nhl_scoreboard(display, transition: bool = False) -> ScreenImage:
    games = _fetch_games_for_date(_scoreboard_date())

    if not games:
        clear_display(display)
        img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(img)
        league_logo = _get_league_logo()
        title_top = 0
        if league_logo:
            logo_x = (WIDTH - league_logo.width) // 2
            img.paste(league_logo, (logo_x, 0), league_logo)
            title_top = league_logo.height + LEAGUE_LOGO_GAP
        try:
            l, t, r, b = draw.textbbox((0, 0), TITLE, font=TITLE_FONT)
            tw, th = r - l, b - t
            tx = (WIDTH - tw) // 2 - l
            ty = title_top - t
        except Exception:
            tw, th = draw.textsize(TITLE, font=TITLE_FONT)
            tx = (WIDTH - tw) // 2
            ty = title_top
        draw.text((tx, ty), TITLE, font=TITLE_FONT, fill=(255, 255, 255))
        _center_text(draw, "No games", STATUS_FONT, 0, WIDTH, HEIGHT // 2 - STATUS_ROW_H // 2, STATUS_ROW_H)
        if transition:
            return ScreenImage(img, displayed=False)
        display.image(img)
        time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)
        return ScreenImage(img, displayed=True)

    full_img = _render_scoreboard(games)
    if transition:
        _scroll_display(display, full_img)
        return ScreenImage(full_img, displayed=True)

    if full_img.height <= HEIGHT:
        display.image(full_img)
        time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)
    else:
        _scroll_display(display, full_img)
    return ScreenImage(full_img, displayed=True)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="NHL scoreboard renderer")
    parser.add_argument(
        "--diagnose-dns",
        action="store_true",
        help="print DNS diagnostics instead of rendering the scoreboard",
    )
    args = parser.parse_args()

    if args.diagnose_dns:
        report = dns_diagnostics()
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        sys.exit(0)

    from utils import Display

    disp = Display()
    try:
        draw_nhl_scoreboard(disp)
    finally:
        clear_display(disp)

