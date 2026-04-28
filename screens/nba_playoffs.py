#!/usr/bin/env python3
"""Render NBA playoff series matchups using NHL-playoffs-style layout."""

from __future__ import annotations

import datetime
import logging
import os
import re
import time
from typing import Any, Optional

from PIL import Image, ImageDraw

from config import (
    WIDTH,
    HEIGHT,
    CENTRAL_TIME,
    FONT_TITLE_SPORTS,
    FONT_TEAM_SPORTS,
    FONT_STATUS,
    IMAGES_DIR,
    SCOREBOARD_SCROLL_STEP,
    SCOREBOARD_SCROLL_DELAY,
    SCOREBOARD_SCROLL_PAUSE_TOP,
    SCOREBOARD_SCROLL_PAUSE_BOTTOM,
    SCOREBOARD_STANDINGS_BOTTOM_PADDING,
    SCOREBOARD_IN_PROGRESS_SCORE_COLOR,
    get_screen_font,
    get_screen_image_scale,
    is_kernel_driven_display,
    is_hyperpixel_next_layout,
    is_hyperpixel_4_square_layout,
    scale_value,
    scale_value_width,
)
from services.http_client import get_session
from utils import (
    ScreenImage,
    clear_display,
    load_team_logo,
    log_missing_team_logo,
    scroll_vertical_content,
    standard_scoreboard_league_logo_height,
)
from screens.nba_scoreboard import (
    _center_text,
    _team_logo_abbr,
    _get_league_logo,
    _NBA_HEADERS,
    _fetch_games_for_date,
    _scoreboard_date,
)

HYPERPIXEL_LAYOUT = is_hyperpixel_next_layout()
HYPERPIXEL_4_SQUARE = is_hyperpixel_4_square_layout()


def _scale_y(value: int) -> int:
    return scale_value(value) if HYPERPIXEL_LAYOUT else scale_value_width(value)


TITLE = "NBA Playoffs"
SUBTITLE = ""
SCREEN_ID = "NBA Playoffs"
TITLE_GAP = _scale_y(8)
SUBTITLE_GAP = _scale_y(4)
BLOCK_SPACING = _scale_y(10)
SCORE_ROW_H = _scale_y(56)
STATUS_ROW_H = _scale_y(18)
REQUEST_TIMEOUT = 8
PAIR_SPACING_BASE = max(8, scale_value_width(16))
LOWER_THAN_HYPERPIXEL_RESOLUTION = (WIDTH * HEIGHT) < (800 * 480)
SERIES_COL_WIDTHS_BASE = [
    scale_value_width(42),
    scale_value_width(32),
    scale_value_width(16),
    scale_value_width(32),
    scale_value_width(42),
]

PAIR_SPACING = PAIR_SPACING_BASE
SERIES_COL_WIDTHS = list(SERIES_COL_WIDTHS_BASE)
SERIES_WIDTH = sum(SERIES_COL_WIDTHS)
CONTENT_WIDTH = SERIES_WIDTH * 2 + PAIR_SPACING
CONTENT_LEFT = max(0, (WIDTH - CONTENT_WIDTH) // 2)
WEST_X = CONTENT_LEFT
EAST_X = CONTENT_LEFT + SERIES_WIDTH + PAIR_SPACING
SERIES_COL_X = [0]
for w in SERIES_COL_WIDTHS:
    SERIES_COL_X.append(SERIES_COL_X[-1] + w)

TITLE_FONT = FONT_TITLE_SPORTS
LOGO_DIR = os.path.join(IMAGES_DIR, "nba")
TEAM_LOGO_BASE_HEIGHT = scale_value_width(26)
PLAYOFF_LOGO_SCALE = 0.9
LEAGUE_LOGO_BASE_HEIGHT = (
    TEAM_LOGO_BASE_HEIGHT
    if (HYPERPIXEL_LAYOUT or is_kernel_driven_display())
    else standard_scoreboard_league_logo_height(TEAM_LOGO_BASE_HEIGHT)
)
if HYPERPIXEL_4_SQUARE:
    LEAGUE_LOGO_BASE_HEIGHT = min(LEAGUE_LOGO_BASE_HEIGHT, scale_value_width(40))
LOGO_HEIGHT = TEAM_LOGO_BASE_HEIGHT
LEAGUE_LOGO_GAP = _scale_y(4)

_SESSION = get_session()

_NBA_ABBR_TO_NICKNAME = {
    "ATL": "Hawks",
    "BOS": "Celtics",
    "BKN": "Nets",
    "BRK": "Nets",
    "CHA": "Hornets",
    "CHI": "Bulls",
    "CLE": "Cavaliers",
    "DAL": "Mavericks",
    "DEN": "Nuggets",
    "DET": "Pistons",
    "GSW": "Warriors",
    "GS": "Warriors",
    "HOU": "Rockets",
    "IND": "Pacers",
    "LAC": "Clippers",
    "LAL": "Lakers",
    "MEM": "Grizzlies",
    "MIA": "Heat",
    "MIL": "Bucks",
    "MIN": "Timberwolves",
    "NOP": "Pelicans",
    "NO": "Pelicans",
    "NYK": "Knicks",
    "NY": "Knicks",
    "OKC": "Thunder",
    "ORL": "Magic",
    "PHI": "76ers",
    "PHX": "Suns",
    "POR": "Trail Blazers",
    "SAC": "Kings",
    "SAS": "Spurs",
    "SA": "Spurs",
    "TOR": "Raptors",
    "UTA": "Jazz",
    "WAS": "Wizards",
    "WSH": "Wizards",
}


def _scoreboard_fonts() -> tuple:
    score = get_screen_font(SCREEN_ID, "score", base_font=FONT_TEAM_SPORTS, default_size=24)
    status = get_screen_font(SCREEN_ID, "status", base_font=FONT_STATUS, default_size=18)
    status_small = get_screen_font(SCREEN_ID, "status_small", base_font=FONT_STATUS, default_size=16)
    center = get_screen_font(SCREEN_ID, "center", base_font=FONT_STATUS, default_size=18)
    return score, status, status_small, center


SCORE_FONT, STATUS_FONT, STATUS_SMALL_FONT, CENTER_FONT = _scoreboard_fonts()
BACKGROUND_COLOR = (0, 0, 0)
_LOGO_CACHE: dict[tuple[str, int], Optional[Image.Image]] = {}


def _fit_widths_to_total(widths: list[int], target_total: int) -> list[int]:
    if not widths:
        return []
    target_total = max(len(widths), target_total)
    current_total = sum(widths)
    if current_total <= 0:
        return [1] * len(widths)
    scaled = [max(1, int(round(w * target_total / current_total))) for w in widths]
    delta = target_total - sum(scaled)
    order = sorted(range(len(widths)), key=lambda idx: widths[idx], reverse=(delta > 0))
    while delta != 0 and order:
        changed = False
        for idx in order:
            if delta == 0:
                break
            if delta < 0 and scaled[idx] <= 1:
                continue
            scaled[idx] += 1 if delta > 0 else -1
            delta += -1 if delta > 0 else 1
            changed = True
        if not changed:
            break
    return scaled


def _use_single_series_per_row_layout() -> bool:
    """Return True when displays are lower resolution than HyperPixel baselines."""
    return LOWER_THAN_HYPERPIXEL_RESOLUTION


def _recompute_series_layout() -> None:
    global PAIR_SPACING, SERIES_COL_WIDTHS, SERIES_WIDTH, CONTENT_WIDTH, CONTENT_LEFT, WEST_X, EAST_X, SERIES_COL_X

    preferred_spacing = max(0, PAIR_SPACING_BASE)
    preferred_series_width = sum(SERIES_COL_WIDTHS_BASE)
    single_series_per_row = _use_single_series_per_row_layout()

    if single_series_per_row:
        spacing = 0
        series_target = max(len(SERIES_COL_WIDTHS_BASE), WIDTH)
        series_col_widths = _fit_widths_to_total(SERIES_COL_WIDTHS_BASE, series_target)
    elif preferred_series_width * 2 + preferred_spacing <= WIDTH:
        spacing = preferred_spacing
        series_col_widths = list(SERIES_COL_WIDTHS_BASE)
    else:
        series_target = max(len(SERIES_COL_WIDTHS_BASE), WIDTH // 2)
        series_col_widths = _fit_widths_to_total(SERIES_COL_WIDTHS_BASE, series_target)
        fitted_series_width = sum(series_col_widths)
        spacing = min(preferred_spacing, max(0, WIDTH - (fitted_series_width * 2)))

    series_width = sum(series_col_widths)
    content_width = min(WIDTH, series_width if single_series_per_row else series_width * 2 + spacing)
    content_left = max(0, (WIDTH - content_width) // 2)

    PAIR_SPACING = spacing
    SERIES_COL_WIDTHS = series_col_widths
    SERIES_WIDTH = series_width
    CONTENT_WIDTH = content_width
    CONTENT_LEFT = content_left
    WEST_X = content_left
    EAST_X = content_left if single_series_per_row else content_left + series_width + spacing
    SERIES_COL_X = [0]
    for w in series_col_widths:
        SERIES_COL_X.append(SERIES_COL_X[-1] + w)


def _apply_style_overrides() -> None:
    global SCORE_FONT, STATUS_FONT, STATUS_SMALL_FONT, CENTER_FONT, LOGO_HEIGHT, BACKGROUND_COLOR

    SCORE_FONT, STATUS_FONT, STATUS_SMALL_FONT, CENTER_FONT = _scoreboard_fonts()
    BACKGROUND_COLOR = (0, 0, 0)
    _recompute_series_layout()
    team_scale = get_screen_image_scale(SCREEN_ID, "team_logo", 1.0)
    target_logo_height = max(1, int(round(TEAM_LOGO_BASE_HEIGHT * team_scale * PLAYOFF_LOGO_SCALE)))
    max_row_fit = max(1, SCORE_ROW_H - scale_value(8))
    LOGO_HEIGHT = min(target_logo_height, max_row_fit)


def _load_logo_cached(abbr: str) -> Optional[Image.Image]:
    key = (abbr or "").strip()
    if not key:
        return None
    cache_key = key.upper()
    cache_token = (cache_key, LOGO_HEIGHT)
    if cache_token in _LOGO_CACHE:
        return _LOGO_CACHE[cache_token]

    logo = load_team_logo(LOGO_DIR, cache_key, height=LOGO_HEIGHT, box_size=LOGO_HEIGHT, trim=True)
    _LOGO_CACHE[cache_token] = logo
    return logo


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _first_present_int(values: list[Any]) -> int:
    for value in values:
        parsed = _as_int(value)
        if parsed is not None:
            return parsed
    return 0


def _team_slot_from_series(series: dict, *keys: str) -> dict:
    for key in keys:
        value = series.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


def _team_from_series(series: dict, *keys: str) -> dict:
    slot = _team_slot_from_series(series, *keys)
    nested_team = slot.get("team") if isinstance(slot, dict) else None
    if isinstance(nested_team, dict) and nested_team:
        return nested_team
    if isinstance(slot, dict):
        return slot
    return {}


def _normalize_conference_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("w"):
        return "west"
    if text.startswith("e"):
        return "east"
    return ""


def _round_rank_from_text(value: Any) -> Optional[int]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if "first round" in text:
        return 1
    if "semifinal" in text:
        return 2
    if "conference final" in text:
        return 3
    if text == "finals" or "nba finals" in text:
        return 4
    short = re.search(r"\br\s*([1-4])\b", text)
    if short:
        return int(short.group(1))
    if text in {"f", "sf", "qf"}:
        return {"qf": 1, "sf": 2, "f": 4}[text]
    return None


def _round_rank_from_series_payload(series: dict) -> Optional[int]:
    for key in ("roundNumber", "round", "playoffRound", "roundNo", "roundId"):
        parsed = _as_int(series.get(key))
        if parsed is not None and parsed > 0:
            return parsed

    text_rank = _round_rank_from_text(
        series.get("roundName")
        or series.get("roundLabel")
        or series.get("seriesStatusShort")
        or series.get("seriesStatus")
        or series.get("seriesText")
    )
    if text_rank is not None:
        return text_rank
    return None


def _extract_next_game_dt(value: Any) -> Optional[datetime.datetime]:
    if isinstance(value, (int, float)):
        try:
            return datetime.datetime.fromtimestamp(float(value), tz=datetime.timezone.utc).astimezone(CENTRAL_TIME)
        except Exception:
            return None
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.datetime.fromisoformat(raw)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(CENTRAL_TIME)


def _extract_series_next_game_dt(series: dict) -> Optional[datetime.datetime]:
    candidate_keys = (
        "nextGameStartTimeUTC",
        "nextGameStartTime",
        "nextGameDateTimeUTC",
        "nextGameDateTime",
        "nextGameTimeUTC",
        "nextGameUtc",
        "nextGameDate",
        "startTimeUTC",
        "gameDateUTC",
        "gameDateTime",
        "gameDate",
    )
    for key in candidate_keys:
        dt = _extract_next_game_dt(series.get(key))
        if dt:
            return dt

    nested_candidates = (
        "nextGame",
        "nextGameInfo",
        "nextGameSchedule",
        "upcomingGame",
    )
    nested_keys = (
        "startTimeUTC",
        "startTime",
        "scheduledStartTimeUTC",
        "gameDateUTC",
        "gameDateTimeUTC",
        "gameDateTime",
        "gameDate",
        "date",
    )
    for container_key in nested_candidates:
        container = series.get(container_key)
        if not isinstance(container, dict):
            continue
        for key in nested_keys:
            dt = _extract_next_game_dt(container.get(key))
            if dt:
                return dt
    return None


def _next_day_label(dt: datetime.datetime, *, now: Optional[datetime.datetime] = None) -> Optional[str]:
    now_dt = now or datetime.datetime.now(CENTRAL_TIME)
    if dt.date() == now_dt.date():
        return "Tonight"
    if dt.date() == (now_dt + datetime.timedelta(days=1)).date():
        return "Tomorrow"
    return None


def _weekday_label(dt: datetime.datetime) -> str:
    return dt.strftime("%A")


def _next_game_day_label(dt: datetime.datetime, *, now: Optional[datetime.datetime] = None) -> str:
    return _next_day_label(dt, now=now) or _weekday_label(dt)


def _format_next_text(series: dict) -> str:
    next_dt = _extract_series_next_game_dt(series)
    if next_dt is None:
        text = str(series.get("nextGameLabel") or series.get("nextGameText") or "").strip()
        return text if text else "TBD"
    day_label = _next_game_day_label(next_dt)
    return f"{day_label} {next_dt.strftime('%-I:%M %p')}"


def _normalize_next_text(text: Any) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "TBD"
    normalized = re.sub(r"^\s*Next:\s*", "", raw, flags=re.IGNORECASE).strip()
    normalized = re.sub(
        r"\s+(?:ET|EST|EDT|CT|CST|CDT|MT|MST|MDT|PT|PST|PDT|UTC)$",
        "",
        normalized,
        flags=re.IGNORECASE,
    ).strip()
    normalized = re.sub(
        r"(?<!\d)(\d{1,2})/(\d{1,2})(?!\d)",
        lambda match: f"{int(match.group(1))}/{int(match.group(2))}",
        normalized,
    )
    date_match = re.match(r"^(\d{1,2})/(\d{1,2})(\s+.+)$", normalized)
    if date_match:
        month = int(date_match.group(1))
        day = int(date_match.group(2))
        suffix = date_match.group(3)
        try:
            next_dt = datetime.datetime.now(CENTRAL_TIME).replace(
                month=month,
                day=day,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
        except ValueError:
            next_dt = None
        if next_dt:
            normalized = f"{_next_game_day_label(next_dt)}{suffix}"
    return normalized or "TBD"


def _normalize_series_item(series: dict) -> Optional[dict]:
    if not isinstance(series, dict):
        return None

    playoff_shape_keys = {
        "awayWins",
        "homeWins",
        "topSeedWins",
        "bottomSeedWins",
        "highSeedWins",
        "lowSeedWins",
        "higherSeedWins",
        "lowerSeedWins",
        "team1Wins",
        "team2Wins",
        "topSeedTeamWins",
        "bottomSeedTeamWins",
        "seriesStatusShort",
        "seriesStatus",
        "seriesText",
        "roundLabel",
        "roundName",
    }
    away_slot = _team_slot_from_series(
        series,
        "awayTeam",
        "topSeedTeam",
        "topSeed",
        "highSeedTeam",
        "higherSeedTeam",
        "higherSeed",
        "team1",
    )
    home_slot = _team_slot_from_series(
        series,
        "homeTeam",
        "bottomSeedTeam",
        "bottomSeed",
        "lowSeedTeam",
        "lowerSeedTeam",
        "lowerSeed",
        "team2",
    )
    has_series_indicators = any(key in series for key in playoff_shape_keys)
    has_nested_series_wins = any(
        _as_int(container.get(field)) is not None
        for container in (away_slot, home_slot)
        if isinstance(container, dict)
        for field in ("wins", "seriesWins")
    )
    if not has_series_indicators and not has_nested_series_wins:
        return None
    away_team = _team_from_series(
        series,
        "awayTeam",
        "topSeedTeam",
        "topSeed",
        "highSeedTeam",
        "higherSeedTeam",
        "higherSeed",
        "team1",
    )
    home_team = _team_from_series(
        series,
        "homeTeam",
        "bottomSeedTeam",
        "bottomSeed",
        "lowSeedTeam",
        "lowerSeedTeam",
        "lowerSeed",
        "team2",
    )

    away_wins = _first_present_int(
        [
            series.get("awayWins"),
            series.get("topSeedWins"),
            series.get("highSeedWins"),
            series.get("higherSeedWins"),
            series.get("team1Wins"),
            series.get("topSeedTeamWins"),
            away_slot.get("wins") if isinstance(away_slot, dict) else None,
            away_slot.get("seriesWins") if isinstance(away_slot, dict) else None,
            away_team.get("wins") if isinstance(away_team, dict) else None,
            away_team.get("seriesWins") if isinstance(away_team, dict) else None,
        ]
    )
    home_wins = _first_present_int(
        [
            series.get("homeWins"),
            series.get("bottomSeedWins"),
            series.get("lowSeedWins"),
            series.get("lowerSeedWins"),
            series.get("team2Wins"),
            series.get("bottomSeedTeamWins"),
            home_slot.get("wins") if isinstance(home_slot, dict) else None,
            home_slot.get("seriesWins") if isinstance(home_slot, dict) else None,
            home_team.get("wins") if isinstance(home_team, dict) else None,
            home_team.get("seriesWins") if isinstance(home_team, dict) else None,
        ]
    )

    away_abbr = _team_logo_abbr(away_team)
    home_abbr = _team_logo_abbr(home_team)
    if not away_abbr or not home_abbr:
        return None

    conference = _normalize_conference_label(
        series.get("conferenceAbbrev") or series.get("conferenceName") or series.get("conference")
    )
    status = (
        series.get("seriesStatusShort")
        or series.get("seriesStatus")
        or series.get("seriesText")
        or series.get("roundLabel")
        or series.get("roundName")
        or "Series"
    )

    return {
        "teams": {
            "away": {"team": away_team, "score": away_wins},
            "home": {"team": home_team, "score": home_wins},
        },
        "status_text": str(status),
        "conference": conference,
        "higher_seed": _as_int(series.get("topSeedRank") or series.get("higherSeedRank") or away_team.get("seed")),
        "lower_seed": _as_int(series.get("bottomSeedRank") or series.get("lowerSeedRank") or home_team.get("seed")),
        "next_text": _format_next_text(series),
        "round_rank": _round_rank_from_series_payload(series),
    }


def _extract_series(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        results: list[dict] = []
        for item in payload:
            results.extend(_extract_series(item))
        return results

    if not isinstance(payload, dict):
        return []

    direct = _normalize_series_item(payload)
    if direct:
        return [direct]

    results: list[dict] = []
    for key in ("series", "seriesList", "matchups", "rounds", "bracket", "playoffBracket", "bracketData"):
        value = payload.get(key)
        if value:
            results.extend(_extract_series(value))

    if not results:
        for value in payload.values():
            if isinstance(value, (list, dict)):
                results.extend(_extract_series(value))

    deduped: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for series in results:
        teams = (series.get("teams") or {})
        away_team = ((teams.get("away") or {}).get("team") or {})
        home_team = ((teams.get("home") or {}).get("team") or {})
        away_abbr = _team_logo_abbr(away_team)
        home_abbr = _team_logo_abbr(home_team)
        key = tuple(sorted((away_abbr, home_abbr)))
        if not away_abbr or not home_abbr or key in seen:
            continue
        seen.add(key)
        deduped.append(series)
    return deduped


def _fetch_playoff_matchups() -> list[dict]:
    urls = [
        "https://cdn.nba.com/static/json/liveData/playoffbracket/playoffbracket_00.json",
        "https://nba-prod-us-east-1-media.s3.amazonaws.com/json/liveData/playoffbracket/playoffbracket_00.json",
        "https://cdn.nba.com/static/json/liveData/bracket/bracket_00.json",
        "https://nba-prod-us-east-1-media.s3.amazonaws.com/json/liveData/bracket/bracket_00.json",
    ]

    for url in urls:
        try:
            response = _SESSION.get(url, timeout=REQUEST_TIMEOUT, headers=_NBA_HEADERS)
            response.raise_for_status()
            matchups = _extract_series(response.json())
            if matchups:
                return matchups
        except Exception as exc:
            logging.debug("NBA playoffs endpoint failed (%s): %s", url, exc)
    return []


def _status_text(game: dict) -> str:
    status = (game or {}).get("status") or {}
    return str(status.get("detailedState") or "").strip()


def _is_final_game(game: dict) -> bool:
    status = (game or {}).get("status") or {}
    code = str(status.get("statusCode") or "").strip()
    abstract = str(status.get("abstractGameState") or "").strip().lower()
    detailed = _status_text(game).lower()
    return code in {"3", "4"} or abstract in {"final", "completed"} or "final" in detailed


def _parse_series_record_from_text(text: str) -> Optional[tuple[int, int]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    lowered = raw.lower()
    if "series" not in lowered and "leads" not in lowered and "tied" not in lowered:
        return None
    match = re.search(r"\b(\d+)\s*-\s*(\d+)\b", raw)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _looks_like_playoff_game(game: dict) -> bool:
    game_id = str(game.get("gamePk") or game.get("id") or game.get("gameId") or "").strip()
    if game_id.startswith("004"):
        return True
    if game_id.startswith("005"):
        return False
    game_type = str(game.get("gameType") or game.get("seasonType") or game.get("seasonStage") or "").strip().lower()
    if game_type in {
        "p",
        "playoffs",
        "postseason",
        "post-season",
        "3",
        "003",
        "004",
    }:
        return True
    if game_type in {"playin", "play-in", "4", "5", "005"}:
        return False
    status_text = _status_text(game).lower()
    return "series" in status_text and ("lead" in status_text or "tied" in status_text)


def _derive_playoff_matchups_from_recent_games(now: Optional[datetime.datetime] = None) -> list[dict]:
    base_day = _scoreboard_date(now)
    recent_days = [base_day - datetime.timedelta(days=offset) for offset in range(14, -1, -1)]
    upcoming_days = [base_day + datetime.timedelta(days=offset) for offset in range(0, 8)]

    all_games: list[dict] = []
    seen_ids: set[str] = set()
    for day in [*recent_days, *upcoming_days]:
        try:
            games_for_day = _fetch_games_for_date(day)
        except Exception as exc:
            logging.debug("NBA playoffs fallback fetch failed (%s): %s", day, exc)
            continue
        for game in games_for_day or []:
            game_id = str(game.get("gamePk") or game.get("id") or "")
            if game_id and game_id in seen_ids:
                continue
            if game_id:
                seen_ids.add(game_id)
            all_games.append(game)

    return _derive_playoff_matchups_from_games(all_games)


def _is_live_game(game: dict) -> bool:
    status = (game or {}).get("status") or {}
    abstract = str(status.get("abstractGameState") or status.get("state") or "").strip().lower()
    code = str(status.get("statusCode") or "").strip()
    detailed = _status_text(game).lower()
    return (
        abstract in {"live", "in progress"}
        or code == "2"
        or any(token in detailed for token in ("in progress", "live", "halftime", "qtr", "quarter"))
    )


def _series_has_live_game_from_games(series: dict, games: list[dict]) -> bool:
    teams = (series or {}).get("teams") or {}
    away = ((teams.get("away") or {}).get("team") or {})
    home = ((teams.get("home") or {}).get("team") or {})
    away_abbr = _team_logo_abbr(away)
    home_abbr = _team_logo_abbr(home)
    if not away_abbr or not home_abbr:
        return False

    for game in games or []:
        game_teams = game.get("teams") or {}
        game_away_abbr = _team_logo_abbr(((game_teams.get("away") or {}).get("team") or {}))
        game_home_abbr = _team_logo_abbr(((game_teams.get("home") or {}).get("team") or {}))
        if {game_away_abbr, game_home_abbr} != {away_abbr, home_abbr}:
            continue
        if _is_live_game(game):
            return True
    return False


def _derive_playoff_matchups_from_games(games: list[dict]) -> list[dict]:
    def _team_from_game_side(game: dict, side: str) -> dict:
        direct = game.get(f"{side}Team") or game.get(side)
        if isinstance(direct, dict) and direct:
            return direct
        teams = game.get("teams")
        if isinstance(teams, dict):
            nested = teams.get(side)
            if isinstance(nested, dict):
                if isinstance(nested.get("team"), dict):
                    return nested.get("team") or {}
                return nested
        return {}

    source_games = [game for game in (games or []) if _looks_like_playoff_game(game)]

    result_by_pair: dict[tuple[str, str], dict] = {}
    for game in source_games:
        away_team = _team_from_game_side(game, "away")
        home_team = _team_from_game_side(game, "home")
        away_abbr = _team_logo_abbr(away_team)
        home_abbr = _team_logo_abbr(home_team)
        if not away_abbr or not home_abbr:
            continue
        key = tuple(sorted((away_abbr, home_abbr)))

        existing = result_by_pair.get(key)
        if existing is None:
            existing = {
                "teams": {
                    "away": {"team": away_team, "score": 0},
                    "home": {"team": home_team, "score": 0},
                },
                "status_text": "Series",
                "conference": _normalize_conference_label(
                    away_team.get("conference") or away_team.get("conferenceName") or home_team.get("conference")
                ),
                "higher_seed": _as_int(away_team.get("seed") or home_team.get("seed")),
                "lower_seed": None,
                "next_text": "TBD",
                "has_live_game": False,
            }
            result_by_pair[key] = existing

        if _is_final_game(game):
            away_score = _as_int(((game.get("teams") or {}).get("away") or {}).get("score"))
            home_score = _as_int(((game.get("teams") or {}).get("home") or {}).get("score"))
            if away_score is not None and home_score is not None and away_score != home_score:
                winner_abbr = away_abbr if away_score > home_score else home_abbr
                stored_away_abbr = _team_logo_abbr((existing["teams"]["away"] or {}).get("team") or {})
                winner_side = "away" if winner_abbr == stored_away_abbr else "home"
                existing["teams"][winner_side]["score"] = int(existing["teams"][winner_side]["score"]) + 1

        detailed = _status_text(game)
        parsed = _parse_series_record_from_text(detailed)
        if parsed:
            existing["teams"]["away"]["score"] = parsed[0]
            existing["teams"]["home"]["score"] = parsed[1]
            existing["status_text"] = detailed

        game_dt = _extract_next_game_dt(game.get("gameDate"))
        if game_dt and game_dt >= datetime.datetime.now(CENTRAL_TIME):
            existing["next_text"] = f"{_next_game_day_label(game_dt)} {game_dt.strftime('%-I:%M %p')}"
        if _is_live_game(game):
            existing["has_live_game"] = True

    return list(result_by_pair.values())


def _is_current_series(series: dict) -> bool:
    teams = (series or {}).get("teams") or {}
    away = teams.get("away") or {}
    home = teams.get("home") or {}
    away_wins = _as_int(away.get("score")) or 0
    home_wins = _as_int(home.get("score")) or 0
    return away_wins < 4 and home_wins < 4


def _has_both_opponents(series: dict) -> bool:
    teams = (series or {}).get("teams") or {}
    away_slot = teams.get("away") or {}
    home_slot = teams.get("home") or {}
    away_team = (away_slot.get("team") or {}) if isinstance(away_slot, dict) else {}
    home_team = (home_slot.get("team") or {}) if isinstance(home_slot, dict) else {}
    away_abbr = _team_logo_abbr(away_team) if isinstance(away_team, dict) else ""
    home_abbr = _team_logo_abbr(home_team) if isinstance(home_team, dict) else ""
    if away_abbr and home_abbr:
        return True
    return bool(away_slot and home_slot)


def _is_completed_series(series: dict) -> bool:
    teams = (series or {}).get("teams") or {}
    away = teams.get("away") or {}
    home = teams.get("home") or {}
    away_wins = _as_int(away.get("score")) or 0
    home_wins = _as_int(home.get("score")) or 0
    return away_wins >= 4 or home_wins >= 4


def _series_winner_name(series: dict) -> str:
    teams = (series or {}).get("teams") or {}
    away_slot = teams.get("away") or {}
    home_slot = teams.get("home") or {}
    away_wins = _as_int(away_slot.get("score")) or 0
    home_wins = _as_int(home_slot.get("score")) or 0
    winner_team = (away_slot if away_wins >= 4 and away_wins >= home_wins else home_slot).get("team") or {}
    for key in ("nickname",):
        candidate = str(winner_team.get(key) or "").strip()
        if candidate:
            return candidate
    winner_abbr = _team_logo_abbr(winner_team)
    if winner_abbr:
        mapped = _NBA_ABBR_TO_NICKNAME.get(winner_abbr)
        if mapped:
            return mapped
    for key in ("teamName", "name"):
        candidate = str(winner_team.get(key) or "").strip()
        if candidate:
            team_city = str(winner_team.get("teamCity") or winner_team.get("city") or "").strip()
            if team_city and candidate.lower().startswith(f"{team_city.lower()} "):
                return candidate[len(team_city) + 1 :].strip()
            return candidate
    return ""


def _series_has_started(series: dict) -> bool:
    if not _has_both_opponents(series):
        return False
    if _is_completed_series(series):
        return True
    teams = (series or {}).get("teams") or {}
    away_wins = _as_int(((teams.get("away") or {}).get("score"))) or 0
    home_wins = _as_int(((teams.get("home") or {}).get("score"))) or 0
    if away_wins > 0 or home_wins > 0:
        return True
    status_text = str(series.get("status_text") or "").strip().lower()
    if any(token in status_text for token in ("lead", "leads", "tied", "final", "game", "in progress", "live")):
        return True
    next_text = _normalize_next_text(series.get("next_text") or "")
    return next_text != "TBD"


def _series_status_line_text(series: dict) -> str:
    if _is_completed_series(series):
        winner_name = _series_winner_name(series)
        return f"{winner_name} win!" if winner_name else "Series over"
    if _series_has_live_game(series):
        return "LIVE!"
    return _normalize_next_text(series.get("next_text") or series.get("status_text") or "TBD")


def _series_has_live_game(series: dict) -> bool:
    if bool(series.get("has_live_game")):
        return True
    status_text = str(series.get("status_text") or "").strip().lower()
    return any(token in status_text for token in ("live", "in progress", "halftime", "qtr", "quarter"))


def _series_status_line_fill(series: dict) -> tuple[int, int, int]:
    return SCOREBOARD_IN_PROGRESS_SCORE_COLOR if _series_has_live_game(series) else (255, 255, 255)


def _series_order_key(series: dict) -> tuple[int, int, str, str]:
    higher_seed = _as_int(series.get("higher_seed"))
    lower_seed = _as_int(series.get("lower_seed"))
    teams = series.get("teams") or {}
    away_team = ((teams.get("away") or {}).get("team") or {})
    home_team = ((teams.get("home") or {}).get("team") or {})
    fallback_high = min(
        _as_int(away_team.get("seed")) or 99,
        _as_int(home_team.get("seed")) or 99,
    )
    high = higher_seed if higher_seed is not None else fallback_high
    low = lower_seed if lower_seed is not None else 99
    return (high, low, _team_logo_abbr(home_team), _team_logo_abbr(away_team))


def _conference_buckets(series: list[dict]) -> tuple[list[dict], list[dict]]:
    west = [item for item in series if _normalize_conference_label(item.get("conference")) == "west"]
    east = [item for item in series if _normalize_conference_label(item.get("conference")) == "east"]
    unknown = [item for item in series if item not in west and item not in east]
    for idx, item in enumerate(unknown):
        (west if idx % 2 == 0 else east).append(item)
    west.sort(key=_series_order_key)
    east.sort(key=_series_order_key)
    return west, east


def _select_current_round_series(series: list[dict]) -> list[dict]:
    if not series:
        return []
    with_opponents = [item for item in series if _has_both_opponents(item)]
    ranked = [item for item in with_opponents if _as_int(item.get("round_rank")) is not None]
    if not ranked:
        return with_opponents

    rounds: dict[int, list[dict]] = {}
    for item in ranked:
        rank = _as_int(item.get("round_rank"))
        if rank is None:
            continue
        rounds.setdefault(rank, []).append(item)
    ordered_ranks = sorted(rounds)
    if not ordered_ranks:
        return with_opponents

    for idx, rank in enumerate(ordered_ranks):
        current_round = rounds[rank]
        if any(not _is_completed_series(item) for item in current_round):
            return current_round
        later_rounds = ordered_ranks[idx + 1 :]
        later_started = any(_series_has_started(item) for later in later_rounds for item in rounds[later])
        if not later_started:
            return current_round

    return rounds[ordered_ranks[-1]]


def _draw_series_block(canvas: Image.Image, draw: ImageDraw.ImageDraw, series: dict, *, left: int, top: int):
    teams = (series or {}).get("teams", {})
    away = teams.get("away", {})
    home = teams.get("home", {})

    for idx, text in ((0, str(away.get("score", 0))), (2, ""), (4, str(home.get("score", 0)))):
        font = SCORE_FONT if idx != 2 else CENTER_FONT
        _center_text(
            draw,
            text,
            font,
            left + SERIES_COL_X[idx],
            SERIES_COL_WIDTHS[idx],
            top,
            SCORE_ROW_H,
            fill=(255, 255, 255),
        )

    for idx, team_side in ((1, away), (3, home)):
        team_obj = (team_side or {}).get("team", {})
        abbr = _team_logo_abbr(team_obj)
        logo = _load_logo_cached(abbr)
        if not logo:
            team_name = (team_obj or {}).get("teamName") or (team_obj or {}).get("teamCity") or "Unknown Team"
            log_missing_team_logo(SCREEN_ID, team_name, abbr)
            continue
        x0 = left + SERIES_COL_X[idx] + (SERIES_COL_WIDTHS[idx] - logo.width) // 2
        y0 = top + (SCORE_ROW_H - logo.height) // 2
        canvas.paste(logo, (x0, y0), logo)

    status_top = top + SCORE_ROW_H
    _center_text(
        draw,
        _series_status_line_text(series),
        STATUS_SMALL_FONT,
        left,
        SERIES_WIDTH,
        status_top,
        STATUS_ROW_H,
        fill=_series_status_line_fill(series),
    )


def _compose_canvas(series: list[dict]) -> Image.Image:
    if not series:
        return Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    west_series, east_series = _conference_buckets(series)
    single_series_per_row = _use_single_series_per_row_layout()
    if single_series_per_row:
        ordered_series = [*west_series, *east_series]
        rows = len(ordered_series)
    else:
        ordered_series = []
        rows = max(len(west_series), len(east_series))

    block_height = SCORE_ROW_H + STATUS_ROW_H
    total_height = block_height * rows
    if rows > 1:
        total_height += BLOCK_SPACING * (rows - 1)
    canvas = Image.new("RGB", (WIDTH, total_height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(canvas)

    y = 0
    for idx in range(rows):
        if single_series_per_row:
            _draw_series_block(canvas, draw, ordered_series[idx], left=WEST_X, top=y)
        else:
            if idx < len(west_series):
                _draw_series_block(canvas, draw, west_series[idx], left=WEST_X, top=y)
            if idx < len(east_series):
                _draw_series_block(canvas, draw, east_series[idx], left=EAST_X, top=y)
        y += block_height
        if idx < rows - 1:
            sep_y = y + BLOCK_SPACING // 2
            draw.line((10, sep_y, WIDTH - 10, sep_y), fill=(45, 45, 45))
            y += BLOCK_SPACING
    return canvas


def _render_playoff_screen(series: list[dict]) -> Image.Image:
    canvas = _compose_canvas(series)

    dummy = Image.new("RGB", (WIDTH, 10), BACKGROUND_COLOR)
    dd = ImageDraw.Draw(dummy)
    try:
        l, t, r, b = dd.textbbox((0, 0), TITLE, font=TITLE_FONT)
        title_h = b - t
    except Exception:
        _, title_h = dd.textsize(TITLE, font=TITLE_FONT)
    subtitle_h = 0
    if SUBTITLE:
        try:
            l, t, r, b = dd.textbbox((0, 0), SUBTITLE, font=STATUS_SMALL_FONT)
            subtitle_h = b - t
        except Exception:
            _, subtitle_h = dd.textsize(SUBTITLE, font=STATUS_SMALL_FONT)

    league_logo = _get_league_logo()
    logo_height = league_logo.height if league_logo else 0
    logo_gap = LEAGUE_LOGO_GAP if league_logo else 0

    subtitle_gap = SUBTITLE_GAP if SUBTITLE else 0
    content_top = logo_height + logo_gap + title_h + subtitle_gap + subtitle_h + TITLE_GAP
    img_height = max(HEIGHT, content_top + canvas.height + SCOREBOARD_STANDINGS_BOTTOM_PADDING)
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
    if SUBTITLE:
        subtitle_top = ty + th + SUBTITLE_GAP
        _center_text(draw, SUBTITLE, STATUS_SMALL_FONT, 0, WIDTH, subtitle_top, subtitle_h)

    img.paste(canvas, (0, content_top))
    return img


def _scroll_display(display, full_img: Image.Image):
    scroll_vertical_content(
        display=display,
        content_height=full_img.height,
        viewport_width=WIDTH,
        viewport_height=HEIGHT,
        render_at_offset=lambda offset: display.image(full_img.crop((0, offset, WIDTH, offset + HEIGHT))),
        base_step=SCOREBOARD_SCROLL_STEP,
        pause_start=SCOREBOARD_SCROLL_PAUSE_TOP,
        pause_end=SCOREBOARD_SCROLL_PAUSE_BOTTOM,
        min_frame_time=SCOREBOARD_SCROLL_DELAY,
    )


def render_nba_playoffs(display, games: list[dict], transition: bool = False) -> ScreenImage:
    _apply_style_overrides()

    merged_games = list(games or [])
    series = _fetch_playoff_matchups()
    if not series:
        series = _derive_playoff_matchups_from_recent_games()
    if not series:
        series = _derive_playoff_matchups_from_games(merged_games)
    else:
        for item in series:
            item["has_live_game"] = _series_has_live_game_from_games(item, merged_games)

    series = _select_current_round_series(series)

    if not series:
        clear_display(display)
        img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(img)
        league_logo = _get_league_logo()
        title_top = 0
        if league_logo:
            logo_x = (WIDTH - league_logo.width) // 2
            img.paste(league_logo, (logo_x, 0), league_logo)
            title_top = league_logo.height + LEAGUE_LOGO_GAP
        _center_text(draw, TITLE, TITLE_FONT, 0, WIDTH, title_top, _scale_y(40))
        subtitle_top = title_top + _scale_y(40)
        msg_top = subtitle_top + _scale_y(16)
        if SUBTITLE:
            _center_text(draw, SUBTITLE, STATUS_SMALL_FONT, 0, WIDTH, subtitle_top, STATUS_ROW_H)
            msg_top = subtitle_top + STATUS_ROW_H + _scale_y(16)
        _center_text(draw, "No active series", STATUS_FONT, 0, WIDTH, msg_top, STATUS_ROW_H)
        if transition:
            return ScreenImage(img, displayed=False)
        display.image(img)
        time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)
        return ScreenImage(img, displayed=True)

    full_img = _render_playoff_screen(series)
    if transition:
        _scroll_display(display, full_img)
        return ScreenImage(full_img, displayed=True)

    if full_img.height <= HEIGHT:
        display.image(full_img)
        time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)
    else:
        _scroll_display(display, full_img)
    return ScreenImage(full_img, displayed=True)
