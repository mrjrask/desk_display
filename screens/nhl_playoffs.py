#!/usr/bin/env python3
"""Render NHL playoff series matchups using scoreboard-style layout."""

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
    FONT_TITLE_SPORTS,
    FONT_TEAM_SPORTS,
    FONT_STATUS,
    CENTRAL_TIME,
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
from services.http_client import NHL_HEADERS, get_session
from utils import (
    ScreenImage,
    clear_display,
    load_team_logo,
    log_missing_team_logo,
    scroll_vertical_content,
    standard_scoreboard_team_logo_height,
    standard_scoreboard_league_logo_height,
)
from screens.nhl_scoreboard import _center_text, _team_logo_abbr, _get_league_logo
from screens.team_abbreviation_mappings import NHL_ABBR_TO_COMMON_NAME

HYPERPIXEL_LAYOUT = is_hyperpixel_next_layout()
HYPERPIXEL_4_SQUARE = is_hyperpixel_4_square_layout()


def _scale_y(value: int) -> int:
    return scale_value(value) if HYPERPIXEL_LAYOUT else scale_value_width(value)


TITLE = "NHL Playoffs"
SUBTITLE = "Best-of-7"
SCREEN_ID = "NHL Playoffs"
TITLE_GAP = _scale_y(8)
SUBTITLE_GAP = _scale_y(8)
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
LOGO_DIR = os.path.join(IMAGES_DIR, "nhl")
# Match NHL Scoreboard v2 logo baseline sizing, then reduce by 10% for playoffs.
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

def _scoreboard_fonts() -> tuple:
    # Keep score typography aligned with NHL Scoreboard v2.
    score = get_screen_font(SCREEN_ID, "score", base_font=FONT_TEAM_SPORTS, default_size=24)
    status = get_screen_font(SCREEN_ID, "status", base_font=FONT_STATUS, default_size=18)
    # Reduce game time/date text by 4pt from prior default (20 -> 16).
    status_small = get_screen_font(SCREEN_ID, "status_small", base_font=FONT_STATUS, default_size=16)
    center = get_screen_font(SCREEN_ID, "center", base_font=FONT_STATUS, default_size=18)
    return score, status, status_small, center


SCORE_FONT, STATUS_FONT, STATUS_SMALL_FONT, CENTER_FONT = _scoreboard_fonts()
BACKGROUND_COLOR = (0, 0, 0)
_LOGO_CACHE: dict[tuple[str, int], Optional[Image.Image]] = {}


def _get_playoffs_logo() -> Optional[Image.Image]:
    """Return the Stanley Cup Playoffs logo for the NHL Playoffs screen."""
    logo = load_team_logo(LOGO_DIR, "SCP", height=LEAGUE_LOGO_BASE_HEIGHT, box_size=LEAGUE_LOGO_BASE_HEIGHT, trim=True)
    if logo:
        return logo
    # Fallback to the shared NHL league logo behavior if SCP.png is unavailable.
    return _get_league_logo()


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

    for candidate in (cache_key, cache_key.lower(), cache_key.title()):
        path = os.path.join(LOGO_DIR, f"{candidate}.png")
        if os.path.exists(path):
            logo = load_team_logo(LOGO_DIR, candidate, height=LOGO_HEIGHT, box_size=LOGO_HEIGHT, trim=True)
            _LOGO_CACHE[cache_token] = logo
            return logo

    _LOGO_CACHE[cache_token] = None
    return None


def _playoff_season(now: Optional[datetime.datetime] = None) -> str:
    now = now or datetime.datetime.now(CENTRAL_TIME)
    if now.month >= 9:
        start_year = now.year
    else:
        start_year = now.year - 1
    return f"{start_year}{start_year + 1}"


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _team_from_series(series: dict, *keys: str) -> dict:
    for key in keys:
        value = series.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _normalize_conference_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("w"):
        return "west"
    if text.startswith("e"):
        return "east"
    return ""


def _conference_from_series_letter(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"A", "B", "C", "D"}:
        return "east"
    if text in {"E", "F", "G", "H"}:
        return "west"
    return ""


def _round_rank_from_text(value: Any) -> Optional[int]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if "first round" in text or text in {"r1", "qf", "quarterfinal", "quarterfinals"}:
        return 1
    if "second round" in text or "semifinal" in text or text in {"r2", "sf"}:
        return 2
    if "conference final" in text or text in {"ecf", "wcf", "cf", "r3"}:
        return 3
    if "stanley cup final" in text or text in {"scf", "final", "f", "r4"}:
        return 4
    short = re.search(r"\br\s*([1-4])\b", text)
    if short:
        return int(short.group(1))
    return None


def _round_rank_from_series_payload(series: dict) -> Optional[int]:
    for key in ("roundNumber", "round", "playoffRound", "roundNo", "roundId"):
        parsed = _as_int(series.get(key))
        if parsed is not None and parsed > 0:
            return parsed
    return _round_rank_from_text(
        series.get("roundName")
        or series.get("roundLabel")
        or series.get("seriesStatusShort")
        or series.get("seriesStatus")
    )


def _first_int(values: list[Any]) -> Optional[int]:
    for value in values:
        parsed = _as_int(value)
        if parsed is not None:
            return parsed
    return None


def _dict_localized_text(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        for key in ("default", "en", "fr", "es", "name"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return ""


def _team_abbr(team: dict) -> str:
    if not isinstance(team, dict):
        return ""

    # Prefer canonical tri-codes/abbreviations first (including nested payloads)
    # before falling back to human-readable labels such as shortName/code.
    for key in ("abbreviation", "abbrev", "teamAbbrev", "triCode"):
        candidate = _dict_localized_text(team.get(key))
        if candidate:
            return candidate.upper()
    nested_team = team.get("team")
    if isinstance(nested_team, dict):
        nested_abbr = _team_abbr(nested_team)
        if nested_abbr:
            return nested_abbr
    for key in ("shortName", "code"):
        candidate = _dict_localized_text(team.get(key))
        if candidate:
            return candidate.upper()
    return _team_logo_abbr(team)


def _parse_datetime(value: Any) -> Optional[datetime.datetime]:
    if not value:
        return None
    if isinstance(value, datetime.datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            dt = datetime.datetime.fromisoformat(text)
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=CENTRAL_TIME)
    return dt.astimezone(CENTRAL_TIME)


def _value_has_time_component(value: Any) -> bool:
    if isinstance(value, datetime.datetime):
        return True
    text = str(value or "").strip()
    if not text:
        return False
    return "T" in text or " " in text


def _extract_next_game_dt(series: dict) -> Optional[datetime.datetime]:
    candidate_keys = (
        "nextGameStartTimeUTC",
        "nextGameStartTime",
        "nextGameDateTimeUTC",
        "nextGameDateTime",
        "nextGameUtc",
        "nextGameDate",
        "startTimeUTC",
        "gameDateUTC",
        "gameDateTime",
        "gameDate",
    )
    for key in candidate_keys:
        dt = _parse_datetime(series.get(key))
        if dt:
            return dt
    next_game = series.get("nextGame")
    if isinstance(next_game, dict):
        for key in ("startTimeUTC", "startTime", "gameDateUTC", "gameDateTime", "gameDate", "date"):
            dt = _parse_datetime(next_game.get(key))
            if dt:
                return dt
    return None


def _extract_next_game_info(series: dict) -> tuple[Optional[datetime.datetime], bool]:
    candidate_keys = (
        "nextGameStartTimeUTC",
        "nextGameStartTime",
        "nextGameDateTimeUTC",
        "nextGameDateTime",
        "nextGameUtc",
        "nextGameDate",
        "startTimeUTC",
        "gameDateUTC",
        "gameDateTime",
        "gameDate",
    )
    for key in candidate_keys:
        value = series.get(key)
        dt = _parse_datetime(value)
        if dt:
            return dt, _value_has_time_component(value)
    next_game = series.get("nextGame")
    if isinstance(next_game, dict):
        for key in ("startTimeUTC", "startTime", "gameDateUTC", "gameDateTime", "gameDate", "date"):
            value = next_game.get(key)
            dt = _parse_datetime(value)
            if dt:
                return dt, _value_has_time_component(value)
    for container_key in ("nextGameSchedule", "nextGameInfo"):
        nested = series.get(container_key)
        if isinstance(nested, dict):
            for key in (
                "startTimeUTC",
                "startTime",
                "scheduledStartTimeUTC",
                "gameDateUTC",
                "gameDateTime",
                "gameDate",
                "date",
            ):
                value = nested.get(key)
                dt = _parse_datetime(value)
                if dt:
                    return dt, _value_has_time_component(value)
    return None, False


def _first_present_int(values: list[Any], default: int = 0) -> int:
    for value in values:
        parsed = _as_int(value)
        if parsed is None and isinstance(value, str):
            match = re.match(r"^\s*(\d+)\s*[-/]", value)
            if match:
                parsed = _as_int(match.group(1))
        if parsed is not None:
            return parsed
    return default


def _format_next_text(series: dict) -> str:
    dt, has_time = _extract_next_game_info(series)
    if not dt:
        return "TBD"
    day_label = _next_game_day_label(dt)
    if not has_time:
        return day_label
    time_text = _format_clock(dt)
    return f"{day_label} {time_text}"


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


def _format_clock(dt: datetime.datetime) -> str:
    if dt.minute == 0:
        return dt.strftime("%-I %p")
    return dt.strftime("%-I:%M %p")


def _normalize_next_text(text: Any) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "TBD"

    normalized = re.sub(r"^\s*Next:\s*", "", raw, flags=re.IGNORECASE).strip()
    if not normalized:
        return "TBD"

    # Drop trailing timezone abbreviations, while preserving meridiem markers (AM/PM).
    normalized = re.sub(
        r"\s+(?:ET|EST|EDT|CT|CST|CDT|MT|MST|MDT|PT|PST|PDT|UTC)$",
        "",
        normalized,
        flags=re.IGNORECASE,
    ).strip()

    # Remove leading zeros from month/day date fragments (e.g., 04/09 -> 4/9).
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
        day_label = _next_game_day_label(next_dt) if next_dt else None
        if day_label:
            normalized = f"{day_label}{suffix}"

    normalized = re.sub(r"(\b\d{1,2}):00(\s+[AP]M\b)", r"\1\2", normalized, flags=re.IGNORECASE)

    return normalized or "TBD"


def _normalize_series_item(series: dict) -> Optional[dict]:
    if not isinstance(series, dict):
        return None

    away_team = _team_from_series(series, "awayTeam", "topSeedTeam", "topSeed", "team1", "homeTeam1")
    home_team = _team_from_series(series, "homeTeam", "bottomSeedTeam", "bottomSeed", "team2", "homeTeam2")

    away_wins = _first_present_int(
        [
            series.get("awayWins"),
            series.get("topSeedWins"),
            series.get("team1Wins"),
            series.get("topSeedTeamWins"),
            series.get("topSeed", {}).get("wins") if isinstance(series.get("topSeed"), dict) else None,
            away_team.get("wins") if isinstance(away_team, dict) else None,
            away_team.get("seriesWins") if isinstance(away_team, dict) else None,
        ]
    )
    home_wins = _first_present_int(
        [
            series.get("homeWins"),
            series.get("bottomSeedWins"),
            series.get("team2Wins"),
            series.get("bottomSeedTeamWins"),
            series.get("bottomSeed", {}).get("wins") if isinstance(series.get("bottomSeed"), dict) else None,
            home_team.get("wins") if isinstance(home_team, dict) else None,
            home_team.get("seriesWins") if isinstance(home_team, dict) else None,
        ]
    )

    away_abbr = _team_abbr(away_team)
    home_abbr = _team_abbr(home_team)
    if not away_abbr or not home_abbr:
        return None

    conference = _normalize_conference_label(
        series.get("conferenceAbbrev")
        or series.get("conferenceName")
        or _conference_from_series_letter(series.get("seriesLetter"))
        or (away_team.get("conferenceAbbrev") if isinstance(away_team, dict) else None)
        or (home_team.get("conferenceAbbrev") if isinstance(home_team, dict) else None)
    )
    higher_seed = _first_int(
        [
            series.get("topSeedRank"),
            series.get("higherSeedRank"),
            away_team.get("seed") if isinstance(away_team, dict) else None,
            home_team.get("seed") if isinstance(home_team, dict) else None,
        ]
    )
    lower_seed = _first_int(
        [
            series.get("bottomSeedRank"),
            series.get("lowerSeedRank"),
            series.get("wildcardSeed"),
        ]
    )

    return {
        "teams": {
            "away": {"team": away_team, "score": away_wins},
            "home": {"team": home_team, "score": home_wins},
        },
        "status_text": "",
        "conference": conference,
        "higher_seed": higher_seed,
        "lower_seed": lower_seed,
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
    for key in ("series", "seriesList", "matchups", "rounds", "bracket", "bracketData"):
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
        key = (_team_abbr(away_team), _team_abbr(home_team))
        if not all(key) or key in seen:
            continue
        seen.add(key)
        deduped.append(series)
    return deduped


def _fetch_playoff_matchups() -> list[dict]:
    season = _playoff_season()
    urls = [
        f"https://api-web.nhle.com/v1/playoff-series/carousel/{season}",
        f"https://api-web.nhle.com/v1/playoff-bracket/{season}",
        "https://api-web.nhle.com/v1/playoff-series/carousel/now",
        "https://api-web.nhle.com/v1/playoff-bracket/now",
    ]

    for url in urls:
        try:
            response = _SESSION.get(url, timeout=REQUEST_TIMEOUT, headers=NHL_HEADERS)
            response.raise_for_status()
            matchups = _extract_series(response.json())
            if matchups:
                return matchups
        except Exception as exc:
            logging.debug("NHL playoffs endpoint failed (%s): %s", url, exc)
    return []


def _team_abbr_from_standings_row(row: dict) -> str:
    value = row.get("teamAbbrev")
    if isinstance(value, dict):
        value = value.get("default") or value.get("fr") or value.get("es")
    if isinstance(value, str) and value.strip():
        return value.strip().upper()
    return ""


def _team_name_from_standings_row(row: dict) -> str:
    for key in ("teamName", "teamCommonName", "teamPlaceName"):
        value = row.get(key)
        if isinstance(value, dict):
            value = value.get("default") or value.get("fr") or value.get("es")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _team_abbr_from_standings_row(row)


def _normalize_conference(value: str) -> str:
    text = (value or "").strip().lower()
    if text.startswith("e"):
        return "east"
    if text.startswith("w"):
        return "west"
    return text


def _normalize_division(value: str) -> str:
    text = (value or "").strip().lower()
    mapping = {
        "a": "atlantic",
        "atl": "atlantic",
        "atlantic": "atlantic",
        "m": "metropolitan",
        "met": "metropolitan",
        "metro": "metropolitan",
        "metropolitan": "metropolitan",
        "c": "central",
        "cen": "central",
        "central": "central",
        "p": "pacific",
        "pac": "pacific",
        "pacific": "pacific",
    }
    return mapping.get(text, text)


def _ranking_tuple(row: dict) -> tuple[int, int, int]:
    return (
        _as_int(row.get("points")) or 0,
        _as_int(row.get("regulationPlusOtWins")) or _as_int(row.get("wins")) or 0,
        _as_int(row.get("goalDifferential")) or 0,
    )


def _standings_team_obj(row: dict) -> dict:
    return {
        "abbreviation": _team_abbr_from_standings_row(row),
        "name": _team_name_from_standings_row(row),
    }


def _projected_series(higher_seed: dict, lower_seed: dict, *, conference: str, higher_seed_rank: int, lower_seed_rank: int) -> dict:
    return {
        "teams": {
            "away": {"team": _standings_team_obj(lower_seed), "score": 0},
            "home": {"team": _standings_team_obj(higher_seed), "score": 0},
        },
        "status_text": "",
        "conference": conference,
        "higher_seed": higher_seed_rank,
        "lower_seed": lower_seed_rank,
        "next_text": "TBD",
    }


def _projected_matchups_from_standings(standings: list[dict]) -> list[dict]:
    conference_rows: dict[str, list[dict]] = {"east": [], "west": []}
    by_conference_division: dict[str, dict[str, list[dict]]] = {"east": {}, "west": {}}

    for row in standings:
        if not isinstance(row, dict):
            continue
        conference = _normalize_conference(str(row.get("conferenceAbbrev") or row.get("conferenceName") or ""))
        division = _normalize_division(str(row.get("divisionAbbrev") or row.get("divisionName") or ""))
        if conference not in {"east", "west"}:
            continue
        if not division:
            continue
        if not _team_abbr_from_standings_row(row):
            continue
        conference_rows[conference].append(row)
        by_conference_division[conference].setdefault(division, []).append(row)

    matchups: list[dict] = []

    for conference, expected_divisions in (
        ("east", ("atlantic", "metropolitan")),
        ("west", ("central", "pacific")),
    ):
        divisions = by_conference_division[conference]
        if any(div not in divisions for div in expected_divisions):
            continue

        top_three_by_div: dict[str, list[dict]] = {}
        division_qualifiers: set[str] = set()
        for division in expected_divisions:
            ranked = sorted(divisions.get(division, []), key=_ranking_tuple, reverse=True)
            if len(ranked) < 3:
                top_three_by_div = {}
                break
            top_three_by_div[division] = ranked[:3]
            division_qualifiers.update(_team_abbr_from_standings_row(team) for team in ranked[:3])
        if not top_three_by_div:
            continue

        wildcard_pool = [
            row
            for row in conference_rows[conference]
            if _team_abbr_from_standings_row(row) and _team_abbr_from_standings_row(row) not in division_qualifiers
        ]
        wildcards = sorted(wildcard_pool, key=_ranking_tuple, reverse=True)[:2]
        if len(wildcards) < 2:
            continue
        wc1, wc2 = wildcards[0], wildcards[1]

        first_division, second_division = expected_divisions
        first_top = top_three_by_div[first_division]
        second_top = top_three_by_div[second_division]

        matchups.extend(
            [
                _projected_series(
                    first_top[0],
                    wc1,
                    conference=conference,
                    higher_seed_rank=1,
                    lower_seed_rank=8,
                ),
                _projected_series(
                    second_top[0],
                    wc2,
                    conference=conference,
                    higher_seed_rank=2,
                    lower_seed_rank=7,
                ),
                _projected_series(
                    first_top[1],
                    first_top[2],
                    conference=conference,
                    higher_seed_rank=3,
                    lower_seed_rank=6,
                ),
                _projected_series(
                    second_top[1],
                    second_top[2],
                    conference=conference,
                    higher_seed_rank=4,
                    lower_seed_rank=5,
                ),
            ]
        )

    return matchups


def _fetch_projected_matchups_from_standings() -> list[dict]:
    try:
        response = _SESSION.get("https://api-web.nhle.com/v1/standings/now", timeout=REQUEST_TIMEOUT, headers=NHL_HEADERS)
        response.raise_for_status()
        standings = (response.json() or {}).get("standings") or []
        return _projected_matchups_from_standings(standings)
    except Exception as exc:
        logging.debug("NHL standings endpoint failed for projected playoffs: %s", exc)
        return []


def _derive_playoff_matchups_from_games(games: list[dict]) -> list[dict]:
    results: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for game in games or []:
        teams = game.get("teams") or {}
        away = teams.get("away") or {}
        home = teams.get("home") or {}
        away_team = away.get("team") or {}
        home_team = home.get("team") or {}
        key = (_team_abbr(away_team), _team_abbr(home_team))
        if not all(key) or key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "teams": {
                    "away": {"team": away_team, "score": _as_int(away.get("leagueRecord", {}).get("wins")) or 0},
                    "home": {"team": home_team, "score": _as_int(home.get("leagueRecord", {}).get("wins")) or 0},
                },
                "status_text": "Series",
                "conference": _normalize_conference_label(
                    game.get("conference")
                    or away_team.get("conferenceAbbrev")
                    or home_team.get("conferenceAbbrev")
                ),
                "higher_seed": None,
                "lower_seed": None,
                "next_text": "TBD",
                "has_live_game": _is_live_schedule_game(game),
            }
        )
    return results


def _series_next_text_from_games(series: dict, games: list[dict]) -> str:
    teams = (series or {}).get("teams") or {}
    away = ((teams.get("away") or {}).get("team") or {})
    home = ((teams.get("home") or {}).get("team") or {})
    away_abbr = _team_abbr(away)
    home_abbr = _team_abbr(home)
    if not away_abbr or not home_abbr:
        return _normalize_next_text(series.get("next_text") or "TBD")

    now = datetime.datetime.now(CENTRAL_TIME)
    next_dt: Optional[datetime.datetime] = None
    next_has_time = True
    for game in games or []:
        game_teams = game.get("teams") or {}
        game_away_abbr = _team_abbr(((game_teams.get("away") or {}).get("team") or {}))
        game_home_abbr = _team_abbr(((game_teams.get("home") or {}).get("team") or {}))
        if {game_away_abbr, game_home_abbr} != {away_abbr, home_abbr}:
            continue
        candidate_dt, candidate_has_time = _extract_next_game_info(game)
        if not candidate_dt:
            continue
        if candidate_has_time and candidate_dt < now:
            continue
        if (not candidate_has_time) and candidate_dt.date() < now.date():
            continue
        if next_dt is None or candidate_dt < next_dt:
            next_dt = candidate_dt
            next_has_time = candidate_has_time

    if not next_dt:
        return _normalize_next_text(series.get("next_text") or "TBD")
    day_label = _next_game_day_label(next_dt, now=now)
    if not next_has_time:
        return day_label
    time_text = _format_clock(next_dt)
    return f"{day_label} {time_text}"


def _is_live_schedule_game(game: dict) -> bool:
    game_state = str((game or {}).get("gameState") or (game or {}).get("gameStatus") or "").strip().lower()
    if game_state in {"live", "crit", "critical", "in", "inprogress", "in-progress"}:
        return True
    game_status = str((game or {}).get("detailedState") or "").strip().lower()
    return any(token in game_status for token in ("live", "in progress", "intermission"))


def _series_has_live_game_from_games(series: dict, games: list[dict]) -> bool:
    teams = (series or {}).get("teams") or {}
    away = ((teams.get("away") or {}).get("team") or {})
    home = ((teams.get("home") or {}).get("team") or {})
    away_abbr = _team_abbr(away)
    home_abbr = _team_abbr(home)
    if not away_abbr or not home_abbr:
        return False

    for game in games or []:
        game_teams = game.get("teams") or {}
        game_away_abbr = _team_abbr(((game_teams.get("away") or {}).get("team") or {}))
        game_home_abbr = _team_abbr(((game_teams.get("home") or {}).get("team") or {}))
        if {game_away_abbr, game_home_abbr} != {away_abbr, home_abbr}:
            continue
        if _is_live_schedule_game(game):
            return True
    return False


def _map_schedule_game_for_series(game: dict) -> Optional[dict]:
    if not isinstance(game, dict):
        return None

    teams = {
        "away": {"team": game.get("awayTeam") or game.get("away") or {}},
        "home": {"team": game.get("homeTeam") or game.get("home") or {}},
    }
    if not _team_abbr((teams["away"] or {}).get("team") or {}):
        return None
    if not _team_abbr((teams["home"] or {}).get("team") or {}):
        return None

    return {
        "gamePk": game.get("id") or game.get("gamePk") or game.get("gameId"),
        "gameDate": game.get("startTimeUTC") or game.get("startTime") or game.get("gameDateUTC") or game.get("gameDate"),
        "gameType": game.get("gameType") or game.get("gameTypeCode") or game.get("seasonType"),
        "gameScheduleState": game.get("gameScheduleState"),
        "gameState": game.get("gameState") or game.get("gameStatus"),
        "detailedState": game.get("gameScheduleState") or game.get("gameState"),
        "seriesStatusShort": game.get("seriesStatusShort"),
        "teams": teams,
    }


def _extract_schedule_games(payload: dict) -> list[dict]:
    if not isinstance(payload, dict):
        return []

    games: list[dict] = []
    for key in ("games",):
        value = payload.get(key)
        if isinstance(value, list):
            for game in value:
                mapped = _map_schedule_game_for_series(game)
                if mapped:
                    games.append(mapped)

    weeks = payload.get("gameWeek")
    if isinstance(weeks, list):
        for week in weeks:
            if not isinstance(week, dict):
                continue
            for game in week.get("games") or []:
                mapped = _map_schedule_game_for_series(game)
                if mapped:
                    games.append(mapped)

    return games


def _is_playoff_schedule_game(game: dict) -> bool:
    game_type = str((game or {}).get("gameType") or "").strip().lower()
    if game_type in {"3", "03", "p", "playoff", "playoffs", "postseason", "post-season"}:
        return True

    schedule_state = str((game or {}).get("gameScheduleState") or "").strip().lower()
    if schedule_state in {"post", "postseason", "post-season", "playoff", "playoffs"}:
        return True

    # api-web schedule payloads can expose abbreviated series statuses for playoff rounds.
    # We only treat explicit playoff-looking statuses as a match.
    series_status = str((game or {}).get("seriesStatusShort") or "").strip().lower()
    if series_status in {"r1", "r2", "ecf", "wcf", "scf", "qf", "sf", "f"}:
        return True

    return False


def _fetch_remaining_playoff_schedule_games() -> list[dict]:
    """Fetch upcoming playoff games from the official NHL schedule feed.

    The schedule page (nhl.com/schedule) is backed by api-web schedule endpoints,
    and can provide concrete start times for future playoff games even when other
    playoff series endpoints still show TBD.
    """

    today = datetime.datetime.now(CENTRAL_TIME).date()
    urls = [
        "https://api-web.nhle.com/v1/schedule/now",
        f"https://api-web.nhle.com/v1/schedule/{today.isoformat()}",
    ]

    merged: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for url in urls:
        try:
            response = _SESSION.get(url, timeout=REQUEST_TIMEOUT, headers=NHL_HEADERS)
            response.raise_for_status()
            games = _extract_schedule_games(response.json())
        except Exception as exc:
            logging.debug("NHL official schedule endpoint failed (%s): %s", url, exc)
            continue

        for game in games:
            if not _is_playoff_schedule_game(game):
                continue
            dt, has_time = _extract_next_game_info(game)
            if not dt:
                continue
            # User-facing requirement: remaining games through June are playoffs.
            # Keep only upcoming schedule entries from now through June 30.
            season_end = datetime.date(dt.year, 6, 30)
            if dt.date() < today or dt.date() > season_end:
                continue
            away_abbr = _team_abbr(((game.get("teams") or {}).get("away") or {}).get("team") or {})
            home_abbr = _team_abbr(((game.get("teams") or {}).get("home") or {}).get("team") or {})
            if not away_abbr or not home_abbr:
                continue
            stamp = dt.isoformat() if has_time else dt.date().isoformat()
            key = (away_abbr, home_abbr, stamp)
            if key in seen:
                continue
            seen.add(key)
            merged.append(game)

    return merged


def _series_order_key(series: dict) -> tuple[int, int, str, str]:
    higher_seed = _as_int(series.get("higher_seed"))
    lower_seed = _as_int(series.get("lower_seed"))
    teams = series.get("teams") or {}
    away_team = ((teams.get("away") or {}).get("team") or {})
    home_team = ((teams.get("home") or {}).get("team") or {})
    fallback_high = min(
        _first_int([away_team.get("seed"), away_team.get("playoffSeed"), 99]) or 99,
        _first_int([home_team.get("seed"), home_team.get("playoffSeed"), 99]) or 99,
    )
    high = higher_seed if higher_seed is not None else fallback_high
    low = lower_seed if lower_seed is not None else 99
    return (high, low, _team_abbr(home_team), _team_abbr(away_team))


def _conference_buckets(series: list[dict]) -> tuple[list[dict], list[dict]]:
    west = [item for item in series if _normalize_conference_label(item.get("conference")) == "west"]
    east = [item for item in series if _normalize_conference_label(item.get("conference")) == "east"]
    unknown = [item for item in series if item not in west and item not in east]
    for idx, item in enumerate(unknown):
        (west if idx % 2 == 0 else east).append(item)
    west.sort(key=_series_order_key)
    east.sort(key=_series_order_key)
    return west, east


def _has_both_opponents(series: dict) -> bool:
    teams = (series or {}).get("teams") or {}
    away_slot = teams.get("away") or {}
    home_slot = teams.get("home") or {}
    away_team = (away_slot.get("team") or {}) if isinstance(away_slot, dict) else {}
    home_team = (home_slot.get("team") or {}) if isinstance(home_slot, dict) else {}
    away_abbr = _team_abbr(away_team) if isinstance(away_team, dict) else ""
    home_abbr = _team_abbr(home_team) if isinstance(home_team, dict) else ""
    if away_abbr and home_abbr:
        return True
    return bool(away_slot and home_slot)


def _is_completed_series(series: dict) -> bool:
    teams = (series or {}).get("teams") or {}
    away_wins = _as_int(((teams.get("away") or {}).get("score"))) or 0
    home_wins = _as_int(((teams.get("home") or {}).get("score"))) or 0
    return away_wins >= 4 or home_wins >= 4


def _series_winner_name(series: dict) -> str:
    teams = (series or {}).get("teams") or {}
    away_slot = teams.get("away") or {}
    home_slot = teams.get("home") or {}
    away_wins = _as_int(away_slot.get("score")) or 0
    home_wins = _as_int(home_slot.get("score")) or 0
    winner_team = (away_slot if away_wins >= 4 and away_wins >= home_wins else home_slot).get("team") or {}
    for key in ("teamName", "commonName", "shortName", "name"):
        candidate = _dict_localized_text(winner_team.get(key))
        if candidate:
            return candidate
    winner_abbr = _team_abbr(winner_team)
    if winner_abbr:
        return NHL_ABBR_TO_COMMON_NAME.get(winner_abbr, "")
    return ""


def _series_status_line_text(series: dict) -> str:
    if _is_completed_series(series):
        winner_name = _series_winner_name(series)
        return f"{winner_name} win!" if winner_name else "Series over"
    if _series_has_live_game(series):
        return "LIVE!"
    return _normalize_next_text(series.get("next_text") or "TBD")


def _series_has_live_game(series: dict) -> bool:
    if bool(series.get("has_live_game")):
        return True
    status_text = str(series.get("status_text") or "").strip().lower()
    return any(token in status_text for token in ("live", "in progress", "critical", "intermission"))


def _series_status_line_fill(series: dict) -> tuple[int, int, int]:
    if _is_completed_series(series):
        return (0, 200, 0)
    return SCOREBOARD_IN_PROGRESS_SCORE_COLOR if _series_has_live_game(series) else (255, 255, 255)


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
    next_text = _normalize_next_text(series.get("next_text") or "")
    if next_text != "TBD":
        return True
    status_text = str(series.get("status_text") or "").strip().lower()
    return any(token in status_text for token in ("lead", "leads", "tied", "final", "game", "live", "in progress"))


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
    score_top = top

    for idx, text in ((0, str(away.get("score", 0))), (2, ""), (4, str(home.get("score", 0)))):
        font = SCORE_FONT if idx != 2 else CENTER_FONT
        _center_text(
            draw,
            text,
            font,
            left + SERIES_COL_X[idx],
            SERIES_COL_WIDTHS[idx],
            score_top,
            SCORE_ROW_H,
            fill=(255, 255, 255),
        )

    for idx, team_side in ((1, away), (3, home)):
        team_obj = (team_side or {}).get("team", {})
        abbr = _team_abbr(team_obj)
        logo = _load_logo_cached(abbr)
        if not logo:
            team_name = (team_obj or {}).get("name") or (team_obj or {}).get("teamName") or "Unknown Team"
            log_missing_team_logo(SCREEN_ID, team_name, abbr)
            continue
        x0 = left + SERIES_COL_X[idx] + (SERIES_COL_WIDTHS[idx] - logo.width) // 2
        y0 = score_top + (SCORE_ROW_H - logo.height) // 2
        canvas.paste(logo, (x0, y0), logo)

    status_top = score_top + SCORE_ROW_H
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
    try:
        l, t, r, b = dd.textbbox((0, 0), SUBTITLE, font=STATUS_SMALL_FONT)
        subtitle_h = b - t
    except Exception:
        _, subtitle_h = dd.textsize(SUBTITLE, font=STATUS_SMALL_FONT)

    league_logo = _get_playoffs_logo()
    logo_height = league_logo.height if league_logo else 0
    logo_gap = LEAGUE_LOGO_GAP if league_logo else 0

    content_top = logo_height + logo_gap + title_h + SUBTITLE_GAP + subtitle_h + TITLE_GAP
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


def render_nhl_playoffs(display, games: list[dict], transition: bool = False) -> ScreenImage:
    _apply_style_overrides()

    upcoming_schedule_games = _fetch_remaining_playoff_schedule_games()
    merged_games = list(games or []) + upcoming_schedule_games

    series = _fetch_playoff_matchups()
    if not series:
        series = _fetch_projected_matchups_from_standings()
    if not series:
        series = _derive_playoff_matchups_from_games(merged_games)
    else:
        for item in series:
            item["next_text"] = _series_next_text_from_games(item, merged_games)
            item["has_live_game"] = _series_has_live_game_from_games(item, merged_games)
    series = _select_current_round_series(series)

    if not series:
        clear_display(display)
        img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(img)
        league_logo = _get_playoffs_logo()
        title_top = 0
        if league_logo:
            logo_x = (WIDTH - league_logo.width) // 2
            img.paste(league_logo, (logo_x, 0), league_logo)
            title_top = league_logo.height + LEAGUE_LOGO_GAP
        _center_text(draw, TITLE, TITLE_FONT, 0, WIDTH, title_top, _scale_y(40))
        subtitle_top = title_top + _scale_y(40)
        _center_text(draw, SUBTITLE, STATUS_SMALL_FONT, 0, WIDTH, subtitle_top, STATUS_ROW_H)
        msg_top = subtitle_top + STATUS_ROW_H + _scale_y(16)
        _center_text(draw, "No playoff series", STATUS_FONT, 0, WIDTH, msg_top, STATUS_ROW_H)
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
