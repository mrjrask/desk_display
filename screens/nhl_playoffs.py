#!/usr/bin/env python3
"""Render NHL playoff series matchups using scoreboard-style layout."""

from __future__ import annotations

import datetime
import logging
import os
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

HYPERPIXEL_LAYOUT = is_hyperpixel_next_layout()
HYPERPIXEL_4_SQUARE = is_hyperpixel_4_square_layout()


def _scale_y(value: int) -> int:
    return scale_value(value) if HYPERPIXEL_LAYOUT else scale_value_width(value)


TITLE = "NHL Playoffs"
SCREEN_ID = "NHL Playoffs"
TITLE_GAP = _scale_y(8)
BLOCK_SPACING = _scale_y(10)
SCORE_ROW_H = _scale_y(56)
STATUS_ROW_H = _scale_y(18)
REQUEST_TIMEOUT = 8
PAIR_SPACING = max(8, scale_value_width(16))
SERIES_COL_WIDTHS = [
    scale_value_width(42),
    scale_value_width(32),
    scale_value_width(16),
    scale_value_width(32),
    scale_value_width(42),
]
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
# Match NHL Scoreboard v2 logo baseline sizing.
TEAM_LOGO_BASE_HEIGHT = scale_value_width(26)
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


def _apply_style_overrides() -> None:
    global SCORE_FONT, STATUS_FONT, STATUS_SMALL_FONT, CENTER_FONT, LOGO_HEIGHT, BACKGROUND_COLOR

    SCORE_FONT, STATUS_FONT, STATUS_SMALL_FONT, CENTER_FONT = _scoreboard_fonts()
    BACKGROUND_COLOR = (0, 0, 0)
    team_scale = get_screen_image_scale(SCREEN_ID, "team_logo", 1.0)
    target_logo_height = max(1, int(round(TEAM_LOGO_BASE_HEIGHT * team_scale)))
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


def _first_int(values: list[Any]) -> Optional[int]:
    for value in values:
        parsed = _as_int(value)
        if parsed is not None:
            return parsed
    return None


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


def _extract_next_game_dt(series: dict) -> Optional[datetime.datetime]:
    candidate_keys = (
        "nextGameStartTimeUTC",
        "nextGameStartTime",
        "nextGameDateTime",
        "nextGameDate",
        "startTimeUTC",
        "gameDate",
    )
    for key in candidate_keys:
        dt = _parse_datetime(series.get(key))
        if dt:
            return dt
    next_game = series.get("nextGame")
    if isinstance(next_game, dict):
        for key in ("startTimeUTC", "startTime", "gameDate", "date"):
            dt = _parse_datetime(next_game.get(key))
            if dt:
                return dt
    return None


def _format_next_text(series: dict) -> str:
    dt = _extract_next_game_dt(series)
    if not dt:
        return "Next: TBD"
    month = dt.month
    day = dt.day
    time_text = dt.strftime("%-I:%M %p")
    return f"Next: {month}/{day} {time_text} CDT"


def _normalize_series_item(series: dict) -> Optional[dict]:
    if not isinstance(series, dict):
        return None

    away_team = _team_from_series(series, "awayTeam", "topSeedTeam", "team1", "homeTeam1")
    home_team = _team_from_series(series, "homeTeam", "bottomSeedTeam", "team2", "homeTeam2")

    away_wins = (
        _as_int(series.get("awayWins"))
        or _as_int(series.get("topSeedWins"))
        or _as_int(series.get("team1Wins"))
        or 0
    )
    home_wins = (
        _as_int(series.get("homeWins"))
        or _as_int(series.get("bottomSeedWins"))
        or _as_int(series.get("team2Wins"))
        or 0
    )

    away_abbr = _team_logo_abbr(away_team)
    home_abbr = _team_logo_abbr(home_team)
    if not away_abbr or not home_abbr:
        return None

    status = (
        series.get("seriesStatusShort")
        or series.get("seriesStatus")
        or series.get("seriesTitle")
        or series.get("roundLabel")
        or "Series"
    )
    conference = _normalize_conference_label(
        series.get("conferenceAbbrev")
        or series.get("conferenceName")
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
        "status_text": str(status),
        "conference": conference,
        "higher_seed": higher_seed,
        "lower_seed": lower_seed,
        "next_text": _format_next_text(series),
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
        key = (_team_logo_abbr(away_team), _team_logo_abbr(home_team))
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
        "status_text": "Projected",
        "conference": conference,
        "higher_seed": higher_seed_rank,
        "lower_seed": lower_seed_rank,
        "next_text": "Next: TBD",
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
        key = (_team_logo_abbr(away_team), _team_logo_abbr(home_team))
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
                "next_text": "Next: TBD",
            }
        )
    return results


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


def _draw_series_block(canvas: Image.Image, draw: ImageDraw.ImageDraw, series: dict, *, left: int, top: int):
    teams = (series or {}).get("teams", {})
    away = teams.get("away", {})
    home = teams.get("home", {})
    score_top = top

    for idx, text in ((0, str(away.get("score", 0))), (2, "-"), (4, str(home.get("score", 0)))):
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
        abbr = _team_logo_abbr(team_obj)
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
        str(series.get("status_text") or "Series"),
        STATUS_FONT,
        left,
        SERIES_WIDTH,
        status_top,
        STATUS_ROW_H,
        fill=(220, 220, 220),
    )
    _center_text(
        draw,
        series.get("next_text") or "Next: TBD",
        STATUS_SMALL_FONT,
        left,
        SERIES_WIDTH,
        status_top + STATUS_ROW_H,
        STATUS_ROW_H,
        fill=(255, 255, 255),
    )


def _compose_canvas(series: list[dict]) -> Image.Image:
    if not series:
        return Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    west_series, east_series = _conference_buckets(series)
    rows = max(len(west_series), len(east_series))
    block_height = SCORE_ROW_H + (STATUS_ROW_H * 2)
    total_height = block_height * rows
    if rows > 1:
        total_height += BLOCK_SPACING * (rows - 1)
    canvas = Image.new("RGB", (WIDTH, total_height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(canvas)

    y = 0
    for idx in range(rows):
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

    league_logo = _get_league_logo()
    logo_height = league_logo.height if league_logo else 0
    logo_gap = LEAGUE_LOGO_GAP if league_logo else 0

    content_top = logo_height + logo_gap + title_h + TITLE_GAP
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

    series = _fetch_playoff_matchups()
    if not series:
        series = _fetch_projected_matchups_from_standings()
    if not series:
        series = _derive_playoff_matchups_from_games(games)

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
        msg_top = title_top + _scale_y(56)
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
