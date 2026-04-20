#!/usr/bin/env python3
"""Render NBA playoff series matchups using scoreboard-style layout."""

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
    SCOREBOARD_BACKGROUND_COLOR,
    get_screen_background_color,
    get_screen_font,
    get_screen_image_scale,
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
    standard_scoreboard_team_logo_height,
    standard_scoreboard_league_logo_height,
)
from screens.nba_scoreboard import _center_text, _team_logo_abbr, _get_league_logo, _NBA_HEADERS

HYPERPIXEL_LAYOUT = is_hyperpixel_next_layout()
HYPERPIXEL_4_SQUARE = is_hyperpixel_4_square_layout()


def _scale_y(value: int) -> int:
    return scale_value(value) if HYPERPIXEL_LAYOUT else scale_value_width(value)


TITLE = "NBA Playoffs"
SCREEN_ID = "NBA Playoffs"
TITLE_GAP = _scale_y(8)
BLOCK_SPACING = _scale_y(10)
SCORE_ROW_H = _scale_y(56)
STATUS_ROW_H = _scale_y(18)
REQUEST_TIMEOUT = 8

COL_WIDTHS = [
    scale_value_width(70),
    scale_value_width(60),
    scale_value_width(60),
    scale_value_width(60),
    scale_value_width(70),
]
_TOTAL_COL_WIDTH = sum(COL_WIDTHS)
_COL_LEFT = max(0, (WIDTH - _TOTAL_COL_WIDTH) // 2)
COL_X = [_COL_LEFT]
for w in COL_WIDTHS:
    COL_X.append(COL_X[-1] + w)

TITLE_FONT = FONT_TITLE_SPORTS
LOGO_DIR = os.path.join(IMAGES_DIR, "nba")
TEAM_LOGO_BASE_HEIGHT = standard_scoreboard_team_logo_height(HEIGHT)
LEAGUE_LOGO_BASE_HEIGHT = standard_scoreboard_league_logo_height(TEAM_LOGO_BASE_HEIGHT)
if HYPERPIXEL_4_SQUARE:
    LEAGUE_LOGO_BASE_HEIGHT = min(LEAGUE_LOGO_BASE_HEIGHT, scale_value_width(40))
LOGO_HEIGHT = TEAM_LOGO_BASE_HEIGHT
LEAGUE_LOGO_GAP = _scale_y(4)

_SESSION = get_session()


def _scoreboard_fonts() -> tuple:
    score = get_screen_font(SCREEN_ID, "score", base_font=FONT_TEAM_SPORTS, default_size=43)
    status = get_screen_font(SCREEN_ID, "status", base_font=FONT_STATUS, default_size=28)
    center = get_screen_font(SCREEN_ID, "center", base_font=FONT_STATUS, default_size=28)
    return score, status, center


SCORE_FONT, STATUS_FONT, CENTER_FONT = _scoreboard_fonts()
BACKGROUND_COLOR = get_screen_background_color(SCREEN_ID, SCOREBOARD_BACKGROUND_COLOR)
_LOGO_CACHE: dict[tuple[str, int], Optional[Image.Image]] = {}


def _apply_style_overrides() -> None:
    global SCORE_FONT, STATUS_FONT, CENTER_FONT, LOGO_HEIGHT, BACKGROUND_COLOR

    SCORE_FONT, STATUS_FONT, CENTER_FONT = _scoreboard_fonts()
    BACKGROUND_COLOR = get_screen_background_color(SCREEN_ID, SCOREBOARD_BACKGROUND_COLOR)
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


def _team_from_series(series: dict, *keys: str) -> dict:
    for key in keys:
        value = series.get(key)
        if isinstance(value, dict):
            return value
    return {}


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


def _next_day_label(dt: datetime.datetime, *, now: Optional[datetime.datetime] = None) -> Optional[str]:
    now_dt = now or datetime.datetime.now(CENTRAL_TIME)
    if dt.date() == now_dt.date():
        return "Tonight"
    if dt.date() == (now_dt + datetime.timedelta(days=1)).date():
        return "Tomorrow"
    return None


def _format_next_text(series: dict) -> str:
    next_dt = _extract_next_game_dt(
        series.get("nextGameStartTimeUTC")
        or series.get("nextGameDateTime")
        or series.get("nextGameTimeUTC")
    )
    if next_dt is None:
        text = str(series.get("nextGameLabel") or series.get("nextGameText") or "").strip()
        return f"Next: {text}" if text else "Next: TBD"
    day_label = _next_day_label(next_dt)
    if day_label:
        return f"Next: {day_label} {next_dt.strftime('%-I:%M %p')}"
    return f"Next: {next_dt.month}/{next_dt.day} {next_dt.strftime('%-I:%M %p')}"


def _normalize_next_text(text: Any) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "Next: TBD"
    normalized = re.sub(
        r"\s+(?:ET|EST|EDT|CT|CST|CDT|MT|MST|MDT|PT|PST|PDT|UTC)$",
        "",
        raw,
        flags=re.IGNORECASE,
    ).strip()
    normalized = re.sub(
        r"(?<!\d)(\d{1,2})/(\d{1,2})(?!\d)",
        lambda match: f"{int(match.group(1))}/{int(match.group(2))}",
        normalized,
    )
    return normalized or "Next: TBD"


def _normalize_series_item(series: dict) -> Optional[dict]:
    if not isinstance(series, dict):
        return None

    away_team = _team_from_series(series, "awayTeam", "topSeedTeam", "topSeed", "team1")
    home_team = _team_from_series(series, "homeTeam", "bottomSeedTeam", "bottomSeed", "team2")

    away_wins = _first_present_int(
        [
            series.get("awayWins"),
            series.get("topSeedWins"),
            series.get("team1Wins"),
            series.get("topSeedTeamWins"),
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
            home_team.get("wins") if isinstance(home_team, dict) else None,
            home_team.get("seriesWins") if isinstance(home_team, dict) else None,
        ]
    )

    away_abbr = _team_logo_abbr(away_team)
    home_abbr = _team_logo_abbr(home_team)
    if not away_abbr or not home_abbr:
        return None

    status = (
        series.get("seriesStatusShort")
        or series.get("seriesStatus")
        or series.get("seriesText")
        or series.get("roundLabel")
        or "Series"
    )

    return {
        "teams": {
            "away": {"team": away_team, "score": away_wins},
            "home": {"team": home_team, "score": home_wins},
        },
        "status_text": str(status),
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
        key = (_team_logo_abbr(away_team), _team_logo_abbr(home_team))
        if not all(key) or key in seen:
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


def _derive_playoff_matchups_from_games(games: list[dict]) -> list[dict]:
    results: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for game in games or []:
        away_team = (game.get("awayTeam") or game.get("away") or {})
        home_team = (game.get("homeTeam") or game.get("home") or {})
        key = (_team_logo_abbr(away_team), _team_logo_abbr(home_team))
        if not all(key) or key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "teams": {
                    "away": {"team": away_team, "score": 0},
                    "home": {"team": home_team, "score": 0},
                },
                "status_text": "Series",
                "next_text": "Next: TBD",
            }
        )
    return results


def _draw_game_block(canvas: Image.Image, draw: ImageDraw.ImageDraw, game: dict, top: int):
    teams = (game or {}).get("teams", {})
    away = teams.get("away", {})
    home = teams.get("home", {})

    away_text = str(away.get("score", 0))
    home_text = str(home.get("score", 0))

    score_top = top
    for idx, text in ((0, away_text), (2, "@"), (4, home_text)):
        font = SCORE_FONT if idx != 2 else CENTER_FONT
        _center_text(draw, text, font, COL_X[idx], COL_WIDTHS[idx], score_top, SCORE_ROW_H, fill=(255, 255, 255))

    for idx, team_side in ((1, away), (3, home)):
        team_obj = (team_side or {}).get("team", {})
        abbr = _team_logo_abbr(team_obj)
        logo = _load_logo_cached(abbr)
        if not logo:
            team_name = (team_obj or {}).get("teamName") or (team_obj or {}).get("teamCity") or "Unknown Team"
            log_missing_team_logo(SCREEN_ID, team_name, abbr)
            continue
        x0 = COL_X[idx] + (COL_WIDTHS[idx] - logo.width) // 2
        y0 = score_top + (SCORE_ROW_H - logo.height) // 2
        canvas.paste(logo, (x0, y0), logo)

    status_top = score_top + SCORE_ROW_H
    status_text = _normalize_next_text(game.get("next_text") or game.get("status_text") or "Next: TBD")
    _center_text(draw, status_text, STATUS_FONT, COL_X[2], COL_WIDTHS[2], status_top, STATUS_ROW_H, fill=(255, 255, 255))


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


def render_nba_playoffs(display, games: list[dict], transition: bool = False) -> ScreenImage:
    _apply_style_overrides()

    series = _fetch_playoff_matchups()
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
