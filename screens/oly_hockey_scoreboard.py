#!/usr/bin/env python3
"""Olympic hockey scoreboards using the NHL scoreboard layout."""

from __future__ import annotations

import datetime
import logging
import os
import time
from typing import Any, Dict, Optional

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
    get_screen_background_color,
    get_screen_font,
    get_screen_image_scale,
    is_hyperpixel_next_layout,
    scale_value,
    scale_value_width,
)
from utils import ScreenImage, clear_display, load_team_logo, log_call, standard_scoreboard_league_logo_height
from services.http_client import get_session

HYPERPIXEL_LAYOUT = is_hyperpixel_next_layout()


def _scale_y(value: int) -> int:
    return scale_value(value) if HYPERPIXEL_LAYOUT else scale_value_width(value)


TITLE_GAP = _scale_y(8)
BLOCK_SPACING = _scale_y(10)
SCORE_ROW_H = _scale_y(56)
STATUS_ROW_H = _scale_y(18)
REQUEST_TIMEOUT = 10

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
LOGO_DIR = os.path.join(IMAGES_DIR, "oly")
LEAGUE_LOGO_KEYS = ("olympics", "OLY", "iihf")
LEAGUE_LOGO_GAP = _scale_y(4)
TEAM_LOGO_BASE_HEIGHT = scale_value_width(36) if HYPERPIXEL_LAYOUT else scale_value_width(52)
LEAGUE_LOGO_BASE_HEIGHT = standard_scoreboard_league_logo_height(TEAM_LOGO_BASE_HEIGHT)
LOGO_HEIGHT = TEAM_LOGO_BASE_HEIGHT
LEAGUE_LOGO_HEIGHT = LEAGUE_LOGO_BASE_HEIGHT

IN_PROGRESS_SCORE_COLOR = SCOREBOARD_IN_PROGRESS_SCORE_COLOR
IN_PROGRESS_STATUS_COLOR = IN_PROGRESS_SCORE_COLOR
FINAL_WINNING_SCORE_COLOR = SCOREBOARD_FINAL_WINNING_SCORE_COLOR
FINAL_LOSING_SCORE_COLOR = SCOREBOARD_FINAL_LOSING_SCORE_COLOR

_SESSION = get_session()
_LOGO_CACHE: dict[tuple[str, int], Optional[Image.Image]] = {}
_LEAGUE_LOGO_CACHE: dict[int, Optional[Image.Image]] = {}

COUNTRY_NAME_TO_CODE = {
    "canada": "CAN", "united states": "USA", "usa": "USA", "sweden": "SWE", "finland": "FIN",
    "czechia": "CZE", "czech republic": "CZE", "switzerland": "SUI", "germany": "GER",
    "slovakia": "SVK", "latvia": "LAT", "denmark": "DEN", "norway": "NOR", "france": "FRA",
    "italy": "ITA", "japan": "JPN", "china": "CHN", "korea": "KOR", "south korea": "KOR",
    "austria": "AUT", "great britain": "GBR", "britain": "GBR",
}


COMPETITIONS = {
    "men": {
        "title": "Olympic Hockey - Men",
        "screen_id": "Olympic Hockey Men Scoreboard",
        "url": "https://site.api.espn.com/apis/site/v2/sports/hockey/mens-olympics/scoreboard",
    },
    "women": {
        "title": "Olympic Hockey - Women",
        "screen_id": "Olympic Hockey Women Scoreboard",
        "url": "https://site.api.espn.com/apis/site/v2/sports/hockey/womens-olympics/scoreboard",
    },
}


def _apply_style_overrides(screen_id: str) -> tuple[Any, Any, Any, tuple[int, int, int]]:
    global LOGO_HEIGHT, LEAGUE_LOGO_HEIGHT
    score_font = get_screen_font(screen_id, "score", base_font=FONT_TEAM_SPORTS, default_size=39)
    status_font = get_screen_font(screen_id, "status", base_font=FONT_STATUS, default_size=28)
    center_font = get_screen_font(screen_id, "center", base_font=FONT_STATUS, default_size=28)
    background = get_screen_background_color(screen_id, SCOREBOARD_BACKGROUND_COLOR)
    team_scale = get_screen_image_scale(screen_id, "team_logo", 1.0)
    LOGO_HEIGHT = max(1, int(round(TEAM_LOGO_BASE_HEIGHT * team_scale)))
    league_scale = get_screen_image_scale(screen_id, "league_logo", team_scale)
    LEAGUE_LOGO_HEIGHT = max(1, int(round(LEAGUE_LOGO_BASE_HEIGHT * league_scale)))
    return score_font, status_font, center_font, background


def _country_code(team: dict) -> str:
    for key in ("abbreviation", "shortDisplayName", "displayName", "name"):
        value = team.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            if len(text) == 3 and text.isalpha():
                return text.upper()
            mapped = COUNTRY_NAME_TO_CODE.get(text.lower())
            if mapped:
                return mapped
    return ""


def _team_fallback_text(team: dict) -> str:
    code = _country_code(team)
    if code:
        return code
    for key in ("shortDisplayName", "displayName", "name"):
        value = team.get(key)
        if isinstance(value, str) and value.strip():
            cleaned = "".join(ch for ch in value.upper() if ch.isalpha())
            if cleaned:
                return cleaned[:3]
    return "?"


def _load_logo_cached(code: str) -> Optional[Image.Image]:
    code = (code or "").strip().upper()
    if not code:
        return None
    cache_token = (code, LOGO_HEIGHT)
    if cache_token in _LOGO_CACHE:
        return _LOGO_CACHE[cache_token]
    for candidate in (code, code.lower()):
        logo = load_team_logo(LOGO_DIR, candidate, height=LOGO_HEIGHT, box_size=LOGO_HEIGHT, trim=True)
        if logo is not None:
            _LOGO_CACHE[cache_token] = logo
            return logo
    _LOGO_CACHE[cache_token] = None
    return None


def _get_league_logo() -> Optional[Image.Image]:
    if LEAGUE_LOGO_HEIGHT in _LEAGUE_LOGO_CACHE:
        return _LEAGUE_LOGO_CACHE[LEAGUE_LOGO_HEIGHT]
    for key in LEAGUE_LOGO_KEYS:
        logo = load_team_logo(LOGO_DIR, key, height=LEAGUE_LOGO_HEIGHT, box_size=LEAGUE_LOGO_HEIGHT)
        if logo is not None:
            _LEAGUE_LOGO_CACHE[LEAGUE_LOGO_HEIGHT] = logo
            return logo
    _LEAGUE_LOGO_CACHE[LEAGUE_LOGO_HEIGHT] = None
    return None


def _center_text(draw, text, font, x, width, y, height, *, fill=(255, 255, 255)):
    if not text:
        return
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    draw.text((x + (width - (r - l)) // 2 - l, y + (height - (b - t)) // 2 - t), text, font=font, fill=fill)


def _status_text(game: dict) -> str:
    status = (game.get("status") or {}).get("type") or {}
    state = (status.get("state") or "").lower()
    short = (status.get("shortDetail") or status.get("detail") or "")
    if state == "pre":
        dt = game.get("date")
        if isinstance(dt, str) and dt:
            try:
                stamp = datetime.datetime.fromisoformat(dt.replace("Z", "+00:00")).astimezone(CENTRAL_TIME)
                return stamp.strftime("%-I:%M %p")
            except Exception:
                pass
    return short or ("Final" if state == "post" else "Live")


def _is_in_progress(game: dict) -> bool:
    state = (((game.get("status") or {}).get("type") or {}).get("state") or "").lower()
    return state == "in"


def _is_final(game: dict) -> bool:
    state = (((game.get("status") or {}).get("type") or {}).get("state") or "").lower()
    return state == "post"


def _draw_game_block(canvas, draw, game, top, *, score_font, status_font, center_font):
    away = game.get("away", {})
    home = game.get("home", {})
    in_progress = _is_in_progress(game)
    final = _is_final(game)
    away_score = str(away.get("score", "")) if (in_progress or final) else ""
    home_score = str(home.get("score", "")) if (in_progress or final) else ""
    away_fill = home_fill = (255, 255, 255)
    if in_progress:
        away_fill = home_fill = IN_PROGRESS_SCORE_COLOR
    elif final:
        if away.get("score", -1) > home.get("score", -1):
            away_fill, home_fill = FINAL_WINNING_SCORE_COLOR, FINAL_LOSING_SCORE_COLOR
        elif home.get("score", -1) > away.get("score", -1):
            home_fill, away_fill = FINAL_WINNING_SCORE_COLOR, FINAL_LOSING_SCORE_COLOR

    _center_text(draw, away_score, score_font, COL_X[0], COL_WIDTHS[0], top, SCORE_ROW_H, fill=away_fill)
    _center_text(draw, "@", center_font, COL_X[2], COL_WIDTHS[2], top, SCORE_ROW_H)
    _center_text(draw, home_score, score_font, COL_X[4], COL_WIDTHS[4], top, SCORE_ROW_H, fill=home_fill)

    for idx, team in ((1, away), (3, home)):
        logo = _load_logo_cached(_country_code(team))
        if logo:
            canvas.paste(logo, (COL_X[idx] + (COL_WIDTHS[idx] - logo.width) // 2, top + (SCORE_ROW_H - logo.height) // 2), logo)
        else:
            _center_text(draw, _team_fallback_text(team), center_font, COL_X[idx], COL_WIDTHS[idx], top, SCORE_ROW_H)

    _center_text(
        draw,
        _status_text(game),
        status_font,
        COL_X[2],
        COL_WIDTHS[2],
        top + SCORE_ROW_H,
        STATUS_ROW_H,
        fill=IN_PROGRESS_STATUS_COLOR if in_progress else (255, 255, 255),
    )


def _compose_canvas(games, *, score_font, status_font, center_font, background):
    if not games:
        return Image.new("RGB", (WIDTH, HEIGHT), background)
    block_h = SCORE_ROW_H + STATUS_ROW_H
    total_h = block_h * len(games) + BLOCK_SPACING * max(0, len(games) - 1)
    canvas = Image.new("RGB", (WIDTH, total_h), background)
    draw = ImageDraw.Draw(canvas)
    y = 0
    for i, game in enumerate(games):
        _draw_game_block(canvas, draw, game, y, score_font=score_font, status_font=status_font, center_font=center_font)
        y += block_h
        if i < len(games) - 1:
            draw.line((10, y + BLOCK_SPACING // 2, WIDTH - 10, y + BLOCK_SPACING // 2), fill=(45, 45, 45))
            y += BLOCK_SPACING
    return canvas


def _fetch_games(url: str) -> list[dict]:
    try:
        response = _SESSION.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        logging.error("Failed to fetch Olympic hockey scoreboard: %s", exc)
        return []

    games: list[dict] = []
    for event in data.get("events", []) or []:
        competitors = (((event.get("competitions") or [{}])[0]).get("competitors") or [])
        away = next((c for c in competitors if (c.get("homeAway") or "").lower() == "away"), None)
        home = next((c for c in competitors if (c.get("homeAway") or "").lower() == "home"), None)
        if not away or not home:
            continue
        away_team = away.get("team") or {}
        home_team = home.get("team") or {}
        games.append(
            {
                "date": event.get("date"),
                "status": event.get("status") or {},
                "away": {
                    "abbreviation": away_team.get("abbreviation"),
                    "displayName": away_team.get("displayName"),
                    "shortDisplayName": away_team.get("shortDisplayName"),
                    "score": int(away.get("score") or 0),
                },
                "home": {
                    "abbreviation": home_team.get("abbreviation"),
                    "displayName": home_team.get("displayName"),
                    "shortDisplayName": home_team.get("shortDisplayName"),
                    "score": int(home.get("score") or 0),
                },
            }
        )
    return games


def _draw(display, division: str, *, transition: bool = False) -> ScreenImage:
    meta = COMPETITIONS[division]
    title = meta["title"]
    screen_id = meta["screen_id"]
    score_font, status_font, center_font, background = _apply_style_overrides(screen_id)
    games = _fetch_games(meta["url"])
    canvas = _compose_canvas(games, score_font=score_font, status_font=status_font, center_font=center_font, background=background)

    dummy = Image.new("RGB", (WIDTH, 10), background)
    title_h = ImageDraw.Draw(dummy).textbbox((0, 0), title, font=TITLE_FONT)[3]
    league_logo = _get_league_logo()
    logo_h = league_logo.height if league_logo else 0
    gap = LEAGUE_LOGO_GAP if league_logo else 0
    content_top = logo_h + gap + title_h + TITLE_GAP
    full_img = Image.new("RGB", (WIDTH, max(HEIGHT, content_top + canvas.height)), background)
    draw = ImageDraw.Draw(full_img)
    if league_logo:
        full_img.paste(league_logo, ((WIDTH - league_logo.width) // 2, 0), league_logo)
    draw.text(((WIDTH - draw.textbbox((0, 0), title, font=TITLE_FONT)[2]) // 2, logo_h + gap), title, font=TITLE_FONT, fill=(255, 255, 255))
    full_img.paste(canvas, (0, content_top))

    if transition:
        clear_display(display)
    if full_img.height <= HEIGHT:
        display.image(full_img)
        return ScreenImage(full_img, displayed=True)

    max_offset = full_img.height - HEIGHT
    display.image(full_img.crop((0, 0, WIDTH, HEIGHT)))
    time.sleep(SCOREBOARD_SCROLL_PAUSE_TOP)
    for offset in range(SCOREBOARD_SCROLL_STEP, max_offset + 1, SCOREBOARD_SCROLL_STEP):
        display.image(full_img.crop((0, offset, WIDTH, offset + HEIGHT)))
        time.sleep(SCOREBOARD_SCROLL_DELAY)
    time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)
    return ScreenImage(full_img, displayed=True)


@log_call
def draw_olympic_mens_hockey_scoreboard(display, transition: bool = False) -> ScreenImage:
    return _draw(display, "men", transition=transition)


@log_call
def draw_olympic_womens_hockey_scoreboard(display, transition: bool = False) -> ScreenImage:
    return _draw(display, "women", transition=transition)
