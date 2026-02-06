#!/usr/bin/env python3
"""Olympic hockey scoreboards v2 (two games per row), modeled after NHL Scoreboard v2."""

from __future__ import annotations

import time
from typing import Optional

from PIL import Image, ImageDraw

from config import (
    WIDTH,
    HEIGHT,
    FONT_TITLE_SPORTS,
    FONT_TEAM_SPORTS,
    FONT_STATUS,
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

from screens.oly_hockey_scoreboard import _country_code, _fetch_games, _team_fallback_text
from screens.oly_hockey_scoreboard import COMPETITIONS, LEAGUE_LOGO_KEYS, LOGO_DIR

HYPERPIXEL_LAYOUT = is_hyperpixel_next_layout()


def _scale_y(value: int) -> int:
    return scale_value(value) if HYPERPIXEL_LAYOUT else scale_value_width(value)


TITLE_GAP = _scale_y(8)
BLOCK_SPACING = _scale_y(8)
PAIR_SPACING = scale_value_width(4)
SCORE_ROW_H = _scale_y(30)
STATUS_ROW_H = _scale_y(14)
GAME_COL_WIDTHS = [scale_value_width(40), scale_value_width(30), scale_value_width(20), scale_value_width(30), scale_value_width(40)]
GAME_WIDTH = sum(GAME_COL_WIDTHS)
GAME_COL_X = [0]
for w in GAME_COL_WIDTHS:
    GAME_COL_X.append(GAME_COL_X[-1] + w)

TITLE_FONT = FONT_TITLE_SPORTS
TEAM_LOGO_BASE_HEIGHT = scale_value_width(26)
LEAGUE_LOGO_BASE_HEIGHT = standard_scoreboard_league_logo_height(TEAM_LOGO_BASE_HEIGHT)
LOGO_HEIGHT = TEAM_LOGO_BASE_HEIGHT
LEAGUE_LOGO_HEIGHT = LEAGUE_LOGO_BASE_HEIGHT
_LOGO_CACHE: dict[tuple[str, int], Optional[Image.Image]] = {}
_LEAGUE_LOGO_CACHE: dict[int, Optional[Image.Image]] = {}


def _apply_style(screen_id: str):
    global LOGO_HEIGHT, LEAGUE_LOGO_HEIGHT
    score_font = get_screen_font(screen_id, "score", base_font=FONT_TEAM_SPORTS, default_size=20)
    status_font = get_screen_font(screen_id, "status", base_font=FONT_STATUS, default_size=18)
    center_font = get_screen_font(screen_id, "center", base_font=FONT_STATUS, default_size=18)
    background = get_screen_background_color(screen_id, SCOREBOARD_BACKGROUND_COLOR)
    team_scale = get_screen_image_scale(screen_id, "team_logo", 1.0)
    LOGO_HEIGHT = max(1, int(round(TEAM_LOGO_BASE_HEIGHT * team_scale)))
    league_scale = get_screen_image_scale(screen_id, "league_logo", team_scale)
    LEAGUE_LOGO_HEIGHT = max(1, int(round(LEAGUE_LOGO_BASE_HEIGHT * league_scale)))
    return score_font, status_font, center_font, background


def _load_logo_cached(code: str) -> Optional[Image.Image]:
    code = (code or "").strip().upper()
    if not code:
        return None
    token = (code, LOGO_HEIGHT)
    if token in _LOGO_CACHE:
        return _LOGO_CACHE[token]
    for candidate in (code, code.lower()):
        logo = load_team_logo(LOGO_DIR, candidate, height=LOGO_HEIGHT, box_size=LOGO_HEIGHT, trim=True)
        if logo:
            _LOGO_CACHE[token] = logo
            return logo
    _LOGO_CACHE[token] = None
    return None


def _get_league_logo() -> Optional[Image.Image]:
    if LEAGUE_LOGO_HEIGHT in _LEAGUE_LOGO_CACHE:
        return _LEAGUE_LOGO_CACHE[LEAGUE_LOGO_HEIGHT]
    for key in LEAGUE_LOGO_KEYS:
        logo = load_team_logo(LOGO_DIR, key, height=LEAGUE_LOGO_HEIGHT, box_size=LEAGUE_LOGO_HEIGHT)
        if logo:
            _LEAGUE_LOGO_CACHE[LEAGUE_LOGO_HEIGHT] = logo
            return logo
    _LEAGUE_LOGO_CACHE[LEAGUE_LOGO_HEIGHT] = None
    return None


def _center(draw, text, font, x, width, y, height, fill=(255, 255, 255)):
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    draw.text((x + (width - (r - l)) // 2 - l, y + (height - (b - t)) // 2 - t), text, font=font, fill=fill)


def _draw_game(canvas, draw, game, left, top, *, score_font, status_font, center_font):
    away = game.get("away", {})
    home = game.get("home", {})
    state = ((((game.get("status") or {}).get("type") or {}).get("state")) or "").lower()
    in_progress = state == "in"
    final = state == "post"
    show_scores = in_progress or final
    away_score = str(away.get("score", "")) if show_scores else ""
    home_score = str(home.get("score", "")) if show_scores else ""
    away_fill = home_fill = (255, 255, 255)
    if in_progress:
        away_fill = home_fill = SCOREBOARD_IN_PROGRESS_SCORE_COLOR
    elif final:
        if away.get("score", -1) > home.get("score", -1):
            away_fill, home_fill = SCOREBOARD_FINAL_WINNING_SCORE_COLOR, SCOREBOARD_FINAL_LOSING_SCORE_COLOR
        elif home.get("score", -1) > away.get("score", -1):
            home_fill, away_fill = SCOREBOARD_FINAL_WINNING_SCORE_COLOR, SCOREBOARD_FINAL_LOSING_SCORE_COLOR

    _center(draw, away_score, score_font, left + GAME_COL_X[0], GAME_COL_WIDTHS[0], top, SCORE_ROW_H, away_fill)
    _center(draw, "@", center_font, left + GAME_COL_X[2], GAME_COL_WIDTHS[2], top, SCORE_ROW_H)
    _center(draw, home_score, score_font, left + GAME_COL_X[4], GAME_COL_WIDTHS[4], top, SCORE_ROW_H, home_fill)

    for idx, team in ((1, away), (3, home)):
        logo = _load_logo_cached(_country_code(team))
        if logo:
            canvas.paste(logo, (left + GAME_COL_X[idx] + (GAME_COL_WIDTHS[idx] - logo.width)//2, top + (SCORE_ROW_H-logo.height)//2), logo)
        else:
            _center(draw, _team_fallback_text(team), center_font, left + GAME_COL_X[idx], GAME_COL_WIDTHS[idx], top, SCORE_ROW_H)

    short = (((game.get("status") or {}).get("type") or {}).get("shortDetail") or "")
    _center(draw, short or ("Final" if final else "Live"), status_font, left, GAME_WIDTH, top + SCORE_ROW_H, STATUS_ROW_H, SCOREBOARD_IN_PROGRESS_SCORE_COLOR if in_progress else (255, 255, 255))


def _render(display, division: str, *, transition=False) -> ScreenImage:
    meta = COMPETITIONS[division]
    title = f"{meta['title']} v2"
    screen_id = f"{meta['screen_id']} v2"
    score_font, status_font, center_font, bg = _apply_style(screen_id)
    games = _fetch_games(meta["url"])

    rows = [games[i:i+2] for i in range(0, len(games), 2)] or [[]]
    row_h = SCORE_ROW_H + STATUS_ROW_H
    total_h = len(rows)*row_h + max(0, len(rows)-1)*BLOCK_SPACING
    canvas = Image.new("RGB", (WIDTH, max(row_h, total_h)), bg)
    draw = ImageDraw.Draw(canvas)
    y = 0
    for r_i, row in enumerate(rows):
        if len(row) == 1:
            _draw_game(canvas, draw, row[0], (WIDTH-GAME_WIDTH)//2, y, score_font=score_font, status_font=status_font, center_font=center_font)
        elif len(row) == 2:
            all_w = GAME_WIDTH*2 + PAIR_SPACING
            left = (WIDTH-all_w)//2
            _draw_game(canvas, draw, row[0], left, y, score_font=score_font, status_font=status_font, center_font=center_font)
            _draw_game(canvas, draw, row[1], left+GAME_WIDTH+PAIR_SPACING, y, score_font=score_font, status_font=status_font, center_font=center_font)
        y += row_h
        if r_i < len(rows)-1:
            draw.line((8, y + BLOCK_SPACING//2, WIDTH-8, y + BLOCK_SPACING//2), fill=(45,45,45))
            y += BLOCK_SPACING

    title_bbox = draw.textbbox((0, 0), title, font=TITLE_FONT)
    title_h = title_bbox[3] - title_bbox[1]
    league_logo = _get_league_logo()
    logo_h = league_logo.height if league_logo else 0
    content_top = logo_h + (4 if league_logo else 0) + title_h + TITLE_GAP
    full = Image.new("RGB", (WIDTH, max(HEIGHT, content_top + canvas.height)), bg)
    d2 = ImageDraw.Draw(full)
    if league_logo:
        full.paste(league_logo, ((WIDTH-league_logo.width)//2, 0), league_logo)
    _center(d2, title, TITLE_FONT, 0, WIDTH, logo_h + (4 if league_logo else 0), title_h, (255,255,255))
    full.paste(canvas, (0, content_top))

    if transition:
        clear_display(display)
    if full.height <= HEIGHT:
        display.image(full)
        return ScreenImage(full, displayed=True)

    display.image(full.crop((0, 0, WIDTH, HEIGHT)))
    time.sleep(SCOREBOARD_SCROLL_PAUSE_TOP)
    for offset in range(SCOREBOARD_SCROLL_STEP, full.height - HEIGHT + 1, SCOREBOARD_SCROLL_STEP):
        display.image(full.crop((0, offset, WIDTH, offset + HEIGHT)))
        time.sleep(SCOREBOARD_SCROLL_DELAY)
    time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)
    return ScreenImage(full, displayed=True)


@log_call
def draw_olympic_mens_hockey_scoreboard_v2(display, transition: bool = False) -> ScreenImage:
    return _render(display, "men", transition=transition)


@log_call
def draw_olympic_womens_hockey_scoreboard_v2(display, transition: bool = False) -> ScreenImage:
    return _render(display, "women", transition=transition)
