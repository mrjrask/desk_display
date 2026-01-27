#!/usr/bin/env python3
"""
nhl_scoreboard_v2.py

Dual-game NHL scoreboard layout - displays 2 games per line.
Compact layout with smaller fonts and logos for a denser presentation.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import time
from typing import Optional

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
    SCOREBOARD_SCROLL_PAUSE_TOP,
    SCOREBOARD_SCROLL_PAUSE_BOTTOM,
    SCOREBOARD_BACKGROUND_COLOR,
    SCOREBOARD_IN_PROGRESS_SCORE_COLOR,
    SCOREBOARD_FINAL_WINNING_SCORE_COLOR,
    SCOREBOARD_FINAL_LOSING_SCORE_COLOR,
    get_screen_background_color,
    get_screen_font,
    get_screen_image_scale,
)
from utils import (
    ScreenImage,
    clear_display,
    load_team_logo,
    log_call,
    standard_scoreboard_league_logo_height,
    standard_scoreboard_team_logo_height,
)

# Import shared NHL data fetching logic
from screens.nhl_scoreboard import (
    _fetch_games_for_date,
    _scoreboard_date,
    _is_game_in_progress,
    _is_game_final,
    _should_display_scores,
    _score_text,
    _score_value,
    _team_result,
    _final_results,
    _format_status,
    _team_logo_abbr,
    _get_league_logo,
)

# ─── Constants ────────────────────────────────────────────────────────────────
TITLE = "NHL Scoreboard v2"
TITLE_GAP = 8
BLOCK_SPACING = 8
PAIR_SPACING = 4
SCORE_ROW_H = 30
STATUS_ROW_H = 14
GAME_WIDTH = 160

# Dual-game column layout (per game, 160px wide)
# [Score 40][Logo 30][@ 20][Logo 30][Score 40] = 160
GAME_COL_WIDTHS = [40, 30, 20, 30, 40]
GAME_COL_X = [0]
for w in GAME_COL_WIDTHS:
    GAME_COL_X.append(GAME_COL_X[-1] + w)

SCREEN_ID = "NHL Scoreboard v2"
TITLE_FONT = FONT_TITLE_SPORTS
TEAM_LOGO_BASE_HEIGHT = standard_scoreboard_team_logo_height(HEIGHT, compact=True)
LEAGUE_LOGO_BASE_HEIGHT = standard_scoreboard_league_logo_height(TEAM_LOGO_BASE_HEIGHT)
LOGO_HEIGHT = TEAM_LOGO_BASE_HEIGHT
LEAGUE_LOGO_HEIGHT = LEAGUE_LOGO_BASE_HEIGHT
SCORE_FONT = get_screen_font(
    SCREEN_ID,
    "score",
    base_font=FONT_TEAM_SPORTS,
    default_size=20,
)
STATUS_FONT = get_screen_font(
    SCREEN_ID,
    "status",
    base_font=FONT_STATUS,
    default_size=18,
)
CENTER_FONT = get_screen_font(
    SCREEN_ID,
    "center",
    base_font=FONT_STATUS,
    default_size=18,
)
LOGO_DIR = os.path.join(IMAGES_DIR, "nhl")
LEAGUE_LOGO_KEYS = ("NHL", "nhl")
LEAGUE_LOGO_GAP = 4
LEAGUE_LOGO_HEIGHT = LEAGUE_LOGO_BASE_HEIGHT

IN_PROGRESS_SCORE_COLOR = SCOREBOARD_IN_PROGRESS_SCORE_COLOR
IN_PROGRESS_STATUS_COLOR = IN_PROGRESS_SCORE_COLOR
FINAL_WINNING_SCORE_COLOR = SCOREBOARD_FINAL_WINNING_SCORE_COLOR
FINAL_LOSING_SCORE_COLOR = SCOREBOARD_FINAL_LOSING_SCORE_COLOR
BACKGROUND_COLOR = get_screen_background_color(SCREEN_ID, SCOREBOARD_BACKGROUND_COLOR)

_LOGO_CACHE: dict[tuple[str, int], Optional[Image.Image]] = {}


def _apply_style_overrides() -> None:
    global SCORE_FONT, STATUS_FONT, CENTER_FONT, LOGO_HEIGHT, LEAGUE_LOGO_HEIGHT, BACKGROUND_COLOR

    SCORE_FONT = get_screen_font(
        SCREEN_ID,
        "score",
        base_font=FONT_TEAM_SPORTS,
        default_size=20,
    )
    STATUS_FONT = get_screen_font(
        SCREEN_ID,
        "status",
        base_font=FONT_STATUS,
        default_size=18,
    )
    CENTER_FONT = get_screen_font(
        SCREEN_ID,
        "center",
        base_font=FONT_STATUS,
        default_size=18,
    )
    BACKGROUND_COLOR = get_screen_background_color(SCREEN_ID, SCOREBOARD_BACKGROUND_COLOR)
    team_scale = get_screen_image_scale(SCREEN_ID, "team_logo", 1.0)
    LOGO_HEIGHT = max(1, int(round(TEAM_LOGO_BASE_HEIGHT * team_scale)))
    league_scale = get_screen_image_scale(SCREEN_ID, "league_logo", team_scale)
    LEAGUE_LOGO_HEIGHT = max(1, int(round(LEAGUE_LOGO_BASE_HEIGHT * league_scale)))


def _load_logo_cached(abbr: str) -> Optional[Image.Image]:
    key = (abbr or "").strip()
    if not key:
        return None
    cache_key = key.upper()
    height = LOGO_HEIGHT
    cache_token = (cache_key, height)
    if cache_token in _LOGO_CACHE:
        return _LOGO_CACHE[cache_token]

    candidates = [cache_key, cache_key.lower(), cache_key.title()]
    for candidate in candidates:
        path = os.path.join(LOGO_DIR, f"{candidate}.png")
        if os.path.exists(path):
            logo = load_team_logo(LOGO_DIR, candidate, height=height, box_size=height, trim=True)
            _LOGO_CACHE[cache_token] = logo
            return logo

    _LOGO_CACHE[cache_token] = None
    return None


def _center_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    x: int,
    width: int,
    y: int,
    height: int,
    *,
    fill=(255, 255, 255),
):
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


def _score_fill(
    team_key: str, *, in_progress: bool, final: bool, results: dict
) -> tuple[int, int, int]:
    if in_progress:
        return IN_PROGRESS_SCORE_COLOR
    if final:
        result = results.get(team_key)
        if result == "loss":
            return FINAL_LOSING_SCORE_COLOR
        if result == "win":
            return FINAL_WINNING_SCORE_COLOR
    return (255, 255, 255)


def _draw_single_game(
    canvas: Image.Image, draw: ImageDraw.ImageDraw, game: dict, x_offset: int, top: int
):
    """Draw a single game within the dual-game layout."""
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

    # Draw scores and @ symbol
    for idx, text in ((0, away_text), (2, "@"), (4, home_text)):
        font = SCORE_FONT if idx != 2 else CENTER_FONT
        if idx == 0:
            fill = _score_fill("away", in_progress=in_progress, final=final, results=results)
        elif idx == 4:
            fill = _score_fill("home", in_progress=in_progress, final=final, results=results)
        else:
            fill = (255, 255, 255)
        _center_text(
            draw,
            text,
            font,
            x_offset + GAME_COL_X[idx],
            GAME_COL_WIDTHS[idx],
            score_top,
            SCORE_ROW_H,
            fill=fill,
        )

    # Draw team logos
    for idx, team_side, team_key in ((1, away, "away"), (3, home, "home")):
        team_obj = (team_side or {}).get("team", {})
        abbr = _team_logo_abbr(team_obj)
        logo = _load_logo_cached(abbr)
        if not logo:
            continue
        x0 = x_offset + GAME_COL_X[idx] + (GAME_COL_WIDTHS[idx] - logo.width) // 2
        y0 = score_top + (SCORE_ROW_H - logo.height) // 2
        canvas.paste(logo, (x0, y0), logo)

    # Draw status
    status_top = score_top + SCORE_ROW_H
    status_text = _format_status(game)
    status_fill = IN_PROGRESS_STATUS_COLOR if in_progress else (255, 255, 255)
    _center_text(
        draw,
        status_text,
        STATUS_FONT,
        x_offset + GAME_COL_X[0],
        GAME_WIDTH,
        status_top,
        STATUS_ROW_H,
        fill=status_fill,
    )


def _draw_game_pair(
    canvas: Image.Image, draw: ImageDraw.ImageDraw, game1: dict, game2: Optional[dict], top: int
):
    """Draw a pair of games side by side."""
    _draw_single_game(canvas, draw, game1, 0, top)
    if game2:
        _draw_single_game(canvas, draw, game2, GAME_WIDTH, top)


def _compose_canvas(games: list[dict]) -> Image.Image:
    if not games:
        return Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)

    # Calculate pairs
    pairs = []
    for i in range(0, len(games), 2):
        game1 = games[i]
        game2 = games[i + 1] if i + 1 < len(games) else None
        pairs.append((game1, game2))

    # Calculate canvas height
    pair_height = SCORE_ROW_H + STATUS_ROW_H
    total_height = pair_height * len(pairs)
    if len(pairs) > 1:
        total_height += BLOCK_SPACING * (len(pairs) - 1)

    canvas = Image.new("RGB", (WIDTH, total_height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(canvas)

    y = 0
    for idx, (game1, game2) in enumerate(pairs):
        _draw_game_pair(canvas, draw, game1, game2, y)
        y += pair_height
        if idx < len(pairs) - 1:
            sep_y = y + BLOCK_SPACING // 2
            draw.line((10, sep_y, WIDTH - 10, sep_y), fill=(45, 45, 45))
            y += BLOCK_SPACING

    return canvas


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

    wait_for_skip = getattr(display, "wait_for_skip", None)
    skip_requested = getattr(display, "skip_requested", None)

    def _should_skip() -> bool:
        return bool(skip_requested and skip_requested())

    def _sleep(duration: float) -> bool:
        if callable(wait_for_skip):
            return bool(wait_for_skip(duration))
        time.sleep(duration)
        return False

    max_offset = full_img.height - HEIGHT
    frame = full_img.crop((0, 0, WIDTH, HEIGHT))
    display.image(frame)
    if _sleep(SCOREBOARD_SCROLL_PAUSE_TOP):
        return

    target_frame_time = 0.016  # ~60 FPS
    for offset in range(SCOREBOARD_SCROLL_STEP, max_offset + 1, SCOREBOARD_SCROLL_STEP):
        if _should_skip():
            return

        frame_start = time.time()
        frame = full_img.crop((0, offset, WIDTH, offset + HEIGHT))
        display.image(frame)
        elapsed = time.time() - frame_start
        sleep_time = max(0, target_frame_time - elapsed)
        if sleep_time > 0 and _sleep(sleep_time):
            return

    _sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)


@log_call
def draw_nhl_scoreboard_v2(display, transition: bool = False) -> ScreenImage:
    _apply_style_overrides()
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
        _center_text(
            draw, "No games", STATUS_FONT, 0, WIDTH, HEIGHT // 2 - STATUS_ROW_H // 2, STATUS_ROW_H
        )
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


if __name__ == "__main__":
    from utils import Display

    disp = Display()
    try:
        draw_nhl_scoreboard_v2(disp)
    finally:
        clear_display(disp)
