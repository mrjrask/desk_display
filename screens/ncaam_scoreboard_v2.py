#!/usr/bin/env python3
"""Dual-game Men's NCAA basketball scoreboard."""

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
    SCOREBOARD_STANDINGS_BOTTOM_PADDING,
    SCOREBOARD_BACKGROUND_COLOR,
    SCOREBOARD_IN_PROGRESS_SCORE_COLOR,
    SCOREBOARD_FINAL_WINNING_SCORE_COLOR,
    SCOREBOARD_FINAL_LOSING_SCORE_COLOR,
    get_screen_background_color,
    get_screen_font,
    scale_value,
    scale_value_width,
)
from utils import ScreenImage, clear_display, scroll_vertical_content

from screens.ncaam_scoreboard import (
    _mode_title_and_logo,
    _team_logo_height,
    _load_remote_logo,
    _team_logo_url,
    _rank_for_display,
    _draw_seed,
    _seed_text_for_display,
    _draw_rank,
    _is_in_progress,
    _is_final,
    _score_text,
    _score_fill,
    _should_display_scores,
    _status_text,
    _center_text,
    _get_league_logo,
    _scoreboard_mode,
    MODE_TOURNAMENT,
    render_ncaam_scoreboard as render_ncaam_scoreboard_v1,
)

SCREEN_ID = "NCAAM Scoreboard v2"
MIN_GAMES_FOR_V2_LAYOUT = 6
V2_DISABLED_RESOLUTIONS = {(320, 240), (240, 320)}
TITLE_GAP = scale_value(8)
BLOCK_SPACING = scale_value(8)
PAIR_SPACING = scale_value_width(4)
SCORE_ROW_H = scale_value(28)
STATUS_ROW_H = scale_value(12)
GAME_COL_WIDTHS = [
    scale_value_width(40),
    scale_value_width(30),
    scale_value_width(20),
    scale_value_width(30),
    scale_value_width(40),
]
GAME_WIDTH = sum(GAME_COL_WIDTHS)
GAME_COL_X = [0]
for w in GAME_COL_WIDTHS:
    GAME_COL_X.append(GAME_COL_X[-1] + w)

SCORE_FONT = get_screen_font(SCREEN_ID, "score", base_font=FONT_TEAM_SPORTS, default_size=20)
STATUS_FONT = get_screen_font(SCREEN_ID, "status", base_font=FONT_STATUS, default_size=18)
CENTER_FONT = get_screen_font(SCREEN_ID, "center", base_font=FONT_STATUS, default_size=18)
BACKGROUND_COLOR = get_screen_background_color(SCREEN_ID, SCOREBOARD_BACKGROUND_COLOR)

IN_PROGRESS_SCORE_COLOR = SCOREBOARD_IN_PROGRESS_SCORE_COLOR
IN_PROGRESS_STATUS_COLOR = IN_PROGRESS_SCORE_COLOR
FINAL_WINNING_SCORE_COLOR = SCOREBOARD_FINAL_WINNING_SCORE_COLOR
FINAL_LOSING_SCORE_COLOR = SCOREBOARD_FINAL_LOSING_SCORE_COLOR


def _v2_team_logo_height() -> int:
    return min(_team_logo_height(), max(1, SCORE_ROW_H - scale_value(4)))


def _draw_single_game(canvas: Image.Image, draw: ImageDraw.ImageDraw, game: dict, x_offset: int, top: int):
    teams = (game or {}).get("teams", {})
    selected_mode = _scoreboard_mode()
    away = teams.get("away", {})
    home = teams.get("home", {})

    show_scores = _should_display_scores(game)
    away_text = _score_text(away, show=show_scores)
    home_text = _score_text(home, show=show_scores)
    in_progress = _is_in_progress(game)
    final = _is_final(game)

    for idx, text in ((0, away_text), (2, "@"), (4, home_text)):
        font = SCORE_FONT if idx != 2 else CENTER_FONT
        fill = (255, 255, 255)
        if idx == 0:
            fill = _score_fill("away", in_progress=in_progress, final=final, away=away, home=home)
        elif idx == 4:
            fill = _score_fill("home", in_progress=in_progress, final=final, away=away, home=home)
        _center_text(draw, text, font, x_offset + GAME_COL_X[idx], GAME_COL_WIDTHS[idx], top, SCORE_ROW_H, fill=fill)

    logo_h = _v2_team_logo_height()
    for idx, team in ((1, away), (3, home)):
        logo = _load_remote_logo(_team_logo_url(team), logo_h)
        if not logo:
            continue
        x0 = x_offset + GAME_COL_X[idx] + (GAME_COL_WIDTHS[idx] - logo.width) // 2
        y0 = top + (SCORE_ROW_H - logo.height) // 2
        canvas.paste(logo, (x0, y0), logo)
        _draw_rank(draw, _rank_for_display(team), x0, y0, logo.width, logo.height)
        if selected_mode == MODE_TOURNAMENT and _seed_text_for_display(team):
            _draw_seed(draw, _seed_text_for_display(team), x0, y0, logo.height)

    _center_text(
        draw,
        _status_text(game),
        STATUS_FONT,
        x_offset + GAME_COL_X[0],
        GAME_WIDTH,
        top + SCORE_ROW_H,
        STATUS_ROW_H,
        fill=IN_PROGRESS_STATUS_COLOR if in_progress else (255, 255, 255),
    )


def _render_scoreboard(games: list[dict]) -> Image.Image:
    pairs = [games[i:i + 2] for i in range(0, len(games), 2)]
    pair_h = SCORE_ROW_H + STATUS_ROW_H
    body_h = max(HEIGHT, len(pairs) * pair_h + max(0, len(pairs) - 1) * BLOCK_SPACING)
    body = Image.new("RGB", (WIDTH, body_h), BACKGROUND_COLOR)
    draw_body = ImageDraw.Draw(body)

    y = 0
    for idx, pair in enumerate(pairs):
        game1 = pair[0]
        game2 = pair[1] if len(pair) > 1 else None
        _draw_single_game(body, draw_body, game1, 0, y)
        if game2:
            _draw_single_game(body, draw_body, game2, GAME_WIDTH + PAIR_SPACING, y)
        y += pair_h
        if idx < len(pairs) - 1:
            sep_y = y + BLOCK_SPACING // 2
            draw_body.line((10, sep_y, WIDTH - 10, sep_y), fill=(45, 45, 45))
            y += BLOCK_SPACING

    title, _ = _mode_title_and_logo()
    dd = ImageDraw.Draw(Image.new("RGB", (WIDTH, 10), BACKGROUND_COLOR))
    try:
        l, t, r, b = dd.textbbox((0, 0), title, font=FONT_TITLE_SPORTS)
        title_h = b - t
    except Exception:
        _, title_h = dd.textsize(title, font=FONT_TITLE_SPORTS)

    league_logo = _get_league_logo()
    logo_h = league_logo.height if league_logo else 0
    logo_gap = scale_value(4) if league_logo else 0
    content_top = logo_h + logo_gap + title_h + TITLE_GAP

    total_h = max(HEIGHT, content_top + body.height + SCOREBOARD_STANDINGS_BOTTOM_PADDING)
    out = Image.new("RGB", (WIDTH, total_h), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(out)
    if league_logo:
        out.paste(league_logo, ((WIDTH - league_logo.width) // 2, 0), league_logo)
    try:
        l, t, r, b = draw.textbbox((0, 0), title, font=FONT_TITLE_SPORTS)
        draw.text(((WIDTH - (r - l)) // 2 - l, logo_h + logo_gap - t), title, font=FONT_TITLE_SPORTS, fill=(255, 255, 255))
    except Exception:
        tw, _ = draw.textsize(title, font=FONT_TITLE_SPORTS)
        draw.text(((WIDTH - tw) // 2, logo_h + logo_gap), title, font=FONT_TITLE_SPORTS, fill=(255, 255, 255))

    out.paste(body, (0, content_top))
    return out


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


def render_ncaam_scoreboard_v2(display, games: list[dict] | None, transition: bool = False) -> ScreenImage:
    games = games or []
    if (WIDTH, HEIGHT) in V2_DISABLED_RESOLUTIONS:
        return render_ncaam_scoreboard_v1(display, games, transition=transition)
    if len(games) < MIN_GAMES_FOR_V2_LAYOUT:
        return render_ncaam_scoreboard_v1(display, games, transition=transition)

    if not games:
        clear_display(display)
        img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
        _center_text(ImageDraw.Draw(img), "No games today", STATUS_FONT, 0, WIDTH, HEIGHT // 2 - STATUS_ROW_H, STATUS_ROW_H)
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
