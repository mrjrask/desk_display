"""Shared drawing helpers for sport scoreboard screens.

The individual sport modules still own API normalization and sport-specific status
formatting, but common layout primitives live here so v1/v2 scoreboards can share
logo placement, text centering, no-games, header, and scroll behavior.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Optional

from PIL import Image, ImageDraw

from config import (
    HEIGHT,
    SCOREBOARD_SCROLL_DELAY,
    SCOREBOARD_SCROLL_PAUSE_BOTTOM,
    SCOREBOARD_SCROLL_PAUSE_TOP,
    SCOREBOARD_SCROLL_STEP,
    SCOREBOARD_STANDINGS_BOTTOM_PADDING,
    WIDTH,
)
from utils import ScreenImage, clear_display, log_missing_team_logo, scroll_vertical_content

WHITE = (255, 255, 255)
SEPARATOR_COLOR = (45, 45, 45)

# Re-export the shared scroll tuning through this module so scoreboards no longer
# have to import each value directly from config.
SCROLL_STEP = SCOREBOARD_SCROLL_STEP
SCROLL_DELAY = SCOREBOARD_SCROLL_DELAY
SCROLL_PAUSE_TOP = SCOREBOARD_SCROLL_PAUSE_TOP
SCROLL_PAUSE_BOTTOM = SCOREBOARD_SCROLL_PAUSE_BOTTOM


def centered_text_bbox(draw: ImageDraw.ImageDraw, text: str, font, x: int, width: int, y: int, height: int) -> tuple[int, int]:
    """Return coordinates that center text in a rectangle."""
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        tw, th = r - l, b - t
        return x + (width - tw) // 2 - l, y + (height - th) // 2 - t
    except Exception:
        tw, th = draw.textsize(text, font=font)
        return x + (width - tw) // 2, y + (height - th) // 2


def center_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    x: int,
    width: int,
    y: int,
    height: int,
    *,
    fill=WHITE,
) -> None:
    """Draw centered text, matching the legacy scoreboard helper behavior."""
    if not text:
        return
    draw.text(centered_text_bbox(draw, text, font, x, width, y, height), text, font=font, fill=fill)


def text_height(text: str, font, *, background_color) -> int:
    dummy = Image.new("RGB", (WIDTH, 10), background_color)
    draw = ImageDraw.Draw(dummy)
    try:
        _l, t, _r, b = draw.textbbox((0, 0), text, font=font)
        return b - t
    except Exception:
        _w, h = draw.textsize(text, font=font)
        return h


def score_value(side: dict) -> Optional[int]:
    """Coerce a competitor's raw score field to an int, if possible."""
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


def team_result(side: dict, opponent: dict) -> Optional[str]:
    """Return "win"/"loss" for *side* against *opponent*, or None if undetermined."""
    for key in ("isWinner", "winner", "won"):
        value = (side or {}).get(key)
        if isinstance(value, bool):
            return "win" if value else "loss"

    side_score = score_value(side)
    opp_score = score_value(opponent)
    if side_score is not None and opp_score is not None:
        if side_score > opp_score:
            return "win"
        if side_score < opp_score:
            return "loss"
    return None


def final_results(away: dict, home: dict) -> dict:
    """Return {"away": ..., "home": ...} win/loss results, reconciled both ways."""
    away_result = team_result(away, home)
    home_result = team_result(home, away)

    if away_result == "win":
        home_result = "loss"
    elif away_result == "loss":
        home_result = "win"
    elif home_result == "win":
        away_result = "loss"
    elif home_result == "loss":
        away_result = "win"

    return {"away": away_result, "home": home_result}


def score_fill(team_key: str, *, in_progress: bool, final: bool, results: dict, in_progress_color, winning_color, losing_color, default=WHITE):
    if in_progress:
        return in_progress_color
    if final:
        result = results.get(team_key)
        if result == "loss":
            return losing_color
        if result == "win":
            return winning_color
    return default


def paste_centered_logo(canvas: Image.Image, logo: Image.Image, x: int, width: int, y: int, height: int) -> tuple[int, int]:
    x0 = x + (width - logo.width) // 2
    y0 = y + (height - logo.height) // 2
    canvas.paste(logo, (x0, y0), logo)
    return x0, y0


def draw_score_game_row(
    *,
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    game: dict,
    x_offset: int,
    top: int,
    col_x: Sequence[int],
    col_widths: Sequence[int],
    score_row_h: int,
    status_row_h: int,
    score_font,
    center_font,
    status_font,
    score_text: Callable[[dict, bool], str],
    should_display_scores: Callable[[dict], bool],
    is_game_in_progress: Callable[[dict], bool],
    is_game_final: Callable[[dict], bool],
    final_results: Callable[[dict, dict], dict],
    format_status: Callable[[dict], str],
    team_logo_abbr: Callable[[dict], str],
    load_logo: Callable[[str], Optional[Image.Image]],
    screen_id: str,
    in_progress_score_color,
    in_progress_status_color,
    final_winning_score_color,
    final_losing_score_color,
    status_x_index: int = 2,
    status_width: Optional[int] = None,
) -> None:
    """Draw the common away-score/logo/@/home-logo/score row plus status text."""
    teams = (game or {}).get("teams", {})
    away = teams.get("away", {})
    home = teams.get("home", {})
    show_scores = should_display_scores(game)
    in_progress = is_game_in_progress(game)
    final = is_game_final(game)
    results = final_results(away, home) if final else {"away": None, "home": None}

    away_text = score_text(away, show_scores)
    home_text = score_text(home, show_scores)
    for idx, text, key in ((0, away_text, "away"), (2, "@", None), (4, home_text, "home")):
        font = center_font if key is None else score_font
        fill = WHITE if key is None else score_fill(
            key,
            in_progress=in_progress,
            final=final,
            results=results,
            in_progress_color=in_progress_score_color,
            winning_color=final_winning_score_color,
            losing_color=final_losing_score_color,
        )
        center_text(draw, text, font, x_offset + col_x[idx], col_widths[idx], top, score_row_h, fill=fill)

    for idx, team_side in ((1, away), (3, home)):
        team_obj = (team_side or {}).get("team", {})
        abbr = team_logo_abbr(team_obj)
        logo = load_logo(abbr)
        if not logo:
            team_name = (
                (team_obj or {}).get("name")
                or (team_obj or {}).get("teamName")
                or (team_obj or {}).get("shortName")
                or "Unknown Team"
            )
            log_missing_team_logo(screen_id, team_name, abbr)
            continue
        paste_centered_logo(canvas, logo, x_offset + col_x[idx], col_widths[idx], top, score_row_h)

    status_top = top + score_row_h
    status_fill = in_progress_status_color if in_progress else WHITE
    center_text(
        draw,
        format_status(game),
        status_font,
        x_offset + col_x[status_x_index],
        col_widths[status_x_index] if status_width is None else status_width,
        status_top,
        status_row_h,
        fill=status_fill,
    )


def draw_title_with_logo(img: Image.Image, draw: ImageDraw.ImageDraw, *, title: str, title_font, league_logo: Optional[Image.Image], league_logo_gap: int, fill=WHITE) -> int:
    title_top = 0
    if league_logo:
        paste_centered_logo(img, league_logo, 0, WIDTH, 0, league_logo.height)
        title_top = league_logo.height + league_logo_gap
    tx, ty = centered_text_bbox(draw, title, title_font, 0, WIDTH, title_top, text_height(title, title_font, background_color=img.getpixel((0, 0))))
    draw.text((tx, ty), title, font=title_font, fill=fill)
    return title_top + text_height(title, title_font, background_color=img.getpixel((0, 0)))


def render_headered_scoreboard(*, canvas: Image.Image, title: str, title_font, league_logo: Optional[Image.Image], league_logo_gap: int, title_gap: int, background_color) -> Image.Image:
    title_h = text_height(title, title_font, background_color=background_color)
    logo_height = league_logo.height if league_logo else 0
    logo_gap = league_logo_gap if league_logo else 0
    content_top = logo_height + logo_gap + title_h + title_gap
    img_height = max(HEIGHT, content_top + canvas.height + SCOREBOARD_STANDINGS_BOTTOM_PADDING)
    img = Image.new("RGB", (WIDTH, img_height), background_color)
    draw = ImageDraw.Draw(img)
    draw_title_with_logo(img, draw, title=title, title_font=title_font, league_logo=league_logo, league_logo_gap=league_logo_gap)
    img.paste(canvas, (0, content_top))
    return img


def render_no_games_image(*, title: str, title_font, status_font, status_row_h: int, league_logo: Optional[Image.Image], league_logo_gap: int, background_color, message: str = "No games") -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), background_color)
    draw = ImageDraw.Draw(img)
    draw_title_with_logo(img, draw, title=title, title_font=title_font, league_logo=league_logo, league_logo_gap=league_logo_gap)
    center_text(draw, message, status_font, 0, WIDTH, HEIGHT // 2 - status_row_h // 2, status_row_h)
    return img


def scroll_display(display, full_img: Image.Image) -> None:
    scroll_vertical_content(
        display=display,
        content_height=full_img.height,
        viewport_width=WIDTH,
        viewport_height=HEIGHT,
        render_at_offset=lambda offset: display.image(full_img.crop((0, offset, WIDTH, offset + HEIGHT))),
        base_step=SCROLL_STEP,
        pause_start=SCROLL_PAUSE_TOP,
        pause_end=SCROLL_PAUSE_BOTTOM,
        min_frame_time=SCROLL_DELAY,
    )


def display_or_scroll_scoreboard(display, full_img: Image.Image, *, transition: bool = False) -> ScreenImage:
    if transition:
        scroll_display(display, full_img)
        return ScreenImage(full_img, displayed=True)
    if full_img.height <= HEIGHT:
        display.image(full_img)
        time.sleep(SCROLL_PAUSE_BOTTOM)
    else:
        scroll_display(display, full_img)
    return ScreenImage(full_img, displayed=True)


def display_no_games(display, img: Image.Image, *, transition: bool = False) -> ScreenImage:
    if transition:
        return ScreenImage(img, displayed=False)
    clear_display(display)
    display.image(img)
    time.sleep(SCROLL_PAUSE_BOTTOM)
    return ScreenImage(img, displayed=True)
