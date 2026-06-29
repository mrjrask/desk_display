#!/usr/bin/env python3
"""Render FIFA World Cup scoreboard."""

from __future__ import annotations

import datetime
import io
import logging
import os
import time
from typing import Any, Optional

from PIL import Image, ImageDraw

try:
    RESAMPLE = Image.ANTIALIAS
except AttributeError:
    RESAMPLE = Image.Resampling.LANCZOS

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
    SCOREBOARD_BACKGROUND_COLOR,
    SCOREBOARD_IN_PROGRESS_SCORE_COLOR,
    SCOREBOARD_FINAL_WINNING_SCORE_COLOR,
    SCOREBOARD_FINAL_LOSING_SCORE_COLOR,
    get_screen_background_color,
    get_screen_font,
    get_screen_image_scale,
    is_hyperpixel_next_layout,
    is_hyperpixel_4_square_layout,
    scale_value,
    scale_value_width,
)
from services.http_client import get_session
from utils import ScreenImage, clear_display, log_call, scroll_vertical_content

HYPERPIXEL_LAYOUT = is_hyperpixel_next_layout()
HYPERPIXEL_4_SQUARE = is_hyperpixel_4_square_layout()


def _scale_y(value: int) -> int:
    return scale_value(value) if HYPERPIXEL_LAYOUT else scale_value_width(value)


REQUEST_TIMEOUT = 10
SCREEN_ID = "World Cup Scoreboard"
LOGO_DIR = os.path.join(IMAGES_DIR, "world_cup")
ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
SCROLL_REPEAT_COUNT = 2

TITLE_GAP = scale_value(8)
BLOCK_SPACING = scale_value(10)
SCORE_ROW_H = scale_value(56)
STATUS_ROW_H = scale_value(18)

COL_WIDTHS = [
    scale_value_width(80),
    scale_value_width(60),
    scale_value_width(40),
    scale_value_width(60),
    scale_value_width(80),
]
_TOTAL_COL_WIDTH = sum(COL_WIDTHS)
_COL_LEFT = max(0, (WIDTH - _TOTAL_COL_WIDTH) // 2)
COL_X = [_COL_LEFT]
for w in COL_WIDTHS:
    COL_X.append(COL_X[-1] + w)

TEAM_LOGO_BASE_HEIGHT = scale_value_width(36) if HYPERPIXEL_LAYOUT else scale_value_width(52)
LEAGUE_LOGO_BASE_HEIGHT = TEAM_LOGO_BASE_HEIGHT
LEAGUE_LOGO_GAP = scale_value(4)

SCORE_FONT = get_screen_font(SCREEN_ID, "score", base_font=FONT_TEAM_SPORTS, default_size=39)
STATUS_FONT = get_screen_font(SCREEN_ID, "status", base_font=FONT_STATUS, default_size=28)
CENTER_FONT = get_screen_font(SCREEN_ID, "center", base_font=FONT_STATUS, default_size=28)

IN_PROGRESS_SCORE_COLOR = SCOREBOARD_IN_PROGRESS_SCORE_COLOR
IN_PROGRESS_STATUS_COLOR = IN_PROGRESS_SCORE_COLOR
FINAL_WINNING_SCORE_COLOR = SCOREBOARD_FINAL_WINNING_SCORE_COLOR
FINAL_LOSING_SCORE_COLOR = SCOREBOARD_FINAL_LOSING_SCORE_COLOR
BACKGROUND_COLOR = get_screen_background_color(SCREEN_ID, SCOREBOARD_BACKGROUND_COLOR)

_SESSION = get_session()
_REMOTE_LOGO_CACHE: dict[tuple[str, int], Optional[Image.Image]] = {}
_LEAGUE_LOGO_CACHE: dict[tuple[str, int], Optional[Image.Image]] = {}


def _team_logo_height() -> int:
    scale = get_screen_image_scale(SCREEN_ID, "team_logo", 1.0)
    target = max(1, int(round(TEAM_LOGO_BASE_HEIGHT * scale)))
    if HYPERPIXEL_4_SQUARE:
        target = max(1, int(round(target * 0.6)))
    return min(target, max(1, SCORE_ROW_H - _scale_y(8)))


def _league_logo_height() -> int:
    team_scale = get_screen_image_scale(SCREEN_ID, "team_logo", 1.0)
    scale = get_screen_image_scale(SCREEN_ID, "league_logo", team_scale)
    return max(1, int(round(LEAGUE_LOGO_BASE_HEIGHT * scale)))


def _center_text(draw: ImageDraw.ImageDraw, text: str, font, x: int, width: int, y: int, height: int, *, fill=(255, 255, 255)):
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


def _fetch_json(params: dict[str, Any]) -> dict[str, Any]:
    resp = _SESSION.get(ESPN_URL, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    return payload if isinstance(payload, dict) else {}


def _normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    comp = (event.get("competitions") or [{}])[0] or {}
    competitors = comp.get("competitors") or []
    away = next((c for c in competitors if str(c.get("homeAway", "")).lower() == "away"), competitors[0] if competitors else {})
    home = next((c for c in competitors if str(c.get("homeAway", "")).lower() == "home"), competitors[1] if len(competitors) > 1 else {})

    status_blob = comp.get("status") or event.get("status") or {}
    status_type = status_blob.get("type") if isinstance(status_blob, dict) else {}
    state = str((status_type or {}).get("state") or "").lower()
    completed = bool((status_type or {}).get("completed"))

    return {
        "id": event.get("id"),
        "date": comp.get("date") or event.get("date"),
        "status": {
            "type": {
                "state": state,
                "completed": completed,
                "shortDetail": (status_type or {}).get("shortDetail") or (status_type or {}).get("description") or status_blob.get("type", {}).get("description", ""),
            },
            "displayClock": status_blob.get("displayClock", ""),
            "period": status_blob.get("period"),
        },
        "teams": {"away": away, "home": home},
    }


def _fetch_games_for_date(day: datetime.date) -> list[dict]:
    date_str = day.strftime("%Y%m%d")
    try:
        payload = _fetch_json({"dates": date_str, "limit": 300})
    except Exception as exc:
        logging.error("Failed to fetch World Cup scoreboard for %s: %s", day, exc)
        return []

    events = payload.get("events") or []
    return [_normalize_event(event) for event in events if isinstance(event, dict)]

def _status_text(game: dict) -> str:
    status = (game or {}).get("status", {}) or {}
    type_info = status.get("type") or {}
    detail = str(type_info.get("shortDetail") or "").strip()
    state = str(type_info.get("state") or "").lower()
    if state == "pre":
        start = _parse_start_time_central(game)
        if start:
            return start
        return "Scheduled"
    return detail or "Scheduled"


def _parse_start_time_central(game: dict[str, Any]) -> str:
    raw = str((game or {}).get("date") or "").strip()
    if not raw:
        return ""
    try:
        dt = datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        local = dt.astimezone(CENTRAL_TIME)
        day_of_week = local.strftime("%a")
        month = local.month
        day = local.day
        gametime = local.strftime("%-I:%M %p")
        return f"{day_of_week} {month}/{day} {gametime}"
    except Exception:
        return ""


def _is_in_progress(game: dict) -> bool:
    return str(((game.get("status") or {}).get("type") or {}).get("state") or "").lower() == "in"


def _is_final(game: dict) -> bool:
    status_type = (game.get("status") or {}).get("type") or {}
    state = str(status_type.get("state") or "").lower()
    return state == "post" or bool(status_type.get("completed"))


def _score_text(team: dict, *, show: bool) -> str:
    if not show:
        return "—"
    score = team.get("score")
    return str(score) if score not in (None, "") else "—"


def _should_display_scores(game: dict) -> bool:
    state = str((((game or {}).get("status") or {}).get("type") or {}).get("state") or "").lower()
    return state in {"in", "post"}


def _score_value(team: dict) -> Optional[int]:
    try:
        return int(str(team.get("score")))
    except Exception:
        return None


def _score_fill(team_key: str, *, in_progress: bool, final: bool, away: dict, home: dict) -> tuple[int, int, int]:
    if in_progress:
        return IN_PROGRESS_SCORE_COLOR
    if not final:
        return (255, 255, 255)
    away_score = _score_value(away)
    home_score = _score_value(home)
    if away_score is None or home_score is None or away_score == home_score:
        return (255, 255, 255)
    if team_key == "away":
        return FINAL_WINNING_SCORE_COLOR if away_score > home_score else FINAL_LOSING_SCORE_COLOR
    return FINAL_WINNING_SCORE_COLOR if home_score > away_score else FINAL_LOSING_SCORE_COLOR


def _load_remote_logo(url: str, height: int) -> Optional[Image.Image]:
    cache_key = (url, height)
    if cache_key in _REMOTE_LOGO_CACHE:
        return _REMOTE_LOGO_CACHE[cache_key]
    try:
        resp = _SESSION.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        ratio = height / max(1, img.height)
        resized = img.resize((max(1, int(round(img.width * ratio))), height), RESAMPLE)
        _REMOTE_LOGO_CACHE[cache_key] = resized
        return resized
    except Exception as exc:
        logging.debug("Unable to load team logo %s: %s", url, exc)
        _REMOTE_LOGO_CACHE[cache_key] = None
        return None


def _team_logo_url(team: dict) -> str:
    team_blob = team.get("team") if isinstance(team.get("team"), dict) else team
    if not isinstance(team_blob, dict):
        return ""

    for source in (team_blob, team):
        logos = source.get("logos") if isinstance(source, dict) else None
        if isinstance(logos, list):
            for logo in logos:
                if isinstance(logo, dict) and logo.get("href"):
                    return str(logo["href"])
        logo_url = source.get("logo") if isinstance(source, dict) else None
        if isinstance(logo_url, str) and logo_url.strip():
            return logo_url.strip()
    return ""


def _get_league_logo() -> Optional[Image.Image]:
    h = _league_logo_height()
    cache_key = ("WC", h)
    if cache_key in _LEAGUE_LOGO_CACHE:
        return _LEAGUE_LOGO_CACHE[cache_key]
    for candidate in ("WC", "world_cup", "fifa"):
        path = os.path.join(LOGO_DIR, f"{candidate}.png")
        if not os.path.exists(path):
            continue
        try:
            img = Image.open(path).convert("RGBA")
            ratio = h / max(1, img.height)
            resized = img.resize((max(1, int(round(img.width * ratio))), h), RESAMPLE)
            _LEAGUE_LOGO_CACHE[cache_key] = resized
            return resized
        except Exception:
            continue
    _LEAGUE_LOGO_CACHE[cache_key] = None
    return None

def _render_scoreboard(games: list[dict]) -> Image.Image:
    title = "World Cup Scores"
    logo_height = _team_logo_height()

    block_h = SCORE_ROW_H + STATUS_ROW_H
    canvas_h = max(HEIGHT, len(games) * block_h + max(0, len(games) - 1) * BLOCK_SPACING)
    canvas = Image.new("RGB", (WIDTH, canvas_h), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(canvas)

    y = 0
    for idx, game in enumerate(games):
        teams = game.get("teams") or {}
        away = teams.get("away") or {}
        home = teams.get("home") or {}

        show_scores = _should_display_scores(game)
        away_score = _score_text(away, show=show_scores)
        home_score = _score_text(home, show=show_scores)
        in_progress = _is_in_progress(game)
        final = _is_final(game)

        for col_idx, text in ((0, away_score), (2, "@"), (4, home_score)):
            font = SCORE_FONT if col_idx != 2 else CENTER_FONT
            fill = (255, 255, 255)
            if col_idx == 0:
                fill = _score_fill("away", in_progress=in_progress, final=final, away=away, home=home)
            elif col_idx == 4:
                fill = _score_fill("home", in_progress=in_progress, final=final, away=away, home=home)
            _center_text(draw, text, font, COL_X[col_idx], COL_WIDTHS[col_idx], y, SCORE_ROW_H, fill=fill)

        for col_idx, team in ((1, away), (3, home)):
            url = _team_logo_url(team)
            logo = _load_remote_logo(url, logo_height) if url else None
            if not logo:
                continue
            x0 = COL_X[col_idx] + (COL_WIDTHS[col_idx] - logo.width) // 2
            y0 = y + (SCORE_ROW_H - logo.height) // 2
            canvas.paste(logo, (x0, y0), logo)

        status_fill = IN_PROGRESS_STATUS_COLOR if in_progress else (255, 255, 255)
        _center_text(draw, _status_text(game), STATUS_FONT, COL_X[0], sum(COL_WIDTHS), y + SCORE_ROW_H, STATUS_ROW_H, fill=status_fill)

        y += block_h
        if idx < len(games) - 1:
            sep_y = y + BLOCK_SPACING // 2
            draw.line((10, sep_y, WIDTH - 10, sep_y), fill=(45, 45, 45))
            y += BLOCK_SPACING

    dummy = Image.new("RGB", (WIDTH, 8), BACKGROUND_COLOR)
    dd = ImageDraw.Draw(dummy)
    try:
        l, t, r, b = dd.textbbox((0, 0), title, font=FONT_TITLE_SPORTS)
        title_h = b - t
    except Exception:
        _, title_h = dd.textsize(title, font=FONT_TITLE_SPORTS)

    league_logo = _get_league_logo()
    league_h = league_logo.height if league_logo else 0
    gap = LEAGUE_LOGO_GAP if league_logo else 0

    content_top = league_h + gap + title_h + TITLE_GAP
    total_h = max(HEIGHT, content_top + canvas.height + SCOREBOARD_STANDINGS_BOTTOM_PADDING)
    out = Image.new("RGB", (WIDTH, total_h), BACKGROUND_COLOR)
    draw_out = ImageDraw.Draw(out)

    if league_logo:
        out.paste(league_logo, ((WIDTH - league_logo.width) // 2, 0), league_logo)
    title_top = league_h + gap
    try:
        l, t, r, b = draw_out.textbbox((0, 0), title, font=FONT_TITLE_SPORTS)
        draw_out.text(((WIDTH - (r - l)) // 2 - l, title_top - t), title, font=FONT_TITLE_SPORTS, fill=(255, 255, 255))
    except Exception:
        tw, _ = draw_out.textsize(title, font=FONT_TITLE_SPORTS)
        draw_out.text(((WIDTH - tw) // 2, title_top), title, font=FONT_TITLE_SPORTS, fill=(255, 255, 255))

    out.paste(canvas, (0, content_top))
    return out


def _repeated_scroll_image(img: Image.Image) -> Image.Image:
    if SCROLL_REPEAT_COUNT <= 1:
        return img

    repeated = Image.new("RGB", (WIDTH, img.height * SCROLL_REPEAT_COUNT), BACKGROUND_COLOR)
    for index in range(SCROLL_REPEAT_COUNT):
        repeated.paste(img, (0, img.height * index))
    return repeated


def _scroll_display(display, img: Image.Image):
    scroll_img = _repeated_scroll_image(img)
    scroll_vertical_content(
        display=display,
        content_height=scroll_img.height,
        viewport_width=WIDTH,
        viewport_height=HEIGHT,
        render_at_offset=lambda offset: display.image(scroll_img.crop((0, offset, WIDTH, offset + HEIGHT))),
        base_step=SCOREBOARD_SCROLL_STEP,
        pause_start=SCOREBOARD_SCROLL_PAUSE_TOP,
        pause_end=SCOREBOARD_SCROLL_PAUSE_BOTTOM,
        min_frame_time=SCOREBOARD_SCROLL_DELAY,
    )


def _viewport_image(img: Image.Image) -> Image.Image:
    return img if img.height <= HEIGHT else img.crop((0, 0, WIDTH, HEIGHT))


def _render_world_cup_scoreboard_v1(display, games: list[dict] | None, transition: bool = False) -> ScreenImage:
    games = games or []
    if not games:
        clear_display(display)
        title = "World Cup Scores"
        img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(img)
        league_logo = _get_league_logo()
        title_top = 0
        title_height = 0
        if league_logo:
            img.paste(league_logo, ((WIDTH - league_logo.width) // 2, 0), league_logo)
            title_top = league_logo.height + LEAGUE_LOGO_GAP
        try:
            l, t, r, b = draw.textbbox((0, 0), title, font=FONT_TITLE_SPORTS)
            title_height = b - t
            draw.text(((WIDTH - (r - l)) // 2 - l, title_top - t), title, font=FONT_TITLE_SPORTS, fill=(255, 255, 255))
        except Exception:
            tw, th = draw.textsize(title, font=FONT_TITLE_SPORTS)
            title_height = th
            draw.text(((WIDTH - tw) // 2, title_top), title, font=FONT_TITLE_SPORTS, fill=(255, 255, 255))

        no_games_top = max(title_top + title_height + LEAGUE_LOGO_GAP, HEIGHT // 2 - STATUS_ROW_H // 2)
        _center_text(draw, "No World Cup games", STATUS_FONT, 0, WIDTH, no_games_top, STATUS_ROW_H)
        if transition:
            return ScreenImage(img, displayed=False)
        display.image(img)
        time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)
        return ScreenImage(img, displayed=True)

    full = _render_scoreboard(games)
    if len(games) == 1:
        viewport = _viewport_image(full)
        if transition:
            return ScreenImage(viewport, displayed=False)
        display.image(viewport)
        time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)
        return ScreenImage(viewport, displayed=True)

    if transition:
        _scroll_display(display, full)
        return ScreenImage(full, displayed=True)

    if full.height <= HEIGHT:
        display.image(full)
        time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)
    else:
        _scroll_display(display, full)
    return ScreenImage(full, displayed=True)


def render_world_cup_scoreboard(display, games: list[dict] | None, transition: bool = False) -> ScreenImage:
    return _render_world_cup_scoreboard_v1(display, games or [], transition=transition)


@log_call
def draw_world_cup_scoreboard(display, transition: bool = False) -> ScreenImage:
    from services.sports.world_cup import fetch_scoreboard

    games = fetch_scoreboard()
    return render_world_cup_scoreboard(display, games, transition=transition)


def _scoreboard_date(now: Optional[datetime.datetime] = None) -> datetime.date:
    if now is None:
        now = datetime.datetime.now(CENTRAL_TIME)
    cutoff = now.replace(hour=10, minute=10, second=0, microsecond=0)
    return (now.date() - datetime.timedelta(days=1)) if now < cutoff else now.date()
