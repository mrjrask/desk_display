#!/usr/bin/env python3
"""Render FIFA World Cup scoreboard."""

from __future__ import annotations

import datetime
import io
import logging
import os
import time
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont

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
NO_GAMES_TEXT = "No Games Today"
OLY_LOGO_PATH = os.path.join(IMAGES_DIR, "oly", "WC.png")

ROUND_QUARTERFINALS = "Quarterfinals"
ROUND_SEMIFINALS = "Semifinals"
ROUND_FINALS = "Finals"
ROUND_LABEL_KEY = "round_label"
ROUND_GAME_KEY = "round_game"

WORLD_CUP_ROUNDS = {
    ROUND_QUARTERFINALS: {
        "dates": (
            datetime.date(2026, 7, 9),
            datetime.date(2026, 7, 10),
            datetime.date(2026, 7, 11),
        ),
        "show_from": datetime.date(2026, 7, 8),
        "show_through": datetime.date(2026, 7, 11),
    },
    ROUND_SEMIFINALS: {
        "dates": (datetime.date(2026, 7, 14), datetime.date(2026, 7, 15)),
        "show_from": datetime.date(2026, 7, 12),
        "show_through": datetime.date(2026, 7, 16),
    },
    ROUND_FINALS: {
        "dates": (datetime.date(2026, 7, 18), datetime.date(2026, 7, 19)),
        "show_from": datetime.date(2026, 7, 17),
        "show_through": datetime.date(2026, 7, 20),
    },
}

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
PK_SCORE_FONT_SCALE = 0.7
PREGAME_SCORE_DISPLAY = os.environ.get("WORLD_CUP_PREGAME_SCORE_DISPLAY", "dash").strip().lower()
PREGAME_SCORE_ABBREVIATION_VALUES = {"abbr", "abbrev", "abbreviation", "team", "team_abbreviation"}


def _scale_font(font: ImageFont.FreeTypeFont, scale: float) -> ImageFont.FreeTypeFont:
    size = getattr(font, "size", None)
    path = getattr(font, "path", None)
    if not size or not path:
        return font
    try:
        return ImageFont.truetype(path, max(1, int(round(size * scale))))
    except OSError:
        logging.debug("Unable to scale font %s by %s", path, scale)
        return font


PK_SCORE_FONT = _scale_font(SCORE_FONT, PK_SCORE_FONT_SCALE)
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
_OLY_LOGO_CACHE: dict[int, Optional[Image.Image]] = {}


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


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int, int, int]:
    try:
        return draw.textbbox((0, 0), text, font=font)
    except Exception:
        width, height = draw.textsize(text, font=font)
        return (0, 0, width, height)


def _team_abbreviation(team: dict) -> str:
    for source in (team, team.get("team") if isinstance(team.get("team"), dict) else None):
        if not isinstance(source, dict):
            continue
        for key in ("abbreviation", "abbr", "shortDisplayName", "displayName", "name"):
            value = str(source.get(key) or "").strip()
            if value:
                return value[:3].upper()
    return ""


def _show_pregame_team_abbreviation() -> bool:
    return PREGAME_SCORE_DISPLAY in PREGAME_SCORE_ABBREVIATION_VALUES


def _score_text_segments(team: dict, *, show: bool) -> list[tuple[str, ImageFont.FreeTypeFont]]:
    if not show:
        abbreviation = _team_abbreviation(team) if _show_pregame_team_abbreviation() else ""
        return [(abbreviation or "—", STATUS_FONT if abbreviation else SCORE_FONT)]
    score = team.get("score")
    base_score = str(score) if score not in (None, "") else "—"
    penalty_score = _penalty_score_value(team)
    if penalty_score is not None:
        return [(base_score, SCORE_FONT), (f" ({penalty_score})", PK_SCORE_FONT)]
    return [(base_score, SCORE_FONT)]


def _center_score_text(draw: ImageDraw.ImageDraw, team: dict, *, show: bool, x: int, width: int, y: int, height: int, fill=(255, 255, 255)):
    segments = _score_text_segments(team, show=show)
    measurements = [_measure_text(draw, text, font) for text, font in segments]
    total_width = sum(r - l for l, _t, r, _b in measurements)
    max_height = max((b - t for _l, t, _r, b in measurements), default=0)
    current_x = x + (width - total_width) // 2
    for (text, font), (l, t, r, b) in zip(segments, measurements):
        segment_width = r - l
        segment_height = b - t
        tx = current_x - l
        ty = y + (height - max_height) // 2 + (max_height - segment_height) // 2 - t
        draw.text((tx, ty), text, font=font, fill=fill)
        current_x += segment_width


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


def _round_for_date(day: datetime.date) -> str | None:
    for round_name, details in WORLD_CUP_ROUNDS.items():
        if details["show_from"] <= day <= details["show_through"]:
            return round_name
    return None


def _round_dates(round_name: str) -> tuple[datetime.date, ...]:
    details = WORLD_CUP_ROUNDS.get(round_name) or {}
    dates = details.get("dates") or ()
    return tuple(day for day in dates if isinstance(day, datetime.date))


def _order_round_games(games: list[dict], round_name: str) -> list[dict]:
    if round_name != ROUND_FINALS:
        return games
    return sorted(games, key=lambda game: str(game.get("date") or ""), reverse=True)


def _with_round_metadata(games: list[dict], round_name: str) -> list[dict]:
    ordered = _order_round_games(games, round_name)
    result: list[dict] = []
    for idx, game in enumerate(ordered):
        copy = dict(game)
        copy[ROUND_LABEL_KEY] = round_name
        if round_name == ROUND_FINALS:
            copy[ROUND_GAME_KEY] = "Championship" if idx == 0 else "3rd Place"
        result.append(copy)
    return result


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
        today = datetime.datetime.now(CENTRAL_TIME).date()
        gametime = local.strftime("%-I:%M %p")
        if local.date() == today:
            label = "Tonight" if local.hour >= 18 else "Today"
            return f"{label} {gametime}"

        day_of_week = local.strftime("%a")
        month = local.month
        day = local.day
        return f"{day_of_week} {month}/{day} {gametime}"
    except Exception:
        return ""


def _is_in_progress(game: dict) -> bool:
    return str(((game.get("status") or {}).get("type") or {}).get("state") or "").lower() == "in"


def _is_final(game: dict) -> bool:
    status_type = (game.get("status") or {}).get("type") or {}
    state = str(status_type.get("state") or "").lower()
    return state == "post" or bool(status_type.get("completed"))


def _penalty_score_value(team: dict) -> Optional[int]:
    for key in ("shootoutScore", "penaltyScore", "penalties", "penaltyKicks", "pkScore"):
        value = team.get(key)
        if isinstance(value, dict):
            score_value = value.get("score")
            value = score_value if score_value not in (None, "") else value.get("value")
        try:
            if value not in (None, ""):
                return int(str(value))
        except Exception:
            continue
    return None


def _score_text(team: dict, *, show: bool) -> str:
    if not show:
        if _show_pregame_team_abbreviation():
            return _team_abbreviation(team) or "—"
        return "—"
    score = team.get("score")
    base_score = str(score) if score not in (None, "") else "—"
    penalty_score = _penalty_score_value(team)
    if penalty_score is not None:
        return f"{base_score} ({penalty_score})"
    return base_score


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
    away_penalties = _penalty_score_value(away)
    home_penalties = _penalty_score_value(home)
    if away_penalties is not None and home_penalties is not None and away_penalties != home_penalties:
        if team_key == "away":
            return FINAL_WINNING_SCORE_COLOR if away_penalties > home_penalties else FINAL_LOSING_SCORE_COLOR
        return FINAL_WINNING_SCORE_COLOR if home_penalties > away_penalties else FINAL_LOSING_SCORE_COLOR
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


def _get_oly_logo() -> Optional[Image.Image]:
    h = max(1, int(round(_league_logo_height() * 0.55)))
    if h in _OLY_LOGO_CACHE:
        return _OLY_LOGO_CACHE[h]
    if not os.path.exists(OLY_LOGO_PATH):
        _OLY_LOGO_CACHE[h] = None
        return None
    try:
        img = Image.open(OLY_LOGO_PATH).convert("RGBA")
        ratio = h / max(1, img.height)
        resized = img.resize((max(1, int(round(img.width * ratio))), h), RESAMPLE)
        _OLY_LOGO_CACHE[h] = resized
        return resized
    except Exception:
        _OLY_LOGO_CACHE[h] = None
        return None


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


def _status_lines(game: dict) -> tuple[str, ...]:
    """Return the status text with a finals-game label on its own line."""
    status_text = _status_text(game)
    round_game = str(game.get(ROUND_GAME_KEY) or "").strip()
    return (round_game, status_text) if round_game else (status_text,)


def _render_scoreboard(games: list[dict], *, title_style: str = "title") -> Image.Image:
    title = "World Cup Scores"
    round_label = str((games[0] or {}).get(ROUND_LABEL_KEY) or "").strip() if games else ""
    logo_height = _team_logo_height()

    game_heights = [SCORE_ROW_H + len(_status_lines(game)) * STATUS_ROW_H for game in games]
    canvas_h = max(HEIGHT, sum(game_heights) + max(0, len(games) - 1) * BLOCK_SPACING)
    canvas = Image.new("RGB", (WIDTH, canvas_h), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(canvas)

    y = 0
    for idx, game in enumerate(games):
        teams = game.get("teams") or {}
        away = teams.get("away") or {}
        home = teams.get("home") or {}

        show_scores = _should_display_scores(game)
        in_progress = _is_in_progress(game)
        final = _is_final(game)

        for col_idx, text in ((2, "@"),):
            _center_text(draw, text, CENTER_FONT, COL_X[col_idx], COL_WIDTHS[col_idx], y, SCORE_ROW_H, fill=(255, 255, 255))

        for team_key, col_idx, team in (("away", 0, away), ("home", 4, home)):
            fill = _score_fill(team_key, in_progress=in_progress, final=final, away=away, home=home)
            _center_score_text(draw, team, show=show_scores, x=COL_X[col_idx], width=COL_WIDTHS[col_idx], y=y, height=SCORE_ROW_H, fill=fill)

        for col_idx, team in ((1, away), (3, home)):
            url = _team_logo_url(team)
            logo = _load_remote_logo(url, logo_height) if url else None
            if not logo:
                continue
            x0 = COL_X[col_idx] + (COL_WIDTHS[col_idx] - logo.width) // 2
            y0 = y + (SCORE_ROW_H - logo.height) // 2
            canvas.paste(logo, (x0, y0), logo)

        status_fill = IN_PROGRESS_STATUS_COLOR if in_progress else (255, 255, 255)
        status_y = y + SCORE_ROW_H
        for status_text in _status_lines(game):
            _center_text(draw, status_text, STATUS_FONT, COL_X[0], sum(COL_WIDTHS), status_y, STATUS_ROW_H, fill=status_fill)
            status_y += STATUS_ROW_H

        y += game_heights[idx]
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

    oly_logo = _get_oly_logo()
    oly_h = oly_logo.height if oly_logo else 0
    oly_gap = LEAGUE_LOGO_GAP if oly_logo else 0

    dummy_round_h = 0
    if round_label:
        try:
            l2, t2, r2, b2 = dd.textbbox((0, 0), round_label, font=STATUS_FONT)
            dummy_round_h = b2 - t2
        except Exception:
            _, dummy_round_h = dd.textsize(round_label, font=STATUS_FONT)

    content_top = oly_h + oly_gap + title_h + (LEAGUE_LOGO_GAP + dummy_round_h if round_label else 0) + TITLE_GAP
    total_h = max(HEIGHT, content_top + canvas.height + SCOREBOARD_STANDINGS_BOTTOM_PADDING)
    out = Image.new("RGB", (WIDTH, total_h), BACKGROUND_COLOR)
    draw_out = ImageDraw.Draw(out)

    if oly_logo:
        out.paste(oly_logo, ((WIDTH - oly_logo.width) // 2, 0), oly_logo)
    title_top = oly_h + oly_gap
    if title_style == "line":
        line_y = title_top + max(1, title_h // 2)
        draw_out.line((scale_value_width(10), line_y, WIDTH - scale_value_width(10), line_y), fill=(45, 45, 45))
    else:
        try:
            l, t, r, b = draw_out.textbbox((0, 0), title, font=FONT_TITLE_SPORTS)
            draw_out.text(((WIDTH - (r - l)) // 2 - l, title_top - t), title, font=FONT_TITLE_SPORTS, fill=(255, 255, 255))
        except Exception:
            tw, _ = draw_out.textsize(title, font=FONT_TITLE_SPORTS)
            draw_out.text(((WIDTH - tw) // 2, title_top), title, font=FONT_TITLE_SPORTS, fill=(255, 255, 255))

    if round_label and title_style != "line":
        _center_text(draw_out, round_label, STATUS_FONT, 0, WIDTH, title_top + title_h + LEAGUE_LOGO_GAP, dummy_round_h, fill=(255, 255, 255))

    out.paste(canvas, (0, content_top))
    return out


def _repeated_scroll_image(img: Image.Image, repeat_images: list[Image.Image] | None = None) -> Image.Image:
    if SCROLL_REPEAT_COUNT <= 1:
        return img

    images = repeat_images or [img]
    repeated_height = sum(images[index % len(images)].height for index in range(SCROLL_REPEAT_COUNT))
    repeated = Image.new("RGB", (WIDTH, repeated_height), BACKGROUND_COLOR)
    y = 0
    for index in range(SCROLL_REPEAT_COUNT):
        cycle_img = images[index % len(images)]
        repeated.paste(cycle_img, (0, y))
        y += cycle_img.height
    return repeated


def _scroll_display(display, img: Image.Image, repeat_images: list[Image.Image] | None = None):
    scroll_img = _repeated_scroll_image(img, repeat_images)
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
        oly_logo = _get_oly_logo()
        title_top = 0
        title_height = 0
        if oly_logo:
            img.paste(oly_logo, ((WIDTH - oly_logo.width) // 2, 0), oly_logo)
            title_top = oly_logo.height + LEAGUE_LOGO_GAP
        try:
            l, t, r, b = draw.textbbox((0, 0), title, font=FONT_TITLE_SPORTS)
            title_height = b - t
            draw.text(((WIDTH - (r - l)) // 2 - l, title_top - t), title, font=FONT_TITLE_SPORTS, fill=(255, 255, 255))
        except Exception:
            tw, th = draw.textsize(title, font=FONT_TITLE_SPORTS)
            title_height = th
            draw.text(((WIDTH - tw) // 2, title_top), title, font=FONT_TITLE_SPORTS, fill=(255, 255, 255))

        no_games_top = max(title_top + title_height + LEAGUE_LOGO_GAP, HEIGHT // 2 - STATUS_ROW_H // 2)
        _center_text(draw, NO_GAMES_TEXT, STATUS_FONT, 0, WIDTH, no_games_top, STATUS_ROW_H)
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

    second_cycle = _render_scoreboard(games, title_style="line")
    repeat_images = [full, second_cycle]

    if transition:
        _scroll_display(display, full, repeat_images=repeat_images)
        return ScreenImage(full, displayed=True)

    if full.height <= HEIGHT:
        display.image(full)
        time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)
    else:
        _scroll_display(display, full, repeat_images=repeat_images)
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
