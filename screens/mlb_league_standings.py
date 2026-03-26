#!/usr/bin/env python3
"""Render MLB league standings grouped by division (AL/NL)."""

from __future__ import annotations

import os
import time
import logging
from typing import Any

from PIL import Image, ImageDraw

import config
from config import (
    WIDTH,
    HEIGHT,
    FONT_TITLE_SPORTS,
    FONT_STATUS,
    IMAGES_DIR,
    SCOREBOARD_BACKGROUND_COLOR,
    SCOREBOARD_SCROLL_STEP,
    SCOREBOARD_SCROLL_DELAY,
    SCOREBOARD_SCROLL_PAUSE_TOP,
    SCOREBOARD_SCROLL_PAUSE_BOTTOM,
    SCOREBOARD_STANDINGS_BOTTOM_PADDING,
    get_screen_background_color,
    get_screen_font,
    scale_value,
)
from services.http_client import get_session
from utils import (
    ScreenImage,
    clear_display,
    clone_font,
    get_mlb_abbreviation,
    get_mlb_tricode,
    load_team_logo,
    log_call,
    log_missing_team_logo,
    scroll_vertical_content,
)

REQUEST_TIMEOUT = 10
CACHE_TTL = 15 * 60

AL_LEAGUE_ID = 103
NL_LEAGUE_ID = 104
DIVISION_ORDER = ("East", "Central", "West")
DIVISION_IDS = {
    AL_LEAGUE_ID: {"East": 201, "Central": 202, "West": 200},
    NL_LEAGUE_ID: {"East": 204, "Central": 205, "West": 203},
}

TITLE_MARGIN_TOP = scale_value(2)
TITLE_GAP = scale_value(3)
DIVISION_GAP_TOP = scale_value(6)
DIVISION_GAP_BOTTOM = scale_value(4)
DIVISION_SECTION_GAP = scale_value(8)
DIVISION_CONTENT_GAP = scale_value(10)
ROW_GAP = scale_value(6)
LEFT_MARGIN = scale_value(5)
RIGHT_MARGIN = scale_value(8)
TEAM_GAP = scale_value(6)
STAT_COLUMN_GAP = scale_value(30)
PCT_TO_GB_EXTRA_GAP = scale_value(10)
LOGO_SIZE = scale_value(24)

_IS_HYPERPIXEL_4_OR_LARGER = (
    config.is_hyperpixel_next_layout()
    or config.is_hyperpixel_4_square_layout()
    or min(int(WIDTH), int(HEIGHT)) >= 480
)

_BASE_FONTS = {
    "title": 48,
    "division": 44,
    "team": 42,
    "stats": 40,
    "gb_half": 28,
    "gb_suffix": 24,
}

_HYPERPIXEL_FONTS = {
    "title": 34,
    "division": 28,
    "team": 28,
    "stats": 26,
    "gb_half": 18,
    "gb_suffix": 16,
}

if _IS_HYPERPIXEL_4_OR_LARGER:
    _font_sizes = _HYPERPIXEL_FONTS
else:
    _font_sizes = _BASE_FONTS

TITLE_FONT = get_screen_font("MLB AL Standings", "title", base_font=FONT_TITLE_SPORTS, default_size=_font_sizes["title"])
DIVISION_FONT = get_screen_font("MLB AL Standings", "division", base_font=FONT_TITLE_SPORTS, default_size=_font_sizes["division"])
TEAM_FONT = get_screen_font("MLB AL Standings", "team", base_font=FONT_STATUS, default_size=_font_sizes["team"])
STATS_FONT = get_screen_font("MLB AL Standings", "stats", base_font=FONT_STATUS, default_size=_font_sizes["stats"])
GB_HALF_FONT = clone_font(STATS_FONT, max(8, _font_sizes["gb_half"]))
GB_SUFFIX_FONT = clone_font(STATS_FONT, max(8, _font_sizes["gb_suffix"] + 4))

SHOW_WIN_PCT = _IS_HYPERPIXEL_4_OR_LARGER

_SESSION = get_session()
_STANDINGS_CACHE: dict[str, Any] = {"timestamp": 0.0, "data": None}
_LOGO_CACHE: dict[tuple[str, int], Image.Image | None] = {}


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _fit_logo(img: Image.Image, target: int) -> Image.Image:
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    scale = min(target / w, target / h)
    return img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.LANCZOS)


def _load_logo(abbr: str, *, team_name: str = "") -> Image.Image | None:
    key = (abbr or "").strip().upper()
    if not key:
        return None
    cache_key = (key, LOGO_SIZE)
    if cache_key in _LOGO_CACHE:
        return _LOGO_CACHE[cache_key]
    logo = load_team_logo(
        os.path.join(IMAGES_DIR, "mlb"),
        key,
        box_size=LOGO_SIZE,
        height=LOGO_SIZE,
        trim=True,
    )
    if logo is not None:
        _LOGO_CACHE[cache_key] = logo
        return logo

    if team_name:
        log_missing_team_logo("MLB League Standings", team_name, key)
    _LOGO_CACHE[cache_key] = None
    return None


def _load_mlb_logo() -> Image.Image | None:
    for name in ("MLB.png", "mlb.png"):
        path = os.path.join(IMAGES_DIR, "mlb", name)
        if os.path.exists(path):
            try:
                return _fit_logo(Image.open(path).convert("RGBA"), scale_value(26))
            except Exception:
                return None
    return None


def _int_text(value: Any) -> str:
    try:
        return str(int(float(value)))
    except Exception:
        return "-"


def _pct_text(value: Any) -> str:
    try:
        return f"{float(value):.3f}".lstrip("0")
    except Exception:
        text = str(value or "-").strip()
        if text.startswith("0."):
            return text[1:]
        return text


def _gb_text(value: Any) -> str:
    text = str(value if value not in (None, "") else "-").strip()
    if not text or text == "-":
        return "-"
    try:
        numeric = float(text)
    except Exception:
        return text
    if numeric < 0:
        numeric = 0.0
    if abs(numeric) < 0.05:
        return "0"
    if abs(numeric - round(numeric)) < 0.01:
        return str(int(round(numeric)))
    return f"{numeric:.1f}"


def _split_gb_text(value: Any) -> tuple[str, str]:
    text = _gb_text(value)
    if text.endswith(".5"):
        whole = text[:-2]
        if whole in ("", "0"):
            return "", ".5"
        return whole, ".5"
    return text, ""


def _team_nickname(record: dict[str, Any]) -> str:
    team = record.get("team") if isinstance(record, dict) else {}
    if not isinstance(team, dict):
        return ""

    for key in ("teamName", "clubName", "name"):
        value = team.get(key)
        if isinstance(value, str) and value.strip():
            if key == "name":
                parts = value.strip().split()
                return " ".join(parts[1:]) if len(parts) > 1 else parts[0]
            return value.strip()
    return ""


def _normalize_row(record: dict[str, Any]) -> dict[str, Any]:
    team = record.get("team") if isinstance(record, dict) else {}
    abbr = get_mlb_tricode(team)
    if not abbr and isinstance(team, dict):
        team_name = team.get("name")
        if isinstance(team_name, str):
            abbr = get_mlb_abbreviation(team_name).upper()
    raw_name = team.get("name") if isinstance(team, dict) else ""
    nickname = _team_nickname(record) or abbr
    if (
        abbr == "BOS"
        and isinstance(raw_name, str)
        and raw_name.strip().lower() == "boston red sox"
    ):
        nickname = "Red Sox"

    return {
        "team_name": nickname,
        "abbr": abbr,
        "wins": _int_text(record.get("wins")),
        "losses": _int_text(record.get("losses")),
        "pct": _pct_text(record.get("winningPercentage")),
        "gb": _gb_text(record.get("gamesBack", "-")),
    }


def _stat_columns() -> tuple[str, ...]:
    cols = ("record", "gb")
    if SHOW_WIN_PCT:
        cols = ("record", "pct", "gb")
    return cols


def _fetch_league_standings() -> dict[int, dict[str, list[dict[str, Any]]]]:
    now = time.time()
    if _STANDINGS_CACHE.get("data") and now - float(_STANDINGS_CACHE.get("timestamp", 0.0)) < CACHE_TTL:
        return _STANDINGS_CACHE["data"]

    url = "https://statsapi.mlb.com/api/v1/standings"
    season = time.gmtime().tm_year
    params = {
        "leagueId": f"{AL_LEAGUE_ID},{NL_LEAGUE_ID}",
        "standingsType": "regularSeason",
        "season": season,
    }

    parsed: dict[int, dict[str, list[dict[str, Any]]]] = {
        AL_LEAGUE_ID: {name: [] for name in DIVISION_ORDER},
        NL_LEAGUE_ID: {name: [] for name in DIVISION_ORDER},
    }
    try:
        response = _SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logging.error("Failed to fetch MLB league standings: %s", exc)
        cached = _STANDINGS_CACHE.get("data")
        return cached if isinstance(cached, dict) else parsed

    for division_blob in payload.get("records", []):
        if not isinstance(division_blob, dict):
            continue
        league_id = int((division_blob.get("league") or {}).get("id") or 0)
        division_id = int((division_blob.get("division") or {}).get("id") or 0)
        if league_id not in DIVISION_IDS:
            continue

        division_name = next((name for name, did in DIVISION_IDS[league_id].items() if did == division_id), None)
        if division_name is None:
            continue

        teams = division_blob.get("teamRecords") or []
        parsed[league_id][division_name] = [_normalize_row(item) for item in teams if isinstance(item, dict)]

    _STANDINGS_CACHE["timestamp"] = now
    _STANDINGS_CACHE["data"] = parsed
    return parsed


def _column_layout(draw: ImageDraw.ImageDraw, rows: list[dict[str, Any]]) -> dict[str, int]:
    team_x = LEFT_MARGIN + LOGO_SIZE + TEAM_GAP
    right_edge = WIDTH - RIGHT_MARGIN
    columns = _stat_columns()

    stat_widths: dict[str, int] = {}
    for key in columns:
        width = 0
        for row in rows:
            if key == "gb":
                main, half = _split_gb_text(row.get("gb", "-"))
                gb_text = main + (half if half else "")
                width = max(width, _text_size(draw, gb_text, STATS_FONT)[0])
                width = max(width, _text_size(draw, "GB", GB_SUFFIX_FONT)[0])
            elif key == "record":
                width = max(width, _text_size(draw, f"{row.get('wins', '-')}-{row.get('losses', '-')}", STATS_FONT)[0])
            else:
                width = max(width, _text_size(draw, str(row.get(key, "-")), STATS_FONT)[0])
        stat_widths[key] = width

    layout = {"team": team_x}
    cursor = right_edge
    for key in reversed(columns):
        layout[key] = cursor
        gap = STAT_COLUMN_GAP
        if key == "gb" and "pct" in columns:
            gap += PCT_TO_GB_EXTRA_GAP
        cursor -= stat_widths[key] + gap

    first_col = columns[0]
    layout["team_max"] = max(scale_value(70), layout[first_col] - team_x - STAT_COLUMN_GAP)
    return layout


def _draw_stat(draw: ImageDraw.ImageDraw, value: str, x: int, y: int) -> None:
    draw.text((x, y), value, font=STATS_FONT, fill=(255, 255, 255), anchor="ra")


def _draw_gb(draw: ImageDraw.ImageDraw, gb_value: Any, x: int, y: int) -> None:
    main, half = _split_gb_text(gb_value)
    main_w, _ = _text_size(draw, main, STATS_FONT) if main else (0, 0)
    half_w, _ = _text_size(draw, half, GB_HALF_FONT)
    suffix_gap = max(1, scale_value(2))
    suffix_text = "GB"
    suffix_w, _ = _text_size(draw, suffix_text, GB_SUFFIX_FONT)
    total_w = main_w + half_w + suffix_gap + suffix_w
    left = x - total_w

    if main:
        draw.text((left, y), main, font=STATS_FONT, fill=(255, 255, 255), anchor="la")
    cursor = left + main_w
    if half:
        half_y = y - max(1, int(round(getattr(STATS_FONT, "size", 12) * 0.25)))
        draw.text((cursor, half_y), half, font=GB_HALF_FONT, fill=(255, 255, 255), anchor="la")
        cursor += half_w
    cursor += suffix_gap
    suffix_y = y + max(1, int(round(getattr(STATS_FONT, "size", 12) * 0.12)))
    draw.text((cursor, suffix_y), suffix_text, font=GB_SUFFIX_FONT, fill=(190, 190, 190), anchor="la")


def _draw_table_title(img: Image.Image, draw: ImageDraw.ImageDraw, title: str) -> int:
    y = TITLE_MARGIN_TOP
    logo = _load_mlb_logo()
    if logo is not None:
        x = (WIDTH - logo.width) // 2
        img.paste(logo, (x, y), logo)
        y += logo.height + TITLE_GAP

    draw.text((WIDTH // 2, y), title, font=TITLE_FONT, fill=(255, 255, 255), anchor="ma")
    return y + _text_size(draw, title, TITLE_FONT)[1] + DIVISION_GAP_TOP


def _draw_league_screen(title: str, league_id: int, screen_id: str) -> Image.Image:
    bg = get_screen_background_color(screen_id, SCOREBOARD_BACKGROUND_COLOR)
    standings = _fetch_league_standings().get(league_id, {})

    probe = ImageDraw.Draw(Image.new("RGB", (WIDTH, HEIGHT), bg))
    all_rows = [row for div in DIVISION_ORDER for row in standings.get(div, [])]
    col = _column_layout(probe, all_rows)

    row_h = max(LOGO_SIZE, _text_size(probe, "SEA", TEAM_FONT)[1], _text_size(probe, "999", STATS_FONT)[1]) + scale_value(2)
    division_title_h = _text_size(probe, "AL East", DIVISION_FONT)[1] + DIVISION_CONTENT_GAP

    visible_divisions = [div for div in DIVISION_ORDER if standings.get(div)]

    section_h = 0
    for idx, div in enumerate(visible_divisions):
        rows = standings.get(div) or []
        section_h += division_title_h
        section_h += len(rows) * (row_h + ROW_GAP)
        section_h += DIVISION_GAP_BOTTOM
        if idx < len(visible_divisions) - 1:
            section_h += DIVISION_SECTION_GAP

    canvas_h = max(HEIGHT, scale_value(80) + section_h + SCOREBOARD_STANDINGS_BOTTOM_PADDING)

    img = Image.new("RGB", (WIDTH, canvas_h), bg)
    draw = ImageDraw.Draw(img)

    y = _draw_table_title(img, draw, title)

    for idx, div in enumerate(visible_divisions):
        rows = standings.get(div) or []

        division_label = f"{title.split()[1]} {div}"
        draw.text((WIDTH // 2, y), division_label, font=DIVISION_FONT, fill=(255, 255, 255), anchor="mt")
        y += _text_size(draw, division_label, DIVISION_FONT)[1] + DIVISION_CONTENT_GAP

        for row in rows:
            row_center = y + row_h // 2
            logo = _load_logo(row.get("abbr", ""), team_name=row.get("team_name", ""))
            if logo is not None:
                img.paste(logo, (LEFT_MARGIN, row_center - logo.height // 2), logo)

            team_name = str(row.get("team_name", "-") or "-")
            max_width = col["team_max"]
            while team_name and _text_size(draw, team_name, TEAM_FONT)[0] > max_width:
                team_name = team_name[:-1]
            if team_name != str(row.get("team_name", "-")):
                team_name = f"{team_name.rstrip()}…"

            draw.text((col["team"], row_center), team_name, font=TEAM_FONT, fill=(255, 255, 255), anchor="lm")
            _draw_stat(draw, f"{row.get('wins', '-')}-{row.get('losses', '-')}", col["record"], row_center)
            if SHOW_WIN_PCT:
                _draw_stat(draw, row.get("pct", "-"), col["pct"], row_center)
            _draw_gb(draw, row.get("gb", "-"), col["gb"], row_center)
            y += row_h + ROW_GAP

        y += DIVISION_GAP_BOTTOM
        if idx < len(visible_divisions) - 1:
            y += DIVISION_SECTION_GAP

    return img


def _render_screen(display, title: str, league_id: int, screen_id: str) -> ScreenImage:
    image = _draw_league_screen(title, league_id, screen_id)
    clear_display(display)

    scroll_vertical_content(
        display=display,
        content_height=image.height,
        viewport_width=WIDTH,
        viewport_height=HEIGHT,
        render_at_offset=lambda offset: display.image(image.crop((0, offset, WIDTH, offset + HEIGHT))),
        base_step=SCOREBOARD_SCROLL_STEP,
        pause_start=SCOREBOARD_SCROLL_PAUSE_TOP,
        pause_end=SCOREBOARD_SCROLL_PAUSE_BOTTOM,
        min_frame_time=SCOREBOARD_SCROLL_DELAY,
    )
    return ScreenImage(image, displayed=True)


@log_call
def draw_mlb_al_standings(display, transition: bool = False) -> ScreenImage:
    _ = transition
    return _render_screen(display, "MLB AL Standings", AL_LEAGUE_ID, "MLB AL Standings")


@log_call
def draw_mlb_nl_standings(display, transition: bool = False) -> ScreenImage:
    _ = transition
    return _render_screen(display, "MLB NL Standings", NL_LEAGUE_ID, "MLB NL Standings")


__all__ = ["draw_mlb_al_standings", "draw_mlb_nl_standings"]
