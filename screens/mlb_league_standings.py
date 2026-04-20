#!/usr/bin/env python3
"""Render MLB league standings grouped by division (AL/NL)."""

from __future__ import annotations

import os
import time
import logging
from typing import Any

from PIL import Image, ImageDraw

import config
from display_profiles import DISPLAY_PROFILE_ADAFRUIT_MINIPITFT_114
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
STAT_HEADER_GAP = scale_value(4)
ROW_GAP = scale_value(6)
LEFT_MARGIN = scale_value(5)
RIGHT_MARGIN = scale_value(8)
TEAM_GAP = scale_value(6)
STAT_COLUMN_GAP = scale_value(30)
PCT_TO_GB_EXTRA_GAP = scale_value(10)
LOGO_SIZE = scale_value(24)
OV_COLS = 3
OV_ROWS = 5
OVERVIEW_DROP_STEPS = 30
OVERVIEW_DROP_STAGGER = 0.4
OVERVIEW_DROP_FRAME_DELAY = 0.02
OVERVIEW_PAUSE_END = 0.5

def _show_win_pct_for_layout(width: int, height: int) -> bool:
    """Return True when there is enough space for the extra PCT standings column."""

    profile_id = config.get_display_profile_id(width, height)
    if profile_id in {"display_hat_mini", DISPLAY_PROFILE_ADAFRUIT_MINIPITFT_114}:
        # Keep miniPiTFT standings aligned with 320x240 compact tables.
        return False

    return (
        config.is_hyperpixel_next_layout(width, height)
        or config.is_hyperpixel_4_square_layout(width, height)
        or min(int(width), int(height)) >= 480
    )


_IS_HYPERPIXEL_4_OR_LARGER = _show_win_pct_for_layout(int(WIDTH), int(HEIGHT))

_BASE_FONTS = {
    "title": 39,
    "division": 35,
    "team": 24,
    "stats": 20,
    "gb_suffix": 17,
    "gb_fraction": 14,
}

_HYPERPIXEL_FONTS = {
    "title": 34,
    "division": 24,
    "team": 22,
    "stats": 20,
    "gb_suffix": 15,
}

if _IS_HYPERPIXEL_4_OR_LARGER:
    _font_sizes = _HYPERPIXEL_FONTS
else:
    _font_sizes = _BASE_FONTS

if config.is_hyperpixel_4_square_layout():
    # HyperPixel 4 Square explicit league-standings font tuning.
    _font_sizes = {
        "title": 30,
        "division": 24,
        "team": 16,
        "stats": 16,
        "gb_suffix": 10,
    }

_COLUMN_GAP_MULTIPLIER = 0.75 if config.is_hyperpixel_4_square_layout() else 1.0
_STAT_COLUMN_GAP = max(1, int(round(STAT_COLUMN_GAP * _COLUMN_GAP_MULTIPLIER)))
_WIDE_STAT_COLUMN_GAP = max(1, int(round(scale_value(22) * _COLUMN_GAP_MULTIPLIER)))
_PCT_TO_GB_EXTRA_GAP = max(0, int(round(PCT_TO_GB_EXTRA_GAP * _COLUMN_GAP_MULTIPLIER)))
_RECORD_TO_GB_EXTRA_GAP = scale_value(22) if int(WIDTH) <= 320 else 0
_TEAM_TO_RECORD_GAP_WIDE = max(1, int(round(scale_value(16) * _COLUMN_GAP_MULTIPLIER)))

TITLE_FONT = get_screen_font("MLB AL Standings", "title", base_font=FONT_TITLE_SPORTS, default_size=_font_sizes["title"])
DIVISION_FONT = get_screen_font("MLB AL Standings", "division", base_font=FONT_TITLE_SPORTS, default_size=_font_sizes["division"])
TEAM_FONT = get_screen_font("MLB AL Standings", "team", base_font=FONT_STATUS, default_size=_font_sizes["team"])
STATS_FONT = get_screen_font("MLB AL Standings", "stats", base_font=FONT_STATUS, default_size=_font_sizes["stats"])
GB_SUFFIX_FONT = clone_font(STATS_FONT, max(8, _font_sizes["gb_suffix"]))
GB_FRACTION_FONT = clone_font(
    STATS_FONT,
    max(
        8,
        int(
            _font_sizes.get(
                "gb_fraction",
                round(getattr(STATS_FONT, "size", _font_sizes["stats"]) * 0.6),
            )
        ),
    ),
)
RECORD_PCT_FONT = clone_font(STATS_FONT, max(8, int(round(getattr(STATS_FONT, "size", 16) * 0.5))))

SHOW_WIN_PCT = _IS_HYPERPIXEL_4_OR_LARGER
SHOW_LAST_10 = int(WIDTH) > 400

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


def _load_logo(abbr: str, *, team_name: str = "", box_size: int = LOGO_SIZE) -> Image.Image | None:
    key = (abbr or "").strip().upper()
    if not key:
        return None
    size = max(1, int(box_size))
    cache_key = (key, size)
    if cache_key in _LOGO_CACHE:
        return _LOGO_CACHE[cache_key]
    logo = load_team_logo(
        os.path.join(IMAGES_DIR, "mlb"),
        key,
        box_size=size,
        height=size,
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
    if abs(numeric) < 0.05:
        return "-"
    if abs(numeric - 0.5) < 0.01:
        return "1/2"
    if abs(numeric - round(numeric)) < 0.01:
        return str(int(round(numeric)))
    if abs((numeric * 2) - round(numeric * 2)) < 0.01:
        whole = int(numeric)
        return f"{whole} 1/2" if whole else "1/2"
    return f"{numeric:.1f}"


def _split_gb_text(value: Any) -> tuple[str, str]:
    text = _gb_text(value)
    if text == "1/2":
        return "", "1/2"
    if text.endswith(" 1/2"):
        return text[:-4], "1/2"
    return text, ""


def _gb_value_width(draw: ImageDraw.ImageDraw, whole_text: str, frac_text: str) -> int:
    width = 0
    if whole_text:
        width += _text_size(draw, whole_text, STATS_FONT)[0]
        if frac_text:
            width += scale_value(2)
    if frac_text:
        width += _text_size(draw, frac_text, GB_FRACTION_FONT)[0]
    return width


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
    if abbr == "BOS" and isinstance(nickname, str) and nickname.strip().lower() == "sox":
        nickname = "Red Sox"
    elif (
        abbr == "BOS"
        and isinstance(raw_name, str)
        and raw_name.strip().lower() == "boston red sox"
    ):
        nickname = "Red Sox"

    split_records = (record.get("records") or {}).get("splitRecords", [])
    last_10 = "-"
    for split in split_records:
        if not isinstance(split, dict):
            continue
        if str(split.get("type", "")).lower() == "lastten":
            last_10 = f"{_int_text(split.get('wins'))}-{_int_text(split.get('losses'))}"
            break

    return {
        "team_name": nickname,
        "abbr": abbr,
        "wins": _int_text(record.get("wins")),
        "losses": _int_text(record.get("losses")),
        "pct": _pct_text(record.get("winningPercentage")),
        "last10": last_10,
        "gb": _gb_text(record.get("gamesBack", "-")),
    }


def _stat_columns() -> tuple[str, ...]:
    if SHOW_LAST_10:
        return ("record", "last10", "gb")
    cols = ("record", "gb")
    if SHOW_WIN_PCT:
        cols = ("record", "pct", "gb")
    return cols


def _record_with_pct_width(draw: ImageDraw.ImageDraw, record_text: str, pct_text: str) -> int:
    record_w = _text_size(draw, record_text, STATS_FONT)[0]
    pct_w = _text_size(draw, f"({pct_text})", RECORD_PCT_FONT)[0]
    return record_w + scale_value(4) + pct_w


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
                gb_text, gb_frac = _split_gb_text(row.get("gb", "-"))
                width = max(width, _gb_value_width(draw, gb_text, gb_frac))
                width = max(width, _text_size(draw, "GB", GB_SUFFIX_FONT)[0])
            elif key == "record":
                record_text = f"{row.get('wins', '-')}-{row.get('losses', '-')}"
                if SHOW_LAST_10:
                    width = max(width, _record_with_pct_width(draw, record_text, str(row.get("pct", "-"))))
                else:
                    width = max(width, _text_size(draw, record_text, STATS_FONT)[0])
            else:
                width = max(width, _text_size(draw, str(row.get(key, "-")), STATS_FONT)[0])
        stat_widths[key] = width

    layout = {"team": team_x}

    def _apply_default_right_aligned_layout() -> None:
        cursor = right_edge
        for key in reversed(columns):
            layout[key] = cursor
            layout[f"{key}_width"] = stat_widths[key]
            gap = _WIDE_STAT_COLUMN_GAP if SHOW_LAST_10 else _STAT_COLUMN_GAP
            if key == "gb" and "pct" in columns:
                gap += _PCT_TO_GB_EXTRA_GAP
            if key == "gb" and "record" in columns:
                gap += _RECORD_TO_GB_EXTRA_GAP
            cursor -= stat_widths[key] + gap

    if SHOW_LAST_10 and columns == ("record", "last10", "gb"):
        # Keep GB at the same right-aligned placement while centering Record
        # and placing L10 midway between Record and GB.
        gb_right = right_edge
        gb_center = gb_right - (stat_widths["gb"] / 2.0)
        record_center = WIDTH / 2.0
        last10_center = (record_center + gb_center) / 2.0

        record_right = int(round(record_center + (stat_widths["record"] / 2.0)))
        last10_right = int(round(last10_center + (stat_widths["last10"] / 2.0)))

        record_right = min(record_right, gb_right - stat_widths["last10"] - stat_widths["record"])
        last10_right = min(last10_right, gb_right - max(2, scale_value(2)))

        record_left = record_right - stat_widths["record"]
        last10_left = last10_right - stat_widths["last10"]
        gb_left = gb_right - stat_widths["gb"]

        if record_right < last10_left and last10_right < gb_left:
            layout["record"] = record_right
            layout["record_width"] = stat_widths["record"]
            layout["last10"] = last10_right
            layout["last10_width"] = stat_widths["last10"]
            layout["gb"] = gb_right
            layout["gb_width"] = stat_widths["gb"]
        else:
            _apply_default_right_aligned_layout()
    else:
        _apply_default_right_aligned_layout()

    first_col = columns[0]
    team_gap = _TEAM_TO_RECORD_GAP_WIDE if SHOW_LAST_10 else _STAT_COLUMN_GAP
    layout["team_max"] = max(scale_value(70), layout[first_col] - team_x - team_gap)
    return layout


def _draw_stat(draw: ImageDraw.ImageDraw, value: str, x: int, y: int) -> None:
    draw.text((x, y), value, font=STATS_FONT, fill=(255, 255, 255), anchor="rm")


def _draw_record_with_pct(draw: ImageDraw.ImageDraw, record_value: str, pct_value: str, x: int, y: int) -> None:
    pct_text = f"({pct_value})"
    record_w = _text_size(draw, record_value, STATS_FONT)[0]
    pct_w = _text_size(draw, pct_text, RECORD_PCT_FONT)[0]
    total_w = record_w + scale_value(4) + pct_w
    left = x - total_w
    draw.text((left, y), record_value, font=STATS_FONT, fill=(255, 255, 255), anchor="lm")
    draw.text(
        (left + record_w + scale_value(4), y),
        pct_text,
        font=RECORD_PCT_FONT,
        fill=(200, 200, 200),
        anchor="lm",
    )


def _stat_header_labels() -> dict[str, str]:
    return {
        "record": "Record",
        "last10": "L10",
        "pct": "Win%",
        "gb": "GB",
    }


def _draw_stat_headers(draw: ImageDraw.ImageDraw, layout: dict[str, int], y: int) -> int:
    labels = _stat_header_labels()
    for key in _stat_columns():
        col_width = int(layout.get(f"{key}_width", 0) or 0)
        center_x = layout[key] - (col_width // 2)
        draw.text(
            (center_x, y),
            labels.get(key, key.upper()),
            font=RECORD_PCT_FONT,
            fill=(200, 200, 200),
            anchor="mm",
        )
    return y + _text_size(draw, "Record", RECORD_PCT_FONT)[1] + STAT_HEADER_GAP


def _draw_gb(draw: ImageDraw.ImageDraw, gb_value: Any, x: int, y: int) -> None:
    gb_text, gb_frac = _split_gb_text(gb_value)
    gb_w = _gb_value_width(draw, gb_text, gb_frac)
    suffix_gap = max(1, scale_value(2))
    suffix_text = "GB"
    suffix_w, _ = _text_size(draw, suffix_text, GB_SUFFIX_FONT)
    value_right = x - suffix_w - suffix_gap

    cursor_x = value_right - gb_w
    if gb_text:
        draw.text((cursor_x, y), gb_text, font=STATS_FONT, fill=(255, 255, 255), anchor="lm")
        cursor_x += _text_size(draw, gb_text, STATS_FONT)[0]
    if gb_frac:
        if gb_text:
            cursor_x += scale_value(2)
        draw.text((cursor_x, y), gb_frac, font=GB_FRACTION_FONT, fill=(255, 255, 255), anchor="lm")
        cursor_x += _text_size(draw, gb_frac, GB_FRACTION_FONT)[0]
    draw.text((x, y), suffix_text, font=GB_SUFFIX_FONT, fill=(190, 190, 190), anchor="rm")


def _draw_table_title(img: Image.Image, draw: ImageDraw.ImageDraw, title: str) -> int:
    y = TITLE_MARGIN_TOP
    logo = _load_mlb_logo()
    if logo is not None:
        x = (WIDTH - logo.width) // 2
        img.paste(logo, (x, y), logo)
        y += logo.height + TITLE_GAP

    draw.text((WIDTH // 2, y), title, font=TITLE_FONT, fill=(255, 255, 255), anchor="ma")
    return y + _text_size(draw, title, TITLE_FONT)[1] + DIVISION_GAP_TOP


def _overview_header_frame(title: str, bg: tuple[int, int, int]) -> tuple[Image.Image, int]:
    img = Image.new("RGB", (WIDTH, HEIGHT), bg)
    draw = ImageDraw.Draw(img)
    title_width, title_height = _text_size(draw, title, TITLE_FONT)
    draw.text(((WIDTH - title_width) // 2, 0), title, font=TITLE_FONT, fill=(255, 255, 255))
    header_pad = scale_value(6) if config.is_hyperpixel_next_layout() else 6
    return img, title_height + header_pad


def _ease_out_cubic(t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    inv = 1.0 - t
    return 1.0 - inv * inv * inv


@log_call
def draw_overview(display, title: str, league_id: int, transition: bool = False):
    wait_for_skip = getattr(display, "wait_for_skip", None)
    skip_requested = getattr(display, "skip_requested", None)

    def _should_skip() -> bool:
        return bool(skip_requested and skip_requested())

    def _sleep(duration: float) -> bool:
        if callable(wait_for_skip):
            return bool(wait_for_skip(duration))
        time.sleep(duration)
        return False

    bg = get_screen_background_color(title, SCOREBOARD_BACKGROUND_COLOR)
    divisions = ["East", "Central", "West"]
    header, top_y = _overview_header_frame(title, bg)
    available_height = max(1, HEIGHT - top_y)

    hyperpixel_layout = config.is_hyperpixel_next_layout()
    hyperpixel4_layout = hyperpixel_layout or config.is_hyperpixel_4_square_layout()
    overview_logo_size = LOGO_SIZE
    if hyperpixel4_layout:
        overview_logo_size = max(1, int(round(LOGO_SIZE * 1.25)))

    if hyperpixel_layout:
        overview_margin = max(LEFT_MARGIN, scale_value(6))
        available_width = max(1, WIDTH - 2 * overview_margin)
        cell_h = available_height / OV_ROWS
        col_width = available_width / OV_COLS
        padding = max(2, scale_value(4))
        logo_box = max(6, int(min(cell_h - padding * 2, col_width - padding * 2)))
        overview_logo_size = min(overview_logo_size, logo_box)
        col_centers = [overview_margin + col_width * (i + 0.5) for i in range(OV_COLS)]
    else:
        cell_h = available_height // OV_ROWS
        col_w = max(scale_value(44), LOGO_SIZE)
        margin_x = (WIDTH - OV_COLS * col_w) // (OV_COLS + 1)
        col_centers = [margin_x * (i + 1) + col_w * i + col_w / 2 for i in range(OV_COLS)]
        logo_box = col_w

    standings = _fetch_league_standings().get(league_id, {})
    logos_per_div: dict[str, list[Image.Image | None]] = {}
    for div in divisions:
        rows = standings.get(div, [])[:OV_ROWS]
        logos: list[Image.Image | None] = []
        for row in rows:
            abbr = str(row.get("abbr", "") or "")
            team_name = str(row.get("team_name", "") or "")
            logos.append(_load_logo(abbr, team_name=team_name, box_size=overview_logo_size))
        while len(logos) < OV_ROWS:
            logos.append(None)
        logos_per_div[div] = logos

    row_positions: list[list[tuple[Image.Image, int, int]]] = []
    for rank in range(OV_ROWS):
        placements: list[tuple[Image.Image, int, int]] = []
        for ci, div in enumerate(divisions):
            icon = logos_per_div[div][rank]
            if not icon:
                continue
            x0 = int(col_centers[ci] - icon.width / 2)
            y_target = int(top_y + rank * cell_h + (cell_h - icon.height) / 2)
            placements.append((icon, x0, y_target))
        row_positions.append(placements)

    steps = max(2, OVERVIEW_DROP_STEPS)
    stagger = max(1, int(round(steps * OVERVIEW_DROP_STAGGER)))

    schedule: list[tuple[int, list[tuple[Image.Image, int, int]]]] = []
    start_step = 0
    for rank in range(len(row_positions) - 1, -1, -1):
        drops = row_positions[rank]
        if not drops:
            continue
        schedule.append((start_step, drops))
        start_step += stagger

    if schedule:
        total_duration = schedule[-1][0] + steps + 1
        placed: list[tuple[Image.Image, int, int]] = []
        completed = [False] * len(schedule)

        for current_step in range(total_duration):
            if _should_skip():
                return header.copy() if transition else None

            frame_start = time.time()

            for idx, (start, drops) in enumerate(schedule):
                if current_step >= start + steps and not completed[idx]:
                    placed.extend(drops)
                    completed[idx] = True

            frame = header.copy()
            for icon, x0, y0 in placed:
                frame.paste(icon, (x0, y0), icon)

            for start, drops in schedule:
                progress = current_step - start
                if progress < 0 or progress >= steps:
                    continue
                frac = progress / (steps - 1) if steps > 1 else 1.0
                eased = _ease_out_cubic(frac)
                for icon, x0, y_target in drops:
                    start_y = -logo_box
                    y_pos = int(start_y + (y_target - start_y) * eased)
                    if y_pos > y_target:
                        y_pos = y_target
                    frame.paste(icon, (x0, y_pos), icon)

            display.image(frame)
            display.show()

            elapsed = time.time() - frame_start
            sleep_time = max(0, OVERVIEW_DROP_FRAME_DELAY - elapsed)
            if sleep_time > 0 and _sleep(sleep_time):
                return header.copy() if transition else None

    final = header.copy()
    for rank in range(OV_ROWS):
        for ci, div in enumerate(divisions):
            icon = logos_per_div[div][rank]
            if not icon:
                continue
            x0 = int(col_centers[ci] - icon.width / 2)
            y0 = int(top_y + rank * cell_h + (cell_h - icon.height) / 2)
            final.paste(icon, (x0, y0), icon)

    display.image(final)
    display.show()
    _sleep(OVERVIEW_PAUSE_END)
    return final if transition else None


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
        y = _draw_stat_headers(draw, col, y)

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
            record_value = f"{row.get('wins', '-')}-{row.get('losses', '-')}"
            if SHOW_LAST_10:
                _draw_record_with_pct(draw, record_value, str(row.get("pct", "-")), col["record"], row_center)
                _draw_stat(draw, str(row.get("last10", "-")), col["last10"], row_center)
            else:
                _draw_stat(draw, record_value, col["record"], row_center)
            if SHOW_WIN_PCT and not SHOW_LAST_10:
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


@log_call
def draw_NL_Overview(display, transition: bool = False):
    return draw_overview(display, "NL Overview", NL_LEAGUE_ID, transition)


@log_call
def draw_AL_Overview(display, transition: bool = False):
    return draw_overview(display, "AL Overview", AL_LEAGUE_ID, transition)


__all__ = [
    "draw_mlb_al_standings",
    "draw_mlb_nl_standings",
    "draw_NL_Overview",
    "draw_AL_Overview",
]
