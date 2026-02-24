#!/usr/bin/env python3
"""
Draw MLB box scores and next/last-game screens in RGB,
with both team logos on the Next Game screen, in AWAY @ HOME order,
and a small W/L flag between the boxscore and date on Cubs 'Last Game'.
"""

import os
import logging
import datetime
from typing import Optional, Tuple
from PIL import Image, ImageDraw, Image, ImageFont

import config
from config import (
    WIDTH, HEIGHT,
    FONT_TITLE_SPORTS, FONT_DATE_SPORTS,
    FONT_TEAM_SPORTS, FONT_SCORE,
    MLB_CUBS_TEAM_ID, MLB_SOX_TEAM_ID,
    CENTRAL_TIME,
    IMAGES_DIR,
    get_screen_background_color,
    is_hyperpixel_4_square_layout,
    is_hyperpixel_next_layout,
)
from utils import (
    LED_INDICATOR_LEVEL,
    ScreenImage,
    get_team_display_name,
    get_mlb_abbreviation,
    get_mlb_tricode,
    log_call,
    load_team_logo,
    standard_next_game_logo_frame_width,
    standard_next_game_logo_height,
    wrap_text,
)
from display_profiles import DISPLAY_PROFILE_DISPLAY_HAT_MINI, DISPLAY_PROFILE_HYPERPIXEL4

# ── Paths ────────────────────────────────────────────────────────────────────
BACKGROUND_COLOR = (0, 0, 0)


def _set_background(screen_id: Optional[str]) -> None:
    global BACKGROUND_COLOR
    if screen_id:
        BACKGROUND_COLOR = get_screen_background_color(screen_id, BACKGROUND_COLOR)
MLB_LOGOS_DIR = os.path.join(IMAGES_DIR, "mlb")

# ── Layout constants ─────────────────────────────────────────────────────────
if is_hyperpixel_4_square_layout():
    BOTTOM_MARGIN = 18
elif is_hyperpixel_next_layout():
    BOTTOM_MARGIN = 15
else:
    BOTTOM_MARGIN = 6
_IS_1080P_LAYOUT        = config.is_hdmi_1080p_layout()
_BOTTOM_TEXT_1080P_OFFSET = 30
_LOGO_SCALE_1080 = config.DISPLAY_PROFILE_LOGO_SCALE_CAP
TITLE_TO_HEADER_GAP     = 6          # space between title baseline and header labels
HEADER_GAP              = 3          # space between R/H/E labels and grid
TABLE_SIDE_MARGIN       = 4          # left/right inset of table
MIN_TEAM_COL_WIDTH      = 40         # never let the team column be narrower
DESIRED_SQUARE_FRACTION = 0.24       # starting point for square width vs total_w
GRID_BG                 = (14, 36, 22)  # dark forest green
NEXT_GAME_LOGO_SCALE    = 0.92       # slightly reduce logo size inside frame

# Cubs mini-flag sizing/reservation
SMALL_RESULT_FLAG_H     = int(os.environ.get("SMALL_RESULT_FLAG_H", "48"))
FLAG_BLOCK_PAD          = 6
FLAG_BLOCK_H            = SMALL_RESULT_FLAG_H + FLAG_BLOCK_PAD  # reserved area (always)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _format_game_label(official_date: str, start_time: str) -> str:
    """Bottom label for next-game screens with relative-day logic."""

    def _parse_date(value: str) -> Optional[datetime.date]:
        value = (value or "").strip()
        if not value:
            return None
        try:
            return datetime.date.fromisoformat(value[:10])
        except Exception:
            return None

    def _parse_time(parts: list[str]) -> Tuple[Optional[datetime.time], str]:
        time_token = ""
        ampm_token = ""
        for part in parts:
            if not time_token:
                time_token = part
                continue
            if not ampm_token and part.upper() in {"AM", "PM"}:
                ampm_token = part.upper()
                break
        if time_token and ampm_token:
            for fmt in ("%I:%M %p", "%I %p"):
                try:
                    tm = datetime.datetime.strptime(f"{time_token} {ampm_token}", fmt).time()
                    break
                except Exception:
                    tm = None
            else:
                tm = None
        else:
            tm = None

        # Build a display string resembling the old formatting.
        disp_time = time_token
        if disp_time.startswith("0"):
            disp_time = disp_time[1:]
        if disp_time.endswith(":00"):
            disp_time = disp_time[:-3]
        display = " ".join(p for p in (disp_time, ampm_token) if p).strip()
        return tm, display

    date_obj = _parse_date(official_date)
    start_raw = (start_time or "").strip()
    parts = start_raw.split()
    time_obj, time_display = _parse_time(parts)

    # If we still do not have a friendly time string, just use the raw input.
    if not time_display:
        time_display = start_raw

    local_dt = None
    if date_obj:
        try:
            if time_obj:
                local_dt = CENTRAL_TIME.localize(
                    datetime.datetime.combine(date_obj, time_obj)
                )
            else:
                # Default to an evening time purely for relative label purposes.
                local_dt = CENTRAL_TIME.localize(
                    datetime.datetime.combine(date_obj, datetime.time(19, 0))
                )
        except Exception:
            local_dt = None

    today = datetime.datetime.now(CENTRAL_TIME)
    label = ""
    if local_dt:
        game_date = local_dt.date()
        if game_date == today.date():
            if time_obj and local_dt.hour >= 18:
                label = "Tonight"
            else:
                label = "Today"
        elif game_date == today.date() + datetime.timedelta(days=1):
            label = "Tomorrow"
        else:
            if os.name == "nt":
                label = game_date.strftime("%a %b %#d")
            else:
                label = game_date.strftime("%a %b %-d")
    elif date_obj:
        if os.name == "nt":
            label = date_obj.strftime("%a %b %#d")
        else:
            label = date_obj.strftime("%a %b %-d")

    parts = []
    if label:
        parts.append(label)
    if time_display:
        parts.append(time_display)
    return " • ".join(parts) if parts else ""

def _rel_date_only(official_date: str) -> str:
    """'Today', 'Tomorrow', 'Yesterday', else 'Tue M/D' (no time)."""
    today = datetime.datetime.now(CENTRAL_TIME).date()
    try:
        d = datetime.datetime.strptime(official_date, "%Y-%m-%d").date()
    except Exception:
        try:
            d = datetime.datetime.strptime(official_date[:10], "%Y-%m-%d").date()
        except Exception:
            return official_date or ""
    if d == today:
        return "Today"
    if d == today + datetime.timedelta(days=1):
        return "Tomorrow"
    if d == today - datetime.timedelta(days=1):
        return "Yesterday"
    return f"{d.strftime('%a')} {d.month}/{d.day}"

def _draw_title_with_bold_result(
    draw: ImageDraw.ImageDraw,
    title: str,
    *,
    y_offset: int = 0,
) -> tuple[int,int]:
    """Center the title. If it ends with ' W' or ' L', faux-bold that letter."""
    tw, th = draw.textsize(title, font=FONT_TITLE_SPORTS)
    x0 = (WIDTH - tw)//2
    draw.text((x0, y_offset), title, font=FONT_TITLE_SPORTS, fill=(255,255,255))
    if title.endswith(" W") or title.endswith(" L"):
        ch = title[-1]
        cw, _ = draw.textsize(ch, font=FONT_TITLE_SPORTS)
        cx = x0 + tw - cw
        cy = y_offset
        draw.text((cx, cy), ch, font=FONT_TITLE_SPORTS, fill=(255,255,255))
        draw.text((cx+1, cy), ch, font=FONT_TITLE_SPORTS, fill=(255,255,255))
    return tw, th


def _center_bottom_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    *,
    margin: int = BOTTOM_MARGIN,
    fill=(255, 255, 255),
) -> None:
    if not text:
        return
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        tw, th = r - l, b - t
        tx = (WIDTH - tw) // 2 - l
        ty = HEIGHT - th - margin - t
    except Exception:
        tw, th = draw.textsize(text, font=font)
        tx = (WIDTH - tw) // 2
        ty = HEIGHT - th - margin
    draw.text((tx, ty), text, font=font, fill=fill)

def _bbox_center(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int,
                 text: str, font, *, fill=(255,255,255)):
    """
    Center text perfectly inside the given box using textbbox to account for
    ascent/descent. This fixes vertical drift.
    """
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        tw, th = (r - l), (b - t)
        tx = x + (w - tw)//2 - l
        ty = y + (h - th)//2 - t
    except Exception:
        # Fallback: approximate with textsize (older Pillow)
        tw, th = draw.textsize(text, font=font)
        tx = x + (w - tw)//2
        ty = y + (h - th)//2
    draw.text((tx, ty), text, font=font, fill=fill)


def _should_show_team_logo_boxscore(screen_id: Optional[str]) -> bool:
    if config.get_display_profile_id() not in {
        DISPLAY_PROFILE_HYPERPIXEL4,
        DISPLAY_PROFILE_DISPLAY_HAT_MINI,
    }:
        return False
    return (screen_id or "").strip().lower() in {
        "cubs live",
        "cubs last",
        "sox live",
        "sox last",
    }


def _draw_left_team_cell_with_logo(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    team: dict,
    abbr: str,
    x: int,
    y: int,
    w: int,
    h: int,
    font,
    fill=(255, 255, 255),
) -> None:
    left_pad = max(2, min(6, w // 12))
    top_pad = max(1, h // 8)
    logo_size = max(1, min(h - (top_pad * 2), int(h * 0.62)))

    logo_key = get_mlb_tricode(team) or get_mlb_abbreviation(get_team_display_name(team))
    logo = load_team_logo(MLB_LOGOS_DIR, logo_key, box_size=logo_size)

    text_x = x + left_pad
    if logo:
        lx = x + left_pad
        ly = y + (h - logo.height) // 2
        img.paste(logo, (lx, ly), logo)
        text_x = lx + logo.width + max(2, left_pad // 2)

    # Keep abbreviation constrained to the cell width.
    max_text_w = max(0, (x + w - left_pad) - text_x)
    display_abbr = str(abbr)
    if max_text_w <= 0:
        return
    try:
        l, t, r, b = draw.textbbox((0, 0), display_abbr, font=font)
        tw, th = r - l, b - t
    except Exception:
        tw, th = draw.textsize(display_abbr, font=font)
        l = t = 0

    while display_abbr and tw > max_text_w:
        display_abbr = display_abbr[:-1]
        try:
            l, t, r, b = draw.textbbox((0, 0), display_abbr, font=font)
            tw, th = r - l, b - t
        except Exception:
            tw, th = draw.textsize(display_abbr, font=font)
            l = t = 0

    ty = y + (h - th) // 2 - t
    draw.text((text_x - l, ty), display_abbr, font=font, fill=fill)

def _compute_table_geometry(
    draw: ImageDraw.ImageDraw,
    top_y: int,
    bottom_y: int,
    reserve_flag_block: bool,
    *,
    table_side_margin: int = TABLE_SIDE_MARGIN,
    min_team_col_width: int = MIN_TEAM_COL_WIDTH,
    header_gap: int = HEADER_GAP,
    reserve_flag_height: int = FLAG_BLOCK_H,
) -> dict:
    """
    Decide sizes so that columns 2–4 are true squares (same width as row height),
    and column 1 takes the rest. Ensures header labels sit above the grid with space.
    Returns a geometry dict.
    """
    # reserve space for the small-flag area (always, so Cubs/Sox align)
    grid_bottom_limit = bottom_y - reserve_flag_height if reserve_flag_block else bottom_y

    # Header row height = label text height + small padding
    hdr_h = draw.textsize("R", font=FONT_DATE_SPORTS)[1] + 2

    # Horizontal extents
    total_w = WIDTH - 2*table_side_margin

    # Start with desired square size; clamp against minimum team width
    desired_sq = int(total_w * DESIRED_SQUARE_FRACTION)
    max_sq_by_width = (total_w - min_team_col_width) // 3
    square = max(18, min(desired_sq, max_sq_by_width))

    # Ensure grid fits vertically (2 rows of 'square' cells)
    grid_top = top_y + hdr_h + header_gap
    max_rows_h = grid_bottom_limit - grid_top
    if max_rows_h > 0:
        square = min(square, max_rows_h // 2)
    square = max(18, square)

    # Derive first column width from final square
    team_w = total_w - 3*square
    if team_w < min_team_col_width:
        square = max(18, (total_w - min_team_col_width) // 3)
        team_w = total_w - 3*square

    xs = [
        table_side_margin,
        table_side_margin + team_w,
        table_side_margin + team_w + square,
        table_side_margin + team_w + 2*square,
        table_side_margin + team_w + 3*square,
    ]

    return {
        "hdr_h": hdr_h,
        "grid_top": grid_top,
        "row_h": square,           # square cells
        "team_w": team_w,
        "square": square,
        "xs": xs,
        "grid_w": total_w,
        "grid_h": square * 2,
    }

def _draw_boxscore_table(img: Image.Image, draw: ImageDraw.ImageDraw, title: str,
                         away_lbl, away_r, away_h, away_e,
                         home_lbl, home_r, home_h, home_e,
                         bottom_text: str,
                         *,
                         away_team: Optional[dict]=None,
                         home_team: Optional[dict]=None,
                         screen_id: Optional[str]=None,
                         reserve_flag_block: bool,
                         live: bool=False,
                         winner_flag: str|None=None,
                         hyperpixel_layout: bool=False,
                         center_content_vertically: bool=False,
                         flag_scale: float=1.0):
    """
    Render the whole screen (title + header + table + optional small flag + bottom line).
    - Columns 2–4 are true squares; column 1 stretches.
    - Values are centered both horizontally and vertically in each cell.
    - Optional small W/L flag drawn only if 'winner_flag' is 'W' or 'L' (Cubs only).
    """
    edge_pad = config.scale_value(2) if hyperpixel_layout else 0
    title_gap = config.scale_value(TITLE_TO_HEADER_GAP) if hyperpixel_layout else TITLE_TO_HEADER_GAP
    header_gap = config.scale_value(HEADER_GAP) if hyperpixel_layout else HEADER_GAP
    table_side_margin = (
        config.scale_value(TABLE_SIDE_MARGIN) if hyperpixel_layout else TABLE_SIDE_MARGIN
    )
    min_team_col_width = (
        config.scale_value(MIN_TEAM_COL_WIDTH) if hyperpixel_layout else MIN_TEAM_COL_WIDTH
    )

    # Title
    _, th = _draw_title_with_bold_result(draw, title, y_offset=edge_pad)

    # Bottom line position (reserve space using accurate text metrics)
    if bottom_text:
        try:
            _, t, _, b = draw.textbbox((0, 0), bottom_text, font=FONT_DATE_SPORTS)
            bh = b - t
        except Exception:
            bh = draw.textsize(bottom_text, font=FONT_DATE_SPORTS)[1]
    else:
        bh = 0
    bottom_margin = config.scale_value(BOTTOM_MARGIN) if hyperpixel_layout else BOTTOM_MARGIN
    bottom_y = HEIGHT - bh - bottom_margin

    flag_h = max(1, int(round(SMALL_RESULT_FLAG_H * max(0.1, flag_scale))))
    flag_block_h = flag_h + FLAG_BLOCK_PAD

    # Geometry
    g = _compute_table_geometry(
        draw,
        top_y=edge_pad + th + title_gap,
        bottom_y=bottom_y,
        reserve_flag_block=reserve_flag_block,
        table_side_margin=table_side_margin,
        min_team_col_width=min_team_col_width,
        header_gap=header_gap,
        reserve_flag_height=flag_block_h,
    )
    hdr_h   = g["hdr_h"]
    grid_top= g["grid_top"]
    row_h   = g["row_h"]
    team_w  = g["team_w"]
    square  = g["square"]
    xs      = g["xs"]
    total_w = g["grid_w"]
    grid_h  = g["grid_h"]

    if center_content_vertically:
        content_top = edge_pad + th + title_gap
        content_bottom = bottom_y
        if reserve_flag_block:
            content_bottom -= flag_block_h
        centered_grid_top = content_top + max(0, (content_bottom - content_top - grid_h) // 2)
        grid_top = max(grid_top, centered_grid_top)

    # Header row (center each label over its column)
    for i, lbl in enumerate(["", "R", "H", "E"]):
        col_w = [team_w, square, square, square][i]
        # center exactly using bbox
        _bbox_center(draw,
                     x=xs[i],
                     y=(grid_top - header_gap) - hdr_h,
                     w=col_w,
                     h=hdr_h,
                     text=lbl,
                     font=FONT_DATE_SPORTS,
                     fill=(255,255,255))

    # Grid background (forest green) – exactly behind the 2×2 rows area
    draw.rectangle(
        (table_side_margin, grid_top, table_side_margin + total_w, grid_top + grid_h),
        fill=GRID_BG
    )

    # Grid outline + interior lines
    draw.rectangle(
        (table_side_margin, grid_top, table_side_margin + total_w, grid_top + grid_h),
        outline=(255,255,255)
    )
    # Vertical separators
    for x in xs[1:-1]:
        draw.line((x, grid_top, x, grid_top + grid_h), fill=(255,255,255))
    # Middle horizontal line
    draw.line((table_side_margin, grid_top + row_h, table_side_margin + total_w, grid_top + row_h),
              fill=(255,255,255))

    # Rows data (centered text in each cell)
    rows = [
        (away_lbl, away_r, away_h, away_e),
        (home_lbl, home_r, home_h, home_e)
    ]
    show_team_logo = _should_show_team_logo_boxscore(screen_id)
    row_teams = [away_team, home_team]
    for ridx, (lbl, r, h, e) in enumerate(rows):
        for cidx, val in enumerate([lbl, r, h, e]):
            txt = str(val)
            fill_col = (255,255,0) if live and cidx > 0 else (255,255,255)
            font_use = FONT_TEAM_SPORTS if cidx == 0 else FONT_SCORE
            col_w    = [team_w, square, square, square][cidx]
            x_left   = xs[cidx]
            y_cell   = grid_top + ridx * row_h
            if cidx == 0 and show_team_logo and row_teams[ridx]:
                _draw_left_team_cell_with_logo(
                    img,
                    draw,
                    team=row_teams[ridx],
                    abbr=txt,
                    x=x_left,
                    y=y_cell,
                    w=col_w,
                    h=row_h,
                    font=font_use,
                    fill=fill_col,
                )
                continue
            _bbox_center(draw, x_left, y_cell, col_w, row_h, txt, font_use, fill=fill_col)

    # Optional small W/L flag (Cubs only) – drawn in the reserved block
    if reserve_flag_block and winner_flag in ("W","L"):
        block_top = grid_top + grid_h + 2
        block_h   = flag_block_h
        flag_path = os.path.join(IMAGES_DIR, "mlb", f"{winner_flag}.png")  # mlb/W.png / mlb/L.png
        if os.path.exists(flag_path):
            try:
                flag = Image.open(flag_path).convert("RGBA")
                w0, h0 = flag.size
                ratio  = flag_h / float(h0)
                flag   = flag.resize((max(1, int(w0*ratio)), flag_h), Image.ANTIALIAS)
                fx     = (WIDTH - flag.width)//2
                fy     = block_top + (block_h - flag.height)//2
                img.paste(flag, (fx, fy), flag)
            except Exception:
                pass

    # Bottom label
    _center_bottom_text(draw, bottom_text, FONT_DATE_SPORTS, margin=bottom_margin)


# ── Screens ─────────────────────────────────────────────────────────────────

@log_call
def draw_last_game(display, game, title="Last Game...", transition=False, screen_id: Optional[str] = None):
    _set_background(screen_id)
    if not game:
        logging.warning(f"No game data for {title}")
        return None

    hyperpixel_layout = config.is_hyperpixel_next_layout() and screen_id in {
        "cubs last",
        "sox last",
    }

    # Determine which team (Cubs vs Sox) to compute W/L and whether to show mini-flag
    tid = int(MLB_CUBS_TEAM_ID) if "Cubs" in title else int(MLB_SOX_TEAM_ID)
    away = game["teams"]["away"]
    home = game["teams"]["home"]
    winner = (
        (away["team"]["id"] == tid and away.get("score",0) > home.get("score",0)) or
        (home["team"]["id"] == tid and home.get("score",0) > away.get("score",0))
    )
    result_char = "W" if winner else "L"
    result_title = f"{title} {result_char}"

    # Bottom label: date only (Today/Tomorrow/Yesterday or Tue M/D)
    od = game.get("officialDate", "") or game.get("gameDate","")[:10]
    bottom = _rel_date_only(od)

    ls      = game.get("linescore", {}).get("teams", {})
    away_ls = ls.get("away", {})
    home_ls = ls.get("home", {})

    img  = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    away_lbl = get_mlb_abbreviation(get_team_display_name(away["team"]))
    home_lbl = get_mlb_abbreviation(get_team_display_name(home["team"]))

    _draw_boxscore_table(
        img, draw, result_title,
        away_lbl, away.get("score", 0), away_ls.get("hits", 0), away_ls.get("errors", 0),
        home_lbl, home.get("score", 0), home_ls.get("hits", 0), home_ls.get("errors", 0),
        bottom,
        away_team=away["team"],
        home_team=home["team"],
        screen_id=screen_id,
        reserve_flag_block=True,                      # keep layout identical Cubs/Sox
        live=False,
        winner_flag=(result_char if "Cubs" in title else None),  # flag only for Cubs
        hyperpixel_layout=hyperpixel_layout,
        center_content_vertically=(screen_id == "sox last"),
        flag_scale=(2.0 if screen_id == "cubs last" and is_hyperpixel_4_square_layout() else 1.0),
    )

    def _as_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    away_score = _as_int(away.get("score"))
    home_score = _as_int(home.get("score"))
    led_color = None
    if away_score is not None and home_score is not None and away_score != home_score:
        led_color = (
            (0.0, LED_INDICATOR_LEVEL, 0.0)
            if winner
            else (LED_INDICATOR_LEVEL, 0.0, 0.0)
        )

    return ScreenImage(img, displayed=False, led_override=led_color)


@log_call
def draw_box_score(display, game, title="Live Game...", transition=False, screen_id: Optional[str] = None):
    _set_background(screen_id)
    if not game:
        # no live game → let main loop advance immediately (no sleep)
        return None

    ls      = game.get("linescore", {})
    inning  = f"{ls.get('inningState','')} {ls.get('currentInningOrdinal','')}".strip() or "In Progress"
    away_ls = ls.get("teams", {}).get("away", {})
    home_ls = ls.get("teams", {}).get("home", {})

    img  = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    away_lbl = get_mlb_abbreviation(get_team_display_name(game["teams"]["away"]["team"]))
    home_lbl = get_mlb_abbreviation(get_team_display_name(game["teams"]["home"]["team"]))

    hyperpixel_layout = config.is_hyperpixel_next_layout() and screen_id in {
        "cubs live",
        "sox live",
    }

    _draw_boxscore_table(
        img, draw, title,
        away_lbl, game["teams"]["away"].get("score", 0),
        away_ls.get("hits", 0), away_ls.get("errors", 0),
        home_lbl, game["teams"]["home"].get("score", 0),
        home_ls.get("hits", 0), home_ls.get("errors", 0),
        inning,
        away_team=game["teams"]["away"]["team"],
        home_team=game["teams"]["home"]["team"],
        screen_id=screen_id,
        reserve_flag_block=True,
        live=True,
        hyperpixel_layout=hyperpixel_layout,
        center_content_vertically=(screen_id in {"cubs live", "sox live"}),
    )

    return ScreenImage(img, displayed=False)


@log_call
def draw_sports_screen(display, game, title, transition=False, screen_id: Optional[str] = None):
    _set_background(screen_id)
    if not game:
        logging.warning(f"No data for {title}")
        return None

    hyperpixel_layout = config.is_hyperpixel_next_layout() and screen_id in {
        "cubs next",
        "cubs next home",
        "sox next",
        "sox next home",
    }
    edge_pad = max(2, config.scale_value(2)) if hyperpixel_layout else 0
    line_gap = max(1, config.scale_value(1)) if hyperpixel_layout else 1

    img  = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    tw, th = draw.textsize(title, font=FONT_TITLE_SPORTS)
    draw.text(
        ((WIDTH - tw) // 2, edge_pad),
        title,
        font=FONT_TITLE_SPORTS,
        fill=(255, 255, 255),
    )

    home_tm = game['teams']['home']['team']
    away_tm = game['teams']['away']['team']
    cubs_id = int(MLB_CUBS_TEAM_ID)
    sox_id = int(MLB_SOX_TEAM_ID)

    focus_key = " ".join(filter(None, [screen_id, title])).lower()
    focus_id = None
    if "cubs" in focus_key:
        focus_id = cubs_id
    elif "sox" in focus_key:
        focus_id = sox_id

    if focus_id is not None:
        if away_tm.get('id') == focus_id:
            prefix, opponent = '@', get_team_display_name(home_tm)
        elif home_tm.get('id') == focus_id:
            prefix, opponent = 'vs.', get_team_display_name(away_tm)
        else:
            prefix, opponent = 'vs.', get_team_display_name(away_tm)
    elif away_tm.get('id') in (cubs_id, sox_id):
        prefix, opponent = '@', get_team_display_name(home_tm)
    else:
        prefix, opponent = 'vs.', get_team_display_name(away_tm)

    wrap_width = WIDTH - (edge_pad * 2) if hyperpixel_layout else WIDTH
    lines = wrap_text(f"{prefix} {opponent}", FONT_TEAM_SPORTS, wrap_width)[:2]
    y_text = edge_pad + th + (config.scale_value(4) if hyperpixel_layout else 4)
    for ln in lines:
        lw, lh = draw.textsize(ln, font=FONT_TEAM_SPORTS)
        draw.text(((WIDTH - lw)//2, y_text), ln, font=FONT_TEAM_SPORTS, fill=(255,255,255))
        y_text += lh + line_gap

    # logos + “@” inline
    def load_logo_for_tm(tm, frame_size: int):
        ab = get_mlb_tricode(tm) or get_mlb_abbreviation(get_team_display_name(tm))
        if not ab or frame_size <= 0:
            return None
        return load_team_logo(MLB_LOGOS_DIR, ab, box_size=frame_size)

    # Desired logo frame height mirrors the Hawks "Next Game" layout for consistency.
    desired_logo_h = standard_next_game_logo_height(HEIGHT)
    if hyperpixel_layout:
        desired_logo_h = max(1, int(round(desired_logo_h * config.DISPLAY_SCALE)))
    elif _IS_1080P_LAYOUT:
        desired_logo_h = max(1, int(round(desired_logo_h * _LOGO_SCALE_1080)))

    raw_date = game.get('officialDate','') or game.get('gameDate','')[:10]
    raw_time = game.get('startTimeCentral','TBD')
    bottom = _format_game_label(raw_date, raw_time)
    if bottom:
        try:
            _, t, _, b = draw.textbbox((0, 0), bottom, font=FONT_DATE_SPORTS)
            bl_h = b - t
        except Exception:
            bl_h = draw.textsize(bottom, font=FONT_DATE_SPORTS)[1]
    else:
        bl_h = 0
    bottom_margin = config.scale_value(BOTTOM_MARGIN) if hyperpixel_layout else BOTTOM_MARGIN
    if _IS_1080P_LAYOUT and (screen_id or "").strip().lower() == "cubs next":
        bottom_margin += _BOTTOM_TEXT_1080P_OFFSET
    bottom_y = HEIGHT - bl_h - bottom_margin

    available_h = max(10, bottom_y - (y_text + (edge_pad if hyperpixel_layout else 2)))
    logo_h = min(desired_logo_h, available_h)

    logo_away = load_logo_for_tm(away_tm, logo_h)
    logo_home = load_logo_for_tm(home_tm, logo_h)

    gap = config.scale_value(10) if hyperpixel_layout else 10
    max_frame_w = max(10, (WIDTH - (gap * 2) - draw.textsize("@", font=FONT_TEAM_SPORTS)[0]) // 2)
    frame_w = min(
        standard_next_game_logo_frame_width(logo_h, (logo_away, logo_home)),
        max_frame_w,
    )

    at_txt = "@"
    at_w, at_h = draw.textsize(at_txt, font=FONT_TEAM_SPORTS)
    block_h = logo_h if (logo_away or logo_home) else at_h

    if hyperpixel_layout:
        row_top = y_text + line_gap
        row_bottom = bottom_y
        row_space = max(0, row_bottom - row_top)
        row_y = row_top + max(0, (row_space - block_h) // 2)
    else:
        centered_top = (HEIGHT - block_h) // 2
        row_y = max(y_text + line_gap, min(centered_top, bottom_y - block_h - line_gap))

    total_w = frame_w * 2 + (gap * 2) + at_w
    start_x = max(0, (WIDTH - total_w) // 2)

    left_x = start_x
    at_x = left_x + frame_w + gap
    right_x = at_x + at_w + gap

    def _fit_logo_to_frame(logo: Image.Image | None) -> Image.Image | None:
        if not logo or frame_w <= 0 or logo_h <= 0:
            return logo
        width, height = logo.size
        if width <= 0 or height <= 0:
            return logo
        scale = min(frame_w / float(width), logo_h / float(height), 1.0)
        scale *= NEXT_GAME_LOGO_SCALE
        if scale >= 1.0:
            return logo
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        return logo.resize((new_width, new_height), Image.LANCZOS)

    def _draw_logo_box(frame_x: int) -> None:
        return None

    def _paste_logo(logo, frame_x):
        logo = _fit_logo_to_frame(logo)
        if not logo:
            return
        lx = frame_x + (frame_w - logo.width) // 2
        ly = row_y + (logo_h - logo.height) // 2
        img.paste(logo, (lx, ly), logo)

    _draw_logo_box(left_x)
    _paste_logo(logo_away, left_x)
    draw.text((at_x, row_y + (block_h - at_h)//2), at_txt, font=FONT_TEAM_SPORTS, fill=(255,255,255))
    _draw_logo_box(right_x)
    _paste_logo(logo_home, right_x)

    _center_bottom_text(draw, bottom, FONT_DATE_SPORTS, margin=bottom_margin)

    return ScreenImage(img, displayed=False)


@log_call
def draw_next_home_game(display, game, transition=False, screen_id: Optional[str] = None):
    """Wrapper to render the 'Next at home...' screen using sports layout."""
    return draw_sports_screen(
        display,
        game,
        "Next at home...",
        transition=transition,
        screen_id=screen_id,
    )

# ── Back-compat: main.py may still import this even though we no longer use it
@log_call
def draw_cubs_result(display, game, transition=False):
    """Deprecated full-screen Cubs flag; keep for import compatibility."""
    _set_background("cubs result")
    return None
