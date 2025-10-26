#!/usr/bin/env python3
"""
Draw MLB box scores and next/last-game screens in RGB,
with both team logos on the Next Game screen, in AWAY @ HOME order,
and a small W/L flag between the boxscore and date on Cubs 'Last Game'.
"""

import os
import time
import logging
import datetime
from typing import Optional, Tuple
from PIL import Image, ImageDraw, Image

from config import (
    WIDTH, HEIGHT,
    FONT_TITLE_SPORTS, FONT_DATE_SPORTS,
    FONT_TEAM_SPORTS, FONT_SCORE,
    MLB_CUBS_TEAM_ID, MLB_SOX_TEAM_ID,
    CENTRAL_TIME
)
from utils import (
    clear_display,
    get_team_display_name,
    wrap_text,
    get_mlb_abbreviation,
    log_call,
    load_team_logo,
    standard_next_game_logo_height,
)

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR    = os.path.join(SCRIPT_DIR, "images")
MLB_LOGOS_DIR = os.path.join(IMAGES_DIR, "mlb")

# ── Layout constants ─────────────────────────────────────────────────────────
BOTTOM_MARGIN           = 4          # keep bottom text safely on-screen
TITLE_TO_HEADER_GAP     = 6          # space between title baseline and header labels
HEADER_GAP              = 3          # space between R/H/E labels and grid
TABLE_SIDE_MARGIN       = 4          # left/right inset of table
MIN_TEAM_COL_WIDTH      = 40         # never let the team column be narrower
DESIRED_SQUARE_FRACTION = 0.24       # starting point for square width vs total_w
GRID_BG                 = (14, 36, 22)  # dark forest green

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

    if label and time_display:
        return f"{label} {time_display}".strip()
    return label or time_display or ""

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

def _draw_title_with_bold_result(draw: ImageDraw.ImageDraw, title: str) -> tuple[int,int]:
    """Center the title. If it ends with ' W' or ' L', faux-bold that letter."""
    tw, th = draw.textsize(title, font=FONT_TITLE_SPORTS)
    x0 = (WIDTH - tw)//2
    draw.text((x0, 0), title, font=FONT_TITLE_SPORTS, fill=(255,255,255))
    if title.endswith(" W") or title.endswith(" L"):
        ch = title[-1]
        cw, _ = draw.textsize(ch, font=FONT_TITLE_SPORTS)
        cx = x0 + tw - cw
        cy = 0
        draw.text((cx, cy), ch, font=FONT_TITLE_SPORTS, fill=(255,255,255))
        draw.text((cx+1, cy), ch, font=FONT_TITLE_SPORTS, fill=(255,255,255))
    return tw, th

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

def _compute_table_geometry(draw: ImageDraw.ImageDraw,
                            top_y: int,
                            bottom_y: int,
                            reserve_flag_block: bool) -> dict:
    """
    Decide sizes so that columns 2–4 are true squares (same width as row height),
    and column 1 takes the rest. Ensures header labels sit above the grid with space.
    Returns a geometry dict.
    """
    # reserve space for the small-flag area (always, so Cubs/Sox align)
    grid_bottom_limit = bottom_y - FLAG_BLOCK_H if reserve_flag_block else bottom_y

    # Header row height = label text height + small padding
    hdr_h = draw.textsize("R", font=FONT_DATE_SPORTS)[1] + 2

    # Horizontal extents
    total_w = WIDTH - 2*TABLE_SIDE_MARGIN

    # Start with desired square size; clamp against minimum team width
    desired_sq = int(total_w * DESIRED_SQUARE_FRACTION)
    max_sq_by_width = (total_w - MIN_TEAM_COL_WIDTH) // 3
    square = max(18, min(desired_sq, max_sq_by_width))

    # Ensure grid fits vertically (2 rows of 'square' cells)
    grid_top = top_y + hdr_h + HEADER_GAP
    max_rows_h = grid_bottom_limit - grid_top
    if max_rows_h > 0:
        square = min(square, max_rows_h // 2)
    square = max(18, square)

    # Derive first column width from final square
    team_w = total_w - 3*square
    if team_w < MIN_TEAM_COL_WIDTH:
        square = max(18, (total_w - MIN_TEAM_COL_WIDTH) // 3)
        team_w = total_w - 3*square

    xs = [
        TABLE_SIDE_MARGIN,
        TABLE_SIDE_MARGIN + team_w,
        TABLE_SIDE_MARGIN + team_w + square,
        TABLE_SIDE_MARGIN + team_w + 2*square,
        TABLE_SIDE_MARGIN + team_w + 3*square,
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
                         reserve_flag_block: bool,
                         live: bool=False,
                         winner_flag: str|None=None):
    """
    Render the whole screen (title + header + table + optional small flag + bottom line).
    - Columns 2–4 are true squares; column 1 stretches.
    - Values are centered both horizontally and vertically in each cell.
    - Optional small W/L flag drawn only if 'winner_flag' is 'W' or 'L' (Cubs only).
    """
    # Title
    _, th = _draw_title_with_bold_result(draw, title)

    # Bottom line position
    bw, bh = draw.textsize(bottom_text, font=FONT_DATE_SPORTS)
    bottom_y = HEIGHT - bh - BOTTOM_MARGIN

    # Geometry
    g = _compute_table_geometry(draw, top_y=th + TITLE_TO_HEADER_GAP, bottom_y=bottom_y,
                                reserve_flag_block=reserve_flag_block)
    hdr_h   = g["hdr_h"]
    grid_top= g["grid_top"]
    row_h   = g["row_h"]
    team_w  = g["team_w"]
    square  = g["square"]
    xs      = g["xs"]
    total_w = g["grid_w"]
    grid_h  = g["grid_h"]

    # Header row (center each label over its column)
    for i, lbl in enumerate(["", "R", "H", "E"]):
        col_w = [team_w, square, square, square][i]
        # center exactly using bbox
        _bbox_center(draw,
                     x=xs[i],
                     y=(grid_top - HEADER_GAP) - hdr_h,
                     w=col_w,
                     h=hdr_h,
                     text=lbl,
                     font=FONT_DATE_SPORTS,
                     fill=(255,255,255))

    # Grid background (forest green) – exactly behind the 2×2 rows area
    draw.rectangle(
        (TABLE_SIDE_MARGIN, grid_top, TABLE_SIDE_MARGIN + total_w, grid_top + grid_h),
        fill=GRID_BG
    )

    # Grid outline + interior lines
    draw.rectangle(
        (TABLE_SIDE_MARGIN, grid_top, TABLE_SIDE_MARGIN + total_w, grid_top + grid_h),
        outline=(255,255,255)
    )
    # Vertical separators
    for x in xs[1:-1]:
        draw.line((x, grid_top, x, grid_top + grid_h), fill=(255,255,255))
    # Middle horizontal line
    draw.line((TABLE_SIDE_MARGIN, grid_top + row_h, TABLE_SIDE_MARGIN + total_w, grid_top + row_h),
              fill=(255,255,255))

    # Rows data (centered text in each cell)
    rows = [
        (away_lbl, away_r, away_h, away_e),
        (home_lbl, home_r, home_h, home_e)
    ]
    for ridx, (lbl, r, h, e) in enumerate(rows):
        for cidx, val in enumerate([lbl, r, h, e]):
            txt = str(val)
            fill_col = (255,255,0) if live and cidx > 0 else (255,255,255)
            font_use = FONT_TEAM_SPORTS if cidx == 0 else FONT_SCORE
            col_w    = [team_w, square, square, square][cidx]
            x_left   = xs[cidx]
            y_cell   = grid_top + ridx * row_h
            _bbox_center(draw, x_left, y_cell, col_w, row_h, txt, font_use, fill=fill_col)

    # Optional small W/L flag (Cubs only) – drawn in the reserved block
    if reserve_flag_block and winner_flag in ("W","L"):
        block_top = grid_top + grid_h + 2
        block_h   = FLAG_BLOCK_H
        flag_h    = SMALL_RESULT_FLAG_H
        flag_path = os.path.join(IMAGES_DIR, f"{winner_flag}.png")  # W.png / L.png
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
    draw.text(((WIDTH - bw)//2, bottom_y), bottom_text, font=FONT_DATE_SPORTS, fill=(255,255,255))


# ── Screens ─────────────────────────────────────────────────────────────────

@log_call
def draw_last_game(display, game, title="Last Game...", transition=False):
    if not game:
        logging.warning(f"No game data for {title}")
        return None

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

    img  = Image.new("RGB", (WIDTH, HEIGHT), "black")
    draw = ImageDraw.Draw(img)

    away_lbl = get_mlb_abbreviation(get_team_display_name(away["team"]))
    home_lbl = get_mlb_abbreviation(get_team_display_name(home["team"]))

    _draw_boxscore_table(
        img, draw, result_title,
        away_lbl, away.get("score", 0), away_ls.get("hits", 0), away_ls.get("errors", 0),
        home_lbl, home.get("score", 0), home_ls.get("hits", 0), home_ls.get("errors", 0),
        bottom,
        reserve_flag_block=True,                      # keep layout identical Cubs/Sox
        live=False,
        winner_flag=(result_char if "Cubs" in title else None)  # flag only for Cubs
    )

    if transition:
        return img

    clear_display(display)
    display.image(img)
    display.show()
    time.sleep(5)
    return None


@log_call
def draw_box_score(display, game, title="Live Game...", transition=False):
    if not game:
        # no live game → let main loop advance immediately (no sleep)
        return None

    ls      = game.get("linescore", {})
    inning  = f"{ls.get('inningState','')} {ls.get('currentInningOrdinal','')}".strip() or "In Progress"
    away_ls = ls.get("teams", {}).get("away", {})
    home_ls = ls.get("teams", {}).get("home", {})

    img  = Image.new("RGB", (WIDTH, HEIGHT), "black")
    draw = ImageDraw.Draw(img)

    away_lbl = get_mlb_abbreviation(get_team_display_name(game["teams"]["away"]["team"]))
    home_lbl = get_mlb_abbreviation(get_team_display_name(game["teams"]["home"]["team"]))

    _draw_boxscore_table(
        img, draw, title,
        away_lbl, game["teams"]["away"].get("score", 0),
        away_ls.get("hits", 0), away_ls.get("errors", 0),
        home_lbl, game["teams"]["home"].get("score", 0),
        home_ls.get("hits", 0), home_ls.get("errors", 0),
        inning,
        reserve_flag_block=False,
        live=True
    )

    if transition:
        return img

    clear_display(display)
    display.image(img)
    display.show()
    time.sleep(5)
    return None


@log_call
def draw_sports_screen(display, game, title, transition=False):
    if not game:
        logging.warning(f"No data for {title}")
        return None

    img  = Image.new("RGB", (WIDTH, HEIGHT), "black")
    draw = ImageDraw.Draw(img)

    tw, th = draw.textsize(title, font=FONT_TITLE_SPORTS)
    draw.text(((WIDTH - tw)//2, 0), title, font=FONT_TITLE_SPORTS, fill=(255,255,255))

    home_tm = game['teams']['home']['team']
    away_tm = game['teams']['away']['team']
    cubs_id = int(MLB_CUBS_TEAM_ID)
    sox_id  = int(MLB_SOX_TEAM_ID)

    if away_tm['id'] == cubs_id or away_tm['id'] == sox_id:
        prefix, opponent = '@', get_team_display_name(home_tm)
    else:
        prefix, opponent = 'vs.', get_team_display_name(away_tm)

    lines = wrap_text(f"{prefix} {opponent}", FONT_TEAM_SPORTS, WIDTH)[:2]
    y_text = th + 4
    for ln in lines:
        lw, lh = draw.textsize(ln, font=FONT_TEAM_SPORTS)
        draw.text(((WIDTH - lw)//2, y_text), ln, font=FONT_TEAM_SPORTS, fill=(255,255,255))
        y_text += lh + 1

    # logos + “@” inline
    def load_logo_for_tm(tm, height: int):
        ab = get_mlb_abbreviation(get_team_display_name(tm)).upper()
        if not ab or height <= 0:
            return None
        return load_team_logo(MLB_LOGOS_DIR, ab, height=height)

    # Desired logo height mirrors the Hawks "Next Game" layout for consistency.
    desired_logo_h = standard_next_game_logo_height(HEIGHT)

    raw_date = game.get('officialDate','') or game.get('gameDate','')[:10]
    raw_time = game.get('startTimeCentral','TBD')
    bottom   = _format_game_label(raw_date, raw_time)
    bl_w, bl_h = draw.textsize(bottom, font=FONT_DATE_SPORTS)
    bottom_y   = HEIGHT - bl_h - BOTTOM_MARGIN

    available_h = max(10, bottom_y - (y_text + 2))
    logo_h = min(desired_logo_h, available_h)

    logo_away = load_logo_for_tm(away_tm, logo_h)
    logo_home = load_logo_for_tm(home_tm, logo_h)

    elems = []
    if logo_away: elems.append(("img", logo_away))
    elems.append(("text", "@"))
    if logo_home: elems.append(("img", logo_home))

    spacing  = 8
    widths_el = [
        (obj.width if tp=="img" else draw.textsize(obj, font=FONT_TEAM_SPORTS)[0])
        for tp,obj in elems
    ]
    total_el_w = sum(widths_el) + spacing*(len(widths_el)-1)
    x0 = (WIDTH - total_el_w)//2

    block_h = logo_h if elems else draw.textsize("@", font=FONT_TEAM_SPORTS)[1]
    centered_top = (HEIGHT - block_h) // 2
    row_y = max(y_text + 1, min(centered_top, bottom_y - block_h - 1))

    x = x0
    for tp, obj in elems:
        if tp=="img":
            paste_y = row_y + (block_h - obj.height)//2
            img.paste(obj, (x, paste_y), obj)
            x += obj.width + spacing
        else:
            w_t, h_t = draw.textsize(obj, font=FONT_TEAM_SPORTS)
            y_o = row_y + (block_h - h_t)//2
            draw.text((x, y_o), obj, font=FONT_TEAM_SPORTS, fill=(255,255,255))
            x += w_t + spacing

    draw.text(((WIDTH - bl_w)//2, bottom_y), bottom, font=FONT_DATE_SPORTS, fill=(255,255,255))

    if transition:
        return img

    clear_display(display)
    display.image(img)
    display.show()
    time.sleep(5)
    return None


@log_call
def draw_next_home_game(display, game, transition=False):
    """Wrapper to render the 'Next at home...' screen using sports layout."""
    return draw_sports_screen(display, game, "Next at home...", transition=transition)

# ── Back-compat: main.py may still import this even though we no longer use it
@log_call
def draw_cubs_result(display, game, transition=False):
    """Deprecated full-screen Cubs flag; keep for import compatibility."""
    return None
