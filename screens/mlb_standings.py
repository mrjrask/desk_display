#!/usr/bin/env python3
"""
mlb_standings.py

Draw MLB division standings, overview, and Wild Card screens in RGB with:
- Drop-in animations on Overview (last place teams fall in first)
- Proper GB / WCGB formatting
- Wild Card screen scrolls bottom → top
"""

import os
import time
import requests
import logging
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageDraw

import config
from config import (
    SCOREBOARD_SCROLL_STEP,
    SCOREBOARD_SCROLL_DELAY,
    SCOREBOARD_SCROLL_PAUSE_TOP,
    SCOREBOARD_SCROLL_PAUSE_BOTTOM,
)
from utils import clear_display, get_mlb_abbreviation, get_mlb_tricode, log_call, log_missing_team_logo, clone_font
from utils import scroll_vertical_content
from screens.mlb_team_standings import format_games_back

# ─── Fonts / geometry from config ────────────────────────────────────────────
WIDTH  = config.WIDTH
HEIGHT = config.HEIGHT
FONT_DIV_HEADER = config.FONT_DIV_HEADER
FONT_DIV_RECORD = config.FONT_DIV_RECORD
FONT_GB_VALUE   = config.FONT_GB_VALUE
FONT_GB_LABEL   = config.FONT_GB_LABEL
if config.is_hyperpixel_4_square_layout():
    FONT_DIV_HEADER = clone_font(FONT_DIV_HEADER, max(8, int(round(getattr(FONT_DIV_HEADER, "size", 24) * 0.78))))
    FONT_DIV_RECORD = clone_font(FONT_DIV_RECORD, max(8, int(round(getattr(FONT_DIV_RECORD, "size", 22) * 0.78))))
    FONT_GB_VALUE = clone_font(FONT_GB_VALUE, max(8, int(round(getattr(FONT_GB_VALUE, "size", 20) * 0.78))))
    FONT_GB_LABEL = clone_font(FONT_GB_LABEL, max(8, int(round(getattr(FONT_GB_LABEL, "size", 20) * 0.78))))

# ─── Tunables ────────────────────────────────────────────────────────────────
LOGO_SIZE   = 52      # max width/height of a division logo
if config.is_hyperpixel_4_square_layout():
    LOGO_SIZE = max(1, int(round(LOGO_SIZE * 0.75)))
MARGIN      = 6       # left/right gutter
ROW_SPACING = 6       # vertical gap between rows

OV_COLS = 3           # East, Central, West columns on Overview
OV_ROWS = 5           # max teams to show per division on Overview
OVERVIEW_DROP_STEPS = 30
OVERVIEW_DROP_STAGGER = 0.4  # fraction of steps before next rank begins dropping
OVERVIEW_DROP_FRAME_DELAY = 0.02
PAUSE_END = 0.5

LEAGUE_DIVISION_IDS: Dict[int, Dict[str, int]] = {
    104: {"East": 204, "Central": 205, "West": 203},  # National League
    103: {"East": 201, "Central": 202, "West": 200},  # American League
}

# Logos live in the shared images directory alongside config.py. Using the
# config.IMAGES_DIR constant keeps the standings screen aligned with the other
# MLB screens (schedule, scoreboard, etc.) and avoids looking for a nonexistent
# ./screens/images/mlb folder.
LOGOS_DIR = os.path.join(config.IMAGES_DIR, "mlb")
TIMEOUT   = 10
BACKGROUND_COLOR = config.SCOREBOARD_BACKGROUND_COLOR


def _set_background(screen_id: str) -> None:
    global BACKGROUND_COLOR
    BACKGROUND_COLOR = config.get_screen_background_color(screen_id, config.SCOREBOARD_BACKGROUND_COLOR)


# ─────────────────────────────────────────────────────────────────────────────
# Data fetchers
# ─────────────────────────────────────────────────────────────────────────────

def _sort_by_int_key(items: List[dict], key: str) -> List[dict]:
    def _k(x):  # MLB API sends ranks as strings
        try:
            return int(x.get(key, 999))
        except Exception:
            return 999
    return sorted(items, key=_k)

def fetch_division_records(league_id: int, division_id: int) -> List[dict]:
    """
    Return teamRecords for a given league+division, sorted by divisionRank (1..N).
    """
    url = (
        "https://statsapi.mlb.com/api/v1/standings"
        f"?season=2025&leagueId={league_id}&divisionId={division_id}"
    )
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        records = r.json().get("records", [])
        rec = next(
            (x for x in records if x.get("division", {}).get("id") == division_id),
            None
        )
        if not rec:
            return []
        teams = rec.get("teamRecords", []) or []
        return _sort_by_int_key(teams, "divisionRank")
    except Exception as e:
        logging.error(f"Fetch standings L{league_id} D{division_id} failed: {e}")
        return []

def fetch_wildcard_records(league_id: int) -> List[dict]:
    """
    Return teamRecords for league Wild Card, sorted by wildCardRank (1..N).
    """
    url = (
        "https://statsapi.mlb.com/api/v1/standings"
        f"?season=2025&leagueId={league_id}&standingsTypes=wildCard"
    )
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json().get("records", [])
        teams = (data[0].get("teamRecords", []) if data else []) or []
        return _sort_by_int_key(teams, "wildCardRank")
    except Exception as e:
        logging.error(f"Fetch wildcard standings L{league_id} failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_logo(abbr: str, target: int, team_name: str = "") -> Optional[Image.Image]:
    """
    Load a team logo (PNG) and resize to fit within target×target box.
    """
    fn = f"{abbr.upper()}.png"
    path = os.path.join(LOGOS_DIR, fn)
    if not os.path.exists(path):
        log_missing_team_logo("MLB Standings", team_name, abbr)
        return None
    try:
        img = Image.open(path).convert("RGBA")
        w0, h0 = img.size
        s = min(target / w0, target / h0)
        return img.resize((max(1, int(w0*s)), max(1, int(h0*s))), Image.LANCZOS)
    except Exception as e:
        logging.warning(f"Logo load error {fn}: {e}")
        return None

def _header_frame(title: str) -> Tuple[Image.Image, int]:
    """
    Create a header-only frame and return (image, header_height).
    """
    img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    d = ImageDraw.Draw(img)
    tw, th = d.textsize(title, font=FONT_DIV_HEADER)
    d.text(((WIDTH - tw)//2, 0), title, font=FONT_DIV_HEADER, fill=(255,255,255))
    header_pad = config.scale_value(6) if config.is_hyperpixel_next_layout() else 6
    return img, th + header_pad


def _ease_out_cubic(t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    inv = 1.0 - t
    return 1.0 - inv * inv * inv


# ─────────────────────────────────────────────────────────────────────────────
# Overview (drop-in animation; last place drops first)
# ─────────────────────────────────────────────────────────────────────────────

@log_call
def draw_overview(display, title: str, league_id: int, transition=False):
    """
    Animated overview showing 3 columns (East, Central, West). Each column drops
    logos from last place up to first, onto a header-only background.
    """
    wait_for_skip = getattr(display, "wait_for_skip", None)
    skip_requested = getattr(display, "skip_requested", None)

    def _should_skip() -> bool:
        return bool(skip_requested and skip_requested())

    def _sleep(duration: float) -> bool:
        if callable(wait_for_skip):
            return bool(wait_for_skip(duration))
        time.sleep(duration)
        return False

    divisions = ["East", "Central", "West"]

    # Header-only base
    header, top_y = _header_frame(title)
    available_height = max(1, HEIGHT - top_y)
    hyperpixel_layout = config.is_hyperpixel_next_layout()
    if hyperpixel_layout:
        overview_margin = max(MARGIN, config.scale_value(MARGIN))
        available_width = max(1, WIDTH - 2 * overview_margin)
        cell_h = available_height / OV_ROWS
        col_width = available_width / OV_COLS
        padding = max(2, config.scale_value(4))
        logo_box = max(6, int(min(cell_h - padding * 2, col_width - padding * 2)))
        col_centers = [overview_margin + col_width * (i + 0.5) for i in range(OV_COLS)]
    else:
        cell_h = available_height // OV_ROWS
        col_w = LOGO_SIZE
        margin_x = (WIDTH - OV_COLS * col_w) // (OV_COLS + 1)
        col_centers = [margin_x * (i + 1) + col_w * i + col_w / 2 for i in range(OV_COLS)]
        logo_box = LOGO_SIZE

    # Load logos per division in standings order (1..N), trimmed to OV_ROWS
    logos_per_div: Dict[str, List[Optional[Image.Image]]] = {}
    for div in divisions:
        div_id = LEAGUE_DIVISION_IDS[league_id][div]
        recs = fetch_division_records(league_id, div_id)[:OV_ROWS]
        logos: List[Optional[Image.Image]] = []
        for rec in recs:
            abbr = get_mlb_tricode(rec.get("team")) or get_mlb_abbreviation(rec["team"]["name"])
            team_name = (rec.get("team") or {}).get("name") or "Unknown Team"
            logos.append(_load_logo(abbr, logo_box, team_name))
        # ensure length OV_ROWS (pad with None if short)
        while len(logos) < OV_ROWS:
            logos.append(None)
        logos_per_div[div] = logos

    row_positions: List[List[Tuple[Image.Image, int, int]]] = []
    for rank in range(OV_ROWS):
        placements: List[Tuple[Image.Image, int, int]] = []
        for ci, div in enumerate(divisions):
            ic = logos_per_div[div][rank]
            if not ic:
                continue
            x0 = int(col_centers[ci] - ic.width / 2)
            y_target = int(top_y + rank * cell_h + (cell_h - ic.height) / 2)
            placements.append((ic, x0, y_target))
        row_positions.append(placements)

    steps = max(2, OVERVIEW_DROP_STEPS)
    stagger = max(1, int(round(steps * OVERVIEW_DROP_STAGGER)))

    schedule: List[Tuple[int, List[Tuple[Image.Image, int, int]]]] = []
    start_step = 0
    for rank in range(len(row_positions) - 1, -1, -1):
        drops = row_positions[rank]
        if not drops:
            continue
        schedule.append((start_step, drops))
        start_step += stagger

    if schedule:
        total_duration = schedule[-1][0] + steps + 1
        placed: List[Tuple[Image.Image, int, int]] = []
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
            for ic, x0, y0 in placed:
                frame.paste(ic, (x0, y0), ic)

            for idx, (start, drops) in enumerate(schedule):
                progress = current_step - start
                if progress < 0 or progress >= steps:
                    continue

                frac = progress / (steps - 1) if steps > 1 else 1.0
                eased = _ease_out_cubic(frac)
                for ic, x0, y_target in drops:
                    start_y = -logo_box
                    y_pos = int(start_y + (y_target - start_y) * eased)
                    if y_pos > y_target:
                        y_pos = y_target
                    frame.paste(ic, (x0, y_pos), ic)

            display.image(frame)
            display.show()

            # Account for rendering time to maintain consistent frame rate
            elapsed = time.time() - frame_start
            sleep_time = max(0, OVERVIEW_DROP_FRAME_DELAY - elapsed)
            if sleep_time > 0 and _sleep(sleep_time):
                return header.copy() if transition else None

    # Final static image
    final = header.copy()
    for ri in range(OV_ROWS):
        for ci, div in enumerate(divisions):
            ic = logos_per_div[div][ri]
            if ic:
                x0 = int(col_centers[ci] - ic.width / 2)
                y0 = int(top_y + ri * cell_h + (cell_h - ic.height) / 2)
                final.paste(ic, (x0, y0), ic)

    display.image(final)
    display.show()
    _sleep(PAUSE_END)

    return final if transition else None


# ─────────────────────────────────────────────────────────────────────────────
# Division screen (static header + vertical scroll of rows)
# ─────────────────────────────────────────────────────────────────────────────

@log_call
def draw_division_screen(display, league_id: int, division_id: int, title: str, transition=False):
    teams = fetch_division_records(league_id, division_id)
    if not teams:
        clear_display(display)
        return None

    hyperpixel_layout = config.is_hyperpixel_next_layout()
    logo_size = config.scale_value_width(LOGO_SIZE) if hyperpixel_layout else LOGO_SIZE
    margin = config.scale_value(MARGIN) if hyperpixel_layout else MARGIN
    row_spacing = config.scale_value(ROW_SPACING) if hyperpixel_layout else ROW_SPACING
    header, header_h = _header_frame(title)

    # Build the list canvas (all rows)
    row_h  = logo_size + row_spacing
    list_h = row_h * len(teams) + config.SCOREBOARD_STANDINGS_BOTTOM_PADDING
    canvas = Image.new("RGB", (WIDTH, list_h), BACKGROUND_COLOR)
    cd     = ImageDraw.Draw(canvas)

    for i, rec in enumerate(teams):
        y = i * row_h
        row_center = y + row_h // 2

        # Logo
        abbr = get_mlb_tricode(rec.get("team")) or get_mlb_abbreviation(rec["team"]["name"])
        team_name = (rec.get("team") or {}).get("name") or "Unknown Team"
        ic = _load_logo(abbr, logo_size, team_name)
        if ic:
            logo_x = margin + (logo_size - ic.width)//2
            logo_y = row_center - ic.height // 2
            canvas.paste(ic, (logo_x, logo_y), ic)

        # GB column (right-aligned, label "GB")
        dgb = rec.get("divisionGamesBack", "-")
        gb_val = format_games_back(dgb) if dgb != "-" else "--"
        num_w, num_h = cd.textsize(gb_val, font=FONT_GB_VALUE)
        lab_w, lab_h = cd.textsize("GB",   font=FONT_GB_LABEL)
        gb_x = WIDTH - margin - (num_w + lab_w)
        gb_y = row_center - num_h // 2
        cd.text((gb_x, gb_y), gb_val, font=FONT_GB_VALUE, fill=(255,255,255))
        cd.text((gb_x + num_w, row_center - lab_h // 2), "GB", font=FONT_GB_LABEL, fill=(255,255,255))

        # W-L centered between logo block and GB
        wins = rec["leagueRecord"]["wins"]
        loss = rec["leagueRecord"]["losses"]
        rec_txt = f"{wins}-{loss}"
        rw2, rh2 = cd.textsize(rec_txt, font=FONT_DIV_RECORD)
        left  = margin + logo_size + margin
        right = gb_x - margin
        rec_x = left + ((right - left) - rw2)//2
        rec_y = row_center - rh2 // 2
        cd.text((rec_x, rec_y), rec_txt, font=FONT_DIV_RECORD, fill=(255,255,255))

    # Show first slice
    slice_first = canvas.crop((0, 0, WIDTH, HEIGHT - header_h))
    frame = header.copy()
    frame.paste(slice_first, (0, header_h))
    display.image(frame)
    display.show()

    visible_h = HEIGHT - header_h

    def _render_at_offset(offset: int) -> None:
        frame = header.copy()
        part = canvas.crop((0, offset, WIDTH, offset + visible_h))
        frame.paste(part, (0, header_h))
        display.image(frame)
        display.show()

    scroll_vertical_content(
        display=display,
        content_height=list_h,
        viewport_width=WIDTH,
        viewport_height=visible_h,
        render_at_offset=_render_at_offset,
        base_step=SCOREBOARD_SCROLL_STEP,
        pause_start=SCOREBOARD_SCROLL_PAUSE_TOP,
        pause_end=SCOREBOARD_SCROLL_PAUSE_BOTTOM,
        min_frame_time=SCOREBOARD_SCROLL_DELAY,
    )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Wild Card (bottom → top scroll; WCGB column)
# ─────────────────────────────────────────────────────────────────────────────

@log_call
def draw_wildcard_screen(display, league_id: int, title: str, transition=False):
    teams = fetch_wildcard_records(league_id)
    if not teams:
        clear_display(display)
        return None

    hyperpixel_layout = config.is_hyperpixel_next_layout()
    logo_size = config.scale_value_width(LOGO_SIZE) if hyperpixel_layout else LOGO_SIZE
    margin = config.scale_value(MARGIN) if hyperpixel_layout else MARGIN
    row_spacing = config.scale_value(ROW_SPACING) if hyperpixel_layout else ROW_SPACING
    header, header_h = _header_frame(title)

    row_h  = logo_size + row_spacing
    list_h = row_h * len(teams) + config.SCOREBOARD_STANDINGS_BOTTOM_PADDING
    canvas = Image.new("RGB", (WIDTH, list_h), BACKGROUND_COLOR)
    cd     = ImageDraw.Draw(canvas)

    for i, rec in enumerate(teams):
        y = i * row_h
        row_center = y + row_h // 2

        # Team logo
        abbr = get_mlb_tricode(rec.get("team")) or get_mlb_abbreviation(rec["team"]["name"])
        team_name = (rec.get("team") or {}).get("name") or "Unknown Team"
        ic = _load_logo(abbr, logo_size, team_name)
        if ic:
            canvas.paste(ic, (margin + (logo_size - ic.width)//2,
                              row_center - ic.height // 2), ic)

        # WCGB formatting
        raw_wcb = rec.get("wildCardGamesBack")
        try:
            wc_val = float(raw_wcb)
        except Exception:
            wc_val = None

        base = format_games_back(raw_wcb) if raw_wcb is not None else "--"
        if wc_val is None or wc_val == 0:
            s = "--"
        elif i < 3:
            s = f"+{base}"
        else:
            s = base

        # Right column labeled WCGB
        nw, nh = cd.textsize(s, font=FONT_GB_VALUE)
        lw, lh = cd.textsize("WCGB", font=FONT_GB_LABEL)
        start_x = WIDTH - margin - (nw + lw)
        y_text  = row_center - nh // 2
        cd.text((start_x, y_text), s, font=FONT_GB_VALUE, fill=(255,255,255))
        cd.text((start_x + nw, row_center - lh // 2), "WCGB", font=FONT_GB_LABEL, fill=(255,255,255))

        # W-L centered between logo block and WCGB
        rw, rl = rec["leagueRecord"]["wins"], rec["leagueRecord"]["losses"]
        rt = f"{rw}-{rl}"
        tw2, th2 = cd.textsize(rt, font=FONT_DIV_RECORD)
        left  = margin + logo_size + margin
        right = start_x - margin
        rec_x = left + ((right - left) - tw2)//2
        cd.text((rec_x, row_center - th2 // 2), rt, font=FONT_DIV_RECORD, fill=(255,255,255))

        # Separator below 3rd team (between #3 and #4): green line
        if i == 2:
            sep_y = y + row_h - row_spacing // 2
            cd.line((margin, sep_y, WIDTH - margin, sep_y), fill=(0, 255, 0))

    visible_h = HEIGHT - header_h

    def _render_at_offset(offset: int) -> None:
        frame = header.copy()
        part = canvas.crop((0, offset, WIDTH, offset + visible_h))
        frame.paste(part, (0, header_h))
        display.image(frame)
        display.show()

    scroll_vertical_content(
        display=display,
        content_height=list_h,
        viewport_width=WIDTH,
        viewport_height=visible_h,
        render_at_offset=_render_at_offset,
        base_step=SCOREBOARD_SCROLL_STEP,
        pause_start=SCOREBOARD_SCROLL_PAUSE_BOTTOM,
        pause_end=SCOREBOARD_SCROLL_PAUSE_TOP,
        reverse=True,
        min_frame_time=SCOREBOARD_SCROLL_DELAY,
    )
    return None


# ─── Wrappers expected by main.py ────────────────────────────────────────────

@log_call
def draw_NL_Overview(display, transition=False):
    _set_background("NL Overview")
    return draw_overview(display, "NL Overview", 104, transition)

@log_call
def draw_AL_Overview(display, transition=False):
    _set_background("AL Overview")
    return draw_overview(display, "AL Overview", 103, transition)

@log_call
def draw_NL_East(display, transition=False):
    _set_background("NL East")
    return draw_division_screen(display, 104, 204, "NL East", transition)

@log_call
def draw_NL_Central(display, transition=False):
    _set_background("NL Central")
    return draw_division_screen(display, 104, 205, "NL Central", transition)

@log_call
def draw_NL_West(display, transition=False):
    _set_background("NL West")
    return draw_division_screen(display, 104, 203, "NL West", transition)

@log_call
def draw_AL_East(display, transition=False):
    _set_background("AL East")
    return draw_division_screen(display, 103, 201, "AL East", transition)

@log_call
def draw_AL_Central(display, transition=False):
    _set_background("AL Central")
    return draw_division_screen(display, 103, 202, "AL Central", transition)

@log_call
def draw_AL_West(display, transition=False):
    _set_background("AL West")
    return draw_division_screen(display, 103, 200, "AL West", transition)

@log_call
def draw_NL_WildCard(display, transition=False):
    _set_background("NL Wild Card")
    return draw_wildcard_screen(display, 104, "NL Wild Card", transition)

@log_call
def draw_AL_WildCard(display, transition=False):
    _set_background("AL Wild Card")
    return draw_wildcard_screen(display, 103, "AL Wild Card", transition)
