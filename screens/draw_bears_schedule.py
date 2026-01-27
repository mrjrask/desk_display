#!/usr/bin/env python3
"""
draw_bears_schedule.py

Shows the next Chicago Bears game with:
  - Title at y=0
  - Opponent wrapped in up to two lines, prefixed by '@' if the Bears are away,
    or 'vs.' if the Bears are home.
  - Between those and the bottom line, a row of logos: AWAY @ HOME, each logo
    auto-sized similarly to the Hawks schedule screen.
  - Bottom lines with week above date/time.
"""

import datetime
import os
import time
from functools import lru_cache
from PIL import Image, ImageDraw
import config
from config import (
    BEARS_BOTTOM_MARGIN,
    BEARS_SCHEDULE,
    NFL_TEAM_ABBREVIATIONS,
    get_screen_background_color,
)
from utils import (
    load_team_logo,
    next_game_from_schedule,
    standard_next_game_logo_frame_width,
    standard_next_game_logo_height_for_space,
    wrap_text,
)


def _text_size(draw, text, *, font):
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return (r - l, b - t)
    except Exception:
        return draw.textsize(text, font=font)


def _format_game_date(date_text: str) -> str:
    if not date_text:
        return ""
    date_text = str(date_text).strip()
    if not date_text:
        return ""
    for fmt in ("%a, %b %d %Y", "%a, %b %d, %Y", "%a, %b %d"):
        try:
            dt0 = datetime.datetime.strptime(date_text, fmt)
            return f"{dt0.month}/{dt0.day}"
        except Exception:
            continue
    return date_text


NFL_LOGO_DIR = os.path.join(config.IMAGES_DIR, "nfl")
DROP_STEPS = 18
DROP_STAGGER = 0.25
DROP_FRAME_DELAY = 0.01


@lru_cache(maxsize=96)
def _cached_team_logo(abbr: str, logo_size: int) -> Image.Image | None:
    return load_team_logo(NFL_LOGO_DIR, abbr, height=logo_size, box_size=logo_size)


def _ease_out_cubic(t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    inv = 1.0 - t
    return 1.0 - inv * inv * inv


def _animate_logo_drop(display, base: Image.Image, row_positions):
    has_logos = any(row for row in row_positions)
    if not has_logos:
        return

    steps = max(2, DROP_STEPS)
    stagger = max(1, int(round(steps * DROP_STAGGER)))

    schedule = []
    start_step = 0
    for row_idx in range(len(row_positions) - 1, -1, -1):
        drops = row_positions[row_idx]
        if not drops:
            continue
        schedule.append((start_step, drops))
        start_step += stagger

    if not schedule:
        return

    total_duration = schedule[-1][0] + steps + 1
    placed = []
    completed = [False] * len(schedule)

    for current_step in range(total_duration):
        frame_start = time.time()

        for idx, (start, drops) in enumerate(schedule):
            if current_step >= start + steps and not completed[idx]:
                placed.extend(drops)
                completed[idx] = True

        frame = base.copy()
        for logo, x0, y0 in placed:
            frame.paste(logo, (x0, y0), logo)

        for idx, (start, drops) in enumerate(schedule):
            progress = current_step - start
            if progress < 0 or progress >= steps:
                continue

            frac = progress / (steps - 1) if steps > 1 else 1.0
            eased = _ease_out_cubic(frac)
            for logo, x0, y_target in drops:
                start_y = -logo.height
                y_pos = int(start_y + (y_target - start_y) * eased)
                if y_pos > y_target:
                    y_pos = y_target
                frame.paste(logo, (x0, y_pos), logo)

        display.image(frame)
        if hasattr(display, "show"):
            display.show()

        elapsed = time.time() - frame_start
        sleep_time = max(0, DROP_FRAME_DELAY - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
def show_bears_next_game(display, transition=False):
    game = next_game_from_schedule(BEARS_SCHEDULE)
    title = "Next for Da Bears:"
    background = get_screen_background_color("bears next", (0, 0, 0))
    img   = Image.new("RGB", (config.WIDTH, config.HEIGHT), background)
    draw  = ImageDraw.Draw(img)

    # Title
    tw, th = draw.textsize(title, font=config.FONT_TITLE_SPORTS)
    draw.text(((config.WIDTH - tw)//2, 0), title,
              font=config.FONT_TITLE_SPORTS, fill=(255,255,255))

    if game:
        opp = game["opponent"]
        ha  = game["home_away"].lower()
        prefix = "@" if ha == "away" else "vs."

        # Opponent text (up to 2 lines)
        lines  = wrap_text(f"{prefix} {opp}", config.FONT_TEAM_SPORTS, config.WIDTH)[:2]
        y_txt  = th + 4
        for ln in lines:
            w_ln, h_ln = draw.textsize(ln, font=config.FONT_TEAM_SPORTS)
            draw.text(((config.WIDTH - w_ln)//2, y_txt),
                      ln, font=config.FONT_TEAM_SPORTS, fill=(255,255,255))
            y_txt += h_ln + 2

        # Logos row: AWAY @ HOME
        bears_ab = "chi"
        opp_key = opp.split()[-1].lower()
        opp_ab = NFL_TEAM_ABBREVIATIONS.get(opp_key, opp_key[:3])
        week_label = str(game.get("week", "") or "")
        if opp.strip().upper() == "TBD":
            if "super bowl" in week_label.lower():
                opp_ab = "afc"
            else:
                opp_ab = "nfc"
        if opp_ab == "was":
            opp_ab = "wsh"
        if ha == "away":
            away_ab, home_ab = bears_ab, opp_ab
        else:
            away_ab, home_ab = opp_ab, bears_ab

        # Bottom lines text — week above date/time
        wk = (game.get("week") or "").strip()
        if not wk:
            game_no = str(game.get("game_no", "")).strip()
            wk = f"Game {game_no}" if game_no else ""
        date_txt = _format_game_date(game.get("date", ""))
        t_txt = game["time"].strip()
        date_time = " ".join(part for part in (date_txt, t_txt) if part).strip()
        bottom_lines = [line for line in (wk, date_time) if line]
        line_gap = 2
        if bottom_lines:
            heights = [
                _text_size(draw, line, font=config.FONT_DATE_SPORTS)[1]
                for line in bottom_lines
            ]
            bottom_h = sum(heights) + (line_gap * (len(bottom_lines) - 1))
        else:
            bottom_h = 0
        bottom_y = config.HEIGHT - bottom_h - BEARS_BOTTOM_MARGIN  # keep on-screen

        available_h = max(10, bottom_y - (y_txt + 2))
        logo_h = standard_next_game_logo_height_for_space(config.HEIGHT, available_h)

        logo_away = load_team_logo(NFL_LOGO_DIR, away_ab, height=logo_h, box_size=logo_h)
        logo_home = load_team_logo(NFL_LOGO_DIR, home_ab, height=logo_h, box_size=logo_h)

        gap = max(6, min(10, config.WIDTH // 30))
        frame_w = standard_next_game_logo_frame_width(logo_h, (logo_away, logo_home))
        at_symbol = "@"
        try:
            l, t, r, b = draw.textbbox((0, 0), at_symbol, font=config.FONT_TEAM_SPORTS)
            at_w, at_h, at_t = r - l, b - t, t
        except Exception:
            at_w, at_h = draw.textsize(at_symbol, font=config.FONT_TEAM_SPORTS)
            at_t = 0

        block_h = logo_h if (logo_away or logo_home) else at_h
        total_w = (frame_w * 2) + (gap * 2) + at_w

        if total_w > config.WIDTH:
            gap = max(4, int(round(gap * (config.WIDTH / max(total_w, 1)))))
            total_w = (frame_w * 2) + (gap * 2) + at_w

        if total_w > config.WIDTH:
            max_frame = max(1, (config.WIDTH - at_w - (gap * 2)) // 2)
            if max_frame < frame_w:
                scale = max_frame / frame_w if frame_w else 1.0
                logo_h = max(1, int(round(logo_h * scale)))
                logo_away = load_team_logo(NFL_LOGO_DIR, away_ab, height=logo_h, box_size=logo_h)
                logo_home = load_team_logo(NFL_LOGO_DIR, home_ab, height=logo_h, box_size=logo_h)
                frame_w = min(standard_next_game_logo_frame_width(logo_h, (logo_away, logo_home)), max_frame)

            def _fit_logo(logo):
                if logo and logo.width > frame_w:
                    ratio = frame_w / logo.width
                    new_h = max(1, int(round(logo.height * ratio)))
                    return logo.resize((frame_w, new_h), Image.ANTIALIAS)
                return logo

            logo_away = _fit_logo(logo_away)
            logo_home = _fit_logo(logo_home)
            block_h = max((logo.height for logo in (logo_away, logo_home) if logo), default=at_h if not (logo_away or logo_home) else logo_h)
            total_w = (frame_w * 2) + (gap * 2) + at_w

        x0 = max(0, (config.WIDTH - total_w) // 2)

        # Vertical center of logos/text block between opponent text and bottom label
        y_logo = y_txt + ((bottom_y - y_txt) - block_h)//2

        left_x = x0
        at_x = left_x + frame_w + gap
        right_x = at_x + at_w + gap

        def _paste_logo(logo, frame_x):
            if not logo:
                return
            lx = frame_x + (frame_w - logo.width)//2
            ly = y_logo + (logo_h - logo.height)//2
            img.paste(logo, (lx, ly), logo)

        _paste_logo(logo_away, left_x)
        at_y = y_logo + (block_h - at_h)//2 - at_t
        draw.text((at_x, at_y), at_symbol, font=config.FONT_TEAM_SPORTS, fill=(255,255,255))
        _paste_logo(logo_home, right_x)

        # Draw bottom text
        if bottom_lines:
            y_bottom_text = bottom_y
            for line in bottom_lines:
                w_line, h_line = _text_size(draw, line, font=config.FONT_DATE_SPORTS)
                draw.text(
                    ((config.WIDTH - w_line) // 2, y_bottom_text),
                    line,
                    font=config.FONT_DATE_SPORTS,
                    fill=(255, 255, 255),
                )
                y_bottom_text += h_line + line_gap

    if transition:
        return img

    display.image(img)
    display.show()
    return None


def show_bears_next_season(display, transition=False):
    title = "2026 Bears Opponents"
    background = get_screen_background_color("bears next season", (0, 0, 0))
    img = Image.new("RGB", (config.WIDTH, config.HEIGHT), background)
    draw = ImageDraw.Draw(img)

    home_opponents = ["det", "gb", "min", "tb", "phi", "jax", "nyj", "ne", "no"]
    away_opponents = ["det", "gb", "min", "buf", "mia", "atl", "car", "sea"]

    title_w, title_h = _text_size(draw, title, font=config.FONT_TITLE_SPORTS)
    draw.text(
        ((config.WIDTH - title_w) // 2, 0),
        title,
        font=config.FONT_TITLE_SPORTS,
        fill=(255, 255, 255),
    )

    column_width = config.WIDTH // 2
    header_y = title_h + 4
    header_font = config.FONT_DATE_SPORTS
    home_label = "Home"
    away_label = "Away"
    home_w, home_h = _text_size(draw, home_label, font=header_font)
    away_w, away_h = _text_size(draw, away_label, font=header_font)

    draw.text(
        ((column_width - home_w) // 2, header_y),
        home_label,
        font=header_font,
        fill=(255, 255, 255),
    )
    draw.text(
        (column_width + (column_width - away_w) // 2, header_y),
        away_label,
        font=header_font,
        fill=(255, 255, 255),
    )

    logos_top = header_y + max(home_h, away_h) + 4
    row_gap = 2
    col_gap = 4
    columns_per_side = 2
    home_rows = (len(home_opponents) + columns_per_side - 1) // columns_per_side
    away_rows = (len(away_opponents) + columns_per_side - 1) // columns_per_side
    rows = max(home_rows, away_rows)
    available_h = config.HEIGHT - logos_top - 2
    subcolumn_width = max(1, (column_width - col_gap) // columns_per_side)
    logo_size = max(
        1,
        min(subcolumn_width, (available_h - row_gap * (rows - 1)) // rows),
    )

    def _logo_position(logo, x, y):
        lx = x + (logo_size - logo.width) // 2
        ly = y + (logo_size - logo.height) // 2
        return lx, ly

    placements_by_row = [[] for _ in range(rows)]
    placements = []

    for idx, abbr in enumerate(home_opponents):
        row = idx // columns_per_side
        col = idx % columns_per_side
        y = logos_top + row * (logo_size + row_gap)
        x = col * (subcolumn_width + col_gap) + (subcolumn_width - logo_size) // 2
        logo = _cached_team_logo(abbr, logo_size)
        if logo:
            lx, ly = _logo_position(logo, x, y)
            placements_by_row[row].append((logo, lx, ly))
            placements.append((logo, lx, ly))

    for idx, abbr in enumerate(away_opponents):
        row = idx // columns_per_side
        col = idx % columns_per_side
        y = logos_top + row * (logo_size + row_gap)
        x = (
            column_width
            + col * (subcolumn_width + col_gap)
            + (subcolumn_width - logo_size) // 2
        )
        logo = _cached_team_logo(abbr, logo_size)
        if logo:
            lx, ly = _logo_position(logo, x, y)
            placements_by_row[row].append((logo, lx, ly))
            placements.append((logo, lx, ly))

    final_img = img.copy()
    for logo, lx, ly in placements:
        final_img.paste(logo, (lx, ly), logo)

    if transition:
        return final_img

    _animate_logo_drop(display, img, placements_by_row)
    display.image(final_img)
    display.show()
    return None
