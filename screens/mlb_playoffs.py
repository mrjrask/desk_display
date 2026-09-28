#!/usr/bin/env python3
"""Render the MLB postseason bracket with the current round's series below it.

The bracket follows mlb.com/postseason: the American League on the left, the
National League on the right, the World Series in the middle, and rounds
running Wild Card → Division Series → LCS toward the centre.  Under it, each
series of the round being played shows both teams, the series score and the
next game (or who won).
"""

from __future__ import annotations

import datetime
import logging
import os
import time
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont

from config import (
    CENTRAL_TIME,
    FONT_STATUS,
    FONT_TEAM_SPORTS,
    FONT_TITLE_SPORTS,
    HEIGHT,
    IMAGES_DIR,
    SCOREBOARD_IN_PROGRESS_SCORE_COLOR,
    SCOREBOARD_SCROLL_DELAY,
    SCOREBOARD_SCROLL_PAUSE_BOTTOM,
    SCOREBOARD_SCROLL_PAUSE_TOP,
    SCOREBOARD_SCROLL_STEP,
    SCOREBOARD_STANDINGS_BOTTOM_PADDING,
    WIDTH,
    get_screen_font,
    scale_value,
)
from screens.scoreboard_components import center_text
from services.sports import mlb_postseason
from utils import (
    ScreenImage,
    clear_display,
    clone_font,
    load_team_logo,
    scroll_vertical_content,
)

SCREEN_ID = "MLB Playoffs"
TITLE = "MLB Playoffs"
LOGO_DIR = os.path.join(IMAGES_DIR, "mlb")

BACKGROUND_COLOR = (0, 0, 0)
WHITE = (255, 255, 255)
DIM_TEXT = (120, 120, 120)
LABEL_TEXT = (170, 170, 170)
BOX_FILL = (22, 22, 22)
BOX_OUTLINE = (70, 70, 70)
CONNECTOR = (90, 90, 90)
WINNER_TEXT = (0, 200, 0)
SEPARATOR = (45, 45, 45)
LOSER_LOGO_ALPHA = 0.35

# Below this width a seven-column bracket's logos are too small to read, so
# the screen lists the series only.
MIN_BRACKET_WIDTH = 200
# At or above this width the series list uses two columns (AL | NL).
TWO_COLUMN_MIN_WIDTH = 640

_LOGO_CACHE: dict[tuple[str, int], Optional[Image.Image]] = {}


# ─── Fonts and logos ──────────────────────────────────────────────────────────


def _font(base: ImageFont.FreeTypeFont, size: int) -> ImageFont.FreeTypeFont:
    return clone_font(base, max(6, int(size)))


def _title_font():
    return get_screen_font(SCREEN_ID, "title", base_font=FONT_TITLE_SPORTS, default_size=30)


def _round_font():
    return get_screen_font(SCREEN_ID, "round", base_font=FONT_STATUS, default_size=16)


def _status_font():
    return get_screen_font(SCREEN_ID, "status", base_font=FONT_STATUS, default_size=15)


def _score_font():
    return get_screen_font(SCREEN_ID, "score", base_font=FONT_TEAM_SPORTS, default_size=26)


def _seed_font():
    return get_screen_font(SCREEN_ID, "seed", base_font=FONT_STATUS, default_size=12)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int, int, int]:
    """Return ``(width, height, left, top)`` of *text*'s ink box."""

    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top, left, top


def _fit_font(draw: ImageDraw.ImageDraw, text: str, font, max_width: int):
    """Shrink *font* until *text* fits in *max_width* pixels."""

    size = int(getattr(font, "size", 0) or 0)
    while size > 6 and _text_size(draw, text, font)[0] > max_width:
        size -= 1
        font = _font(font, size)
    return font


def _logo(abbr: Optional[str], size: int) -> Optional[Image.Image]:
    if not abbr or size <= 0:
        return None
    key = (abbr.upper(), size)
    if key not in _LOGO_CACHE:
        _LOGO_CACHE[key] = load_team_logo(LOGO_DIR, abbr.upper(), height=size, box_size=size, trim=True)
    return _LOGO_CACHE[key]


def _dimmed(logo: Image.Image) -> Image.Image:
    faded = logo.convert("RGBA")
    alpha = faded.getchannel("A").point(lambda value: int(value * LOSER_LOGO_ALPHA))
    faded.putalpha(alpha)
    return faded


def _paste_logo(canvas: Image.Image, abbr: Optional[str], size: int, x: int, y: int, *, dim: bool = False) -> bool:
    logo = _logo(abbr, size)
    if logo is None:
        return False
    if dim:
        logo = _dimmed(logo)
    x0 = x + (size - logo.width) // 2
    y0 = y + (size - logo.height) // 2
    canvas.paste(logo, (x0, y0), logo if logo.mode == "RGBA" else None)
    return True


def _draw_text_left(draw, text: str, font, x: int, y: int, height: int, fill) -> int:
    width, text_h, left, top = _text_size(draw, text, font)
    draw.text((x - left, y + (height - text_h) // 2 - top), text, font=font, fill=fill)
    return width


def _draw_text_right(draw, text: str, font, right: int, y: int, height: int, fill) -> int:
    width, text_h, left, top = _text_size(draw, text, font)
    draw.text((right - width - left, y + (height - text_h) // 2 - top), text, font=font, fill=fill)
    return width


# ─── Series text ──────────────────────────────────────────────────────────────


def _format_clock(moment: datetime.datetime) -> str:
    return moment.strftime("%-I %p") if moment.minute == 0 else moment.strftime("%-I:%M %p")


def _day_label(moment: datetime.datetime, now: datetime.datetime) -> str:
    if moment.date() == now.date():
        return "Tonight" if moment.hour >= 17 else "Today"
    if moment.date() == (now + datetime.timedelta(days=1)).date():
        return "Tomorrow"
    if 0 <= (moment.date() - now.date()).days < 7:
        return moment.strftime("%A")
    return f"{moment.strftime('%a')} {moment.month}/{moment.day}"


def _next_game_text(series: dict, now: datetime.datetime) -> str:
    start = mlb_postseason._parse_iso(series.get("next_start"))
    if start is None:
        return ""
    local = start.astimezone(CENTRAL_TIME)
    when = _day_label(local, now)
    if not series.get("next_time_tbd"):
        when = f"{when} {_format_clock(local)}"
    number = series.get("next_game")
    return f"Game {number} · {when}" if number else when


def _team_name(data: dict, abbr: str) -> str:
    return str(((data or {}).get("names") or {}).get(abbr) or abbr)


def series_status(slot: dict, data: dict, now: Optional[datetime.datetime] = None) -> tuple[str, tuple[int, int, int]]:
    """The line under a series: who won, a live game, or the next game."""

    now = now or datetime.datetime.now(CENTRAL_TIME)
    series = slot.get("series")
    if not series:
        return "Projected", DIM_TEXT
    teams = slot["teams"]
    wins = [w or 0 for w in slot["wins"]]
    winner = series.get("winner")
    if winner:
        high, low = max(wins), min(wins)
        return f"{_team_name(data, winner)} win {high}-{low}", WINNER_TEXT
    if series.get("live"):
        played = sum(wins) + 1
        return f"Game {played} · LIVE", SCOREBOARD_IN_PROGRESS_SCORE_COLOR
    next_text = _next_game_text(series, now)
    if next_text:
        return next_text, WHITE
    if wins[0] == wins[1]:
        return f"Series tied {wins[0]}-{wins[1]}", WHITE
    leader = 0 if wins[0] > wins[1] else 1
    return f"{teams[leader]} leads {wins[leader]}-{wins[1 - leader]}", WHITE


# ─── Bracket ──────────────────────────────────────────────────────────────────


class _BracketLayout:
    """Pixel geometry of the seven-column bracket for the current display."""

    def __init__(self, width: int, height: int):
        self.margin = max(2, width // 80)
        self.col_gap = max(2, width // 64)
        self.col_w = max(10, (width - 2 * self.margin - 6 * self.col_gap) // 7)
        used = 7 * self.col_w + 6 * self.col_gap
        self.left = (width - used) // 2
        self.row_h = max(8, min(int(self.col_w * 0.45), height // 9))
        self.slot_h = 2 * self.row_h
        self.label_h = max(8, int(self.row_h * 0.8))
        self.line_w = max(1, self.row_h // 14)
        # Division Series slots sit at rows 0 and 3; the Wild Card slots feed
        # their bottom (wild-card) row, so they sit half a row lower.
        self.body_h = int(self.row_h * 5.5)
        self.height = self.label_h + self.body_h + self.row_h // 2

    def col_x(self, index: int) -> int:
        return self.left + index * (self.col_w + self.col_gap)


def _draw_slot(canvas: Image.Image, draw: ImageDraw.ImageDraw, layout: _BracketLayout, slot: dict,
               seeds: dict[str, int], x: int, y: int) -> None:
    row_h, col_w = layout.row_h, layout.col_w
    radius = max(1, row_h // 6)
    draw.rounded_rectangle((x, y, x + col_w - 1, y + layout.slot_h - 1), radius=radius,
                           fill=BOX_FILL, outline=BOX_OUTLINE, width=layout.line_w)
    draw.line((x + 1, y + row_h, x + col_w - 2, y + row_h), fill=BOX_OUTLINE, width=layout.line_w)

    series = slot.get("series")
    winner = slot.get("winner")
    live = bool(series and series.get("live"))
    pad = max(1, row_h // 10)
    logo_size = row_h - 2 * pad
    score_font = _font(FONT_TEAM_SPORTS, row_h * 0.72)
    seed_font = _font(FONT_STATUS, row_h * 0.46)
    seed_w = _text_size(draw, "6", seed_font)[0] + pad
    show_seeds = bool(seeds) and col_w >= seed_w + logo_size + _text_size(draw, "4", score_font)[0] + 4 * pad
    for index, team in enumerate(slot["teams"]):
        top = y + index * row_h
        if not team:
            center_text(draw, "TBD", _font(FONT_STATUS, row_h * 0.5), x, col_w, top, row_h, fill=DIM_TEXT)
            continue
        lost = bool(winner) and team != winner
        cursor = x + pad
        seed = seeds.get(team)
        if show_seeds:
            if seed:
                _draw_text_left(draw, str(seed), seed_font, cursor, top, row_h, DIM_TEXT)
            cursor += seed_w
        if not _paste_logo(canvas, team, logo_size, cursor, top + pad, dim=lost):
            _draw_text_left(draw, team, _font(FONT_STATUS, row_h * 0.45), cursor, top, row_h,
                            DIM_TEXT if lost else WHITE)
        wins = slot["wins"][index]
        if series is None or wins is None:
            continue
        fill = SCOREBOARD_IN_PROGRESS_SCORE_COLOR if live else (DIM_TEXT if lost else WHITE)
        _draw_text_right(draw, str(wins), score_font, x + col_w - pad - 1, top, row_h, fill)


def _hline(draw, layout, x0, x1, y):
    draw.line((min(x0, x1), y, max(x0, x1), y), fill=CONNECTOR, width=layout.line_w)


def _draw_bracket(canvas: Image.Image, top: int, bracket: dict) -> None:
    layout = _BracketLayout(canvas.width, HEIGHT)
    draw = ImageDraw.Draw(canvas)
    row_h = layout.row_h
    seeds = bracket.get("seeds") or {}

    label_font = _font(FONT_STATUS, layout.label_h * 0.85)
    labels = ["WC", "DS", "LCS", "WS", "LCS", "DS", "WC"]
    for index, label in enumerate(labels):
        center_text(draw, label, label_font, layout.col_x(index), layout.col_w, top, layout.label_h, fill=LABEL_TEXT)

    body = top + layout.label_h + max(1, row_h // 4)
    division_y = [body, body + 3 * row_h]
    wild_card_y = [y + row_h // 2 for y in division_y]
    center_y = body + int(2.5 * row_h)
    lcs_y = center_y - row_h

    for league, cols, inward in (("AL", (0, 1, 2), 1), ("NL", (6, 5, 4), -1)):
        rounds = bracket.get(league) or {}
        wc_x, ds_x, lcs_x = (layout.col_x(c) for c in cols)
        for index, slot in enumerate(rounds.get("F") or []):
            _draw_slot(canvas, draw, layout, slot, seeds, wc_x, wild_card_y[index])
            edge = wc_x + layout.col_w if inward > 0 else wc_x
            target = ds_x if inward > 0 else ds_x + layout.col_w
            _hline(draw, layout, edge, target, wild_card_y[index] + row_h)
        for index, slot in enumerate(rounds.get("D") or []):
            _draw_slot(canvas, draw, layout, slot, seeds, ds_x, division_y[index])
        # Division Series → LCS elbow.
        ds_edge = ds_x + layout.col_w if inward > 0 else ds_x
        lcs_edge = lcs_x if inward > 0 else lcs_x + layout.col_w
        elbow_x = (ds_edge + lcs_edge) // 2
        for y in division_y:
            _hline(draw, layout, ds_edge, elbow_x, y + row_h)
        draw.line((elbow_x, division_y[0] + row_h, elbow_x, division_y[1] + row_h), fill=CONNECTOR, width=layout.line_w)
        _hline(draw, layout, elbow_x, lcs_edge, center_y)
        for slot in rounds.get("L") or []:
            _draw_slot(canvas, draw, layout, slot, seeds, lcs_x, lcs_y)
        # League logo above its LCS slot.
        logo_size = min(int(row_h * 1.1), lcs_y - body - 2)
        if logo_size > 4:
            _paste_logo(canvas, league, logo_size, lcs_x + (layout.col_w - logo_size) // 2, body)
        ws_x = layout.col_x(3)
        _hline(draw, layout, lcs_x + layout.col_w if inward > 0 else lcs_x,
               ws_x if inward > 0 else ws_x + layout.col_w, center_y)

    ws_x = layout.col_x(3)
    for slot in bracket.get("W") or []:
        _draw_slot(canvas, draw, layout, slot, {}, ws_x, lcs_y)
        champion = slot.get("winner")
        if champion:
            size = min(int(row_h * 1.2), lcs_y - body - 2)
            if size > 4:
                _paste_logo(canvas, champion, size, ws_x + (layout.col_w - size) // 2, body)


# ─── Series list ──────────────────────────────────────────────────────────────


def _list_slots(bracket: dict, round_code: str) -> list[tuple[str, dict]]:
    if round_code == "W":
        return [("", slot) for slot in bracket.get("W") or []]
    return [(league, slot) for league in mlb_postseason.LEAGUES for slot in (bracket.get(league) or {}).get(round_code) or []]


def _series_row_height() -> tuple[int, int]:
    return scale_value(34), scale_value(20)


def _draw_series_row(canvas: Image.Image, draw: ImageDraw.ImageDraw, slot: dict, data: dict,
                     seeds: dict[str, int], x: int, width: int, y: int, now: datetime.datetime) -> None:
    row_h, status_h = _series_row_height()
    logo_size = max(6, row_h - scale_value(4))
    score_font = _score_font()
    seed_font = _seed_font()
    series = slot.get("series")
    winner = slot.get("winner")
    live = bool(series and series.get("live"))
    gap = max(2, scale_value(6))
    dash_w = max(scale_value(18), _text_size(draw, "-", score_font)[0] + 2 * gap)
    center = x + width // 2
    for index, team in enumerate(slot["teams"]):
        direction = -1 if index == 0 else 1
        lost = bool(winner) and team != winner
        edge = center + direction * dash_w // 2
        wins = slot["wins"][index]
        score_text = "" if series is None or wins is None else str(wins)
        fill = SCOREBOARD_IN_PROGRESS_SCORE_COLOR if live else (DIM_TEXT if lost else WHITE)
        if index == 0:
            score_w = _draw_text_right(draw, score_text, score_font, edge, y, row_h, fill) if score_text else 0
            logo_x = edge - score_w - gap - logo_size
        else:
            score_w = _draw_text_left(draw, score_text, score_font, edge, y, row_h, fill) if score_text else 0
            logo_x = edge + score_w + gap
        if not team:
            center_text(draw, "TBD", _status_font(), logo_x, logo_size, y, row_h, fill=DIM_TEXT)
            continue
        if not _paste_logo(canvas, team, logo_size, logo_x, y + (row_h - logo_size) // 2, dim=lost):
            center_text(draw, team, _status_font(), logo_x, logo_size, y, row_h, fill=DIM_TEXT if lost else WHITE)
        seed = seeds.get(team)
        if seed:
            seed_text = f"({seed})"
            if index == 0:
                _draw_text_right(draw, seed_text, seed_font, logo_x - gap // 2, y, row_h, DIM_TEXT)
            else:
                _draw_text_left(draw, seed_text, seed_font, logo_x + logo_size + gap // 2, y, row_h, DIM_TEXT)
    if slot["teams"][0] or slot["teams"][1]:
        center_text(draw, "-", score_font, center - dash_w // 2, dash_w, y, row_h, fill=DIM_TEXT)
    text, fill = series_status(slot, data, now)
    status_font = _fit_font(draw, text, _status_font(), width - 2 * gap)
    center_text(draw, text, status_font, x, width, y + row_h, status_h, fill=fill)


def _round_heading(data: dict, round_code: str, projected: bool) -> str:
    name = mlb_postseason.ROUND_NAMES[round_code]
    if projected:
        return f"Projected {name}"
    series = [s for s in data.get("series") or [] if s.get("round") == round_code]
    best_of = max((s.get("best_of") or 0 for s in series), default=0) or mlb_postseason.DEFAULT_BEST_OF[round_code]
    return f"{name} · Best of {best_of}"


def _compose_series_list(width: int, data: dict, bracket: dict, now: datetime.datetime) -> Optional[Image.Image]:
    round_code = mlb_postseason.current_round(data)
    projected = round_code is None
    if projected:
        round_code = "F"
    slots = [(league, slot) for league, slot in _list_slots(bracket, round_code) if any(slot["teams"])]
    if not slots:
        return None
    seeds = bracket.get("seeds") or {}
    row_h, status_h = _series_row_height()
    block_h = row_h + status_h
    spacing = scale_value(10)
    margin = max(2, width // 60)
    two_columns = width >= TWO_COLUMN_MIN_WIDTH and round_code != "W"
    if two_columns:
        columns = [[slot for league, slot in slots if league == "AL"], [slot for league, slot in slots if league == "NL"]]
    else:
        columns = [[slot for _, slot in slots]]
    rows = max(len(column) for column in columns)

    heading = _round_heading(data, round_code, projected)
    probe = ImageDraw.Draw(Image.new("RGB", (width, 10)))
    heading_font = _fit_font(probe, heading, _round_font(), width - 2 * margin)
    heading_h = _text_size(probe, heading, heading_font)[1] + scale_value(8)
    height = heading_h + rows * block_h + (rows - 1) * spacing
    canvas = Image.new("RGB", (width, height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(canvas)
    center_text(draw, heading, heading_font, 0, width, 0, heading_h, fill=LABEL_TEXT)

    gutter = scale_value(16) if len(columns) > 1 else 0
    col_w = (width - 2 * margin - gutter * (len(columns) - 1)) // len(columns)
    for col_index, column in enumerate(columns):
        x = margin + col_index * (col_w + gutter)
        y = heading_h
        for row_index, slot in enumerate(column):
            _draw_series_row(canvas, draw, slot, data, seeds, x, col_w, y, now)
            y += block_h
            if row_index < len(column) - 1:
                sep = y + spacing // 2
                draw.line((x + margin, sep, x + col_w - margin, sep), fill=SEPARATOR)
                y += spacing
    return canvas


# ─── Screen ───────────────────────────────────────────────────────────────────


def _header(width: int) -> Image.Image:
    title_font = _title_font()
    probe = ImageDraw.Draw(Image.new("RGB", (width, 10)))
    title_w, title_h, left, top = _text_size(probe, TITLE, title_font)
    logo_size = max(8, scale_value(26))
    logo = _logo("MLB", logo_size)
    gap = scale_value(4)
    logo_h = logo.height + gap if logo is not None else 0
    header = Image.new("RGB", (width, logo_h + title_h + scale_value(6)), BACKGROUND_COLOR)
    if logo is not None:
        header.paste(logo, ((width - logo.width) // 2, 0), logo if logo.mode == "RGBA" else None)
    ImageDraw.Draw(header).text(((width - title_w) // 2 - left, logo_h - top), TITLE, font=title_font, fill=WHITE)
    return header


def compose_playoffs_image(data: Any, *, now: Optional[datetime.datetime] = None) -> Image.Image:
    """Render the whole screen (header, bracket, series list) as one tall image."""

    now = now or datetime.datetime.now(CENTRAL_TIME)
    data = data if isinstance(data, dict) else {}
    header = _header(WIDTH)
    parts: list[Image.Image] = [header]
    has_data = bool(data.get("series") or data.get("seeds"))
    if has_data:
        bracket = mlb_postseason.build_bracket(data)
        if WIDTH >= MIN_BRACKET_WIDTH:
            bracket_img = Image.new("RGB", (WIDTH, _BracketLayout(WIDTH, HEIGHT).height), BACKGROUND_COLOR)
            _draw_bracket(bracket_img, 0, bracket)
            parts.append(bracket_img)
        series_img = _compose_series_list(WIDTH, data, bracket, now)
        if series_img is not None:
            parts.append(series_img)
    else:
        message = Image.new("RGB", (WIDTH, scale_value(40)), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(message)
        text = "No postseason data"
        center_text(draw, text, _fit_font(draw, text, _status_font(), WIDTH - 4), 0, WIDTH, 0, message.height)
        parts.append(message)

    gap = scale_value(8)
    total = sum(part.height for part in parts) + gap * (len(parts) - 1) + SCOREBOARD_STANDINGS_BOTTOM_PADDING
    image = Image.new("RGB", (WIDTH, max(HEIGHT, total)), BACKGROUND_COLOR)
    y = 0
    for part in parts:
        image.paste(part, (0, y))
        y += part.height + gap
    return image


def _scroll_display(display, full_img: Image.Image) -> None:
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


def render_mlb_playoffs(display, data: Any = None, transition: bool = False) -> ScreenImage:
    """Draw the MLB Playoffs screen.

    *data* is the ``mlb_postseason`` feed value; ``None`` fetches it here
    (standalone mode, where screens fetch what they draw).
    """

    if data is None:
        try:
            data = mlb_postseason.fetch_postseason()
        except Exception as exc:  # noqa: BLE001 - draw the empty state instead
            logging.warning("MLB postseason fetch failed: %s", exc)
            data = {}

    full_img = compose_playoffs_image(data)
    if full_img.height <= HEIGHT:
        if not transition:
            clear_display(display)
        display.image(full_img)
        time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)
    else:
        _scroll_display(display, full_img)
    return ScreenImage(full_img, displayed=True)


__all__ = ["SCREEN_ID", "compose_playoffs_image", "render_mlb_playoffs", "series_status"]
