"""Shared drawing for the playoff bracket screens (MLB, NHL and NBA).

Every league's screen has the same shape, first built for MLB Playoffs after
mlb.com/postseason: a header (league logo and title), a seven-column bracket
(one conference or league on each side, rounds running toward the final in
the middle), and under it one full-width line per series of the round being
played with both teams, the series score and the next game or the result.

A league module supplies a :class:`PlayoffScreen` (title, logos, round labels)
and turns its feed data into a bracket and a series list.  A bracket is::

    {"left": [outer, middle, inner], "right": [...], "center": slot,
     "seeds": {abbr: seed}, "mode": "pairs" | "byes",
     "left_logo": key, "right_logo": key, "left_label": text, "right_label": text}

where each column is a list of slots, and a slot is::

    {"teams": [top, bottom], "wins": [n, n], "series": dict | None, "winner": abbr | None}

``series`` carries ``live``, ``winner``, ``next_start`` (ISO time),
``next_time_tbd`` and ``next_game``.  ``mode`` "pairs" is a 16-team bracket
(four outer slots a side, two feeding each next slot); "byes" is MLB's, where
two top seeds skip the Wild Card round and each Wild Card slot feeds the
bottom row of one Division Series slot.
"""

from __future__ import annotations

import datetime
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
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
from utils import (
    ScreenImage,
    clear_display,
    clone_font,
    load_team_logo,
    scroll_vertical_content,
)

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

_LOGO_CACHE: dict[tuple[str, str, int], Optional[Image.Image]] = {}


@dataclass(frozen=True)
class PlayoffScreen:
    """What differs between the leagues' playoff screens."""

    screen_id: str
    title: str
    logo_dir: str  # team logos, by abbreviation
    header_logo: str  # key in logo_dir (or an absolute directory's key, see header_logo_dir)
    round_labels: tuple[str, str, str, str]  # outer, middle, inner, final
    header_logo_dir: Optional[str] = None
    empty_text: str = "No postseason data"


# ─── Fonts, text and logos ────────────────────────────────────────────────────


def _font(base: ImageFont.FreeTypeFont, size: float) -> ImageFont.FreeTypeFont:
    return clone_font(base, max(6, int(size)))


def _screen_font(spec: PlayoffScreen, key: str, base, size: int):
    return get_screen_font(spec.screen_id, key, base_font=base, default_size=size)


def _title_font(spec):
    return _screen_font(spec, "title", FONT_TITLE_SPORTS, 30)


def _round_font(spec):
    return _screen_font(spec, "round", FONT_STATUS, 20)


def _status_font(spec):
    return _screen_font(spec, "status", FONT_STATUS, 20)


def _score_font(spec):
    return _screen_font(spec, "score", FONT_TEAM_SPORTS, 36)


def _seed_font(spec):
    return _screen_font(spec, "seed", FONT_STATUS, 16)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int, int, int]:
    """Return ``(width, height, left, top)`` of *text*'s ink box."""

    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top, left, top


def fit_font(draw: ImageDraw.ImageDraw, text: str, font, max_width: int):
    """Shrink *font* until *text* fits in *max_width* pixels."""

    size = int(getattr(font, "size", 0) or 0)
    while size > 6 and _text_size(draw, text, font)[0] > max_width:
        size -= 1
        font = _font(font, size)
    return font


def _logo(directory: str, abbr: Optional[str], size: int) -> Optional[Image.Image]:
    if not abbr or size <= 0:
        return None
    key = (directory, abbr.upper(), size)
    if key not in _LOGO_CACHE:
        _LOGO_CACHE[key] = load_team_logo(directory, abbr.upper(), height=size, box_size=size, trim=True)
    return _LOGO_CACHE[key]


def _dimmed(logo: Image.Image) -> Image.Image:
    faded = logo.convert("RGBA")
    alpha = faded.getchannel("A").point(lambda value: int(value * LOSER_LOGO_ALPHA))
    faded.putalpha(alpha)
    return faded


def _paste_logo(canvas: Image.Image, directory: str, abbr: Optional[str], size: int, x: int, y: int,
                *, dim: bool = False) -> bool:
    logo = _logo(directory, abbr, size)
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


def parse_iso(value: Any) -> Optional[datetime.datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed


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
    start = parse_iso(series.get("next_start"))
    if start is None:
        return ""
    local = start.astimezone(CENTRAL_TIME)
    when = _day_label(local, now)
    if not series.get("next_time_tbd"):
        when = f"{when} {_format_clock(local)}"
    number = series.get("next_game")
    return f"Game {number} · {when}" if number else when


def series_status(slot: dict, names: dict, now: Optional[datetime.datetime] = None) -> tuple[str, tuple[int, int, int]]:
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
        return f"{(names or {}).get(winner) or winner} win {high}-{low}", WINNER_TEXT
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


class BracketLayout:
    """Pixel geometry of the seven-column bracket for the current display."""

    def __init__(self, width: int, height: int, mode: str = "pairs"):
        self.mode = mode
        self.margin = max(2, width // 80)
        self.col_gap = max(2, width // 64)
        self.col_w = max(10, (width - 2 * self.margin - 6 * self.col_gap) // 7)
        used = 7 * self.col_w + 6 * self.col_gap
        self.left = (width - used) // 2
        self.row_h = max(8, min(int(self.col_w * 0.45), height // 9))
        self.slot_h = 2 * self.row_h
        self.label_h = max(8, int(self.row_h * 0.8))
        self.line_w = max(1, self.row_h // 14)
        self.slot_gap = self.row_h // 2
        if mode == "byes":
            # Division Series slots sit at rows 0 and 3; the Wild Card slots
            # feed their bottom (wild-card) row, so they sit half a row lower.
            self.body_h = int(self.row_h * 5.5)
        else:
            self.body_h = 4 * self.slot_h + 3 * self.slot_gap
        self.body_top = self.label_h + max(1, self.row_h // 4)
        self.height = self.label_h + self.body_h + self.row_h // 2

    def col_x(self, index: int) -> int:
        return self.left + index * (self.col_w + self.col_gap)

    def column_tops(self, top: int) -> list[list[int]]:
        """Slot tops for the outer, middle and inner columns (and the final)."""

        body = top + self.body_top
        row_h = self.row_h
        if self.mode == "byes":
            middle = [body, body + 3 * row_h]
            outer = [y + row_h // 2 for y in middle]
            inner = [body + int(2.5 * row_h) - row_h]
            return [outer, middle, inner, inner]
        outer = [body + i * (self.slot_h + self.slot_gap) for i in range(4)]
        centers = [y + row_h for y in outer]
        middle_c = [(centers[0] + centers[1]) // 2, (centers[2] + centers[3]) // 2]
        inner_c = (middle_c[0] + middle_c[1]) // 2
        return [outer, [c - row_h for c in middle_c], [inner_c - row_h], [inner_c - row_h]]

    def feeders(self, column: int, index: int) -> tuple[int, ...]:
        """Indexes in the column before *column* that feed its slot *index*."""

        if self.mode == "byes" and column == 1:
            return (index,)
        return (2 * index, 2 * index + 1)


def _draw_slot(canvas: Image.Image, draw: ImageDraw.ImageDraw, layout: BracketLayout, spec: PlayoffScreen,
               slot: dict, seeds: dict, x: int, y: int) -> None:
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
    slot_seeds = [str(seeds.get(team) or "") for team in slot["teams"] if team]
    widest = max(slot_seeds + ["6"], key=lambda text: _text_size(draw, text, seed_font)[0])
    seed_w = _text_size(draw, widest, seed_font)[0] + pad
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
        if not _paste_logo(canvas, spec.logo_dir, team, logo_size, cursor, top + pad, dim=lost):
            _draw_text_left(draw, team, _font(FONT_STATUS, row_h * 0.45), cursor, top, row_h,
                            DIM_TEXT if lost else WHITE)
        wins = slot["wins"][index]
        if series is None or wins is None:
            continue
        fill = SCOREBOARD_IN_PROGRESS_SCORE_COLOR if live else (DIM_TEXT if lost else WHITE)
        _draw_text_right(draw, str(wins), score_font, x + col_w - pad - 1, top, row_h, fill)


def _hline(draw, layout, x0, x1, y):
    draw.line((min(x0, x1), y, max(x0, x1), y), fill=CONNECTOR, width=layout.line_w)


def _connect(draw, layout: BracketLayout, sources: list[int], target: int, src_edge: int, dst_edge: int) -> None:
    """Join feeder slot centres *sources* to the slot centred at *target*."""

    if len(sources) == 1:
        _hline(draw, layout, src_edge, dst_edge, sources[0])
        return
    elbow_x = (src_edge + dst_edge) // 2
    for y in sources:
        _hline(draw, layout, src_edge, elbow_x, y)
    draw.line((elbow_x, min(sources), elbow_x, max(sources)), fill=CONNECTOR, width=layout.line_w)
    _hline(draw, layout, elbow_x, dst_edge, target)


def _side_marker(canvas, draw, layout: BracketLayout, spec: PlayoffScreen, logo: Optional[str], label: str,
                 x: int, slot_y: int, body: int, scale: float) -> None:
    """A league or conference logo (or its name) above a slot."""

    row_h = layout.row_h
    size = min(int(row_h * scale), slot_y - body - 2)
    if size <= 4:
        return
    y = max(body, slot_y - size - int(row_h * 0.4))
    if logo and _paste_logo(canvas, spec.logo_dir, logo, size, x + (layout.col_w - size) // 2, y):
        return
    if label:
        font = fit_font(draw, label, _font(FONT_STATUS, row_h * 0.6), layout.col_w)
        center_text(draw, label, font, x, layout.col_w, y, size, fill=LABEL_TEXT)


def draw_bracket(canvas: Image.Image, top: int, bracket: dict, spec: PlayoffScreen) -> None:
    layout = BracketLayout(canvas.width, HEIGHT, bracket.get("mode", "pairs"))
    draw = ImageDraw.Draw(canvas)
    row_h = layout.row_h
    seeds = bracket.get("seeds") or {}

    label_font = _font(FONT_STATUS, layout.label_h * 0.85)
    outer, middle, inner, final = spec.round_labels
    for index, label in enumerate((outer, middle, inner, final, inner, middle, outer)):
        font = fit_font(draw, label, label_font, layout.col_w)
        center_text(draw, label, font, layout.col_x(index), layout.col_w, top, layout.label_h, fill=LABEL_TEXT)

    body = top + layout.body_top
    tops = layout.column_tops(top)
    center_x = layout.col_x(3)
    final_y = tops[3][0]
    for side, cols, inward in (("left", (0, 1, 2), 1), ("right", (6, 5, 4), -1)):
        columns = bracket.get(side) or [[], [], []]
        for depth, slots in enumerate(columns):
            x = layout.col_x(cols[depth])
            for index, slot in enumerate(slots[: len(tops[depth])]):
                _draw_slot(canvas, draw, layout, spec, slot, seeds, x, tops[depth][index])
            if depth == 0:
                continue
            prev_x = layout.col_x(cols[depth - 1])
            src_edge = prev_x + layout.col_w if inward > 0 else prev_x
            dst_edge = x if inward > 0 else x + layout.col_w
            for index, y in enumerate(tops[depth]):
                sources = [tops[depth - 1][i] + row_h for i in layout.feeders(depth, index)]
                # A single feeder lines up with the slot row it fills.
                target = sources[0] if len(sources) == 1 else y + row_h
                _connect(draw, layout, sources, target, src_edge, dst_edge)
        inner_x = layout.col_x(cols[2])
        _connect(draw, layout, [final_y + row_h], final_y + row_h,
                 inner_x + layout.col_w if inward > 0 else inner_x,
                 center_x if inward > 0 else center_x + layout.col_w)
        _side_marker(canvas, draw, layout, spec, bracket.get(f"{side}_logo"), bracket.get(f"{side}_label") or "",
                     inner_x, tops[2][0], body, 1.1)

    center = bracket.get("center")
    if center:
        _draw_slot(canvas, draw, layout, spec, center, {}, center_x, final_y)
        if center.get("winner"):
            _side_marker(canvas, draw, layout, spec, center["winner"], "", center_x, final_y, body, 1.2)


# ─── Series list ──────────────────────────────────────────────────────────────


def _series_row_height() -> tuple[int, int]:
    return scale_value(48), scale_value(26)


def draw_series_row(canvas: Image.Image, draw: ImageDraw.ImageDraw, spec: PlayoffScreen, slot: dict,
                    names: dict, seeds: dict, x: int, width: int, y: int, now: datetime.datetime) -> None:
    row_h, status_h = _series_row_height()
    logo_size = max(6, row_h - scale_value(4))
    score_font = _score_font(spec)
    seed_font = _seed_font(spec)
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
            center_text(draw, "TBD", _status_font(spec), logo_x, logo_size, y, row_h, fill=DIM_TEXT)
            continue
        if not _paste_logo(canvas, spec.logo_dir, team, logo_size, logo_x, y + (row_h - logo_size) // 2, dim=lost):
            center_text(draw, team, _status_font(spec), logo_x, logo_size, y, row_h,
                        fill=DIM_TEXT if lost else WHITE)
        seed = seeds.get(team)
        if seed:
            seed_text = f"({seed})"
            if index == 0:
                _draw_text_right(draw, seed_text, seed_font, logo_x - gap // 2, y, row_h, DIM_TEXT)
            else:
                _draw_text_left(draw, seed_text, seed_font, logo_x + logo_size + gap // 2, y, row_h, DIM_TEXT)
    if slot["teams"][0] or slot["teams"][1]:
        center_text(draw, "-", score_font, center - dash_w // 2, dash_w, y, row_h, fill=DIM_TEXT)
    text, fill = series_status(slot, names, now)
    status_font = fit_font(draw, text, _status_font(spec), width - 2 * gap)
    center_text(draw, text, status_font, x, width, y + row_h, status_h, fill=fill)


def compose_series_list(spec: PlayoffScreen, width: int, heading: str, slots: list[dict], names: dict,
                        seeds: dict, now: datetime.datetime,
                        draw_row: Optional[Callable[..., None]] = None) -> Optional[Image.Image]:
    """One full-width line per series, under a heading naming the round."""

    slots = [slot for slot in slots if any(slot["teams"])]
    if not slots:
        return None
    draw_row = draw_row or draw_series_row
    row_h, status_h = _series_row_height()
    block_h = row_h + status_h
    spacing = scale_value(10)
    margin = max(2, width // 60)
    rows = len(slots)

    probe = ImageDraw.Draw(Image.new("RGB", (width, 10)))
    heading_font = fit_font(probe, heading, _round_font(spec), width - 2 * margin)
    heading_h = _text_size(probe, heading, heading_font)[1] + scale_value(8)
    height = heading_h + rows * block_h + (rows - 1) * spacing
    canvas = Image.new("RGB", (width, height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(canvas)
    center_text(draw, heading, heading_font, 0, width, 0, heading_h, fill=LABEL_TEXT)

    x, col_w = margin, width - 2 * margin
    y = heading_h
    for index, slot in enumerate(slots):
        draw_row(canvas, draw, spec, slot, names, seeds, x, col_w, y, now)
        y += block_h
        if index < rows - 1:
            sep = y + spacing // 2
            draw.line((x + margin, sep, x + col_w - margin, sep), fill=SEPARATOR)
            y += spacing
    return canvas


# ─── Screen ───────────────────────────────────────────────────────────────────


def _header(spec: PlayoffScreen, width: int) -> Image.Image:
    title_font = _title_font(spec)
    probe = ImageDraw.Draw(Image.new("RGB", (width, 10)))
    title_w, title_h, left, top = _text_size(probe, spec.title, title_font)
    logo_size = max(8, scale_value(26))
    logo = _logo(spec.header_logo_dir or spec.logo_dir, spec.header_logo, logo_size)
    gap = scale_value(4)
    logo_h = logo.height + gap if logo is not None else 0
    header = Image.new("RGB", (width, logo_h + title_h + scale_value(6)), BACKGROUND_COLOR)
    if logo is not None:
        header.paste(logo, ((width - logo.width) // 2, 0), logo if logo.mode == "RGBA" else None)
    ImageDraw.Draw(header).text(((width - title_w) // 2 - left, logo_h - top), spec.title, font=title_font, fill=WHITE)
    return header


def compose_screen(spec: PlayoffScreen, bracket: Optional[dict], series_list: Optional[Image.Image],
                   *, include_bracket: bool = True) -> Image.Image:
    """Stack the header, bracket (when the display is wide enough) and series list.

    *include_bracket* lets a screen drop the bracket on a display where it
    would not read, whatever the width; the series list still shows.
    """

    parts: list[Image.Image] = [_header(spec, WIDTH)]
    if bracket is not None:
        if include_bracket and WIDTH >= MIN_BRACKET_WIDTH:
            layout = BracketLayout(WIDTH, HEIGHT, bracket.get("mode", "pairs"))
            bracket_img = Image.new("RGB", (WIDTH, layout.height), BACKGROUND_COLOR)
            draw_bracket(bracket_img, 0, bracket, spec)
            parts.append(bracket_img)
        if series_list is not None:
            parts.append(series_list)
    else:
        message = Image.new("RGB", (WIDTH, scale_value(40)), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(message)
        text = spec.empty_text
        center_text(draw, text, fit_font(draw, text, _status_font(spec), WIDTH - 4), 0, WIDTH, 0, message.height)
        parts.append(message)

    gap = scale_value(8)
    total = sum(part.height for part in parts) + gap * (len(parts) - 1) + SCOREBOARD_STANDINGS_BOTTOM_PADDING
    image = Image.new("RGB", (WIDTH, max(HEIGHT, total)), BACKGROUND_COLOR)
    y = 0
    for part in parts:
        image.paste(part, (0, y))
        y += part.height + gap
    return image


def show(display, full_img: Image.Image, transition: bool = False) -> ScreenImage:
    """Put the screen on *display*, scrolling when it is taller than the panel."""

    if full_img.height <= HEIGHT:
        if not transition:
            clear_display(display)
        display.image(full_img)
        time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)
    else:
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
    return ScreenImage(full_img, displayed=True)


def compose_sixteen_team_image(spec: PlayoffScreen, data: Any, round_names: dict[int, str], *,
                               left_logo: Optional[str] = None, right_logo: Optional[str] = None,
                               left_label: str = "WEST", right_label: str = "EAST",
                               now: Optional[datetime.datetime] = None) -> Image.Image:
    """The NHL/NBA screen: a 16-team bracket (West left, East right) and the current round's series."""

    from services.sports import playoff_bracket16

    now = now or datetime.datetime.now(CENTRAL_TIME)
    data = data if isinstance(data, Mapping) else {}
    if not data.get("series"):
        return compose_screen(spec, None, None)
    bracket = playoff_bracket16.build_bracket(data)
    bracket.update(left_logo=left_logo, right_logo=right_logo, left_label=left_label, right_label=right_label)
    round_number = playoff_bracket16.current_round(data)
    projected = round_number is None
    round_number = round_number or 1
    name = round_names[round_number]
    heading = f"Projected {name}" if projected else f"{name} · Best of {playoff_bracket16.BEST_OF}"
    series_list = compose_series_list(
        spec, WIDTH, heading, playoff_bracket16.round_slots(bracket, round_number),
        dict(data.get("names") or {}), bracket["seeds"], now,
    )
    return compose_screen(spec, bracket, series_list)


def images_dir(league: str) -> str:
    return os.path.join(IMAGES_DIR, league)


__all__ = [
    "BracketLayout",
    "PlayoffScreen",
    "compose_screen",
    "compose_series_list",
    "compose_sixteen_team_image",
    "draw_bracket",
    "draw_series_row",
    "fit_font",
    "images_dir",
    "parse_iso",
    "series_status",
    "show",
]
