#!/usr/bin/env python3
"""
draw_date_time.py

Two screens; both show date AND time:
  • Screen A (draw_date):    DATE on top half,  TIME on bottom half
  • Screen B (draw_time):    TIME on top half,  DATE on bottom half

- Bright, readable random colors (no dark combos)
- GitHub update indicator is a tiny GitHub PNG at bottom-right
- No "flash": when called with transition=True (as main.py does),
  we render a static image; any optional color cycling only runs
  when transition=False (i.e., if you ever direct-render these).

Options:
- GH_ICON_INVERT: invert gh.png colors (useful if your icon is dark).
- GH_ICON_SIZE:   height of the GitHub icon in pixels.
"""

import threading
import time
import datetime
from typing import Tuple, Literal, Callable

import logging

from PIL import Image, ImageDraw, ImageFont

from config import (
    WIDTH,
    HEIGHT,
    FONT_DAY_DATE,
    FONT_DATE,
    FONT_TIME,
    FONT_AM_PM,
    DATE_TIME_GH_ICON_INVERT,
    DATE_TIME_GH_ICON_SIZE,
    DATE_TIME_GH_ICON_PATHS,
    is_hyperpixel_next_layout,
    is_hyperpixel_4_square_layout,
    is_kernel_driven_display,
    get_display_profile_id,
    get_screen_background_color,
)
from utils import (
    ScreenImage,
    bright_color,
    check_apt_updates,
    check_github_updates,
    clear_display,
    date_strings,
    get_update_status,
    load_github_icon,
    measure_text,
    time_strings,
)


def _color_cycle_profile(
    *,
    kernel_driven: bool,
    display_profile_id: str,
    hyperpixel_layout: bool,
    hyperpixel_square: bool,
) -> tuple[float, float, int | None]:
    """Return cycle timing: initial delay, frame interval, and max steps."""

    # Date/time screens should start cycling immediately across all profiles.
    # Keeping the cadence fast preserves the intended "rapid" effect.
    _ = display_profile_id
    initial_delay = 0.0
    interval = 0.08
    infinite_cycle = kernel_driven or hyperpixel_layout or hyperpixel_square
    # Non-kernel/non-HyperPixel profiles still cycle for roughly one rotation
    # window (default SCREEN_DELAY is 4s), then naturally stop.
    steps = None if infinite_cycle else 60
    return initial_delay, interval, steps

# -----------------------------------------------------------------------------
# Layout helpers

def _compose_frame(
    order: Literal["date_time", "time_date"],
    col_top: Tuple[int,int,int],
    col_bottom: Tuple[int,int,int],
    gh_on: bool,
    screen_id: str,
) -> Image.Image:
    """
    Build a single static frame with the requested order.
    Top block and bottom block are vertically centered within their halves.
    """
    background = get_screen_background_color(screen_id, (0, 0, 0))
    img  = Image.new("RGB", (WIDTH, HEIGHT), background)
    draw = ImageDraw.Draw(img)

    now = datetime.datetime.now()
    weekday, date_str = date_strings(now)
    time_str, ampm = time_strings(now)

    # Top/bottom halves
    top_box    = (0, 0, WIDTH, HEIGHT//2)
    bottom_box = (0, HEIGHT//2, WIDTH, HEIGHT)

    # ----- Build content per half
    def draw_date_block(box, color):
        x0, y0, x1, y1 = box
        area_w = x1 - x0
        area_h = y1 - y0

        # Wednesday can be long—shrink weekday by 2pt if available
        try:
            day_font = ImageFont.truetype(FONT_DAY_DATE.path, max(8, FONT_DAY_DATE.size - (2 if weekday=="Wednesday" else 0)))
        except Exception:
            day_font = FONT_DAY_DATE

        w1, h1 = measure_text(draw, weekday, day_font)
        w2, h2 = measure_text(draw, date_str, FONT_DATE)
        gap = 2
        block_h = h1 + gap + h2
        y_start = y0 + (area_h - block_h)//2

        draw.text((x0 + (area_w - w1)//2, y_start),
                  weekday, font=day_font, fill=color)
        draw.text((x0 + (area_w - w2)//2, y_start + h1 + gap),
                  date_str, font=FONT_DATE, fill=color)

    def draw_time_block(box, color):
        x0, y0, x1, y1 = box
        area_w = x1 - x0
        area_h = y1 - y0

        w_t, h_t = measure_text(draw, time_str, FONT_TIME)
        w_a, h_a = measure_text(draw, ampm,     FONT_AM_PM)

        total_w = w_t + w_a
        max_h   = max(h_t, h_a)
        x_start = x0 + (area_w - total_w)//2
        y_start = y0 + (area_h - max_h)//2

        draw.text((x_start, y_start), time_str, font=FONT_TIME,   fill=color)
        draw.text((x_start + w_t, y_start + (h_t - h_a)//2),
                  ampm, font=FONT_AM_PM, fill=color)

    if order == "date_time":
        draw_date_block(top_box,    col_top)
        draw_time_block(bottom_box, col_bottom)
    else:
        draw_time_block(top_box,    col_top)
        draw_date_block(bottom_box, col_bottom)

    # GitHub update indicator (tiny GitHub logo, bottom-right)
    if gh_on:
        ic = load_github_icon(
            size=DATE_TIME_GH_ICON_SIZE,
            invert=DATE_TIME_GH_ICON_INVERT,
            paths=DATE_TIME_GH_ICON_PATHS,
        )
        if ic:
            x_pos = WIDTH - ic.width - 2
            y_pos = HEIGHT - ic.height - 2 + 4
            y_pos = min(HEIGHT - ic.height, y_pos)
            y_pos = max(0, y_pos)
            img.paste(ic, (x_pos, y_pos), ic)

    return img

def _cycle_colors_after_load(
    display,
    base_order: Literal["date_time", "time_date"],
    gh_state: Callable[[], bool],
    screen_id: str,
    frame_state: dict | None = None,
):
    """
    Optional subtle color-cycle that runs AFTER the first full static frame is already shown.
    Only used when transition=False (direct rendering).
    """
    # small delay so the initial frame is already visible
    hyperpixel_layout = is_hyperpixel_next_layout()
    hyperpixel_square = is_hyperpixel_4_square_layout()
    kernel_driven = is_kernel_driven_display()
    display_profile_id = get_display_profile_id()
    initial_delay, color_cycle_interval, steps = _color_cycle_profile(
        kernel_driven=kernel_driven,
        display_profile_id=display_profile_id,
        hyperpixel_layout=hyperpixel_layout,
        hyperpixel_square=hyperpixel_square,
    )
    # HyperPixel frame ids can drift due driver-side refresh behavior. When
    # that happens we should keep animating instead of treating those bumps as
    # a screen takeover.
    enforce_takeover_tracking = not (hyperpixel_layout or hyperpixel_square)
    time.sleep(initial_delay)
    expected_frame_id = display.frame_id() if hasattr(display, "frame_id") else None
    if frame_state is not None:
        with frame_state["lock"]:
            expected_frame_id = frame_state["value"]
    if expected_frame_id is not None and hasattr(display, "frame_id"):
        current_frame_id = display.frame_id()
        if current_frame_id > expected_frame_id:
            # A sibling writer (typically update-check refresh) can advance the
            # display before its shared frame_state commit is visible.
            # Reconcile once so that startup races do not cancel animation.
            if frame_state is not None:
                with frame_state["lock"]:
                    expected_frame_id = max(frame_state["value"], current_frame_id)
            else:
                expected_frame_id = current_frame_id
    # Keep cycling while this screen remains active, but stop immediately once
    # another screen has updated the display frame buffer.
    count = 0
    takeover_observations = 0
    while steps is None or count < steps:
        if (
            enforce_takeover_tracking
            and
            expected_frame_id is not None
            and hasattr(display, "frame_id")
        ):
            current_frame_id = display.frame_id()
            if frame_state is not None:
                with frame_state["lock"]:
                    expected_frame_id = max(expected_frame_id, frame_state["value"])
            # Another renderer taking over will advance the frame id beyond
            # what this screen last wrote. Ignore transient races where the
            # shared expected id has already advanced but display.frame_id()
            # has not yet caught up.
            if current_frame_id > expected_frame_id:
                if frame_state is not None:
                    # The update-check worker can advance display.frame_id()
                    # shortly before its shared frame_state write lands.
                    # Reconcile once more before treating this as takeover.
                    with frame_state["lock"]:
                        expected_frame_id = max(expected_frame_id, frame_state["value"])
                    if current_frame_id <= expected_frame_id:
                        takeover_observations = 0
                        continue
                takeover_observations += 1
                # Treat a single leading frame read as transient. We only stop
                # once takeover is observed on consecutive checks.
                if takeover_observations >= 2:
                    break
                continue
            takeover_observations = 0
        img = _compose_frame(base_order, bright_color(), bright_color(), gh_state(), screen_id)
        display.image(img)
        try:
            # Some display drivers (notably HyperPixel/HDMI paths) only flush
            # framebuffer updates when show() is called.
            display.show()
        except AttributeError:
            # Drivers that auto-refresh on image() do not expose show().
            pass
        if hasattr(display, "frame_id"):
            latest_frame_id = display.frame_id()
            # Track the frame id that *this* screen last rendered so we only stop
            # when another screen takes over.
            if frame_state is not None:
                with frame_state["lock"]:
                    frame_state["value"] = latest_frame_id
                    expected_frame_id = latest_frame_id
            else:
                expected_frame_id = latest_frame_id
        time.sleep(color_cycle_interval)
        count += 1


def _start_update_checks(
    order: Literal["date_time", "time_date"],
    colors: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
    gh_state: dict,
    display,
    screen_id: str,
    expected_frame_id: int | None = None,
    frame_state: dict | None = None,
):
    """Kick off apt/GitHub checks in the background, updating the screen when ready."""

    def _worker():
        try:
            worker_expected_frame_id = expected_frame_id
            gh_on = check_github_updates()
            gh_state["value"] = gh_on
            check_apt_updates()

            if display is None:
                return
            if worker_expected_frame_id is not None:
                current_frame_id = display.frame_id()
                if frame_state is not None:
                    with frame_state["lock"]:
                        worker_expected_frame_id = frame_state["value"]
                if current_frame_id > worker_expected_frame_id:
                    logging.info(
                        "Background update checks skipped; display already updated."
                    )
                    return

            refreshed = _compose_frame(order, colors[0], colors[1], gh_state["value"], screen_id)
            display.image(refreshed)
            if frame_state is not None and hasattr(display, "frame_id"):
                with frame_state["lock"]:
                    frame_state["value"] = display.frame_id()
            try:
                display.show()
            except AttributeError:
                pass
        except Exception:
            logging.exception("Background update checks failed")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

# -----------------------------------------------------------------------------
# Public API

def draw_date(display, transition: bool=False):
    """
    Screen A: DATE on top, TIME on bottom.
    When transition=True (used by main.py), returns a single static frame
    to avoid any initial flash. No cycling occurs in transition mode.
    When transition=False, we show the first frame immediately and keep the
    colors stable for the rest of the screen's dwell time.
    """
    col_top    = bright_color()
    col_bottom = bright_color()
    gh_state   = {"value": get_update_status().github}

    img = _compose_frame("date_time", col_top, col_bottom, gh_state["value"], "date")

    if transition:
        _start_update_checks("date_time", (col_top, col_bottom), gh_state, None, "date")
        return img

    clear_display(display)
    display.image(img)
    try:
        display.show()
    except AttributeError:
        # Some display drivers immediately refresh when image() is called.
        pass
    frame_id = display.frame_id()
    frame_state = {"value": frame_id, "lock": threading.Lock()}
    _start_update_checks(
        "date_time",
        (col_top, col_bottom),
        gh_state,
        display,
        "date",
        expected_frame_id=frame_id,
        frame_state=frame_state,
    )
    if is_hyperpixel_next_layout() or is_hyperpixel_4_square_layout():
        threading.Thread(
            target=_cycle_colors_after_load,
            args=(display, "date_time", lambda: gh_state["value"], "date", frame_state),
            daemon=True,
        ).start()
    return ScreenImage(img, displayed=True)


def draw_time(display, transition: bool=False):
    """
    Screen B: TIME on top, DATE on bottom.
    Same transition policy as draw_date().
    """
    col_top    = bright_color()
    col_bottom = bright_color()
    gh_state   = {"value": get_update_status().github}

    img = _compose_frame("time_date", col_top, col_bottom, gh_state["value"], "time")

    if transition:
        _start_update_checks("time_date", (col_top, col_bottom), gh_state, None, "time")
        return img

    clear_display(display)
    display.image(img)
    try:
        display.show()
    except AttributeError:
        pass
    frame_id = display.frame_id()
    frame_state = {"value": frame_id, "lock": threading.Lock()}
    _start_update_checks(
        "time_date",
        (col_top, col_bottom),
        gh_state,
        display,
        "time",
        expected_frame_id=frame_id,
        frame_state=frame_state,
    )
    return ScreenImage(img, displayed=True)
