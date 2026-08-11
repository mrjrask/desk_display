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

import contextlib
import datetime
import logging
import threading
import time
from collections.abc import Callable
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

_IP_OVERLAY_FONT: ImageFont.ImageFont | None = None
_IP_OVERLAY_BOTTOM_PADDING = 6

from config import (
    DATE_TIME_GH_ICON_INVERT,
    DATE_TIME_GH_ICON_PATHS,
    DATE_TIME_GH_ICON_SIZE,
    FONT_AM_PM,
    FONT_DATE,
    FONT_DAY_DATE,
    FONT_TIME,
    HEIGHT,
    IP_WITH_TIME,
    SCREEN_DELAY,
    WIDTH,
    get_display_profile_id,
    get_screen_background_color,
    is_hyperpixel_4_square_layout,
    is_hyperpixel_next_layout,
    is_kernel_driven_display,
)
from services.wifi_utils import get_assigned_ipv4
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

    initial_delay = 0.0
    # Kernel-driven displays (HyperPixel/HDMI) can visibly tear/flicker if we
    # push updates too aggressively. Use a calmer cadence that still animates.
    if kernel_driven or hyperpixel_layout or hyperpixel_square:
        interval = 0.20
        cycle_window_seconds = max(0.5, float(SCREEN_DELAY) - 0.2)
    elif display_profile_id == "display_hat_mini":
        # Display HAT Mini can overrun into subsequent screens if this worker
        # animates through most of the dwell window. Keep animation subtle and
        # short so color changes stay scoped to date/time only.
        interval = 0.12
        cycle_window_seconds = min(max(0.4, float(SCREEN_DELAY) * 0.35), 1.2)
    else:
        interval = 0.08
        # Always cap cycling to this screen's dwell window so the worker cannot
        # continue writing into later screens.
        cycle_window_seconds = max(0.5, float(SCREEN_DELAY) - 0.2)

    steps = max(1, int(cycle_window_seconds / interval))
    return initial_delay, interval, steps

def _ip_overlay_font() -> ImageFont.ImageFont:
    """Return a compact, legible font for the assigned IP overlay text."""

    global _IP_OVERLAY_FONT
    if _IP_OVERLAY_FONT is not None:
        return _IP_OVERLAY_FONT

    target_size = max(8, int(round(FONT_AM_PM.size * 0.45)))
    try:
        _IP_OVERLAY_FONT = ImageFont.truetype(FONT_AM_PM.path, target_size)
    except Exception:
        _IP_OVERLAY_FONT = FONT_AM_PM

    return _IP_OVERLAY_FONT

def _assigned_ip_overlay_text() -> str:
    """Return the bottom-left IP overlay label for the date/time screen."""

    ip_address = get_assigned_ipv4()
    if ip_address:
        return f"IP: {ip_address}"
    return "IP: --"


# -----------------------------------------------------------------------------
# Layout helpers

def _compose_frame(
    order: Literal["date_time", "time_date"],
    col_top: tuple[int,int,int],
    col_bottom: tuple[int,int,int],
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

    # Assigned IPv4 indicator (bottom-left)
    if IP_WITH_TIME:
        ip_text = _assigned_ip_overlay_text()
        ip_font = _ip_overlay_font()
        left, top, right, bottom = draw.textbbox((0, 0), ip_text, font=ip_font)
        ip_x = 2 - left
        ip_y = max(0, HEIGHT - _IP_OVERLAY_BOTTOM_PADDING - bottom)
        draw.text((ip_x, ip_y), ip_text, font=ip_font, fill=(200, 200, 200))

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
    # Kernel-driven outputs (HyperPixel/HDMI framebuffers) can report frame-id
    # drift while this same worker is active. That makes "takeover" detection
    # look like another screen has rendered even when it has not, which can
    # stop color cycling immediately. For those displays, rely on the bounded
    # step count instead of frame-id takeover checks.
    enforce_takeover_tracking = not (
        kernel_driven or hyperpixel_layout or hyperpixel_square
    )
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
    while count < steps:
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
        # Some display drivers (notably HyperPixel/HDMI paths) only flush
        # framebuffer updates when show() is called. Drivers that auto-refresh
        # on image() do not expose show().
        with contextlib.suppress(AttributeError):
            display.show()
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
    colors: tuple[tuple[int, int, int], tuple[int, int, int]],
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
            with contextlib.suppress(AttributeError):
                display.show()
        except Exception:
            logging.exception("Background update checks failed")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def _start_color_cycle(
    display,
    order: Literal["date_time", "time_date"],
    gh_state: dict,
    screen_id: str,
    frame_state: dict,
):
    """Run the date/time color animation in a background worker."""

    def _worker():
        try:
            _cycle_colors_after_load(
                display,
                order,
                lambda: gh_state["value"],
                screen_id,
                frame_state,
            )
        except Exception:
            logging.exception("Date/time color cycle failed")

    threading.Thread(target=_worker, daemon=True).start()

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
    # Some display drivers immediately refresh when image() is called.
    with contextlib.suppress(AttributeError):
        display.show()
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
    _start_color_cycle(display, "date_time", gh_state, "date", frame_state)
    return ScreenImage(img, displayed=True)

