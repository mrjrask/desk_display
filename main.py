#!/usr/bin/env python3
"""
Main display loop driving the Pimoroni Display HAT Mini LCD,
with optional screenshot capture, H.264 MP4 video capture, Wi-Fi triage,
screen-config sequencing, and batch screenshot archiving.

Changes:
- Stop pruning single files; instead, when screenshots/ has >= ARCHIVE_THRESHOLD
  images, archive the whole set into screenshot_archive/<screen>/.
- Avoid creating empty archive folders.
- Guard logo screens when the image file is missing.
- Sort archived screenshots inside screenshot_archive/<screen>/ so they mirror
  the live screenshots/ folder structure.
"""
import warnings
from gpiozero.exc import PinFactoryFallback, NativePinFactoryFallback

warnings.filterwarnings("ignore", category=PinFactoryFallback)
warnings.filterwarnings("ignore", category=NativePinFactoryFallback)
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API",
    category=UserWarning,
    module=r"pygame\.pkgdata",
)

import glob
import os
import time
import logging
import threading
import datetime
import json
import hashlib
import signal
import shutil
import subprocess
import sys
from collections import deque
from contextlib import nullcontext
from typing import Callable, Dict, List, Optional, Set, Tuple
try:
    import pygame
except Exception:
    pygame = None

os.environ.setdefault("CONFIG_LOAD_DOTENV", "1")

gc = __import__('gc')

from PIL import Image

from config import (
    WIDTH,
    HEIGHT,
    SCREEN_DELAY,
    SCHEDULE_UPDATE_INTERVAL,
    FONT_DATE_SPORTS,
    ENABLE_SCREENSHOTS,
    ENABLE_VIDEO,
    VIDEO_FPS,
    ENABLE_WEATHER,
    ENABLE_WIFI_MONITOR,
    CENTRAL_TIME,
    TRAVEL_ACTIVE_WINDOW,
    DARK_HOURS_ENABLED,
    is_within_dark_hours,
    AHL_TEAM_TRICODE,
    ENABLE_WIFI_RECOVERY,
    WEATHER_REFRESH_SECONDS,
    initialise_runtime_probes,
)
from utils import (
    Display,
    ScreenImage,
    animate_fade_in,
    clear_display,
    clear_update_indicator,
    defer_clear_display,
    display_updates_enabled,
    resume_display_updates,
    suspend_display_updates,
    temporary_display_led,
)
import data_fetch
from services.data_provider import provider as data_provider
try:
    from services import wifi_utils as _wifi_utils
    wifi_utils = _wifi_utils
except Exception as exc:
    logging.getLogger(__name__).warning(
        "Wi-Fi utilities unavailable; Wi-Fi monitoring disabled: %s", exc
    )

    class _WifiUtilsFallback:
        @staticmethod
        def start_monitor(*args, **kwargs):
            return None

        @staticmethod
        def stop_monitor():
            return None

        @staticmethod
        def get_wifi_state():
            return "ok", None

    wifi_utils = _WifiUtilsFallback()
from paths import resolve_screens_config_paths, resolve_storage_paths

from screens.registry import ScreenContext, ScreenDefinition, build_screen_registry
from schedule import ScreenScheduler, build_scheduler, load_schedule_config, sanitize_schedule_config

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Config path precedence/fallback rules are centralized in paths.py.
_screens_config_paths = resolve_screens_config_paths()
DEFAULT_CONFIG_PATH = str(_screens_config_paths.default_path)
LOCAL_CONFIG_PATH = str(_screens_config_paths.local_override_path)
CONFIG_PATH = str(_screens_config_paths.active_path)


def _active_config_path() -> str:
    """Return the current schedule config path, preferring local overrides."""
    return str(resolve_screens_config_paths().active_path)

# ─── Screenshot archiving (batch) ────────────────────────────────────────────
ARCHIVE_THRESHOLD = 500  # archive when we reach this many images
ARCHIVE_DEFAULT_FOLDER = "Screens"
ALLOWED_SCREEN_EXTS = (".png", ".jpg", ".jpeg")  # images only
MAX_SCREENSHOTS_PER_SCREEN = 5
MAX_ARCHIVED_SCREENSHOTS_PER_SCREEN = 50

_storage_paths = None
SCREENSHOT_DIR = ""
CURRENT_SCREENSHOT_DIR = ""
SCREENSHOT_ARCHIVE_BASE = ""
SCREENSHOT_ARCHIVE_MIRROR = ""
DISPLAY_STATUS_PATH = ""

_screen_config_mtime: Optional[float] = None
screen_scheduler: Optional[ScreenScheduler] = None
_requested_screen_ids: Set[str] = set()
_registry_cache_key: Optional[
    Tuple[
        Optional[float],
        bool,
        bool,
        Tuple[int, int],
        Optional[datetime.datetime],
        int,
        int,
    ]
] = None
_registry_cache_value: Optional[Tuple[Dict[str, ScreenDefinition], Dict[str, object]]] = None
_registry_cache_nonce = 0

_skip_request_pending = False
_last_screen_id: Optional[str] = None

_SKIP_BUTTON_SCREEN_IDS = {"date"}

_shutdown_event = threading.Event()
_shutdown_complete = threading.Event()
_display_cleared = threading.Event()

BUTTON_POLL_INTERVAL = 0.1
_BUTTON_NAMES = ("A", "B", "X", "Y")
_BUTTON_STATE = {name: False for name in _BUTTON_NAMES}
_BUTTON_PRESS_STARTED_AT = {name: 0.0 for name in _BUTTON_NAMES}
_BUTTON_PRESS_HANDLED = {name: False for name in _BUTTON_NAMES}
_BUTTON_MIN_HOLD_SECONDS = {"B": 0.6}
_BUTTON_NOISE_WINDOW_SECONDS = 15.0
_BUTTON_NOISE_WARNING_THRESHOLD = 5
_BUTTON_NOISE_TIMESTAMPS = deque(maxlen=32)
_manual_skip_event = threading.Event()
_button_monitor_thread: Optional[threading.Thread] = None
_pending_previous_screen_id: Optional[str] = None
_SCREEN_HISTORY_LIMIT = 50
_screen_history = []
_screen_history_lock = threading.Lock()

_dark_hours_active = False
_manual_display_off = False
_manual_backlight_level: Optional[float] = None
_wifi_outage_active = False
_wifi_outage_started_at: Optional[datetime.datetime] = None
_wifi_outage_live_games = False
_wifi_monitor_enabled = ENABLE_WIFI_MONITOR

GC_COLLECT_INTERVAL = max(5.0, float(os.environ.get("DESK_DISPLAY_GC_INTERVAL_SECONDS", "30")))
_last_gc_collect_monotonic = 0.0
_TOUCH_DOUBLE_TAP_MAX_INTERVAL_SECONDS = max(
    0.1, float(os.environ.get("TOUCH_DOUBLE_TAP_MAX_INTERVAL_SECONDS", "0.45"))
)
_last_touch_tap_monotonic = 0.0


def _run_gc_maintenance(*, force: bool = False) -> bool:
    """Run `gc.collect()` only when needed to reduce per-frame pause time."""

    global _last_gc_collect_monotonic

    now = time.monotonic()
    if not force and (now - _last_gc_collect_monotonic) < GC_COLLECT_INTERVAL:
        return False

    gc.collect()
    _last_gc_collect_monotonic = now
    return True


def _frame_id_changed(display: object, before: Optional[int]) -> bool:
    """Return whether the display frame counter advanced after rendering."""

    if before is None or not hasattr(display, "frame_id"):
        return True

    try:
        after = display.frame_id()
    except Exception:
        return True

    return after != before


def _request_next_screen() -> bool:
    """Request that the scheduler advance to the next eligible screen."""

    global _skip_request_pending

    logging.info("⏭️  Skip requested – advancing to next screen.")
    _skip_request_pending = True
    _manual_skip_event.set()
    return True


def _request_previous_screen() -> bool:
    """Request that the scheduler return to the previously shown screen."""

    global _pending_previous_screen_id

    with _screen_history_lock:
        previous_id = _screen_history[-2] if len(_screen_history) >= 2 else None

    if not previous_id:
        logging.info("⏮️  Previous screen requested, but no history is available.")
        return False

    logging.info("⏮️  Returning to previous screen '%s'.", previous_id)
    _pending_previous_screen_id = previous_id
    _manual_skip_event.set()
    return True


def _check_touch_skip_request() -> bool:
    """Handle touchscreen double-tap gestures that request a screen skip."""

    global _last_touch_tap_monotonic

    if display is None or pygame is None:
        return False

    if _shutdown_event.is_set():
        return False

    fingerdown = getattr(pygame, "FINGERDOWN", None)
    mousebuttondown = getattr(pygame, "MOUSEBUTTONDOWN", None)
    if fingerdown is None and mousebuttondown is None:
        return False

    event_types = [event_type for event_type in (fingerdown, mousebuttondown) if event_type is not None]
    try:
        events = pygame.event.get(event_types)
    except Exception:
        return False

    if not events:
        return False

    finger_taps: list[float] = []
    mouse_taps: list[float] = []
    right_third_start = (2.0 * float(getattr(display, "width", WIDTH))) / 3.0

    for event in events:
        event_type = getattr(event, "type", None)
        x_pos = None

        if fingerdown is not None and event_type == fingerdown:
            x_pos = float(getattr(event, "x", 0.0)) * float(getattr(display, "width", WIDTH))
            if x_pos >= right_third_start:
                finger_taps.append(time.monotonic())
            continue

        if mousebuttondown is not None and event_type == mousebuttondown:
            button = getattr(event, "button", 1)
            if button not in (1,):
                continue
            pos = getattr(event, "pos", None)
            if isinstance(pos, tuple) and len(pos) >= 1:
                x_pos = float(pos[0])
            if x_pos is not None and x_pos >= right_third_start:
                mouse_taps.append(time.monotonic())

    tap_times = finger_taps if finger_taps else mouse_taps
    if not tap_times:
        return False

    for tap_time in tap_times:
        if (
            _last_touch_tap_monotonic > 0
            and (tap_time - _last_touch_tap_monotonic) <= _TOUCH_DOUBLE_TAP_MAX_INTERVAL_SECONDS
        ):
            _last_touch_tap_monotonic = 0.0
            logging.info("👆 Double tap detected on right-third touch zone; advancing screen.")
            return _request_next_screen()

        _last_touch_tap_monotonic = tap_time

    return False


def _handle_button_down(name: str) -> bool:
    """React to a newly pressed control button."""

    name = name.upper()
    if display is None:
        return False
    if name == "X":
        logging.info("⬇️  X button pressed – pulling latest desk_display changes and restarting service…")
        _git_pull_and_restart_desk_display_service()
        return False
    if name == "A":
        return _request_next_screen()
    if name == "B":
        return _toggle_display_updates()
    if name == "Y":
        logging.info("🔁 Y button pressed – restarting desk_display service…")
        _restart_desk_display_service()
        return False
    return False


def _toggle_display_updates() -> bool:
    """Toggle the display on/off without stopping the main loop."""

    global _manual_display_off, _manual_backlight_level

    if display is None:
        return False

    if _manual_display_off:
        _manual_display_off = False
        logging.info("🔆 Display toggled on.")
        if _manual_backlight_level is not None and hasattr(display, "set_backlight"):
            try:
                display.set_backlight(_manual_backlight_level)
            except Exception:
                pass
        _manual_backlight_level = None
        if not _dark_hours_active:
            resume_display_updates()
        return True

    _manual_display_off = True
    logging.info("🌑 Display toggled off.")
    if hasattr(display, "backlight_level") and hasattr(display, "set_backlight"):
        try:
            _manual_backlight_level = display.backlight_level()
            display.set_backlight(0.0)
        except Exception:
            _manual_backlight_level = None
    try:
        resume_display_updates()
        clear_display(display)
        display.show()
    except Exception:
        pass
    suspend_display_updates()
    return True


def _cache_has_live_games(data_cache: Dict[str, object]) -> bool:
    for team in ("hawks", "wolves", "bulls", "cubs", "sox"):
        entry = data_cache.get(team)
        if isinstance(entry, dict) and entry.get("live"):
            return True
    return False


def _update_wifi_outage_state(wifi_state: str) -> None:
    global _wifi_outage_active, _wifi_outage_started_at, _wifi_outage_live_games

    now = datetime.datetime.now(datetime.timezone.utc)

    if wifi_state != "ok":
        if not _wifi_outage_active:
            _wifi_outage_active = True
            _wifi_outage_started_at = now
            _wifi_outage_live_games = _cache_has_live_games(cache)
            logging.warning(
                "⚠️  Wi-Fi outage detected; caching enabled (%s).",
                "live games active" if _wifi_outage_live_games else "no live games",
            )
        return

    if _wifi_outage_active:
        logging.info("✅ Wi-Fi restored; resuming live refreshes.")
        _wifi_outage_active = False
        _wifi_outage_started_at = None
        _wifi_outage_live_games = False


def _button_event_callback(name: str) -> None:
    """Hardware callback fired when a control button is pressed."""

    upper = name.upper()
    if upper not in _BUTTON_STATE:
        return

    if _BUTTON_STATE[upper]:
        return

    _BUTTON_STATE[upper] = True
    _BUTTON_PRESS_STARTED_AT[upper] = time.monotonic()
    _BUTTON_PRESS_HANDLED[upper] = False


def _button_press_can_fire(name: str, now: float) -> bool:
    """Return whether a currently pressed button is eligible to trigger."""

    if _BUTTON_PRESS_HANDLED[name]:
        return False

    hold_seconds = _BUTTON_MIN_HOLD_SECONDS.get(name, 0.0)
    if hold_seconds <= 0:
        return True

    pressed_for = now - _BUTTON_PRESS_STARTED_AT[name]
    return pressed_for >= hold_seconds


def _load_scheduler_from_config() -> Optional[ScreenScheduler]:
    config_path = _active_config_path()

    try:
        config_data = load_schedule_config(config_path)
    except Exception as exc:
        logging.warning(f"Could not load schedule configuration: {exc}")
        return None

    sanitized_config, removed_ids = sanitize_schedule_config(config_data)
    if removed_ids:
        try:
            with open(config_path, "w", encoding="utf-8") as fh:
                json.dump(sanitized_config, fh, indent=2)
                fh.write("\n")
            logging.info(
                "Removed %d deprecated/unknown screen id(s) from schedule configuration.",
                len(removed_ids),
            )
            config_data = sanitized_config
        except OSError as exc:
            logging.warning("Could not persist cleaned schedule configuration: %s", exc)

    try:
        scheduler = build_scheduler(config_data)
    except ValueError as exc:
        logging.error(f"Invalid schedule configuration: {exc}")
        return None

    return scheduler


def refresh_schedule_if_needed(force: bool = False) -> None:
    global _screen_config_mtime, screen_scheduler, _requested_screen_ids
    global _registry_cache_key, _registry_cache_value
    global _last_screen_id, _skip_request_pending, _pending_previous_screen_id

    config_path = _active_config_path()

    try:
        mtime = os.path.getmtime(config_path)
    except OSError:
        mtime = None

    if not force and mtime == _screen_config_mtime and screen_scheduler is not None:
        return

    scheduler = _load_scheduler_from_config()
    if scheduler is None:
        return

    screen_scheduler = scheduler
    _requested_screen_ids = scheduler.requested_ids
    _screen_config_mtime = mtime
    _last_screen_id = None
    _skip_request_pending = False
    _pending_previous_screen_id = None
    _registry_cache_key = None
    _registry_cache_value = None
    with _screen_history_lock:
        _screen_history.clear()
    logging.info("🔁 Loaded schedule configuration with %d node(s).", scheduler.node_count)


def _registry_cache_inputs(
    offline: bool,
    skip_scoreboards: bool,
    weather_fetched_at: Optional[datetime.datetime],
    weather_object_id: int,
) -> Tuple[
    Optional[float],
    bool,
    bool,
    Tuple[int, int],
    Optional[datetime.datetime],
    int,
    int,
]:
    """Return the cache key that determines whether registry rebuild is required."""

    display_mode = (int(WIDTH), int(HEIGHT))
    return (
        _screen_config_mtime,
        bool(offline),
        bool(skip_scoreboards),
        display_mode,
        weather_fetched_at,
        weather_object_id,
        _registry_cache_nonce,
    )


def _build_registry_if_needed(context: ScreenContext) -> Tuple[Dict[str, ScreenDefinition], Dict[str, object]]:
    """Build and cache the screen registry when runtime inputs change."""

    global _registry_cache_key, _registry_cache_value

    cache_key = _registry_cache_inputs(
        context.offline,
        context.skip_scoreboards,
        context.weather_fetched_at,
        id(context.cache.get("weather")),
    )
    if _registry_cache_value is not None and _registry_cache_key == cache_key:
        return _registry_cache_value

    registry, metadata = build_screen_registry(context)
    _registry_cache_key = cache_key
    _registry_cache_value = (registry, metadata)
    return registry, metadata


def _bump_registry_cache_nonce() -> None:
    """Invalidate cached screen registry after data updates."""

    global _registry_cache_nonce
    _registry_cache_nonce += 1


display: Optional[Display] = None
_background_refresh_thread: Optional[threading.Thread] = None
_startup_refresh_thread: Optional[threading.Thread] = None
_config_ui_process: Optional[subprocess.Popen] = None
_runtime_initialized = False


def _clear_display_immediately(reason: Optional[str] = None) -> None:
    """Clear the LCD as soon as a shutdown is requested."""

    already_cleared = _display_cleared.is_set()

    if display is None:
        _display_cleared.set()
        return

    if reason and not already_cleared:
        logging.info("🧹 Clearing display (%s)…", reason)

    try:
        resume_display_updates()
        clear_display(display)
        try:
            display.show()
        except Exception:
            pass
    except Exception:
        pass
    finally:
        _display_cleared.set()
        suspend_display_updates()


def request_shutdown(reason: str) -> None:
    """Signal the main loop to exit and blank the screen immediately."""

    if _shutdown_event.is_set():
        _clear_display_immediately(reason)
        return

    logging.info("✋ Shutdown requested (%s).", reason)
    _shutdown_event.set()
    _clear_display_immediately(reason)


def _restart_desk_display_service() -> None:
    """Restart the desk_display systemd service."""

    request_shutdown("service restart")
    try:
        result = subprocess.run(
            ["sudo", "systemctl", "--no-block", "restart", "desk_display.service"],
            check=False,
        )
        if result.returncode != 0:
            logging.error(
                "desk_display.service restart returned non-zero exit code: %s",
                result.returncode,
            )
    except Exception as exc:
        logging.error("Failed to restart desk_display.service: %s", exc)


def _git_pull_and_restart_desk_display_service() -> None:
    """Pull latest code in this repo, then restart the desk_display service."""

    request_shutdown("git pull + service restart")
    try:
        subprocess.run(
            ["git", "-C", SCRIPT_DIR, "pull"],
            check=False,
        )
    except Exception as exc:
        logging.error("Failed to run git pull in %s: %s", SCRIPT_DIR, exc)

    try:
        result = subprocess.run(
            ["sudo", "systemctl", "--no-block", "restart", "desk_display.service"],
            check=False,
        )
        if result.returncode != 0:
            logging.error(
                "desk_display.service restart returned non-zero exit code: %s",
                result.returncode,
            )
    except Exception as exc:
        logging.error("Failed to restart desk_display.service: %s", exc)


def _start_config_ui() -> None:
    global _config_ui_process

    if os.environ.get("SCREEN_CONFIG_AUTOSTART", "1").strip().lower() in {"0", "false", "no"}:
        logging.info("⚙️  Screen configuration UI autostart disabled.")
        return

    if _config_ui_process is not None and _config_ui_process.poll() is None:
        return

    config_ui_path = os.path.join(SCRIPT_DIR, "config_ui.py")
    if not os.path.exists(config_ui_path):
        logging.warning("Screen configuration UI entrypoint missing at %s.", config_ui_path)
        return

    logging.info("🧭 Launching screen configuration UI…")
    _config_ui_process = subprocess.Popen(
        [sys.executable, config_ui_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _check_control_buttons() -> bool:
    """Handle Display HAT Mini control buttons.

    Returns True when the caller should skip to the next screen immediately.
    """

    global _skip_request_pending

    if display is None:
        return False

    if _shutdown_event.is_set():
        return False

    if _check_touch_skip_request():
        return True

    new_presses = []
    skip_requested = False

    for name in _BUTTON_NAMES:
        try:
            pressed = display.is_button_pressed(name)
        except Exception as exc:
            logging.debug("Button poll failed for %s: %s", name, exc)
            pressed = False

        previously_pressed = _BUTTON_STATE[name]

        if pressed and not previously_pressed:
            new_presses.append(name)
            _BUTTON_PRESS_STARTED_AT[name] = time.monotonic()
            _BUTTON_PRESS_HANDLED[name] = False
        elif not pressed and previously_pressed:
            logging.debug("Button %s released.", name)
            _BUTTON_PRESS_STARTED_AT[name] = 0.0
            _BUTTON_PRESS_HANDLED[name] = False

        _BUTTON_STATE[name] = pressed

    if len(new_presses) > 1:
        _record_button_noise_event(new_presses)
        logging.warning(
            "Ignoring simultaneous button presses (%s); treating as noise.",
            ", ".join(new_presses),
        )
        for name in new_presses:
            _BUTTON_STATE[name] = False
            _BUTTON_PRESS_STARTED_AT[name] = 0.0
            _BUTTON_PRESS_HANDLED[name] = False
        return False

    now = time.monotonic()
    for name in _BUTTON_NAMES:
        if not _BUTTON_STATE[name]:
            continue
        if not _button_press_can_fire(name, now):
            continue

        _BUTTON_PRESS_HANDLED[name] = True
        if _handle_button_down(name):
            skip_requested = True

    if skip_requested or _manual_skip_event.is_set():
        return True

    return False


def _record_button_noise_event(new_presses: list[str]) -> None:
    """Track repeated simultaneous button reads and emit actionable diagnostics."""

    now = time.monotonic()
    _BUTTON_NOISE_TIMESTAMPS.append(now)

    while _BUTTON_NOISE_TIMESTAMPS:
        age = now - _BUTTON_NOISE_TIMESTAMPS[0]
        if age <= _BUTTON_NOISE_WINDOW_SECONDS:
            break
        _BUTTON_NOISE_TIMESTAMPS.popleft()

    if len(_BUTTON_NOISE_TIMESTAMPS) < _BUTTON_NOISE_WARNING_THRESHOLD:
        return

    _BUTTON_NOISE_TIMESTAMPS.clear()
    diagnostic = None
    get_power_diagnostic = getattr(wifi_utils, "get_power_diagnostic", None)
    if callable(get_power_diagnostic):
        try:
            diagnostic = get_power_diagnostic()
        except Exception as exc:
            logging.debug("Power diagnostic probe failed: %s", exc)

    detail = (
        f" Power diagnostic: {diagnostic}." if diagnostic else ""
    )
    logging.error(
        "⚠️ Repeated simultaneous button noise detected (%s). "
        "This can indicate undervoltage or electrical noise that may also impact Wi-Fi.%s",
        ", ".join(new_presses),
        detail,
    )


def _wait_with_button_checks(duration: float) -> bool:
    """Sleep for *duration* seconds while checking for control button presses.

    Returns True if the caller should skip the rest of the current screen.
    """

    if _manual_skip_event.is_set() or _skip_request_pending:
        _manual_skip_event.clear()
        return True

    end = time.monotonic() + duration
    last_frame_id = None
    can_poll_frame_id = hasattr(display, "frame_id")
    can_refresh_display = hasattr(display, "show")
    if can_poll_frame_id:
        try:
            last_frame_id = display.frame_id()
        except Exception:
            can_poll_frame_id = False

    while not _shutdown_event.is_set():
        if _manual_skip_event.is_set() or _skip_request_pending:
            _manual_skip_event.clear()
            return True

        if _check_control_buttons():
            _manual_skip_event.clear()
            return True

        if can_poll_frame_id and can_refresh_display:
            try:
                current_frame_id = display.frame_id()
            except Exception:
                can_poll_frame_id = False
            else:
                if current_frame_id != last_frame_id:
                    try:
                        display.show()
                    except Exception:
                        can_refresh_display = False
                    last_frame_id = current_frame_id

        remaining = end - time.monotonic()
        if remaining <= 0:
            break

        sleep_for = min(BUTTON_POLL_INTERVAL, remaining)
        if sleep_for > 0:
            if _manual_skip_event.wait(sleep_for):
                _manual_skip_event.clear()
                return True

            if _shutdown_event.is_set():
                return False

    return False


def _monitor_control_buttons() -> None:
    """Background poller to catch brief button presses."""

    logging.debug("Starting control button monitor thread.")

    try:
        while not _shutdown_event.is_set():
            try:
                _check_control_buttons()
            except Exception as exc:
                logging.debug("Button monitor loop failed: %s", exc)

            if _shutdown_event.wait(BUTTON_POLL_INTERVAL):
                break
    finally:
        logging.debug("Control button monitor thread exiting.")


_button_monitor_thread = threading.Thread(
    target=_monitor_control_buttons,
    name="control-button-monitor",
    daemon=True,
)
_button_monitor_thread.start()


def _next_screen_from_registry(
    registry: Dict[str, ScreenDefinition]
) -> Optional[ScreenDefinition]:
    """Return the next screen, honoring any pending skip requests."""

    global _skip_request_pending, _pending_previous_screen_id

    if _pending_previous_screen_id:
        previous_id = _pending_previous_screen_id
        _pending_previous_screen_id = None
        previous_entry = registry.get(previous_id)
        if previous_entry and previous_entry.available:
            logging.info("⏮️  Returning to previous screen '%s'.", previous_id)
            _skip_request_pending = False
            return previous_entry
        logging.info(
            "⏮️  Previous screen '%s' unavailable; resuming scheduled rotation.",
            previous_id,
        )

    scheduler = screen_scheduler
    if scheduler is None:
        _skip_request_pending = False
        return None

    entry = scheduler.next_available(registry)
    if entry is None:
        _skip_request_pending = False
        return None

    if not _skip_request_pending:
        return entry

    first_entry = entry
    avoided = set(_SKIP_BUTTON_SCREEN_IDS)
    if _last_screen_id:
        avoided.add(_last_screen_id)

    attempts = scheduler.node_count
    while entry and entry.id in avoided and attempts > 1:
        logging.debug(
            "Manual skip dropping '%s' from queue.",
            entry.id,
        )
        entry = scheduler.next_available(registry)
        attempts -= 1

    if entry and entry.id in avoided:
        logging.debug(
            "Manual skip fallback to '%s' (no alternative available).",
            entry.id,
        )
        entry = first_entry

    _skip_request_pending = False
    return entry

# ─── Screenshot / video outputs ──────────────────────────────────────────────
video_out = None

_archive_lock = threading.Lock()
_screenshot_count_lock = threading.Lock()
_screenshot_count: Optional[int] = None
_archive_pending = False


def _release_video_writer() -> None:
    global video_out

    if video_out:
        video_out.release()
        logging.info("🎬 Video finalized cleanly.")
        video_out = None


def _finalize_shutdown() -> None:
    """Run the shutdown cleanup sequence once."""

    if _shutdown_complete.is_set():
        return

    _clear_display_immediately("final cleanup")

    if video_out:
        logging.info("🎬 Finalizing video…")
    _release_video_writer()

    if _wifi_monitor_enabled and hasattr(wifi_utils, "stop_monitor"):
        try:
            wifi_utils.stop_monitor()
        except Exception as exc:
            logging.debug("Wi-Fi monitor shutdown skipped: %s", exc)

    global _button_monitor_thread
    if _button_monitor_thread and _button_monitor_thread.is_alive():
        _button_monitor_thread.join(timeout=1.0)
        _button_monitor_thread = None

    global _config_ui_process
    if _config_ui_process and _config_ui_process.poll() is None:
        logging.info("🛑 Stopping screen configuration UI…")
        _config_ui_process.terminate()
        try:
            _config_ui_process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            _config_ui_process.kill()
        _config_ui_process = None

    clear_update_indicator(display)
    _shutdown_complete.set()
    logging.info("👋 Shutdown cleanup finished.")


def _sanitize_directory_name(name: str) -> str:
    """Return a filesystem-friendly directory name while keeping spaces."""

    safe = name.strip().replace("/", "-").replace("\\", "-")
    safe = "".join(ch for ch in safe if ch.isalnum() or ch in (" ", "-", "_"))
    return safe or "Screens"


def _sanitize_filename_prefix(name: str) -> str:
    """Return a filesystem-friendly filename prefix."""

    safe = name.strip().replace("/", "-").replace("\\", "-")
    safe = safe.replace(" ", "_")
    safe = "".join(ch for ch in safe if ch.isalnum() or ch in ("_", "-"))
    return safe or "screen"


def _compute_existing_screenshot_count() -> int:
    count = 0
    try:
        for root, _, files in os.walk(SCREENSHOT_DIR):
            if os.path.abspath(root) == os.path.abspath(CURRENT_SCREENSHOT_DIR):
                continue
            count += sum(
                1
                for fname in files
                if fname.lower().endswith(ALLOWED_SCREEN_EXTS)
            )
    except Exception:
        count = 0
    return count


def _prune_screenshots_in_dir(dir_path: str, limit: int) -> int:
    if limit < 1:
        limit = 0
    if not os.path.isdir(dir_path):
        return 0

    entries = []
    for entry in os.scandir(dir_path):
        if not entry.is_file():
            continue
        if not entry.name.lower().endswith(ALLOWED_SCREEN_EXTS):
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        entries.append((mtime, entry.path))

    if len(entries) <= limit:
        return 0

    entries.sort(key=lambda item: (item[0], item[1]))
    to_remove = entries[: max(0, len(entries) - limit)]
    removed = 0
    for _, path in to_remove:
        try:
            os.remove(path)
            removed += 1
        except OSError as exc:
            logging.warning("Failed to prune screenshot %s: %s", path, exc)
    return removed


def _apply_screenshot_limits_on_startup() -> None:
    if not ENABLE_SCREENSHOTS:
        return

    total_removed = 0
    pruned_root = False
    pruned_archive_root = False
    try:
        for entry in os.scandir(SCREENSHOT_DIR):
            if entry.is_dir():
                if os.path.abspath(entry.path) == os.path.abspath(CURRENT_SCREENSHOT_DIR):
                    continue
                total_removed += _prune_screenshots_in_dir(
                    entry.path,
                    MAX_SCREENSHOTS_PER_SCREEN,
                )
            elif entry.is_file() and not pruned_root:
                if entry.name.lower().endswith(ALLOWED_SCREEN_EXTS):
                    total_removed += _prune_screenshots_in_dir(
                        SCREENSHOT_DIR,
                        MAX_SCREENSHOTS_PER_SCREEN,
                    )
                    pruned_root = True
    except FileNotFoundError:
        return

    try:
        for entry in os.scandir(SCREENSHOT_ARCHIVE_BASE):
            if entry.is_dir():
                total_removed += _prune_screenshots_in_dir(
                    entry.path,
                    MAX_ARCHIVED_SCREENSHOTS_PER_SCREEN,
                )
            elif entry.is_file() and not pruned_archive_root:
                if entry.name.lower().endswith(ALLOWED_SCREEN_EXTS):
                    total_removed += _prune_screenshots_in_dir(
                        SCREENSHOT_ARCHIVE_BASE,
                        MAX_ARCHIVED_SCREENSHOTS_PER_SCREEN,
                    )
                    pruned_archive_root = True
    except FileNotFoundError:
        return

    if total_removed:
        logging.info(
            "🧹 Pruned %d screenshot(s) to enforce limits on startup.",
            total_removed,
        )


def _ensure_screenshot_counter_locked() -> int:
    global _screenshot_count, _archive_pending

    if _screenshot_count is None:
        _screenshot_count = _compute_existing_screenshot_count()
        _archive_pending = _screenshot_count >= ARCHIVE_THRESHOLD
    return _screenshot_count or 0


def _register_screenshot_saved() -> Tuple[int, bool]:
    global _screenshot_count, _archive_pending

    with _screenshot_count_lock:
        current = _ensure_screenshot_counter_locked()
        _screenshot_count = current + 1
        if _screenshot_count >= ARCHIVE_THRESHOLD:
            _archive_pending = True
        return _screenshot_count, _archive_pending


def _register_screenshots_removed(count: int) -> Tuple[int, bool]:
    global _screenshot_count, _archive_pending

    count = max(0, count)
    with _screenshot_count_lock:
        current = _ensure_screenshot_counter_locked()
        if count:
            _screenshot_count = max(0, current - count)
        _archive_pending = (_screenshot_count or 0) >= ARCHIVE_THRESHOLD
        return _screenshot_count or 0, _archive_pending


def _save_screenshot(sid: str, img: Image.Image) -> Optional[Tuple[str, bool]]:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = _sanitize_directory_name(sid)
    prefix = _sanitize_filename_prefix(sid)
    target_dir = os.path.join(SCREENSHOT_DIR, folder)
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, f"{prefix}_{ts}.png")

    saved = False
    try:
        img.save(path)
        saved = True
    except Exception:
        logging.warning(f"⚠️ Screenshot save failed for '{sid}'")

    archive_needed = False
    if saved:
        _, archive_needed = _register_screenshot_saved()

    try:
        os.makedirs(CURRENT_SCREENSHOT_DIR, exist_ok=True)
        for entry in os.scandir(CURRENT_SCREENSHOT_DIR):
            if not entry.is_file():
                continue
            stem, ext = os.path.splitext(entry.name)
            if stem == prefix and ext.lower() in ALLOWED_SCREEN_EXTS:
                os.remove(entry.path)
        current_path = os.path.join(CURRENT_SCREENSHOT_DIR, f"{prefix}.png")
        img.save(current_path)
    except Exception:
        logging.warning(f"⚠️ Failed to update current screenshot for '{sid}'")

    if saved:
        removed = _prune_screenshots_in_dir(target_dir, MAX_SCREENSHOTS_PER_SCREEN)
        if removed:
            _register_screenshots_removed(removed)
            logging.info(
                "🧹 Pruned %d screenshot(s) from %s (limit %d).",
                removed,
                target_dir,
                MAX_SCREENSHOTS_PER_SCREEN,
            )

    if saved:
        return folder, archive_needed
    return None


def _write_display_status(
    sid: str,
    img: Image.Image,
    *,
    loop_iteration: int,
    rendered_at: Optional[datetime.datetime] = None,
) -> None:
    """Persist a heartbeat of what should currently be on the physical display."""

    if not DISPLAY_STATUS_PATH:
        return

    timestamp = rendered_at or datetime.datetime.now(datetime.timezone.utc)
    frame_id = None
    if hasattr(display, "frame_id"):
        try:
            frame_id = display.frame_id()
        except Exception:
            frame_id = None

    payload = {
        "screen_id": sid,
        "loop_iteration": loop_iteration,
        "rendered_at": timestamp.isoformat(),
        "image_digest": hashlib.sha1(img.tobytes()).hexdigest()[:12],
        "frame_id": frame_id,
    }

    temp_path = f"{DISPLAY_STATUS_PATH}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temp_path, DISPLAY_STATUS_PATH)
    except Exception as exc:
        logging.debug("Failed to update display heartbeat file: %s", exc)

def maybe_archive_screenshots(latest_folder: str) -> None:
    """Archive the newest screen's folder once the rolling counter hits the threshold."""

    global _archive_pending

    if not ENABLE_SCREENSHOTS:
        return
    if not latest_folder:
        return

    target_dir = os.path.join(SCREENSHOT_DIR, latest_folder)
    if not os.path.isdir(target_dir):
        return

    with _screenshot_count_lock:
        _ensure_screenshot_counter_locked()
        if not _archive_pending:
            return

    with _archive_lock:
        with _screenshot_count_lock:
            _ensure_screenshot_counter_locked()
            if not _archive_pending:
                return
            _archive_pending = False

        files = [
            path
            for path in glob.glob(os.path.join(target_dir, "**", "*"), recursive=True)
            if os.path.isfile(path) and path.lower().endswith(ALLOWED_SCREEN_EXTS)
        ]

        if not files:
            _register_screenshots_removed(0)
            return

        moved = 0
        created_archive_dirs = set()

        for src in files:
            rel_path = os.path.relpath(src, SCREENSHOT_DIR)
            try:
                dest = os.path.join(SCREENSHOT_ARCHIVE_MIRROR, rel_path)
                dest_dir = os.path.dirname(dest)
                if dest_dir and not os.path.exists(dest_dir):
                    os.makedirs(dest_dir, exist_ok=True)
                    created_archive_dirs.add(dest_dir)
                shutil.move(src, dest)
                moved += 1
            except Exception as e:
                logging.warning(f"⚠️  Could not move '{rel_path}' to archive: {e}")

        if moved == 0:
            for archive_dir in sorted(created_archive_dirs, reverse=True):
                if os.path.isdir(archive_dir) and not os.listdir(archive_dir):
                    try:
                        shutil.rmtree(archive_dir)
                    except Exception:
                        pass

        _register_screenshots_removed(moved)

        if moved:
            logging.info(
                "🗃️  Archived %s screenshot(s) from %s → %s/",
                moved,
                latest_folder,
                SCREENSHOT_ARCHIVE_MIRROR,
            )
            archive_dir = os.path.join(SCREENSHOT_ARCHIVE_MIRROR, latest_folder)
            removed = _prune_screenshots_in_dir(
                archive_dir,
                MAX_ARCHIVED_SCREENSHOTS_PER_SCREEN,
            )
            if removed:
                logging.info(
                    "🧹 Pruned %d archived screenshot(s) from %s (limit %d).",
                    removed,
                    archive_dir,
                    MAX_ARCHIVED_SCREENSHOTS_PER_SCREEN,
                )

# ─── SIGTERM handler ─────────────────────────────────────────────────────────
def _handle_sigterm(signum, frame):
    logging.info("✋ SIGTERM caught—requesting shutdown…")
    request_shutdown("SIGTERM")

signal.signal(signal.SIGTERM, _handle_sigterm)

# ─── Logos ───────────────────────────────────────────────────────────────────
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")
# Logos scroll across the screen; keep them just a bit shorter than the display
# while preserving aspect ratio during resize. Use a fixed width so replacement
# logos with different aspect ratios render consistently.
_logo_height_limit = HEIGHT - 30
if WIDTH >= 1280 or HEIGHT >= 720:
    short_edge = max(1, min(WIDTH, HEIGHT))
    _logo_height_limit = min(_logo_height_limit, int(round(short_edge * 0.55)))

LOGO_SCREEN_HEIGHT = max(1, _logo_height_limit)
TEAM_LOGO_HEIGHT   = LOGO_SCREEN_HEIGHT
LOGO_SCREEN_WIDTH = max(1, min(WIDTH, int(round(LOGO_SCREEN_HEIGHT * 1.5))))


def load_logo(fn, height=LOGO_SCREEN_HEIGHT, width=LOGO_SCREEN_WIDTH):
    path = os.path.join(IMAGES_DIR, fn)
    try:
        with Image.open(path) as img:
            has_transparency = (
                img.mode in ("RGBA", "LA")
                or (img.mode == "P" and "transparency" in img.info)
            )
            target_mode = "RGBA" if has_transparency else "RGB"
            img = img.convert(target_mode)
            target_height = max(1, int(height))
            target_width = max(1, int(width))
            if img.height == 0 or img.width == 0:
                return None
            width_ratio = target_width / img.width
            height_ratio = target_height / img.height
            scale = min(width_ratio, height_ratio)
            resized_size = (
                max(1, int(round(img.width * scale))),
                max(1, int(round(img.height * scale))),
            )
            resized = img.resize(resized_size, Image.ANTIALIAS)
            if resized_size == (target_width, target_height):
                return resized
            background = (0, 0, 0, 0) if has_transparency else (0, 0, 0)
            canvas = Image.new(target_mode, (target_width, target_height), background)
            offset = (
                (target_width - resized_size[0]) // 2,
                (target_height - resized_size[1]) // 2,
            )
            if has_transparency:
                canvas.paste(resized, offset, resized)
            else:
                canvas.paste(resized, offset)
        return canvas
    except Exception as e:
        logging.warning(f"Logo load failed '{fn}': {e}")
        return None


def _load_wolves_logo() -> Optional[Image.Image]:
    wolves_tri = (AHL_TEAM_TRICODE or "CHI").strip() or "CHI"
    for variant in {wolves_tri.upper(), wolves_tri.lower()}:
        wolves_logo = load_logo(f"ahl/{variant}.png", height=TEAM_LOGO_HEIGHT)
        if wolves_logo:
            return wolves_logo
    return load_logo("wolves.jpg", height=TEAM_LOGO_HEIGHT)


_LOGO_LOADERS: Dict[str, Callable[[], Optional[Image.Image]]] = {
    "weather logo": lambda: load_logo("weather.jpg"),
    "verano logo": lambda: load_logo("verano.jpg"),
    "bears logo": lambda: load_logo("nfl/chi.png"),
    "nfl logo": lambda: load_logo("nfl/nfl.png"),
    "hawks logo": lambda: load_logo("nhl/CHI.png", height=TEAM_LOGO_HEIGHT),
    "nhl logo": lambda: load_logo("nhl/nhl.png") or load_logo("nhl/NHL.png"),
    "wolves logo": _load_wolves_logo,
    "cubs logo": lambda: load_logo("mlb/CUBS.png", height=TEAM_LOGO_HEIGHT),
    "sox logo": lambda: load_logo("mlb/SOX.png", height=TEAM_LOGO_HEIGHT),
    "mlb logo": lambda: load_logo("mlb/MLB.png"),
    "nba logo": lambda: load_logo("nba/NBA.png"),
    "bulls logo": lambda: load_logo("nba/CHI.png", height=TEAM_LOGO_HEIGHT),
}


class LogoCache:
    def __init__(self, loaders: Dict[str, Callable[[], Optional[Image.Image]]]):
        self._loaders = loaders
        self._cache: Dict[str, Optional[Image.Image]] = {}

    def get(self, name: str) -> Optional[Image.Image]:
        if name in self._cache:
            return self._cache[name]

        loader = self._loaders.get(name)
        image = loader() if loader else None
        self._cache[name] = image
        return image


logo_cache = LogoCache(_LOGO_LOADERS)

# ─── Data cache & refresh ────────────────────────────────────────────────────
cache = {
    "bears":  {"stand": None},
    "weather": None,
    "hawks":   {"stand":None, "last":None, "live":None, "next":None, "next_home":None},
    "wolves":  {"last":None, "live":None, "next":None, "next_home":None},
    "bulls":   {"stand":None, "last":None, "live":None, "next":None, "next_home":None},
    "cubs":    {"stand":None, "last":None, "last_alt":None, "live":None, "next":None, "next_alt":None, "next_home":None},
    "sox":     {"stand":None, "last":None, "last_alt":None, "live":None, "next":None, "next_alt":None, "next_home":None},
    "scoreboards": {"nfl": None, "mlb": None, "nba": None, "ncaam": None, "nhl": None},
}

_FEED_DEPENDENCIES: Dict[str, Set[str]] = {
    "weather": {
        "weather1",
        "weather2",
        "weather hourly",
        "weather daily",
        "weather radar",
        "weather logo",
    },
    "bears": {"bears stand1", "bears stand2"},
    "hawks": {"hawks stand1", "hawks last", "hawks live", "hawks next", "hawks next home", "hawks logo"},
    "wolves": {"wolves last", "wolves live", "wolves next", "wolves next home", "wolves logo"},
    "bulls": {"bulls stand1", "bulls last", "bulls live", "bulls next", "bulls next home", "bulls logo"},
    "cubs": {
        "cubs stand1",
        "cubs stand2",
        "cubs last",
        "cubs result",
        "cubs live",
        "cubs next",
        "cubs next 2",
        "cubs next home",
        "cubs logo",
    },
    "sox": {
        "sox stand1",
        "sox stand2",
        "sox last",
        "sox live",
        "sox next",
        "sox next 2",
        "sox next home",
        "sox logo",
    },
    "scoreboards": {
        "NFL Scoreboard",
        "NFL Scoreboard v2",
        "NHL Scoreboard",
        "NHL Scoreboard v2",
        "MLB Scoreboard",
        "MLB Scoreboard v2",
        "NBA Scoreboard",
        "NBA Scoreboard v2",
        "NCAAM Scoreboard",
    },
}

_FEED_REFRESH_INTERVALS: Dict[str, int] = {
    "weather": SCHEDULE_UPDATE_INTERVAL,
    "hawks": SCHEDULE_UPDATE_INTERVAL,
    "bulls": SCHEDULE_UPDATE_INTERVAL,
    "wolves": SCHEDULE_UPDATE_INTERVAL,
    "bears": 1800,
    "cubs": 1800,
    "sox": 1800,
    "scoreboards": 120,
}

_SCOREBOARD_SCREEN_IDS = {
    "NFL Scoreboard",
    "NFL Scoreboard v2",
    "NHL Scoreboard",
    "NHL Scoreboard v2",
    "MLB Scoreboard",
    "MLB Scoreboard v2",
    "NBA Scoreboard",
    "NBA Scoreboard v2",
    "NCAAM Scoreboard",
}

_last_feed_refresh: Dict[str, float] = {}


def _requested_data_feeds() -> Set[str]:
    feeds: Set[str] = set()
    for feed, screen_ids in _FEED_DEPENDENCIES.items():
        if feed == "weather" and not ENABLE_WEATHER:
            continue
        if _requested_screen_ids & screen_ids:
            feeds.add(feed)
    return feeds


def _refresh_weather() -> None:
    cache["weather"] = data_provider.read_weather(ttl_seconds=WEATHER_REFRESH_SECONDS)
    if not cache["weather"]:
        logging.warning(
            "Weather feed returned no data; weather screens will remain hidden until a successful refresh."
        )


def _refresh_scoreboards() -> None:
    sports_payloads = data_provider.read_sports_payloads(ttl_seconds=120) or {}
    cache["scoreboards"].update(sports_payloads.get("scoreboards") or {})


def _refresh_scoreboards_fresh() -> None:
    sports_payloads = data_provider.read_sports_payloads(ttl_seconds=0) or {}
    cache["scoreboards"].update(sports_payloads.get("scoreboards") or {})


def _is_live_scoreboard_game(game: object) -> bool:
    if not isinstance(game, dict):
        return False

    status_fields: list[str] = []
    status_blob = game.get("status")
    if isinstance(status_blob, dict):
        for key in (
            "detailedState",
            "abstractGameState",
            "gameStatus",
            "gameStatusText",
            "state",
            "gameState",
            "displayClock",
        ):
            value = status_blob.get(key)
            if value:
                status_fields.append(str(value))
        coded = str(status_blob.get("codedGameState") or "").upper()
        status_code = str(status_blob.get("statusCode") or "").upper()
    else:
        coded = str(game.get("codedGameState") or "").upper()
        status_code = str(game.get("statusCode") or game.get("gameStatus") or "").upper()

    for key in (
        "gameStatusText",
        "gameStatus",
        "detailedState",
        "abstractGameState",
        "status",
        "gameState",
        "displayClock",
    ):
        value = game.get(key)
        if value:
            status_fields.append(str(value))

    status_text = " ".join(part.strip().lower() for part in status_fields if str(part).strip())

    if any(
        token in status_text
        for token in ("final", "postponed", "canceled", "cancelled", "suspend", "scheduled", "preview", "pregame")
    ):
        return False

    if any(
        token in status_text
        for token in ("live", "in progress", "in-progress", "intermission", "halftime", "quarter", "period", "ot", "top", "bottom")
    ):
        return True

    return coded == "I" or status_code in {"I", "2", "3"}


def _scoreboards_have_live_games(scoreboards: object) -> bool:
    if not isinstance(scoreboards, dict):
        return False

    for games in scoreboards.values():
        if not isinstance(games, list):
            continue
        for game in games:
            if _is_live_scoreboard_game(game):
                return True

    return False


def _should_force_refresh_scoreboards(screen_id: str, *, offline: bool) -> bool:
    """Return whether the current screen should trigger a fresh scoreboard pull."""

    return screen_id in _SCOREBOARD_SCREEN_IDS and not offline


def _refresh_bears() -> None:
    cache["bears"].update({
        "stand": data_fetch.fetch_bears_standings(),
    })


def _refresh_hawks() -> None:
    cache["hawks"].update({
        "stand": data_fetch.fetch_blackhawks_standings(),
        "last": data_fetch.fetch_blackhawks_last_game(),
        "live": data_fetch.fetch_blackhawks_live_game(),
        "next": data_fetch.fetch_blackhawks_next_game(),
        "next_home": data_fetch.fetch_blackhawks_next_home_game(),
    })


def _refresh_wolves() -> None:
    wolves_games = data_fetch.fetch_wolves_games() or {}
    cache["wolves"].update({
        "last": wolves_games.get("last_game"),
        "live": wolves_games.get("live_game"),
        "next": wolves_games.get("next_game"),
        "next_home": wolves_games.get("next_home_game"),
    })


def _refresh_bulls() -> None:
    cache["bulls"].update({
        "stand": data_fetch.fetch_bulls_standings(),
        "last": data_fetch.fetch_bulls_last_game(),
        "live": data_fetch.fetch_bulls_live_game(),
        "next": data_fetch.fetch_bulls_next_game(),
        "next_home": data_fetch.fetch_bulls_next_home_game(),
    })


def _refresh_cubs() -> None:
    cubg = data_fetch.fetch_cubs_games() or {}
    cache["cubs"].update({
        "stand": data_fetch.fetch_cubs_standings(),
        "last":  cubg.get("last_game"),
        "last_alt": cubg.get("last_game_alt"),
        "live":  cubg.get("live_game"),
        "next":  cubg.get("next_game"),
        "next_alt": cubg.get("next_game_alt"),
        "next_home": cubg.get("next_home_game"),
    })


def _refresh_sox() -> None:
    soxg = data_fetch.fetch_sox_games() or {}
    cache["sox"].update({
        "stand": data_fetch.fetch_sox_standings(),
        "last":  soxg.get("last_game"),
        "last_alt": soxg.get("last_game_alt"),
        "live":  soxg.get("live_game"),
        "next":  soxg.get("next_game"),
        "next_alt": soxg.get("next_game_alt"),
        "next_home": soxg.get("next_home_game"),
    })


_FEED_REFRESHERS: Dict[str, Callable[[], None]] = {
    "weather": _refresh_weather,
    "bears": _refresh_bears,
    "hawks": _refresh_hawks,
    "wolves": _refresh_wolves,
    "bulls": _refresh_bulls,
    "cubs": _refresh_cubs,
    "sox": _refresh_sox,
    "scoreboards": _refresh_scoreboards,
}


def refresh_all(force: bool = False) -> None:
    if _wifi_outage_active and not force:
        logging.info("⏸️  Skipping refresh during Wi-Fi outage.")
        return

    required_feeds = _requested_data_feeds()
    if not required_feeds:
        logging.info("⏭️  No scheduled data-dependent screens; skipping refresh.")
        return

    now = time.monotonic()
    due_feeds: Set[str] = set()
    for feed in required_feeds:
        interval = _FEED_REFRESH_INTERVALS.get(feed, SCHEDULE_UPDATE_INTERVAL)
        last_run = _last_feed_refresh.get(feed, 0.0)
        elapsed = now - last_run if last_run else float("inf")

        if force or elapsed >= interval:
            due_feeds.add(feed)
        else:
            remaining = int(interval - elapsed)
            logging.info("⏭️  Skipping %s refresh; %ds until next update.", feed, remaining)

    if not due_feeds:
        return

    logging.info("🔄 Refreshing data for feeds: %s", ", ".join(sorted(due_feeds)))
    for feed in sorted(due_feeds):
        refresher = _FEED_REFRESHERS.get(feed)
        if not refresher:
            continue
        try:
            refresher()
            _last_feed_refresh[feed] = time.monotonic()
            _bump_registry_cache_nonce()
        except Exception as exc:
            logging.error("Failed to refresh %s feed: %s", feed, exc)

def _background_refresh() -> None:
    time.sleep(30)
    while not _shutdown_event.is_set():
        feeds = _requested_data_feeds()
        if not feeds:
            logging.info("⏸️  Background refresh idle; no data-driven screens active.")
        else:
            refresh_all()

        if _shutdown_event.wait(SCHEDULE_UPDATE_INTERVAL):
            break




def _scheduled_startup_feed_order(limit: int = 4) -> List[str]:
    """Return feed names ordered by first upcoming scheduled appearance."""

    scheduler = screen_scheduler
    if scheduler is None:
        return sorted(_requested_data_feeds())

    scheduled_ids = scheduler.preview_scheduled_ids(limit)
    if not scheduled_ids:
        return sorted(_requested_data_feeds())

    ordered_feeds: List[str] = []
    for screen_id in scheduled_ids:
        for feed, feed_screen_ids in _FEED_DEPENDENCIES.items():
            if feed == "weather" and not ENABLE_WEATHER:
                continue
            if screen_id in feed_screen_ids and feed not in ordered_feeds:
                ordered_feeds.append(feed)

    for feed in sorted(_requested_data_feeds()):
        if feed not in ordered_feeds:
            ordered_feeds.append(feed)

    return ordered_feeds


def _refresh_feeds_in_order(feeds: List[str]) -> None:
    """Refresh feeds in the provided order with force semantics."""

    for feed in feeds:
        refresher = _FEED_REFRESHERS.get(feed)
        if not refresher:
            continue
        try:
            refresher()
            _last_feed_refresh[feed] = time.monotonic()
        except Exception as exc:
            logging.error("Failed to refresh %s feed: %s", feed, exc)


def _startup_refresh() -> None:
    """Prime cached data asynchronously so the first frame can render quickly."""

    try:
        ordered_feeds = _scheduled_startup_feed_order()
        if not ordered_feeds:
            logging.info("⏭️  Startup refresh idle; no data-driven screens active.")
            return

        first_wave = ordered_feeds[:2]
        remaining = ordered_feeds[2:]

        logging.info("🚀 Startup refresh first wave: %s", ", ".join(first_wave))
        _refresh_feeds_in_order(first_wave)

        if remaining and not _shutdown_event.is_set():
            logging.info("🧵 Startup refresh background wave: %s", ", ".join(remaining))
            _refresh_feeds_in_order(remaining)
    except Exception as exc:
        logging.error("Startup refresh failed: %s", exc)


def init_runtime() -> None:
    """Configure logging, storage paths, hardware, and background workers."""

    global SCREENSHOT_DIR, CURRENT_SCREENSHOT_DIR, SCREENSHOT_ARCHIVE_BASE
    global DISPLAY_STATUS_PATH
    global SCREENSHOT_ARCHIVE_MIRROR, _storage_paths, display, video_out
    global _background_refresh_thread, _startup_refresh_thread
    global _runtime_initialized, _wifi_monitor_enabled

    if _runtime_initialized:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.info("🖥️  Starting display service…")
    initialise_runtime_probes()

    _start_config_ui()

    _storage_paths = resolve_storage_paths(logger=logging.getLogger(__name__))
    SCREENSHOT_DIR = str(_storage_paths.screenshot_dir)
    CURRENT_SCREENSHOT_DIR = str(_storage_paths.current_screenshot_dir)
    SCREENSHOT_ARCHIVE_BASE = str(_storage_paths.archive_base)
    SCREENSHOT_ARCHIVE_MIRROR = SCREENSHOT_ARCHIVE_BASE
    DISPLAY_STATUS_PATH = os.path.join(CURRENT_SCREENSHOT_DIR, "display_status.json")

    # Display & Wi-Fi monitor
    display = Display()
    display.register_skip_event(_manual_skip_event)
    try:
        display.set_button_callback(_button_event_callback)
    except Exception:
        logging.debug("Button callback registration unavailable.")
    clear_update_indicator(display)

    _wifi_monitor_enabled = ENABLE_WIFI_MONITOR
    if _wifi_monitor_enabled and hasattr(wifi_utils, "should_monitor_wifi"):
        try:
            _wifi_monitor_enabled = bool(wifi_utils.should_monitor_wifi())
        except Exception as exc:
            logging.debug("Wi-Fi monitor eligibility check failed: %s", exc)

    if _wifi_monitor_enabled and hasattr(wifi_utils, "start_monitor"):
        logging.info("🔌 Starting Wi-Fi monitor…")
        try:
            wifi_utils.start_monitor(allow_recovery=ENABLE_WIFI_RECOVERY)
        except Exception as exc:
            logging.warning("Wi-Fi monitor unavailable: %s", exc)
    elif ENABLE_WIFI_MONITOR:
        logging.info("🔌 Wi-Fi monitor skipped for current network setup.")

    refresh_schedule_if_needed(force=True)

    if ENABLE_SCREENSHOTS:
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        os.makedirs(CURRENT_SCREENSHOT_DIR, exist_ok=True)
        os.makedirs(SCREENSHOT_ARCHIVE_BASE, exist_ok=True)
        _apply_screenshot_limits_on_startup()

    if ENABLE_VIDEO:
        import cv2, numpy as np
        FOURCC     = cv2.VideoWriter_fourcc(*"mp4v")
        video_path = os.path.join(SCREENSHOT_DIR, "display_output.mp4")
        logging.info(
            "🎥 Starting video capture → %s @ %s FPS using mp4v",
            video_path,
            VIDEO_FPS,
        )
        video_out = cv2.VideoWriter(video_path, FOURCC, VIDEO_FPS, (WIDTH, HEIGHT))
        if not video_out.isOpened():
            logging.error("❌ Cannot open video writer; disabling video output")
            video_out = None

    if _background_refresh_thread is None:
        _background_refresh_thread = threading.Thread(
            target=_background_refresh,
            daemon=True,
        )
        _background_refresh_thread.start()

    if _startup_refresh_thread is None or not _startup_refresh_thread.is_alive():
        _startup_refresh_thread = threading.Thread(
            target=_startup_refresh,
            daemon=True,
        )
        _startup_refresh_thread.start()

    _runtime_initialized = True

# ─── Main loop ───────────────────────────────────────────────────────────────
loop_count = 0

def main_loop():
    global loop_count, _last_screen_id, _dark_hours_active

    refresh_schedule_if_needed(force=True)

    try:
        while not _shutdown_event.is_set():
            refresh_schedule_if_needed()

            # Always drain button events, but keep skip requests active so the
            # currently visible screen (or the very next one) can react
            # immediately instead of idling on the previous frame for another
            # full iteration.
            _check_control_buttons()

            current_time = datetime.datetime.now(CENTRAL_TIME)

            if DARK_HOURS_ENABLED and is_within_dark_hours(current_time):
                if not _dark_hours_active:
                    logging.info("🌙 Entering configured dark hours; blanking display.")
                    try:
                        resume_display_updates()
                        clear_display(display)
                        display.show()
                    except Exception:
                        pass
                    suspend_display_updates()
                _dark_hours_active = True

                if _shutdown_event.is_set():
                    break

                if _wait_with_button_checks(SCREEN_DELAY):
                    continue

                _run_gc_maintenance()
                continue

            if _dark_hours_active:
                logging.info("🌅 Leaving dark hours; resuming screen rotation.")
                _dark_hours_active = False
                if not _manual_display_off:
                    resume_display_updates()

            if not _manual_display_off and not _dark_hours_active:
                if not display_updates_enabled():
                    logging.warning(
                        "Display updates were suspended unexpectedly; resuming."
                    )
                    resume_display_updates()

            # Wi-Fi outage handling
            if _wifi_monitor_enabled and hasattr(wifi_utils, "get_wifi_state"):
                try:
                    wifi_state, wifi_ssid = wifi_utils.get_wifi_state()
                except Exception as exc:
                    logging.debug("Wi-Fi state unavailable: %s", exc)
                    wifi_state, wifi_ssid = ("ok", None)
            else:
                wifi_state, wifi_ssid = ("ok", None)

            if _wifi_monitor_enabled:
                _update_wifi_outage_state(wifi_state)

            if screen_scheduler is None:
                logging.warning(
                    "No schedule available; sleeping for %s seconds.", SCREEN_DELAY
                )
                if _shutdown_event.is_set():
                    break
                if _wait_with_button_checks(SCREEN_DELAY):
                    continue
                _run_gc_maintenance()
                continue

            offline = _wifi_outage_active if _wifi_monitor_enabled else False
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            weather_fetched_at = data_fetch.get_weather_cache_timestamp()
            context = ScreenContext(
                display=display,
                cache=cache,
                logos=logo_cache,
                image_dir=IMAGES_DIR,
                now=current_time,
                now_utc=now_utc,
                offline=offline,
                weather_fetched_at=weather_fetched_at,
                skip_scoreboards=offline and _wifi_outage_live_games,
            )
            registry, _metadata = _build_registry_if_needed(context)

            entry = _next_screen_from_registry(registry)
            if entry is None:
                logging.info(
                    "No eligible screens available; sleeping for %s seconds.",
                    SCREEN_DELAY,
                )
                if _shutdown_event.is_set():
                    break
                if _wait_with_button_checks(SCREEN_DELAY):
                    continue
                _run_gc_maintenance()
                continue

            sid = entry.id
            if _should_force_refresh_scoreboards(sid, offline=offline):
                try:
                    _refresh_scoreboards_fresh()
                    _last_feed_refresh["scoreboards"] = time.monotonic()
                    _bump_registry_cache_nonce()
                except Exception as exc:
                    logging.error("Failed to force-refresh scoreboards for '%s': %s", sid, exc)

            loop_count += 1
            logging.info("🎬 Presenting '%s' (iteration %d)", sid, loop_count)

            frame_id_before_render = None
            if hasattr(display, "frame_id"):
                try:
                    frame_id_before_render = display.frame_id()
                except Exception:
                    frame_id_before_render = None

            try:
                with defer_clear_display():
                    result = entry.render()
            except Exception as exc:
                logging.error(f"Error in screen '{sid}': {exc}")
                _run_gc_maintenance()
                if _shutdown_event.is_set():
                    break
                if _wait_with_button_checks(SCREEN_DELAY):
                    continue
                continue

            already_displayed = False
            led_override = None
            consumed_delay = False
            img = None

            if result is None:
                logging.info(
                    "Screen '%s' returned no image; using current display buffer for outputs.",
                    sid,
                )
                current = getattr(display, "current_image", None)
                if isinstance(current, Image.Image):
                    img = current.copy()
                    already_displayed = True
            elif isinstance(result, ScreenImage):
                img = result.image
                already_displayed = result.displayed
                led_override = result.led_override
                consumed_delay = bool(getattr(result, "consumed_delay", False))
            elif isinstance(result, Image.Image):
                img = result

            if img is None:
                logging.info("Screen '%s' produced no drawable image.", sid)
                _run_gc_maintenance()
                if _shutdown_event.is_set():
                    break
                if _wait_with_button_checks(SCREEN_DELAY):
                    continue
                continue

            skip_delay = False
            led_context = (
                temporary_display_led(*led_override)
                if led_override is not None
                else nullcontext()
            )
            with led_context:
                if isinstance(img, Image.Image):
                    if "logo" in sid:
                        if not _frame_id_changed(display, frame_id_before_render):
                            logging.warning(
                                "Logo screen '%s' did not refresh the display; forcing frame output.",
                                sid,
                            )
                            animate_fade_in(display, img, steps=1, delay=0.01)
                        if ENABLE_SCREENSHOTS:
                            saved = _save_screenshot(sid, img)
                            if saved and saved[1]:
                                maybe_archive_screenshots(saved[0])
                        if ENABLE_VIDEO and video_out:
                            import cv2, numpy as np

                            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                            video_out.write(frame)
                    else:
                        if already_displayed and not _frame_id_changed(display, frame_id_before_render):
                            logging.warning(
                                "Screen '%s' reported displayed=True without refreshing the display; forcing frame output.",
                                sid,
                            )
                            already_displayed = False

                        if not already_displayed:
                            animate_fade_in(display, img, steps=1, delay=0.015)
                        if ENABLE_SCREENSHOTS:
                            saved = _save_screenshot(sid, img)
                            if saved and saved[1]:
                                maybe_archive_screenshots(saved[0])
                        if ENABLE_VIDEO and video_out:
                            import cv2, numpy as np

                            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                            video_out.write(frame)
                else:
                    logging.info("Screen '%s' produced no drawable image.", sid)

                if _shutdown_event.is_set():
                    break

                _write_display_status(sid, img, loop_iteration=loop_count)
                _last_screen_id = sid
                with _screen_history_lock:
                    _screen_history.append(sid)
                    if len(_screen_history) > _SCREEN_HISTORY_LIMIT:
                        _screen_history[:] = _screen_history[-_SCREEN_HISTORY_LIMIT:]
                wait_duration = 0.0 if consumed_delay else SCREEN_DELAY
                skip_delay = _wait_with_button_checks(wait_duration)

            if _shutdown_event.is_set():
                break

            if skip_delay:
                continue
            _run_gc_maintenance()

    finally:
        _finalize_shutdown()


def main() -> None:
    init_runtime()

    try:
        main_loop()
    except KeyboardInterrupt:
        logging.info("✋ CTRL-C caught—requesting shutdown…")
        request_shutdown("CTRL-C")
    finally:
        _finalize_shutdown()

    os._exit(0)


if __name__ == "__main__":
    main()
