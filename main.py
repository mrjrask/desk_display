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

import glob
import os
import time
import logging
import threading
import datetime
import signal
import shutil
import subprocess
import sys
from contextlib import nullcontext
from typing import Callable, Dict, Optional, Set, Tuple

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
from paths import resolve_storage_paths

from screens.draw_travel_time import (
    get_travel_active_window,
    is_travel_screen_active,
)
from screens.registry import ScreenContext, ScreenDefinition, build_screen_registry
from schedule import ScreenScheduler, build_scheduler, load_schedule_config

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "screens_config.json")
LOCAL_CONFIG_PATH = os.environ.get(
    "SCREENS_CONFIG_LOCAL_PATH", os.path.join(SCRIPT_DIR, "screens_config.local.json")
)
CONFIG_PATH = LOCAL_CONFIG_PATH if os.path.exists(LOCAL_CONFIG_PATH) else DEFAULT_CONFIG_PATH

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

_screen_config_mtime: Optional[float] = None
screen_scheduler: Optional[ScreenScheduler] = None
_requested_screen_ids: Set[str] = set()

_skip_request_pending = False
_last_screen_id: Optional[str] = None

_SKIP_BUTTON_SCREEN_IDS = {"date", "time"}

_shutdown_event = threading.Event()
_shutdown_complete = threading.Event()
_display_cleared = threading.Event()

BUTTON_POLL_INTERVAL = 0.1
_BUTTON_NAMES = ("A", "B", "X", "Y")
_BUTTON_STATE = {name: False for name in _BUTTON_NAMES}
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


def _handle_button_down(name: str) -> bool:
    """React to a newly pressed control button."""

    name = name.upper()
    if display is None:
        return False
    if name == "X":
        return _request_previous_screen()
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
    _handle_button_down(upper)


def _load_scheduler_from_config() -> Optional[ScreenScheduler]:
    try:
        config_data = load_schedule_config(CONFIG_PATH)
    except Exception as exc:
        logging.warning(f"Could not load schedule configuration: {exc}")
        return None

    try:
        scheduler = build_scheduler(config_data)
    except ValueError as exc:
        logging.error(f"Invalid schedule configuration: {exc}")
        return None

    return scheduler


def refresh_schedule_if_needed(force: bool = False) -> None:
    global _screen_config_mtime, screen_scheduler, _requested_screen_ids
    global _last_screen_id, _skip_request_pending, _pending_previous_screen_id

    try:
        mtime = os.path.getmtime(CONFIG_PATH)
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
    with _screen_history_lock:
        _screen_history.clear()
    logging.info("🔁 Loaded schedule configuration with %d node(s).", scheduler.node_count)


display: Optional[Display] = None
_background_refresh_thread: Optional[threading.Thread] = None
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
        subprocess.run(
            ["sudo", "systemctl", "restart", "desk_display.service"],
            check=False,
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
        elif not pressed and previously_pressed:
            logging.debug("Button %s released.", name)

        _BUTTON_STATE[name] = pressed

    if len(new_presses) > 1:
        logging.warning(
            "Ignoring simultaneous button presses (%s); treating as noise.",
            ", ".join(new_presses),
        )
        for name in new_presses:
            _BUTTON_STATE[name] = False
        return False

    for name in new_presses:
        if _handle_button_down(name):
            skip_requested = True

    if skip_requested or _manual_skip_event.is_set():
        return True

    return False


def _wait_with_button_checks(duration: float) -> bool:
    """Sleep for *duration* seconds while checking for control button presses.

    Returns True if the caller should skip the rest of the current screen.
    """

    if _manual_skip_event.is_set() or _skip_request_pending:
        _manual_skip_event.clear()
        return True

    end = time.monotonic() + duration
    while not _shutdown_event.is_set():
        if _manual_skip_event.is_set() or _skip_request_pending:
            _manual_skip_event.clear()
            return True

        if _check_control_buttons():
            _manual_skip_event.clear()
            return True

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

    if ENABLE_WIFI_MONITOR and hasattr(wifi_utils, "stop_monitor"):
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
LOGO_SCREEN_HEIGHT = max(1, HEIGHT - 30)
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
    "cubs":    {"stand":None, "last":None, "live":None, "next":None, "next_home":None},
    "sox":     {"stand":None, "last":None, "live":None, "next":None, "next_home":None},
}

_FEED_DEPENDENCIES: Dict[str, Set[str]] = {
    "weather": {"weather1", "weather2", "weather hourly", "weather radar", "weather logo"},
    "bears": {"bears stand1", "bears stand2"},
    "hawks": {"hawks stand1", "hawks stand2", "hawks last", "hawks live", "hawks next", "hawks next home", "hawks logo"},
    "wolves": {"wolves last", "wolves live", "wolves next", "wolves next home", "wolves logo"},
    "bulls": {"bulls stand1", "bulls stand2", "bulls last", "bulls live", "bulls next", "bulls next home", "bulls logo"},
    "cubs": {
        "cubs stand1",
        "cubs stand2",
        "cubs last",
        "cubs result",
        "cubs live",
        "cubs next",
        "cubs next home",
        "cubs logo",
    },
    "sox": {
        "sox stand1",
        "sox stand2",
        "sox last",
        "sox live",
        "sox next",
        "sox next home",
        "sox logo",
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
    cache["weather"] = data_fetch.fetch_weather(force_refresh=True)


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
        "live":  cubg.get("live_game"),
        "next":  cubg.get("next_game"),
        "next_home": cubg.get("next_home_game"),
    })


def _refresh_sox() -> None:
    soxg = data_fetch.fetch_sox_games() or {}
    cache["sox"].update({
        "stand": data_fetch.fetch_sox_standings(),
        "last":  soxg.get("last_game"),
        "live":  soxg.get("live_game"),
        "next":  soxg.get("next_game"),
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


def init_runtime() -> None:
    """Configure logging, storage paths, hardware, and background workers."""

    global SCREENSHOT_DIR, CURRENT_SCREENSHOT_DIR, SCREENSHOT_ARCHIVE_BASE
    global SCREENSHOT_ARCHIVE_MIRROR, _storage_paths, display, video_out
    global _background_refresh_thread, _runtime_initialized

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

    _start_config_ui()

    _storage_paths = resolve_storage_paths(logger=logging.getLogger(__name__))
    SCREENSHOT_DIR = str(_storage_paths.screenshot_dir)
    CURRENT_SCREENSHOT_DIR = str(_storage_paths.current_screenshot_dir)
    SCREENSHOT_ARCHIVE_BASE = str(_storage_paths.archive_base)
    SCREENSHOT_ARCHIVE_MIRROR = SCREENSHOT_ARCHIVE_BASE

    # Display & Wi-Fi monitor
    display = Display()
    display.register_skip_event(_manual_skip_event)
    try:
        display.set_button_callback(_button_event_callback)
    except Exception:
        logging.debug("Button callback registration unavailable.")
    clear_update_indicator(display)
    if ENABLE_WIFI_MONITOR and hasattr(wifi_utils, "start_monitor"):
        logging.info("🔌 Starting Wi-Fi monitor…")
        try:
            wifi_utils.start_monitor(allow_recovery=ENABLE_WIFI_RECOVERY)
        except Exception as exc:
            logging.warning("Wi-Fi monitor unavailable: %s", exc)

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

    refresh_all(force=True)
    _runtime_initialized = True

# ─── Main loop ───────────────────────────────────────────────────────────────
loop_count = 0
_travel_schedule_state: Optional[str] = None

def main_loop():
    global loop_count, _travel_schedule_state, _last_screen_id, _dark_hours_active

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

                gc.collect()
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
            if ENABLE_WIFI_MONITOR and hasattr(wifi_utils, "get_wifi_state"):
                try:
                    wifi_state, wifi_ssid = wifi_utils.get_wifi_state()
                except Exception as exc:
                    logging.debug("Wi-Fi state unavailable: %s", exc)
                    wifi_state, wifi_ssid = ("ok", None)
            else:
                wifi_state, wifi_ssid = ("ok", None)

            if ENABLE_WIFI_MONITOR:
                _update_wifi_outage_state(wifi_state)

            if screen_scheduler is None:
                logging.warning(
                    "No schedule available; sleeping for %s seconds.", SCREEN_DELAY
                )
                if _shutdown_event.is_set():
                    break
                if _wait_with_button_checks(SCREEN_DELAY):
                    continue
                gc.collect()
                continue

            travel_requested = "travel" in _requested_screen_ids
            offline = _wifi_outage_active if ENABLE_WIFI_MONITOR else False
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            weather_fetched_at = data_fetch.get_weather_cache_timestamp()
            context = ScreenContext(
                display=display,
                cache=cache,
                logos=logo_cache,
                image_dir=IMAGES_DIR,
                travel_requested=travel_requested,
                travel_active=is_travel_screen_active(),
                travel_window=get_travel_active_window(),
                previous_travel_state=_travel_schedule_state,
                now=current_time,
                now_utc=now_utc,
                offline=offline,
                weather_fetched_at=weather_fetched_at,
                skip_scoreboards=offline and _wifi_outage_live_games,
            )
            registry, metadata = build_screen_registry(context)
            _travel_schedule_state = metadata.get("travel_state", _travel_schedule_state)

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
                gc.collect()
                continue

            sid = entry.id
            loop_count += 1
            logging.info("🎬 Presenting '%s' (iteration %d)", sid, loop_count)

            try:
                with defer_clear_display():
                    result = entry.render()
            except Exception as exc:
                logging.error(f"Error in screen '{sid}': {exc}")
                gc.collect()
                if _shutdown_event.is_set():
                    break
                if _wait_with_button_checks(SCREEN_DELAY):
                    continue
                continue

            already_displayed = False
            led_override = None
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
            elif isinstance(result, Image.Image):
                img = result

            if img is None:
                logging.info("Screen '%s' produced no drawable image.", sid)
                gc.collect()
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
                        if ENABLE_SCREENSHOTS:
                            saved = _save_screenshot(sid, img)
                            if saved and saved[1]:
                                maybe_archive_screenshots(saved[0])
                        if ENABLE_VIDEO and video_out:
                            import cv2, numpy as np

                            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                            video_out.write(frame)
                    else:
                        if not already_displayed:
                            animate_fade_in(display, img, steps=8, delay=0.015)
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

                _last_screen_id = sid
                with _screen_history_lock:
                    _screen_history.append(sid)
                    if len(_screen_history) > _SCREEN_HISTORY_LIMIT:
                        _screen_history[:] = _screen_history[-_SCREEN_HISTORY_LIMIT:]
                skip_delay = _wait_with_button_checks(SCREEN_DELAY)

            if _shutdown_event.is_set():
                break

            if skip_delay:
                continue
            gc.collect()

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
