#!/usr/bin/env python3
"""
Main display loop driving the Pimoroni Display HAT Mini LCD,
with optional screenshot capture, H.264 MP4 video capture, Wi-Fi triage,
screen-config sequencing, and batch screenshot archiving.

Changes:
- Stop pruning single files; instead, when screenshots/ has >= ARCHIVE_THRESHOLD
  images, archive the whole set into screenshot_archive/dated_folders/<screen>/
  YYYYMMDD/HHMMSS/.
- Avoid creating empty archive folders.
- Guard logo screens when the image file is missing.
- Sort archived screenshots inside screenshot_archive/dated_folders/<screen>/
  YYYYMMDD/HHMMSS/ so they mirror the live screenshots/ folder structure.
"""
import warnings
from gpiozero.exc import PinFactoryFallback, NativePinFactoryFallback

warnings.filterwarnings("ignore", category=PinFactoryFallback)
warnings.filterwarnings("ignore", category=NativePinFactoryFallback)

import os
import sys
import time
import logging
import threading
import datetime
import signal
import shutil
from typing import Optional, Set

gc = __import__('gc')

from PIL import Image, ImageDraw

from config import (
    WIDTH,
    HEIGHT,
    SCREEN_DELAY,
    SCHEDULE_UPDATE_INTERVAL,
    FONT_DATE_SPORTS,
    ENABLE_SCREENSHOTS,
    ENABLE_VIDEO,
    VIDEO_FPS,
    ENABLE_WIFI_MONITOR,
    CENTRAL_TIME,
    TRAVEL_ACTIVE_WINDOW,
)
from utils import (
    Display,
    ScreenImage,
    clear_display,
    draw_text_centered,
    animate_fade_in,
)
import data_fetch
from services import wifi_utils

from screens.draw_date_time import draw_date, draw_time
from screens.draw_travel_time import (
    get_travel_active_window,
    is_travel_screen_active,
)
from screens.registry import ScreenContext, build_screen_registry
from schedule import ScreenScheduler, build_scheduler, load_schedule_config

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.info("🖥️  Starting display service…")

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "screens_config.json")

# ─── Screenshot archiving (batch) ────────────────────────────────────────────
ARCHIVE_THRESHOLD        = 500                  # archive when we reach this many images
SCREENSHOT_ARCHIVE_BASE  = os.path.join(SCRIPT_DIR, "screenshot_archive")
SCREENSHOT_ARCHIVE_DATED = os.path.join(SCREENSHOT_ARCHIVE_BASE, "dated_folders")
ARCHIVE_DEFAULT_FOLDER   = "Screens"
ALLOWED_SCREEN_EXTS      = (".png", ".jpg", ".jpeg")  # images only

_screen_config_mtime: Optional[float] = None
screen_scheduler: Optional[ScreenScheduler] = None
_requested_screen_ids: Set[str] = set()


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
    logging.info("🔁 Loaded schedule configuration with %d node(s).", scheduler.node_count)

# ─── Display & Wi-Fi monitor ─────────────────────────────────────────────────
display = Display()
if ENABLE_WIFI_MONITOR:
    logging.info("🔌 Starting Wi-Fi monitor…")
    wifi_utils.start_monitor()

# ─── Screenshot / video outputs ──────────────────────────────────────────────
SCREENSHOT_DIR = os.path.join(SCRIPT_DIR, "screenshots")
if ENABLE_SCREENSHOTS:
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    os.makedirs(SCREENSHOT_ARCHIVE_BASE, exist_ok=True)

video_out = None
if ENABLE_VIDEO:
    import cv2, numpy as np
    FOURCC     = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(SCREENSHOT_DIR, "display_output.mp4")
    logging.info(f"🎥 Starting video capture → {video_path} @ {VIDEO_FPS} FPS using mp4v")
    video_out = cv2.VideoWriter(video_path, FOURCC, VIDEO_FPS, (WIDTH, HEIGHT))
    if not video_out.isOpened():
        logging.error("❌ Cannot open video writer; disabling video output")
        video_out = None

_archive_lock = threading.Lock()


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


def _save_screenshot(sid: str, img: Image.Image) -> None:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = _sanitize_directory_name(sid)
    prefix = _sanitize_filename_prefix(sid)
    target_dir = os.path.join(SCREENSHOT_DIR, folder)
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, f"{prefix}_{ts}.png")

    try:
        img.save(path)
    except Exception:
        logging.warning(f"⚠️ Screenshot save failed for '{sid}'")


def _list_screenshot_files():
    try:
        results = []
        for root, _dirs, files in os.walk(SCREENSHOT_DIR):
            for fname in files:
                if not fname.lower().endswith(ALLOWED_SCREEN_EXTS):
                    continue
                rel_dir = os.path.relpath(root, SCREENSHOT_DIR)
                rel_path = fname if rel_dir == "." else os.path.join(rel_dir, fname)
                results.append(rel_path)
        return sorted(results)
    except Exception:
        return []

def maybe_archive_screenshots():
    """
    When screenshots/ reaches ARCHIVE_THRESHOLD images, move the current images
    into screenshot_archive/dated_folders/<screen>/YYYYMMDD/HHMMSS/ so the
    archive mirrors the live screenshots/ folder layout. Avoid creating empty
    archive folders.
    """
    if not ENABLE_SCREENSHOTS:
        return
    files = _list_screenshot_files()
    if len(files) < ARCHIVE_THRESHOLD:
        return

    with _archive_lock:
        files = _list_screenshot_files()
        if len(files) < ARCHIVE_THRESHOLD:
            return

        moved = 0
        day_stamp = None
        time_stamp = None
        created_batch_dirs = set()

        for fname in files:
            src = os.path.join(SCREENSHOT_DIR, fname)
            try:
                if day_stamp is None or time_stamp is None:
                    now = datetime.datetime.now()
                    day_stamp = now.strftime("%Y%m%d")
                    time_stamp = now.strftime("%H%M%S")

                parts = fname.split(os.sep)
                if len(parts) > 1:
                    screen_folder, remainder = parts[0], os.path.join(*parts[1:])
                else:
                    screen_folder, remainder = ARCHIVE_DEFAULT_FOLDER, parts[0]

                batch_dir = os.path.join(
                    SCREENSHOT_ARCHIVE_DATED,
                    screen_folder,
                    day_stamp,
                    time_stamp,
                )
                created_batch_dirs.add(batch_dir)
                dest = os.path.join(batch_dir, remainder)
                dest_dir = os.path.dirname(dest)
                if dest_dir and not os.path.exists(dest_dir):
                    os.makedirs(dest_dir, exist_ok=True)
                shutil.move(src, dest)
                moved += 1
            except Exception as e:
                logging.warning(f"⚠️  Could not move '{fname}' to archive: {e}")

        if moved == 0:
            for batch_dir in sorted(created_batch_dirs, reverse=True):
                if os.path.isdir(batch_dir):
                    try:
                        shutil.rmtree(batch_dir)
                    except Exception:
                        pass

        if moved:
            logging.info(
                "🗃️  Archived %s screenshot(s) → dated_folders/%s/%s",
                moved,
                day_stamp,
                time_stamp,
            )

# ─── SIGTERM handler ─────────────────────────────────────────────────────────
def _handle_sigterm(signum, frame):
    logging.info("✋ SIGTERM caught—clearing display & finalizing video…")
    try:
        clear_display(display)
    except Exception:
        pass
    if video_out:
        video_out.release()
        logging.info("🎬 Video finalized cleanly.")
    sys.exit(0)

signal.signal(signal.SIGTERM, _handle_sigterm)

# ─── Logos ───────────────────────────────────────────────────────────────────
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")
def load_logo(fn, height=80):
    path = os.path.join(IMAGES_DIR, fn)
    try:
        with Image.open(path) as img:
            has_transparency = (
                img.mode in ("RGBA", "LA")
                or (img.mode == "P" and "transparency" in img.info)
            )
            target_mode = "RGBA" if has_transparency else "RGB"
            img = img.convert(target_mode)
            ratio = height / img.height if img.height else 1
            resized = img.resize((int(img.width * ratio), height), Image.ANTIALIAS)
        return resized
    except Exception as e:
        logging.warning(f"Logo load failed '{fn}': {e}")
        return None

cubs_logo   = load_logo("cubs.jpg")
hawks_logo  = load_logo("hawks.jpg")
bulls_logo  = load_logo("nba/CHI.png")
sox_logo    = load_logo("sox.jpg")
weather_img = load_logo("weather.jpg")
mlb_logo    = load_logo("mlb.jpg")
nba_logo    = load_logo("nba/NBA.png")
nhl_logo    = load_logo("nhl/nhl.png") or load_logo("nhl/NHL.png")
nfl_logo    = load_logo("nfl/nfl.png")
verano_img  = load_logo("verano.jpg")
bears_logo  = load_logo("bears.png")

LOGOS = {
    "weather logo": weather_img,
    "verano logo": verano_img,
    "bears logo": bears_logo,
    "nfl logo": nfl_logo,
    "hawks logo": hawks_logo,
    "nhl logo": nhl_logo,
    "cubs logo": cubs_logo,
    "sox logo": sox_logo,
    "mlb logo": mlb_logo,
    "nba logo": nba_logo,
    "bulls logo": bulls_logo,
}

# ─── Data cache & refresh ────────────────────────────────────────────────────
cache = {
    "weather": None,
    "hawks":   {"last":None, "live":None, "next":None, "next_home":None},
    "bulls":   {"last":None, "live":None, "next":None, "next_home":None},
    "cubs":    {"stand":None, "last":None, "live":None, "next":None, "next_home":None},
    "sox":     {"stand":None, "last":None, "live":None, "next":None, "next_home":None},
}

def refresh_all():
    logging.info("🔄 Refreshing all data…")
    cache["weather"] = data_fetch.fetch_weather()
    cache["hawks"].update({
        "last": data_fetch.fetch_blackhawks_last_game(),
        "live": data_fetch.fetch_blackhawks_live_game(),
        "next": data_fetch.fetch_blackhawks_next_game(),
        "next_home": data_fetch.fetch_blackhawks_next_home_game(),
    })
    cache["bulls"].update({
        "last": data_fetch.fetch_bulls_last_game(),
        "live": data_fetch.fetch_bulls_live_game(),
        "next": data_fetch.fetch_bulls_next_game(),
        "next_home": data_fetch.fetch_bulls_next_home_game(),
    })
    cubg = data_fetch.fetch_cubs_games() or {}
    cache["cubs"].update({
        "stand": data_fetch.fetch_cubs_standings(),
        "last":  cubg.get("last_game"),
        "live":  cubg.get("live_game"),
        "next":  cubg.get("next_game"),
        "next_home": cubg.get("next_home_game"),
    })
    soxg = data_fetch.fetch_sox_games() or {}
    cache["sox"].update({
        "stand": data_fetch.fetch_sox_standings(),
        "last":  soxg.get("last_game"),
        "live":  soxg.get("live_game"),
        "next":  soxg.get("next_game"),
        "next_home": soxg.get("next_home_game"),
    })

def _background_refresh() -> None:
    time.sleep(30)
    while True:
        refresh_all()
        time.sleep(SCHEDULE_UPDATE_INTERVAL)


threading.Thread(
    target=_background_refresh,
    daemon=True
).start()
refresh_all()

# ─── Main loop ───────────────────────────────────────────────────────────────
loop_count = 0
_travel_schedule_state: Optional[str] = None

def main_loop():
    global loop_count, _travel_schedule_state

    refresh_schedule_if_needed(force=True)

    try:
        while True:
            refresh_schedule_if_needed()

            # Wi-Fi outage handling
            if ENABLE_WIFI_MONITOR:
                wifi_state, wifi_ssid = wifi_utils.get_wifi_state()
            else:
                wifi_state, wifi_ssid = ("ok", None)

            if ENABLE_WIFI_MONITOR and wifi_state != "ok":
                img = Image.new("RGB", (WIDTH, HEIGHT), "black")
                d   = ImageDraw.Draw(img)
                if wifi_state == "no_wifi":
                    draw_text_centered(d, "No Wi-Fi.", FONT_DATE_SPORTS, fill=(255,0,0))
                else:
                    draw_text_centered(d, "Wi-Fi ok.",     FONT_DATE_SPORTS, y_offset=-12, fill=(255,255,0))
                    draw_text_centered(d, wifi_ssid or "", FONT_DATE_SPORTS, fill=(255,255,0))
                    draw_text_centered(d, "No internet.",  FONT_DATE_SPORTS, y_offset=12,  fill=(255,0,0))
                display.image(img); display.show(); time.sleep(SCREEN_DELAY)
                for fn in (draw_date, draw_time):
                    img2 = fn(display, transition=True)
                    animate_fade_in(display, img2, steps=8, delay=0.015)
                    time.sleep(SCREEN_DELAY)
                gc.collect()
                continue

            if screen_scheduler is None:
                logging.warning(
                    "No schedule available; sleeping for %s seconds.", SCREEN_DELAY
                )
                time.sleep(SCREEN_DELAY)
                gc.collect()
                continue

            travel_requested = "travel" in _requested_screen_ids
            context = ScreenContext(
                display=display,
                cache=cache,
                logos=LOGOS,
                image_dir=IMAGES_DIR,
                travel_requested=travel_requested,
                travel_active=is_travel_screen_active(),
                travel_window=get_travel_active_window(),
                previous_travel_state=_travel_schedule_state,
                now=datetime.datetime.now(CENTRAL_TIME),
            )
            registry, metadata = build_screen_registry(context)
            _travel_schedule_state = metadata.get("travel_state", _travel_schedule_state)

            entry = screen_scheduler.next_available(registry)
            if entry is None:
                logging.info(
                    "No eligible screens available; sleeping for %s seconds.",
                    SCREEN_DELAY,
                )
                time.sleep(SCREEN_DELAY)
                gc.collect()
                continue

            sid = entry.id
            loop_count += 1
            logging.info("🎬 Presenting '%s' (iteration %d)", sid, loop_count)

            try:
                result = entry.render()
            except Exception as exc:
                logging.error(f"Error in screen '{sid}': {exc}")
                gc.collect()
                time.sleep(SCREEN_DELAY)
                continue

            if result is None:
                logging.info("Screen '%s' returned no image.", sid)
                gc.collect()
                time.sleep(SCREEN_DELAY)
                continue

            already_displayed = False
            img = None
            if isinstance(result, ScreenImage):
                img = result.image
                already_displayed = result.displayed
            elif isinstance(result, Image.Image):
                img = result

            if isinstance(img, Image.Image):
                if "logo" in sid:
                    if ENABLE_SCREENSHOTS:
                        _save_screenshot(sid, img)
                        maybe_archive_screenshots()
                    if ENABLE_VIDEO and video_out:
                        import cv2, numpy as np

                        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        video_out.write(frame)
                else:
                    if not already_displayed:
                        animate_fade_in(display, img, steps=8, delay=0.015)
                    if ENABLE_SCREENSHOTS:
                        _save_screenshot(sid, img)
                        maybe_archive_screenshots()
                    if ENABLE_VIDEO and video_out:
                        import cv2, numpy as np

                        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        video_out.write(frame)
            else:
                logging.info("Screen '%s' produced no drawable image.", sid)

            time.sleep(SCREEN_DELAY)
            gc.collect()

    finally:
        if video_out:
            logging.info("🎬 Finalizing video…")
            video_out.release()

if __name__ == '__main__':
    try:
        main_loop()
    except KeyboardInterrupt:
        logging.info("✋ CTRL-C caught—clearing display…")
        clear_display(display)
        if video_out:
            video_out.release()
        sys.exit(0)
