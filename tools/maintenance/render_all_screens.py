#!/usr/bin/env python3
"""Render every available screen to PNG and archive them into a dated ZIP."""
from __future__ import annotations

import argparse
import datetime as _dt
import io
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

from PIL import Image, ImageDraw

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow<9 compatibility
    RESAMPLE_LANCZOS = Image.LANCZOS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure the shared .env file is loaded before importing project modules so that
# configuration values (API keys, flags, etc.) are available to the renderer.
os.environ.setdefault("CONFIG_LOAD_DOTENV", "1")

import data_fetch
from config import (
    AHL_TEAM_TRICODE,
    CENTRAL_TIME,
    ENABLE_SCREENSHOTS,
    HEIGHT,
    WIDTH,
)
from screens.draw_travel_time import get_travel_active_window, is_travel_screen_active
from screens.registry import ScreenContext, ScreenDefinition, build_screen_registry
from schedule import build_scheduler, load_schedule_config
from screens_catalog import SCREEN_IDS
from utils import ScreenImage
from paths import resolve_storage_paths

try:
    import utils
except ImportError:  # pragma: no cover
    utils = None  # type: ignore


CONFIG_PATH = PROJECT_ROOT / "screens_config.json"
IMAGES_DIR = PROJECT_ROOT / "images"

_storage_paths = resolve_storage_paths(logger=logging.getLogger(__name__))
SCREENSHOT_DIR = str(_storage_paths.screenshot_dir)
CURRENT_SCREENSHOT_DIR = str(_storage_paths.current_screenshot_dir)
ARCHIVE_DIR = str(_storage_paths.archive_base)


class HeadlessDisplay:
    """Minimal display stub that captures the latest image frame."""

    def __init__(self, width: int = WIDTH, height: int = HEIGHT):
        self.width = width
        self.height = height
        self._current = Image.new("RGB", (self.width, self.height), "black")

    def clear(self) -> None:
        self._current = Image.new("RGB", (self.width, self.height), "black")

    def image(self, pil_img: Image.Image) -> None:
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        self._current = pil_img.copy()

    def show(self) -> None:  # pragma: no cover - no hardware interaction
        pass

    @property
    def current_image(self) -> Image.Image:
        return self._current


def _sanitize_directory_name(name: str) -> str:
    safe = name.strip().replace("/", "-").replace("\\", "-")
    safe = "".join(ch for ch in safe if ch.isalnum() or ch in (" ", "-", "_"))
    return safe or "Screens"


def _sanitize_filename_prefix(name: str) -> str:
    safe = name.strip().replace("/", "-").replace("\\", "-")
    safe = safe.replace(" ", "_")
    safe = "".join(ch for ch in safe if ch.isalnum() or ch in ("_", "-"))
    return safe or "screen"


LOGO_SCREEN_HEIGHT = max(1, HEIGHT - 30)
TEAM_LOGO_HEIGHT   = LOGO_SCREEN_HEIGHT
LOGO_SCREEN_WIDTH = max(1, min(WIDTH, int(round(LOGO_SCREEN_HEIGHT * 1.5))))


def load_logo(
    filename: str,
    height: int = LOGO_SCREEN_HEIGHT,
    width: int = LOGO_SCREEN_WIDTH,
) -> Optional[Image.Image]:
    path = IMAGES_DIR / filename
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
            resized = img.resize(resized_size, RESAMPLE_LANCZOS)
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
    except Exception as exc:
        logging.warning("Logo load failed '%s': %s", filename, exc)
        return None


def build_logo_map() -> Dict[str, Optional[Image.Image]]:
    wolves_logo = None
    wolves_tri = (AHL_TEAM_TRICODE or "CHI").strip() or "CHI"
    for variant in {wolves_tri.upper(), wolves_tri.lower()}:
        wolves_logo = load_logo(f"ahl/{variant}.png", height=TEAM_LOGO_HEIGHT)
        if wolves_logo:
            break
    if wolves_logo is None:
        wolves_logo = load_logo("wolves.jpg", height=TEAM_LOGO_HEIGHT)

    return {
        "weather logo": load_logo("weather.jpg"),
        "verano logo": load_logo("verano.jpg"),
        "bears logo": load_logo("nfl/chi.png"),
        "nfl logo": load_logo("nfl/nfl.png"),
        "hawks logo": load_logo("nhl/CHI.png", height=TEAM_LOGO_HEIGHT),
        "nhl logo": load_logo("nhl/nhl.png") or load_logo("nhl/NHL.png"),
        "wolves logo": wolves_logo,
        "cubs logo": load_logo("mlb/CUBS.png", height=TEAM_LOGO_HEIGHT),
        "sox logo": load_logo("mlb/SOX.png", height=TEAM_LOGO_HEIGHT),
        "mlb logo": load_logo("mlb/MLB.png"),
        "nba logo": load_logo("nba/NBA.png"),
        "bulls logo": load_logo("nba/CHI.png", height=TEAM_LOGO_HEIGHT),
    }


def build_cache() -> Dict[str, object]:
    logging.info("Refreshing data feeds…")
    cache: Dict[str, object] = {
        "weather": None,
        "bears": {"stand": None},
        "hawks": {"last": None, "live": None, "next": None, "next_home": None},
        "wolves": {"last": None, "live": None, "next": None, "next_home": None},
        "bulls": {
            "stand": None,
            "last": None,
            "live": None,
            "next": None,
            "next_home": None,
        },
        "cubs": {
            "stand": None,
            "last": None,
            "live": None,
            "next": None,
            "next_home": None,
        },
        "sox": {
            "stand": None,
            "last": None,
            "live": None,
            "next": None,
            "next_home": None,
        },
    }

    cache["weather"] = data_fetch.fetch_weather()
    cache["bears"]["stand"] = data_fetch.fetch_bears_standings()
    cache["hawks"].update(
        {
            "last": data_fetch.fetch_blackhawks_last_game(),
            "live": data_fetch.fetch_blackhawks_live_game(),
            "next": data_fetch.fetch_blackhawks_next_game(),
            "next_home": data_fetch.fetch_blackhawks_next_home_game(),
            "stand": data_fetch.fetch_blackhawks_standings(),
        }
    )
    wolves_games = data_fetch.fetch_wolves_games() or {}
    cache["wolves"].update(
        {
            "last": wolves_games.get("last_game"),
            "live": wolves_games.get("live_game"),
            "next": wolves_games.get("next_game"),
            "next_home": wolves_games.get("next_home_game"),
        }
    )
    cache["bulls"].update(
        {
            "last": data_fetch.fetch_bulls_last_game(),
            "live": data_fetch.fetch_bulls_live_game(),
            "next": data_fetch.fetch_bulls_next_game(),
            "next_home": data_fetch.fetch_bulls_next_home_game(),
            "stand": data_fetch.fetch_bulls_standings(),
        }
    )

    cubs_games = data_fetch.fetch_cubs_games() or {}
    cache["cubs"].update(
        {
            "stand": data_fetch.fetch_cubs_standings(),
            "last": cubs_games.get("last_game"),
            "live": cubs_games.get("live_game"),
            "next": cubs_games.get("next_game"),
            "next_home": cubs_games.get("next_home_game"),
        }
    )

    sox_games = data_fetch.fetch_sox_games() or {}
    cache["sox"].update(
        {
            "stand": data_fetch.fetch_sox_standings(),
            "last": sox_games.get("last_game"),
            "live": sox_games.get("live_game"),
            "next": sox_games.get("next_game"),
            "next_home": sox_games.get("next_home_game"),
        }
    )

    return cache


def load_requested_screen_ids() -> Tuple[set[str], Optional[str]]:
    try:
        config = load_schedule_config(CONFIG_PATH)
        scheduler = build_scheduler(config)
        logging.info("Loaded %d schedule entries", scheduler.node_count)
        return scheduler.requested_ids, None
    except Exception as exc:
        logging.warning("Failed to load schedule configuration: %s", exc)
        return set(), str(exc)


def _extract_image(result: object, display: HeadlessDisplay) -> Optional[Image.Image]:
    if isinstance(result, ScreenImage):
        if result.image is not None:
            return result.image
        if result.displayed:
            return display.current_image.copy()
        return None
    if isinstance(result, Image.Image):
        return result
    return display.current_image.copy()


def _write_zip(assets: Iterable[Tuple[str, Image.Image]], timestamp: _dt.datetime) -> str:
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    zip_name = f"screens_{timestamp.strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = os.path.join(ARCHIVE_DIR, zip_name)

    counts: Dict[str, int] = {}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for screen_id, image in assets:
            prefix = _sanitize_filename_prefix(screen_id)
            counts[prefix] = counts.get(prefix, 0) + 1
            suffix = "" if counts[prefix] == 1 else f"_{counts[prefix] - 1:02d}"
            filename = f"{prefix}{suffix}.png"

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            zf.writestr(filename, buf.getvalue())
    return zip_path


def _write_screenshots(
    assets: Iterable[Tuple[str, Image.Image]], timestamp: _dt.datetime
) -> list[str]:
    dated_dir = os.path.join(SCREENSHOT_DIR, timestamp.strftime("%Y%m%d"))
    os.makedirs(dated_dir, exist_ok=True)
    os.makedirs(CURRENT_SCREENSHOT_DIR, exist_ok=True)

    saved: list[str] = []
    current_written: set[str] = set()
    ts_suffix = timestamp.strftime("%Y%m%d_%H%M%S")
    counts: Dict[str, int] = {}

    for screen_id, image in assets:
        prefix = _sanitize_filename_prefix(screen_id)
        counts[prefix] = counts.get(prefix, 0) + 1
        suffix = "" if counts[prefix] == 1 else f"_{counts[prefix] - 1:02d}"
        filename = f"{prefix}{suffix}_{ts_suffix}.png"
        path = os.path.join(dated_dir, filename)
        image.save(path)
        saved.append(path)

        current_filename = f"{prefix}{suffix}.png"
        current_path = os.path.join(CURRENT_SCREENSHOT_DIR, current_filename)
        image.save(current_path)
        current_written.add(current_path)

    # Remove outdated current screenshots so the folder only reflects the latest run
    for existing in os.listdir(CURRENT_SCREENSHOT_DIR):
        existing_path = os.path.join(CURRENT_SCREENSHOT_DIR, existing)
        if existing_path not in current_written:
            try:
                os.remove(existing_path)
            except OSError as exc:
                logging.warning("Failed to remove stale screenshot %s: %s", existing_path, exc)

    return saved


def _cleanup_screenshots(paths: Iterable[str]) -> None:
    removed = 0
    for path in paths:
        try:
            os.remove(path)
            removed += 1
        except FileNotFoundError:
            logging.debug("Screenshot already removed: %s", path)
        except OSError as exc:
            logging.warning("Failed to remove screenshot %s: %s", path, exc)

    if removed:
        logging.info("Cleaned up %d screenshot(s) from %s", removed, SCREENSHOT_DIR)


def _suppress_animation_delay():
    if utils is None:
        return lambda: None
    original_sleep = utils.time.sleep

    def restore() -> None:
        utils.time.sleep = original_sleep

    utils.time.sleep = lambda *_args, **_kwargs: None
    return restore


def _suppress_image_loading() -> Callable[[], None]:
    original_open = Image.open

    def _build_placeholder(size: tuple[int, int]) -> Image.Image:
        width, height = max(1, size[0]), max(1, size[1])
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        for inset in range(0, 6, 2):
            draw.rectangle(
                [inset, inset, width - 1 - inset, height - 1 - inset],
                outline=(255, 255, 255, 255),
            )
        draw.line((0, 0, width - 1, height - 1), fill=(255, 255, 255, 255), width=1)
        draw.line((0, height - 1, width - 1, 0), fill=(255, 255, 255, 255), width=1)
        return img

    def placeholder_open(*args, **kwargs) -> Image.Image:
        size = (max(1, WIDTH // 2), max(1, HEIGHT // 2))
        try:
            with original_open(*args, **kwargs) as opened:
                size = opened.size
        except Exception:
            pass
        return _build_placeholder(size)

    def restore() -> None:
        Image.open = original_open

    Image.open = placeholder_open
    return restore


def _render_all_screens_impl(
    *,
    sync_screenshots: bool = ENABLE_SCREENSHOTS,
    create_archive: bool = True,
    ignore_schedule: bool = False,
    suppress_images: bool = False,
) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    restore_sleep = _suppress_animation_delay()
    restore_images = _suppress_image_loading() if suppress_images else lambda: None
    assets: list[Tuple[str, Image.Image]] = []
    now = _dt.datetime.now(CENTRAL_TIME)
    try:
        display = HeadlessDisplay()
        logos = build_logo_map()
        cache = build_cache()

        schedule_error: Optional[str] = None
        requested_ids: set[str] = set()
        travel_requested = True
        if not ignore_schedule:
            requested_ids, schedule_error = load_requested_screen_ids()
            if schedule_error:
                logging.info("Continuing without schedule data (%s)", schedule_error)

        now = _dt.datetime.now(CENTRAL_TIME)
        context = ScreenContext(
            display=display,
            cache=cache,
            logos=logos,
            image_dir=str(IMAGES_DIR),
            travel_requested=travel_requested,
            travel_active=is_travel_screen_active(),
            travel_window=get_travel_active_window(),
            previous_travel_state=None,
            now=now,
        )

        registry, _metadata = build_screen_registry(context)

        screen_ids = sorted(set(SCREEN_IDS) | set(registry.keys()))
        for screen_id in screen_ids:
            definition: Optional[ScreenDefinition] = registry.get(screen_id)
            if definition is None:
                logging.warning(
                    "No renderer registered for '%s'; creating placeholder image.",
                    screen_id,
                )
                assets.append((screen_id, Image.new("RGB", (display.width, display.height), "black")))
                continue
            if not definition.available:
                logging.info("Rendering '%s' (marked unavailable)", screen_id)
            else:
                logging.info("Rendering '%s'", screen_id)
            try:
                result = definition.render()
            except Exception as exc:
                logging.error("Failed to render '%s': %s", screen_id, exc)
                continue

            if result is None:
                logging.info(
                    "Screen '%s' returned no image; capturing current frame.",
                    screen_id,
                )
                image = display.current_image.copy()
            else:
                image = _extract_image(result, display)
            if image is None:
                logging.warning("No image returned for '%s'", screen_id)
                continue
            assets.append((screen_id, image))
            display.clear()

    finally:
        restore_images()
        restore_sleep()

    if not assets:
        logging.error("No screen images were produced.")
        return 1

    saved: list[str] = []
    if sync_screenshots:
        saved = _write_screenshots(assets, now)
        target_dir = os.path.dirname(saved[0]) if saved else SCREENSHOT_DIR
        logging.info(
            "Updated %d screenshot(s) in %s", len(saved), target_dir
        )

    if create_archive:
        archive_path = _write_zip(assets, now)
        logging.info("Archived %d screen(s) → %s", len(assets), archive_path)
        print(archive_path)
    elif not create_archive and not sync_screenshots:
        logging.info("Rendered %d screen(s) (no outputs written)", len(assets))

    return 0


def render_all_screens(
    *,
    sync_screenshots: bool = ENABLE_SCREENSHOTS,
    create_archive: bool = True,
    ignore_schedule: bool = False,
    suppress_images: bool = False,
) -> int:
    return _render_all_screens_impl(
        sync_screenshots=sync_screenshots,
        create_archive=create_archive,
        ignore_schedule=ignore_schedule,
        suppress_images=suppress_images,
    )


def render_all_screens_without_images(
    *,
    sync_screenshots: bool = ENABLE_SCREENSHOTS,
    create_archive: bool = True,
    ignore_schedule: bool = False,
) -> int:
    return render_all_screens(
        sync_screenshots=sync_screenshots,
        create_archive=create_archive,
        ignore_schedule=ignore_schedule,
        suppress_images=True,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-a",
        "--all",
        dest="ignore_schedule",
        action="store_true",
        help="Ignore screens_config.json and render every available screen.",
    )
    parser.add_argument(
        "--sync-screenshots",
        dest="sync_screenshots",
        action="store_true",
        help=(
            "Write PNG files for each rendered screen to the screenshots directory "
            "(default mirrors ENABLE_SCREENSHOTS from config.py)."
        ),
    )
    parser.add_argument(
        "--no-sync-screenshots",
        dest="sync_screenshots",
        action="store_false",
        help="Disable screenshot syncing even if ENABLE_SCREENSHOTS is true.",
    )
    parser.set_defaults(sync_screenshots=ENABLE_SCREENSHOTS)
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Skip creating the ZIP archive of rendered screens.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Render with placeholder frames instead of loading images.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return _render_all_screens_impl(
        sync_screenshots=args.sync_screenshots,
        create_archive=not args.no_archive,
        ignore_schedule=args.ignore_schedule,
        suppress_images=args.no_images,
    )


if __name__ == "__main__":
    sys.exit(main())
