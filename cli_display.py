#!/usr/bin/env python3
"""Windowed CLI entry point for driving the desk display scheduler."""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
import signal
import threading
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image

try:
    import pygame
except Exception as exc:  # pragma: no cover - import guard
    pygame = None
    _PYGAME_IMPORT_ERROR = exc
else:  # pragma: no cover - exercised in integration usage
    _PYGAME_IMPORT_ERROR = None

from config import (
    CENTRAL_TIME,
    SCREEN_DELAY,
    SCHEDULE_UPDATE_INTERVAL,
    VIDEO_FPS,
    WIDTH,
    HEIGHT,
)
from render_all_screens import build_cache, build_logo_map
from runtime_events import shutdown_event as _shutdown_event
from schedule import ScreenScheduler, build_scheduler, load_schedule_config
from screens.draw_travel_time import get_travel_active_window, is_travel_screen_active
from screens.registry import ScreenContext, build_screen_registry
from utils import ScreenImage, animate_fade_in

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "screens_config.json"
IMAGES_DIR = SCRIPT_DIR / "images"


def request_shutdown(reason: str) -> None:
    """Trigger a graceful shutdown using the shared shutdown event."""

    if not _shutdown_event.is_set():
        logging.info("✋ Shutdown requested (%s).", reason)
    _shutdown_event.set()


def _sanitize_directory_name(name: str) -> str:
    safe = name.strip().replace("/", "-").replace("\\", "-")
    safe = "".join(ch for ch in safe if ch.isalnum() or ch in (" ", "-", "_"))
    return safe or "Screens"


def _sanitize_filename_prefix(name: str) -> str:
    safe = name.strip().replace("/", "-").replace("\\", "-")
    safe = safe.replace(" ", "_")
    safe = "".join(ch for ch in safe if ch.isalnum() or ch in ("_", "-"))
    return safe or "screen"


class FrameCapture:
    """Optional helpers for screenshot/video capture."""

    def __init__(
        self,
        *,
        directory: Optional[Path] = None,
        video_path: Optional[Path] = None,
        fps: int = VIDEO_FPS,
    ) -> None:
        self._directory = directory
        self._video_path = video_path
        self._video_writer = None
        if self._directory is not None:
            self._directory.mkdir(parents=True, exist_ok=True)
        if self._video_path is not None:
            self._video_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._video_writer = self._build_writer(self._video_path, fps)
            except Exception as exc:  # pragma: no cover - requires cv2 backend
                logging.warning("⚠️  Video capture disabled: %s", exc)
                self._video_writer = None

    @staticmethod
    def _build_writer(path: Path, fps: int):  # pragma: no cover - requires cv2
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(str(path), fourcc, fps, (WIDTH, HEIGHT))

    def handle(self, screen_id: str, image: Image.Image) -> None:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        if self._directory is not None:
            folder = self._directory / _sanitize_directory_name(screen_id)
            folder.mkdir(parents=True, exist_ok=True)
            prefix = _sanitize_filename_prefix(screen_id)
            target = folder / f"{prefix}_{ts}.png"
            try:
                image.save(target)
            except Exception as exc:  # pragma: no cover - disk errors
                logging.warning("⚠️  Screenshot save failed (%s): %s", screen_id, exc)

        if self._video_writer is not None:
            import cv2
            import numpy as np

            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            self._video_writer.write(frame)

    def close(self) -> None:
        if self._video_writer is not None:  # pragma: no cover - requires cv2
            self._video_writer.release()
            self._video_writer = None


class CacheManager:
    """Refresh the data cache in the background at a fixed cadence."""

    def __init__(self, interval: float) -> None:
        self._interval = max(1.0, float(interval))
        self._cache = {}
        self._lock = threading.Lock()
        self.refresh(force=True)
        self._thread = threading.Thread(target=self._loop, name="cache-refresh", daemon=True)
        self._thread.start()

    def refresh(self, force: bool = False) -> None:
        if not force and _shutdown_event.is_set():
            return
        try:
            cache = build_cache()
        except Exception as exc:
            logging.warning("Cache refresh failed: %s", exc)
            return
        with self._lock:
            self._cache = cache
            logging.info("Refreshed data cache (%d keys).", len(cache))

    def snapshot(self) -> dict:
        with self._lock:
            return self._cache

    def _loop(self) -> None:
        while not _shutdown_event.wait(self._interval):
            self.refresh(force=True)


class SchedulerDriver:
    """Reload the playlist scheduler when the configuration changes."""

    def __init__(self, config_path: Path) -> None:
        self._path = config_path
        self._mtime: Optional[float] = None
        self.scheduler: Optional[ScreenScheduler] = None
        self.requested_ids: set[str] = set()
        self.refresh(force=True)

    def refresh(self, force: bool = False) -> None:
        try:
            mtime = self._path.stat().st_mtime
        except OSError:
            mtime = None

        if not force and mtime == self._mtime and self.scheduler is not None:
            return

        try:
            config = load_schedule_config(str(self._path))
            scheduler = build_scheduler(config)
        except Exception as exc:
            logging.warning("Failed to load schedule configuration: %s", exc)
            if force:
                self.scheduler = None
                self.requested_ids = set()
            return

        self.scheduler = scheduler
        self.requested_ids = scheduler.requested_ids
        self._mtime = mtime
        logging.info("🔁 Loaded schedule configuration with %d node(s).", scheduler.node_count)


class WindowDisplay:
    """Minimal pygame window that mirrors the Pimoroni display API."""

    def __init__(
        self,
        *,
        display_index: Optional[int] = None,
        fullscreen: bool = False,
        kiosk: bool = False,
        scale: float = 2.0,
        caption: str = "Desk Display",
        video_driver: Optional[str] = None,
    ) -> None:
        if pygame is None:  # pragma: no cover - import guard
            raise RuntimeError(
                "pygame is required for the CLI display backend"  # pragma: no cover
            ) from _PYGAME_IMPORT_ERROR

        if video_driver:
            os.environ.setdefault("SDL_VIDEODRIVER", video_driver)

        pygame.display.init()
        pygame.font.init()
        allowed = [pygame.QUIT, pygame.KEYDOWN, pygame.VIDEORESIZE]
        if hasattr(pygame, "WINDOWCLOSE"):
            allowed.append(pygame.WINDOWCLOSE)
        pygame.event.set_allowed(allowed)

        self.width = WIDTH
        self.height = HEIGHT
        self.rotation = 0
        self._fullscreen = fullscreen or kiosk
        self._kiosk = kiosk
        self._scale = max(0.1, scale)
        self._flags = pygame.DOUBLEBUF
        if self._fullscreen:
            self._flags |= pygame.FULLSCREEN
        else:
            self._flags |= pygame.RESIZABLE
        if self._kiosk:
            self._flags |= pygame.NOFRAME

        size = (int(self.width * self._scale), int(self.height * self._scale))
        if self._fullscreen:
            size = self._resolve_display_size(display_index)

        kwargs = {}
        if display_index is not None:
            kwargs["display"] = display_index

        self._surface = pygame.display.set_mode(size, self._flags, **kwargs)
        pygame.display.set_caption(caption)
        if kiosk:
            pygame.mouse.set_visible(False)
        self._current = Image.new("RGB", (self.width, self.height), "black")
        self._update_blit_rect()

    @staticmethod
    def _resolve_display_size(display_index: Optional[int]) -> tuple[int, int]:
        try:
            sizes = pygame.display.get_desktop_sizes()
        except Exception:
            sizes = []
        if sizes:
            if display_index is not None and 0 <= display_index < len(sizes):
                return sizes[display_index]
            return sizes[0]
        modes = pygame.display.list_modes(display=display_index or 0) or [(WIDTH, HEIGHT)]
        return modes[0]

    def _update_blit_rect(self) -> None:
        surface_w, surface_h = self._surface.get_size()
        aspect = self.width / self.height
        target_w = surface_w
        target_h = int(target_w / aspect)
        if target_h > surface_h:
            target_h = surface_h
            target_w = int(target_h * aspect)
        x = (surface_w - target_w) // 2
        y = (surface_h - target_h) // 2
        self._blit_rect = pygame.Rect(x, y, target_w, target_h)

    def _render(self) -> None:
        frame_bytes = self._current.tobytes()
        surface = pygame.image.frombuffer(frame_bytes, self._current.size, "RGB")
        self._surface.fill((0, 0, 0))
        if self._blit_rect.size != self._current.size:
            surface = pygame.transform.smoothscale(surface, self._blit_rect.size)
        self._surface.blit(surface, self._blit_rect.topleft)
        pygame.display.flip()

    def clear(self) -> None:
        self._current = Image.new("RGB", (self.width, self.height), "black")
        self._render()

    def image(self, pil_img: Image.Image) -> None:
        if pil_img.size != (self.width, self.height):
            pil_img = pil_img.resize((self.width, self.height), Image.ANTIALIAS)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        self._current = pil_img
        self._render()

    def show(self) -> None:
        self._render()

    def capture(self) -> Image.Image:
        return self._current.copy()

    def poll(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                return True
            if hasattr(pygame, "WINDOWCLOSE") and event.type == pygame.WINDOWCLOSE:
                return True
            if event.type == pygame.VIDEORESIZE and not self._fullscreen:
                self._surface = pygame.display.set_mode(event.size, self._flags)
                self._update_blit_rect()
                self._render()
        return False

    def close(self) -> None:
        try:
            pygame.display.quit()
        finally:
            pygame.quit()


def _list_displays(video_driver: Optional[str]) -> None:  # pragma: no cover - CLI helper
    if pygame is None:
        raise RuntimeError("pygame is required to list displays") from _PYGAME_IMPORT_ERROR
    if video_driver:
        os.environ.setdefault("SDL_VIDEODRIVER", video_driver)
    pygame.display.init()
    try:
        count = pygame.display.get_num_video_displays()
    except Exception:
        count = 1
    try:
        sizes = pygame.display.get_desktop_sizes()
    except Exception:
        sizes = []
    for idx in range(count):
        if idx < len(sizes):
            w, h = sizes[idx]
        else:
            modes = pygame.display.list_modes(display=idx) or [(WIDTH, HEIGHT)]
            w, h = modes[0]
        print(f"Display {idx}: {w}×{h}")
    pygame.display.quit()


def _extract_image(result: object, display: WindowDisplay) -> Optional[Image.Image]:
    if isinstance(result, ScreenImage):
        if result.image is not None:
            return result.image
        if result.displayed:
            try:
                return display.capture()
            except Exception:
                return None
        return None
    if isinstance(result, Image.Image):
        return result
    return None


def run_display(args: argparse.Namespace) -> int:
    def _wrap(signame: str):
        return lambda *_: request_shutdown(signame)

    for sig, name in (
        (getattr(signal, "SIGINT", None), "SIGINT"),
        (getattr(signal, "SIGTERM", None), "SIGTERM"),
    ):
        if sig is None:
            continue
        try:
            signal.signal(sig, _wrap(name))
        except Exception:  # pragma: no cover - platform dependent
            pass

    if args.list_displays:
        _list_displays(args.video_driver)
        return 0

    capture_dir = Path(args.capture_dir).expanduser().resolve() if args.capture_dir else None
    video_path = Path(args.video_path).expanduser().resolve() if args.video_path else None

    capture = FrameCapture(directory=capture_dir, video_path=video_path, fps=args.video_fps)

    display = WindowDisplay(
        display_index=args.display,
        fullscreen=args.fullscreen,
        kiosk=args.kiosk,
        scale=args.scale,
        caption="Desk Display",
        video_driver=args.video_driver,
    )

    cache_manager = CacheManager(args.refresh_interval)
    logos = build_logo_map()
    scheduler_driver = SchedulerDriver(Path(args.schedule))

    travel_state: Optional[str] = None
    delay = max(0.1, float(args.delay))

    try:
        while not _shutdown_event.is_set():
            if display.poll():
                request_shutdown("window closed")
                break

            scheduler_driver.refresh()
            scheduler = scheduler_driver.scheduler
            if scheduler is None:
                logging.warning("No schedule available—waiting %.1f seconds.", delay)
                if _shutdown_event.wait(delay):
                    break
                continue

            context = ScreenContext(
                display=display,
                cache=cache_manager.snapshot(),
                logos=logos,
                image_dir=str(IMAGES_DIR),
                travel_requested="travel" in scheduler_driver.requested_ids,
                travel_active=is_travel_screen_active(),
                travel_window=get_travel_active_window(),
                previous_travel_state=travel_state,
                now=_dt.datetime.now(CENTRAL_TIME),
            )

            registry, metadata = build_screen_registry(context)
            travel_state = metadata.get("travel_state", travel_state)

            entry = scheduler.next_available(registry)
            if entry is None:
                logging.info("No eligible screens—waiting %.1f seconds.", delay)
                if _shutdown_event.wait(delay):
                    break
                continue

            logging.info("🎬 Presenting '%s'", entry.id)

            try:
                result = entry.render()
            except Exception as exc:
                logging.error("Error rendering '%s': %s", entry.id, exc)
                if _shutdown_event.wait(delay):
                    break
                continue

            image = _extract_image(result, display)
            if image is None:
                logging.info("Screen '%s' returned no image.", entry.id)
                if _shutdown_event.wait(delay):
                    break
                continue

            already_displayed = isinstance(result, ScreenImage) and result.displayed
            if already_displayed:
                display.show()
            else:
                animate_fade_in(display, image, steps=8, delay=0.015)

            capture.handle(entry.id, image)

            if _shutdown_event.wait(delay):
                break

    finally:
        capture.close()
        display.close()

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", default=str(CONFIG_PATH), help="Path to screens_config.json")
    parser.add_argument("--display", type=int, default=None, help="SDL display index to target")
    parser.add_argument("--video-driver", default=None, help="Force a specific SDL video driver (e.g. kmsdrm, x11)")
    parser.add_argument("--fullscreen", action="store_true", help="Launch fullscreen")
    parser.add_argument("--kiosk", action="store_true", help="Fullscreen without window borders")
    parser.add_argument("--scale", type=float, default=2.0, help="Scale factor for windowed mode")
    parser.add_argument("--capture-dir", help="Save each frame as a PNG under this directory")
    parser.add_argument("--video-path", help="Record frames to an MP4 at this path")
    parser.add_argument("--video-fps", type=int, default=VIDEO_FPS, help="Frame rate for video capture")
    parser.add_argument("--delay", type=float, default=SCREEN_DELAY, help="Seconds between screens")
    parser.add_argument("--refresh-interval", type=float, default=SCHEDULE_UPDATE_INTERVAL, help="Cache refresh cadence in seconds")
    parser.add_argument("--list-displays", action="store_true", help="List available displays and exit")
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, ...)")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    if _PYGAME_IMPORT_ERROR is not None:
        parser.error(f"pygame is required for CLI rendering: {_PYGAME_IMPORT_ERROR}")

    try:
        return run_display(args)
    except KeyboardInterrupt:
        request_shutdown("KeyboardInterrupt")
        return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

