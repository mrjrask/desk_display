#!/usr/bin/env python3
"""
utils.py

Core utilities for the desk display project:
- Display wrapper
- Drawing helpers
- Animations
- Text wrapping/centering
- Team/MLB helpers
- GitHub update checker
"""
import datetime
import errno
import html
import logging
import math
import os
import pwd
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import functools
import PIL.ImageDraw as _ID
import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

# ─── Pillow compatibility shim ─────────────────────────────────────────────
# Re-add ImageDraw.textsize if missing (Pillow ≥10 compatibility)
if not hasattr(_ID.ImageDraw, "textsize"):
    def _textsize(self, text, font=None, *args, **kwargs):
        bbox = self.textbbox((0, 0), text, font=font)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])
    _ID.ImageDraw.textsize = _textsize
# Compatibility for ANTIALIAS (Pillow ≥11)
try:
    Image.ANTIALIAS
except AttributeError:
    Image.ANTIALIAS = Image.Resampling.LANCZOS

# Display HAT Mini driver (optional at import time)
try:  # pragma: no cover - hardware import
    from displayhatmini import DisplayHATMini  # type: ignore
except (ImportError, RuntimeError) as _displayhat_exc:  # pragma: no cover - hardware import
    DisplayHATMini = None  # type: ignore
    _DISPLAY_HAT_ERROR = _displayhat_exc
else:  # pragma: no cover - hardware import
    _DISPLAY_HAT_ERROR = None

try:  # pragma: no cover - hardware import
    from gpiozero import Button as GpioButton  # type: ignore
except Exception as _gpio_button_exc:  # pragma: no cover - hardware import
    GpioButton = None  # type: ignore
    _GPIO_BUTTON_ERROR = _gpio_button_exc
else:  # pragma: no cover - hardware import
    _GPIO_BUTTON_ERROR = None

RPiGPIO = None  # type: ignore
_RPI_GPIO_ERROR: Optional[Exception] = None

try:  # pragma: no cover - hardware import
    import RPi.GPIO as RPiGPIO  # type: ignore
except Exception as _rpi_gpio_exc:  # pragma: no cover - hardware import
    RPiGPIO = None  # type: ignore
    _RPI_GPIO_ERROR = _rpi_gpio_exc
else:  # pragma: no cover - hardware import
    _RPI_GPIO_ERROR = None

try:  # pragma: no cover - hardware import
    import board  # type: ignore
    import digitalio  # type: ignore
    from adafruit_rgb_display import st7789  # type: ignore
except Exception as _minipitft_exc:  # pragma: no cover - hardware import
    board = None  # type: ignore
    digitalio = None  # type: ignore
    st7789 = None  # type: ignore
    _MINIPITFT_ERROR = _minipitft_exc
else:  # pragma: no cover - hardware import
    _MINIPITFT_ERROR = None

_FORCE_HEADLESS = os.environ.get("DESK_DISPLAY_FORCE_HEADLESS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

_DISPLAY_OUTPUT = os.environ.get("DESK_DISPLAY_OUTPUT", "auto").strip().lower()
_FRAMEBUFFER_DEVICE = os.environ.get("DISPLAY_FB_DEVICE", "/dev/fb0")
_FRAMEBUFFER_PIXEL_FORMAT = os.environ.get("DISPLAY_FB_PIXEL_FORMAT", "").strip().lower()
_FRAMEBUFFER_PIXEL_ORDER = os.environ.get("DISPLAY_FB_PIXEL_ORDER", "").strip().lower()
_PYGAME_MODULE = None
_PYGAME_ERROR: Optional[Exception] = None
_CURSOR_WIGGLE_DELAY_SECONDS = 30.0


class _RPiGPIOButton:
    """Minimal gpiozero.Button-compatible wrapper backed by RPi.GPIO."""

    def __init__(self, gpio_module, pin: int):
        self._gpio = gpio_module
        self._pin = pin
        self._gpio.setmode(self._gpio.BCM)
        self._gpio.setup(self._pin, self._gpio.IN, pull_up_down=self._gpio.PUD_UP)

    @property
    def is_pressed(self) -> bool:
        return self._gpio.input(self._pin) == self._gpio.LOW

    def close(self) -> None:
        self._gpio.cleanup(self._pin)


def _load_rpi_gpio():
    global RPiGPIO, _RPI_GPIO_ERROR
    if RPiGPIO is not None or _RPI_GPIO_ERROR is not None:  # pragma: no cover - hardware import
        return RPiGPIO
    try:  # pragma: no cover - hardware import
        import RPi.GPIO as _RPiGPIO  # type: ignore
    except Exception as exc:  # pragma: no cover - hardware import
        _RPI_GPIO_ERROR = exc
        return None
    RPiGPIO = _RPiGPIO  # type: ignore
    return RPiGPIO


def _maybe_configure_desktop_env() -> None:
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return

    uid = os.getuid()
    user = os.environ.get("DESK_DISPLAY_SESSION_USER")
    if not user:
        user = os.environ.get("SUDO_USER")
    if not user:
        try:
            user = pwd.getpwuid(uid).pw_name
        except KeyError:
            user = None
    if user:
        try:
            uid = pwd.getpwnam(user).pw_uid
        except KeyError:
            pass

    runtime_dir = f"/run/user/{uid}"
    if not os.environ.get("XDG_RUNTIME_DIR") and Path(runtime_dir).is_dir():
        os.environ["XDG_RUNTIME_DIR"] = runtime_dir

    if user:
        try:
            result = subprocess.run(
                ["loginctl", "show-user", user, "-p", "Sessions", "--value"],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            result = None
        if result and result.returncode == 0:
            sessions = result.stdout.strip().split()
            for session in sessions:
                active = subprocess.run(
                    ["loginctl", "show-session", session, "-p", "Active", "--value"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if active.stdout.strip() != "yes":
                    continue
                session_type = subprocess.run(
                    ["loginctl", "show-session", session, "-p", "Type", "--value"],
                    check=False,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                display = subprocess.run(
                    ["loginctl", "show-session", session, "-p", "Display", "--value"],
                    check=False,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                if session_type == "x11" and display and not os.environ.get("DISPLAY"):
                    os.environ["DISPLAY"] = display
                if session_type == "wayland":
                    wayland_socket = Path(runtime_dir) / "wayland-0"
                    if wayland_socket.is_socket() and not os.environ.get("WAYLAND_DISPLAY"):
                        os.environ["WAYLAND_DISPLAY"] = "wayland-0"
                if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
                    break

    if not os.environ.get("WAYLAND_DISPLAY"):
        wayland_socket = Path(runtime_dir) / "wayland-0"
        if wayland_socket.is_socket():
            os.environ["WAYLAND_DISPLAY"] = "wayland-0"

    if not os.environ.get("DISPLAY"):
        if Path("/tmp/.X11-unix/X0").is_socket():
            os.environ["DISPLAY"] = ":0"

    if os.environ.get("DISPLAY") and not os.environ.get("XAUTHORITY"):
        home = Path(os.path.expanduser("~"))
        xauthority = home / ".Xauthority"
        if xauthority.is_file():
            os.environ["XAUTHORITY"] = str(xauthority)

    if os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("DESK_DISPLAY_SDL_DRIVERS", "wayland,x11,kmsdrm,fbcon,directfb")
    elif os.environ.get("DISPLAY"):
        os.environ.setdefault("DESK_DISPLAY_SDL_DRIVERS", "x11,wayland,kmsdrm,fbcon,directfb")


def _load_pygame():
    global _PYGAME_MODULE, _PYGAME_ERROR

    if _PYGAME_MODULE is not None or _PYGAME_ERROR is not None:
        return _PYGAME_MODULE

    if "SDL_VIDEODRIVER" not in os.environ:
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            _maybe_configure_desktop_env()

    try:
        warnings.filterwarnings(
            "ignore",
            message=r"pkg_resources is deprecated as an API",
            category=UserWarning,
            module=r"pygame\.pkgdata",
        )
        import pygame  # type: ignore
    except Exception as exc:  # pragma: no cover - platform import
        _PYGAME_ERROR = exc
        return None

    _PYGAME_MODULE = pygame
    return _PYGAME_MODULE


def _sdl_driver_candidates() -> List[Optional[str]]:
    if os.environ.get("SDL_VIDEODRIVER"):
        return [os.environ["SDL_VIDEODRIVER"]]

    drivers_env = os.environ.get("DESK_DISPLAY_SDL_DRIVERS", "")
    if drivers_env:
        drivers = [driver.strip() for driver in drivers_env.split(",") if driver.strip()]
        if drivers:
            return drivers

    candidates: List[Optional[str]] = [None]
    if os.environ.get("WAYLAND_DISPLAY"):
        candidates.append("wayland")
    if os.environ.get("DISPLAY"):
        candidates.append("x11")
    candidates.extend(["kmsdrm", "fbcon", "directfb"])
    return list(dict.fromkeys(candidates))


def _read_sysfs_value(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _parse_virtual_size(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if not value:
        return None
    try:
        width_str, height_str = value.split(",", 1)
        return int(width_str), int(height_str)
    except (ValueError, AttributeError):
        return None


def _parse_mode_size(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if not value:
        return None
    match = re.search(r"(\d+)\s*x\s*(\d+)", value)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _read_framebuffer_mode_size(device_path: str) -> Optional[Tuple[int, int]]:
    fb_name = Path(device_path).name
    sysfs_base = Path("/sys/class/graphics") / fb_name
    mode_value = _parse_mode_size(_read_sysfs_value(str(sysfs_base / "mode")))
    if mode_value:
        return mode_value
    modes_raw = _read_sysfs_value(str(sysfs_base / "modes"))
    if not modes_raw:
        return None
    first_line = modes_raw.splitlines()[0]
    return _parse_mode_size(first_line)


def _read_framebuffer_fbset_size(device_path: str) -> Optional[Tuple[int, int]]:
    try:
        result = subprocess.run(
            ["fbset", "-fb", device_path, "-s"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    for line in output.splitlines():
        if "geometry" not in line:
            continue
        match = re.search(r"geometry\s+(\d+)\s+(\d+)", line)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _detect_framebuffer_device(preferred_device: str, requested_size: Tuple[int, int]) -> str:
    candidates: List[str] = []
    if preferred_device:
        candidates.append(preferred_device)

    for fb_path in sorted(Path('/dev').glob('fb*')):
        device = str(fb_path)
        if device not in candidates:
            candidates.append(device)

    fallback_device: Optional[str] = None
    for device in candidates:
        if not Path(device).exists():
            continue

        if fallback_device is None:
            fallback_device = device

        detected_size = (
            _read_framebuffer_mode_size(device)
            or _read_framebuffer_fbset_size(device)
            or _parse_virtual_size(
                _read_sysfs_value(str(Path("/sys/class/graphics") / Path(device).name / "virtual_size"))
            )
        )
        if detected_size == requested_size:
            return device

    if fallback_device is not None:
        return fallback_device

    return '/dev/fb0'


def _detect_exact_framebuffer_device(
    preferred_device: str,
    requested_size: Tuple[int, int],
) -> Optional[str]:
    """Return a framebuffer device only when it exactly matches *requested_size*."""

    candidates: List[str] = []
    if preferred_device:
        candidates.append(preferred_device)

    for fb_path in sorted(Path("/dev").glob("fb*")):
        device = str(fb_path)
        if device not in candidates:
            candidates.append(device)

    for device in candidates:
        if not Path(device).exists():
            continue

        detected_size = (
            _read_framebuffer_mode_size(device)
            or _read_framebuffer_fbset_size(device)
            or _parse_virtual_size(
                _read_sysfs_value(str(Path("/sys/class/graphics") / Path(device).name / "virtual_size"))
            )
        )
        if detected_size == requested_size:
            return device

    return None


def _init_framebuffer_output(
    *,
    requested_size: Tuple[int, int],
    configured_device: str,
) -> Optional["_FrameBufferDevice"]:
    framebuffer_device = _detect_framebuffer_device(configured_device, requested_size)
    if framebuffer_device != configured_device:
        logging.info(
            "Auto-selected framebuffer device %s (requested size %dx%d; configured %s).",
            framebuffer_device,
            requested_size[0],
            requested_size[1],
            configured_device,
        )

    framebuffer = _FrameBufferDevice(framebuffer_device)
    if framebuffer._fd is None and framebuffer_device != "/dev/fb0":
        logging.warning(
            "Failed to open framebuffer %s; retrying with /dev/fb0.",
            framebuffer_device,
        )
        framebuffer = _FrameBufferDevice("/dev/fb0")

    if framebuffer._fd is None:
        return None

    logging.info(
        "🖼️  Framebuffer initialized (%dx%d, %dbpp, %s).",
        framebuffer.width,
        framebuffer.height,
        framebuffer.bpp,
        framebuffer.device_path,
    )
    if requested_size != (framebuffer.width, framebuffer.height):
        logging.info(
            "Framebuffer size differs from render size (%dx%d); frames will be scaled.",
            requested_size[0],
            requested_size[1],
        )

    return framebuffer


def _resolve_framebuffer_info(device_path: str) -> Tuple[int, int, int, Optional[int]]:
    fb_name = Path(device_path).name
    sysfs_base = Path("/sys/class/graphics") / fb_name
    mode_size = _read_framebuffer_mode_size(device_path) or _read_framebuffer_fbset_size(device_path)
    virtual_size = _parse_virtual_size(_read_sysfs_value(str(sysfs_base / "virtual_size")))
    bpp_value = _read_sysfs_value(str(sysfs_base / "bits_per_pixel"))
    stride_value = _read_sysfs_value(str(sysfs_base / "stride"))

    size = mode_size or virtual_size
    width = size[0] if size else WIDTH
    height = size[1] if size else HEIGHT
    bpp = int(bpp_value) if bpp_value and bpp_value.isdigit() else 16
    stride = int(stride_value) if stride_value and stride_value.isdigit() else None
    return width, height, bpp, stride


def _resolve_framebuffer_sysfs_format(device_path: str) -> Optional[str]:
    fb_name = Path(device_path).name
    sysfs_base = Path("/sys/class/graphics") / fb_name
    return _read_sysfs_value(str(sysfs_base / "format"))


def _normalize_display_output(value: str) -> str:
    value = value.strip().lower()
    if value in {"displayhatmini", "display-hat-mini", "hatmini", "hat"}:
        return "displayhatmini"
    if value in {"minipitft", "mini-pitft", "adafruit-minipitft", "pitft"}:
        return "minipitft"
    if value in {"framebuffer", "fb", "framebuffer-device"}:
        return "framebuffer"
    if value in {"kernel", "kms", "drm", "sdl", "fullscreen"}:
        return "kernel"
    if value in {"window", "windowed", "macos_window", "mac-window", "sdl-window"}:
        return "window"
    if value in {"headless", "none", "off"}:
        return "headless"
    if value in {"auto", ""}:
        return "auto"
    return value


def _framebuffer_pixel_format(bpp: int) -> str:
    if _FRAMEBUFFER_PIXEL_FORMAT:
        return _FRAMEBUFFER_PIXEL_FORMAT
    if bpp == 16:
        return "rgb565"
    if bpp == 24:
        return "rgb888"
    return "xrgb8888"


def _framebuffer_pixel_order(fmt: str) -> str:
    if _FRAMEBUFFER_PIXEL_ORDER in {"rgb", "bgr"}:
        return _FRAMEBUFFER_PIXEL_ORDER
    if fmt in {"bgr565", "bgr888", "xbgr8888", "abgr8888"}:
        return "bgr"
    return "rgb"


def _infer_pixel_order(format_value: str) -> Optional[str]:
    value = format_value.strip().lower().replace(" ", "").replace("_", "")
    if "bgr" in value or "b8g8r8" in value or "b5g6r5" in value:
        return "bgr"
    if "rgb" in value or "r8g8b8" in value or "r5g6b5" in value:
        return "rgb"
    return None


def _infer_pixel_format(format_value: str, *, order: Optional[str] = None) -> Optional[str]:
    value = format_value.strip().lower().replace(" ", "").replace("_", "")
    resolved_order = order if order in {"rgb", "bgr"} else "rgb"
    if "565" in value:
        return f"{resolved_order}565"
    if "8888" in value or "x8" in value or "a8" in value:
        if "a8" in value:
            return "abgr8888" if resolved_order == "bgr" else "argb8888"
        return "xbgr8888" if resolved_order == "bgr" else "xrgb8888"
    if "888" in value:
        return "bgr888" if resolved_order == "bgr" else "rgb888"
    return None


def _resolve_framebuffer_pixel_settings(
    device_path: str, bpp: int
) -> Tuple[str, str]:
    fmt = _framebuffer_pixel_format(bpp)
    order = _framebuffer_pixel_order(fmt)
    sysfs_format = _resolve_framebuffer_sysfs_format(device_path)
    if sysfs_format:
        inferred_order = _infer_pixel_order(sysfs_format)
        if inferred_order:
            order = inferred_order
        inferred_format = _infer_pixel_format(sysfs_format, order=order)
        if inferred_format:
            fmt = inferred_format
    if _FRAMEBUFFER_PIXEL_FORMAT:
        fmt = _FRAMEBUFFER_PIXEL_FORMAT
    if _FRAMEBUFFER_PIXEL_ORDER in {"rgb", "bgr"}:
        order = _FRAMEBUFFER_PIXEL_ORDER
    return fmt, order


def _disable_framebuffer_cursor() -> None:
    for path in ("/sys/class/graphics/fbcon/cursor_blink", "/sys/class/graphics/fbcon/cursor"):
        try:
            Path(path).write_text("0", encoding="utf-8")
        except OSError:
            continue


def _convert_rgb565(image: Image.Image, *, order: str = "rgb") -> bytes:
    import numpy as np

    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    if order == "bgr":
        rgb = rgb[..., ::-1]
    r = rgb[..., 0].astype(np.uint16)
    g = rgb[..., 1].astype(np.uint16)
    b = rgb[..., 2].astype(np.uint16)
    rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
    return rgb565.astype("<u2").tobytes()


def _convert_rgb888(image: Image.Image, *, order: str = "rgb") -> bytes:
    image = image.convert("RGB")
    raw_mode = "BGR" if order == "bgr" else "RGB"
    return image.tobytes("raw", raw_mode)


def _convert_rgbx8888(image: Image.Image, *, order: str = "rgb") -> bytes:
    image = image.convert("RGB")
    raw_mode = "BGRX" if order == "bgr" else "RGBX"
    return image.tobytes("raw", raw_mode)


def _convert_argb8888(image: Image.Image, *, order: str = "rgb") -> bytes:
    import numpy as np

    rgba = np.array(image.convert("RGBA"), dtype=np.uint8)
    alpha = np.full(rgba.shape[:2] + (1,), 255, dtype=np.uint8)
    rgb = rgba[..., :3]
    if order == "bgr":
        rgb = rgb[..., ::-1]
    argb = np.concatenate([alpha, rgb], axis=2)
    return argb.tobytes()


class _FrameBufferDevice:
    def __init__(self, device_path: str):
        self.device_path = device_path
        self.width, self.height, self.bpp, self.stride = _resolve_framebuffer_info(device_path)
        self.bytes_per_pixel = max(1, self.bpp // 8)
        self.pixel_format, self.pixel_order = _resolve_framebuffer_pixel_settings(device_path, self.bpp)
        self._fd: Optional[int] = None

        try:
            self._fd = os.open(self.device_path, os.O_RDWR)
        except OSError as exc:
            logging.warning("Failed to open framebuffer device %s: %s", self.device_path, exc)
            self._fd = None
        else:
            _disable_framebuffer_cursor()

    def close(self) -> None:
        if self._fd is None:
            return
        try:
            os.close(self._fd)
        except OSError:
            pass
        self._fd = None

    def _convert_image(self, image: Image.Image) -> bytes:
        fmt = self.pixel_format
        if fmt in {"rgb565", "bgr565"}:
            return _convert_rgb565(image, order=self.pixel_order)
        if fmt in {"rgb888", "bgr888"}:
            return _convert_rgb888(image, order=self.pixel_order)
        if fmt in {"argb8888", "abgr8888"}:
            return _convert_argb8888(image, order=self.pixel_order)
        return _convert_rgbx8888(image, order=self.pixel_order)

    def write_image(self, image: Image.Image) -> None:
        if self._fd is None:
            return
        if image.size != (self.width, self.height):
            scale = min(
                self.width / image.width if image.width else 1.0,
                self.height / image.height if image.height else 1.0,
            )
            target_w = max(1, int(round(image.width * scale)))
            target_h = max(1, int(round(image.height * scale)))
            resized = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (self.width, self.height), "black")
            offset_x = (self.width - target_w) // 2
            offset_y = (self.height - target_h) // 2
            canvas.paste(resized, (offset_x, offset_y))
            image = canvas

        raw = self._convert_image(image)
        row_bytes = self.width * self.bytes_per_pixel
        stride = self.stride or row_bytes

        try:
            os.lseek(self._fd, 0, os.SEEK_SET)
            if stride <= row_bytes:
                os.write(self._fd, raw)
                return

            padded = bytearray(stride * self.height)
            for row in range(self.height):
                start_src = row * row_bytes
                start_dst = row * stride
                padded[start_dst:start_dst + row_bytes] = raw[start_src:start_src + row_bytes]
            os.write(self._fd, padded)
        except OSError as exc:
            logging.warning("Failed to write framebuffer: %s", exc)


class _KernelDisplay:
    def __init__(self, width: int, height: int, *, window_mode: bool = False):
        self.render_width = width
        self.render_height = height
        self.window_mode = bool(window_mode)
        self._pygame = _load_pygame()
        if self._pygame is None:
            raise RuntimeError(f"pygame not available: {_PYGAME_ERROR}")

        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            _maybe_configure_desktop_env()

        fullscreen_default = "0" if self.window_mode else "1"
        self._fullscreen = os.environ.get(
            "DESK_DISPLAY_SDL_FULLSCREEN",
            fullscreen_default,
        ).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        try:
            self._window_scale = max(
                1.0,
                float(
                    os.environ.get(
                        "DESK_DISPLAY_WINDOW_SCALE",
                        "1.0" if self.window_mode else "1.0",
                    )
                ),
            )
        except (TypeError, ValueError):
            self._window_scale = 1.0 if self.window_mode else 1.0
        self._window_resizable = os.environ.get(
            "DESK_DISPLAY_WINDOW_RESIZABLE",
            "1",
        ).strip().lower() not in {"0", "false", "no", "off"}

        self._sdl_driver: Optional[str] = None
        self._screen = self._init_display_surface()
        self.screen_width, self.screen_height = self._screen.get_size()
        self._scale_to_screen = (self.screen_width, self.screen_height) != (
            self.render_width,
            self.render_height,
        )
        self._pygame.display.set_caption("Desk Display")
        try:
            self._pygame.mouse.set_visible(False)
            _park_mouse_cursor(self._pygame)
            _wiggle_mouse_cursor(self._pygame)
            _schedule_mouse_cursor_wiggle(
                self._pygame,
                delay_seconds=_CURSOR_WIGGLE_DELAY_SECONDS,
                repeat=True,
            )
        except Exception:  # pragma: no cover - optional behavior
            pass

    def _drain_window_resize_events(self) -> None:
        """Drop queued resize/window events to avoid backlog while dragging."""

        event_get = getattr(self._pygame.event, "get", None)
        if event_get is None:
            return

        resize_event_types: list[int] = []
        for event_name in (
            "WINDOWRESIZED",
            "WINDOWSIZECHANGED",
            "VIDEORESIZE",
            "WINDOWEVENT",
        ):
            event_type = getattr(self._pygame, event_name, None)
            if isinstance(event_type, int):
                resize_event_types.append(event_type)
        if not resize_event_types:
            return

        try:
            event_get(resize_event_types)
        except Exception:
            return

    def _scale_surface_to_target(self, surface: Any, target_size: tuple[int, int]) -> Any:
        """Choose the lowest-latency scaler for the current platform/mode."""

        transform = getattr(self._pygame, "transform", None)
        if transform is None:
            return surface

        # AppKit can emit dense resize traffic while dragging macOS windows.
        # Prefer the faster nearest-neighbor path during windowed rendering to
        # keep resize interactions responsive.
        if self.window_mode and sys.platform == "darwin":
            fast_scale = getattr(transform, "scale", None)
            if callable(fast_scale):
                return fast_scale(surface, target_size)

        smoothscale = getattr(transform, "smoothscale", None)
        if callable(smoothscale):
            return smoothscale(surface, target_size)

        fast_scale = getattr(transform, "scale", None)
        if callable(fast_scale):
            return fast_scale(surface, target_size)
        return surface

    def _init_display_surface(self):
        if self._fullscreen:
            flags = self._pygame.FULLSCREEN | self._pygame.SCALED
            requested_size = (0, 0)
        else:
            # Keep windowed mode fully user-resizable (including below the
            # native render size). SDL's SCALED flag can enforce a minimum
            # window size that blocks shrinking on desktop platforms.
            flags = 0
            if self._window_resizable:
                flags |= self._pygame.RESIZABLE
            requested_size = (
                max(1, int(round(self.render_width * self._window_scale))),
                max(1, int(round(self.render_height * self._window_scale))),
            )
        errors: List[str] = []
        for driver in _sdl_driver_candidates():
            if driver:
                os.environ["SDL_VIDEODRIVER"] = driver
            else:
                os.environ.pop("SDL_VIDEODRIVER", None)
            try:
                self._pygame.display.quit()
            except Exception:
                pass
            try:
                self._pygame.display.init()
                try:
                    screen = self._pygame.display.set_mode(requested_size, flags)
                except Exception as exc:
                    message = str(exc)
                    if "0 sized" in message or "0-sized" in message:
                        screen = self._pygame.display.set_mode(
                            (self.render_width, self.render_height),
                            flags,
                        )
                    else:
                        raise
            except Exception as exc:
                errors.append(f"{driver or 'default'}: {exc}")
                continue
            self._sdl_driver = driver
            return screen
        raise RuntimeError("Failed to initialize SDL display: " + "; ".join(errors))

    def write_image(self, image: Image.Image) -> None:
        if (
            sys.platform == "darwin"
            and threading.current_thread() is not threading.main_thread()
        ):
            # Cocoa-backed SDL must only be touched from the process main
            # thread on macOS. Background render workers can still prepare PIL
            # frames, but they must not blit/flip/pump pygame events.
            return
        if image.mode != "RGB":
            image = image.convert("RGB")
        surface = self._pygame.image.frombuffer(
            image.tobytes(), image.size, "RGB"
        )
        current_size = self._screen.get_size()
        if current_size != (self.screen_width, self.screen_height):
            self.screen_width, self.screen_height = current_size
            self._scale_to_screen = current_size != (
                self.render_width,
                self.render_height,
            )
        if self._scale_to_screen:
            scale_factor = min(
                self.screen_width / self.render_width,
                self.screen_height / self.render_height,
            )
            target_size = (
                max(1, int(round(self.render_width * scale_factor))),
                max(1, int(round(self.render_height * scale_factor))),
            )
            surface = self._scale_surface_to_target(surface, target_size)
            self._screen.fill((0, 0, 0))
            offset = (
                (self.screen_width - target_size[0]) // 2,
                (self.screen_height - target_size[1]) // 2,
            )
            self._screen.blit(surface, offset)
        else:
            self._screen.blit(surface, (0, 0))
        self._pygame.display.flip()
        self._drain_window_resize_events()
        self._pygame.event.pump()


def _wiggle_mouse_cursor(pygame_module: Any, *, distance: int = 1) -> None:
    """Nudge the pointer slightly so desktop cursor-hide daemons can engage."""

    if distance <= 0:
        return

    try:
        x, y = pygame_module.mouse.get_pos()
        pygame_module.mouse.set_pos((x + distance, y))
        pygame_module.event.pump()
        pygame_module.mouse.set_pos((x, y))
        pygame_module.event.pump()
    except Exception:
        return


def _park_mouse_cursor(pygame_module: Any) -> None:
    """Move the pointer to the bottom-right corner of the active display."""

    try:
        surface = pygame_module.display.get_surface()
        if surface is None:
            return
        width, height = surface.get_size()
        pygame_module.mouse.set_pos((max(0, width - 1), max(0, height - 1)))
        pygame_module.event.pump()
    except Exception:
        return


def _schedule_mouse_cursor_wiggle(
    pygame_module: Any,
    *,
    delay_seconds: float,
    distance: int = 1,
    repeat: bool = False,
) -> Optional[threading.Timer]:
    """Schedule a cursor wiggle after startup, optionally repeating."""

    if delay_seconds <= 0:
        return None
    if sys.platform == "darwin":
        # AppKit event processing must run on the main thread on macOS.
        # threading.Timer callbacks can invoke pygame event pumping from a
        # worker thread, which triggers NSInternalInconsistencyException.
        return None

    callback: Callable[[], None]
    if repeat:
        def _repeat() -> None:
            _wiggle_mouse_cursor(pygame_module, distance=distance)
            _schedule_mouse_cursor_wiggle(
                pygame_module,
                delay_seconds=delay_seconds,
                distance=distance,
                repeat=True,
            )

        callback = _repeat
    else:
        callback = functools.partial(
            _wiggle_mouse_cursor,
            pygame_module=pygame_module,
            distance=distance,
        )

    timer = threading.Timer(
        delay_seconds,
        callback,
    )
    timer.daemon = True
    timer.start()
    return timer

_ACTIVE_DISPLAY: Optional["Display"] = None
_LED_INDICATOR_ANIMATOR: Optional["_LedAnimator"] = None


@dataclass
class _UpdateStatus:
    github: bool = False
    apt: bool = False


_UPDATE_STATUS = _UpdateStatus()
_UPDATE_INDICATOR_ENABLED = True

_DISPLAY_UPDATE_GATE = threading.Event()
_DISPLAY_UPDATE_GATE.set()
_DEFER_CLEAR_DISPLAY = threading.Event()


def get_update_status() -> _UpdateStatus:
    """Return the last known update status for GitHub and apt."""

    return _UPDATE_STATUS


def update_indicator_enabled() -> bool:
    """Return whether update indicator LED/border output is enabled."""

    return _UPDATE_INDICATOR_ENABLED


def suspend_display_updates() -> None:
    """Prevent subsequent display updates from reaching the hardware."""

    _DISPLAY_UPDATE_GATE.clear()


def resume_display_updates() -> None:
    """Allow display updates to be pushed to the hardware again."""

    _DISPLAY_UPDATE_GATE.set()


def display_updates_enabled() -> bool:
    """Return True when display updates are currently allowed."""

    return _DISPLAY_UPDATE_GATE.is_set()


@contextmanager
def defer_clear_display() -> Iterable[None]:
    """Temporarily suppress immediate clears to avoid black flashes."""

    _DEFER_CLEAR_DISPLAY.set()
    try:
        yield
    finally:
        _DEFER_CLEAR_DISPLAY.clear()

def _clamp_led_level(value: float) -> float:
    """Clamp LED channel values to normalized [0.0, 1.0] range."""

    return max(0.0, min(1.0, value))


def _normalized_led_to_driver_channel(value: float) -> int:
    """Convert a normalized LED value to the Display HAT Mini 8-bit channel range."""

    return int(round(_clamp_led_level(value) * 255))


def _get_led_indicator_level() -> float:
    """Return the normalized indicator LED level from environment config."""

    default_level = 0.08
    raw = os.environ.get("DISPLAY_HAT_MINI_LED_LEVEL")
    if raw is None:
        return default_level
    try:
        return _clamp_led_level(float(raw))
    except (TypeError, ValueError):
        logging.warning(
            "Invalid DISPLAY_HAT_MINI_LED_LEVEL %r; using default %.2f.",
            raw,
            default_level,
        )
        return default_level


LED_INDICATOR_LEVEL = _get_led_indicator_level()

# Project config
from config import (
    WIDTH,
    HEIGHT,
    CENTRAL_TIME,
    DISPLAY_ROTATION,
    EMOJI_EMBEDDED_COLOR,
    WEATHER_USE_EMOJI_ICONS,
    get_emoji_font,
    is_hyperpixel_next_layout,
    is_hyperpixel_4_square_layout,
    HYPERPIXEL_LED_INDICATOR_BORDER_ENABLED,
    HYPERPIXEL_LED_INDICATOR_BORDER_WIDTH,
    DISPLAY_HAT_MINI_LED_INDICATOR_BORDER_ENABLED,
    DISPLAY_HAT_MINI_LED_ENABLED,
    DISPLAY_HAT_MINI_REINIT_SECONDS,
    DISPLAY_FADE_IN_ENABLED,
    DISPLAY_FADE_IN_DISPLAY_HAT_MINI_STEPS,
    DISPLAY_FADE_IN_HYPERPIXEL_STEPS,
    DISPLAY_FADE_IN_HDMI_1080P_STEPS,
    DISPLAY_FADE_IN_STEPS_BY_PROFILE,
    get_display_profile_id,
)

EMOJI_DRAW_KWARGS = {"embedded_color": True} if EMOJI_EMBEDDED_COLOR else {}
# Color utilities
from screens.color_palettes import random_color
# ─── Logging decorator ──────────────────────────────────────────────────────
def log_call(func):
    """
    Decorator that logs entry & exit at DEBUG level only.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.debug(f"→ {func.__name__}()")
        result = func(*args, **kwargs)
        logging.debug(f"← {func.__name__}()")
        return result
    return wrapper


def _is_device_busy_error(exc: BaseException) -> bool:
    """Return True when an exception chain contains an EBUSY error."""

    visited = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, OSError) and current.errno == errno.EBUSY:
            return True
        current = current.__cause__ or current.__context__
    return False

# ─── Display wrapper ────────────────────────────────────────────────────────
class Display:
    """Wrapper around the Pimoroni Display HAT Mini (320×240 LCD)."""

    _BUTTON_NAMES = ("A", "B", "X", "Y")
    _BOTTOM_SAFE_BUFFER_PX = 5
    _KERNEL_BOTTOM_SAFE_BUFFER_PX = 25
    _INDICATOR_BOTTOM_SAFE_BUFFER_PX = 5
    _DISPLAY_REINIT_RETRY_SECONDS = 60

    def __init__(self):
        global _ACTIVE_DISPLAY

        self.width = WIDTH
        self.height = HEIGHT
        self.rotation = DISPLAY_ROTATION % 360
        if self.rotation not in (0, 90, 180, 270):
            logging.warning(
                "Unsupported display rotation %d°; falling back to 0°.",
                self.rotation,
            )
            self.rotation = 0
        self._buffer = Image.new("RGB", (self.width, self.height), "black")
        self._display = None
        self._framebuffer: Optional[_FrameBufferDevice] = None
        self._kernel_display: Optional[_KernelDisplay] = None
        self._button_pins: Dict[str, Optional[int]] = {name: None for name in self._BUTTON_NAMES}
        self._gpio_buttons: Dict[str, Any] = {}
        self._button_callback: Optional[Callable[[str], None]] = None
        self._backlight_level = 1.0
        self._backlight_lock = threading.Lock()
        self._skip_event: Optional[threading.Event] = None
        self._frame_id = 0
        self._frame_lock = threading.Lock()
        self._led_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._hyperpixel_indicator_border = (
            HYPERPIXEL_LED_INDICATOR_BORDER_ENABLED
            and is_hyperpixel_next_layout(self.width, self.height)
        )
        self._display_hat_mini_indicator_border = False
        self._uses_kernel_output = False
        self._display_reinit_seconds = DISPLAY_HAT_MINI_REINIT_SECONDS
        self._last_display_reinit = time.monotonic()
        self._next_display_reinit_retry = 0.0
        self._display_reinit_disabled = False
        self._display_reinit_lock = threading.Lock()
        self._display_io_lock = threading.RLock()
        self._rotation_requires_expand = self.rotation in (90, 270)
        self._frame_transform: Callable[[Image.Image], Image.Image] = lambda img: img
        self._frame_writer: Callable[[Image.Image], None] = lambda img: None
        self._output_strategy = "headless"
        self._display_driver = "none"
        self._minipitft_backlight = None

        output = _normalize_display_output(_DISPLAY_OUTPUT)
        if _FORCE_HEADLESS or output == "headless":
            reason = (
                "DESK_DISPLAY_FORCE_HEADLESS"
                if _FORCE_HEADLESS
                else "DESK_DISPLAY_OUTPUT=headless"
            )
            logging.info("Display initialization skipped; running headless (%s).", reason)
        elif output == "framebuffer":
            requested_size = (self.width, self.height)
            self._framebuffer = _init_framebuffer_output(
                requested_size=requested_size,
                configured_device=_FRAMEBUFFER_DEVICE,
            )
            if self._framebuffer is None:
                logging.warning(
                    "Framebuffer output requested but unavailable; running headless."
                )
        elif output == "auto":
            requested_size = (self.width, self.height)
            exact_framebuffer_device = _detect_exact_framebuffer_device(
                _FRAMEBUFFER_DEVICE,
                requested_size,
            )
            if exact_framebuffer_device:
                logging.info(
                    "Auto mode detected exact framebuffer match at %s; preferring framebuffer output.",
                    exact_framebuffer_device,
                )
                self._framebuffer = _init_framebuffer_output(
                    requested_size=requested_size,
                    configured_device=exact_framebuffer_device,
                )
                if self._framebuffer is None:
                    logging.warning(
                        "Exact framebuffer device %s was detected but failed to initialize; continuing auto detection.",
                        exact_framebuffer_device,
                    )
            if self._framebuffer is None and output == "auto":
                if DisplayHATMini is None:  # pragma: no cover - hardware import
                    if _DISPLAY_HAT_ERROR:
                        logging.warning(
                            "Display HAT Mini driver unavailable; running headless (%s)",
                            _DISPLAY_HAT_ERROR,
                        )
                    else:
                        logging.warning(
                            "Display HAT Mini driver unavailable; running headless."
                        )
                else:
                    try:  # pragma: no cover - hardware import
                        self._display = self._create_display_hat_mini(self._buffer)
                    except Exception as exc:  # pragma: no cover - hardware import
                        self._display = None
                        logging.warning(
                            "Failed to initialize Display HAT Mini hardware in auto mode (%s); trying framebuffer fallback.",
                            exc,
                        )
                        self._framebuffer = _init_framebuffer_output(
                            requested_size=(self.width, self.height),
                            configured_device=_FRAMEBUFFER_DEVICE,
                        )
                        if self._framebuffer is None:
                            logging.warning(
                                "Framebuffer fallback unavailable after Display HAT Mini init failure; running headless."
                            )
                    else:  # pragma: no cover - hardware import
                        self._display_driver = "displayhatmini"
                        logging.info(
                            "🖼️  Display HAT Mini initialized (%dx%d, rotation %d°).",
                            self.width,
                            self.height,
                            self.rotation,
                        )
        elif output in {"kernel", "window"}:
            self._uses_kernel_output = True
            window_mode = output == "window"
            try:
                self._kernel_display = _KernelDisplay(
                    self.width,
                    self.height,
                    window_mode=window_mode,
                )
            except Exception as exc:
                mode_label = "Windowed SDL" if window_mode else "Kernel display"
                logging.warning("%s output unavailable (%s).", mode_label, exc)
                if window_mode:
                    logging.warning(
                        "Window mode requested but SDL window initialization failed; running headless instead of framebuffer fallback."
                    )
                    self._kernel_display = None
                    self._uses_kernel_output = False
                else:
                    logging.info(
                        "Attempting framebuffer fallback after %s output failure.",
                        mode_label.lower(),
                    )
                    self._framebuffer = _init_framebuffer_output(
                        requested_size=(self.width, self.height),
                        configured_device=_FRAMEBUFFER_DEVICE,
                    )
                    if self._framebuffer is None:
                        logging.warning(
                            "Framebuffer fallback unavailable; running headless."
                        )
                    else:
                        self._uses_kernel_output = False
                self._kernel_display = None
            else:
                driver_label = self._kernel_display._sdl_driver or "default"
                if window_mode:
                    logging.info(
                        "🪟 Window display initialized (%dx%d window, SDL driver: %s).",
                        self._kernel_display.screen_width,
                        self._kernel_display.screen_height,
                        driver_label,
                    )
                else:
                    logging.info(
                        "🖥️  Kernel display initialized (%dx%d fullscreen, SDL driver: %s).",
                        self._kernel_display.screen_width,
                        self._kernel_display.screen_height,
                        driver_label,
                    )
                if (self.width, self.height) != (
                    self._kernel_display.screen_width,
                    self._kernel_display.screen_height,
                ):
                    logging.info(
                        "Kernel display size differs from render size (%dx%d); frames will be scaled.",
                        self.width,
                        self.height,
                    )
        elif output == "minipitft":
            if st7789 is None or board is None or digitalio is None:  # pragma: no cover - hardware import
                if _MINIPITFT_ERROR:
                    logging.warning(
                        "Adafruit miniPiTFT driver unavailable; running headless (%s)",
                        _MINIPITFT_ERROR,
                    )
                else:
                    logging.warning(
                        "Adafruit miniPiTFT driver unavailable; running headless."
                    )
            else:
                try:  # pragma: no cover - hardware import
                    self._display = self._create_adafruit_minipitft(self._buffer)
                except Exception as exc:  # pragma: no cover - hardware import
                    self._display = None
                    logging.warning(
                        "Failed to initialize Adafruit miniPiTFT hardware; running headless (%s)",
                        exc,
                    )
                else:  # pragma: no cover - hardware import
                    self._display_driver = "minipitft"
                    logging.info(
                        "🖼️  Adafruit miniPiTFT initialized (%dx%d, rotation %d°).",
                        self.width,
                        self.height,
                        self.rotation,
                    )
        elif DisplayHATMini is None:  # pragma: no cover - hardware import
            if output == "displayhatmini":
                if _DISPLAY_HAT_ERROR:
                    logging.warning(
                        "Display HAT Mini driver unavailable; running headless (%s)",
                        _DISPLAY_HAT_ERROR,
                    )
                else:
                    logging.warning(
                        "Display HAT Mini driver unavailable; running headless."
                    )
        else:
            try:  # pragma: no cover - hardware import
                self._display = self._create_display_hat_mini(self._buffer)
            except Exception as exc:  # pragma: no cover - hardware import
                self._display = None
                if output == "displayhatmini":
                    logging.warning(
                        "Failed to initialize Display HAT Mini hardware; running headless (%s)",
                        exc,
                    )
            else:  # pragma: no cover - hardware import
                self._display_driver = "displayhatmini"
                logging.info(
                    "🖼️  Display HAT Mini initialized (%dx%d, rotation %d°).",
                    self.width,
                    self.height,
                    self.rotation,
                )

        self._configure_output_strategy()
        self._initialize_gpio_buttons()

        _ACTIVE_DISPLAY = self

    def _configure_output_strategy(self) -> None:
        """Select frame transform/output handlers once during initialization."""

        display_hat_indicator_eligible = (
            DISPLAY_HAT_MINI_LED_INDICATOR_BORDER_ENABLED
            and (self.width, self.height) in {(320, 240), (240, 320)}
        )

        if self._framebuffer is not None:
            self._frame_transform = self._build_rotation_transform()
            self._frame_writer = self._framebuffer.write_image
            self._output_strategy = "framebuffer"
            self._display_hat_mini_indicator_border = False
            return

        if self._kernel_display is not None:
            self._frame_transform = self._build_rotation_transform()
            self._frame_writer = self._kernel_display.write_image
            self._output_strategy = "kernel"
            self._display_hat_mini_indicator_border = False
            return

        if self._display is not None:
            self._frame_transform = self._build_rotation_transform(
                resize_to_display=True,
            )
            if self._display_driver == "minipitft":
                self._frame_writer = self._write_adafruit_minipitft_frame
                self._output_strategy = "minipitft"
                self._display_hat_mini_indicator_border = False
            else:
                self._frame_writer = self._write_display_hat_mini_frame
                self._output_strategy = "display_hat_mini"
                self._display_hat_mini_indicator_border = display_hat_indicator_eligible
            return

        self._frame_transform = lambda img: img
        self._frame_writer = lambda img: None
        self._output_strategy = "headless"
        self._display_hat_mini_indicator_border = False

    def _build_rotation_transform(
        self,
        *,
        resize_to_display: bool = False,
    ) -> Callable[[Image.Image], Image.Image]:
        """Precompute the image transform strategy used by ``_update_display``."""

        if not self.rotation and not resize_to_display:
            return lambda img: img

        def _transform(img: Image.Image) -> Image.Image:
            transformed = img
            if self.rotation:
                transformed = transformed.rotate(
                    self.rotation,
                    expand=self._rotation_requires_expand,
                )
            if resize_to_display and transformed.size != (self.width, self.height):
                transformed = transformed.resize((self.width, self.height), Image.ANTIALIAS)
            return transformed

        return _transform

    def register_skip_event(self, event: Optional[threading.Event]) -> None:
        """Associate a skip event so long-running screens can bail out early."""

        self._skip_event = event

    def skip_requested(self) -> bool:
        """Return True when a registered skip event is active."""

        return bool(self._skip_event and self._skip_event.is_set())

    def wait_for_skip(self, timeout: float, *, poll_interval: float = 0.05) -> bool:
        """Sleep up to *timeout* seconds, returning True if a skip is requested."""

        if not self._skip_event:
            time.sleep(timeout)
            return False

        end = time.monotonic() + timeout
        while True:
            if self._skip_event.is_set():
                return True

            remaining = end - time.monotonic()
            if remaining <= 0:
                break

            time.sleep(min(poll_interval, remaining))

        return False

    def _update_display(self):
        if not display_updates_enabled():
            return
        if self._output_strategy == "headless" and self._display is not None:
            self._configure_output_strategy()
        buffer_to_display = self._frame_transform(self._indicator_buffer())
        self._frame_writer(buffer_to_display)

    def _write_display_hat_mini_frame(self, buffer_to_display: Image.Image) -> None:
        """Push a frame to Display HAT Mini, including hot reinitialization logic."""

        if self._display is None:  # pragma: no cover - hardware import
            return

        with self._display_io_lock:
            try:
                self._maybe_reinitialize_display_hat_mini()
                if self._display is None:
                    return
                self._display.buffer = buffer_to_display
                self._display.display()
            except Exception as exc:  # pragma: no cover - hardware import
                logging.warning("Display refresh failed: %s", exc)

    def _write_adafruit_minipitft_frame(self, buffer_to_display: Image.Image) -> None:
        """Push a frame to Adafruit miniPiTFT."""

        if self._display is None:  # pragma: no cover - hardware import
            return

        with self._display_io_lock:
            try:
                self._display.image(buffer_to_display)
            except Exception as exc:  # pragma: no cover - hardware import
                logging.warning("miniPiTFT refresh failed: %s", exc)

    def _button_gpio_pin(self, name: str) -> Optional[int]:
        raw_value = os.environ.get(f"BUTTON_{name}")
        if raw_value is None:
            return None
        raw_value = raw_value.strip()
        if not raw_value:
            return None
        try:
            return int(raw_value, 0)
        except ValueError:
            logging.warning("Invalid BUTTON_%s pin '%s'; ignoring.", name, raw_value)
            return None

    def _initialize_gpio_buttons(self) -> None:
        configured_button_names = [
            name for name in self._BUTTON_NAMES if os.environ.get(f"BUTTON_{name}")
        ]
        if not configured_button_names:
            return

        button_class = GpioButton
        using_rpi_gpio_fallback = False
        if button_class is None:  # pragma: no cover - hardware import
            fallback_enabled = os.environ.get(
                "DESK_DISPLAY_RPI_GPIO_FALLBACK",
                "1",
            ).strip().lower() not in {"0", "false", "no", "off"}
            rpi_gpio = _load_rpi_gpio() if fallback_enabled else None
            if rpi_gpio is not None:
                button_class = functools.partial(_RPiGPIOButton, rpi_gpio)
                using_rpi_gpio_fallback = True
                logging.warning(
                    "gpiozero unavailable (%s); using RPi.GPIO fallback for hardware buttons.",
                    _GPIO_BUTTON_ERROR,
                )
            else:
                logging.warning(
                    "Hardware button pins configured but no GPIO input backend is available "
                    "(gpiozero error: %s; RPi.GPIO error: %s).",
                    _GPIO_BUTTON_ERROR,
                    _RPI_GPIO_ERROR,
                )
                return

        if using_rpi_gpio_fallback and hasattr(RPiGPIO, "setwarnings"):  # pragma: no cover - hardware import
            RPiGPIO.setwarnings(False)

        for name in self._BUTTON_NAMES:
            pin = self._button_gpio_pin(name)
            if pin is None:
                continue
            try:  # pragma: no cover - hardware import
                if using_rpi_gpio_fallback:
                    button = button_class(pin)
                else:
                    button = button_class(pin, pull_up=True, bounce_time=0.05)
            except Exception as exc:
                logging.warning("Failed to initialize BUTTON_%s on GPIO%d: %s", name, pin, exc)
                continue

            self._gpio_buttons[name] = button
            self._button_pins[name] = pin
            logging.info("Configured BUTTON_%s on GPIO%d", name, pin)

    def _create_display_hat_mini(self, initial_buffer: Image.Image):
        """Create and configure a Display HAT Mini driver instance."""

        display = DisplayHATMini(initial_buffer)
        display_module = None
        try:
            display_module = __import__(display.__class__.__module__, fromlist=["_"])
        except Exception:
            display_module = None

        for name in self._BUTTON_NAMES:
            pin_name = f"BUTTON_{name}"
            pin = getattr(display, pin_name, None)
            if pin is None:
                pin = getattr(display.__class__, pin_name, None)
            if pin is None and display_module is not None:
                pin = getattr(display_module, pin_name, None)
            self._button_pins[name] = pin

        if hasattr(display, "on_button_pressed"):
            try:
                display.on_button_pressed(self._handle_hw_button_event)
            except Exception as exc:  # pragma: no cover - hardware import
                logging.debug("Failed to register hardware button callback: %s", exc)

        return display

    def _create_adafruit_minipitft(self, _initial_buffer: Image.Image):
        """Create and configure an Adafruit miniPiTFT 1.14 display driver instance."""

        spi = board.SPI()
        cs = digitalio.DigitalInOut(getattr(board, "CE0"))
        dc = digitalio.DigitalInOut(getattr(board, "D25"))
        rst = digitalio.DigitalInOut(getattr(board, "D27"))
        backlight = digitalio.DigitalInOut(getattr(board, "D22"))
        backlight.switch_to_output(value=True)

        baudrate = int(os.environ.get("MINIPITFT_BAUDRATE", "64000000"))
        rotation = int(os.environ.get("MINIPITFT_DRIVER_ROTATION", "90"))
        width = int(os.environ.get("MINIPITFT_DRIVER_WIDTH", "135"))
        height = int(os.environ.get("MINIPITFT_DRIVER_HEIGHT", "240"))
        x_offset = int(os.environ.get("MINIPITFT_X_OFFSET", "53"))
        y_offset = int(os.environ.get("MINIPITFT_Y_OFFSET", "40"))

        display = st7789.ST7789(
            spi,
            cs=cs,
            dc=dc,
            rst=rst,
            baudrate=baudrate,
            width=width,
            height=height,
            x_offset=x_offset,
            y_offset=y_offset,
            rotation=rotation,
        )
        self._minipitft_backlight = backlight
        return display

    def _release_display_hat_mini(self, display, *, call_destructor: bool = True) -> None:
        """Best-effort cleanup for a Display HAT Mini driver instance."""

        if display is None:
            return

        for pwm_name in ("led_r_pwm", "led_g_pwm", "led_b_pwm", "backlight_pwm"):
            pwm = getattr(display, pwm_name, None)
            stop = getattr(pwm, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception as exc:  # pragma: no cover - hardware import
                    logging.debug("Failed to stop %s during display cleanup: %s", pwm_name, exc)

        module = None
        gpio = None
        try:
            module = __import__(display.__class__.__module__, fromlist=["GPIO"])
            gpio = getattr(module, "GPIO", None)
        except Exception:  # pragma: no cover - hardware import
            gpio = None

        if gpio is not None:
            for pin_name in ("BUTTON_A", "BUTTON_B", "BUTTON_X", "BUTTON_Y"):
                pin = getattr(display, pin_name, None)
                if pin is None:
                    continue
                try:
                    gpio.remove_event_detect(pin)
                except Exception:
                    pass

            cleanup_pins = [
                getattr(display, pin_name, None)
                for pin_name in ("LED_R", "LED_G", "LED_B", "BACKLIGHT")
            ]
            cleanup_pins = [pin for pin in cleanup_pins if pin is not None]
            if cleanup_pins:
                try:
                    gpio.cleanup(cleanup_pins)
                except Exception as exc:  # pragma: no cover - hardware import
                    logging.debug("Failed GPIO cleanup for Display HAT Mini pins %s: %s", cleanup_pins, exc)

        teardown_methods = ["cleanup", "deinit", "close"]
        if call_destructor:
            teardown_methods.append("__del__")

        for method_name in teardown_methods:
            method = getattr(display, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception as exc:  # pragma: no cover - hardware import
                    logging.debug("Failed to %s Display HAT Mini driver: %s", method_name, exc)

    def _release_display_hat_mini_compat(self, display, *, call_destructor: bool) -> None:
        """Release helper tolerant of tests that monkeypatch the cleanup hook."""

        try:
            self._release_display_hat_mini(display, call_destructor=call_destructor)
        except TypeError as exc:
            if "call_destructor" not in str(exc):
                raise
            self._release_display_hat_mini(display)



    def _maybe_reinitialize_display_hat_mini(self) -> None:
        """Periodically recreate the Display HAT Mini driver to avoid long-run stalls."""

        if self._display is None:
            return
        if self._display_reinit_seconds <= 0:
            return
        if self._display_reinit_disabled:
            return

        now = time.monotonic()
        if now - self._last_display_reinit < self._display_reinit_seconds:
            return
        if now < self._next_display_reinit_retry:
            return

        with self._display_reinit_lock:
            # Another thread may have already reinitialized while we waited.
            now = time.monotonic()
            if self._display is None:
                return
            if now - self._last_display_reinit < self._display_reinit_seconds:
                return
            if now < self._next_display_reinit_retry:
                return

            old_display = self._display
            self._display = None

            with self._display_io_lock:
                self._release_display_hat_mini_compat(old_display, call_destructor=False)

            try:
                new_display = self._create_display_hat_mini(self._buffer)
            except Exception as exc:  # pragma: no cover - hardware import
                with self._display_io_lock:
                    self._display = old_display
                try:  # pragma: no cover - hardware import
                    with self._display_io_lock:
                        old_display.set_backlight(self._backlight_level)
                        if (
                            DISPLAY_HAT_MINI_LED_ENABLED
                            or self._display_hat_mini_indicator_border
                        ) and any(self._led_color):
                            old_display.set_led(
                                r=_normalized_led_to_driver_channel(self._led_color[0]),
                                g=_normalized_led_to_driver_channel(self._led_color[1]),
                                b=_normalized_led_to_driver_channel(self._led_color[2]),
                            )
                except Exception as restore_exc:  # pragma: no cover - hardware import
                    logging.debug("Failed to restore previous Display HAT Mini state after reinit failure: %s", restore_exc)

                if _is_device_busy_error(exc):
                    self._display_reinit_disabled = True
                    self._next_display_reinit_retry = 0.0
                    logging.info(
                        "Display HAT Mini periodic reinit disabled after device busy error; "
                        "continuing with current driver (%s)",
                        exc,
                    )
                    return

                self._next_display_reinit_retry = now + self._DISPLAY_REINIT_RETRY_SECONDS
                logging.warning(
                    "Display HAT Mini reinit failed; restored previous driver and retrying in %ds (%s)",
                    self._DISPLAY_REINIT_RETRY_SECONDS,
                    exc,
                )
                return

            with self._display_io_lock:
                self._display = new_display

            self._last_display_reinit = now
            self._next_display_reinit_retry = 0.0

            try:
                with self._display_io_lock:
                    new_display.set_backlight(self._backlight_level)
            except Exception as exc:  # pragma: no cover - hardware import
                logging.debug("Failed to restore backlight after display reinit: %s", exc)

            if (
                DISPLAY_HAT_MINI_LED_ENABLED
                or self._display_hat_mini_indicator_border
            ) and any(self._led_color):
                try:
                    with self._display_io_lock:
                        new_display.set_led(
                            r=_normalized_led_to_driver_channel(self._led_color[0]),
                            g=_normalized_led_to_driver_channel(self._led_color[1]),
                            b=_normalized_led_to_driver_channel(self._led_color[2]),
                        )
                except Exception as exc:  # pragma: no cover - hardware import
                    logging.debug("Failed to restore LED state after display reinit: %s", exc)

            logging.info(
                "Display HAT Mini driver reinitialized after %d seconds to keep output active.",
                self._display_reinit_seconds,
            )


    def _bump_frame_id(self) -> None:
        with self._frame_lock:
            self._frame_id += 1

    def clear(self):
        self._buffer = Image.new("RGB", (self.width, self.height), "black")
        self._bump_frame_id()
        self._update_display()

    def image(self, pil_img: Image.Image):
        pil_img = self._apply_bottom_safe_buffer(pil_img)
        pil_img = self._apply_indicator_bottom_safe_buffer(pil_img)
        if pil_img.size != (self.width, self.height):
            pil_img = pil_img.resize((self.width, self.height), Image.ANTIALIAS)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        self._buffer = pil_img.copy()
        self._bump_frame_id()
        self._update_display()

    def _apply_bottom_safe_buffer(self, pil_img: Image.Image) -> Image.Image:
        """Always clear the bottom safety strip so content never touches the edge."""

        bottom_buffer = self._BOTTOM_SAFE_BUFFER_PX
        if self._uses_kernel_output and not (
            self._hyperpixel_indicator_border or self._display_hat_mini_indicator_border
        ):
            bottom_buffer = self._KERNEL_BOTTOM_SAFE_BUFFER_PX

        bottom_buffer = max(0, bottom_buffer)
        if bottom_buffer <= 0:
            return pil_img

        source = pil_img
        if source.mode != "RGB":
            source = source.convert("RGB")
        if source.size != (self.width, self.height):
            source = source.resize((self.width, self.height), Image.ANTIALIAS)

        buffered_img = source.copy()
        ImageDraw.Draw(buffered_img).rectangle(
            [(0, self.height - bottom_buffer), (self.width - 1, self.height - 1)],
            fill="black",
        )
        return buffered_img

    def _apply_indicator_bottom_safe_buffer(self, pil_img: Image.Image) -> Image.Image:
        """Clear a bottom buffer to avoid indicator border overlap, when enabled."""

        if not (self._hyperpixel_indicator_border or self._display_hat_mini_indicator_border):
            return pil_img

        bottom_buffer = max(0, self._INDICATOR_BOTTOM_SAFE_BUFFER_PX)
        if bottom_buffer <= 0:
            return pil_img

        source = pil_img
        if source.mode != "RGB":
            source = source.convert("RGB")
        if source.size != (self.width, self.height):
            source = source.resize((self.width, self.height), Image.ANTIALIAS)

        buffered_img = source.copy()
        ImageDraw.Draw(buffered_img).rectangle(
            [(0, self.height - bottom_buffer), (self.width - 1, self.height - 1)],
            fill="black",
        )
        return buffered_img

    def show(self):
        # No additional action required; display() is triggered during image()
        self._update_display()

    def capture(self) -> Image.Image:
        """Return a copy of the currently buffered frame."""

        return self._buffer.copy()

    def frame_id(self) -> int:
        """Return the current frame identifier."""

        with self._frame_lock:
            return self._frame_id

    # ----- Hardware helpers -------------------------------------------------
    def set_backlight(self, level: float) -> float:
        """Set the LCD backlight brightness (0.0 – 1.0)."""

        # Clamp the requested level to keep the screen visible but not blinding.
        # Allow 0.0 for explicit "off" requests (used by display toggle).
        level = max(0.0, min(1.0, level))
        # Treat the top-end as true full brightness so 100% matches forcing
        # the underlying driver brightness to 1.0.
        if level >= 1.0:
            level = 1.0

        with self._backlight_lock:
            self._backlight_level = level

            if self._display is None:  # pragma: no cover - hardware import
                return self._backlight_level

            try:  # pragma: no cover - hardware import
                with self._display_io_lock:
                    self._display.set_backlight(self._backlight_level)
                    if self._backlight_level == 1.0 and hasattr(self._display, "brightness"):
                        self._display.brightness = 1.0
            except Exception as exc:  # pragma: no cover - hardware import
                logging.debug("Failed to set backlight level: %s", exc)

        return self._backlight_level

    def adjust_backlight(self, delta: float) -> float:
        """Adjust the backlight brightness by *delta* (0.0 – 1.0)."""

        with self._backlight_lock:
            new_level = self._backlight_level + delta

        return self.set_backlight(new_level)

    def backlight_level(self) -> float:
        """Return the current backlight level."""

        with self._backlight_lock:
            return self._backlight_level

    def set_led(self, r: float = 0.0, g: float = 0.0, b: float = 0.0) -> None:
        """Set the onboard RGB LED, if hardware is available."""

        r = _clamp_led_level(r)
        g = _clamp_led_level(g)
        b = _clamp_led_level(b)
        self._led_color = (r, g, b)
        if self._hyperpixel_indicator_border or self._display_hat_mini_indicator_border:
            self._update_display()

        should_drive_hardware_led = (
            DISPLAY_HAT_MINI_LED_ENABLED or self._display_hat_mini_indicator_border
        )
        if self._display is None or not should_drive_hardware_led:  # pragma: no cover - hardware import
            return
        try:  # pragma: no cover - hardware import
            if self._display_hat_mini_indicator_border:
                led_kwargs = {
                    "r": self._indicator_channel_to_pixel(r),
                    "g": self._indicator_channel_to_pixel(g),
                    "b": self._indicator_channel_to_pixel(b),
                }
            else:
                led_kwargs = {
                    "r": _normalized_led_to_driver_channel(r),
                    "g": _normalized_led_to_driver_channel(g),
                    "b": _normalized_led_to_driver_channel(b),
                }
            with self._display_io_lock:
                self._display.set_led(**led_kwargs)
        except Exception as exc:  # pragma: no cover - hardware import
            logging.debug("Display LED update failed: %s", exc)

    def _indicator_buffer(self) -> Image.Image:
        """Return a frame with the border LED indicator overlay when enabled."""

        if not (
            self._hyperpixel_indicator_border
            or self._display_hat_mini_indicator_border
        ):
            return self._buffer

        color = tuple(self._indicator_channel_to_pixel(value) for value in self._led_color)
        if not any(color):
            return self._buffer

        # Compose onto a fresh image each frame so concurrent ``_update_display()``
        # calls cannot mutate a buffer while a previous frame is still being
        # transferred to the panel. Reusing a shared work buffer can surface as
        # partial-frame flicker/tearing near the top edge.
        indicator_frame = self._buffer.copy()
        ImageDraw.Draw(indicator_frame).rectangle(
            [(0, 0), (self.width - 1, self.height - 1)],
            outline=color,
            width=HYPERPIXEL_LED_INDICATOR_BORDER_WIDTH,
        )
        return indicator_frame

    @staticmethod
    def _indicator_channel_to_pixel(value: float) -> int:
        value = _clamp_led_level(value)
        if value <= 0:
            return 0
        if LED_INDICATOR_LEVEL <= 0:
            return min(255, int(round(value * 255)))
        normalized = value / LED_INDICATOR_LEVEL
        return max(1, min(255, int(round(normalized * 255))))

    def is_button_pressed(self, name: str) -> bool:
        """Return True if the named button is currently pressed."""

        normalized_name = name.upper()

        gpio_button = self._gpio_buttons.get(normalized_name)
        if gpio_button is not None:
            try:  # pragma: no cover - hardware import
                return bool(gpio_button.is_pressed)
            except Exception as exc:
                logging.debug("GPIO button read failed (%s): %s", name, exc)

        if self._display is None:  # pragma: no cover - hardware import
            return False

        pin = self._button_pins.get(normalized_name)

        try:  # pragma: no cover - hardware import
            with self._display_io_lock:
                read_button = getattr(self._display, "read_button", None)
                if callable(read_button):
                    try:
                        if pin is not None:
                            raw_state = read_button(pin)
                        else:
                            raw_state = read_button(normalized_name)
                    except Exception:
                        raw_state = read_button(normalized_name)
                else:
                    raise AttributeError("Display driver does not expose read_button")
        except Exception as exc:  # pragma: no cover - hardware import
            logging.debug("Display button read failed (%s): %s", name, exc)
            return False

        if isinstance(raw_state, bool):  # pragma: no cover - hardware import
            return raw_state

        if isinstance(raw_state, (int, float)):  # pragma: no cover - hardware import
            # Buttons are wired active-low; a ``0`` reading means the button is
            # being held down.  ``read_button`` previously returned ``True``
            # when pressed but newer firmware returns the raw ``0/1`` GPIO
            # value.  Treat both styles uniformly so the skip button works
            # regardless of driver version.
            return raw_state == 0

        return bool(raw_state)

    def set_button_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        """Register a callable invoked when a hardware button is pressed."""

        self._button_callback = callback

    def _handle_hw_button_event(self, pin) -> None:  # pragma: no cover - hardware import
        name = None

        if isinstance(pin, str):
            candidate = pin.strip().upper()
            if candidate in self._BUTTON_NAMES:
                name = candidate

        if name is None and isinstance(pin, int):
            ordered_buttons = list(self._BUTTON_NAMES)
            if 0 <= pin < len(ordered_buttons):
                name = ordered_buttons[pin]

        if name is None:
            for button_name, button_pin in self._button_pins.items():
                if button_pin == pin:
                    name = button_name
                    break

        if not name or self._display is None:
            return

        read_identifier = self._button_pins.get(name, pin)
        if read_identifier is None:
            read_identifier = name

        try:
            with self._display_io_lock:
                state = self._display.read_button(read_identifier)
        except Exception as exc:
            logging.debug("Hardware button callback read failed: %s", exc)
            return

        if isinstance(state, bool):
            pressed = state
        elif isinstance(state, (int, float)):
            pressed = state == 0
        else:
            pressed = bool(state)

        if not pressed:
            return

        callback = self._button_callback
        if callback is None:
            return

        try:
            callback(name)
        except Exception as exc:
            logging.debug("Button callback raised %s", exc)


def get_active_display() -> Optional["Display"]:
    """Return the most recently constructed :class:`Display` instance, if any."""

    return _ACTIVE_DISPLAY


@dataclass
class ScreenImage:
    """Container for a rendered screen image.

    Attributes
    ----------
    image:
        The full PIL image representing the screen.
    displayed:
        Whether the image has already been pushed to the display by the
        originating function. This allows callers to skip redundant redraws
        while still accessing the image data (e.g., for screenshots).
    led_override:
        Optional RGB tuple describing an LED color override that should remain
        active while the image is shown.
    consumed_delay:
        True when the screen renderer has already consumed the normal dwell
        delay internally (for example, by rendering an animation loop).
    """

    image: Image.Image
    displayed: bool = False
    led_override: Optional[Tuple[float, float, float]] = None
    consumed_delay: bool = False

# ─── Basic utilities ────────────────────────────────────────────────────────
@log_call
def clear_display(display):
    """
    Clear the connected display, falling back to a blank frame.
    """
    if _DEFER_CLEAR_DISPLAY.is_set():
        # When clears are deferred we should leave the current frame buffer
        # untouched so transitions can blend from the live image. Mutating the
        # in-memory buffer to black causes a visible blank fade between screens.
        return
    try:
        display.clear()
    except Exception:
        try:
            blank = Image.new("RGB", (getattr(display, "width", WIDTH), getattr(display, "height", HEIGHT)), "black")
            display.image(blank)
            display.show()
        except Exception:
            pass

@log_call
def draw_text_centered(
    draw: ImageDraw.Draw,
    text: str,
    font: ImageFont.FreeTypeFont,
    y_offset: int = 0,
    width: int = WIDTH,
    height: int = HEIGHT,
    *,
    fill=(255,255,255)
):
    """
    Draw `text` centered horizontally at vertical center + y_offset.
    """
    w, h = draw.textsize(text, font=font)
    x = (width - w) // 2
    y = (height - h) // 2 + y_offset
    draw.text((x, y), text, font=font, fill=fill)

@log_call
def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int):
    """
    Break `text` into lines so each line fits within max_width.
    """
    words = text.split()
    if not words:
        return []
    dummy = Image.new("RGB", (max_width, 1))
    draw = ImageDraw.Draw(dummy)
    lines = [words[0]]
    for w in words[1:]:
        test = f"{lines[-1]} {w}"
        if draw.textsize(test, font=font)[0] <= max_width:
            lines[-1] = test
        else:
            lines.append(w)
    return lines


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    try:
        return draw.textsize(text, font=font)
    except Exception:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top


def clone_font(font: ImageFont.FreeTypeFont, size: int) -> ImageFont.FreeTypeFont:
    path = getattr(font, "path", None)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return font


def fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    base_font: ImageFont.FreeTypeFont,
    max_width: int,
    max_height: int,
    *,
    min_pt: int = 8,
    max_pt: int | None = None,
) -> ImageFont.FreeTypeFont:
    base_size = getattr(base_font, "size", 16)
    hi = max_pt if max_pt else base_size
    lo = min_pt
    best = clone_font(base_font, lo)
    while lo <= hi:
        mid = (lo + hi) // 2
        test_font = clone_font(base_font, mid)
        width, height = measure_text(draw, text, test_font)
        if width <= max_width and height <= max_height:
            best = test_font
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def format_voc_ohms(value) -> str:
    if value is None:
        return "N/A"
    try:
        val = float(value)
    except Exception:
        return "N/A"
    if val >= 1_000_000:
        return f"{val / 1_000_000:.1f} MΩ"
    if val >= 1_000:
        return f"{val / 1_000:.1f} kΩ"
    return f"{val:.0f} Ω"


def temperature_color(temp_f: float, lo: float = 50.0, hi: float = 80.0) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, (temp_f - lo) / (hi - lo + 1e-6)))
    if t < 0.5:
        alpha = t / 0.5
        r = int(0 + (80 - 0) * alpha)
        g = int(150 + (220 - 150) * alpha)
        b = int(255 + (180 - 255) * alpha)
    else:
        alpha = (t - 0.5) / 0.5
        r = int(80 + (255 - 80) * alpha)
        g = int(220 + (120 - 220) * alpha)
        b = int(180 + (0 - 180) * alpha)
    return (r, g, b)

def _resolve_fade_steps(display: Display, steps: int | None) -> int:
    """Resolve fade step count from explicit args, policy defaults, and env/config flags."""

    if steps is not None:
        return max(0, int(steps))

    if not DISPLAY_FADE_IN_ENABLED:
        return 0

    profile_id = get_display_profile_id(display.width, display.height)
    return DISPLAY_FADE_IN_STEPS_BY_PROFILE.get(
        profile_id,
        DISPLAY_FADE_IN_DISPLAY_HAT_MINI_STEPS,
    )


@log_call
def animate_fade_in(
    display: Display,
    new_image: Image.Image,
    steps: int | None = None,
    delay: float = 0.02,
    *,
    from_image: Image.Image | None = None,
):
    """
    Fade from the current display buffer (or ``from_image``) into ``new_image``.
    """

    resolved_steps = _resolve_fade_steps(display, steps)
    if resolved_steps <= 0:
        display.image(new_image)
        return

    if from_image is None:
        try:
            base = display.capture()
        except AttributeError:
            base = None
        if base is None:
            base = Image.new("RGB", new_image.size, (0, 0, 0))
    else:
        base = from_image

    base = base.convert("RGB")
    if base.size != new_image.size:
        base = base.resize(new_image.size, Image.ANTIALIAS)

    target = new_image.convert("RGB")

    for i in range(resolved_steps + 1):
        frame_start = time.time()

        alpha = i / resolved_steps
        frame = Image.blend(base, target, alpha)
        display.image(frame)

        # Account for rendering time to maintain consistent frame rate
        elapsed = time.time() - frame_start
        sleep_time = max(0, delay - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

@log_call
def animate_scroll(display: Display, image: Image.Image, speed=3.0, y_offset=None):
    """
    Scroll an image across the display.
    """
    if image is None:
        return

    bands = image.getbands() if hasattr(image, "getbands") else ()
    has_alpha = "A" in bands
    image = image.convert("RGBA" if has_alpha else "RGB")

    w, h = display.width, display.height
    img_w, img_h = image.size
    y = y_offset if y_offset is not None else (h - img_h) // 2
    direction = random.choice(("ltr", "rtl"))
    speed = float(speed)
    if speed == 0:
        return
    start, end = ((-img_w, w) if direction == "ltr" else (w, -img_w))
    step = abs(speed) if direction == "ltr" else -abs(speed)

    background_color = (0, 0, 0, 0) if has_alpha else (0, 0, 0)
    frame_mode = "RGBA" if has_alpha else "RGB"

    target_frame_time = 0.016  # ~60 FPS for smoother animation

    wait_for_skip = getattr(display, "wait_for_skip", None)
    skip_requested = getattr(display, "skip_requested", None)

    def _should_skip() -> bool:
        return bool(callable(skip_requested) and skip_requested())

    def _sleep(duration: float) -> bool:
        if duration <= 0:
            return _should_skip()
        if callable(wait_for_skip):
            return bool(wait_for_skip(duration))
        time.sleep(duration)
        return _should_skip()

    x = float(start)
    while (x <= end if step > 0 else x >= end):
        if _should_skip():
            return
        frame_start = time.time()

        x_pos = int(round(x))
        frame = Image.new(frame_mode, (w, h), background_color)
        if has_alpha:
            frame.paste(image, (x_pos, y), image)
            frame_to_show = frame.convert("RGB")
        else:
            frame.paste(image, (x_pos, y))
            frame_to_show = frame
        display.image(frame_to_show)

        # Account for rendering time to maintain consistent frame rate
        elapsed = time.time() - frame_start
        sleep_time = max(0, target_frame_time - elapsed)
        if _sleep(sleep_time):
            return
        x += step

    if _should_skip():
        return

    # End on a visible centered logo frame so the next screen can fade from the
    # brand card instead of a fully black buffer.
    centered_frame = Image.new(frame_mode, (w, h), background_color)
    centered_x = (w - img_w) // 2
    if has_alpha:
        centered_frame.paste(image, (centered_x, y), image)
        display.image(centered_frame.convert("RGB"))
    else:
        centered_frame.paste(image, (centered_x, y))
        display.image(centered_frame)


@dataclass(frozen=True)
class AdaptiveScrollParams:
    step: int
    target_frame_time: float
    use_page_jump: bool


def compute_adaptive_scroll_params(
    *,
    content_height: int,
    viewport_height: int,
    viewport_width: int,
    base_step: int,
    min_frame_time: float = 0.016,
    page_jump_mode: bool = True,
    page_jump_threshold_ratio: float = 8.0,
) -> AdaptiveScrollParams:
    """Compute resolution-aware scroll step and frame pacing.

    Larger displays can scroll farther per frame, while very tall boards run at
    a lower FPS target to keep motion readable and reduce frame churn.

    Page-jump mode is intentionally conservative so long scoreboards keep a
    readable line-by-line cadence instead of skipping most rows.
    """

    safe_base_step = max(1, int(base_step))
    max_dimension = max(int(viewport_width), int(viewport_height), 1)
    resolution_scale = max(1.0, max_dimension / 320.0)
    step = max(safe_base_step, int(round(safe_base_step * resolution_scale)))

    overflow = max(0, int(content_height) - int(viewport_height))
    overflow_ratio = overflow / max(1, int(viewport_height))
    # For very tall content increase frame time (lower FPS).
    target_frame_time = min_frame_time * (1.0 + min(2.0, overflow_ratio * 0.6))

    return AdaptiveScrollParams(
        step=step,
        target_frame_time=target_frame_time,
        use_page_jump=bool(page_jump_mode and overflow_ratio >= page_jump_threshold_ratio),
    )


def scroll_vertical_content(
    *,
    display,
    content_height: int,
    viewport_width: int,
    viewport_height: int,
    render_at_offset: Callable[[int], None],
    base_step: int,
    pause_start: float,
    pause_end: float,
    reverse: bool = False,
    page_jump_mode: bool = True,
    min_frame_time: float = 0.016,
) -> None:
    """Shared vertical scroll driver with adaptive timing and skip support."""

    wait_for_skip = getattr(display, "wait_for_skip", None)
    skip_requested = getattr(display, "skip_requested", None)

    def _should_skip() -> bool:
        return bool(callable(skip_requested) and skip_requested())

    def _sleep(duration: float) -> bool:
        if duration <= 0:
            return _should_skip()
        if callable(wait_for_skip):
            return bool(wait_for_skip(duration))
        time.sleep(duration)
        return _should_skip()

    max_offset = max(0, int(content_height) - int(viewport_height))
    params = compute_adaptive_scroll_params(
        content_height=content_height,
        viewport_height=viewport_height,
        viewport_width=viewport_width,
        base_step=base_step,
        page_jump_mode=page_jump_mode,
        min_frame_time=min_frame_time,
    )

    stride = params.step
    if params.use_page_jump:
        stride = max(stride, int(viewport_height * 0.4))

    start_offset = max_offset if reverse else 0
    render_at_offset(start_offset)

    if _sleep(pause_start):
        return
    if max_offset <= 0:
        _sleep(pause_end)
        return

    if reverse:
        offsets = range(start_offset - stride, -stride, -stride)
    else:
        offsets = range(stride, max_offset + stride, stride)

    for raw_offset in offsets:
        if _should_skip():
            return
        frame_start = time.time()
        offset = max(0, min(max_offset, raw_offset))
        render_at_offset(offset)
        elapsed = time.time() - frame_start
        sleep_time = max(0, params.target_frame_time - elapsed)
        if _sleep(sleep_time):
            return

    _sleep(pause_end)

# ─── Date & Time Helpers ─────────────────────────────────────────────────────
def parse_game_date(iso_date_str: str, time_str: str = "TBD") -> str:
    try:
        d = datetime.datetime.strptime(iso_date_str, "%Y-%m-%d").date()
    except Exception:
        return time_str
    today = datetime.datetime.now(CENTRAL_TIME).date()
    if d == today:
        day = "Today"
    elif d == today + datetime.timedelta(days=1):
        day = "Tomorrow"
    else:
        day = d.strftime("%a %-m/%-d")
    return f"{day} {time_str}" if time_str.upper() != "TBD" else f"{day} TBD"

def format_date_no_leading(dt_date: datetime.date) -> str:
    return f"{dt_date.month}/{dt_date.day}"

def format_time_no_leading(dt_time: datetime.time) -> str:
    return dt_time.strftime("%I:%M %p").lstrip("0")

def split_time_period(dt_time: datetime.time) -> tuple[str,str]:
    full = dt_time.strftime("%I:%M %p").lstrip("0")
    parts = full.rsplit(" ", 1)
    return (parts[0], parts[1]) if len(parts)==2 else (full, "")

# ─── Team & Standings Helpers ────────────────────────────────────────────────
def get_team_display_name(team) -> str:
    if not isinstance(team, dict):
        return str(team)
    t = team.get("team", team)
    for key in ("commonName","name","teamName","fullName","city"): 
        val = t.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return "UNK"

def get_opponent_last_game(team) -> str:
    if not isinstance(team, dict):
        return str(team)
    city = team.get("placeName", {}).get("default", "").strip()
    return city or get_team_display_name(team)

def extract_split_record(split_records: list, record_type: str) -> str:
    for sp in split_records:
        if sp.get("type", "").lower() == record_type.lower():
            w = sp.get("wins", "N/A")
            l = sp.get("losses", "N/A")
            p = sp.get("pct", "N/A")
            return f"{w}-{l} ({p})"
    return "N/A"

def wind_direction(degrees: float) -> str:
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
            "S","SSW","SW","WSW","W","WNW","NW","NNW"]
    try:
        idx = int((degrees / 22.5) + 0.5) % 16
        return dirs[idx]
    except Exception:
        return ""

wind_deg_to_compass = wind_direction

def center_coords(
    img_size: tuple[int,int],
    content_size: tuple[int,int],
    y_offset: int = 0
) -> tuple[int,int]:
    w, h = img_size
    cw, ch = content_size
    return ((w - cw)//2, (h - ch)//2 + y_offset)

def _week_sort_value(week_label: str) -> float:
    """Return a numeric sort key for week labels.

    Supports values like ``"Week 16"`` or preseason fractions such as ``"0.2"``.
    Unknown or missing labels sort to the end of the schedule.
    """

    label = (week_label or "").strip().lower()
    if label.startswith("week"):
        try:
            return float(label.split()[1])
        except Exception:
            return float("inf")
    try:
        return float(label)
    except Exception:
        return float("inf")


def _game_sort_value(entry: Dict[str, Any]) -> float:
    """Return the numeric sort order for a schedule entry.

    Prefers ``game_no`` (to support decimal preseason numbering) and falls back to
    ``week`` labels.
    """

    try:
        game_no = entry.get("game_no")
    except AttributeError:  # pragma: no cover - defensive
        game_no = None
    if game_no is not None:
        try:
            return float(str(game_no))
        except Exception:
            pass

    return _week_sort_value(str(entry.get("week", "")))


def _parse_game_date(
    date_text: str, *, default_year: int
) -> Optional[datetime.date]:
    if not date_text:
        return None
    date_text = str(date_text).strip()
    if not date_text or date_text.upper() in {"TBD", "BYE"}:
        return None
    for fmt in ("%a, %b %d %Y", "%a, %b %d, %Y"):
        try:
            parsed = datetime.datetime.strptime(date_text, fmt)
            return parsed.date()
        except Exception:
            continue
    try:
        parsed = datetime.datetime.strptime(date_text, "%a, %b %d")
        return datetime.date(default_year, parsed.month, parsed.day)
    except Exception:
        return None


def next_game_from_schedule(
    schedule: List[Dict[str, Any]], today: Optional[datetime.date] = None
) -> Optional[Dict[str, Any]]:
    today = today or datetime.date.today()
    year = today.year

    candidates: List[tuple[Optional[datetime.date], float, int, Dict[str, Any]]] = []
    for idx, entry in enumerate(schedule):
        if entry.get("opponent") == "—":
            continue

        parsed_date = _parse_game_date(entry.get("date", ""), default_year=year)

        sort_value = _game_sort_value(entry)
        candidates.append((parsed_date, sort_value, idx, entry))

    if not candidates:
        return None

    future_dated = [
        (parsed_date, sort_value, idx, entry)
        for parsed_date, sort_value, idx, entry in candidates
        if parsed_date is not None and parsed_date >= today
    ]

    if future_dated:
        parsed_date, _, _, entry = min(
            future_dated, key=lambda item: (item[0], item[1], item[2])
        )
        return entry

    dated_past = [
        (parsed_date, sort_value, idx, entry)
        for parsed_date, sort_value, idx, entry in candidates
        if parsed_date is not None and parsed_date <= today
    ]
    if dated_past:
        _, last_sort_value, _, _ = max(
            dated_past, key=lambda item: (item[0], item[1], item[2])
        )
        higher_games = [
            (parsed_date, sort_value, idx, entry)
            for parsed_date, sort_value, idx, entry in candidates
            if sort_value > last_sort_value
        ]
        if higher_games:
            _, _, _, entry = min(
                higher_games,
                key=lambda item: (item[1], item[0] or datetime.date.max, item[2]),
            )
            return entry

    _, _, _, entry = min(
        candidates, key=lambda item: (item[1], item[0] or datetime.date.max, item[2])
    )
    return entry


_LOGO_BRIGHTNESS_OVERRIDES: dict[tuple[str, str], float] = {
    ("nhl", "WAS"): 1.35,
    ("nhl", "TBL"): 1.35,
    ("nhl", "TB"): 1.35,
    ("nfl", "NYJ"): 1.4,
    ("mlb", "SD"): 1.35,
    ("mlb", "DET"): 1.35,
    ("mlb", "NYY"): 1.35,
}


def _adjust_logo_brightness(logo: Image.Image, base_dir: str, abbr: str) -> Image.Image:
    sport = os.path.basename(os.path.normpath(base_dir or ""))
    key = (sport.lower(), (abbr or "").upper())
    factor = _LOGO_BRIGHTNESS_OVERRIDES.get(key)
    if not factor:
        return logo
    return ImageEnhance.Brightness(logo).enhance(factor)


def standard_next_game_logo_height(panel_height: int) -> int:
    """Return the shared next-game logo height used across team screens."""
    if panel_height >= 128:
        return 150
    if panel_height >= 96:
        return 109
    return 89


def standard_next_game_logo_height_for_space(
    panel_height: int,
    available_height: int,
    *,
    scale: float = 1.0,
) -> int:
    """Return a shared next-game logo height constrained by available space."""
    clamped_scale = max(0.5, min(float(scale or 1.0), 1.2))
    desired = max(1, int(round(standard_next_game_logo_height(panel_height) * clamped_scale)))
    return max(1, min(desired, max(1, available_height)))


def standard_scoreboard_team_logo_height(panel_height: int, *, compact: bool = False) -> int:
    """Return the shared scoreboard team logo height for the given panel size."""
    from config import is_hyperpixel_next_layout, scale_value, scale_value_width

    if compact:
        return scale_value_width(20) if is_hyperpixel_next_layout() else scale_value(26)
    return scale_value_width(36) if is_hyperpixel_next_layout() else scale_value(52)


def standard_scoreboard_league_logo_height(team_logo_height: int) -> int:
    """Return the shared scoreboard league logo height for the given team logo height."""
    return int(round(max(1, team_logo_height) * 1.25))


def standard_next_game_logo_frame_width(
    logo_height: int, logos: Iterable[Image.Image | None] = ()
) -> int:
    """Width to reserve for each logo on next-game screens.

    The returned width ensures that both logos share the same frame size, avoiding
    "crowding" around the centered "@" regardless of each logo's aspect ratio.
    """

    # Slightly wider than tall to give horizontally oriented marks breathing room.
    min_width = int(round(max(1, logo_height) * 1.1))
    max_logo_width = max((logo.width for logo in logos if logo), default=0)
    return max(min_width, max_logo_width)


def fit_logo_to_box(logo: Image.Image | None, box_size: int) -> Image.Image | None:
    """Resize a logo to fit within a square box, preserving aspect ratio."""
    if logo is None or box_size <= 0:
        return logo
    width, height = logo.size
    if width <= 0 or height <= 0:
        return logo
    scale = min(box_size / float(width), box_size / float(height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    if new_width == width and new_height == height:
        return logo
    return logo.resize((new_width, new_height), Image.LANCZOS)


MLB_LOGO_FILE_ALIASES = {
    "AUS": "Australia",
    "BRA": "Brazil",
    "CAN": "Canada",
    "COL": "Colombia",
    "CUB": "Cuba",
    "CZE": "Czech Republic",
    "DOM": "Dominican Republic",
    "GBR": "Great Britain",
    "ISR": "Israel",
    "ITA": "Italy",
    "JPN": "Japan",
    "KOR": "South Korea",
    "MEX": "Mexico",
    "NCA": "Nicaragua",
    "NED": "Kingdom of the Netherlands",
    "PAN": "Panama",
    "PUR": "Puerto Rico",
    "TPE": "Chinese Taipei",
    "TWN": "Chinese Taipei",
    "USA": "United States",
    "VEN": "Venezuela",
}


def _logo_filename_candidates(abbr: str) -> list[str]:
    cleaned = (abbr or "").strip()
    if not cleaned:
        return []

    raw_variants = [cleaned, cleaned.replace("_", " "), cleaned.replace("-", " ")]
    normalized_key = cleaned.upper().replace("_", " ").replace("-", " ")
    aliased = MLB_LOGO_FILE_ALIASES.get(normalized_key)
    if aliased:
        raw_variants.append(aliased)

    candidates: list[str] = []
    for variant in raw_variants:
        token = (variant or "").strip()
        if not token:
            continue
        for candidate in (token, token.upper(), token.lower(), token.title()):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def load_team_logo(
    base_dir: str,
    abbr: str,
    height: int = 36,
    *,
    box_size: int | None = None,
    trim: bool = False,
) -> Image.Image | None:
    candidates = _logo_filename_candidates(abbr)
    if not candidates:
        return None
    last_error: Optional[Exception] = None
    for candidate in candidates:
        filename = f"{candidate}.png"
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            logo = Image.open(path).convert("RGBA")
            logo = _adjust_logo_brightness(logo, base_dir, candidate)
            if trim:
                alpha = logo.split()[-1]
                bbox = alpha.getbbox()
                if bbox:
                    logo = logo.crop(bbox)
            if box_size and box_size > 0:
                return fit_logo_to_box(logo, box_size)
            ratio = height / logo.height
            return logo.resize((int(logo.width * ratio), height), Image.ANTIALIAS)
        except Exception as exc:  # pragma: no cover - rare file corruption
            last_error = exc
            continue
    if last_error:
        logging.warning("Could not load logo '%s': %s", abbr, last_error)
    return None


_MISSING_TEAM_LOGO_WARNINGS: set[tuple[str, str, str]] = set()


def log_missing_team_logo(screen_id: str, team_name: str, expected_abbr: str) -> None:
    """Emit a deduplicated warning when a screen cannot find a team logo."""
    cleaned_abbr = str(expected_abbr or "").strip().upper()
    if not cleaned_abbr:
        return
    cleaned_name = str(team_name or "").strip() or "Unknown Team"
    cache_key = (str(screen_id or "unknown"), cleaned_name, cleaned_abbr)
    if cache_key in _MISSING_TEAM_LOGO_WARNINGS:
        return
    _MISSING_TEAM_LOGO_WARNINGS.add(cache_key)
    logging.warning(
        "Missing team logo on %s: team='%s', expected_abbr='%s'",
        cache_key[0],
        cleaned_name,
        cleaned_abbr,
    )

@log_call
def colored_image(mono_img: Image.Image, screen_key: str) -> Image.Image:
    rgb = Image.new("RGB", mono_img.size, (0,0,0))
    pix = mono_img.load()
    draw = ImageDraw.Draw(rgb)
    col = random_color(screen_key)
    for y in range(mono_img.height):
        for x in range(mono_img.width):
            if pix[x, y]:
                draw.point((x, y), fill=col)
    return rgb

@log_call
def load_svg(key, url) -> Image.Image | None:
    cache_dir = os.path.join(os.path.dirname(__file__), "images", "nhl")
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, f"{key}.svg")
    if not os.path.exists(local):
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            with open(local, "wb") as f:
                f.write(r.content)
        except Exception as e:
            logging.warning(f"Failed to download NHL logo: {e}")
            return None
    try:
        from cairosvg import svg2png
        png = svg2png(url=local)
        return Image.open(BytesIO(png))
    except Exception:
        return None

# ─── Update Indicator LED ───────────────────────────────────────────────────
class _LedAnimator:
    """Animate the onboard LED using a cycle of colors."""

    def __init__(
        self,
        display: "Display",
        pattern: Tuple[Tuple[float, float, float], ...],
        interval: float = 0.6,
    ) -> None:
        self._display = display
        self._pattern = pattern
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def is_running_for(
        self,
        display: "Display",
        pattern: Tuple[Tuple[float, float, float], ...],
        interval: float,
    ) -> bool:
        return (
            self._display is display
            and self._thread.is_alive()
            and self._pattern == pattern
            and abs(self._interval - interval) < 1e-6
        )

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.5)
        self._display.set_led(r=0.0, g=0.0, b=0.0)

    def _run(self) -> None:
        idx = 0
        while not self._stop.is_set():
            r, g, b = self._pattern[idx]
            self._display.set_led(r=r, g=g, b=b)
            idx = (idx + 1) % len(self._pattern)
            if self._stop.wait(self._interval):
                break
        self._display.set_led(r=0.0, g=0.0, b=0.0)


def _led_pattern(status: _UpdateStatus) -> Tuple[Tuple[Tuple[float, float, float], ...], float] | tuple[None, None]:
    blue = (0.0, 0.0, LED_INDICATOR_LEVEL)
    yellow = (LED_INDICATOR_LEVEL, LED_INDICATOR_LEVEL, 0.0)

    if status.github and status.apt:
        # Alternate blue/yellow only when both update sources are pending.
        return ((blue, yellow), 0.6)
    if status.apt:
        return ((yellow,), 0.8)
    if status.github:
        return ((blue,), 0.8)
    return (None, None)


def _refresh_led_indicator(display: Optional["Display"] = None) -> None:
    """Reflect update status on the Display HAT Mini LED."""

    global _LED_INDICATOR_ANIMATOR

    display = display or get_active_display()
    status = _UPDATE_STATUS

    if display is None:
        if _LED_INDICATOR_ANIMATOR is not None and not (status.github or status.apt):
            try:  # pragma: no cover - hardware import
                _LED_INDICATOR_ANIMATOR.stop()
            except Exception as exc:
                logging.debug("Failed to stop LED animator without display: %s", exc)
            finally:
                _LED_INDICATOR_ANIMATOR = None
        return

    pattern, interval = _led_pattern(status)

    indicator_border_enabled = bool(
        getattr(display, "_hyperpixel_indicator_border", False)
        or getattr(display, "_display_hat_mini_indicator_border", False)
    )

    if not _UPDATE_INDICATOR_ENABLED:
        if _LED_INDICATOR_ANIMATOR is not None:
            try:  # pragma: no cover - hardware import
                _LED_INDICATOR_ANIMATOR.stop()
            except Exception as exc:
                logging.debug("Failed to stop LED animator while disabling indicator: %s", exc)
            finally:
                _LED_INDICATOR_ANIMATOR = None
        try:
            display.set_led(r=0.0, g=0.0, b=0.0)
        except Exception as exc:  # pragma: no cover - hardware import
            logging.debug("Failed to clear LED while indicator disabled: %s", exc)
        return

    if pattern is None:
        if _LED_INDICATOR_ANIMATOR is not None:
            try:  # pragma: no cover - hardware import
                _LED_INDICATOR_ANIMATOR.stop()
            except Exception as exc:
                logging.debug("Failed to stop LED animator: %s", exc)
            finally:
                _LED_INDICATOR_ANIMATOR = None
        else:
            try:
                display.set_led(r=0.0, g=0.0, b=0.0)
            except Exception as exc:  # pragma: no cover - hardware import
                logging.debug("Failed to clear LED: %s", exc)
        return

    # Border indicators are painted onto the frame buffer via ``set_led()``.
    # Running a background LED animator in this mode can race with normal
    # screen rendering and briefly re-present stale buffers.
    if indicator_border_enabled:
        if _LED_INDICATOR_ANIMATOR is not None:
            try:  # pragma: no cover - hardware import
                _LED_INDICATOR_ANIMATOR.stop()
            except Exception as exc:
                logging.debug("Failed to stop LED animator for border indicator mode: %s", exc)
            finally:
                _LED_INDICATOR_ANIMATOR = None

        r, g, b = pattern[0]
        try:
            display.set_led(r=r, g=g, b=b)
        except Exception as exc:  # pragma: no cover - hardware import
            logging.debug("Failed to set static border indicator LED: %s", exc)
        return

    if _LED_INDICATOR_ANIMATOR is not None:
        if _LED_INDICATOR_ANIMATOR.is_running_for(display, pattern, interval):
            return
        try:  # pragma: no cover - hardware import
            _LED_INDICATOR_ANIMATOR.stop()
        except Exception as exc:
            logging.debug("Failed to stop existing LED animator: %s", exc)
    animator = _LedAnimator(display, pattern, interval)
    _LED_INDICATOR_ANIMATOR = animator
    try:  # pragma: no cover - hardware import
        animator.start()
    except Exception as exc:
        logging.debug("Failed to start LED animator: %s", exc)
        _LED_INDICATOR_ANIMATOR = None


def _set_update_status(github: Optional[bool] = None, apt: Optional[bool] = None) -> None:
    global _UPDATE_STATUS

    if github is None and apt is None:
        return

    status = _UPDATE_STATUS
    _UPDATE_STATUS = _UpdateStatus(
        github=status.github if github is None else github,
        apt=status.apt if apt is None else apt,
    )
    _refresh_led_indicator()


def clear_update_indicator(display: Optional["Display"] = None) -> None:
    """Stop the update LED animation and turn the LED off."""

    global _LED_INDICATOR_ANIMATOR, _UPDATE_STATUS

    _UPDATE_STATUS = _UpdateStatus()
    display = display or get_active_display()

    if _LED_INDICATOR_ANIMATOR is not None:
        try:  # pragma: no cover - hardware import
            _LED_INDICATOR_ANIMATOR.stop()
        except Exception as exc:
            logging.debug("Failed to stop LED animator during cleanup: %s", exc)
        finally:
            _LED_INDICATOR_ANIMATOR = None

    if display is None:
        return

    try:
        display.set_led(r=0.0, g=0.0, b=0.0)
    except Exception as exc:  # pragma: no cover - hardware import
        logging.debug("Failed to clear LED during cleanup: %s", exc)


def set_update_indicator_enabled(
    enabled: bool,
    display: Optional["Display"] = None,
) -> bool:
    """Enable/disable the update indicator and immediately apply the new state."""

    global _UPDATE_INDICATOR_ENABLED
    _UPDATE_INDICATOR_ENABLED = bool(enabled)
    _refresh_led_indicator(display)
    return _UPDATE_INDICATOR_ENABLED


@contextmanager
def temporary_display_led(r: float, g: float, b: float):
    """Temporarily override the display LED, restoring update status after."""

    global _LED_INDICATOR_ANIMATOR

    display = get_active_display()
    if display is None:
        yield
        return

    pattern, interval = _led_pattern(_UPDATE_STATUS)

    animator = _LED_INDICATOR_ANIMATOR
    if animator is not None and (pattern is None or interval is None or not animator.is_running_for(display, pattern, interval)):
        animator = None

    update_led_active = bool(_UPDATE_STATUS.github or _UPDATE_STATUS.apt)
    if animator is not None:
        update_led_active = True

    if animator is not None:
        try:  # pragma: no cover - hardware import
            animator.stop()
        except Exception as exc:
            logging.debug("Failed to stop LED animator before override: %s", exc)
        finally:
            _LED_INDICATOR_ANIMATOR = None

    try:
        display.set_led(r=r, g=g, b=b)
        yield
    finally:
        if update_led_active or _UPDATE_STATUS.github or _UPDATE_STATUS.apt:
            _refresh_led_indicator(display)
        else:
            try:
                display.set_led(r=0.0, g=0.0, b=0.0)
            except Exception as exc:
                logging.debug("Failed to reset LED after override: %s", exc)

_GIT_COMMAND_TIMEOUT = 10
_APT_COMMAND_TIMEOUT = 20


def check_github_updates() -> bool:
    """
    Return True if the local branch differs from its upstream tracking branch.
    Also logs the list of files that have changed on the remote.

    Safe fallbacks:
      - Handles non-git directories gracefully.
      - Skips detached HEADs or branches without an upstream.
      - Silently returns False if remote can't be fetched.
    """
    repo_dir = os.path.dirname(__file__)

    def _github_check_failed() -> bool:
        _set_update_status(github=False)
        return False

    # Is this a git repo?
    try:
        subprocess.check_call(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=repo_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_GIT_COMMAND_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        logging.warning("check_github_updates: git probe timed out")
        return _github_check_failed()
    except Exception:
        logging.info("check_github_updates: not a git repository, skipping check")
        return _github_check_failed()

    # Local branch name (skip detached HEADs)
    try:
        local_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
            timeout=_GIT_COMMAND_TIMEOUT,
        ).decode().strip()
    except subprocess.TimeoutExpired:
        logging.warning("check_github_updates: git branch lookup timed out")
        return _github_check_failed()
    except Exception:
        logging.exception("check_github_updates: failed to determine local branch")
        return _github_check_failed()

    if local_branch in {"HEAD", ""}:
        logging.info("check_github_updates: detached HEAD, skipping check")
        return _github_check_failed()

    # Local SHA
    try:
        local_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
            timeout=_GIT_COMMAND_TIMEOUT,
        ).decode().strip()
    except subprocess.TimeoutExpired:
        logging.warning("check_github_updates: git HEAD lookup timed out")
        return _github_check_failed()
    except Exception:
        logging.exception("check_github_updates: failed to read local HEAD")
        return _github_check_failed()

    # Upstream branch for the current branch
    try:
        upstream_ref = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
            timeout=_GIT_COMMAND_TIMEOUT,
        ).decode().strip()
    except subprocess.TimeoutExpired:
        logging.warning("check_github_updates: git upstream lookup timed out")
        return _github_check_failed()
    except Exception:
        logging.info(
            "check_github_updates: no upstream tracking branch for %s, skipping check",
            local_branch,
        )
        return _github_check_failed()

    # Fetch remote so we can diff against it
    try:
        subprocess.check_call(
            ["git", "fetch", "--quiet", "origin"],
            cwd=repo_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_GIT_COMMAND_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        logging.warning("check_github_updates: git fetch timed out")
        return _github_check_failed()
    except Exception:
        logging.warning("check_github_updates: failed to fetch from origin")
        return _github_check_failed()

    # Remote SHA for the upstream branch
    try:
        remote_sha = subprocess.check_output(
            ["git", "rev-parse", upstream_ref],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
            timeout=_GIT_COMMAND_TIMEOUT,
        ).decode().strip()
    except subprocess.TimeoutExpired:
        logging.warning("check_github_updates: git upstream SHA lookup timed out")
        return _github_check_failed()
    except Exception:
        logging.warning(
            "check_github_updates: failed to resolve upstream %s for %s",
            upstream_ref,
            local_branch,
        )
        return _github_check_failed()

    updated = (local_sha != remote_sha)
    logging.info(f"check_github_updates: updates available = {updated}")
    _set_update_status(github=updated)

    # If updated, log which files changed
    if updated:
        try:
            changed = subprocess.check_output(
                ["git", "diff", "--name-only", f"{local_sha}..{remote_sha}"],
                cwd=repo_dir,
                timeout=_GIT_COMMAND_TIMEOUT,
            ).decode().splitlines()
        except subprocess.TimeoutExpired:
            logging.warning("check_github_updates: git diff timed out")
            return updated
        except Exception:
            logging.exception("check_github_updates: failed to list changed files")
        else:
            if not changed:
                logging.info("check_github_updates: no file list available (empty diff?)")
            else:
                # Keep the log readable if there are many files
                MAX_LIST = 100
                shown = changed[:MAX_LIST]
                logging.info(
                    f"check_github_updates: {len(changed)} file(s) differ from {upstream_ref}:"
                )
                for p in shown:
                    logging.info(f"  • {p}")
                if len(changed) > MAX_LIST:
                    logging.info(f"  …and {len(changed) - MAX_LIST} more")

    return updated


_APT_CACHE_TTL_SECONDS = 4 * 60 * 60
_APT_CACHE_RESULT: Optional[bool] = None
_APT_CACHE_AT: float = 0.0


def check_apt_updates() -> bool:
    """Return True if `apt` has upgradeable packages (cached for four hours)."""

    global _APT_CACHE_RESULT, _APT_CACHE_AT

    now = time.time()
    if _APT_CACHE_RESULT is not None and (now - _APT_CACHE_AT) < _APT_CACHE_TTL_SECONDS:
        logging.info(
            "check_apt_updates: using cached result (%s)",
            "updates" if _APT_CACHE_RESULT else "no updates",
        )
        _set_update_status(apt=_APT_CACHE_RESULT)
        return _APT_CACHE_RESULT

    if shutil.which("apt-get") is None:
        logging.info("check_apt_updates: apt-get unavailable on this platform; skipping")
        _set_update_status(apt=False)
        return False

    try:
        proc = subprocess.run(
            ["apt-get", "-s", "-o", "Debug::NoLocking=1", "upgrade"],
            capture_output=True,
            text=True,
            check=False,
            timeout=_APT_COMMAND_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        logging.warning("check_apt_updates: apt-get simulation timed out")
        _set_update_status(apt=False)
        return False
    except Exception:
        logging.exception("check_apt_updates: failed to run apt-get simulation")
        _set_update_status(apt=False)
        return False

    if proc.returncode != 0:
        logging.warning("check_apt_updates: apt-get exited with %s", proc.returncode)
        _set_update_status(apt=False)
        return False

    updates_available = any(line.startswith("Inst ") for line in proc.stdout.splitlines())
    logging.info("check_apt_updates: updates available = %s", updates_available)

    _APT_CACHE_RESULT = updates_available
    _APT_CACHE_AT = now

    _set_update_status(apt=updates_available)
    return updates_available

MLB_ABBREVIATIONS = {
    # National League
    "Arizona Diamondbacks": "ARI",
    "Diamondbacks": "ARI",
    "D-backs": "ARI",
    "Atlanta Braves": "ATL",
    "Braves": "ATL",
    "Chicago Cubs": "CUBS",
    "Cubs": "CUBS",
    "Cincinnati Reds": "CIN",
    "Reds": "CIN",
    "Colorado Rockies": "COL",
    "Rockies": "COL",
    "Los Angeles Dodgers": "LAD",
    "Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Brewers": "MIL",
    "New York Mets": "NYM",
    "Mets": "NYM",
    "Philadelphia Phillies": "PHI",
    "Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "Pirates": "PIT",
    "San Diego Padres": "SD",
    "Padres": "SD",
    "San Francisco Giants": "SF",
    "Giants": "SF",
    "St. Louis Cardinals": "STL",
    "Cardinals": "STL",
    "Washington Nationals": "WSH",
    "Nationals": "WSH",

    # American League
    "Baltimore Orioles": "BAL",
    "Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Red Sox": "BOS",
    "Chicago White Sox": "SOX",
    "White Sox": "SOX",
    "Cleveland Guardians": "CLE",
    "Guardians": "CLE",
    "Detroit Tigers": "DET",
    "Tigers": "DET",
    "Houston Astros": "HOU",
    "Astros": "HOU",
    "Kansas City Royals": "KC",
    "Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Angels": "LAA",
    "Minnesota Twins": "MIN",
    "Twins": "MIN",
    "New York Yankees": "NYY",
    "Yankees": "NYY",
    "Oakland Athletics": "ATH",
    "Seattle Mariners": "SEA",
    "Mariners": "SEA",
    "Tampa Bay Rays": "TB",
    "Rays": "TB",
    "Texas Rangers": "TEX",
    "Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Blue Jays": "TOR",
    "Las Vegas Athletics": "ATH",
    "Athletics": "ATH",
}

MLB_LOGO_OVERRIDES = {
    "AZ": "ARI",
    "CHC": "CUBS",
    "CWS": "SOX",
    "CHW": "SOX",
    "KCR": "KC",
    "OAK": "ATH",
    "SDP": "SD",
    "SFG": "SF",
    "TAM": "TB",
    "TBR": "TB",
    "WAS": "WSH",
    "WSN": "WSH",
    "ATHLETICS": "ATH",
    "RED SOX": "BOS",
    "REDSOX": "BOS",
    "WHITE SOX": "SOX",
    "WHITESOX": "SOX",
}


def _normalize_mlb_tricode(abbr: str | None) -> str:
    if not isinstance(abbr, str):
        return ""
    normalized = abbr.strip().upper()
    return MLB_LOGO_OVERRIDES.get(normalized, normalized)


def get_mlb_abbreviation(team_name: str) -> str:
    abbr = MLB_ABBREVIATIONS.get(team_name)
    if isinstance(abbr, str):
        return _normalize_mlb_tricode(abbr)
    return str(team_name)


def get_mlb_tricode(team: dict | str | None) -> str:
    if isinstance(team, dict):
        for key in (
            "triCode",
            "tricode",
            "teamTricode",
            "abbreviation",
            "teamCode",
            "fileCode",
        ):
            val = team.get(key)
            normalized = _normalize_mlb_tricode(val)
            if normalized:
                return normalized
        team_name = team.get("name")
        if isinstance(team_name, str):
            return get_mlb_abbreviation(team_name).upper()
        return ""
    if isinstance(team, str):
        return get_mlb_abbreviation(team).upper()
    return ""

# ─── Weather helpers ──────────────────────────────────────────────────────────
def _draw_cloud(draw: ImageDraw.ImageDraw, center: tuple[float, float], radius: float, color: tuple[int, int, int]):
    cx, cy = center
    for dx, dy, scale in [(-radius * 0.8, 0, 1), (0, -radius * 0.5, 1.1), (radius * 0.8, 0, 1)]:
        r = radius * scale
        draw.ellipse(
            (cx + dx - r, cy + dy - r, cx + dx + r, cy + dy + r),
            fill=color,
        )
    draw.rectangle((cx - radius * 1.6, cy, cx + radius * 1.6, cy + radius * 1.1), fill=color)


def _render_sun(size: int) -> Image.Image:
    icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)
    center = size / 2
    radius = size * 0.24
    draw.ellipse((center - radius, center - radius, center + radius, center + radius), fill=(255, 204, 0, 255))
    for i in range(12):
        angle = math.radians(i * 30)
        x0 = center + math.cos(angle) * radius * 1.5
        y0 = center + math.sin(angle) * radius * 1.5
        x1 = center + math.cos(angle) * radius * 2.1
        y1 = center + math.sin(angle) * radius * 2.1
        draw.line((x0, y0, x1, y1), fill=(255, 215, 0, 255), width=max(2, size // 20))
    return icon


def _render_cloudy(size: int, with_sun: bool = False) -> Image.Image:
    icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)
    center = (size * 0.5, size * 0.55)
    radius = size * 0.18
    if with_sun:
        sun = _render_sun(size)
        icon.alpha_composite(sun, (int(size * 0.05), int(size * 0.05)))
    _draw_cloud(draw, center, radius, (220, 220, 220, 255))
    return icon


def _render_precip(size: int, kind: str) -> Image.Image:
    icon = _render_cloudy(size)
    draw = ImageDraw.Draw(icon)
    base_y = int(size * 0.65)
    spacing = size * 0.12
    start_x = size * 0.32
    color = (100, 170, 255, 255)
    for idx in range(3):
        x = start_x + idx * spacing
        if kind == "snow":
            arm = size * 0.05
            draw.line((x, base_y, x, base_y + size * 0.18), fill=color, width=max(1, size // 30))
            draw.line((x - arm, base_y + size * 0.08, x + arm, base_y + size * 0.1), fill=color, width=max(1, size // 30))
            draw.line((x - arm, base_y + size * 0.12, x + arm, base_y + size * 0.14), fill=color, width=max(1, size // 30))
        elif kind == "sleet":
            draw.line((x, base_y, x, base_y + size * 0.18), fill=color, width=max(2, size // 18))
            draw.text((x - size * 0.04, base_y + size * 0.16), "•", font=ImageFont.load_default(), fill=color)
        else:
            draw.line((x, base_y, x, base_y + size * 0.2), fill=color, width=max(2, size // 18))
    if kind == "storm":
        bolt = [
            (size * 0.62, base_y - size * 0.05),
            (size * 0.55, base_y + size * 0.15),
            (size * 0.66, base_y + size * 0.12),
            (size * 0.6, base_y + size * 0.35),
            (size * 0.74, base_y + size * 0.12),
            (size * 0.64, base_y + size * 0.12),
        ]
        draw.polygon(bolt, fill=(255, 204, 0, 255))
    return icon


def _render_fog(size: int) -> Image.Image:
    icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)
    y = size * 0.35
    for _ in range(4):
        draw.rounded_rectangle((size * 0.18, y, size * 0.82, y + size * 0.08), radius=size * 0.04, fill=(200, 200, 200, 200))
        y += size * 0.12
    return icon


def _render_wind(size: int) -> Image.Image:
    icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)
    y = size * 0.35
    for idx in range(3):
        draw.arc((size * 0.18, y - size * 0.05, size * 0.9, y + size * 0.25), start=200, end=350, fill=(180, 220, 255, 255), width=max(2, size // 24))
        y += size * 0.18
    return icon


ICON_RENDERERS = {
    "sunny": _render_sun,
    "partly-cloudy": lambda size: _render_cloudy(size, with_sun=True),
    "cloudy": _render_cloudy,
    "rain": lambda size: _render_precip(size, "rain"),
    "snow": lambda size: _render_precip(size, "snow"),
    "sleet": lambda size: _render_precip(size, "sleet"),
    "storm": lambda size: _render_precip(size, "storm"),
    "fog": _render_fog,
    "wind": _render_wind,
}


@dataclass(frozen=True)
class WeatherConditionInfo:
    display_name: str
    day_emoji: str
    night_emoji: str
    precipitation_type: str
    severity: int


WEATHERKIT_CONDITION_EMOJI: dict[str, WeatherConditionInfo] = {
    "Clear": WeatherConditionInfo("Clear", "☀️", "🌙", "none", 0),
    "MostlyClear": WeatherConditionInfo("Mostly Clear", "🌤️", "🌙✨", "none", 0),
    "PartlyCloudy": WeatherConditionInfo("Partly Cloudy", "⛅", "☁️🌙", "none", 0),
    "MostlyCloudy": WeatherConditionInfo("Mostly Cloudy", "🌥️", "☁️🌙", "none", 0),
    "Cloudy": WeatherConditionInfo("Cloudy", "☁️", "☁️🌙", "none", 0),
    "Foggy": WeatherConditionInfo("Fog", "🌫️", "🌫️🌙", "none", 1),
    "Haze": WeatherConditionInfo("Haze", "🌫️", "🌫️🌙", "none", 1),
    "Smoky": WeatherConditionInfo("Smoke", "🌫️", "🌫️🌙", "none", 1),
    "BlowingDust": WeatherConditionInfo("Blowing Dust", "🌪️", "🌪️🌙", "none", 2),
    "Breezy": WeatherConditionInfo("Breezy", "🍃", "🍃🌙", "none", 0),
    "Windy": WeatherConditionInfo("Windy", "🌬️", "🌬️🌙", "none", 1),
    "Drizzle": WeatherConditionInfo("Drizzle", "🌦️", "🌧️🌙", "rain", 1),
    "Rain": WeatherConditionInfo("Rain", "🌧️", "🌧️🌙", "rain", 1),
    "HeavyRain": WeatherConditionInfo("Heavy Rain", "🌧️🌧️", "🌧️🌧️🌙", "rain", 2),
    "SunShowers": WeatherConditionInfo("Sun Showers", "🌦️", "🌧️🌙", "rain", 1),
    "IsolatedThunderstorms": WeatherConditionInfo("Isolated Thunderstorms", "🌩️", "🌩️🌙", "thunderstorm", 2),
    "ScatteredThunderstorms": WeatherConditionInfo("Scattered Thunderstorms", "⛈️", "⛈️🌙", "thunderstorm", 2),
    "Thunderstorms": WeatherConditionInfo("Thunderstorms", "⛈️", "⛈️🌙", "thunderstorm", 3),
    "StrongStorms": WeatherConditionInfo("Strong Storms", "⛈️⚡", "⛈️⚡🌙", "thunderstorm", 4),
    "Flurries": WeatherConditionInfo("Flurries", "🌨️", "🌨️🌙", "snow", 1),
    "SunFlurries": WeatherConditionInfo("Sun Flurries", "🌨️☀️", "🌨️🌙", "snow", 1),
    "Snow": WeatherConditionInfo("Snow", "❄️", "❄️🌙", "snow", 2),
    "HeavySnow": WeatherConditionInfo("Heavy Snow", "❄️❄️", "❄️❄️🌙", "snow", 3),
    "BlowingSnow": WeatherConditionInfo("Blowing Snow", "🌬️❄️", "🌬️❄️🌙", "snow", 3),
    "Sleet": WeatherConditionInfo("Sleet", "🌨️🌧️", "🌨️🌧️🌙", "wintry_mix", 2),
    "WintryMix": WeatherConditionInfo("Wintry Mix", "🌨️🌧️", "🌨️🌧️🌙", "wintry_mix", 2),
    "FreezingDrizzle": WeatherConditionInfo("Freezing Drizzle", "🌧️🧊", "🌧️🧊🌙", "ice", 3),
    "FreezingRain": WeatherConditionInfo("Freezing Rain", "🌧️🧊", "🌧️🧊🌙", "ice", 4),
    "Hail": WeatherConditionInfo("Hail", "🌨️🧊", "🌨️🧊🌙", "hail", 4),
    "Blizzard": WeatherConditionInfo("Blizzard", "🌨️🌪️", "🌨️🌪️🌙", "snow", 5),
    "Frigid": WeatherConditionInfo("Frigid", "🥶", "🥶🌙", "none", 3),
    "Hot": WeatherConditionInfo("Hot", "🔥", "🔥🌙", "none", 3),
    "TropicalStorm": WeatherConditionInfo("Tropical Storm", "🌀🌧️", "🌀🌧️🌙", "storm", 4),
    "Hurricane": WeatherConditionInfo("Hurricane", "🌀", "🌀🌙", "storm", 5),
}

_WEATHERKIT_CONDITION_ALIASES = {
    "Fog": "Foggy",
    "Hazy": "Haze",
    "MostlySunny": "MostlyClear",
    "Tornado": "BlowingDust",
}

_WEATHERKIT_NIGHT_ICON_NAMES = {
    "Clear": "Clear_night",
    "MostlyClear": "MostlyClear_night",
    "PartlyCloudy": "PartlyCloudy_night",
}


def _resolve_weatherkit_condition(condition_code: Optional[str], icon_code: Optional[str]) -> WeatherConditionInfo:
    raw_code = condition_code or icon_code or ""
    code = str(raw_code).strip()
    if code.endswith("_night"):
        code = code[: -len("_night")]
    code = _WEATHERKIT_CONDITION_ALIASES.get(code, code)
    info = WEATHERKIT_CONDITION_EMOJI.get(code)
    if not info:
        logging.warning("Unknown WeatherKit condition %s; falling back to Cloudy emoji.", code)
        info = WEATHERKIT_CONDITION_EMOJI["Cloudy"]
    return info


def _resolve_daylight(icon_code: Optional[str], is_daylight: Optional[bool]) -> bool:
    if is_daylight is None:
        return not (isinstance(icon_code, str) and icon_code.endswith("_night"))
    return bool(is_daylight)


def _split_emoji_sequence(emoji: str) -> list[str]:
    tokens: list[str] = []
    current = ""
    for char in emoji:
        if char in ("\uFE0F", "\uFE0E", "\u200D"):
            current += char
            continue
        if current:
            tokens.append(current)
        current = char
    if current:
        tokens.append(current)
    return tokens


def _render_emoji_icon(emoji: str, size: int, *, stack_emojis: bool = False) -> Image.Image:
    size = max(1, int(size))
    icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)
    tokens = _split_emoji_sequence(emoji)
    if stack_emojis and len(tokens) > 1:
        count = len(tokens)
        slot_height = max(1, size // count)
        font_size = max(1, int(slot_height * 0.9))
        for idx, token in enumerate(tokens):
            font = get_emoji_font(font_size)
            bbox = draw.textbbox((0, 0), token, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            slot_top = idx * slot_height
            x = (size - text_w) // 2 - bbox[0]
            y = slot_top + (slot_height - text_h) // 2 - bbox[1]
            try:
                draw.text((x, y), token, font=font, fill=(255, 255, 255, 255), **EMOJI_DRAW_KWARGS)
            except TypeError:
                draw.text((x, y), token, font=font, fill=(255, 255, 255, 255))
        return icon

    font = get_emoji_font(size)
    bbox = draw.textbbox((0, 0), emoji, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (size - text_w) // 2 - bbox[0]
    y = (size - text_h) // 2 - bbox[1]
    try:
        draw.text((x, y), emoji, font=font, fill=(255, 255, 255, 255), **EMOJI_DRAW_KWARGS)
    except TypeError:
        draw.text((x, y), emoji, font=font, fill=(255, 255, 255, 255))
    return icon


@log_call
def fetch_weather_icon(
    icon_code: str,
    size: int,
    *,
    condition_code: Optional[str] = None,
    is_daylight: Optional[bool] = None,
    stack_emojis: bool = False,
) -> Image.Image | None:
    if not icon_code and not condition_code:
        return None

    if WEATHER_USE_EMOJI_ICONS:
        info = _resolve_weatherkit_condition(condition_code, icon_code)
        daylight = _resolve_daylight(icon_code, is_daylight)
        emoji = info.day_emoji if daylight else info.night_emoji
        return _render_emoji_icon(emoji, size, stack_emojis=stack_emojis)

    icon_lookup = str(icon_code).strip()
    if not icon_lookup:
        return None

    daylight = _resolve_daylight(icon_lookup, is_daylight)

    alias_map = {
        "sunny": "Clear",
        "partly-cloudy": "PartlyCloudy",
        "cloudy": "Cloudy",
        "rain": "Rain",
        "snow": "Snow",
        "sleet": "Sleet",
        "storm": "Thunderstorms",
        "fog": "Fog",
        "wind": "Windy",
    }
    icon_name = alias_map.get(icon_lookup.lower(), icon_lookup)
    if not daylight:
        icon_name = _WEATHERKIT_NIGHT_ICON_NAMES.get(icon_name, icon_name)

    icon_dir = Path(__file__).resolve().parent / "images" / "WeatherKit"
    candidates = [icon_dir / f"{icon_name}.png"]
    if icon_name != "Cloudy":
        candidates.append(icon_dir / "Cloudy.png")

    for candidate in candidates:
        try:
            if candidate.is_file():
                icon = Image.open(candidate).convert("RGBA")
                if icon.size != (size, size):
                    icon = icon.resize((size, size), Image.ANTIALIAS)
                return icon
        except Exception as exc:  # pragma: no cover - drawing failures are non-fatal
            logging.warning("Weather icon load failed for %s: %s", candidate, exc)

    logging.warning("Weather icon %s not found; returning None", icon_name)
    return None


def uv_index_color(uvi: int) -> tuple[int, int, int]:
    if uvi <= 1:
        return (0, 255, 0)
    if uvi == 2:
        return (200, 120, 255)
    if 3 <= uvi <= 5:
        return (255, 255, 0)
    if 6 <= uvi <= 7:
        return (255, 165, 0)
    if 8 <= uvi <= 10:
        return (255, 0, 0)
    return (128, 0, 128)


def timestamp_to_datetime(value, tz) -> datetime.datetime | None:
    try:
        return datetime.datetime.fromtimestamp(value, tz)
    except Exception:
        return None


def bright_color(min_luma: int = 160) -> tuple[int, int, int]:
    for _ in range(20):
        r = random.randint(80, 255)
        g = random.randint(80, 255)
        b = random.randint(80, 255)
        luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if luma >= min_luma:
            return (r, g, b)
    return (255, 255, 255)


_GH_ICON_CACHE: dict[tuple[int, bool, tuple[str, ...]], Image.Image | None] = {}


def load_github_icon(size: int, invert: bool, paths: list[str]) -> Image.Image | None:
    key = (size, bool(invert), tuple(paths))
    if key in _GH_ICON_CACHE:
        return _GH_ICON_CACHE[key]

    path = next((p for p in paths if os.path.exists(p)), None)
    if not path:
        _GH_ICON_CACHE[key] = None
        return None

    try:
        icon = Image.open(path).convert("RGBA")
        if icon.height != size:
            ratio = size / float(icon.height)
            icon = icon.resize((max(1, int(round(icon.width * ratio))), size), Image.ANTIALIAS)

        if invert:
            r, g, b, a = icon.split()
            rgb_inv = ImageOps.invert(Image.merge("RGB", (r, g, b)))
            icon = Image.merge("RGBA", (*rgb_inv.split(), a))

        _GH_ICON_CACHE[key] = icon
        return icon
    except Exception:
        _GH_ICON_CACHE[key] = None
        return None


def time_strings(now: datetime.datetime) -> tuple[str, str]:
    time_str = now.strftime("%-I:%M")
    am_pm = now.strftime("%p")
    if time_str.startswith("0"):
        time_str = time_str[1:]
    return time_str, am_pm


def date_strings(now: datetime.datetime) -> tuple[str, str]:
    weekday = now.strftime("%A")
    return weekday, f"{now.strftime('%B')} {now.day}, {now.year}"


def decode_html(text: str) -> str:
    try:
        return html.unescape(text)
    except Exception:
        return text


def fetch_directions_routes(
    origin: str,
    destination: str,
    api_key: str,
    *,
    avoid_highways: bool = False,
    avoid_tolls: bool = False,
    url: str,
) -> List[Dict[str, Any]]:
    if not api_key:
        logging.warning("Travel: no GOOGLE_MAPS_API_KEY configured.")
        return []

    params = {
        "origin": origin,
        "destination": destination,
        "alternatives": "true",
        "departure_time": "now",
        "traffic_model": "best_guess",
        "region": "us",
        "key": api_key,
    }
    avoid = []
    if avoid_highways:
        avoid.append("highways")
    if avoid_tolls:
        avoid.append("tolls")
    if avoid:
        params["avoid"] = "|".join(avoid)

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logging.warning("Directions request failed: %s", exc)
        return []

    if payload.get("status") != "OK":
        logging.warning(
            "Directions status=%s, error_message=%s",
            payload.get("status"),
            payload.get("error_message"),
        )
        return []

    routes = payload.get("routes", []) or []
    for route in routes:
        leg = (route.get("legs") or [{}])[0]
        route["_summary"] = decode_html(route.get("summary", "")).lower()
        duration = leg.get("duration_in_traffic") or leg.get("duration") or {}
        route["_duration_text"] = duration.get("text", "")
        route["_duration_sec"] = duration.get("value", 0)
        steps = leg.get("steps", []) or []
        fragments = []
        for step in steps:
            instruction = decode_html(step.get("html_instructions", "")).lower()
            fragments.append(instruction)
        route["_steps_text"] = " ".join(fragments)
    return routes


def route_contains(route: Dict[str, Any], token: str) -> bool:
    token = token.lower()
    return token in route.get("_summary", "") or token in route.get("_steps_text", "")


def choose_route_by_token(routes: List[Dict[str, Any]], token: str) -> Optional[Dict[str, Any]]:
    for route in routes:
        if route_contains(route, token):
            return route
    return None


def choose_route_by_any(routes: List[Dict[str, Any]], tokens: List[str]) -> Optional[Dict[str, Any]]:
    for token in tokens:
        match = choose_route_by_token(routes, token)
        if match:
            return match
    return None


def format_duration_text(route: Optional[Dict[str, Any]]) -> str:
    if not route:
        return "N/A"
    text = route.get("_duration_text") or ""
    return text if text else "N/A"


def fastest_route(routes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not routes:
        return None
    return min(routes, key=lambda r: r.get("_duration_sec", math.inf))
