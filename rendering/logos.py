"""Logo loading for renders at any display profile.

The standalone display loop (``main.py``) sizes its logos once, for its own
display.  A render server renders many profiles, so this module applies the
same sizing rules per profile and caches the result per profile.
"""
from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path

from PIL import Image

LOGGER = logging.getLogger("desk_display.logos")
IMAGES_DIR = str(Path(__file__).resolve().parents[1] / "images")

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - Pillow < 9.1
    LANCZOS = Image.LANCZOS


def logo_dimensions(width: int, height: int) -> tuple[int, int, int]:
    """Return ``(logo_height, team_logo_height, logo_width)`` like ``main.py``."""

    limit = height - 30
    if width >= 1280 or height >= 720:
        limit = min(limit, round(max(1, min(width, height)) * 0.55))
    logo_height = max(1, limit)
    return logo_height, logo_height, max(1, min(width, round(logo_height * 1.5)))


def load_logo(images_dir: str, filename: str, *, height: int, width: int) -> Image.Image | None:
    """Load a logo centered on a fixed ``width`` x ``height`` canvas."""

    path = os.path.join(images_dir, filename)
    try:
        with Image.open(path) as img:
            transparent = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
            mode = "RGBA" if transparent else "RGB"
            img = img.convert(mode)
            if img.width == 0 or img.height == 0:
                return None
            scale = min(width / img.width, height / img.height)
            size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
            resized = img.resize(size, LANCZOS)
            if size == (width, height):
                return resized
            canvas = Image.new(mode, (width, height), (0, 0, 0, 0) if transparent else (0, 0, 0))
            offset = ((width - size[0]) // 2, (height - size[1]) // 2)
            canvas.paste(resized, offset, resized if transparent else None)
            return canvas
    except Exception as exc:  # noqa: BLE001 - a missing logo must not stop a render
        LOGGER.warning("Logo load failed '%s': %s", filename, exc)
        return None


def logo_loaders(
    width: int,
    height: int,
    *,
    images_dir: str = IMAGES_DIR,
    ahl_tricode: str | None = None,
) -> dict[str, Callable[[], Image.Image | None]]:
    """The named logos screens read from ``ScreenContext.logos``."""

    logo_h, team_h, logo_w = logo_dimensions(width, height)

    def logo(filename: str, logo_height: int = logo_h) -> Callable[[], Image.Image | None]:
        return lambda: load_logo(images_dir, filename, height=logo_height, width=logo_w)

    def wolves() -> Image.Image | None:
        tricode = (ahl_tricode or "CHI").strip() or "CHI"
        for variant in {tricode.upper(), tricode.lower()}:
            image = load_logo(images_dir, f"ahl/{variant}.png", height=team_h, width=logo_w)
            if image:
                return image
        return load_logo(images_dir, "wolves.jpg", height=team_h, width=logo_w)

    return {
        "weather logo": logo("weather.jpg"),
        "verano logo": logo("verano.jpg"),
        "bears logo": logo("nfl/chi.png"),
        "nfl logo": logo("nfl/nfl.png"),
        "hawks logo": logo("nhl/CHI.png", team_h),
        "nhl logo": logo("nhl/nhl.png"),
        "wolves logo": wolves,
        "cubs logo": logo("mlb/CUBS.png", team_h),
        "sox logo": logo("mlb/SOX.png", team_h),
        "mlb logo": logo("mlb/MLB.png"),
        "nba logo": logo("nba/NBA.png"),
        "bulls logo": logo("nba/CHI.png", team_h),
    }


class LogoCache:
    """Lazily loaded logos for one size; safe to share between render workers."""

    def __init__(self, loaders: dict[str, Callable[[], Image.Image | None]]) -> None:
        self._loaders = loaders
        self._cache: dict[str, Image.Image | None] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> Image.Image | None:
        with self._lock:
            if name in self._cache:
                return self._cache[name]
        loader = self._loaders.get(name)
        image = loader() if loader else None
        with self._lock:
            self._cache.setdefault(name, image)
            return self._cache[name]


class ProfileLogos:
    """One :class:`LogoCache` per render size."""

    def __init__(self, *, images_dir: str = IMAGES_DIR, ahl_tricode: str | None = None) -> None:
        self.images_dir = images_dir
        self.ahl_tricode = ahl_tricode
        self._caches: dict[tuple[int, int], LogoCache] = {}
        self._lock = threading.Lock()

    def for_size(self, width: int, height: int) -> LogoCache:
        with self._lock:
            cache = self._caches.get((width, height))
            if cache is None:
                cache = LogoCache(logo_loaders(width, height, images_dir=self.images_dir,
                                               ahl_tricode=self.ahl_tricode))
                self._caches[(width, height)] = cache
            return cache


__all__ = ["IMAGES_DIR", "LogoCache", "ProfileLogos", "load_logo", "logo_dimensions", "logo_loaders"]
