"""Play one render package locally, from the client's cache.

A :class:`PackagePlayback` turns a validated package (see
remote_display/render_package.py) into frames as a pure function of the
time since the screen started, so playback never contacts the server and
tests can step through it deterministically. It imports no provider or
server rendering code; only clock packages draw with the shared clock
faces, because the time is exactly what the server cannot send ahead.
"""
from __future__ import annotations

import datetime as dt
import math
import random
from collections.abc import Callable, Hashable, Mapping
from typing import Any

from PIL import Image

from display_profiles import RenderProfile
from remote_display.render_package import asset_image

DEFAULT_FRAME_SECONDS = 1 / 30
TICKER_FRAME_SECONDS = 0.045


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class PackagePlayback:
    """Frames for one showing of a screen.

    ``hold_seconds`` is the playlist's time for the screen: a finite motion
    (scroll, logo slide, frames) plays once and then holds its last frame
    that long, as the standalone display does; a ticker, quad or clock runs
    for its own window or the hold, whichever is longer.
    """

    def __init__(
        self,
        package: Mapping[str, Any],
        profile: RenderProfile,
        *,
        hold_seconds: float,
        clock: Callable[[], dt.datetime] = _utc_now,
        rng: random.Random | None = None,
        ip_text: Callable[[], str | None] | None = None,
    ) -> None:
        self.package = package
        self.profile = profile
        self.kind = str(package["kind"])
        self.body: Mapping[str, Any] = package[self.kind]
        self.hold_seconds = max(0.0, float(hold_seconds))
        self._clock = clock
        self._rng = rng or random.Random()
        self._ip_text = ip_text
        self._images: dict[str, Image.Image] = {}
        self._last: tuple[Hashable, Image.Image] | None = None
        self._colors: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None
        self.direction = "ltr"
        if self.kind == "animation" and "slide" in self.body:
            self.direction = self._rng.choice(("ltr", "rtl"))

    # Assets

    def _image(self, asset_id: str) -> Image.Image:
        image = self._images.get(asset_id)
        if image is None:
            image = asset_image(self.package, asset_id)
            self._images[asset_id] = image
        return image

    def _mode(self, image: Image.Image) -> Image.Image:
        return image if image.mode == self.profile.color_mode else image.convert(self.profile.color_mode)

    # Timing

    @property
    def motion_seconds(self) -> float:
        """How long the package's own motion lasts (0 when it only holds)."""

        body = self.body
        if self.kind == "scroll":
            canvas = self.package["assets"][body["canvas"]]
            steps = math.ceil((canvas["height"] - self.profile.height) / body["step_px"])
            return body["pause_start_seconds"] + steps * body["frame_seconds"] + body["pause_end_seconds"]
        if self.kind == "animation" and "frames" in body:
            return sum(f["duration_ms"] for f in body["frames"]) / 1000 * body["loops"]
        if self.kind == "animation":
            sprite = self.package["assets"][body["slide"]["sprite"]]
            return (self.profile.width + sprite["width"]) / body["slide"]["speed_px_per_second"]
        if self.kind in ("ticker", "composite"):
            return float(body["duration_seconds"])
        return 0.0

    @property
    def duration(self) -> float:
        if self.kind in ("ticker", "composite", "clock"):
            return max(self.motion_seconds, self.hold_seconds)
        return self.motion_seconds + self.hold_seconds

    @property
    def frame_seconds(self) -> float:
        """How often the picture can change."""

        body = self.body
        if self.kind == "scroll":
            return max(0.01, float(body["frame_seconds"]))
        if self.kind == "ticker":
            return TICKER_FRAME_SECONDS
        if self.kind == "composite":
            return max(0.03, float(body["frame_seconds"]))
        if self.kind == "clock":
            return 1.0 if body["layout"]["face"] == "nixie" else 5.0
        if "frames" in body:
            return max(0.03, min(f["duration_ms"] for f in body["frames"]) / 1000)
        return DEFAULT_FRAME_SECONDS

    # Frames

    def key_at(self, t: float) -> Hashable:
        """Changes exactly when the frame at *t* differs from the previous one."""

        t = max(0.0, t)
        body = self.body
        if self.kind == "scroll":
            return ("scroll", self._scroll_offset(t))
        if self.kind == "ticker":
            return ("ticker", int(t / TICKER_FRAME_SECONDS))
        if self.kind == "composite":
            return ("composite", int(t / self.frame_seconds))
        if self.kind == "clock":
            now = self._clock()
            return ("clock", now.replace(microsecond=0) if body["layout"]["face"] == "nixie"
                    else now.replace(second=0, microsecond=0))
        if "frames" in body:
            return ("frame", self._frame_index(t))
        return ("slide", self._slide_x(t))

    def frame_at(self, t: float) -> Image.Image:
        key = self.key_at(t)
        if self._last is not None and self._last[0] == key:
            return self._last[1]
        image = self._mode(self._draw(max(0.0, t), key))
        self._last = (key, image)
        return image

    def _scroll_offset(self, t: float) -> int:
        body = self.body
        canvas = self.package["assets"][body["canvas"]]
        max_offset = canvas["height"] - self.profile.height
        moving = t - body["pause_start_seconds"]
        travelled = 0 if moving <= 0 else min(max_offset, int(moving / body["frame_seconds"]) * body["step_px"])
        return max_offset - travelled if body["direction"] == "up" else travelled

    def _frame_index(self, t: float) -> int:
        frames = self.body["frames"]
        loop = sum(f["duration_ms"] for f in frames) / 1000
        if loop <= 0 or t >= loop * self.body["loops"]:
            return len(frames) - 1
        t %= loop
        for index, frame in enumerate(frames):
            t -= frame["duration_ms"] / 1000
            if t < 0:
                return index
        return len(frames) - 1

    def _slide_x(self, t: float) -> int | None:
        """Sprite x position, or ``None`` once it has crossed (then centred)."""

        slide = self.body["slide"]
        width = self.package["assets"][slide["sprite"]]["width"]
        travelled = t * slide["speed_px_per_second"]
        if travelled >= self.profile.width + width:
            return None
        return int(-width + travelled) if self.direction == "ltr" else int(self.profile.width - travelled)

    def _draw(self, t: float, key: Hashable) -> Image.Image:
        body = self.body
        size = (self.profile.width, self.profile.height)
        if self.kind == "scroll":
            offset = key[1]
            return self._image(body["canvas"]).crop((0, offset, size[0], offset + size[1]))
        if self.kind == "ticker":
            frame = self._image(body["base"]).copy()
            for lane in body["lanes"]:
                left, top, right, bottom = lane["bounds"]
                strip = self._image(lane["strip"])
                offset = int(lane["offset_px"] + lane["speed_px_per_second"] * t) % strip.width
                window = Image.new(strip.mode, (right - left, bottom - top))
                x = -offset
                while x < window.width:
                    window.paste(strip, (x, 0))
                    x += strip.width
                frame.paste(window.convert(frame.mode), (left, top))
            return frame
        if self.kind == "composite":
            frame = self._image(body["base"]).copy()
            step = key[1]
            for tile in body["tiles"]:
                frames = tile["frames"]
                left, top, _right, _bottom = tile["bounds"]
                frame.paste(self._image(frames[step % len(frames)]), (left, top))
            return frame
        if self.kind == "clock":
            from rendering.clock_faces import render_clock

            if self._colors is None:
                from utils import bright_color

                self._colors = (bright_color(), bright_color())
            ip = self._ip_text() if self._ip_text is not None else None
            return render_clock(body["layout"], self.profile, self._clock(), ip_text=ip, colors=self._colors)
        if "frames" in body:
            return self._image(body["frames"][key[1]]["asset"])
        slide = body["slide"]
        sprite = self._image(slide["sprite"])
        frame = Image.new("RGB", size, tuple(slide["background"][:3]))
        x = key[1]
        if x is None:
            x = (size[0] - sprite.width) // 2
        frame.paste(sprite, (x, slide["y"]), sprite if "A" in sprite.getbands() else None)
        return frame

    # Interaction

    def focus_targets(self) -> dict[str, tuple[int, int, int, int]]:
        """Tile bounds (logical) that open a screen full screen when tapped."""

        if self.kind != "composite":
            return {}
        return {tile["focus_screen"]: tuple(tile["bounds"]) for tile in self.body["tiles"]
                if tile.get("focus_screen")}


__all__ = ["PackagePlayback"]
