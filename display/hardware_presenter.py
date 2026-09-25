"""The hardware boundary for final presentation and local input."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PIL import Image

from display_profiles import RenderProfile
from rendering.screen_renderer import RenderArtifact


class HardwarePresenter:
    """Adapt profile-native artifacts to a physical ``utils.Display``."""

    def __init__(self, display: Any | None = None, *, profile: RenderProfile | None = None) -> None:
        if display is None:
            from utils import Display

            display = Display()
        self.display = display
        self.profile = profile

    def convert(self, image: Image.Image) -> Image.Image:
        profile = self.profile
        if profile is None:
            return image.copy()
        # ``utils.Display`` owns the physical output transform. Frames passed
        # here must remain at logical dimensions to avoid resize and rotation
        # being applied a second time by the selected output driver.
        return image.resize((profile.width, profile.height)).convert(profile.color_mode)

    def present(self, artifact: RenderArtifact | Image.Image) -> Image.Image:
        image = artifact.image if isinstance(artifact, RenderArtifact) else artifact
        converted = self.convert(image)
        self.display.image(converted)
        show = getattr(self.display, "show", None)
        if callable(show):
            show()
        return converted

    def set_backlight(self, level: float) -> float:
        return float(self.display.set_backlight(level))

    def is_button_pressed(self, name: str) -> bool:
        return bool(self.display.is_button_pressed(name))

    def set_button_callback(self, callback: Callable[[str], None] | None) -> None:
        self.display.set_button_callback(callback)

    @property
    def rotation(self) -> int:
        """The rotation the output driver applies (after the double-rotation guard)."""

        return int(getattr(self.display, "rotation", 0) or 0) % 360

    def touch_to_logical(self, x: float, y: float) -> tuple[int, int]:
        """Map a panel touch point into the profile's logical coordinates."""

        from display.rotation import to_logical

        profile = self.profile
        if profile is None:
            return int(x), int(y)
        return to_logical(x, y, profile.width, profile.height, self.rotation)

    def poll_touch(self) -> Any:
        poll = getattr(self.display, "poll_touch", None)
        return poll() if callable(poll) else None

    def poll_taps(self) -> list[tuple[float, float]]:
        """Taps since the last poll, as panel (physical) pixel points.

        Reads the SDL event queue of window and kernel outputs; SPI panels
        without a touch layer report nothing.
        """

        display = self.display
        pygame = getattr(display, "_pygame", None)
        event_api = getattr(pygame, "event", None)
        get = getattr(event_api, "get", None)
        finger = getattr(pygame, "FINGERDOWN", None)
        mouse = getattr(pygame, "MOUSEBUTTONDOWN", None)
        types = [t for t in (finger, mouse) if isinstance(t, int)]
        if not callable(get) or not types:
            return []
        try:
            events = get(types)
        except Exception:  # noqa: BLE001 - a closed or headless SDL session has no input
            return []
        panel_w = float(getattr(display, "render_width", 0) or getattr(display, "width", 0) or 1)
        panel_h = float(getattr(display, "render_height", 0) or getattr(display, "height", 0) or 1)
        window_w = float(getattr(display, "screen_width", 0) or panel_w)
        window_h = float(getattr(display, "screen_height", 0) or panel_h)
        taps: list[tuple[float, float]] = []
        for event in events or ():
            kind = getattr(event, "type", None)
            if finger is not None and kind == finger:
                taps.append((float(getattr(event, "x", 0.0)) * panel_w, float(getattr(event, "y", 0.0)) * panel_h))
            elif mouse is not None and kind == mouse and getattr(event, "button", 1) == 1:
                pos = getattr(event, "pos", None)
                if pos is not None and len(pos) >= 2:
                    taps.append((float(pos[0]) * panel_w / window_w, float(pos[1]) * panel_h / window_h))
        return taps

    def close(self) -> None:
        for name in ("cleanup", "close"):
            method = getattr(self.display, name, None)
            if callable(method):
                method()
                return
