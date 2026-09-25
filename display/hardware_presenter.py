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

    def close(self) -> None:
        for name in ("cleanup", "close"):
            method = getattr(self.display, name, None)
            if callable(method):
                method()
                return
