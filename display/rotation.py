"""Physical rotation, applied only at final presentation.

Every render profile is defined in its canonical logical orientation, and
artifacts, render keys and server scheduling never depend on how a panel is
mounted. A client rotates each frame just before driver output and maps
touch coordinates from the panel back into logical coordinates, so two
identical displays mounted differently share the same artifacts.

Rotation follows ``PIL.Image.rotate``: positive degrees turn the frame
counter-clockwise, the same transform ``utils.Display`` applies.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from PIL import Image

LOGGER = logging.getLogger("desk_display.rotation")
ROTATIONS = (0, 90, 180, 270)


def parse_rotation(value: object) -> int:
    """Accept 0/90/180/270 or the legacy 0-3 form; raise ``ValueError`` otherwise."""

    try:
        number = int(str(value).strip()) if value is not None and str(value).strip() else 0
    except ValueError:
        raise ValueError(f"invalid rotation {value!r}; expected 0, 90, 180, 270 or 0-3") from None
    if number in (1, 2, 3):
        number *= 90
    if number not in ROTATIONS:
        raise ValueError(f"invalid rotation {value!r}; expected 0, 90, 180, 270 or 0-3")
    return number


@dataclass(frozen=True)
class RotationDecision:
    """The rotation the application applies, and why."""

    configured: int
    kernel_overlay: int | None
    applied: int
    reason: str

    def describe(self) -> str:
        kernel = "none" if self.kernel_overlay is None else f"{self.kernel_overlay}"
        return f"rotation {self.applied} (configured {self.configured}, kernel overlay {kernel})"


def resolve_rotation(configured: int, *, kernel_overlay: int | None, strict: bool) -> RotationDecision:
    """Apply the double-rotation guardrail.

    When a kernel overlay already rotates the panel and strict mode is on,
    the application does not rotate again. Without strict mode both apply,
    which is the documented legacy behavior.
    """

    if kernel_overlay is not None and configured and strict:
        return RotationDecision(configured, kernel_overlay, 0, "kernel overlay already rotates; strict mode")
    if kernel_overlay is not None and configured:
        return RotationDecision(configured, kernel_overlay, configured, "kernel overlay and application both rotate")
    return RotationDecision(configured, kernel_overlay, configured, "configured")


def physical_size(width: int, height: int, rotation: int) -> tuple[int, int]:
    return (height, width) if rotation in (90, 270) else (width, height)


def rotate_frame(image: Image.Image, rotation: int) -> Image.Image:
    """The frame as the panel shows it."""

    return image.rotate(rotation, expand=True) if rotation else image


def to_physical(x: int, y: int, width: int, height: int, rotation: int) -> tuple[int, int]:
    """Where logical pixel ``(x, y)`` lands on the panel."""

    if rotation == 90:
        return y, width - 1 - x
    if rotation == 180:
        return width - 1 - x, height - 1 - y
    if rotation == 270:
        return height - 1 - y, x
    return x, y


def to_logical(px: float, py: float, width: int, height: int, rotation: int) -> tuple[int, int]:
    """Map a panel touch point back into logical coordinates, clamped to the frame."""

    px, py = int(px), int(py)
    if rotation == 90:
        x, y = width - 1 - py, px
    elif rotation == 180:
        x, y = width - 1 - px, height - 1 - py
    elif rotation == 270:
        x, y = py, height - 1 - px
    else:
        x, y = px, py
    return min(max(x, 0), width - 1), min(max(y, 0), height - 1)


def hit_test(bounds: dict[str, tuple[int, int, int, int]], x: int, y: int) -> str | None:
    """Return the target whose logical ``(left, top, right, bottom)`` contains the point."""

    for name, (left, top, right, bottom) in bounds.items():
        if left <= x < right and top <= y < bottom:
            return name
    return None


__all__ = [
    "ROTATIONS",
    "RotationDecision",
    "hit_test",
    "parse_rotation",
    "physical_size",
    "resolve_rotation",
    "rotate_frame",
    "to_logical",
    "to_physical",
]
