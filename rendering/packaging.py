"""Turn what the server captured while rendering into a render package."""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from PIL import Image

from display_profiles import RenderProfile
from remote_display.render_package import PackageBuilder, validate_package
from rendering.screen_classes import CLASSIFICATIONS, ScreenClass

LOGGER = logging.getLogger("desk_display.packaging")


def _package(key: Any, profile: RenderProfile, screen: ScreenClass, builder: PackageBuilder, kind: str,
             body: Mapping[str, Any]) -> dict[str, Any]:
    package = builder.build(
        screen_id=key.screen_id, render_profile=profile.profile_id, width=profile.width,
        height=profile.height, color_mode=profile.color_mode, render_key_digest=key.digest,
        classification=screen.kind, kind=kind, body=body,
    )
    return validate_package(package, key=key, verify_assets=False)


def _source_mode(profile: RenderProfile) -> str:
    """The mode for images the client composes into frames.

    A 1-bit display dithers each colour frame it shows, so the parts of a
    moving frame (a scroll canvas, ticker strips, quad tiles) stay in colour
    and the client dithers the composed frame. Dithering the whole canvas
    once, or going through greyscale first, would give scrolled frames a
    different pattern from the standalone display's.
    """

    return "RGB" if profile.color_mode == "1" else profile.color_mode


def _frame(image: Image.Image, profile: RenderProfile, mode: str | None = None) -> Image.Image:
    if image.size != (profile.width, profile.height):
        image = image.resize((profile.width, profile.height))
    return image.convert(mode or profile.color_mode)


def clock_package(key: Any, profile: RenderProfile, layout: Mapping[str, Any],
                  background: Image.Image) -> dict[str, Any]:
    builder = PackageBuilder()
    body = {"background": builder.add(_frame(background, profile)), "layout": dict(layout)}
    return _package(key, profile, CLASSIFICATIONS[key.screen_id], builder, "clock", body)


def build_package(key: Any, profile: RenderProfile, artifact: Any) -> dict[str, Any] | None:
    """The package for a rendered screen, or ``None`` when its still is enough.

    A screen whose content fits (a scoreboard with few games), or that did
    not move this time, has no package; the client shows the still image.
    """

    screen = CLASSIFICATIONS.get(key.screen_id)
    if screen is None or screen.package_kind in (None, "clock"):
        return None
    capture = artifact.capture or {}
    builder = PackageBuilder()
    kind = screen.package_kind
    if kind == "scroll" and capture.get("kind") == "scroll":
        body = {
            "canvas": builder.add(capture["canvas"].convert(_source_mode(profile))),
            "viewport": [profile.width, profile.height],
            **{k: capture[k] for k in ("step_px", "frame_seconds", "pause_start_seconds",
                                       "pause_end_seconds", "direction")},
        }
    elif kind == "ticker" and capture.get("kind") == "ticker":
        lanes = [{
            "bounds": [int(v) for v in lane["bounds"]],
            "strip": builder.add(lane["strip"].convert(_source_mode(profile))),
            "speed_px_per_second": round(float(lane["speed_px_per_second"]), 3),
            "offset_px": round(float(lane["offset_px"]), 3),
            "background": [int(c) for c in lane["background"]],
        } for lane in capture["lanes"]]
        if not lanes:
            return None
        body = {"base": builder.add(_frame(capture["base"], profile, _source_mode(profile))), "lanes": lanes,
                "duration_seconds": capture["duration_seconds"]}
    elif kind == "animation" and capture.get("kind") == "slide":
        body = {"slide": {
            "sprite": builder.add(capture["sprite"]),
            "y": capture["y"],
            "speed_px_per_second": round(capture["speed_px_per_second"], 3),
            "background": [int(c) for c in capture["background"]],
        }}
    elif kind == "animation":
        if capture.get("kind") == "frames":
            frames, loops = capture["frames"], capture["loops"]
        else:
            frames, loops = list(artifact.recorded_frames), 1
        if len(frames) < 2:
            return None
        body = {"frames": [{"asset": builder.add(_frame(image, profile)),
                            "duration_ms": int(round(seconds * 1000))} for image, seconds in frames],
                "loops": max(1, int(loops))}
    elif kind == "composite" and capture.get("kind") == "composite":
        tiles = []
        for tile in capture["tiles"]:
            left, top, right, bottom = (int(v) for v in tile["bounds"])
            size = (right - left, bottom - top)
            label = tile.get("label")
            focus = label if screen.interactive and label in CLASSIFICATIONS and label != key.screen_id else None
            tiles.append({
                "bounds": [left, top, right, bottom],
                "frames": [builder.add(f.resize(size).convert(_source_mode(profile))) for f in tile["frames"]],
                "focus_screen": focus,
            })
        body = {"base": builder.add(_frame(artifact.image, profile, _source_mode(profile))), "tiles": tiles,
                "frame_seconds": capture["frame_seconds"], "duration_seconds": capture["duration_seconds"]}
    else:
        return None
    return _package(key, profile, screen, builder, kind, body)


__all__ = ["build_package", "clock_package"]
