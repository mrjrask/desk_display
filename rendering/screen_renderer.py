"""Hardware-free screen rendering."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable

from PIL import Image

from config import CENTRAL_TIME
from display_profiles import RenderProfile
from services.data_coordinator import DataSnapshot
from utils import ScreenImage


def _thaw_legacy_data(value: Any) -> Any:
    """Return a mutable copy using the container types legacy screens expect."""

    if isinstance(value, Mapping):
        return {key: _thaw_legacy_data(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_legacy_data(item) for item in value]
    if isinstance(value, frozenset):
        return {_thaw_legacy_data(item) for item in value}
    # Keep the immutable snapshot isolated from mutations to any custom values
    # performed by a legacy renderer.
    try:
        return copy.deepcopy(value)
    except (TypeError, ValueError):
        return value


@dataclass(frozen=True)
class ServerPreferenceSnapshot:
    revision: int
    values: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderArtifact:
    screen_id: str
    profile_id: str
    image: Image.Image
    data_revision: int
    configuration_revision: int
    source_revisions: Mapping[str, int]
    rendered_at: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Motion recorded for a render package; see _CaptureDisplay.
    capture: Mapping[str, Any] | None = None
    recorded_frames: tuple[tuple[Image.Image, float], ...] = ()
    ticker_data: Mapping[str, Any] | None = None

    @property
    def config_revision(self) -> int:
        """Concise alias used by the wire/package layer."""

        return self.configuration_revision

    @property
    def revisions(self) -> Mapping[str, Any]:
        """Machine-readable declaration of every input revision."""

        return {
            "configuration": self.configuration_revision,
            "data": self.data_revision,
            "sources": self.source_revisions,
        }


# Frames an animation capture holds before thinning (see _CaptureDisplay).
_FRAME_BUFFER_PIXELS = 48_000_000


def _sample(items: list[Any], count: int) -> list[Any]:
    """*count* evenly spaced items, always keeping the first and last."""

    if len(items) <= count:
        return list(items)
    span = len(items) - 1
    return [items[round(i * span / (count - 1))] for i in range(count)]


class _CaptureDisplay:
    """Display-shaped sink which only records pixels in memory.

    Besides the still frame, it records motion as data for render packages
    (remote_display/render_package.py): scroll canvases, logo slides, frame
    sequences, quad tiles and ticker lanes arrive through the ``capture_*``
    hooks that utils.package_capture exposes to renderers, and with
    ``record_frames`` every frame written, with the time it was shown, is
    kept (thinned to a bounded count) for finite animations.
    """

    def __init__(self, profile: RenderProfile, *, record_frames: bool = False) -> None:
        from remote_display.render_package import MAX_ANIMATION_FRAMES

        self.width, self.height = profile.width, profile.height
        self.mode = profile.color_mode
        self.current_image = Image.new(self.mode, (self.width, self.height), 0)
        self._frame_id = 0
        self.capture: dict[str, Any] | None = None
        self._record = record_frames
        self._max_frames = max(2, min(MAX_ANIMATION_FRAMES,
                                      _FRAME_BUFFER_PIXELS // (2 * self.width * self.height)))
        self._frames: list[list[Any]] = []

    def image(self, image: Image.Image) -> None:
        self.current_image = image.resize((self.width, self.height)).convert(self.mode).copy()
        self._frame_id += 1
        if self._record and self.capture is None:
            if self._frames and self._frames[-1][1] <= 0:
                self._frames[-1][0] = self.current_image  # never shown; replace it
            else:
                self._frames.append([self.current_image, 0.0])
            if len(self._frames) > 2 * self._max_frames:
                self._frames = self._thin(self._frames)

    @staticmethod
    def _thin(frames: list[list[Any]]) -> list[list[Any]]:
        """Merge neighbours, keeping the later image so the final frame survives."""

        merged = [frames[0]] if len(frames) % 2 else []
        rest = frames[len(merged):]
        for first, second in zip(rest[::2], rest[1::2], strict=True):
            merged.append([second[0], first[1] + second[1]])
        return merged

    def recorded_frames(self) -> list[tuple[Image.Image, float]]:
        frames = self._frames
        while len(frames) > self._max_frames:
            frames = self._thin(frames)
        return [(image, seconds) for image, seconds in frames]

    def show(self) -> None:
        return None

    # Composing an artifact captures frames, it does not play them back.
    # Screens with loops bounded by wall-clock time (tickers, animated quads)
    # check this flag and stop after their first complete frame.
    render_only = True

    def wait_for_skip(self, duration: float) -> bool:
        # Return at once without sleeping or skipping, so frame-bounded
        # animations (drop-ins, scrolls) still run to their final frame.
        if self._record and self._frames and isinstance(duration, int | float):
            self._frames[-1][1] += max(0.0, float(duration))
        return False

    def frame_id(self) -> int:
        return self._frame_id

    # ── Render-package capture hooks ───────────────────────────────────────

    def _set_capture(self, capture: dict[str, Any]) -> None:
        if self.capture is None:  # the screen's first motion defines it
            self.capture = capture

    def capture_scroll(self, *, content_height: int, viewport_width: int, viewport_height: int,
                       render_at_offset: Callable[[int], None], step_px: int, frame_seconds: float,
                       pause_start_seconds: float, pause_end_seconds: float, reverse: bool) -> None:
        from remote_display.render_package import MAX_ASSET_PIXELS

        max_offset = max(0, content_height - viewport_height)
        fits = (viewport_width, viewport_height) == (self.width, self.height)
        if max_offset and fits and self.width * content_height <= MAX_ASSET_PIXELS:
            canvas = Image.new(self.mode, (self.width, content_height), 0)
            for offset in [*range(0, max_offset, viewport_height), max_offset]:
                render_at_offset(offset)
                canvas.paste(self.current_image, (0, offset))
            self._set_capture({
                "kind": "scroll", "canvas": canvas, "step_px": max(1, min(int(step_px), self.height)),
                "frame_seconds": frame_seconds, "pause_start_seconds": pause_start_seconds,
                "pause_end_seconds": pause_end_seconds, "direction": "up" if reverse else "down",
            })
        render_at_offset(max_offset if reverse else 0)

    def capture_slide(self, *, sprite: Image.Image, y: int, speed_px_per_second: float,
                      background: tuple[int, ...]) -> None:
        self._set_capture({"kind": "slide", "sprite": sprite.copy(), "y": int(y),
                           "speed_px_per_second": float(speed_px_per_second),
                           "background": tuple(background[:3])})
        frame = Image.new("RGB", (self.width, self.height), tuple(background[:3]))
        position = ((self.width - sprite.width) // 2, int(y))
        frame.paste(sprite, position, sprite if "A" in sprite.getbands() else None)
        self.image(frame)

    def capture_frames(self, frames: list[Image.Image], *, frame_seconds: float, loops: int = 1) -> None:
        from remote_display.render_package import MAX_ANIMATION_FRAMES

        kept = _sample(list(frames), MAX_ANIMATION_FRAMES)
        seconds = float(frame_seconds) * len(frames) / max(1, len(kept))
        self._set_capture({"kind": "frames", "frames": [(f, seconds) for f in kept], "loops": int(loops)})
        self.image(kept[-1])

    def capture_composite(self, *, tiles: list[dict[str, Any]], frame_seconds: float,
                          duration_seconds: float) -> None:
        from remote_display.render_package import MAX_TILE_FRAMES

        self._set_capture({
            "kind": "composite", "frame_seconds": float(frame_seconds), "duration_seconds": float(duration_seconds),
            "tiles": [{**tile, "frames": _sample(tile["frames"], MAX_TILE_FRAMES)} for tile in tiles],
        })

    def capture_ticker(self, *, base: Image.Image, lanes: list[dict[str, Any]], duration_seconds: float) -> None:
        self._set_capture({"kind": "ticker", "base": base, "lanes": lanes,
                           "duration_seconds": float(duration_seconds)})


class ScreenRenderer:
    """Compose one artifact without touching hardware or scheduler state."""

    def __init__(self, registry_factory: Callable[..., Any] | None = None) -> None:
        self._registry_factory = registry_factory

    def render(
        self,
        screen_id: str,
        profile: RenderProfile,
        preferences: ServerPreferenceSnapshot,
        data: DataSnapshot,
        *,
        record_frames: bool = False,
    ) -> RenderArtifact:
        capture = _CaptureDisplay(profile, record_frames=record_frames)
        if self._registry_factory is None:
            from screens.registry import ScreenContext, build_screen_registry

            now = datetime.now(CENTRAL_TIME)
            context = ScreenContext(
                display=capture,
                cache=_thaw_legacy_data(data.values),
                logos=preferences.values.get("logos", {}),
                image_dir=str(preferences.values.get("image_dir", "images")),
                now=now,
                now_utc=now.astimezone(UTC),
                offline=bool(preferences.values.get("offline", False)),
                weather_fetched_at=preferences.values.get("weather_fetched_at"),
                skip_scoreboards=bool(preferences.values.get("skip_scoreboards", False)),
                render_profile=profile,
                allow_upstream_requests=False,
            )
            registry, _ = build_screen_registry(context)
        else:
            registry = self._registry_factory(capture, profile, preferences, data)
            if isinstance(registry, tuple):
                registry = registry[0]

        definition = registry.get(screen_id)
        if definition is None or not definition.available:
            raise KeyError(f"Screen is not available: {screen_id}")
        result = definition.render()
        metadata: dict[str, Any] = dict(getattr(definition, "metadata", {}) or {})
        if isinstance(result, ScreenImage):
            image = capture.current_image if result.displayed else result.image
            metadata["consumed_delay"] = bool(result.consumed_delay)
        elif isinstance(result, Image.Image):
            image = result
        elif result is None:
            image = capture.current_image
        else:
            raise TypeError(f"Screen {screen_id!r} returned unsupported artifact {type(result)!r}")
        image = image.resize((profile.width, profile.height)).convert(profile.color_mode).copy()
        return RenderArtifact(
            screen_id=screen_id,
            profile_id=profile.profile_id,
            image=image,
            data_revision=data.revision,
            configuration_revision=preferences.revision,
            source_revisions=data.source_revisions,
            rendered_at=datetime.now(UTC),
            metadata=metadata,
            capture=capture.capture,
            recorded_frames=tuple(capture.recorded_frames()) if record_frames else (),
            ticker_data=getattr(result, "ticker_data", None),
        )
