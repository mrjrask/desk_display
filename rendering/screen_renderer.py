"""Hardware-free screen rendering."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable

from PIL import Image

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


class _CaptureDisplay:
    """Display-shaped sink which only records pixels in memory."""

    def __init__(self, profile: RenderProfile) -> None:
        self.width, self.height = profile.width, profile.height
        self.mode = profile.color_mode
        self.current_image = Image.new(self.mode, (self.width, self.height), 0)
        self._frame_id = 0

    def image(self, image: Image.Image) -> None:
        self.current_image = image.resize((self.width, self.height)).convert(self.mode).copy()
        self._frame_id += 1

    def show(self) -> None:
        return None

    def frame_id(self) -> int:
        return self._frame_id


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
    ) -> RenderArtifact:
        capture = _CaptureDisplay(profile)
        if self._registry_factory is None:
            from screens.registry import ScreenContext, build_screen_registry

            context = ScreenContext(
                display=capture,
                cache=_thaw_legacy_data(data.values),
                logos=preferences.values.get("logos", {}),
                image_dir=str(preferences.values.get("image_dir", "images")),
                now=datetime.now().astimezone(),
                now_utc=datetime.now(UTC),
                offline=bool(preferences.values.get("offline", False)),
                weather_fetched_at=preferences.values.get("weather_fetched_at"),
                skip_scoreboards=bool(preferences.values.get("skip_scoreboards", False)),
                render_profile=profile,
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
            image = result.image
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
        )
