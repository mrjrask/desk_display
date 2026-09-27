"""Pure screen composition."""
from __future__ import annotations

from typing import Any

__all__ = ["RenderArtifact", "ScreenRenderer", "ServerPreferenceSnapshot"]


def __getattr__(name: str) -> Any:
    # Imported on first use: screen_renderer imports config, which must not
    # load before a display client has configured it for its panel
    # (rendering.profile_process.configure_native).
    if name in __all__:
        from rendering import screen_renderer

        return getattr(screen_renderer, name)
    raise AttributeError(f"module 'rendering' has no attribute {name!r}")
