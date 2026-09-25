"""Production renderer and revision source for the render coordinator.

Renders with :class:`rendering.screen_renderer.ScreenRenderer` from the
shared :mod:`services.data_coordinator` snapshot, so rendering never calls an
upstream API.  Revisions:

* ``style_revision`` hashes the style and layout documents, so editing them
  rerenders every screen.
* ``data_revision`` is the data coordinator's snapshot revision.  Phase 14b
  narrows this to the feeds each screen actually uses.
* ``renderer_revision`` is the application version, so an upgrade rerenders.
"""
from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from protocol_versions import APPLICATION_VERSION
from remote_display.models import RenderKey, ScreenRevisions
from remote_display.render_coordinator import RenderOutput

DEFAULT_REFRESH_SECONDS = 300
# Screens that show the time must be refreshed every minute.
CLOCK_SCREENS = frozenset({"date", "nixie"})
CLOCK_REFRESH_SECONDS = 60


class StyleRevision:
    """Content hash of the style and layout documents, cached on mtime."""

    def __init__(self, paths: Iterable[Path] | None = None) -> None:
        self._paths = None if paths is None else tuple(Path(p) for p in paths)
        self._lock = threading.Lock()
        self._signature: tuple | None = None
        self._revision = "s-none"

    def _resolve(self) -> tuple[Path, ...]:
        if self._paths is not None:
            return self._paths
        import paths

        return (paths.resolve_style_config_path(), paths.resolve_layouts_config_path())

    def __call__(self) -> str:
        files = self._resolve()
        signature = tuple((str(p), *_stat(p)) for p in files)
        with self._lock:
            if signature != self._signature:
                digest = hashlib.sha256()
                for path in files:
                    digest.update(str(path).encode("utf-8") + b"\0")
                    try:
                        digest.update(path.read_bytes())
                    except OSError:
                        digest.update(b"<missing>")
                    digest.update(b"\0")
                self._signature = signature
                self._revision = f"s-{digest.hexdigest()[:16]}"
            return self._revision


def _stat(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
    except OSError:
        return (-1, -1)
    return (stat.st_mtime_ns, stat.st_size)


class ServerRendering:
    """Revision source, renderer and data health for :class:`RenderCoordinator`."""

    def __init__(self, data_coordinator: Any = None, style_revision: StyleRevision | None = None) -> None:
        if data_coordinator is None:
            from services.data_coordinator import coordinator as data_coordinator
        self.data = data_coordinator
        self.style_revision = style_revision or StyleRevision()

    def revisions(self, screens: Iterable[str]) -> Mapping[str, ScreenRevisions]:
        revisions = ScreenRevisions(
            style_revision=self.style_revision(),
            data_revision=f"d{self.data.snapshot().revision}",
            renderer_revision=f"v{APPLICATION_VERSION}",
        )
        return {screen: revisions for screen in screens}

    def render(self, key: RenderKey) -> RenderOutput:
        from display_profiles import PROFILE_PRESETS
        from rendering.screen_renderer import ScreenRenderer, ServerPreferenceSnapshot

        snapshot = self.data.snapshot()
        artifact = ScreenRenderer().render(
            key.screen_id,
            PROFILE_PRESETS[key.render_profile],
            ServerPreferenceSnapshot(revision=0, values={}),
            snapshot,
        )
        metadata = {k: v for k, v in artifact.metadata.items() if k in {"animation", "required_capabilities"}}
        refresh = CLOCK_REFRESH_SECONDS if key.screen_id in CLOCK_SCREENS else DEFAULT_REFRESH_SECONDS
        return RenderOutput(image=artifact.image, refresh_seconds=refresh, metadata=metadata)

    def health(self) -> Mapping[str, Any]:
        snapshot = self.data.snapshot()
        return {
            "revision": snapshot.revision,
            "sources": dict(snapshot.source_revisions),
            "snapshot_at": snapshot.created_at.isoformat(timespec="seconds"),
        }


__all__ = ["CLOCK_SCREENS", "ServerRendering", "StyleRevision"]
