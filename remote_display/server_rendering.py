"""Production renderer and revision source for the render coordinator.

Renders with :class:`rendering.screen_renderer.ScreenRenderer` from the
shared :mod:`services.data_coordinator` snapshot, so rendering never calls an
upstream API.  Revisions:

* ``style_revision`` hashes the style and layout documents together with the
  content preferences (location, teams, feeds and other non-secret content
  settings), so editing any of them rerenders every screen.
* ``data_revision`` covers exactly the feeds a screen reads (from
  :mod:`services.feeds`), so a weather update rerenders only weather screens.
  A screen no catalogued feed serves falls back to the whole snapshot
  revision, which is conservative but never stale.
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
# Settings sections whose values change what screens show.
CONTENT_SECTIONS = frozenset({"location", "weather", "air_quality", "sports", "news", "adsb", "maps"})
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


def preferences_revision(env: Mapping[str, str] | None = None) -> str:
    """Hash of the content preferences a render depends on.

    Secret settings (API keys and tokens) are excluded: they change how data
    is fetched, not how a screen looks, and must never feed a digest that
    leaves the server.
    """

    import os

    import deployment_config

    source = os.environ if env is None else env
    digest = hashlib.sha256()
    for setting in sorted(deployment_config.SETTINGS, key=lambda s: s.name):
        if setting.section not in CONTENT_SECTIONS or setting.secret or setting.provider:
            continue
        if deployment_config.Role.SERVER not in setting.roles:
            continue
        value = source.get(setting.name)
        digest.update(f"{setting.name}={setting.default if value is None else value}\0".encode())
    return f"p-{digest.hexdigest()[:16]}"


class ServerRendering:
    """Revision source, renderer and data health for :class:`RenderCoordinator`."""

    def __init__(
        self,
        data_coordinator: Any = None,
        style_revision: StyleRevision | None = None,
        feeds: Any = None,
        *,
        preferences: str | None = None,
        logos: Any = None,
    ) -> None:
        if data_coordinator is None:
            from services.data_coordinator import coordinator as data_coordinator
        if logos is None:
            import os

            from rendering.logos import ProfileLogos

            logos = ProfileLogos(ahl_tricode=os.environ.get("AHL_TEAM_TRICODE", "CHI"))
        self.data = data_coordinator
        self.style_revision = style_revision or StyleRevision()
        self.feeds = feeds
        # Settings are read once at start-up, like the config module does.
        self.preferences = preferences or preferences_revision()
        self.logos = logos

    def revisions(self, screens: Iterable[str]) -> Mapping[str, ScreenRevisions]:
        snapshot = self.data.snapshot()
        style = f"{self.style_revision()}+{self.preferences}"
        renderer = f"v{APPLICATION_VERSION}"
        result = {}
        for screen in screens:
            data_revision = None
            if self.feeds is not None:
                data_revision = self.feeds.data_revision(screen, snapshot.source_revisions)
            result[screen] = ScreenRevisions(
                style_revision=style,
                data_revision=data_revision or f"d{snapshot.revision}",
                renderer_revision=renderer,
            )
        return result

    def render(self, key: RenderKey) -> RenderOutput:
        from display_profiles import PROFILE_PRESETS
        from rendering.screen_renderer import ScreenRenderer, ServerPreferenceSnapshot

        from rendering.logos import IMAGES_DIR

        snapshot = self.data.snapshot()
        profile = PROFILE_PRESETS[key.render_profile]
        timestamp = getattr(self.data, "weather_cache_timestamp", None)
        preferences = ServerPreferenceSnapshot(revision=0, values={
            "logos": self.logos.for_size(profile.width, profile.height),
            "image_dir": IMAGES_DIR,
            "weather_fetched_at": timestamp() if callable(timestamp) else None,
        })
        artifact = ScreenRenderer().render(key.screen_id, profile, preferences, snapshot)
        metadata = {k: v for k, v in artifact.metadata.items() if k in {"animation", "required_capabilities"}}
        refresh = CLOCK_REFRESH_SECONDS if key.screen_id in CLOCK_SCREENS else DEFAULT_REFRESH_SECONDS
        return RenderOutput(image=artifact.image, refresh_seconds=refresh, metadata=metadata)

    def health(self) -> Mapping[str, Any]:
        snapshot = self.data.snapshot()
        return {
            "revision": snapshot.revision,
            "sources": dict(snapshot.source_revisions),
            "snapshot_at": snapshot.created_at.isoformat(timespec="seconds"),
            "feeds": None if self.feeds is None else self.feeds.health(),
        }


__all__ = ["CLOCK_SCREENS", "ServerRendering", "StyleRevision", "preferences_revision"]
