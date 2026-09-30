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
import json
import threading
from dataclasses import replace
from datetime import datetime, timezone
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from protocol_versions import APPLICATION_VERSION
from remote_display.locations import Location, scoped_values
from remote_display.models import RenderKey, ScreenRevisions
from remote_display.render_coordinator import RenderOutput

DEFAULT_REFRESH_SECONDS = 300
# Settings sections whose values change what screens show.
CONTENT_SECTIONS = frozenset({"location", "weather", "air_quality", "sports", "news", "adsb", "maps"})
# Screens that show the time must be refreshed every minute.
CLOCK_SCREENS = frozenset({"date", "nixie"})
CLOCK_REFRESH_SECONDS = 60


def _scroll_settings(path: Path) -> bytes:
    """The scroll settings in a rotation config, which every render reads.

    ``utils`` takes global and per-screen scroll speed from the active
    screens config (the config UI's Screens page) whatever the playlist, so
    they belong in the style revision; the rest of that file does not.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return b"<missing>"
    if not isinstance(payload, dict):
        return b"<invalid>"
    screens = payload.get("screens") if isinstance(payload.get("screens"), dict) else {}
    scroll = {
        "global": payload.get("scroll"),
        "screens": {sid: spec.get("scroll") for sid, spec in screens.items()
                    if isinstance(spec, dict) and spec.get("scroll") is not None},
    }
    return json.dumps(scroll, sort_keys=True, default=str).encode("utf-8")


class StyleRevision:
    """Content hash of the style and layout documents, cached on mtime.

    With the default paths it also covers the scroll settings in the active
    screens config (see :func:`_scroll_settings`).
    """

    def __init__(self, paths: Iterable[Path] | None = None, scroll_path: Path | None = None) -> None:
        self._paths = None if paths is None else tuple(Path(p) for p in paths)
        self._scroll_path = None if scroll_path is None else Path(scroll_path)
        self._lock = threading.Lock()
        self._signature: tuple | None = None
        self._revision = "s-none"

    def _resolve(self) -> tuple[Path, ...]:
        if self._paths is not None:
            return self._paths
        import paths

        return (paths.resolve_style_config_path(), paths.resolve_layouts_config_path())

    def _resolve_scroll(self) -> Path | None:
        if self._scroll_path is not None or self._paths is not None:
            return self._scroll_path
        import paths

        return paths.resolve_screens_config_paths().active_path

    def __call__(self) -> str:
        files = self._resolve()
        scroll = self._resolve_scroll()
        signature = tuple((str(p), *_stat(p)) for p in (*files, *((scroll,) if scroll else ())))
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
                if scroll is not None:
                    digest.update(b"scroll\0" + _scroll_settings(scroll))
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
        profile_processes: Any = None,
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
        # A rendering.profile_process.ProfileProcessPool composes each profile
        # in a process configured for it, as the v0.1 standalone display was.
        # Without one, profiles are composed in this process by substitution.
        self.profile_processes = profile_processes

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

    def scoped_revisions(self, pairs: Iterable[tuple[str, str]]) -> Mapping[tuple[str, str], ScreenRevisions]:
        """Revisions of screens rendered for a location scope (``loc-…``)."""

        snapshot = self.data.snapshot()
        style = f"{self.style_revision()}+{self.preferences}"
        renderer = f"v{APPLICATION_VERSION}"
        result = {}
        for screen, scope in pairs:
            data_revision = None
            if self.feeds is not None:
                data_revision = self.feeds.data_revision(screen, snapshot.source_revisions, scope=scope)
            result[(screen, scope)] = ScreenRevisions(
                style_revision=style,
                data_revision=data_revision or f"d{snapshot.revision}",
                renderer_revision=renderer,
            )
        return result

    def render(self, key: RenderKey) -> RenderOutput:
        from display_profiles import PROFILE_PRESETS
        from rendering import screen_classes
        from rendering.packaging import clock_package

        profile = PROFILE_PRESETS[key.render_profile]
        screen = screen_classes.CLASSIFICATIONS.get(key.screen_id)
        refresh = DEFAULT_REFRESH_SECONDS
        if screen is not None and screen.kind == screen_classes.CLIENT_TIMED:
            # Clients draw the time; the still is a fallback for clients that
            # cannot, drawn without the server's own IP or update state.
            from rendering.clock_faces import clock_background, clock_layout, render_clock

            layout = clock_layout(key.screen_id, profile)
            now = datetime.now(timezone.utc)
            if self.profile_processes is not None:
                image = self.profile_processes.render_clock(layout, profile, now)
            else:
                image = render_clock(layout, profile, now)
            package = clock_package(key, profile, layout, clock_background(layout, profile))
            return RenderOutput(image=image, refresh_seconds=CLOCK_REFRESH_SECONDS, package=package)
        if screen is not None and screen.kind == screen_classes.PERIODIC:
            refresh = screen_classes.PERIODIC_REFRESH_SECONDS

        snapshot = self.data.snapshot()
        timestamp = getattr(self.data, "weather_cache_timestamp", None)
        location = Location.from_scope(key.client_scope)
        if not callable(timestamp):
            fetched_at = None
        elif location is None:
            fetched_at = timestamp()
        else:
            fetched_at = timestamp((location.latitude, location.longitude))
        if self.profile_processes is not None:
            reply = self.profile_processes.render_screen(key, profile, snapshot, fetched_at)
            image, metadata, package = reply["image"], reply["metadata"], reply["package"]
        else:
            image, metadata, package = compose_screen(key, profile, snapshot, self.logos, fetched_at)
        return RenderOutput(image=image, refresh_seconds=refresh, metadata=metadata, package=package)

    def close(self) -> None:
        if self.profile_processes is not None:
            self.profile_processes.close()

    def health(self) -> Mapping[str, Any]:
        snapshot = self.data.snapshot()
        return {
            "revision": snapshot.revision,
            "sources": dict(snapshot.source_revisions),
            "snapshot_at": snapshot.created_at.isoformat(timespec="seconds"),
            "feeds": None if self.feeds is None else self.feeds.health(),
        }


def compose_screen(key: RenderKey, profile: Any, snapshot: Any, logos: Any,
                   weather_fetched_at: Any = None) -> tuple[Any, dict[str, Any], Any]:
    """Compose *key* in this process: ``(image, metadata, package)``."""

    from rendering import screen_classes
    from rendering.logos import IMAGES_DIR
    from rendering.packaging import build_package
    from rendering.screen_renderer import ScreenRenderer, ServerPreferenceSnapshot

    screen = screen_classes.CLASSIFICATIONS.get(key.screen_id)
    if Location.from_scope(key.client_scope) is not None:
        # A display with its own location: its place's weather under the
        # keys the screens read.
        snapshot = replace(snapshot, values=scoped_values(snapshot.values, key.client_scope))
    preferences = ServerPreferenceSnapshot(revision=0, values={
        "logos": logos.for_size(profile.width, profile.height),
        "image_dir": IMAGES_DIR,
        "weather_fetched_at": weather_fetched_at,
    })
    record = screen is not None and screen.kind == screen_classes.FINITE_ANIMATION
    artifact = ScreenRenderer().render(key.screen_id, profile, preferences, snapshot, record_frames=record)
    metadata = {k: v for k, v in artifact.metadata.items() if k in {"animation", "required_capabilities", "led"}}
    return artifact.image, metadata, build_package(key, profile, artifact)


__all__ = ["CLOCK_SCREENS", "ServerRendering", "StyleRevision", "compose_screen", "preferences_revision"]
