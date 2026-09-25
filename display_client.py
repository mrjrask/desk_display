#!/usr/bin/env python3
"""Thin display client: play server-rendered artifacts from a local cache.

Start-up order matters.  The client loads its settings, initializes the
display, and starts playing the last-known-good playlist and manifest from
its cache *before* it contacts the server, so a warm client works without
one.  :class:`remote_display.client_sync.ClientSync` runs in a background
thread; the display loop only reads the content it last activated, so
network trouble never delays a frame or a button press.

With nothing cached the client shows a local diagnostic screen with its
identity, server host and sync state (never a credential) until the first
complete sync.  Physical rotation is applied only by the hardware presenter
at final presentation; artifacts stay at logical dimensions throughout.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from PIL import Image, ImageDraw, ImageFont

from display_profiles import RenderProfile, resolve_display_profile_by_id
from playback.client_player import ClientPlayer
from protocol import CLIENT_SUPPORTED_RENDER_PACKAGE_SCHEMA_VERSIONS
from protocol_versions import APPLICATION_VERSION, NETWORK_PROTOCOL_VERSION
from remote_display.client_cache import ClientCache, reconcile_playback
from remote_display.client_sync import (
    ActiveContent,
    ArtifactCache,
    ClientSync,
    PlaybackReport,
    RequestsTransport,
)
from remote_display.models import ClientCapabilities

LOGGER = logging.getLogger("desk_display.client")
_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = _PROJECT_ROOT / "cache" / "client"
DEFAULT_SCREEN_SECONDS = 10.0
POLL_SECONDS = 0.05


def _rotation(value: Any) -> int:
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return {1: 90, 2: 180, 3: 270}.get(number, number if number in (0, 90, 180, 270) else 0)


def capabilities_for(client_id: str, profile: RenderProfile, *, has_touch: bool = False,
                     buttons: tuple[str, ...] = ()) -> ClientCapabilities:
    return ClientCapabilities(
        protocol_version=NETWORK_PROTOCOL_VERSION,
        client_software_version=APPLICATION_VERSION,
        client_id=client_id,
        display_profile=profile.profile_id,
        logical_width=profile.width,
        logical_height=profile.height,
        image_formats=("PNG",),
        color_modes=(profile.color_mode,),
        render_package_versions=tuple(sorted(CLIENT_SUPPORTED_RENDER_PACKAGE_SCHEMA_VERSIONS)),
        has_touch=has_touch,
        buttons=buttons,
    )


def safe_server_label(url: str | None) -> str:
    """Host and port only: never a path, query or embedded credentials."""

    if not url:
        return "not configured"
    parts = urlsplit(url)
    host = parts.hostname or "unknown"
    return f"{host}:{parts.port}" if parts.port else host


def diagnostic_image(profile: RenderProfile, lines: list[str]) -> Image.Image:
    """A locally drawn status screen at the profile's logical size."""

    image = Image.new("RGB", (profile.width, profile.height), "black")
    draw = ImageDraw.Draw(image)
    size = max(10, min(profile.width, profile.height) // 14)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    y = size // 2
    for index, line in enumerate(lines):
        draw.text((size // 2, y), line, fill="white" if index else (255, 200, 0), font=font)
        y += int(size * 1.4)
        if y > profile.height - size:
            break
    return image.convert(profile.color_mode)


@dataclass
class Controls:
    """Button and touch requests, consumed by the display loop."""

    skip: bool = False
    back: bool = False


class DisplayClient:
    """Cached playback loop; the synchronizer feeds it in the background."""

    def __init__(
        self,
        profile: RenderProfile,
        presenter: Any,
        sync: ClientSync,
        cache: ClientCache,
        artifacts: ArtifactCache,
        *,
        server_url: str | None = None,
        physical_rotation: int = 0,
        screen_seconds: float = DEFAULT_SCREEN_SECONDS,
        offline_max_age_seconds: float = 0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.profile = profile
        self.presenter = presenter
        self.sync = sync
        self.cache = cache
        self.artifacts = artifacts
        self.server_url = server_url
        self.physical_rotation = physical_rotation
        self.screen_seconds = screen_seconds
        self.offline_max_age_seconds = offline_max_age_seconds
        self._monotonic = monotonic
        self.controls = Controls()
        self._content_revision: tuple[str | None, str | None] | None = None
        self._player: ClientPlayer | None = None
        self.playback = cache.load_playback()
        self.report = PlaybackReport(physical_rotation=physical_rotation)
        self._stop = threading.Event()
        sync.report = lambda: self.report

    # Playback

    def _rebuild(self, content: ActiveContent) -> None:
        from schedule import build_scheduler

        self._content_revision = content.revision
        self._player = None
        playlist = content.playlist
        if playlist is None or content.manifest is None:
            return
        document = playlist.to_dict()["document"]
        try:
            scheduler = build_scheduler(document)
        except ValueError as exc:
            LOGGER.error("Cached playlist cannot be scheduled: %s", exc)
            return
        player = ClientPlayer(scheduler, default_duration=self.screen_seconds)
        packages = {e["screen_id"]: e for e in content.manifest.get("artifacts") or () if e.get("sha256")}
        try:
            player.load_cache(dict(content.manifest), {}, packages)
        except ValueError as exc:
            LOGGER.error("Cached manifest is not playable: %s", exc)
            return
        self.playback = reconcile_playback(self.playback, playlist)
        player.history = list(self.playback.history)
        self._player = player

    def _too_old(self) -> bool:
        """Offline and the cache is older than DESK_DISPLAY_OFFLINE_MAX_AGE_HOURS."""

        if self.offline_max_age_seconds <= 0 or self.sync.connected:
            return False
        age = self.sync.cache_age()
        return age is not None and age > self.offline_max_age_seconds

    def _diagnostic(self, state: str) -> Image.Image:
        age = self.sync.last_sync_age()
        active = self.sync.active()
        lines = [
            "Desk Display client",
            f"ID: {self.sync.client_id}",
            f"Server: {safe_server_label(self.server_url)}",
            f"State: {state}",
            "Last sync: never" if age is None else f"Last sync: {int(age)}s ago",
            f"Playlist: {active.playlist.playlist_id if active.playlist else 'none cached'}",
            f"Version: {APPLICATION_VERSION}",
        ]
        return diagnostic_image(self.profile, lines)

    def step(self) -> tuple[str | None, float]:
        """Present one frame; return ``(screen_id, seconds to show it)``."""

        content = self.sync.active()
        if content.revision != self._content_revision:
            self._rebuild(content)
        player = self._player
        back = self.controls.back
        self.controls.back = self.controls.skip = False
        item = None
        if player is not None:
            if back:
                item = player.previous()
            if item is None:
                item = player.next()
        frame = None
        too_old = self._too_old()
        if item is not None and item.package is not None and not too_old:
            data = self.artifacts.read(item.package)
            if data is not None:
                import io

                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    frame = image.copy()
        if frame is None:
            state = "waiting for first sync" if content.playlist is None else "cached content unavailable"
            if not self.sync.connected and content.playlist is None:
                state = "server unreachable"
            if too_old:
                state = "offline; cached content expired"
            self.report.playback_state = "error" if content.playlist else "starting"
            self.report.current_screen = None
            self.presenter.present(self._diagnostic(state))
            return None, 5.0
        self.report.playback_state = "playing" if self.sync.connected else "offline"
        self.report.current_screen = item.screen_id
        self.presenter.present(frame)
        self.playback.current_screen = item.screen_id
        self.playback.remember(item.screen_id)
        if content.playlist is not None:
            self.playback.playlist_id = content.playlist.playlist_id
            self.playback.playlist_revision = content.playlist.playlist_revision
        try:
            self.cache.save_playback(self.playback)
        except OSError as exc:
            LOGGER.warning("Could not save playback state: %s", exc)
        return item.screen_id, item.duration

    def wait(self, seconds: float) -> None:
        """Hold the current frame, returning early on a control or stop."""

        deadline = self._monotonic() + seconds
        while not self._stop.is_set() and self._monotonic() < deadline:
            if self.controls.skip or self.controls.back:
                return
            if self.sync.active().revision != self._content_revision and self._player is None:
                return
            self._stop.wait(POLL_SECONDS)

    def on_button(self, name: str) -> None:
        if name in {"B", "Y", "right", "next"}:
            self.controls.skip = True
        elif name in {"A", "X", "left", "previous"}:
            self.controls.back = True

    def run(self) -> None:
        with_buttons = getattr(self.presenter, "set_button_callback", None)
        if callable(with_buttons):
            try:
                with_buttons(self.on_button)
            except Exception:
                LOGGER.debug("No button support", exc_info=True)
        self.sync.start()
        while not self._stop.is_set():
            _screen, seconds = self.step()
            self.wait(seconds)

    def stop(self) -> None:
        self._stop.set()
        self.sync.stop()


def _load_env_files() -> None:
    if os.environ.get("CONFIG_LOAD_DOTENV", "1").strip().lower() in {"0", "false", "no", "off"}:
        return
    import deployment_config

    for name in (".env.client", ".env"):
        path = _PROJECT_ROOT / name
        if path.is_file():
            for key, value in deployment_config.parse_env_file(path).items():
                os.environ.setdefault(key, value)
            return


def build_client(settings: dict[str, Any], *, presenter: Any = None, transport: Any = None) -> DisplayClient:
    profile = resolve_display_profile_by_id(settings["DESK_DISPLAY_PROFILE"])
    if profile is None:
        raise SystemExit(f"Unknown display profile {settings['DESK_DISPLAY_PROFILE']!r}")
    cache_dir = Path(settings.get("DESK_DISPLAY_CLIENT_CACHE_DIR") or DEFAULT_CACHE_DIR).expanduser()
    cache = ClientCache(cache_dir)
    artifacts = ArtifactCache(cache_dir, max_bytes=int(settings.get("DESK_DISPLAY_CLIENT_CACHE_MAX_MB") or 256) << 20)
    if presenter is None:
        from display.hardware_presenter import HardwarePresenter

        presenter = HardwarePresenter(profile=profile)
    server_url = settings.get("DESK_DISPLAY_SERVER_URL")
    if transport is None:
        verify: bool | str = bool(settings.get("DESK_DISPLAY_TLS_VERIFY", True))
        if verify and settings.get("DESK_DISPLAY_SERVER_CA_BUNDLE"):
            verify = str(settings["DESK_DISPLAY_SERVER_CA_BUNDLE"])
        transport = RequestsTransport(server_url, verify=verify)
    sync = ClientSync(
        capabilities_for(settings["DESK_DISPLAY_CLIENT_ID"], profile),
        transport,
        cache,
        artifacts,
        enrollment_token=settings.get("DESK_DISPLAY_CLIENT_TOKEN"),
        sync_interval_seconds=int(settings.get("DESK_DISPLAY_SYNC_INTERVAL_SECONDS") or 30),
    )
    return DisplayClient(
        profile, presenter, sync, cache, artifacts,
        server_url=server_url,
        physical_rotation=_rotation(settings.get("DISPLAY_ROTATION")),
        offline_max_age_seconds=float(settings.get("DESK_DISPLAY_OFFLINE_MAX_AGE_HOURS") or 0) * 3600,
    )


def main() -> None:  # pragma: no cover - exercised on hardware
    import deployment_config

    _load_env_files()
    os.environ.setdefault(deployment_config.ROLE_ENV, "client")
    logging.basicConfig(level=deployment_config.resolve_log_level())
    deployment_config.install_secret_log_redaction()
    deployment_config.startup_check("display client")
    settings = deployment_config.load_settings(deployment_config.Role.CLIENT)
    client = build_client(settings)
    try:
        client.run()
    except KeyboardInterrupt:
        pass
    finally:
        client.stop()
        close = getattr(client.presenter, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":  # pragma: no cover
    main()
