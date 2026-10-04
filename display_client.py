#!/usr/bin/env python3
"""Thin display client: play server-rendered artifacts from a local cache.

Start-up order matters.  The client loads its settings, initializes the
display, and starts playing the last-known-good playlist and manifest from
its cache *before* it contacts the server, so a warm client works without
one (unless ``DESK_DISPLAY_OFFLINE_START=0``, which waits for a first sync).  :class:`remote_display.client_sync.ClientSync` runs in a background
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
import signal
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

# Before any project import: config.py loads a dotenv file on import, and a
# client must read .env.client, never the server's .env beside it. Only for
# these imports, so a process that also imports other modules is unaffected;
# the client process itself keeps it (prepare_environment).
_DOTENV_FILE_SET = "DESK_DISPLAY_DOTENV_FILE" not in os.environ
os.environ.setdefault("DESK_DISPLAY_DOTENV_FILE", ".env.client")

from PIL import Image, ImageDraw, ImageFont

from display.rotation import RotationDecision, parse_rotation, resolve_rotation, to_logical
from display_profiles import RenderProfile, resolve_display_profile_by_id
from playback.client_player import ClientPlayer
from playback.local_screens import default_local_screens, is_local, local_entries
from playback.motion_clock import MotionClock
from playback.package_player import PackagePlayback
from protocol import CLIENT_SUPPORTED_RENDER_PACKAGE_SCHEMA_VERSIONS
from protocol_versions import APPLICATION_VERSION, NETWORK_PROTOCOL_VERSION
from remote_display.client_cache import ClientCache, PlaybackState, reconcile_playback
from remote_display.client_commands import CommandRunner
from remote_display.client_screenshots import ClientScreenshots
from remote_display.screenshot_uploads import UploadQueue
from remote_display.client_sync import (
    ActiveContent,
    ArtifactCache,
    ClientSync,
    PlaybackReport,
    RequestsTransport,
)
from remote_display.models import ClientCapabilities, HardwareDescription

if _DOTENV_FILE_SET:
    os.environ.pop("DESK_DISPLAY_DOTENV_FILE", None)

LOGGER = logging.getLogger("desk_display.client")
_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = _PROJECT_ROOT / "cache" / "client"
# v0.1's hold for a screen before its extra seconds (config.SCREEN_DELAY).
DEFAULT_SCREEN_SECONDS = 4.0
POLL_SECONDS = 0.05


def _rotation(value: Any) -> int:
    try:
        return parse_rotation(value)
    except ValueError:
        LOGGER.warning("Invalid DISPLAY_ROTATION %r; using 0", value)
        return 0


def capabilities_for(client_id: str, profile: RenderProfile, *, has_touch: bool = False,
                     buttons: tuple[str, ...] = (),
                     rotation: RotationDecision | None = None,
                     supports_animation: bool = True) -> ClientCapabilities:
    """Capabilities in canonical logical orientation; rotation is diagnostic only."""

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
        supports_animation=supports_animation,
        has_touch=has_touch,
        buttons=buttons,
        hardware=None if rotation is None else HardwareDescription(driver=rotation.describe()),
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


class DarkHours:
    """This client's own dark hours and backlight levels.

    ``DARK_HOURS`` is read as wall-clock time in the content timezone.  During
    dark hours the panel is either blanked with its backlight off (``off``)
    or keeps playing at a lower backlight (``dim``).
    """

    def __init__(
        self,
        spec: str | None = None,
        *,
        mode: str = "off",
        level: int = 100,
        dark_level: int = 10,
        zone: Any = None,
        now: Callable[[], Any] | None = None,
    ) -> None:
        import datetime as dt
        from zoneinfo import ZoneInfo

        from dark_hours import _parse_dark_hours_spec
        from display_time import DEFAULT_CONTENT_TIMEZONE

        self.segments = _parse_dark_hours_spec(spec)
        self.mode = mode if mode in {"off", "dim"} else "off"
        self.level = max(0, min(100, int(level)))
        self.dark_level = max(0, min(100, int(dark_level)))
        self.zone = zone or ZoneInfo(DEFAULT_CONTENT_TIMEZONE)
        self._now = now or (lambda: dt.datetime.now(dt.timezone.utc))

    @classmethod
    def from_settings(cls, settings: dict[str, Any], **kwargs: Any) -> DarkHours:
        from zoneinfo import ZoneInfo

        from display_time import content_timezone_name

        def number(name: str, default: int) -> int:
            value = settings.get(name)
            return default if value in (None, "") else int(value)

        return cls(
            settings.get("DARK_HOURS"),
            mode=str(settings.get("DESK_DISPLAY_DARK_HOURS_MODE") or "off").strip().lower(),
            level=number("DESK_DISPLAY_BACKLIGHT_LEVEL", 100),
            dark_level=number("DESK_DISPLAY_DARK_HOURS_BACKLIGHT_LEVEL", 10),
            zone=ZoneInfo(content_timezone_name({k: str(v) for k, v in settings.items() if v is not None})),
            **kwargs,
        )

    def active(self) -> bool:
        from dark_hours import segments_contain

        return bool(self.segments) and segments_contain(self.segments, self._now(), self.zone)

    def state(self) -> str:
        """``normal``, ``dim`` or ``dark`` for the current moment."""

        if not self.active():
            return "normal"
        return "dim" if self.mode == "dim" else "dark"

    def backlight(self, state: str) -> float:
        """The backlight fraction (0 to 1) for *state*."""

        if state == "dark":
            return 0.0
        return (self.dark_level if state == "dim" else self.level) / 100


def _live_screens() -> frozenset[str]:
    from rendering.screen_classes import CLASSIFICATIONS, PERIODIC

    return frozenset(sid for sid, entry in CLASSIFICATIONS.items()
                     if entry.kind == PERIODIC or "Scoreboard" in sid)


# Screens whose content goes out of date within minutes (see DisplayClient._playable).
LIVE_SCREENS = _live_screens()


def _vertical_speed_adjustment(manifest: Any) -> float | None:
    """This display's own vertical scroll adjustment from its manifest, if set."""

    configuration = manifest.get("configuration") if isinstance(manifest, Mapping) else None
    value = configuration.get("vertical_speed_adjustment") if isinstance(configuration, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return min(3.0, max(-0.9, float(value)))


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


DARK_POLL_SECONDS = 30.0
UPDATE_CHECK_SECONDS = 60.0
LIGHT_CHECK_SECONDS = 1.0


@dataclass
class Controls:
    """Button and touch requests, consumed by the display loop."""

    skip: bool = False
    back: bool = False
    focus: str | None = None  # a quad tile's screen to open full screen
    unfocus: bool = False  # leave a focused tile and return to its quad
    # v0.1's B, X and Y buttons, applied on the display loop's thread.
    toggle_display: bool = False
    toggle_indicator: bool = False
    restart: bool = False


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
        clock: Callable[[], Any] | None = None,
        ip_text: Callable[[], str | None] | None = None,
        dark_hours: DarkHours | None = None,
        offline_start: bool = True,
        screenshots: ClientScreenshots | None = None,
        update_check: Callable[[], Any] | None = None,
        restart_service: Callable[[], Any] | None = None,
        update_available: Callable[[], bool] | None = None,
        local_screens: dict[str, Any] | None = None,
        max_fps: float = 0.0,
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
        # This display's own vertical scroll adjustment (Clients page), or
        # None to play scrolls as the server paced them.
        self._vertical_speed_adjustment: float | None = None
        self._player: ClientPlayer | None = None
        # A restart always begins at the top of the Starter playlist: only the
        # history (for the Back button) survives from the saved position.
        self.playback = PlaybackState(history=cache.load_playback().history)
        self._start_at_starter = True
        self.report = PlaybackReport(physical_rotation=physical_rotation)
        self._stop = threading.Event()
        self.restart_requested = False
        sync.report = lambda: self.report
        caps = sync.capabilities
        self.has_touch = bool(caps.has_touch)
        self.supports_animation = bool(caps.supports_animation)
        self.animation: PackagePlayback | None = None
        self._shown_at = 0.0
        self._current: str | None = None
        self._focus_return: str | None = None  # quad to return to after a focused tile
        self._returning = False
        self._clock = clock
        self._ip_text = ip_text
        self.dark_hours = dark_hours or DarkHours()
        self.light_state: str | None = None  # last applied: normal, dim or dark
        # DESK_DISPLAY_OFFLINE_START=0: play nothing cached until a sync in this
        # process has activated the server's current content, so a restarted
        # client never shows stale content.
        self.offline_start = offline_start
        # Feeds the config UI's Screenshots/Feed pages and display heartbeat.
        self.screenshots = screenshots
        self._led: tuple[float, float, float] | None = None
        self._update_check = update_check
        # The clock faces' GitHub update icon: this device's own update status.
        self._update_available = update_available
        self._update_check_at: float | None = None
        self._update_check_thread: threading.Thread | None = None
        # v0.1's B button: the panel is blanked until B is pressed again.
        self.display_off = False
        self._restart_service = restart_service
        # Screens this device draws from its own hardware (the inside sensor).
        self.local_screens = dict(local_screens or {})
        # DESK_DISPLAY_CLIENT_MAX_FPS: the fewest seconds between animation
        # frames (0 = as fast as each package asks). Motion keeps its speed
        # and moves further per frame instead.
        self.min_frame_seconds = 1.0 / max_fps if max_fps and max_fps > 0 else 0.0

    # Playback

    def _rebuild(self, content: ActiveContent) -> None:
        from schedule import build_scheduler, starter_screen_ids

        self._content_revision = content.revision
        self._player = None
        self._vertical_speed_adjustment = _vertical_speed_adjustment(content.manifest)
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
        packages.update(local_entries(scheduler.requested_ids, self.local_screens))
        # Checked live, not baked into `packages`: a failed download can be
        # repaired by a later sync without the manifest revision changing.
        # Only the still is required: step() falls back to it when a render
        # package is missing or bad.
        player.is_locally_usable = self._playable
        try:
            player.load_cache(dict(content.manifest), {}, packages)
        except ValueError as exc:
            LOGGER.error("Cached manifest is not playable: %s", exc)
            return
        previous_screen = self.playback.current_screen
        self.playback = reconcile_playback(self.playback, playlist)
        # A playlist update is configuration, not a command: resume this
        # client's own position, or continue after the screen it was showing.
        if self._start_at_starter:
            self._start_at_starter = False
            scheduler.start_at(starter_screen_ids(document))
        else:
            resumed = bool(self.playback.scheduler) and scheduler.restore_state(self.playback.scheduler)
            if not resumed and previous_screen in playlist.screens:
                scheduler.seek_after(previous_screen)
        player.history = list(self.playback.history)
        self._player = player

    def _playable(self, entry: Any) -> bool:
        """A cached screen that can play now.

        While the server is unreachable, a live screen (scoreboards and live
        games) whose refresh deadline has passed is skipped, so a frozen score
        never looks current. v0.1 likewise skipped scoreboards during an
        outage with games live.
        """

        if is_local(entry):
            drawer = self.local_screens.get(entry["screen_id"])
            return drawer is not None and drawer.available
        if not self.artifacts.has(entry):
            return False
        if self.sync.connected or not isinstance(entry, dict) or entry.get("screen_id") not in LIVE_SCREENS:
            return True
        deadline = _parse_time(entry.get("refresh_deadline"))
        if deadline is None:
            return True
        now = self._clock() if self._clock is not None else datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now <= deadline

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

    def _next_item(self, player: ClientPlayer | None) -> Any:
        """The next screen: a tapped tile, the quad it returns to, or the rotation."""

        controls = self.controls
        focus, unfocus, back = controls.focus, controls.unfocus, controls.back
        controls.back = controls.skip = controls.unfocus = False
        controls.focus = None
        self._returning = False
        if player is None:
            return None
        if focus is not None and self._current is not None:
            item = player.item_for(focus)
            if item is not None:
                # Opening a tile never moves the rotation: the quad resumes after.
                self._focus_return = self._current
                self.playback.focus_return = self._current
                return item
        if self._focus_return is not None:
            returning, self._focus_return = self._focus_return, None
            self.playback.focus_return = None
            if unfocus or not back:
                item = player.item_for(returning)
                if item is not None:
                    self._returning = True
                    return item
        item = player.previous() if back else None
        return item if item is not None else player.next()

    def _fallback(self, screen: str) -> Any:
        from remote_display.fallbacks import playback_mode

        return playback_mode(screen, supports_animation=self.supports_animation, has_touch=self.has_touch,
                             color_mode=self.profile.color_mode)

    def _animation_for(self, item: Any) -> PackagePlayback | None:
        """Local playback of the screen's render package, when it applies."""

        if item.package is None or not self._fallback(item.screen_id).animated:
            return None
        package = self.artifacts.read_package(item.package)
        if package is None:
            return None
        kwargs: dict[str, Any] = {"hold_seconds": item.duration}
        if self._clock is not None:
            kwargs["clock"] = self._clock
        if self._ip_text is not None:
            kwargs["ip_text"] = self._ip_text
        if self._update_available is not None:
            kwargs["update_available"] = self._update_available
        if self._vertical_speed_adjustment is not None:
            kwargs["vertical_speed_adjustment"] = self._vertical_speed_adjustment
        try:
            return PackagePlayback(package, self.profile, **kwargs)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            LOGGER.warning("Render package for %s is not playable; showing its still: %s", item.screen_id, exc)
            return None

    def _apply_light(self, state: str) -> None:
        """Set the backlight for *state* once, when it changes."""

        if state == self.light_state:
            return
        previous, self.light_state = self.light_state, state
        if state == "dark":
            LOGGER.info("Entering dark hours; blanking the display")
        elif previous == "dark":
            LOGGER.info("Leaving dark hours; resuming playback")
        set_backlight = getattr(self.presenter, "set_backlight", None)
        if callable(set_backlight):
            try:
                set_backlight(self.dark_hours.backlight(state))
            except Exception:  # noqa: BLE001 - a panel without backlight control still plays
                LOGGER.debug("Backlight control failed", exc_info=True)

    def _set_led(self, color: Any) -> None:
        """Show *color* (a screen's notification) on the LED and border, once.

        ``None`` returns them to the update status, as v0.1's
        ``temporary_display_led`` did when a screen ended.
        """

        led = None
        if isinstance(color, (list, tuple)) and len(color) == 3:
            try:
                led = tuple(max(0.0, min(1.0, float(c))) for c in color)
            except (TypeError, ValueError):
                led = None
            if led is not None and not any(led):
                led = None
        if led == self._led:
            return
        self._led = led
        set_led = getattr(self.presenter, "set_led", None)
        if callable(set_led):
            try:
                set_led(led)
            except Exception:  # noqa: BLE001 - an LED fault must never stop playback
                LOGGER.debug("LED update failed", exc_info=True)

    def _maybe_check_updates(self, screen_id: str) -> None:
        """Run v0.1's GitHub/apt update check when a clock face shows.

        The standalone ``date`` and ``nixie`` screens started this check each
        time they appeared; its result drives the update LED and border.  It
        runs in the background and at most once per UPDATE_CHECK_SECONDS.
        """

        from remote_display.server_rendering import CLOCK_SCREENS

        if self._update_check is None or screen_id not in CLOCK_SCREENS:
            return
        now = self._monotonic()
        if self._update_check_at is not None and now - self._update_check_at < UPDATE_CHECK_SECONDS:
            return
        thread = self._update_check_thread
        if thread is not None and thread.is_alive():
            return
        self._update_check_at = now

        def run() -> None:
            try:
                self._update_check()
            except Exception:  # noqa: BLE001 - a failed check leaves the indicator as it was
                LOGGER.debug("Update check failed", exc_info=True)

        self._update_check_thread = threading.Thread(target=run, name="update-check", daemon=True)
        self._update_check_thread.start()

    def _apply_buttons(self) -> None:
        """Run v0.1's B (display on/off), X (update indicator) and Y (restart) actions."""

        controls = self.controls
        toggle_display, toggle_indicator, restart = (
            controls.toggle_display, controls.toggle_indicator, controls.restart)
        controls.toggle_display = controls.toggle_indicator = controls.restart = False
        if toggle_indicator:
            toggle = getattr(self.presenter, "toggle_update_indicator", None)
            if callable(toggle):
                try:
                    enabled = toggle()
                    LOGGER.info("X button: update indicator %s", "enabled" if enabled else "disabled")
                except Exception:  # noqa: BLE001 - an LED fault must never stop playback
                    LOGGER.debug("Update indicator toggle failed", exc_info=True)
        if toggle_display:
            self.display_off = not self.display_off
            LOGGER.info("B button: display toggled %s", "off" if self.display_off else "on")
            if not self.display_off:
                self.light_state = None  # restore the backlight for the current light state
        if restart and self._restart_service is not None:
            LOGGER.info("Y button: restarting the display client service")
            try:
                self._restart_service()
            except Exception:  # noqa: BLE001
                LOGGER.warning("Could not restart the display client service", exc_info=True)

    def _go_off(self) -> tuple[None, float]:
        """The B button's blank panel, held until B is pressed again."""

        entering = self.light_state != "off"
        if entering:
            self.light_state = "off"
            set_backlight = getattr(self.presenter, "set_backlight", None)
            if callable(set_backlight):
                try:
                    set_backlight(0.0)
                except Exception:  # noqa: BLE001
                    LOGGER.debug("Backlight control failed", exc_info=True)
        self._blank("paused", entering)  # the protocol's state for a panel switched off by hand
        return None, DARK_POLL_SECONDS

    def _blank(self, state: str, entering: bool) -> None:
        self._set_led(None)
        self.animation = None
        self._current = None
        self.report.playback_state = state
        self.report.current_screen = None
        if entering:
            self.presenter.present(Image.new(self.profile.color_mode, (self.profile.width, self.profile.height)))

    def _go_dark(self) -> tuple[None, float]:
        entering = self.light_state != "dark"
        self._apply_light("dark")
        self.controls = Controls()  # skip and back do nothing while the panel is dark
        self._blank("dark", entering)
        return None, DARK_POLL_SECONDS

    def step(self) -> tuple[str | None, float]:
        """Present a screen's first frame; return ``(screen_id, seconds to show it)``."""

        self._apply_buttons()
        if self.display_off:
            return self._go_off()
        light = self.dark_hours.state()
        if light == "dark":
            return self._go_dark()
        self._apply_light(light)
        if not self.offline_start and not self.sync.confirmed:
            self.animation = None
            self._current = None
            self.report.playback_state = "starting"
            self.report.current_screen = None
            state = "waiting for first sync" if self.sync.connected else "server unreachable"
            self._set_led(None)
            self.presenter.present(self._diagnostic(state))
            return None, 5.0
        content = self.sync.active()
        if content.revision != self._content_revision:
            self._rebuild(content)
        player = self._player
        item = self._next_item(player)
        frame = None
        self.animation = None
        too_old = self._too_old()
        if item is not None and is_local(item.package):
            # Drawn here from this device's own sensor: never stale.
            frame = self.local_screens[item.screen_id].render(
                self.profile.width, self.profile.height, self.profile.color_mode)
        elif item is not None and item.package is not None and not too_old:
            self.animation = self._animation_for(item)
            if self.animation is not None:
                try:
                    frame = self.animation.frame_at(0.0)
                except (KeyError, TypeError, ValueError, OSError) as exc:
                    LOGGER.warning("Render package for %s failed; showing its still: %s", item.screen_id, exc)
                    self.animation = None
            if frame is None:
                data = self.artifacts.read(item.package)
                if data is not None:
                    import io

                    with Image.open(io.BytesIO(data)) as image:
                        image.load()
                        frame = image.copy()
        if frame is None:
            self.animation = None
            self._current = None
            state = "waiting for first sync" if content.playlist is None else "cached content unavailable"
            if not self.sync.connected and content.playlist is None:
                state = "server unreachable"
            if too_old:
                state = "offline; cached content expired"
            self.report.playback_state = "error" if content.playlist else "starting"
            self.report.current_screen = None
            self._set_led(None)
            self.presenter.present(self._diagnostic(state))
            return None, 5.0
        self.report.playback_state = "playing" if self.sync.connected else "offline"
        self.report.current_screen = item.screen_id
        # Before presenting, as v0.1 lit it before the frame went out.
        self._set_led(item.package.get("led") if isinstance(item.package, dict) else None)
        self.presenter.present(frame)
        self._maybe_check_updates(item.screen_id)
        if self.screenshots is not None:
            screenshot = self.animation.screenshot_image() if self.animation is not None else frame
            border = getattr(self.presenter, "apply_indicator_border", None)
            if callable(border):
                # Saved screenshots show the notification border, as in v0.1.
                screenshot = border(screenshot)
            self.screenshots.record(item.screen_id, screenshot)
        self._shown_at = self._monotonic()
        self._current = item.screen_id
        self.playback.current_screen = item.screen_id
        if self._focus_return is None and not self._returning:
            self.playback.remember(item.screen_id)
        self.playback.scheduler = player.scheduler.export_state()
        if content.playlist is not None:
            self.playback.playlist_id = content.playlist.playlist_id
            self.playback.playlist_revision = content.playlist.playlist_revision
        try:
            self.cache.save_playback(self.playback)
        except OSError as exc:
            LOGGER.warning("Could not save playback state: %s", exc)
        seconds = self.animation.duration if self.animation is not None else item.duration
        return item.screen_id, seconds

    def _controls_pending(self) -> bool:
        c = self.controls
        return (c.skip or c.back or c.focus is not None or c.unfocus
                or c.toggle_display or c.toggle_indicator or c.restart)

    def _poll_taps(self) -> None:
        poll = getattr(self.presenter, "poll_taps", None)
        if not self.has_touch or not callable(poll):
            return
        try:
            taps = poll() or ()
        except Exception:  # noqa: BLE001 - input trouble must never stop playback
            LOGGER.debug("Touch polling failed", exc_info=True)
            return
        for x, y in taps:
            self.on_touch(x, y)

    def wait(self, seconds: float) -> None:
        """Hold or animate the current screen, returning early on a control or stop.

        Motion is drawn from the cached package on this device; nothing here
        waits on the network.
        """

        started = self._monotonic()
        deadline = started + seconds
        next_light_check = started + LIGHT_CHECK_SECONDS
        # Motion runs on a MotionClock: a panel slower than the package's
        # frame rate moves one step per frame, as v0.1 did, and the screen's
        # time grows by however far it fell behind.
        animation = self.animation
        motion = MotionClock(started - self._shown_at, started,
                             max_lag=animation.motion_seconds if animation is not None else 0.0)
        presented_key = animation.key_at(motion.t) if animation is not None else None
        next_frame_at = started  # the earliest the next animation frame may go out
        while not self._stop.is_set() and self._monotonic() < deadline + motion.lag:
            self._poll_taps()
            if self._controls_pending():
                return
            if self._monotonic() >= next_light_check:
                # A dark-hours boundary ends the hold, so the panel blanks,
                # dims or wakes promptly even during a long animation.
                next_light_check = self._monotonic() + LIGHT_CHECK_SECONDS
                if self.dark_hours.state() != self.light_state:
                    return
            if self.sync.active().revision != self._content_revision and self._player is None:
                return
            interval = POLL_SECONDS
            animation = self.animation
            now = self._monotonic()
            if animation is not None and now < next_frame_at:
                # DESK_DISPLAY_CLIENT_MAX_FPS: not time for another frame yet.
                interval = min(POLL_SECONDS, next_frame_at - now)
            elif animation is not None:
                frame_seconds = max(animation.frame_seconds, self.min_frame_seconds)
                t = motion.tick(now, frame_seconds)
                key = animation.key_at(t)
                drawn = key != presented_key
                motion.drew(drawn)
                interval = min(POLL_SECONDS, frame_seconds)
                if drawn:
                    try:
                        self.presenter.present(animation.frame_at(t))
                    except (KeyError, TypeError, ValueError, OSError) as exc:
                        LOGGER.warning("Animation stopped; holding the last frame: %s", exc)
                        self.animation = None
                    presented_key = key
                    next_frame_at = now + self.min_frame_seconds
                    # The next frame is due one frame after this one started,
                    # not one frame after the (slow) push finished.
                    interval = min(POLL_SECONDS, max(0.0, now + frame_seconds - self._monotonic()))
            self._stop.wait(interval)

    def on_touch(self, x: float, y: float) -> tuple[int, int]:
        """Handle a tap in panel coordinates, entirely on this device.

        The point is mapped into logical coordinates first, so the gesture is
        the same however the panel is mounted. On an interactive quad a tap
        opens the tile's screen full screen from the cache; on a focused tile
        a tap returns to the quad. Elsewhere the left half goes back and the
        right half skips.
        """

        mapper = getattr(self.presenter, "touch_to_logical", None)
        if callable(mapper):
            lx, ly = mapper(x, y)
        else:
            lx, ly = to_logical(x, y, self.profile.width, self.profile.height, self.physical_rotation)
        if self._focus_return is not None:
            self.controls.unfocus = True
            return lx, ly
        target = self._tile_at(lx, ly)
        if target is not None:
            self.controls.focus = target
        elif lx < self.profile.width // 2:
            self.controls.back = True
        else:
            self.controls.skip = True
        return lx, ly

    def _tile_at(self, x: int, y: int) -> str | None:
        from display.rotation import hit_test

        animation, player = self.animation, self._player
        if animation is None or player is None or self._current is None:
            return None
        if not self._fallback(self._current).expands:
            return None
        target = hit_test(animation.focus_targets(), x, y)
        return target if target is not None and player.item_for(target) is not None else None

    def on_button(self, name: str) -> None:
        """v0.1's buttons: A next screen, B display on/off, X update indicator, Y restart."""

        if name in {"A", "right", "next"}:
            self.controls.skip = True
        elif name in {"left", "previous"}:
            self.controls.back = True
        elif name == "B":
            self.controls.toggle_display = True
        elif name == "X":
            self.controls.toggle_indicator = True
        elif name == "Y":
            self.controls.restart = True

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

    def request_restart(self) -> None:
        """Stop so the service manager starts the client again (Restart=always)."""

        LOGGER.info("Restarting the client as requested from the Display Clients page")
        self.restart_requested = True
        self.stop()


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


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _check_for_updates() -> None:
    """v0.1's update check: sets the update status the LED and border show."""

    from utils import check_apt_updates, check_github_updates

    check_github_updates()
    check_apt_updates()


def _github_update_available() -> bool:
    """Whether the last update check found new commits (v0.1's clock-face icon)."""

    from utils import get_update_status

    return bool(get_update_status().github)


def _restart_client_service() -> None:
    """v0.1's Y button: restart this device's display service (the client's here)."""

    import subprocess

    from service_units import CLIENT_SERVICE

    result = subprocess.run(["sudo", "systemctl", "--no-block", "restart", CLIENT_SERVICE], check=False)
    if result.returncode != 0:
        LOGGER.error("%s restart returned exit code %s", CLIENT_SERVICE, result.returncode)


def start_wifi_monitor(settings: dict[str, Any], wifi: Any = None) -> bool:
    """Start v0.1's Wi-Fi monitor and recovery; return whether it started.

    A client depends on the network to stay current, so it watches Wi-Fi the
    way main.py does. The monitor skips wired-only hosts itself.
    """

    # Unset, both follow v0.1's default: on unless DESK_DISPLAY_LOW_POWER.
    low_power = _truthy(settings.get("DESK_DISPLAY_LOW_POWER", False))

    def enabled(name: str) -> bool:
        if low_power and not (os.environ.get(name) or "").strip():
            return False
        return _truthy(settings.get(name, True))

    if not enabled("ENABLE_WIFI_MONITOR"):
        return False
    if wifi is None:
        from services import wifi_utils as wifi
    try:
        if not wifi.should_monitor_wifi():
            LOGGER.info("Wi-Fi monitor skipped for the current network setup")
            return False
        wifi.start_monitor(allow_recovery=enabled("ENABLE_WIFI_RECOVERY"))
    except Exception as exc:  # noqa: BLE001 - playback never depends on the monitor
        LOGGER.warning("Wi-Fi monitor unavailable: %s", exc)
        return False
    LOGGER.info("Wi-Fi monitor started")
    return True


def client_max_fps(settings: dict[str, Any]) -> float:
    """DESK_DISPLAY_CLIENT_MAX_FPS; 0 (the default) leaves animations uncapped."""

    value = settings.get("DESK_DISPLAY_CLIENT_MAX_FPS")
    if value is None or (isinstance(value, str) and not value.strip()):
        return 0.0
    try:
        fps = float(value)
    except (TypeError, ValueError):
        LOGGER.warning("Invalid DESK_DISPLAY_CLIENT_MAX_FPS %r; animations run uncapped", value)
        return 0.0
    return max(0.0, fps)


def _client_ip_text() -> str:
    """Return the address label drawn by client-rendered clock screens."""

    from services.wifi_utils import get_assigned_ipv4

    address = get_assigned_ipv4()
    return f"IP: {address}" if address else "IP: --"


def build_client(settings: dict[str, Any], *, presenter: Any = None, transport: Any = None,
                 screenshots: ClientScreenshots | None = None) -> DisplayClient:
    profile = resolve_display_profile_by_id(settings["DESK_DISPLAY_PROFILE"])
    if profile is None:
        raise SystemExit(f"Unknown display profile {settings['DESK_DISPLAY_PROFILE']!r}")
    cache_dir = Path(settings.get("DESK_DISPLAY_CLIENT_CACHE_DIR") or DEFAULT_CACHE_DIR).expanduser()
    cache = ClientCache(cache_dir)
    artifacts = ArtifactCache(cache_dir, max_bytes=int(settings.get("DESK_DISPLAY_CLIENT_CACHE_MAX_MB") or 512) << 20)
    configured = _rotation(settings.get("DISPLAY_ROTATION"))
    kernel_overlay = None
    hardware = presenter is None
    if presenter is None:
        from display.hardware_presenter import HardwarePresenter

        presenter = HardwarePresenter(profile=profile)
        if screenshots is None:
            # Mirror the physical panel for the config UI's Screenshots page.
            screenshots = ClientScreenshots.from_settings(settings, profile)
        import config

        kernel_overlay = getattr(config, "_kernel_overlay_rotation", None)
    strict = settings.get("DISPLAY_ROTATION_STRICT")
    decision = resolve_rotation(configured, kernel_overlay=kernel_overlay,
                                strict=bool(strict) if strict is not None else kernel_overlay is not None)
    applied = getattr(presenter, "rotation", decision.applied)
    if isinstance(applied, int) and applied != decision.applied:
        decision = RotationDecision(configured, kernel_overlay, applied, "reported by the output driver")
    server_url = settings.get("DESK_DISPLAY_SERVER_URL")
    if transport is None:
        verify: bool | str = bool(settings.get("DESK_DISPLAY_TLS_VERIFY", True))
        if verify and settings.get("DESK_DISPLAY_SERVER_CA_BUNDLE"):
            verify = str(settings["DESK_DISPLAY_SERVER_CA_BUNDLE"])
        transport = RequestsTransport(server_url, verify=verify)
    touch = str(settings.get("DESK_DISPLAY_CLIENT_TOUCH") or "auto").strip().lower()
    has_touch = touch == "on" or (touch == "auto" and "hyperpixel" in profile.profile_id.lower())
    animation = settings.get("DESK_DISPLAY_CLIENT_ANIMATION", True)
    # Update/restart requests from the Display Clients page; results wait here.
    commands = CommandRunner(cache_dir / "command_results.json", project_dir=_PROJECT_ROOT)
    sync = ClientSync(
        capabilities_for(settings["DESK_DISPLAY_CLIENT_ID"], profile, rotation=decision, has_touch=has_touch,
                         supports_animation=_truthy(animation)),
        transport,
        cache,
        artifacts,
        enrollment_token=settings.get("DESK_DISPLAY_CLIENT_TOKEN"),
        sync_interval_seconds=int(settings.get("DESK_DISPLAY_SYNC_INTERVAL_SECONDS") or 30),
        heartbeat_interval_seconds=int(settings.get("DESK_DISPLAY_HEARTBEAT_INTERVAL_SECONDS") or 60),
        commands=commands,
    )
    if screenshots is not None and screenshots.feed_summary is None:
        screenshots.feed_summary = lambda: sync.display_status
    if screenshots is not None and _truthy(settings.get("DESK_DISPLAY_CLIENT_UPLOAD_SCREENSHOTS", False)):
        # For a display on another network, which the collector cannot reach.
        minutes = float(settings.get("DESK_DISPLAY_CLIENT_SCREENSHOT_UPLOAD_MINUTES") or 10)
        uploads = UploadQueue(minutes * 60)
        screenshots.uploads = uploads
        sync.screenshot_uploads = uploads
        if not screenshots.enabled:
            LOGGER.warning("DESK_DISPLAY_CLIENT_UPLOAD_SCREENSHOTS needs ENABLE_SCREENSHOTS=1; nothing will upload")
    client = DisplayClient(
        profile, presenter, sync, cache, artifacts,
        server_url=server_url,
        physical_rotation=decision.applied,
        offline_max_age_seconds=float(settings.get("DESK_DISPLAY_OFFLINE_MAX_AGE_HOURS") or 0) * 3600,
        dark_hours=DarkHours.from_settings(settings),
        offline_start=_truthy(settings.get("DESK_DISPLAY_OFFLINE_START", True)),
        screenshots=screenshots,
        ip_text=_client_ip_text,
        update_check=_check_for_updates if hardware else None,
        restart_service=_restart_client_service if hardware else None,
        update_available=_github_update_available if hardware else None,
        local_screens=default_local_screens() if hardware else None,
        max_fps=client_max_fps(settings),
    )
    commands.restart = client.request_restart
    return client


# systemd sends SIGTERM and SIGKILLs the client 10 s later (TimeoutStopSec);
# a clean stop that stalls (a panel driver teardown, say) exits by force first.
SHUTDOWN_GRACE_SECONDS = 5.0


class StopOnSignal:
    """Turn SIGTERM into a prompt, clean stop of the client.

    SDL (pygame, used by the HyperPixel and window panels) replaces a default
    SIGTERM action with a handler that only queues a quit event, which the
    client never reads, so ``systemctl stop`` used to wait out its timeout and
    SIGKILL the client. A Python handler installed before the panel opens
    keeps SDL from taking the signal over.
    """

    def __init__(self, *, grace_seconds: float = SHUTDOWN_GRACE_SECONDS,
                 force_exit: Callable[[int], Any] = os._exit) -> None:
        self.grace_seconds = grace_seconds
        self.force_exit = force_exit
        self.target: Callable[[], None] | None = None
        self.received = False

    def install(self) -> None:
        signal.signal(signal.SIGTERM, self.handle)

    def handle(self, signum: int, _frame: Any = None) -> None:
        if self.received:
            return
        self.received = True
        LOGGER.info("Received %s; stopping the client", signal.Signals(signum).name)
        watchdog = threading.Timer(self.grace_seconds, self._stalled)
        watchdog.daemon = True
        watchdog.start()
        target = self.target
        if target is None:
            # Still starting up: nothing to stop cleanly yet.
            raise SystemExit(0)
        # Stop from another thread: the handler runs on the main thread, which
        # may hold the locks stop() needs.
        threading.Thread(target=target, name="client-stop", daemon=True).start()

    def _stalled(self) -> None:
        LOGGER.warning("Client still running %.0f s after SIGTERM; exiting now", self.grace_seconds)
        self.force_exit(0)


def prepare_environment() -> None:
    """Load the client's settings; call before anything imports ``config``.

    ``config`` loads a dotenv file when it is first imported, which in a
    client process happens after start-up (see
    rendering.profile_process.configure_native), so name .env.client for the
    whole process: it must never load the server's .env beside it.
    """

    import deployment_config

    os.environ.setdefault("DESK_DISPLAY_DOTENV_FILE", ".env.client")
    _load_env_files()
    os.environ.setdefault(deployment_config.ROLE_ENV, "client")


def main() -> None:  # pragma: no cover - exercised on hardware
    import deployment_config

    prepare_environment()
    # Before the panel opens: SDL leaves SIGTERM alone once it has a handler.
    os.environ.setdefault("SDL_NO_SIGNAL_HANDLERS", "1")
    stopper = StopOnSignal()
    stopper.install()
    logging.basicConfig(level=deployment_config.resolve_log_level())
    deployment_config.install_secret_log_redaction()
    deployment_config.require_role("display_client.py", deployment_config.Role.CLIENT)
    deployment_config.startup_check("display client")
    settings = deployment_config.load_settings(deployment_config.Role.CLIENT)
    profile = resolve_display_profile_by_id(settings["DESK_DISPLAY_PROFILE"])
    if profile is not None:
        # Clock faces are drawn here; size their fonts for this panel as v0.1 did.
        from rendering.profile_process import configure_native

        configure_native(profile)
    client = build_client(settings)
    stopper.target = client.stop
    start_wifi_monitor(settings)
    try:
        client.run()
    except KeyboardInterrupt:
        pass
    finally:
        client.stop()
        close = getattr(client.presenter, "close", None)
        if callable(close):
            close()
    if client.restart_requested or stopper.received:
        # Exit even if a helper thread lingers; on a restart systemd
        # (Restart=always) starts the client again.
        logging.shutdown()
        os._exit(0)


if __name__ == "__main__":  # pragma: no cover
    main()
