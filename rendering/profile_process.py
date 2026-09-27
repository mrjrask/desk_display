"""Render each display profile in a process configured for that display.

The v0.1 standalone display imported :mod:`config` and the screen modules
once, for its own panel: fonts, logo sizes and layout constants were all
derived from ``DISPLAY_WIDTH``/``DISPLAY_HEIGHT`` (and a few install-time
settings) at import time.  A render server draws many profiles, and swapping
those values per composition inside one interpreter
(:func:`screens.registry._profile_composition_globals`) cannot reach every
value a renderer derives at import time, so fonts and layouts drifted from
v0.1.

This module keeps one long-lived worker process per profile instead.  Each
worker is started with the environment the v0.1 installer wrote for that
display (:func:`composition_env`), so its module state is exactly what the
standalone display had, and it composes only that profile.

Requests and replies are pickled frames on the worker's stdin and a private
copy of its stdout; the worker points its own stdout at stderr so a stray
``print`` in a renderer cannot corrupt the stream.  A worker exits when its
parent closes the pipe.
"""
from __future__ import annotations

import logging
import os
import pickle
import struct
import subprocess
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO

from display_profiles import (
    DISPLAY_PROFILE_ADAFRUIT_MINIPITFT_114,
    DISPLAY_PROFILE_DISPLAY_HAT_MINI,
    DISPLAY_PROFILE_FALLBACK_DEFAULT,
    DISPLAY_PROFILE_FALLBACK_HD,
    DISPLAY_PROFILE_HDMI_1080P,
    DISPLAY_PROFILE_HYPERPIXEL4,
    DISPLAY_PROFILE_HYPERPIXEL4_SQUARE,
    DISPLAY_PROFILE_WAVESHARE_LCD_320X240,
    DISPLAY_PROFILE_WAVESHARE_OLED_128X64,
    RenderProfile,
    resolve_display_profile_by_id,
)

LOGGER = logging.getLogger("desk_display.profile_process")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_ENV = "DESK_DISPLAY_RENDER_WORKER_PROFILE"
_HEADER = struct.Struct(">Q")

# What each v0.1 installer wrote to .env for its display, limited to the
# settings screens read while composing.  An empty DESK_DISPLAY_PROFILE means
# "resolve from the size", which is what v0.1 did; the Waveshare LCD was a
# 320x240 display_hat_mini profile marked by its OLED font settings.
_WAVESHARE = {
    "DESK_DISPLAY_OUTPUT": "framebuffer",
    "WAVESHARE_OLED_MAX_VALUE_FONT_SIZE": "26",
    "WAVESHARE_OLED_MAX_TIME_FONT_SIZE": "24",
}
_INSTALL_ENV: dict[str, dict[str, str]] = {
    DISPLAY_PROFILE_DISPLAY_HAT_MINI: {"DESK_DISPLAY_OUTPUT": "displayhatmini"},
    DISPLAY_PROFILE_ADAFRUIT_MINIPITFT_114: {"DESK_DISPLAY_OUTPUT": "minipitft"},
    DISPLAY_PROFILE_HYPERPIXEL4: {"DESK_DISPLAY_OUTPUT": "kernel", "HYPERPIXEL_PANEL": "hyperpixel4"},
    DISPLAY_PROFILE_HYPERPIXEL4_SQUARE: {"DESK_DISPLAY_OUTPUT": "kernel", "HYPERPIXEL_PANEL": "hyperpixel4sq"},
    DISPLAY_PROFILE_WAVESHARE_LCD_320X240: dict(_WAVESHARE),
    # Not a main display at v0.1; composed as its own profile.
    DISPLAY_PROFILE_WAVESHARE_OLED_128X64: {**_WAVESHARE, "DESK_DISPLAY_PROFILE": DISPLAY_PROFILE_WAVESHARE_OLED_128X64},
    DISPLAY_PROFILE_HDMI_1080P: {"DESK_DISPLAY_OUTPUT": "kernel"},
    DISPLAY_PROFILE_FALLBACK_HD: {"DESK_DISPLAY_OUTPUT": "kernel"},
    DISPLAY_PROFILE_FALLBACK_DEFAULT: {"DESK_DISPLAY_OUTPUT": "auto", "DESK_DISPLAY_PROFILE": DISPLAY_PROFILE_FALLBACK_DEFAULT},
}
# Display settings a server's own .env may carry that must not leak into a
# worker composing some other display.
_CLEARED = ("HYPERPIXEL_PANEL", "WAVESHARE_OLED_MAX_VALUE_FONT_SIZE", "WAVESHARE_OLED_MAX_TIME_FONT_SIZE",
            "WAVESHARE_OLED_LCD_HAT_A_INSTALLED")


def composition_env(profile: RenderProfile) -> dict[str, str | None]:
    """Environment overrides that configure a process for *profile* as v0.1 did.

    ``None`` means the variable is removed.
    """

    env: dict[str, str | None] = dict.fromkeys(_CLEARED)
    env.update({
        "DISPLAY_WIDTH": str(profile.width),
        "DISPLAY_HEIGHT": str(profile.height),
        "DESK_DISPLAY_PROFILE": None,
    })
    env.update(_INSTALL_ENV.get(profile.profile_id, {"DESK_DISPLAY_PROFILE": profile.profile_id}))
    return env


class ProfileProcessError(RuntimeError):
    """A render worker failed, or a render in it raised."""


def _encode(message: Any) -> bytes:
    body = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    return _HEADER.pack(len(body)) + body


def _write(stream: BinaryIO, message: Any) -> None:
    stream.write(_encode(message))
    stream.flush()


def _read(stream: BinaryIO) -> Any:
    header = stream.read(_HEADER.size)
    if len(header) < _HEADER.size:
        raise EOFError("render worker pipe closed")
    (size,) = _HEADER.unpack(header)
    body = stream.read(size)
    if len(body) < size:
        raise EOFError("render worker pipe closed")
    return pickle.loads(body)  # noqa: S301 - frames come from our own child process


def _snapshot_message(snapshot: Any) -> tuple[Any, ...]:
    from rendering.screen_renderer import _thaw_legacy_data

    return (snapshot.revision, snapshot.created_at, _thaw_legacy_data(snapshot.values),
            dict(snapshot.source_revisions))


def _snapshot_from_message(message: tuple[Any, ...]) -> Any:
    from services.data_coordinator import DataSnapshot, _freeze

    revision, created_at, values, sources = message
    return DataSnapshot(revision=revision, created_at=created_at, values=_freeze(values),
                        source_revisions=MappingProxyType(dict(sources)))


class _Worker:
    def __init__(self, profile: RenderProfile, python: str, timeout: float) -> None:
        self.profile = profile
        self._python = python
        self._timeout = timeout
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._snapshot_tag: Any = None

    def _start(self) -> subprocess.Popen[bytes]:
        env = dict(os.environ)
        for name, value in composition_env(self.profile).items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
        env[PROFILE_ENV] = self.profile.profile_id
        # Settings were loaded into this process already; the worker inherits them.
        env["CONFIG_LOAD_DOTENV"] = "0"
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing if existing else "")
        LOGGER.info("Starting render worker for %s", self.profile.profile_id)
        process = subprocess.Popen(  # noqa: S603 - our own interpreter and module
            [self._python, "-m", "rendering.profile_process"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=str(PROJECT_ROOT), env=env,
        )
        self._snapshot_tag = None
        try:
            ready = self._guarded(lambda: _read(process.stdout), process)
        except (EOFError, OSError, pickle.UnpicklingError) as exc:
            process.kill()
            process.wait()
            raise ProfileProcessError(f"render worker for {self.profile.profile_id} did not start") from exc
        if not (isinstance(ready, Mapping) and ready.get("ready")):
            process.kill()
            process.wait()
            raise ProfileProcessError(f"render worker for {self.profile.profile_id} failed: {ready!r}")
        return process

    def request(self, message: dict[str, Any], snapshot: Any = None) -> dict[str, Any]:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._process = self._start()
            process = self._process
            tag = None
            if snapshot is not None:
                tag = (snapshot.revision, tuple(sorted(snapshot.source_revisions.items())))
                if tag != self._snapshot_tag:
                    message = {**message, "snapshot": _snapshot_message(snapshot)}
            try:
                frame = _encode(message)
            except (pickle.PickleError, TypeError, AttributeError) as exc:
                raise ProfileProcessError(f"render input cannot be sent to the worker: {exc}") from exc

            def exchange() -> Any:
                process.stdin.write(frame)
                process.stdin.flush()
                return _read(process.stdout)

            try:
                reply = self._guarded(exchange, process)
            except (EOFError, OSError, pickle.PickleError) as exc:
                self._stop(process)
                raise ProfileProcessError(
                    f"render worker for {self.profile.profile_id} stopped: {exc}") from exc
            if tag is not None:
                self._snapshot_tag = tag
        if not reply.get("ok"):
            error = reply.get("error", "render failed")
            if reply.get("type") == "KeyError":
                raise KeyError(error)
            raise ProfileProcessError(error)
        return reply

    def _guarded(self, call: Any, process: subprocess.Popen[bytes]) -> Any:
        """Run *call*, killing the worker if it takes longer than the timeout.

        A hung render would otherwise hold this profile's lock for good; the
        kill turns it into an EOF, and the next request starts a new worker.
        """

        timer = threading.Timer(self._timeout, process.kill)
        timer.daemon = True
        timer.start()
        try:
            return call()
        finally:
            timer.cancel()

    def _stop(self, process: subprocess.Popen[bytes]) -> None:
        self._process = None
        self._snapshot_tag = None
        try:
            process.kill()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):  # pragma: no cover - best effort
            pass

    def close(self) -> None:
        with self._lock:
            process, self._process = self._process, None
        if process is None:
            return
        try:
            process.stdin.close()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()


class ProfileProcessPool:
    """One render worker per profile, started on first use."""

    def __init__(self, *, python: str | None = None, timeout_seconds: float = 120) -> None:
        """*timeout_seconds* bounds a worker's start-up and each render in it."""

        self._python = python or sys.executable
        self._timeout = float(timeout_seconds)
        self._workers: dict[str, _Worker] = {}
        self._lock = threading.Lock()

    def _worker(self, profile: RenderProfile) -> _Worker:
        with self._lock:
            worker = self._workers.get(profile.profile_id)
            if worker is None:
                worker = self._workers[profile.profile_id] = _Worker(profile, self._python, self._timeout)
            return worker

    def render_screen(self, key: Any, profile: RenderProfile, snapshot: Any,
                      weather_fetched_at: Any = None) -> dict[str, Any]:
        """``{"image", "metadata", "package"}`` for *key*, composed natively."""

        return self._worker(profile).request(
            {"op": "screen", "key": key, "weather_fetched_at": weather_fetched_at}, snapshot)

    def render_clock(self, layout: Mapping[str, Any], profile: RenderProfile, now: Any) -> Any:
        return self._worker(profile).request({"op": "clock", "layout": dict(layout), "now": now})["image"]

    def close(self) -> None:
        with self._lock:
            workers, self._workers = list(self._workers.values()), {}
        for worker in workers:
            worker.close()


def configure_native(profile: RenderProfile) -> bool:
    """Import ``config`` for *profile* in this process; return whether it took.

    Call before anything imports :mod:`config`.  Used by the display client,
    whose only profile is its own panel's.
    """

    if "config" not in sys.modules:
        for name, value in composition_env(profile).items():
            if name in {"DISPLAY_WIDTH", "DISPLAY_HEIGHT"} and value is not None:
                os.environ[name] = value
    from screens.registry import set_native_profile

    native = set_native_profile(profile)
    if not native:
        LOGGER.warning("Display settings do not match profile %s; composing with substituted sizes.",
                       profile.profile_id)
    return native


def _serve(profile: RenderProfile, requests: BinaryIO, replies: BinaryIO) -> None:
    from remote_display.server_rendering import compose_screen
    from rendering.clock_faces import render_clock
    from rendering.logos import ProfileLogos

    logos = ProfileLogos(ahl_tricode=os.environ.get("AHL_TEAM_TRICODE", "CHI"))
    snapshot = None
    _write(replies, {"ready": True})
    while True:
        try:
            message = _read(requests)
        except EOFError:
            return
        try:
            if "snapshot" in message:
                snapshot = _snapshot_from_message(message["snapshot"])
            if message["op"] == "clock":
                reply = {"ok": True, "image": render_clock(message["layout"], profile, message["now"])}
            else:
                if snapshot is None:
                    raise ProfileProcessError("render worker has no data snapshot")
                image, metadata, package = compose_screen(
                    message["key"], profile, snapshot, logos, message.get("weather_fetched_at"))
                reply = {"ok": True, "image": image, "metadata": metadata, "package": package}
        except Exception as exc:  # noqa: BLE001 - reported to the parent, which records it
            LOGGER.exception("Render failed in worker for %s", profile.profile_id)
            reply = {"ok": False, "type": type(exc).__name__, "error": f"{type(exc).__name__}: {exc}"}
        try:
            _write(replies, reply)
        except (pickle.PickleError, TypeError, AttributeError) as exc:
            _write(replies, {"ok": False, "type": "PicklingError", "error": f"unsendable render result: {exc}"})


def main() -> None:  # pragma: no cover - exercised through ProfileProcessPool
    replies = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    requests = sys.stdin.buffer
    import deployment_config

    logging.basicConfig(level=deployment_config.resolve_log_level(),
                        format="%(asctime)s %(levelname)-8s [render %(process)d] %(message)s",
                        datefmt="%H:%M:%S")
    deployment_config.install_secret_log_redaction()
    profile = resolve_display_profile_by_id(os.environ.get(PROFILE_ENV, ""))
    if profile is None:
        _write(replies, {"ready": False, "error": "unknown profile"})
        return
    from screens.registry import set_native_profile

    if not set_native_profile(profile):
        _write(replies, {"ready": False, "error": "display settings did not apply"})
        return
    _serve(profile, requests, replies)


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["ProfileProcessError", "ProfileProcessPool", "composition_env", "configure_native"]
