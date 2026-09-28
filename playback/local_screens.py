"""Screens a display client draws itself from its own hardware.

The ``inside`` screen (class ``client_sensor``) reads the indoor sensor wired
to this display. The server never renders it: each client probes its own
sensor, using the same ``INSIDE_SENSOR``, ``INSIDE_I2C_BUSES`` and
``INSIDE_I2C_ADDRESS`` settings (from ``.env.client``) that a standalone
display reads from ``.env``, and draws the screen with the standalone layout
code, so it looks exactly as it did on v0.1. A client without a sensor skips
the screen, as the standalone display did.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from PIL import Image

LOGGER = logging.getLogger("desk_display.client")
LOCAL_MARKER = "local"


class SensorScreen:
    """The ``inside`` screen, drawn on this device from its own sensor.

    The first probe of the I2C buses can take a few seconds, so it runs in
    the background the first time a playlist asks for the screen; until it
    finishes, and forever when no sensor answers, the screen is skipped.
    Each showing reads the sensor once, as v0.1 did.
    """

    def __init__(
        self,
        screen_id: str = "inside",
        *,
        probe: Callable[[], bool] | None = None,
        render: Callable[[], Image.Image] | None = None,
    ) -> None:
        self.screen_id = screen_id
        self._probe = probe or _probe_inside_sensor
        self._render = render or _render_inside_screen
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._available: bool | None = None  # None until the probe finishes

    def start(self) -> None:
        """Probe the sensor in the background, once."""

        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run_probe, name="sensor-probe", daemon=True)
            self._thread.start()

    def _run_probe(self) -> None:
        try:
            available = bool(self._probe())
        except Exception:  # noqa: BLE001 - a sensor fault only hides this screen
            LOGGER.warning("Indoor sensor probe failed; the %s screen is skipped", self.screen_id, exc_info=True)
            available = False
        self._available = available
        if available:
            LOGGER.info("Indoor sensor found; this display draws the %s screen", self.screen_id)
        else:
            LOGGER.info("No indoor sensor on this display; the %s screen is skipped", self.screen_id)

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the probe (tests and diagnostics); return whether it finished."""

        thread = self._thread
        if thread is None:
            return False
        thread.join(timeout)
        return not thread.is_alive()

    @property
    def available(self) -> bool:
        return bool(self._available)

    def render(self, width: int, height: int, color_mode: str) -> Image.Image | None:
        """Read the sensor and draw the screen at the client's logical size."""

        if not self.available:
            return None
        try:
            image = self._render()
        except Exception:  # noqa: BLE001 - one bad read must not stop playback
            LOGGER.warning("Could not draw the %s screen", self.screen_id, exc_info=True)
            return None
        if not isinstance(image, Image.Image):
            return None
        if image.size != (width, height):
            image = image.resize((width, height), Image.LANCZOS)
        return image if image.mode == color_mode else image.convert(color_mode)


def local_entries(screens: Iterable[str], local: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Playback entries for the client-drawn screens among *screens*, starting their probes."""

    entries = {}
    for screen in screens:
        drawer = local.get(screen)
        if drawer is None:
            continue
        drawer.start()
        entries[screen] = {"screen_id": screen, LOCAL_MARKER: True}
    return entries


def is_local(entry: Any) -> bool:
    return isinstance(entry, Mapping) and entry.get(LOCAL_MARKER) is True


def default_local_screens() -> dict[str, SensorScreen]:
    """The client-drawn screens a hardware client offers."""

    from rendering.screen_classes import CLASSIFICATIONS, CLIENT_SENSOR

    return {sid: SensorScreen(sid) for sid, entry in CLASSIFICATIONS.items() if entry.kind == CLIENT_SENSOR}


def _probe_inside_sensor() -> bool:
    # Imported here: draw_inside imports config, which a client may import
    # only after rendering.profile_process.configure_native has run.
    from screens.draw_inside import is_inside_sensor_available

    return is_inside_sensor_available()


def _render_inside_screen() -> Image.Image:
    from screens.draw_inside import render_inside_screen

    return render_inside_screen()


__all__ = ["LOCAL_MARKER", "SensorScreen", "default_local_screens", "is_local", "local_entries"]
