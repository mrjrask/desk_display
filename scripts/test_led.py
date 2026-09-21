#!/usr/bin/env python3
"""Drive the Display HAT Mini's RGB status LED through a color cycle.

Confirms on real hardware that ``Display.set_led()`` actually reaches the
physical LED. The ``displayhatmini`` driver takes 0.0-1.0 floats and raises
``ValueError`` for anything outside that range, and ``Display.set_led()``
turns such a failure into a single log warning, so a dead LED is otherwise
easy to miss. This script watches for that warning and reports a pass or fail
instead of leaving it to be spotted in the log scroll.

The LED sits on the same GPIO lines the running display service holds, so stop
the service before running it::

    sudo systemctl stop desk_display.service
    python3 scripts/test_led.py
    bash scripts/restart_services.sh

Exits 0 when the whole cycle is accepted by the driver, and 1 when no display
driver is attached or the driver rejects an update.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    # Only re-exec when this script is run directly, not when it is imported as
    # a library (e.g. by the tests) -- see _venv_bootstrap.py.
    try:
        from scripts._venv_bootstrap import reexec_with_project_venv
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _venv_bootstrap import reexec_with_project_venv
    reexec_with_project_venv()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils  # noqa: E402 -- the project root has to reach sys.path first.

DISPLAY_SERVICE = "desk_display.service"

#: Substring of the warning ``Display.set_led()`` logs when the driver rejects
#: an update. Kept in sync with ``utils.Display.set_led``.
LED_FAILURE_MARKER = "Display LED update failed"

#: The LED is driven by RPi.GPIO software PWM, which applies a new duty cycle
#: from a background thread. Exiting immediately after the final
#: ``set_led(0, 0, 0)`` can kill that thread before it drives the pins, which
#: leaves the LED stuck lit at whatever color came last.
LED_SETTLE_SECONDS = 0.5

DEFAULT_HOLD_SECONDS = 2.0

COLOR_CYCLE: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("red", (1.0, 0.0, 0.0)),
    ("green", (0.0, 1.0, 0.0)),
    ("blue", (0.0, 0.0, 1.0)),
    ("white", (1.0, 1.0, 1.0)),
)


class LedFailureWatcher(logging.Handler):
    """Collect the warnings ``Display.set_led()`` logs when the driver fails."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.failures: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if LED_FAILURE_MARKER in message:
            self.failures.append(message)


@contextlib.contextmanager
def watch_for_led_failures() -> Iterator[LedFailureWatcher]:
    """Capture swallowed LED driver failures for the duration of the block."""

    watcher = LedFailureWatcher()
    root_logger = logging.getLogger()
    root_logger.addHandler(watcher)
    try:
        yield watcher
    finally:
        root_logger.removeHandler(watcher)


def describe_configuration() -> list[str]:
    """Return the LED-relevant configuration, as printable lines."""

    return [
        f"LED_INDICATOR_ENABLED        : {utils.LED_INDICATOR_ENABLED}",
        f"LED_INDICATOR_BORDER_ENABLED : {utils.LED_INDICATOR_BORDER_ENABLED}",
        f"LED_INDICATOR_LEVEL          : {utils.LED_INDICATOR_LEVEL}",
        f"displayhatmini driver        : {utils.DisplayHATMini}",
    ]


def check_driver_channel_range() -> tuple[bool, str]:
    """Verify normalized LED values reach the driver as 0.0-1.0 floats.

    The driver rejects anything outside that range, so a regression back to an
    8-bit 0-255 channel would silently stop the LED from ever lighting.
    """

    full = utils._normalized_led_to_driver_channel(1.0)
    above = utils._normalized_led_to_driver_channel(1.5)
    below = utils._normalized_led_to_driver_channel(-0.5)
    ok = (full, above, below) == (1.0, 1.0, 0.0)
    detail = (
        f"Driver channel for 1.0/1.5/-0.5: {full}/{above}/{below} "
        f"({'expected 1.0/1.0/0.0' if ok else 'EXPECTED 1.0/1.0/0.0'})"
    )
    return ok, detail


def display_service_is_active(
    service: str = DISPLAY_SERVICE,
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> bool:
    """Return True when systemd reports *service* as running."""

    try:
        completed = runner(["systemctl", "is-active", "--quiet", service], check=False)
    except (OSError, ValueError):
        # No systemd, or systemctl is missing: nothing to contend with.
        return False
    return completed.returncode == 0


def cycle_colors(
    display: utils.Display,
    *,
    level: float,
    hold_seconds: float,
    colors: Sequence[tuple[str, tuple[float, float, float]]] = COLOR_CYCLE,
    sleep: Callable[[float], None] = time.sleep,
    announce: Callable[[str], None] = print,
) -> None:
    """Light each color in *colors* at *level* brightness, in turn."""

    for name, (red, green, blue) in colors:
        announce(f"  LED should now be {name.upper()}")
        display.set_led(r=red * level, g=green * level, b=blue * level)
        sleep(hold_seconds)


def turn_off_and_release(
    display: utils.Display,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Turn the LED off, let the change settle, then release the GPIO lines.

    Both steps matter: the settle gives the software PWM thread time to drive
    the pins, and releasing hands them back instead of leaving them held at
    their last level once this process goes away.
    """

    display.set_led(r=0.0, g=0.0, b=0.0)
    sleep(LED_SETTLE_SECONDS)

    driver = getattr(display, "_display", None)
    if driver is None:
        return
    try:
        display._release_display_hat_mini_compat(driver, call_destructor=True)
    except Exception as exc:
        logging.debug("Failed to release the Display HAT Mini driver: %s", exc)


def _led_level(raw: str) -> float:
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("LED level must be between 0.0 and 1.0")
    return value


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hold",
        type=float,
        default=DEFAULT_HOLD_SECONDS,
        metavar="SECONDS",
        help=f"seconds to hold each color; defaults to {DEFAULT_HOLD_SECONDS}",
    )
    parser.add_argument(
        "--level",
        type=_led_level,
        default=1.0,
        metavar="LEVEL",
        help=(
            "LED brightness for the cycle, 0.0-1.0; defaults to full brightness "
            "so a working LED is unmistakable, unlike the dim level used for "
            "update notifications"
        ),
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="print the LED configuration and exit without touching hardware",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="log at debug level",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    for line in describe_configuration():
        print(line)

    range_ok, range_detail = check_driver_channel_range()
    print(range_detail)
    if not range_ok:
        print("FAIL: normalized LED values are not reaching the driver as 0.0-1.0 floats.")
        return 1

    if args.config_only:
        return 0

    if display_service_is_active():
        print(
            f"WARNING: {DISPLAY_SERVICE} is still running and holds the same GPIO/SPI "
            f"lines. Stop it first, or this test's results are unreliable."
        )

    display = utils.Display()
    print(f"Output strategy: {display._output_strategy} | driver: {display._display_driver}")
    if display._display is None:
        print("FAIL: no display driver is attached, so the LED cannot light.")
        return 1

    with watch_for_led_failures() as watcher:
        try:
            cycle_colors(display, level=args.level, hold_seconds=args.hold)
        finally:
            print("  LED should now be OFF")
            turn_off_and_release(display)

    if watcher.failures:
        print(f"FAIL: the driver rejected {len(watcher.failures)} LED update(s):")
        for failure in watcher.failures:
            print(f"  {failure}")
        return 1

    print("PASS: the driver accepted every LED update.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
