"""Tests for the standalone Display HAT Mini LED diagnostic script."""

from __future__ import annotations

import logging

import pytest

import utils
from scripts import test_led


class _RecordingDriver:
    """Stand-in for the ``displayhatmini`` driver, recording what it receives."""

    def __init__(self, events: list, *, reject: bool = False):
        self._events = events
        self._reject = reject

    def set_led(self, r=0.0, g=0.0, b=0.0):
        if self._reject or not (0.0 <= r <= 1.0 and 0.0 <= g <= 1.0 and 0.0 <= b <= 1.0):
            # Matches the real driver, which only accepts 0.0-1.0 floats.
            raise ValueError("r, g, and b must be in the range 0.0 to 1.0")
        self._events.append(("led", (r, g, b)))

    def display(self):
        """Accept a frame push, as the real driver does after ``buffer`` is set.

        ``Display.set_led()`` repaints the indicator border, which routes a
        frame here; a driver that cannot take one trips the consecutive-refresh
        failure limit and exits the process.
        """

    def cleanup(self):
        self._events.append(("released", None))


def _headless_display(driver) -> utils.Display:
    display = utils.Display()
    display._display = driver
    return display


def test_check_driver_channel_range_accepts_current_implementation():
    ok, detail = test_led.check_driver_channel_range()

    assert ok is True
    assert "1.0/1.0/0.0" in detail


def test_check_driver_channel_range_rejects_8_bit_scaling(monkeypatch):
    """A regression to the old 0-255 channel must be reported, not tolerated."""

    monkeypatch.setattr(
        utils,
        "_normalized_led_to_driver_channel",
        lambda value: round(max(0.0, min(1.0, value)) * 255),
    )

    ok, detail = test_led.check_driver_channel_range()

    assert ok is False
    assert "EXPECTED" in detail


def test_watch_for_led_failures_captures_swallowed_driver_error():
    """``Display.set_led()`` only logs a driver rejection, so the watcher is
    the script's sole evidence that the physical LED never lit."""

    display = _headless_display(_RecordingDriver([], reject=True))

    with test_led.watch_for_led_failures() as watcher:
        display.set_led(r=1.0, g=0.0, b=0.0)

    assert len(watcher.failures) == 1
    assert test_led.LED_FAILURE_MARKER in watcher.failures[0]
    assert watcher not in logging.getLogger().handlers


def test_watch_for_led_failures_stays_quiet_when_the_driver_accepts():
    display = _headless_display(_RecordingDriver([]))

    with test_led.watch_for_led_failures() as watcher:
        display.set_led(r=1.0, g=0.5, b=0.0)

    assert watcher.failures == []


def test_watch_for_led_failures_ignores_unrelated_warnings():
    with test_led.watch_for_led_failures() as watcher:
        logging.warning("Failed to set backlight level: nope")

    assert watcher.failures == []


def test_cycle_colors_scales_each_color_by_the_requested_level():
    events: list = []
    display = _headless_display(_RecordingDriver(events))

    test_led.cycle_colors(
        display,
        level=0.5,
        hold_seconds=1.5,
        sleep=lambda seconds: events.append(("slept", seconds)),
        announce=lambda _message: None,
    )

    assert events == [
        ("led", (0.5, 0.0, 0.0)),
        ("slept", 1.5),
        ("led", (0.0, 0.5, 0.0)),
        ("slept", 1.5),
        ("led", (0.0, 0.0, 0.5)),
        ("slept", 1.5),
        ("led", (0.5, 0.5, 0.5)),
        ("slept", 1.5),
    ]


def test_turn_off_and_release_settles_before_releasing_the_pins():
    """Regression test: releasing (or exiting) immediately after the final
    ``set_led(0, 0, 0)`` can kill the software PWM thread before it drives the
    pins, leaving the LED stuck lit."""

    events: list = []
    display = _headless_display(_RecordingDriver(events))

    test_led.turn_off_and_release(
        display,
        sleep=lambda seconds: events.append(("slept", seconds)),
    )

    assert events == [
        ("led", (0.0, 0.0, 0.0)),
        ("slept", test_led.LED_SETTLE_SECONDS),
        ("released", None),
    ]


def test_turn_off_and_release_without_hardware_is_a_no_op():
    display = utils.Display()
    display._display = None

    test_led.turn_off_and_release(display, sleep=lambda _seconds: None)

    assert display._led_color == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("returncode", "expected"),
    [(0, True), (3, False)],
)
def test_display_service_is_active_reads_the_systemctl_exit_code(returncode, expected):
    def _runner(command, check=False):
        assert command == ["systemctl", "is-active", "--quiet", "desk_display.service"]
        assert check is False
        return type("Completed", (), {"returncode": returncode})()

    assert test_led.display_service_is_active(runner=_runner) is expected


def test_display_service_is_active_tolerates_a_missing_systemctl():
    def _runner(_command, check=False):
        raise FileNotFoundError("systemctl")

    assert test_led.display_service_is_active(runner=_runner) is False


def test_config_only_run_reports_configuration_without_touching_hardware(monkeypatch, capsys):
    def _fail(*_args, **_kwargs):
        raise AssertionError("--config-only must not construct a Display")

    monkeypatch.setattr(utils, "Display", _fail)

    assert test_led.main(["--config-only"]) == 0

    output = capsys.readouterr().out
    assert "LED_INDICATOR_ENABLED" in output
    assert "Driver channel for 1.0/1.5/-0.5" in output


def test_level_argument_rejects_values_outside_the_driver_range():
    with pytest.raises(SystemExit):
        test_led.main(["--level", "255"])
