"""Tests for Display HAT Mini button handling utilities."""

from types import SimpleNamespace

import utils


def _make_display(return_value):
    display = utils.Display()
    display._display = SimpleNamespace(read_button=lambda pin: return_value)  # type: ignore[attr-defined]
    display._button_pins["X"] = 16
    return display


def test_is_button_pressed_true_from_bool():
    display = _make_display(True)

    assert display.is_button_pressed("X") is True


def test_is_button_pressed_false_from_bool():
    display = _make_display(False)

    assert display.is_button_pressed("X") is False


def test_is_button_pressed_handles_active_low_int():
    display = _make_display(0)

    assert display.is_button_pressed("X") is True


def test_is_button_pressed_handles_inactive_int():
    display = _make_display(1)

    assert display.is_button_pressed("X") is False
