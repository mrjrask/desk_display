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


def test_update_display_resizes_rotated_hardware_buffer_to_native_size():
    display = utils.Display()

    captured = {}

    class _FakeHardwareDisplay:
        buffer = None

        def display(self):
            captured["size"] = self.buffer.size

    display._display = _FakeHardwareDisplay()
    display.rotation = 90
    display._buffer = utils.Image.new("RGB", (display.width, display.height), "black")

    display._update_display()

    assert captured["size"] == (display.width, display.height)


def test_hyperpixel_indicator_border_renders_led_color(monkeypatch):
    monkeypatch.setattr(utils, "is_hyperpixel_next_layout", lambda w, h: True)

    display = utils.Display()
    display._buffer = utils.Image.new("RGB", (display.width, display.height), "black")

    display.set_led(r=0.0, g=0.0, b=utils.LED_INDICATOR_LEVEL)

    pixel = display._indicator_buffer().getpixel((0, 0))
    assert pixel == (0, 0, 255)


def test_hyperpixel_indicator_border_clears_when_led_is_off(monkeypatch):
    monkeypatch.setattr(utils, "is_hyperpixel_next_layout", lambda w, h: True)

    display = utils.Display()
    display._buffer = utils.Image.new("RGB", (display.width, display.height), "black")

    display.set_led(r=0.0, g=0.0, b=0.0)

    pixel = display._indicator_buffer().getpixel((0, 0))
    assert pixel == (0, 0, 0)


def test_hyperpixel_indicator_border_respects_config_disable(monkeypatch):
    monkeypatch.setattr(utils, "is_hyperpixel_next_layout", lambda w, h: True)
    monkeypatch.setattr(utils, "HYPERPIXEL_LED_INDICATOR_BORDER_ENABLED", False)

    display = utils.Display()
    display._buffer = utils.Image.new("RGB", (display.width, display.height), "black")
    display.set_led(r=0.0, g=0.0, b=utils.LED_INDICATOR_LEVEL)

    pixel = display._indicator_buffer().getpixel((0, 0))
    assert pixel == (0, 0, 0)
