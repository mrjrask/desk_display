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
    monkeypatch.setattr(utils, "DISPLAY_HAT_MINI_LED_INDICATOR_BORDER_ENABLED", False)

    display = utils.Display()
    display._buffer = utils.Image.new("RGB", (display.width, display.height), "black")
    display.set_led(r=0.0, g=0.0, b=utils.LED_INDICATOR_LEVEL)

    pixel = display._indicator_buffer().getpixel((0, 0))
    assert pixel == (0, 0, 0)


def test_display_hat_mini_indicator_border_renders_led_color(monkeypatch):
    monkeypatch.setattr(utils, "WIDTH", 320)
    monkeypatch.setattr(utils, "HEIGHT", 240)
    monkeypatch.setattr(utils, "DISPLAY_HAT_MINI_LED_INDICATOR_BORDER_ENABLED", True)

    display = utils.Display()
    display._buffer = utils.Image.new("RGB", (display.width, display.height), "black")

    display.set_led(r=0.0, g=0.0, b=utils.LED_INDICATOR_LEVEL)

    pixel = display._indicator_buffer().getpixel((0, 0))
    assert pixel == (0, 0, 255)


def test_display_hat_mini_led_respects_config_disable(monkeypatch):
    monkeypatch.setattr(utils, "DISPLAY_HAT_MINI_LED_ENABLED", False)

    class _FakeHardwareDisplay:
        def __init__(self):
            self.called = False

        def set_led(self, **kwargs):
            self.called = True

    display = utils.Display()
    fake_display = _FakeHardwareDisplay()
    display._display = fake_display

    display.set_led(r=0.1, g=0.2, b=0.3)

    assert fake_display.called is False


def test_image_always_applies_bottom_safe_buffer(monkeypatch):
    monkeypatch.setattr(utils, "is_hyperpixel_next_layout", lambda w, h: True)

    display = utils.Display()
    source = utils.Image.new("RGB", (display.width, display.height), (255, 0, 0))

    display.image(source)

    assert display.capture().getpixel((10, 0)) == (255, 0, 0)
    assert display.capture().getpixel((10, display.height - 6)) == (255, 0, 0)
    assert display.capture().getpixel((10, display.height - 1)) == (0, 0, 0)


def test_image_applies_bottom_safe_buffer_when_indicator_border_disabled(monkeypatch):
    monkeypatch.setattr(utils, "is_hyperpixel_next_layout", lambda w, h: False)
    monkeypatch.setattr(utils, "DISPLAY_HAT_MINI_LED_INDICATOR_BORDER_ENABLED", False)

    display = utils.Display()
    source = utils.Image.new("RGB", (display.width, display.height), (255, 0, 0))

    display.image(source)

    assert display.capture().getpixel((10, 0)) == (255, 0, 0)
    assert display.capture().getpixel((10, display.height - 6)) == (255, 0, 0)
    assert display.capture().getpixel((10, display.height - 1)) == (0, 0, 0)
