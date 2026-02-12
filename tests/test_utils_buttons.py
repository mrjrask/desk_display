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


def test_kernel_output_uses_25px_bottom_safe_buffer_without_indicator_border(monkeypatch):
    monkeypatch.setattr(utils, "is_hyperpixel_next_layout", lambda w, h: False)
    monkeypatch.setattr(utils, "DISPLAY_HAT_MINI_LED_INDICATOR_BORDER_ENABLED", False)

    display = utils.Display()
    display._uses_kernel_output = True
    source = utils.Image.new("RGB", (display.width, display.height), (255, 0, 0))

    display.image(source)

    assert display.capture().getpixel((10, display.height - 26)) == (255, 0, 0)
    assert display.capture().getpixel((10, display.height - 25)) == (0, 0, 0)
    assert display.capture().getpixel((10, display.height - 1)) == (0, 0, 0)


def test_kernel_output_keeps_indicator_border_exception(monkeypatch):
    monkeypatch.setattr(utils, "is_hyperpixel_next_layout", lambda w, h: True)

    display = utils.Display()
    display._uses_kernel_output = True
    source = utils.Image.new("RGB", (display.width, display.height), (255, 0, 0))

    display.image(source)

    assert display.capture().getpixel((10, display.height - 6)) == (255, 0, 0)
    assert display.capture().getpixel((10, display.height - 5)) == (0, 0, 0)


def test_release_display_hat_mini_prefers_close():
    calls = []

    class _FakeDisplay:
        def close(self):
            calls.append("close")

        def cleanup(self):
            calls.append("cleanup")

    display = utils.Display()
    display._release_display_hat_mini(_FakeDisplay())

    assert calls == ["close"]


def test_reinitialize_display_releases_previous_driver(monkeypatch):
    display = utils.Display()
    display._display_reinit_seconds = 1
    display._last_display_reinit = 0

    released = []

    old_display = object()
    display._display = old_display

    new_display = object()
    monkeypatch.setattr(display, "_create_display_hat_mini", lambda _: new_display)
    monkeypatch.setattr(display, "_release_display_hat_mini", lambda driver: released.append(driver))
    monkeypatch.setattr(utils.time, "monotonic", lambda: 10)

    display._maybe_reinitialize_display_hat_mini()

    assert display._display is new_display
    assert released == [old_display]


def test_reinitialize_display_retries_after_failure(monkeypatch):
    display = utils.Display()
    display._display_reinit_seconds = 1
    display._last_display_reinit = 0
    display._display = object()

    released = []
    monkeypatch.setattr(display, "_release_display_hat_mini", lambda driver: released.append(driver))
    monkeypatch.setattr(
        display,
        "_create_display_hat_mini",
        lambda _: (_ for _ in ()).throw(RuntimeError("busy")),
    )
    monkeypatch.setattr(utils.time, "monotonic", lambda: 10)

    display._maybe_reinitialize_display_hat_mini()

    assert display._display is None
    assert len(released) == 1
    assert display._next_display_reinit_retry == 10 + display._DISPLAY_REINIT_RETRY_SECONDS


def test_reinitialize_display_skips_attempt_during_retry_window(monkeypatch):
    display = utils.Display()
    display._display_reinit_seconds = 1
    display._last_display_reinit = 0
    display._next_display_reinit_retry = 100
    display._display = object()

    create_calls = []
    monkeypatch.setattr(display, "_create_display_hat_mini", lambda _: create_calls.append("called"))
    monkeypatch.setattr(utils.time, "monotonic", lambda: 50)

    display._maybe_reinitialize_display_hat_mini()

    assert create_calls == []
