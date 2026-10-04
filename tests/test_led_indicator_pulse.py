"""LED_INDICATOR_PULSE: the notification border breathes instead of staying static."""

import threading
import time

import pytest

import utils


def _display(monkeypatch, *, pulse):
    monkeypatch.setattr(utils, "LED_INDICATOR_BORDER_ENABLED", True)
    monkeypatch.setattr(utils, "LED_INDICATOR_PULSE", pulse)
    display = utils.Display()
    display._buffer = utils.Image.new("RGB", (display.width, display.height), "black")
    return display


def test_pulse_level_starts_full_dips_and_returns():
    period = utils._INDICATOR_PULSE_PERIOD_SECONDS
    assert utils.indicator_pulse_level(0.0) == pytest.approx(1.0)
    assert utils.indicator_pulse_level(period / 2) == pytest.approx(utils._INDICATOR_PULSE_MIN_LEVEL)
    assert utils.indicator_pulse_level(period) == pytest.approx(1.0)
    levels = [utils.indicator_pulse_level(period * i / 50) for i in range(51)]
    assert min(levels) >= utils._INDICATOR_PULSE_MIN_LEVEL - 1e-9
    assert max(levels) <= 1.0 + 1e-9


def test_pulse_is_off_by_default_and_border_stays_static(monkeypatch):
    monkeypatch.setattr(utils, "LED_INDICATOR_BORDER_ENABLED", True)
    display = utils.Display()
    assert display._indicator_pulse_enabled is False
    display._buffer = utils.Image.new("RGB", (display.width, display.height), "black")
    display.set_led(r=0.0, g=0.0, b=utils.LED_INDICATOR_LEVEL)

    # Half a period in, a pulsing border would be at its dimmest.
    display._indicator_pulse_started_at -= utils._INDICATOR_PULSE_PERIOD_SECONDS / 2
    assert display._indicator_buffer().getpixel((0, 0)) == (0, 0, 255)
    assert display._indicator_pulse_thread is None
    display.close()


def test_pulsing_border_dims_on_the_panel_but_not_in_screenshots(monkeypatch):
    display = _display(monkeypatch, pulse=True)
    display.set_led(r=0.0, g=0.0, b=utils.LED_INDICATOR_LEVEL)
    try:
        display._indicator_pulse_started_at = (
            time.monotonic() - utils._INDICATOR_PULSE_PERIOD_SECONDS / 2
        )
        r, g, b = display._indicator_buffer().getpixel((0, 0))
        assert (r, g) == (0, 0)
        assert 0 < b < 255
        assert b == pytest.approx(255 * utils._INDICATOR_PULSE_MIN_LEVEL, abs=6)

        screenshot = utils.Image.new("RGB", (display.width, display.height), "black")
        assert display.apply_indicator_border(screenshot).getpixel((0, 0)) == (0, 0, 255)
    finally:
        display.close()


def test_pulse_thread_resends_idle_frames_and_stops_when_led_clears(monkeypatch):
    monkeypatch.setattr(utils, "_INDICATOR_PULSE_FRAME_SECONDS", 0.01)
    display = _display(monkeypatch, pulse=True)
    frames = []
    sent = threading.Event()

    def _writer(image):
        frames.append(image.getpixel((0, 0)))
        if len(frames) >= 5:
            sent.set()

    display._frame_writer = _writer
    try:
        display.set_led(r=utils.LED_INDICATOR_LEVEL, g=0.0, b=0.0)
        thread = display._indicator_pulse_thread
        assert thread is not None and thread.name == "led-border-pulse"
        assert sent.wait(2.0)
        assert all(pixel[0] > 0 and pixel[1:] == (0, 0) for pixel in frames)

        display.set_led(r=0.0, g=0.0, b=0.0)
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert display._indicator_pulse_thread is None
    finally:
        display.close()


def test_pulse_needs_the_border(monkeypatch):
    monkeypatch.setattr(utils, "LED_INDICATOR_BORDER_ENABLED", False)
    monkeypatch.setattr(utils, "LED_INDICATOR_PULSE", True)
    display = utils.Display()
    display.set_led(r=utils.LED_INDICATOR_LEVEL, g=0.0, b=0.0)
    assert display._indicator_pulse_thread is None
    display.close()


def test_close_stops_the_pulse_thread(monkeypatch):
    display = _display(monkeypatch, pulse=True)
    display.set_led(r=utils.LED_INDICATOR_LEVEL, g=0.0, b=0.0)
    thread = display._indicator_pulse_thread
    assert thread is not None
    display.close()
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_pulse_setting_is_documented_for_clients():
    import deployment_config

    names = {setting.name for setting in deployment_config.SETTINGS}
    assert "LED_INDICATOR_PULSE" in names
