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


def _sdl_display(monkeypatch):
    """A headless Display marked SDL-backed: frames must not come from a thread."""

    display = _display(monkeypatch, pulse=True)
    display._uses_kernel_output = True
    frames = []
    display._frame_writer = lambda image: frames.append(image.getpixel((0, 0)))
    return display, frames


def test_sdl_output_pulses_from_ticks_not_a_thread(monkeypatch):
    display, frames = _sdl_display(monkeypatch)
    try:
        display.set_led(r=utils.LED_INDICATOR_LEVEL, g=0.0, b=0.0)
        assert display._indicator_pulse_thread is None
        assert len(frames) == 1  # set_led presents the lit border once
        assert frames[0][0] == 255  # ...at full brightness, whatever the clock says

        # Half a period later the border is at its dimmest, and the owner
        # loop's tick presents that frame.
        display._indicator_pulse_started_at = (
            time.monotonic() - utils._INDICATOR_PULSE_PERIOD_SECONDS / 2
        )
        display._last_present_at = time.monotonic() - 1.0
        assert display.tick_indicator_pulse() is True
        assert frames[-1][0] == pytest.approx(255 * utils._INDICATOR_PULSE_MIN_LEVEL, abs=6)
    finally:
        display.close()


def test_tick_is_idle_when_a_frame_just_went_out_or_nothing_changed(monkeypatch):
    display, frames = _sdl_display(monkeypatch)
    try:
        display.set_led(r=utils.LED_INDICATOR_LEVEL, g=0.0, b=0.0)
        display._indicator_pulse_started_at = (
            time.monotonic() - utils._INDICATOR_PULSE_PERIOD_SECONDS / 4
        )
        before = len(frames)

        display._last_present_at = time.monotonic()  # an animation frame, say
        assert display.tick_indicator_pulse() is False

        display._last_present_at = time.monotonic() - 1.0
        assert display.tick_indicator_pulse() is True
        display._last_present_at = time.monotonic() - 1.0
        # Same brightness step as the frame just sent: nothing to repaint.
        monkeypatch.setattr(utils, "indicator_pulse_level", lambda _elapsed: 0.5)
        display._last_pulse_step = round(0.5 * utils._INDICATOR_PULSE_STEPS)
        assert display.tick_indicator_pulse() is False
        assert len(frames) == before + 1
    finally:
        display.close()


def test_tick_does_nothing_when_pulse_is_off_or_border_dark(monkeypatch):
    off = _display(monkeypatch, pulse=False)
    off.set_led(r=utils.LED_INDICATOR_LEVEL, g=0.0, b=0.0)
    assert off.tick_indicator_pulse() is False
    off.close()

    dark, frames = _sdl_display(monkeypatch)
    dark._last_present_at = 0.0
    assert dark.tick_indicator_pulse() is False
    assert frames == []
    dark.close()


def test_pulse_restarts_at_full_brightness_when_the_border_relights(monkeypatch):
    display, _frames = _sdl_display(monkeypatch)
    try:
        display.set_led(r=utils.LED_INDICATOR_LEVEL, g=0.0, b=0.0)
        display._indicator_pulse_started_at -= 10.0
        display.set_led(r=0.0, g=0.0, b=0.0)
        display.set_led(r=0.0, g=utils.LED_INDICATOR_LEVEL, b=0.0)
        assert display._indicator_pulse_level() == pytest.approx(1.0, abs=0.01)
    finally:
        display.close()


def test_main_wait_loop_ticks_the_pulse(monkeypatch):
    import main

    class _Display:
        ticks = 0

        def tick_indicator_pulse(self):
            _Display.ticks += 1

    main.display = _Display()
    main._shutdown_event.clear()
    main._manual_skip_event.clear()
    main._skip_request_pending = False
    monkeypatch.setattr(main, "BUTTON_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(main, "_check_control_buttons", lambda *a, **k: False)

    assert main._wait_with_button_checks(0.1) is False
    assert _Display.ticks >= 3


def test_hardware_presenter_forwards_the_tick():
    from display.hardware_presenter import HardwarePresenter

    class _Display:
        def tick_indicator_pulse(self):
            return True

    assert HardwarePresenter(_Display()).tick_indicator_pulse() is True
    assert HardwarePresenter(object()).tick_indicator_pulse() is False
