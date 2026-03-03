"""Tests for Display HAT Mini button handling utilities."""

import subprocess
import threading
import time
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


def test_is_button_pressed_falls_back_to_named_button_read():
    calls = []

    def _read_button(value):
        calls.append(value)
        if isinstance(value, int):
            raise RuntimeError("pin lookup unavailable")
        return 0

    display = utils.Display()
    display._display = SimpleNamespace(read_button=_read_button)  # type: ignore[attr-defined]
    display._button_pins["A"] = None

    assert display.is_button_pressed("A") is True
    assert calls[-1] == "A"


def test_create_display_hat_mini_reads_button_pins_from_class(monkeypatch):
    class _FakeDisplay:
        BUTTON_A = 17
        BUTTON_B = 18
        BUTTON_X = 19
        BUTTON_Y = 20

        def __init__(self, _buffer):
            pass

    display = utils.Display()
    monkeypatch.setattr(utils, "DisplayHATMini", _FakeDisplay)

    created = display._create_display_hat_mini(display._buffer)

    assert isinstance(created, _FakeDisplay)
    assert display._button_pins == {"A": 17, "B": 18, "X": 19, "Y": 20}


def test_hardware_button_callback_accepts_button_name():
    events = []
    display = utils.Display()
    display._display = SimpleNamespace(read_button=lambda _pin: True)  # type: ignore[attr-defined]
    display.set_button_callback(events.append)

    display._handle_hw_button_event("x")

    assert events == ["X"]


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


def test_indicator_buffer_returns_fresh_frame_when_border_enabled(monkeypatch):
    monkeypatch.setattr(utils, "is_hyperpixel_next_layout", lambda w, h: True)

    display = utils.Display()
    display._buffer = utils.Image.new("RGB", (display.width, display.height), "black")
    display.set_led(r=utils.LED_INDICATOR_LEVEL, g=0.0, b=0.0)

    frame_a = display._indicator_buffer()
    frame_b = display._indicator_buffer()

    assert frame_a is not frame_b

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


def test_display_hat_mini_indicator_border_renders_led_color_when_rotated(monkeypatch):
    monkeypatch.setattr(utils, "WIDTH", 240)
    monkeypatch.setattr(utils, "HEIGHT", 320)
    monkeypatch.setattr(utils, "DISPLAY_HAT_MINI_LED_INDICATOR_BORDER_ENABLED", True)

    display = utils.Display()
    display._buffer = utils.Image.new("RGB", (display.width, display.height), "black")

    display.set_led(r=0.0, g=0.0, b=utils.LED_INDICATOR_LEVEL)

    pixel = display._indicator_buffer().getpixel((0, 0))
    assert pixel == (0, 0, 255)


def test_hyperpixel_indicator_border_renders_led_color_for_hyperpixel_size(monkeypatch):
    monkeypatch.setattr(utils, "WIDTH", 800)
    monkeypatch.setattr(utils, "HEIGHT", 480)
    monkeypatch.setattr(utils, "DISPLAY_HAT_MINI_LED_INDICATOR_BORDER_ENABLED", True)
    monkeypatch.setattr(utils, "HYPERPIXEL_LED_INDICATOR_BORDER_ENABLED", True)

    display = utils.Display()
    display._buffer = utils.Image.new("RGB", (display.width, display.height), "black")

    display.set_led(r=0.0, g=utils.LED_INDICATOR_LEVEL, b=0.0)

    pixel = display._indicator_buffer().getpixel((0, 0))
    assert pixel == (0, 255, 0)


def test_display_hat_mini_led_respects_config_disable_when_border_is_disabled(monkeypatch):
    monkeypatch.setattr(utils, "DISPLAY_HAT_MINI_LED_ENABLED", False)
    monkeypatch.setattr(utils, "DISPLAY_HAT_MINI_LED_INDICATOR_BORDER_ENABLED", False)

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


def test_display_hat_mini_led_is_still_updated_when_indicator_border_is_enabled(monkeypatch):
    monkeypatch.setattr(utils, "WIDTH", 320)
    monkeypatch.setattr(utils, "HEIGHT", 240)
    monkeypatch.setattr(utils, "DISPLAY_HAT_MINI_LED_ENABLED", False)
    monkeypatch.setattr(utils, "DISPLAY_HAT_MINI_LED_INDICATOR_BORDER_ENABLED", True)

    class _FakeHardwareDisplay:
        def __init__(self):
            self.called = False
            self.color = None

        def set_led(self, **kwargs):
            self.called = True
            self.color = kwargs

    display = utils.Display()
    fake_display = _FakeHardwareDisplay()
    display._display = fake_display

    display.set_led(r=0.1, g=0.2, b=0.3)

    assert fake_display.called is True
    assert fake_display.color == {"r": 0.1, "g": 0.2, "b": 0.3}


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


def test_release_display_hat_mini_runs_all_cleanup_hooks():
    calls = []

    class _FakeDisplay:
        def close(self):
            calls.append("close")

        def cleanup(self):
            calls.append("cleanup")

    display = utils.Display()
    display._release_display_hat_mini(_FakeDisplay())

    assert calls == ["cleanup", "close"]


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


def test_check_github_updates_clears_status_when_not_git_repo(monkeypatch):
    monkeypatch.setattr(
        utils,
        "_UPDATE_STATUS",
        utils._UpdateStatus(github=True, apt=False),
    )

    def _raise_not_git(*_args, **_kwargs):
        raise subprocess.CalledProcessError(returncode=128, cmd="git")

    monkeypatch.setattr(utils.subprocess, "check_call", _raise_not_git)

    assert utils.check_github_updates() is False
    assert utils.get_update_status().github is False


def test_check_github_updates_clears_status_when_fetch_fails(monkeypatch):
    monkeypatch.setattr(
        utils,
        "_UPDATE_STATUS",
        utils._UpdateStatus(github=True, apt=False),
    )

    def _fake_check_call(args, **_kwargs):
        if args[:2] == ["git", "fetch"]:
            raise subprocess.CalledProcessError(returncode=1, cmd="git fetch")
        return 0

    def _fake_check_output(args, **_kwargs):
        if args == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return b"main\n"
        if args == ["git", "rev-parse", "HEAD"]:
            return b"localsha\n"
        if args == ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]:
            return b"origin/main\n"
        raise AssertionError(f"Unexpected git command: {args}")

    monkeypatch.setattr(utils.subprocess, "check_call", _fake_check_call)
    monkeypatch.setattr(utils.subprocess, "check_output", _fake_check_output)

    assert utils.check_github_updates() is False
    assert utils.get_update_status().github is False


def test_led_pattern_is_yellow_only_for_apt_updates():
    pattern, interval = utils._led_pattern(utils._UpdateStatus(github=False, apt=True))

    assert pattern == ((utils.LED_INDICATOR_LEVEL, utils.LED_INDICATOR_LEVEL, 0.0),)
    assert interval == 0.8


def test_led_pattern_alternates_only_when_apt_and_github_updates():
    pattern, interval = utils._led_pattern(utils._UpdateStatus(github=True, apt=True))

    assert pattern == (
        (0.0, 0.0, utils.LED_INDICATOR_LEVEL),
        (utils.LED_INDICATOR_LEVEL, utils.LED_INDICATOR_LEVEL, 0.0),
    )
    assert interval == 0.6


def test_refresh_led_indicator_uses_static_color_for_indicator_border(monkeypatch):
    class _FakeAnimator:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    class _FakeDisplay:
        _hyperpixel_indicator_border = True
        _display_hat_mini_indicator_border = False

        def __init__(self):
            self.calls = []

        def set_led(self, *, r, g, b):
            self.calls.append((r, g, b))

    animator = _FakeAnimator()
    fake_display = _FakeDisplay()

    monkeypatch.setattr(utils, "_UPDATE_STATUS", utils._UpdateStatus(github=True, apt=True))
    monkeypatch.setattr(utils, "_LED_INDICATOR_ANIMATOR", animator)

    utils._refresh_led_indicator(fake_display)

    assert animator.stopped is True
    assert utils._LED_INDICATOR_ANIMATOR is None
    assert fake_display.calls == [(0.0, 0.0, utils.LED_INDICATOR_LEVEL)]


def test_reinitialize_display_retries_after_failure(monkeypatch):
    display = utils.Display()
    display._display_reinit_seconds = 1
    display._last_display_reinit = 0
    original_display = object()
    display._display = original_display

    released = []
    monkeypatch.setattr(display, "_release_display_hat_mini", lambda driver: released.append(driver))
    monkeypatch.setattr(
        display,
        "_create_display_hat_mini",
        lambda _: (_ for _ in ()).throw(RuntimeError("busy")),
    )
    monkeypatch.setattr(utils.time, "monotonic", lambda: 10)

    display._maybe_reinitialize_display_hat_mini()

    assert display._display is original_display
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


def test_display_lock_serializes_refresh_and_button_reads():
    display = utils.Display()

    class _FakeHardwareDisplay:
        def __init__(self):
            self.buffer = None
            self._busy = threading.Lock()

        def _enter(self):
            if not self._busy.acquire(blocking=False):
                raise RuntimeError("concurrent hardware access")

        def _exit(self):
            self._busy.release()

        def display(self):
            self._enter()
            try:
                time.sleep(0.002)
            finally:
                self._exit()

        def read_button(self, _pin):
            self._enter()
            try:
                time.sleep(0.001)
                return 0
            finally:
                self._exit()

    display._display = _FakeHardwareDisplay()
    display._button_pins["A"] = 5

    errors = []

    def _poll_buttons():
        for _ in range(30):
            try:
                display.is_button_pressed("A")
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(exc)

    reader = threading.Thread(target=_poll_buttons)
    reader.start()

    for _ in range(30):
        display._update_display()

    reader.join(timeout=1.0)

    assert reader.is_alive() is False
    assert errors == []


def test_set_led_clamps_channels_to_normalized_range():
    display = utils.Display()

    display.set_led(r=2.5, g=-1.0, b=0.5)

    assert display._led_color == (1.0, 0.0, 0.5)


def test_indicator_channel_to_pixel_clamps_out_of_range_values():
    assert utils.Display._indicator_channel_to_pixel(5.0) == 255
    assert utils.Display._indicator_channel_to_pixel(-0.2) == 0


def test_get_led_indicator_level_from_env(monkeypatch):
    monkeypatch.setenv("DISPLAY_HAT_MINI_LED_LEVEL", "0.05")

    assert utils._get_led_indicator_level() == 0.05


def test_get_led_indicator_level_invalid_env_uses_default(monkeypatch):
    monkeypatch.setenv("DISPLAY_HAT_MINI_LED_LEVEL", "not-a-number")

    assert utils._get_led_indicator_level() == 0.08
