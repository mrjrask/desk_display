import time

import pytest

import utils
from utils import compute_adaptive_scroll_params, scroll_vertical_content




@pytest.fixture(autouse=True)
def _fast_scroll_sleep(monkeypatch):
    now = [time.monotonic()]

    def fake_monotonic():
        return now[0]

    def fake_sleep(duration):
        now[0] += max(0.0, duration)

    monkeypatch.setattr(utils.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(utils.time, "sleep", fake_sleep)
    yield

class DummyDisplay:
    def __init__(self):
        self.frames = []

    def wait_for_skip(self, _duration: float) -> bool:
        return False

    def skip_requested(self) -> bool:
        return False


def test_compute_adaptive_scroll_params_increases_step_for_high_resolution():
    params = compute_adaptive_scroll_params(
        content_height=2000,
        viewport_height=480,
        viewport_width=1920,
        base_step=1,
    )
    assert params.step > 1


def test_compute_adaptive_scroll_params_enables_page_jump_for_very_tall_content():
    params = compute_adaptive_scroll_params(
        content_height=3200,
        viewport_height=320,
        viewport_width=320,
        base_step=1,
    )
    assert params.use_page_jump is True
    assert params.target_frame_time > 0.016


def test_scroll_vertical_content_uses_adaptive_stride_when_page_jump_enabled():
    display = DummyDisplay()

    scroll_vertical_content(
        display=display,
        content_height=2440,
        viewport_width=1080,
        viewport_height=240,
        render_at_offset=lambda offset: display.frames.append(offset),
        base_step=1,
        pause_start=0,
        pause_end=0,
    )

    assert display.frames[0] == 0
    assert display.frames[-1] == 2200
    assert len(display.frames) <= 24
    assert display.frames[1] - display.frames[0] >= 96


def test_scroll_vertical_content_keeps_dense_readable_frames_for_long_content():
    display = DummyDisplay()

    scroll_vertical_content(
        display=display,
        content_height=1200,
        viewport_width=1080,
        viewport_height=240,
        render_at_offset=lambda offset: display.frames.append(offset),
        base_step=1,
        pause_start=0,
        pause_end=0,
    )

    assert display.frames[0] == 0
    assert display.frames[-1] == 960
    assert len(display.frames) > 80


class ImmediateWaitCaptureDisplay:
    def __init__(self, frame_limit=1):
        self.frames = []
        self.wait_calls = []
        self.frame_limit = frame_limit

    def wait_for_skip(self, duration: float) -> bool:
        self.wait_calls.append(duration)
        return False

    def skip_requested(self) -> bool:
        return len(self.frames) >= self.frame_limit


def test_scroll_vertical_content_preserves_immediate_wait_for_skip_capture_semantics():
    display = ImmediateWaitCaptureDisplay(frame_limit=3)

    scroll_vertical_content(
        display=display,
        content_height=100,
        viewport_width=100,
        viewport_height=50,
        render_at_offset=display.frames.append,
        base_step=10,
        pause_start=0,
        pause_end=0,
        page_jump_mode=False,
        min_frame_time=0.1,
    )

    assert len(display.frames) == 3
    assert display.wait_calls


def test_scroll_vertical_content_does_not_fallback_sleep_after_wait_for_skip_returns_immediately(monkeypatch):
    display = ImmediateWaitCaptureDisplay(frame_limit=4)
    sleep_calls = []

    monkeypatch.setattr(utils.time, "sleep", lambda duration: sleep_calls.append(duration))

    scroll_vertical_content(
        display=display,
        content_height=80,
        viewport_width=100,
        viewport_height=50,
        render_at_offset=display.frames.append,
        base_step=10,
        pause_start=0,
        pause_end=0,
        page_jump_mode=False,
        min_frame_time=0.05,
    )

    assert display.frames == [0, 10, 20, 30]
    assert display.wait_calls
    assert sleep_calls == []

def test_compute_adaptive_scroll_params_honors_max_step(monkeypatch):
    monkeypatch.setattr("utils.get_global_scroll_settings", lambda: {"speed": 3.0, "smoothness": 1.0})

    params = compute_adaptive_scroll_params(
        content_height=1200,
        viewport_height=240,
        viewport_width=320,
        base_step=1,
        max_step=1,
    )

    assert params.step == 1


def test_compute_adaptive_scroll_params_applies_min_frame_floor_after_overflow_scaling(monkeypatch):
    monkeypatch.setattr("utils.get_global_scroll_settings", lambda: {"speed": 1.0, "smoothness": 3.0})

    params = compute_adaptive_scroll_params(
        content_height=720,
        viewport_height=240,
        viewport_width=320,
        base_step=1,
        min_frame_time=0.100,
        min_frame_time_floor=0.100,
    )

    assert params.target_frame_time == 0.100


def test_compute_adaptive_scroll_params_preserves_target_above_min_frame_floor(
    monkeypatch,
):
    monkeypatch.setattr(
        "utils.get_global_scroll_settings",
        lambda: {"speed": 1.0, "smoothness": 3.0},
    )

    params = compute_adaptive_scroll_params(
        content_height=720,
        viewport_height=240,
        viewport_width=320,
        base_step=1,
        min_frame_time=0.100,
        min_frame_time_floor=0.050,
    )

    assert params.target_frame_time == pytest.approx(
        0.100 * (1.0 + 2.0 * 0.6) / 3.0
    )


def test_scroll_vertical_content_caps_page_jump_stride_with_max_step():
    display = DummyDisplay()

    scroll_vertical_content(
        display=display,
        content_height=2440,
        viewport_width=1080,
        viewport_height=240,
        render_at_offset=lambda offset: display.frames.append(offset),
        base_step=1,
        pause_start=0,
        pause_end=0,
        max_step=1,
    )

    assert display.frames[:3] == [0, 1, 2]
    assert display.frames[-1] == 2200
