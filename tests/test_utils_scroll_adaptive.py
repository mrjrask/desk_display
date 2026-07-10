from utils import compute_adaptive_scroll_params, scroll_vertical_content


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
        pause_start=30,
        pause_end=30,
        page_jump_mode=False,
        min_frame_time=0.1,
    )

    assert len(display.frames) == 3
    assert display.wait_calls
    assert len(display.wait_calls) <= 3
    assert all(duration <= 0.05 for duration in display.wait_calls)


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
