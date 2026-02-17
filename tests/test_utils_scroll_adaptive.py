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
        content_height=2500,
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
        content_height=2200,
        viewport_width=1080,
        viewport_height=240,
        render_at_offset=lambda offset: display.frames.append(offset),
        base_step=1,
        pause_start=0,
        pause_end=0,
    )

    assert display.frames[0] == 0
    assert display.frames[-1] == 1960
    assert len(display.frames) <= 12
    assert display.frames[1] - display.frames[0] >= 180
