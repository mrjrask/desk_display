from types import SimpleNamespace

import utils


class FakeTime:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def time(self):
        return self.now

    def sleep(self, duration):
        self.now += max(0.0, float(duration))


class FakeEventQueue:
    def __init__(self, batches):
        self.batches = list(batches)

    def get(self, _event_types):
        if self.batches:
            return self.batches.pop(0)
        return []


class FakeMouse:
    def __init__(self):
        self.visibility = []

    def set_visible(self, visible):
        self.visibility.append(visible)


class FakePygame:
    FINGERDOWN = 1
    FINGERMOTION = 2
    FINGERUP = 3
    MOUSEBUTTONDOWN = 4
    MOUSEMOTION = 5
    MOUSEBUTTONUP = 6

    def __init__(self, batches):
        self.event = FakeEventQueue(batches)
        self.mouse = FakeMouse()


class FakeDisplay:
    width = 100
    height = 50

    def __init__(self):
        self._kernel_display = None
        self.skip = False

    def skip_requested(self):
        return self.skip


def _event(event_type, **attrs):
    return SimpleNamespace(type=event_type, **attrs)


def test_scroll_vertical_content_touch_drag_changes_offsets_and_pauses_auto_scroll(monkeypatch):
    fake_time = FakeTime()
    display = FakeDisplay()
    frames = []
    fake_pygame = FakePygame(
        [
            [
                _event(FakePygame.FINGERDOWN, x=0.5, y=0.8),
                _event(FakePygame.FINGERMOTION, x=0.5, y=0.2),
                _event(FakePygame.FINGERUP, x=0.5, y=0.2),
            ]
        ]
    )

    monkeypatch.setattr(utils, "_PYGAME_MODULE", fake_pygame)
    monkeypatch.setattr(utils, "_PYGAME_ERROR", None)
    monkeypatch.setattr(utils.time, "monotonic", fake_time.monotonic)
    monkeypatch.setattr(utils.time, "time", fake_time.time)
    monkeypatch.setattr(utils.time, "sleep", fake_time.sleep)

    utils.scroll_vertical_content(
        display=display,
        content_height=100,
        viewport_width=100,
        viewport_height=50,
        render_at_offset=lambda offset: frames.append((fake_time.monotonic(), offset)),
        base_step=10,
        pause_start=0,
        pause_end=0,
        page_jump_mode=False,
        min_frame_time=0.1,
    )

    assert frames[0] == (0.0, 0)
    assert frames[1] == (0.0, 30)
    assert all(offset <= 30 for timestamp, offset in frames if timestamp < 2.0)
    resumed_frames = [(timestamp, offset) for timestamp, offset in frames if timestamp >= 2.0]
    assert resumed_frames[0][1] > 30
    assert frames[-1][1] == 50
    assert fake_pygame.mouse.visibility
    assert all(visible is False for visible in fake_pygame.mouse.visibility)


def test_scroll_vertical_content_mouse_drag_clamps_offsets(monkeypatch):
    fake_time = FakeTime()
    display = FakeDisplay()
    frames = []
    fake_pygame = FakePygame(
        [
            [
                _event(FakePygame.MOUSEBUTTONDOWN, button=1, pos=(10, 45)),
                _event(FakePygame.MOUSEMOTION, buttons=(1, 0, 0), pos=(10, -50)),
                _event(FakePygame.MOUSEBUTTONUP, button=1, pos=(10, -50)),
            ]
        ]
    )

    monkeypatch.setattr(utils, "_PYGAME_MODULE", fake_pygame)
    monkeypatch.setattr(utils, "_PYGAME_ERROR", None)
    monkeypatch.setattr(utils.time, "monotonic", fake_time.monotonic)
    monkeypatch.setattr(utils.time, "time", fake_time.time)
    monkeypatch.setattr(utils.time, "sleep", fake_time.sleep)

    utils.scroll_vertical_content(
        display=display,
        content_height=100,
        viewport_width=100,
        viewport_height=50,
        render_at_offset=lambda offset: frames.append(offset),
        base_step=10,
        pause_start=0,
        pause_end=0,
        page_jump_mode=False,
        min_frame_time=0.1,
    )

    assert 45 in frames
    assert frames[-1] == 50
    assert fake_pygame.mouse.visibility
    assert all(visible is False for visible in fake_pygame.mouse.visibility)


def test_scroll_vertical_content_skip_still_exits_promptly(monkeypatch):
    display = FakeDisplay()
    frames = []
    fake_pygame = FakePygame([[_event(FakePygame.FINGERDOWN, x=0.5, y=0.8)]])

    def skip_requested():
        display.skip = True
        return True

    display.skip_requested = skip_requested
    monkeypatch.setattr(utils, "_PYGAME_MODULE", fake_pygame)

    utils.scroll_vertical_content(
        display=display,
        content_height=100,
        viewport_width=100,
        viewport_height=50,
        render_at_offset=frames.append,
        base_step=10,
        pause_start=0,
        pause_end=0,
        page_jump_mode=False,
        min_frame_time=0.1,
    )

    assert frames == [0]
