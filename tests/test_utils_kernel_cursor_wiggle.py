"""Tests for kernel cursor wiggle behavior."""

import utils


class _FakeMouse:
    def __init__(self):
        self.pos = (100, 50)
        self.moves = []

    def get_pos(self):
        return self.pos

    def set_pos(self, pos):
        self.moves.append(pos)
        self.pos = pos


class _FakeEvent:
    def __init__(self):
        self.pump_count = 0

    def pump(self):
        self.pump_count += 1


class _FakePygame:
    def __init__(self):
        self.mouse = _FakeMouse()
        self.event = _FakeEvent()
        self.display = _FakeDisplay()


class _FakeSurface:
    def __init__(self, size=(320, 240)):
        self._size = size

    def get_size(self):
        return self._size


class _FakeDisplay:
    def __init__(self):
        self.surface = _FakeSurface()

    def get_surface(self):
        return self.surface


def test_wiggle_mouse_cursor_moves_and_restores_pointer():
    fake_pygame = _FakePygame()

    utils._wiggle_mouse_cursor(fake_pygame, distance=3)

    assert fake_pygame.mouse.moves == [(103, 50), (100, 50)]
    assert fake_pygame.event.pump_count == 2


def test_wiggle_mouse_cursor_noops_for_nonpositive_distance():
    fake_pygame = _FakePygame()

    utils._wiggle_mouse_cursor(fake_pygame, distance=0)

    assert fake_pygame.mouse.moves == []
    assert fake_pygame.event.pump_count == 0


def test_schedule_mouse_cursor_wiggle_starts_daemon_timer(monkeypatch):
    fake_pygame = _FakePygame()
    captured = {}

    class _FakeTimer:
        def __init__(self, interval, callback):
            captured["interval"] = interval
            captured["callback"] = callback
            self.daemon = False
            self.started = False

        def start(self):
            self.started = True
            captured["started"] = True

    monkeypatch.setattr(utils.threading, "Timer", _FakeTimer)

    timer = utils._schedule_mouse_cursor_wiggle(fake_pygame, delay_seconds=30)

    assert isinstance(timer, _FakeTimer)
    assert timer.daemon is True
    assert captured["interval"] == 30
    captured["callback"]()
    assert fake_pygame.mouse.moves == [(101, 50), (100, 50)]
    assert captured["started"] is True


def test_schedule_mouse_cursor_wiggle_noops_for_nonpositive_delay(monkeypatch):
    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("Timer should not be constructed")

    monkeypatch.setattr(utils.threading, "Timer", _raise_if_called)

    assert utils._schedule_mouse_cursor_wiggle(_FakePygame(), delay_seconds=0) is None


def test_park_mouse_cursor_moves_pointer_to_bottom_right():
    fake_pygame = _FakePygame()

    utils._park_mouse_cursor(fake_pygame)

    assert fake_pygame.mouse.moves[-1] == (319, 239)
    assert fake_pygame.event.pump_count == 1


def test_schedule_mouse_cursor_wiggle_repeat_rearms_timer(monkeypatch):
    fake_pygame = _FakePygame()
    captured_callbacks = []

    class _FakeTimer:
        def __init__(self, _interval, callback):
            captured_callbacks.append(callback)
            self.daemon = False

        def start(self):
            return None

    monkeypatch.setattr(utils.threading, "Timer", _FakeTimer)

    timer = utils._schedule_mouse_cursor_wiggle(fake_pygame, delay_seconds=30, repeat=True)
    assert isinstance(timer, _FakeTimer)

    captured_callbacks[0]()
    assert fake_pygame.mouse.moves == [(101, 50), (100, 50)]
    assert len(captured_callbacks) == 2
