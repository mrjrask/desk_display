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
        def __init__(self, interval, callback, kwargs):
            captured["interval"] = interval
            captured["callback"] = callback
            captured["kwargs"] = kwargs
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
    assert captured["callback"] is utils._wiggle_mouse_cursor
    assert captured["kwargs"] == {"pygame_module": fake_pygame, "distance": 1}
    assert captured["started"] is True


def test_schedule_mouse_cursor_wiggle_noops_for_nonpositive_delay(monkeypatch):
    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("Timer should not be constructed")

    monkeypatch.setattr(utils.threading, "Timer", _raise_if_called)

    assert utils._schedule_mouse_cursor_wiggle(_FakePygame(), delay_seconds=0) is None
