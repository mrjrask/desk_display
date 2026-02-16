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
