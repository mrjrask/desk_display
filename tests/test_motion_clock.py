"""MotionClock paces package motion to the panel, as v0.1's scroll loop did."""

import pytest

from playback.motion_clock import MotionClock


def test_follows_the_wall_clock_while_nothing_is_drawn():
    clock = MotionClock(0.0, 10.0, max_lag=5.0)
    assert clock.tick(10.75, 0.02) == pytest.approx(0.75)  # a pause runs in real time
    assert clock.lag == 0.0


def test_advances_one_frame_per_drawn_frame_on_a_slow_panel():
    clock = MotionClock(0.0, 0.0, max_lag=5.0)
    clock.tick(0.0, 0.02)
    now = 0.0
    times = []
    for _ in range(5):
        clock.drew(True)
        now += 0.05  # each push takes two and a half frames
        times.append(clock.tick(now, 0.02))
    assert times == pytest.approx([0.02, 0.04, 0.06, 0.08, 0.10])
    assert clock.lag == pytest.approx(0.15)


def test_lag_is_bounded():
    clock = MotionClock(0.0, 0.0, max_lag=0.1)
    clock.drew(True)
    clock.tick(1.0, 0.02)
    assert clock.lag == pytest.approx(0.1)
