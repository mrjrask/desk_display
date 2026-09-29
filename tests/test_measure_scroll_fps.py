"""The on-device scroll frame rate probe (scripts/measure_scroll_fps.py)."""

import time

from display_profiles import PROFILE_PRESETS
from playback.package_player import PackagePlayback
from scripts import measure_scroll_fps as probe


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _run(legacy):
    profile = PROFILE_PRESETS["display_hat_mini"]
    clock = _Clock()
    playback = PackagePlayback(probe.scroll_package(profile, pages=2), profile, hold_seconds=0)

    def push(_image):
        clock.now += 0.03  # a Display HAT Mini frame over SPI

    return probe.measure(playback, push, legacy=legacy, show=lambda: push(None), clock=clock, sleep=clock.sleep)


def test_current_playback_moves_one_pixel_every_frame():
    stats = _run(legacy=False)
    assert set(stats.steps) == {1}
    assert stats.frames == PROFILE_PRESETS["display_hat_mini"].height
    assert "fps" in stats.report()


def test_legacy_playback_reproduces_the_judder():
    stats = _run(legacy=True)
    assert max(stats.steps) > 1
    assert stats.fps < _run(legacy=False).fps


def test_probe_runs_without_hardware_quickly():
    started = time.monotonic()
    _run(legacy=False)
    assert time.monotonic() - started < 5
