"""Pace package motion so a slow panel shows every step, as v0.1 did.

v0.1 scrolled by drawing each offset in turn and sleeping what was left of
the frame, so a panel slower than the requested frame rate (a Pi Zero 2 W
pushing 320x240 over SPI) scrolled a little slower but never skipped a
pixel. Package playback is a function of time instead; driven by the wall
clock, a slow push makes the next frame jump two or three pixels at
uneven intervals, which reads as judder. :class:`MotionClock` is the time
fed to playback: it follows the wall clock while nothing moves (pauses,
holds) and advances at most one frame per frame drawn while something does.
"""
from __future__ import annotations


class MotionClock:
    """Playback time that never runs more than one frame ahead of the panel.

    ``lag`` is how far it has fallen behind the wall clock (at most
    ``max_lag``); the caller extends the screen's time by it so a slowed
    scroll still reaches the bottom and holds for the full hold time.
    """

    def __init__(self, t: float, now: float, *, max_lag: float) -> None:
        self.t = max(0.0, t)
        self.lag = 0.0
        self.max_lag = max(0.0, max_lag)
        self._at = now
        self._drew = False

    def tick(self, now: float, frame_seconds: float) -> float:
        """Advance to *now* and return the playback time to draw."""

        elapsed = max(0.0, now - self._at)
        step = min(elapsed, frame_seconds) if self._drew else elapsed
        self.lag = min(self.max_lag, self.lag + elapsed - step)
        self.t += step
        self._at = now
        return self.t

    def drew(self, drawn: bool) -> None:
        """Record whether the frame for the last tick was pushed to the panel."""

        self._drew = drawn


__all__ = ["MotionClock"]
