"""Deterministic capability fallbacks for playing a screen on one client.

The same function decides what a client actually plays and what the
assignment UI warns about before a playlist is deployed, so a warning never
disagrees with the display. The decision depends only on the screen's
remote-mode class (:mod:`rendering.screen_classes`) and the client's
declared capabilities:

=====================  =================================================
Condition              Result
=====================  =================================================
``unsupported`` class  ``unavailable``: skipped on remote clients
no animation support   ``still``: moving screens show their still image
                       (clocks are drawn locally either way)
no touchscreen         interactive quads play (still or animated), but a
                       tap does not open a tile (``expands`` is false)
one-bit display        colour-dependent screens play, with a warning
otherwise              ``full``
=====================  =================================================
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from rendering import screen_classes

FULL = "full"
STILL = "still"
UNAVAILABLE = "unavailable"
_MOVING = frozenset({screen_classes.SCROLLING_CANVAS, screen_classes.TICKER_OVERLAY,
                     screen_classes.FINITE_ANIMATION, screen_classes.COMPOSITE,
                     screen_classes.INTERACTIVE_FOCUS})


@dataclass(frozen=True)
class Fallback:
    screen_id: str
    mode: str
    expands: bool = False
    notes: tuple[tuple[str, str, str], ...] = ()  # (code, severity, message)

    @property
    def animated(self) -> bool:
        return self.mode == FULL


def playback_mode(screen_id: str, *, supports_animation: bool, has_touch: bool,
                  color_mode: str) -> Fallback:
    entry = screen_classes.CLASSIFICATIONS.get(screen_id)
    if entry is None or entry.kind == screen_classes.UNSUPPORTED:
        reason = entry.note if entry is not None else "unknown screen"
        return Fallback(screen_id, UNAVAILABLE, False, (("unsupported", "error", f"not available on remote clients ({reason})"),))
    notes: list[tuple[str, str, str]] = []
    mode = FULL
    if entry.kind in _MOVING and not supports_animation:
        mode = STILL
        notes.append(("no_animation", "warning", "shows its still image; the client does not animate"))
    if entry.interactive and not has_touch:
        notes.append(("no_touch", "info", "tiles cannot be tapped open without a touchscreen"))
    if color_mode == "1" and screen_id in screen_classes.COLOR_DEPENDENT:
        notes.append(("monochrome", "warning", "depends on colour; a one-bit display loses detail"))
    return Fallback(screen_id, mode, entry.interactive and has_touch, tuple(notes))


def plan(screens: Iterable[str], *, supports_animation: bool, has_touch: bool,
         color_mode: str) -> dict[str, Fallback]:
    return {s: playback_mode(s, supports_animation=supports_animation, has_touch=has_touch,
                             color_mode=color_mode) for s in sorted(set(screens))}


__all__ = ["FULL", "Fallback", "STILL", "UNAVAILABLE", "plan", "playback_mode"]
