"""Tests for display frame refresh detection helpers."""

import importlib
import sys

import pytest

import screens.draw_hawks_schedule as hawks


class _DisplayWithFrameCounter:
    def __init__(self, value):
        self._value = value

    def frame_id(self):
        return self._value


class _DisplayWithShowAndFrames:
    def __init__(self, frames):
        self._frames = list(frames)
        self._last = self._frames[-1] if self._frames else 0
        self.shows = 0

    def frame_id(self):
        if self._frames:
            self._last = self._frames.pop(0)
        return self._last

    def show(self):
        self.shows += 1


def _load_main():
    sys.modules.pop("main", None)
    return importlib.import_module("main")


def test_frame_id_changed_returns_true_without_prior_frame_id():
    main = _load_main()

    assert main._frame_id_changed(object(), None) is True


def test_frame_id_changed_detects_no_refresh():
    main = _load_main()
    display = _DisplayWithFrameCounter(42)

    assert main._frame_id_changed(display, 42) is False


def test_frame_id_changed_detects_refresh():
    main = _load_main()
    display = _DisplayWithFrameCounter(43)

    assert main._frame_id_changed(display, 42) is True


def test_wait_with_button_checks_flushes_when_frame_changes(monkeypatch):
    main = _load_main()
    display = _DisplayWithShowAndFrames([1, 2, 2])
    main.display = display
    main._shutdown_event.clear()
    main._manual_skip_event.clear()
    main._skip_request_pending = False
    monkeypatch.setattr(main, "BUTTON_POLL_INTERVAL", 0.0)

    # Keep the wait loop deterministic and short.
    times = iter([0.0, 0.0, 0.0, 1.0])
    monkeypatch.setattr(main.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(main, "_check_control_buttons", lambda: False)

    assert main._wait_with_button_checks(0.1) is False
    assert display.shows == 1


def test_hawks_live_feed_resolves_nhl_id_instead_of_ics_uid(monkeypatch):
    main = _load_main()
    requested = []
    monkeypatch.setattr(hawks, "fetch_schedule", lambda days_back, days_fwd: {"s": 1})
    monkeypatch.setattr(hawks, "classify_games", lambda s: ({"gamePk": 2025020123}, None, None))
    monkeypatch.setattr(
        hawks, "fetch_game_feed", lambda pk: requested.append(pk) or {"homeScore": 1}
    )

    feed = main._fetch_hawks_live_feed({"id": "abc@ecal.com", "gamePk": "abc@ecal.com"})

    assert feed == {"homeScore": 1}
    assert requested == [2025020123]


def test_hawks_live_feed_skips_non_numeric_id_without_schedule_match(monkeypatch):
    main = _load_main()
    monkeypatch.setattr(hawks, "fetch_schedule", lambda days_back, days_fwd: None)
    monkeypatch.setattr(hawks, "fetch_game_feed", lambda pk: pytest.fail("must not fetch"))

    assert main._fetch_hawks_live_feed({"id": "abc@ecal.com"}) is None
