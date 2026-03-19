"""Tests for display frame refresh detection helpers."""

import importlib
import sys


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
