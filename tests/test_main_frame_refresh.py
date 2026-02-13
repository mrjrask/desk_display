"""Tests for display frame refresh detection helpers."""

import importlib
import sys


class _DisplayWithFrameCounter:
    def __init__(self, value):
        self._value = value

    def frame_id(self):
        return self._value


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
