"""Tests for duplicate frame suppression in Display output."""

from types import SimpleNamespace

import utils


def _headless_display_with_writer():
    display = utils.Display()
    writes = []
    display._output_strategy = "test"
    display._frame_transform = lambda image: image
    display._frame_writer = writes.append
    display._last_written_frame_signature = None
    return display, writes


def test_update_display_suppresses_unchanged_transformed_frame():
    display, writes = _headless_display_with_writer()
    frame = utils.Image.new("RGB", (display.width, display.height), "navy")

    display.image(frame)
    display.image(frame)

    assert len(writes) == 1


def test_update_display_writes_when_transformed_frame_changes():
    display, writes = _headless_display_with_writer()

    display.image(utils.Image.new("RGB", (display.width, display.height), "navy"))
    display.image(utils.Image.new("RGB", (display.width, display.height), "maroon"))

    assert len(writes) == 2


def test_clear_force_bypasses_unchanged_frame_suppression():
    display, writes = _headless_display_with_writer()

    display.clear()
    display.clear()
    display.clear(force=True)

    assert len(writes) == 2


def test_display_hat_mini_recovery_due_bypasses_unchanged_frame_suppression(monkeypatch):
    display, writes = _headless_display_with_writer()
    display._output_strategy = "display_hat_mini"
    display._display = SimpleNamespace()
    display._display_reinit_seconds = 10
    display._display_reinit_disabled = False
    display._last_display_reinit = 0.0
    display._next_display_reinit_retry = 0.0
    monkeypatch.setattr(utils.time, "monotonic", lambda: 20.0)

    frame = utils.Image.new("RGB", (display.width, display.height), "black")
    display.image(frame)
    display.image(frame)

    assert len(writes) == 2
