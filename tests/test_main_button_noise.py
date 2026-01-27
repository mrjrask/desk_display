"""Tests for filtering out spurious simultaneous button presses."""

import importlib
import sys

import pytest

import data_fetch
from services import wifi_utils


@pytest.fixture
def main_for_buttons(monkeypatch):
    monkeypatch.setattr(wifi_utils, "start_monitor", lambda *args, **kwargs: None)
    monkeypatch.setattr(wifi_utils, "stop_monitor", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_weather", lambda: {})
    monkeypatch.setattr(data_fetch, "fetch_blackhawks_last_game", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_blackhawks_live_game", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_blackhawks_next_game", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_blackhawks_next_home_game", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_wolves_games", lambda: {})
    monkeypatch.setattr(data_fetch, "fetch_bulls_last_game", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_bulls_live_game", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_bulls_next_game", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_bulls_next_home_game", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_cubs_games", lambda: {})
    monkeypatch.setattr(data_fetch, "fetch_cubs_standings", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_sox_games", lambda: {})
    monkeypatch.setattr(data_fetch, "fetch_sox_standings", lambda: None)

    sys.modules.pop("main", None)
    main = importlib.import_module("main")

    if main._button_monitor_thread and main._button_monitor_thread.is_alive():
        main._shutdown_event.set()
        main._button_monitor_thread.join(timeout=1.0)
        main._shutdown_event.clear()
        main._button_monitor_thread = None

    main._manual_skip_event.clear()
    main._skip_request_pending = False
    main._BUTTON_STATE = {name: False for name in main._BUTTON_NAMES}
    main._manual_display_off = False

    yield main

    main.request_shutdown("tests")
    main._finalize_shutdown()
    sys.modules.pop("main", None)


class _FakeDisplay:
    def __init__(self, pressed=None):
        self.pressed = set(pressed or [])

    def is_button_pressed(self, name: str) -> bool:
        return name in self.pressed


def test_simultaneous_presses_are_treated_as_noise(main_for_buttons, monkeypatch):
    fake_display = _FakeDisplay({"A", "B", "X", "Y"})
    main_for_buttons.display = fake_display

    handled = []

    def fake_handle(name: str) -> bool:
        handled.append(name)
        return False

    monkeypatch.setattr(main_for_buttons, "_handle_button_down", fake_handle)

    result = main_for_buttons._check_control_buttons()

    assert result is False
    assert handled == []
    assert all(state is False for state in main_for_buttons._BUTTON_STATE.values())
    assert main_for_buttons._skip_request_pending is False


def test_single_press_still_processed(main_for_buttons, monkeypatch):
    fake_display = _FakeDisplay({"X"})
    main_for_buttons.display = fake_display

    handled = []

    def fake_handle(name: str) -> bool:
        handled.append(name)
        return name == "X"

    monkeypatch.setattr(main_for_buttons, "_handle_button_down", fake_handle)

    result = main_for_buttons._check_control_buttons()

    assert result is True
    assert handled == ["X"]
    assert main_for_buttons._BUTTON_STATE["X"] is True

    fake_display.pressed.clear()
    handled.clear()

    result = main_for_buttons._check_control_buttons()

    assert result is False
    assert handled == []
    assert main_for_buttons._BUTTON_STATE["X"] is False


def test_a_button_advances_to_next_screen(main_for_buttons):
    # Provide a non-None display to allow the handler to run.
    main_for_buttons.display = object()

    assert main_for_buttons._handle_button_down("A") is True
    assert main_for_buttons._skip_request_pending is True
    assert main_for_buttons._manual_skip_event.is_set()


def test_x_button_returns_to_previous_screen(main_for_buttons):
    main_for_buttons.display = object()
    main_for_buttons._screen_history[:] = ["first", "second", "third"]

    assert main_for_buttons._handle_button_down("X") is True
    assert main_for_buttons._pending_previous_screen_id == "second"
    assert main_for_buttons._manual_skip_event.is_set()


def test_b_button_toggles_display(main_for_buttons, monkeypatch):
    class _PowerDisplay:
        def __init__(self):
            self.cleared = False
            self.shown = False

        def clear(self):
            self.cleared = True

        def show(self):
            self.shown = True

    display = _PowerDisplay()
    main_for_buttons.display = display
    main_for_buttons._manual_display_off = False
    main_for_buttons._dark_hours_active = False

    called = {"resume": 0, "suspend": 0}

    def fake_resume():
        called["resume"] += 1

    def fake_suspend():
        called["suspend"] += 1

    monkeypatch.setattr(main_for_buttons, "resume_display_updates", fake_resume)
    monkeypatch.setattr(main_for_buttons, "suspend_display_updates", fake_suspend)

    assert main_for_buttons._handle_button_down("B") is True
    assert main_for_buttons._manual_display_off is True
    assert called["resume"] == 1
    assert called["suspend"] == 1
    assert display.cleared is True
    assert display.shown is True

    assert main_for_buttons._handle_button_down("B") is True
    assert main_for_buttons._manual_display_off is False
    assert called["resume"] == 2
