"""Tests for manual skip behaviour in main loop."""

import importlib
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, Optional

import pytest

import data_fetch
from screens.registry import ScreenDefinition
from services import wifi_utils


@dataclass
class _FakeScheduler:
    order: Iterable[str]

    def __post_init__(self) -> None:
        self._order = list(self.order)
        self._cursor = 0
        self.node_count = len(self._order)

    def next_available(self, registry: dict[str, ScreenDefinition]) -> Optional[ScreenDefinition]:
        if not self._order:
            return None
        sid = self._order[self._cursor % len(self._order)]
        self._cursor += 1
        return registry.get(sid)


@pytest.fixture
def main_module(monkeypatch):
    monkeypatch.setattr(data_fetch, "fetch_weather", dict)
    monkeypatch.setattr(data_fetch, "fetch_blackhawks_last_game", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_blackhawks_live_game", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_blackhawks_next_game", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_blackhawks_next_home_game", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_wolves_games", dict)
    monkeypatch.setattr(data_fetch, "fetch_bulls_last_game", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_bulls_live_game", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_bulls_next_game", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_bulls_next_home_game", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_cubs_games", dict)
    monkeypatch.setattr(data_fetch, "fetch_cubs_standings", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_sox_games", dict)
    monkeypatch.setattr(data_fetch, "fetch_sox_standings", lambda: None)
    monkeypatch.setattr(wifi_utils, "start_monitor", lambda *args, **kwargs: None)

    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    main.screen_scheduler = None
    main._last_screen_id = None
    main._skip_request_pending = False
    main._manual_skip_event.clear()

    yield main

    main.request_shutdown("tests")
    sys.modules.pop("main", None)


def _build_registry(*ids: str) -> dict[str, ScreenDefinition]:
    return {sid: ScreenDefinition(id=sid, render=lambda: None) for sid in ids}


def test_next_screen_skips_date_when_manual_skip_requested(main_module):
    registry = _build_registry("date", "weather1")
    main_module.screen_scheduler = _FakeScheduler(["date", "weather1"])
    main_module._skip_request_pending = True
    main_module._last_screen_id = "date"

    entry = main_module._next_screen_from_registry(registry)

    assert entry is not None
    assert entry.id == "weather1"
    assert main_module._skip_request_pending is False


def test_next_screen_skips_previous_screen_when_possible(main_module):
    registry = _build_registry("weather1", "inside")
    main_module.screen_scheduler = _FakeScheduler(["weather1", "inside", "weather1"])
    main_module._skip_request_pending = True
    main_module._last_screen_id = "weather1"

    entry = main_module._next_screen_from_registry(registry)

    assert entry is not None
    assert entry.id == "inside"


def test_next_screen_uses_first_candidate_when_only_avoided_options(main_module):
    registry = _build_registry("date", "nixie")
    main_module.screen_scheduler = _FakeScheduler(["nixie", "date"])
    main_module._skip_request_pending = True
    main_module._last_screen_id = "date"

    entry = main_module._next_screen_from_registry(registry)

    assert entry is not None
    assert entry.id == "nixie"


def test_next_screen_falls_back_when_no_alternative_available(main_module):
    registry = _build_registry("date")
    main_module.screen_scheduler = _FakeScheduler(["date"])
    main_module._skip_request_pending = True
    main_module._last_screen_id = "date"

    entry = main_module._next_screen_from_registry(registry)

    assert entry is not None
    assert entry.id == "date"
    assert main_module._skip_request_pending is False


def test_next_screen_returns_none_without_scheduler(main_module):
    main_module.screen_scheduler = None
    main_module._skip_request_pending = True

    entry = main_module._next_screen_from_registry({})

    assert entry is None
    assert main_module._skip_request_pending is False


def test_wait_with_button_checks_honors_pending_skip_event(main_module):
    main_module._manual_skip_event.set()

    assert main_module._wait_with_button_checks(5.0) is True
    assert main_module._manual_skip_event.is_set() is False


def test_touch_double_tap_on_right_third_requests_next_screen(main_module):
    class _FakeEvent:
        def __init__(self, event_type, x):
            self.type = event_type
            self.x = x

    class _FakePygame:
        FINGERDOWN = 1
        MOUSEBUTTONDOWN = 2

        class event:
            @staticmethod
            def get(_event_types):
                return [_FakeEvent(1, 0.9), _FakeEvent(1, 0.9)]

    main_module.display = type("D", (), {"width": 320})()
    main_module.pygame = _FakePygame()
    main_module._skip_request_pending = False
    main_module._last_touch_tap_monotonic = 0.0

    assert main_module._check_touch_skip_request() is True
    assert main_module._skip_request_pending is True


def test_touch_double_tap_ignores_left_side_taps(main_module):
    class _FakeEvent:
        def __init__(self, event_type, x):
            self.type = event_type
            self.x = x

    class _FakePygame:
        FINGERDOWN = 1
        MOUSEBUTTONDOWN = 2

        class event:
            @staticmethod
            def get(_event_types):
                return [_FakeEvent(1, 0.2), _FakeEvent(1, 0.2)]

    main_module.display = type("D", (), {"width": 320})()
    main_module.pygame = _FakePygame()
    main_module._skip_request_pending = False
    main_module._last_touch_tap_monotonic = 0.0

    assert main_module._check_touch_skip_request() is False
    assert main_module._skip_request_pending is False


def test_touch_tap_on_quad_tile_requests_fullscreen(main_module):
    class _FakeEvent:
        def __init__(self, event_type, x, y):
            self.type = event_type
            self.x = x
            self.y = y

    class _FakePygame:
        FINGERDOWN = 1
        MOUSEBUTTONDOWN = 2

        class event:
            @staticmethod
            def get(_event_types):
                return [_FakeEvent(1, 0.8, 0.2)]

    main_module.display = type("D", (), {"width": 320, "height": 240})()
    main_module.pygame = _FakePygame()
    main_module._pending_touch_focus_screen_id = None
    main_module._pending_touch_return_screen_id = None

    handled = main_module._check_touch_skip_request(
        current_screen_id="quad",
        current_quad_tiles=["date", "inside", "weather1", "weather2"],
    )

    assert handled is True
    assert main_module._pending_touch_focus_screen_id == "inside"
    assert main_module._pending_touch_return_screen_id == "quad"


def test_touch_tap_on_quad_prefers_most_recent_tap(main_module):
    class _FakeEvent:
        def __init__(self, event_type, x, y):
            self.type = event_type
            self.x = x
            self.y = y

    class _FakePygame:
        FINGERDOWN = 1
        MOUSEBUTTONDOWN = 2

        class event:
            @staticmethod
            def get(_event_types):
                return [
                    _FakeEvent(1, 0.2, 0.2),
                    _FakeEvent(1, 0.8, 0.8),
                ]

    main_module.display = type("D", (), {"width": 320, "height": 240, "rotation": 0})()
    main_module.pygame = _FakePygame()
    main_module._pending_touch_focus_screen_id = None

    handled = main_module._check_touch_skip_request(
        current_screen_id="quad",
        current_quad_tiles=["date", "inside", "weather1", "weather2"],
    )

    assert handled is True
    assert main_module._pending_touch_focus_screen_id == "weather2"


def test_touch_tap_on_quad_tile_honors_display_rotation(main_module):
    class _FakeEvent:
        def __init__(self, event_type, x, y):
            self.type = event_type
            self.x = x
            self.y = y

    class _FakePygame:
        FINGERDOWN = 1
        MOUSEBUTTONDOWN = 2

        class event:
            @staticmethod
            def get(_event_types):
                return [_FakeEvent(1, 0.8, 0.2)]

    main_module.display = type("D", (), {"width": 320, "height": 240, "rotation": 90})()
    main_module.pygame = _FakePygame()
    main_module._pending_touch_focus_screen_id = None

    handled = main_module._check_touch_skip_request(
        current_screen_id="quad",
        current_quad_tiles=["date", "inside", "weather1", "weather2"],
    )

    assert handled is True
    assert main_module._pending_touch_focus_screen_id == "weather2"


def test_touch_focused_screen_skips_screenshot_once(main_module):
    main_module._pending_touch_focus_screenshot_skip_ids.clear()
    main_module._request_touch_focus("inside", return_screen_id="quad")

    assert main_module._consume_touch_focus_screenshot_skip("inside") is True
    assert main_module._consume_touch_focus_screenshot_skip("inside") is False


def test_escape_double_press_stops_service(main_module, monkeypatch):
    class _FakeEvent:
        def __init__(self, event_type, key):
            self.type = event_type
            self.key = key

    class _FakePygame:
        KEYDOWN = 1
        K_ESCAPE = 27

        class event:
            @staticmethod
            def get(_event_types):
                return [_FakeEvent(1, 27)]

    timestamps = iter((100.0, 100.5))
    monkeypatch.setattr(main_module.time, "monotonic", lambda: next(timestamps))

    called = {"count": 0}
    monkeypatch.setattr(main_module, "_ESC_DOUBLE_PRESS_ACTION", "stop")
    monkeypatch.setattr(main_module, "_stop_desk_display_service", lambda: called.__setitem__("count", called["count"] + 1))

    main_module.pygame = _FakePygame()
    main_module._last_escape_key_monotonic = 0.0

    assert main_module._check_keyboard_shutdown_request() is False
    assert main_module._check_keyboard_shutdown_request() is True
    assert called["count"] == 1


def test_escape_double_press_expires_outside_interval(main_module, monkeypatch):
    class _FakeEvent:
        def __init__(self, event_type, key):
            self.type = event_type
            self.key = key

    class _FakePygame:
        KEYDOWN = 1
        K_ESCAPE = 27

        class event:
            @staticmethod
            def get(_event_types):
                return [_FakeEvent(1, 27)]

    timestamps = iter((100.0, 101.2))
    monkeypatch.setattr(main_module.time, "monotonic", lambda: next(timestamps))

    called = {"count": 0}
    monkeypatch.setattr(main_module, "_ESC_DOUBLE_PRESS_ACTION", "stop")
    monkeypatch.setattr(main_module, "_stop_desk_display_service", lambda: called.__setitem__("count", called["count"] + 1))

    main_module.pygame = _FakePygame()
    main_module._last_escape_key_monotonic = 0.0

    assert main_module._check_keyboard_shutdown_request() is False
    assert main_module._check_keyboard_shutdown_request() is False
    assert called["count"] == 0


def test_escape_double_press_restarts_service_when_configured(main_module, monkeypatch):
    class _FakeEvent:
        def __init__(self, event_type, key):
            self.type = event_type
            self.key = key

    class _FakePygame:
        KEYDOWN = 1
        K_ESCAPE = 27

        class event:
            @staticmethod
            def get(_event_types):
                return [_FakeEvent(1, 27)]

    timestamps = iter((100.0, 100.5))
    monkeypatch.setattr(main_module.time, "monotonic", lambda: next(timestamps))

    called = {"restart": 0, "stop": 0}
    monkeypatch.setattr(main_module, "_ESC_DOUBLE_PRESS_ACTION", "restart")
    monkeypatch.setattr(
        main_module,
        "_restart_desk_display_service",
        lambda: called.__setitem__("restart", called["restart"] + 1),
    )
    monkeypatch.setattr(
        main_module,
        "_stop_desk_display_service",
        lambda: called.__setitem__("stop", called["stop"] + 1),
    )

    main_module.pygame = _FakePygame()
    main_module._last_escape_key_monotonic = 0.0

    assert main_module._check_keyboard_shutdown_request() is False
    assert main_module._check_keyboard_shutdown_request() is True
    assert called["restart"] == 1
    assert called["stop"] == 0
