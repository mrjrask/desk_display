"""Tests for manual skip behaviour in main loop."""

from dataclasses import dataclass
import importlib
import sys
from typing import Dict, Iterable, Optional

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

    def next_available(self, registry: Dict[str, ScreenDefinition]) -> Optional[ScreenDefinition]:
        if not self._order:
            return None
        sid = self._order[self._cursor % len(self._order)]
        self._cursor += 1
        return registry.get(sid)


@pytest.fixture
def main_module(monkeypatch):
    monkeypatch.setattr(data_fetch, "fetch_weather", lambda: {})
    monkeypatch.setattr(data_fetch, "fetch_blackhawks_last_game", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_blackhawks_live_game", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_blackhawks_next_game", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_blackhawks_next_home_game", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_bulls_last_game", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_bulls_live_game", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_bulls_next_game", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_bulls_next_home_game", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_cubs_games", lambda: {})
    monkeypatch.setattr(data_fetch, "fetch_cubs_standings", lambda: None)
    monkeypatch.setattr(data_fetch, "fetch_sox_games", lambda: {})
    monkeypatch.setattr(data_fetch, "fetch_sox_standings", lambda: None)
    monkeypatch.setattr(wifi_utils, "start_monitor", lambda: None)

    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    main.screen_scheduler = None
    main._last_screen_id = None
    main._skip_request_pending = False
    main._manual_skip_event.clear()

    yield main

    main.request_shutdown("tests")
    sys.modules.pop("main", None)


def _build_registry(*ids: str) -> Dict[str, ScreenDefinition]:
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
