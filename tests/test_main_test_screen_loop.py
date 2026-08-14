"""Tests for the DESK_DISPLAY_TEST_SCREEN single-screen looping mode."""

import importlib
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

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
    main.TEST_LOOP_SCREEN_ID = None

    yield main

    main.request_shutdown("tests")
    sys.modules.pop("main", None)


def _build_registry(*ids: str) -> dict[str, ScreenDefinition]:
    return {sid: ScreenDefinition(id=sid, render=lambda: None) for sid in ids}


def test_select_entry_uses_normal_rotation_when_test_mode_disabled(main_module):
    registry = _build_registry("date", "weather1")
    main_module.screen_scheduler = _FakeScheduler(["date", "weather1"])

    entry = main_module._select_entry_for_iteration(registry)

    assert entry is not None
    assert entry.id == "date"


def test_select_entry_repeats_test_screen_every_iteration(main_module):
    registry = _build_registry("news headlines", "weather1")
    main_module.screen_scheduler = _FakeScheduler(["news headlines", "weather1"])
    main_module.TEST_LOOP_SCREEN_ID = "news headlines"

    first = main_module._select_entry_for_iteration(registry)
    second = main_module._select_entry_for_iteration(registry)

    assert first is not None and first.id == "news headlines"
    assert second is not None and second.id == "news headlines"


def test_select_entry_falls_back_when_test_screen_missing(main_module, caplog):
    registry = _build_registry("date", "weather1")
    main_module.screen_scheduler = _FakeScheduler(["date", "weather1"])
    main_module.TEST_LOOP_SCREEN_ID = "does not exist"

    entry = main_module._select_entry_for_iteration(registry)

    assert entry is not None
    assert entry.id == "date"


def test_select_entry_falls_back_when_test_screen_unavailable(main_module):
    registry = _build_registry("date")
    registry["news headlines"] = main_module.ScreenDefinition(
        id="news headlines", render=lambda: None, available=False
    )
    main_module.screen_scheduler = _FakeScheduler(["date"])
    main_module.TEST_LOOP_SCREEN_ID = "news headlines"

    entry = main_module._select_entry_for_iteration(registry)

    assert entry is not None
    assert entry.id == "date"
