"""Tests for coordinated snapshot sources."""

import screens.nfl_standings as nfl_standings
import screens.nhl_standings as nhl_standings
from services.data_coordinator import DataCoordinator
from services.data_provider import DataProvider


def test_nfl_standings_publish_season_metadata_beside_rows(monkeypatch):
    standings = {"NFC": {"North": [{"abbr": "CHI"}]}, "AFC": {}}
    monkeypatch.setattr(
        nfl_standings,
        "_fetch_standings_data",
        lambda **_kwargs: (standings, None, "2025 season"),
    )
    coordinator = DataCoordinator(DataProvider())

    assert coordinator.read_nfl_league_standings() == standings

    snapshot = coordinator.snapshot()
    assert snapshot["nfl_standings"]["NFC"]["North"][0]["abbr"] == "CHI"
    assert dict(snapshot["nfl_standings_meta"]) == {
        "fallback_message": None,
        "season_note": "2025 season",
    }


def test_nfl_offseason_message_reaches_snapshot(monkeypatch):
    monkeypatch.setattr(
        nfl_standings,
        "_fetch_standings_data",
        lambda **_kwargs: (
            {"NFC": {}, "AFC": {}}, nfl_standings.FALLBACK_MESSAGE_OFFSEASON, None
        ),
    )
    coordinator = DataCoordinator(DataProvider())
    coordinator.read_nfl_league_standings()

    meta = coordinator.snapshot()["nfl_standings_meta"]
    assert meta["fallback_message"] == nfl_standings.FALLBACK_MESSAGE_OFFSEASON


def test_forced_nfl_read_bypasses_module_cache(monkeypatch):
    stale = {"NFC": {"North": [{"abbr": "OLD"}]}, "AFC": {}}
    fresh = {"NFC": {"North": [{"abbr": "NEW"}]}, "AFC": {}}
    monkeypatch.setattr(
        nfl_standings,
        "_STANDINGS_CACHE",
        {"data": stale, "timestamp": 10**12, "message": None, "season_note": None},
    )
    monkeypatch.setattr(nfl_standings, "_in_offseason", lambda *args, **kwargs: False)
    monkeypatch.setattr(nfl_standings, "_target_season_year", lambda *args, **kwargs: 2026)
    monkeypatch.setattr(
        nfl_standings._SESSION,
        "get",
        lambda *args, **kwargs: type(
            "Response", (), {"text": "csv", "raise_for_status": lambda self: None}
        )(),
    )
    monkeypatch.setattr(nfl_standings, "_parse_csv_standings", lambda text, season: (fresh, 2026))
    coordinator = DataCoordinator(DataProvider())

    assert coordinator.read_nfl_league_standings() == stale
    assert coordinator.read_nfl_league_standings(force=True) == fresh


def test_forced_nhl_read_bypasses_module_cache(monkeypatch):
    stale = {"Western": {"Central": [{"abbr": "OLD"}]}}
    fresh = {"Western": {"Central": [{"abbr": "NEW"}]}}
    monkeypatch.setattr(
        nhl_standings, "_STANDINGS_CACHE", {"data": stale, "timestamp": 10**12}
    )
    monkeypatch.setattr(nhl_standings, "_fetch_standings_api_web", lambda: fresh)
    coordinator = DataCoordinator(DataProvider())

    assert coordinator.read_nhl_league_standings() == stale
    assert coordinator.read_nhl_league_standings(force=True) == fresh
