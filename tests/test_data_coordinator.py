"""Tests for coordinated snapshot sources."""

import screens.nfl_standings as nfl_standings
from services.data_coordinator import DataCoordinator
from services.data_provider import DataProvider


def test_nfl_standings_publish_season_metadata_beside_rows(monkeypatch):
    standings = {"NFC": {"North": [{"abbr": "CHI"}]}, "AFC": {}}
    monkeypatch.setattr(
        nfl_standings,
        "_fetch_standings_data",
        lambda: (standings, None, "2025 season"),
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
        lambda: ({"NFC": {}, "AFC": {}}, nfl_standings.FALLBACK_MESSAGE_OFFSEASON, None),
    )
    coordinator = DataCoordinator(DataProvider())
    coordinator.read_nfl_league_standings()

    meta = coordinator.snapshot()["nfl_standings_meta"]
    assert meta["fallback_message"] == nfl_standings.FALLBACK_MESSAGE_OFFSEASON
