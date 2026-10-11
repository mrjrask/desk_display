"""Tests for coordinated snapshot sources."""


import data_fetch
import screens.draw_hawks_schedule as hawks
import screens.nfl_standings as nfl_standings
import screens.nhl_standings as nhl_standings
from services import data_coordinator
from services.data_coordinator import DataCoordinator, fetch_hawks_live_feed
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


def test_identical_publishes_keep_revisions():
    coordinator = DataCoordinator(DataProvider())
    first = coordinator.publish("weather", {"temp": 70, "hourly": [1, 2]})
    again = coordinator.publish("weather", {"temp": 70, "hourly": [1, 2]})
    assert again.revision == first.revision
    assert again.source_revisions["weather"] == first.source_revisions["weather"]
    changed = coordinator.publish("weather", {"temp": 71, "hourly": [1, 2]})
    assert changed.revision == first.revision + 1
    assert changed.source_revisions["weather"] == first.source_revisions["weather"] + 1


def test_refresh_from_a_ttl_cache_keeps_revisions():
    coordinator = DataCoordinator(DataProvider())
    payload = {"games": [1]}
    coordinator.register_source("cubs", lambda: payload, ttl_seconds=300)
    first = coordinator.refresh()
    assert coordinator.refresh().source_revisions == first.source_revisions
    payload = {"games": [1, 2]}
    assert coordinator.refresh(force=True).source_revisions["cubs"] == first.source_revisions["cubs"] + 1


def test_values_without_equality_always_count_as_changed():
    class Opaque:
        def __eq__(self, other):
            raise RuntimeError("no comparison")

    coordinator = DataCoordinator(DataProvider())
    value = Opaque()
    first = coordinator.publish("x", value)
    assert coordinator.publish("x", value).source_revisions["x"] == first.source_revisions["x"] + 1


def test_restored_values_are_the_baseline_for_changes():
    coordinator = DataCoordinator(DataProvider())
    coordinator.restore({"weather": ({"temp": 70}, 7)})
    assert coordinator.publish("weather", {"temp": 70}).source_revisions["weather"] == 7
    assert coordinator.publish("weather", {"temp": 72}).source_revisions["weather"] == 8


def test_snapshot_reuses_the_values_frozen_at_publish(monkeypatch):
    import services.data_coordinator as module

    coordinator = DataCoordinator(DataProvider())
    coordinator.publish("scores", {"games": [{"id": 1}]})

    def refuse(_value):
        raise AssertionError("snapshot() must not re-freeze published data")

    monkeypatch.setattr(module, "_freeze", refuse)
    snapshot = coordinator.snapshot()
    assert snapshot.values["scores"]["games"] == ({"id": 1},)
    assert snapshot.revision == coordinator.snapshot().revision


def test_hawks_live_feed_resolves_nhl_id_instead_of_ics_uid(monkeypatch):
    requested = []
    monkeypatch.setattr(hawks, "fetch_schedule", lambda days_back, days_fwd: {"s": 1})
    monkeypatch.setattr(hawks, "classify_games", lambda s: ({"gamePk": 2025020123}, None, None))
    monkeypatch.setattr(
        hawks, "fetch_game_feed", lambda pk: requested.append(pk) or {"homeScore": 1}
    )

    feed = fetch_hawks_live_feed({"id": "abc@ecal.com", "gamePk": "abc@ecal.com"})

    assert feed == {"homeScore": 1}
    assert requested == [2025020123]


def _forbidden():
    raise AssertionError("must not fetch")


def test_hawks_live_feed_skips_non_numeric_id_without_schedule_match(monkeypatch):
    monkeypatch.setattr(hawks, "fetch_schedule", lambda days_back, days_fwd: None)
    monkeypatch.setattr(hawks, "fetch_game_feed", lambda pk: _forbidden())

    assert fetch_hawks_live_feed({"id": "abc@ecal.com"}) is None


def test_hawks_team_feed_includes_live_feed(monkeypatch):
    live = {"id": "abc@ecal.com"}
    feed = {"awayScore": 1, "homeScore": 2, "perOrdinal": 2, "clock": "05:12"}
    for name, value in {
        "fetch_blackhawks_live_game": live,
        "fetch_blackhawks_standings": {},
        "fetch_blackhawks_last_game": None,
        "fetch_blackhawks_next_game": None,
        "fetch_blackhawks_next_home_game": None,
    }.items():
        monkeypatch.setattr(data_fetch, name, lambda value=value: value)
    monkeypatch.setattr(data_coordinator, "fetch_hawks_live_feed", lambda game: feed)

    coordinator = DataCoordinator(DataProvider())
    coordinator.read_legacy_team("hawks", force=True)

    assert coordinator.snapshot().values["hawks"]["live_feed"]["clock"] == "05:12"
