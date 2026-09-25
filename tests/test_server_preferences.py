"""Phase 14c: content preferences, logos, and last-good feed data on the server."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest

from remote_display.models import RenderKey, ScreenRevisions
from remote_display.server_rendering import ServerRendering, StyleRevision, preferences_revision
from rendering.logos import ProfileLogos, logo_dimensions, logo_loaders
from services import feeds
from services.air_quality import AirQualityReport
from services.data_coordinator import DataCoordinator
from services.feed_state import FeedStateFile, UnsupportedValue, decode, encode
from services.server_feeds import ServerFeedService

# ── Preference revision ─────────────────────────────────────────────────────


def test_content_settings_change_the_preference_revision():
    base = preferences_revision({})
    assert preferences_revision({"AHL_TEAM_TRICODE": "CHI"}) == base  # the default value
    assert preferences_revision({"AHL_TEAM_TRICODE": "MIL"}) != base


def test_secrets_and_non_content_settings_do_not():
    base = preferences_revision({})
    assert preferences_revision({"AIRNOW_API_KEY": "k" * 20}) == base
    assert preferences_revision({"DESK_DISPLAY_SERVER_PORT": "9999"}) == base


def test_preferences_are_part_of_every_render_key(tmp_path):
    data = DataCoordinator()
    style = StyleRevision([tmp_path / "none.json"])
    first = ServerRendering(data, style, preferences="p-1", logos=ProfileLogos()).revisions(["date", "weather1"])
    second = ServerRendering(data, style, preferences="p-2", logos=ProfileLogos()).revisions(["date", "weather1"])
    for screen in ("date", "weather1"):
        assert first[screen].style_revision.endswith("+p-1")
        key_one = RenderKey.for_screen(screen, "hyperpixel4", first[screen])
        key_two = RenderKey.for_screen(screen, "hyperpixel4", second[screen])
        assert key_one.digest != key_two.digest
    ScreenRevisions(**{"style_revision": first["date"].style_revision, "data_revision": "d1",
                       "renderer_revision": "r1"})  # still a valid revision token


# ── Logos ───────────────────────────────────────────────────────────────────


def test_logo_sizes_follow_the_standalone_rules():
    assert logo_dimensions(320, 240) == (210, 210, 315)
    assert logo_dimensions(1920, 1080) == (594, 594, 891)


def test_logos_load_per_profile_size():
    small = logo_loaders(320, 240)["cubs logo"]()
    large = logo_loaders(800, 480)["cubs logo"]()
    assert small.size == (315, 210) and large.size == (675, 450)
    logos = ProfileLogos()
    cache = logos.for_size(320, 240)
    assert logos.for_size(320, 240) is cache and cache.get("cubs logo") is cache.get("cubs logo")
    assert cache.get("no such logo") is None


def test_missing_logo_file_renders_without_one(tmp_path):
    assert logo_loaders(320, 240, images_dir=str(tmp_path))["cubs logo"]() is None


def test_server_renders_logo_screens(tmp_path):
    data = DataCoordinator()
    for key, value in feeds.default_cache().items():
        data.publish(key, value)
    key = RenderKey.for_screen("weather logo", "hyperpixel4", ScreenRevisions("s", "d", "r"))
    with_logos = ServerRendering(data, StyleRevision([]), preferences="p-1").render(key)
    assert with_logos.image.getbbox() is not None

    class NoLogos:
        def for_size(self, width, height):
            return {}

    # Before Phase 14c the server passed no logos, so logo screens were unavailable.
    with pytest.raises(KeyError):
        ServerRendering(data, StyleRevision([]), preferences="p-1", logos=NoLogos()).render(key)


# ── Saved feed data ─────────────────────────────────────────────────────────


def test_feed_values_round_trip():
    report = AirQualityReport(aqi_value=40, aqi_category="Good", primary_pollutant="O3",
                              pollutant_breakdown=(("O3", 40.0),), component_history=((1.0, 2, 3, 4),))
    value = {"report": report, "when": datetime(2026, 9, 25, 12, tzinfo=UTC), "day": date(2026, 9, 25),
             "standings": {2026: {"al": [1, 2]}}, "items": (1, "x", None, True)}
    restored = decode(json.loads(json.dumps(encode(value))))
    assert restored["report"] == report
    assert restored["when"] == value["when"] and restored["day"] == value["day"]
    assert restored["standings"] == {2026: {"al": [1, 2]}}
    assert restored["items"] == [1, "x", None, True]


def test_only_known_types_are_saved_or_loaded():
    with pytest.raises(UnsupportedValue):
        encode(object())
    with pytest.raises(UnsupportedValue):
        encode({"__type__": "dataclass"})
    with pytest.raises(ValueError):
        decode({"__type__": "dataclass", "name": "Popen", "fields": {}})


def test_saved_file_excludes_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("AIRNOW_API_KEY", "secret-airnow-key-123456")
    state = FeedStateFile(tmp_path / "state.json")
    state.save({"weather": {"value": {"AIRNOW_API_KEY": "x", "note": "secret-airnow-key-123456", "temp": 70},
                            "source_revision": 3, "saved_at": 1.0}})
    text = (tmp_path / "state.json").read_text()
    assert "secret-airnow-key-123456" not in text and "AIRNOW_API_KEY" not in text
    assert state.load()["weather"]["value"]["temp"] == 70


@pytest.mark.parametrize("content", ["{", "[]", json.dumps({"schema_version": 99, "feeds": {}})])
def test_unreadable_or_outdated_file_is_ignored(tmp_path, content):
    path = tmp_path / "state.json"
    path.write_text(content)
    assert FeedStateFile(path).load() == {}


def test_one_bad_entry_does_not_drop_the_others(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"schema_version": 1, "feeds": {
        "weather": {"value": {"temp": 1}, "source_revision": 2, "saved_at": 5},
        "cubs": {"value": {"__type__": "evil"}, "source_revision": 2, "saved_at": 5},
        "hawks": {"value": {}, "source_revision": -1, "saved_at": 5},
    }}))
    assert set(FeedStateFile(path).load()) == {"weather"}


class Clock:
    def __init__(self):
        self.now = 50_000.0

    def __call__(self):
        return self.now


class Provider:
    def __init__(self):
        self.calls = []

    def read(self, key, fetcher, *, ttl_seconds=300, force=False):
        self.calls.append(key)
        return {"stand": {"team": key}}

    def read_weather(self, *, ttl_seconds=300):
        self.calls.append("weather")
        return {"current": {"temp": 70}}

    def read_sports_payloads(self, **kwargs):
        return {}


def service(tmp_path, clock, provider=None):
    provider = provider or Provider()
    settings = SimpleNamespace(ENABLE_WEATHER=True, ENABLE_AIR_QUALITY=False, WEATHER_REFRESH_SECONDS=1800)
    return ServerFeedService(DataCoordinator(provider), provider, fetch_air_quality=lambda *a, **k: None,
                             settings=settings, history_path=str(tmp_path / "aq.json"),
                             state_path=str(tmp_path / "state.json"), clock=clock, wall_clock=clock), provider


def test_restart_restores_data_and_revisions(tmp_path):
    clock = Clock()
    first, _ = service(tmp_path, clock)
    first.refresh({"weather1", "cubs last"})
    revisions = {s: first.data_revision(s) for s in ("weather1", "cubs last")}
    weather = first.data.snapshot().values["weather"]

    clock.now += 60
    second, provider = service(tmp_path, clock)
    assert second.data.snapshot().values["weather"] == weather
    assert {s: second.data_revision(s) for s in revisions} == revisions
    # Feeds refresh on their normal schedule, not all at once after a restart.
    assert second.refresh({"weather1", "cubs last"}) == {}
    assert provider.calls == []
    clock.now += feeds.FEED_REFRESH_INTERVALS["cubs"]
    assert "cubs" in second.refresh({"weather1", "cubs last"})


def test_long_outage_refreshes_expired_feeds_at_start(tmp_path):
    clock = Clock()
    first, _ = service(tmp_path, clock)
    first.refresh({"weather1"})
    clock.now += 100_000
    second, _ = service(tmp_path, clock)
    assert second.refresh({"weather1"}) == {"weather": True}


def test_restored_server_renders_before_its_first_refresh(tmp_path):
    clock = Clock()
    first, _ = service(tmp_path, clock)
    first.refresh({"weather1"})
    second, provider = service(tmp_path, clock)
    rendering = ServerRendering(second.data, StyleRevision([]), feeds=second, preferences="p-1",
                                logos=ProfileLogos())
    revision = rendering.revisions(["weather1"])["weather1"]
    assert revision.data_revision == first.data_revision("weather1")
    assert provider.calls == []


def test_failed_refresh_keeps_the_saved_value(tmp_path):
    clock = Clock()
    first, provider = service(tmp_path, clock)
    first.refresh({"weather1"})
    provider.read_weather = lambda **kwargs: None
    clock.now += 100_000
    assert first.refresh({"weather1"}) == {"weather": False}
    saved = FeedStateFile(tmp_path / "state.json").load()
    assert saved["weather"]["value"] == {"current": {"temp": 70}}
