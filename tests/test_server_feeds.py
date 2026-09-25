"""Tests for headless feed collection on the render server."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from PIL import Image

from display_profiles import PROFILE_PRESETS
from services import feeds
from services.air_quality import AirQualityReport
from services.data_coordinator import DataCoordinator
from services.server_feeds import LIVE_REFRESH_SECONDS, ServerFeedService


class Clock:
    def __init__(self):
        self.now = 10_000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeProvider:
    """Stands in for services.data_provider: counts upstream reads."""

    def __init__(self):
        self.calls = []
        self.weather = {"current": {"temp": 70}}
        self.teams = {team: {"stand": {"team": team}} for team in feeds.LIVE_TEAM_SCREEN_TO_FEED.values()}
        self.scoreboards = {"scoreboards": {"mlb": [{"id": 1}]}, "scoreboard_metadata": {"mlb": {"stale": False}}}

    def read(self, key, fetcher, *, ttl_seconds=300, force=False):
        self.calls.append((key, ttl_seconds, force))
        team = key.split(":", 1)[1]
        return dict(self.teams.get(team, {"stand": {"team": team}}))

    def read_weather(self, *, ttl_seconds=300):
        self.calls.append(("weather", ttl_seconds, False))
        return self.weather

    def read_sports_payloads(self, *, ttl_seconds=120, leagues=None, force_refresh_leagues=None):
        self.calls.append(("sports", ttl_seconds, tuple(sorted(leagues or ())), tuple(sorted(force_refresh_leagues or ()))))
        return self.scoreboards


def settings(**overrides):
    values = dict(
        ENABLE_WEATHER=True, ENABLE_AIR_QUALITY=True, AIR_QUALITY_LATITUDE=41.9, AIR_QUALITY_LONGITUDE=-87.6,
        AIRNOW_API_KEY="k", AIR_QUALITY_ENABLE_POLLEN=False, WEATHER_REFRESH_SECONDS=1800,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def env(tmp_path):
    clock = Clock()
    provider = FakeProvider()
    data = DataCoordinator(provider)
    reports = []

    def fetch_air_quality(lat, lon, *, api_key=None, include_pollen=False):
        reports.append((lat, lon))
        return AirQualityReport(aqi_value=40, aqi_category="Good", primary_pollutant="O3",
                                us_aqi_pm2_5=10 + len(reports), us_aqi_pm10=5, us_aqi_ozone=40)

    service = ServerFeedService(
        data, provider, fetch_air_quality=fetch_air_quality, settings=settings(),
        history_path=str(tmp_path / "aq.json"), clock=clock, wall_clock=clock,
    )
    return SimpleNamespace(clock=clock, provider=provider, data=data, service=service, reports=reports, tmp=tmp_path)


def feeds_called(provider):
    names = set()
    for call in provider.calls:
        names.add("scoreboards" if call[0] == "sports" else call[0].replace("team:", ""))
    return names


# ── Selection and scheduling ────────────────────────────────────────────────


def test_seeds_the_standalone_cache_shape(env):
    values = env.data.snapshot().values
    assert set(values) == set(feeds.default_cache())
    assert values["cubs"]["schedule_covers_today"] is False


def test_only_demanded_and_enabled_feeds_refresh(env):
    results = env.service.refresh({"weather1", "cubs last", "date"})
    assert results == {"weather": True, "cubs": True}
    assert feeds_called(env.provider) == {"weather", "cubs"}
    env.service.settings.ENABLE_WEATHER = False
    assert env.service.required_feeds({"weather1", "air quality"}) == {"air_quality"}


def test_feeds_refresh_on_their_intervals(env):
    env.service.refresh({"cubs last", "hawks last"})
    env.provider.calls.clear()
    assert env.service.refresh({"cubs last", "hawks last"}) == {}
    env.clock.advance(feeds.FEED_REFRESH_INTERVALS["hawks"])
    assert env.service.refresh({"cubs last", "hawks last"}) == {"hawks": True}
    env.clock.advance(1800)
    assert set(env.service.refresh({"cubs last", "hawks last"})) == {"cubs", "hawks"}


def test_force_refreshes_every_demanded_feed(env):
    env.service.refresh({"weather1"})
    assert env.service.refresh({"weather1"}, force=True) == {"weather": True}


def test_live_team_screen_refreshes_fresh_on_the_live_interval(env):
    env.service.refresh({"cubs live"})
    assert env.provider.calls[-1] == ("team:cubs", 120, True)
    env.provider.calls.clear()
    env.clock.advance(LIVE_REFRESH_SECONDS)
    assert env.service.refresh({"cubs live"}) == {"cubs": True}


def test_scoreboards_live_window_and_date_rollover(env, monkeypatch):
    screens = {"MLB Scoreboard", "NFL Scoreboard"}
    env.service.refresh(screens)
    assert env.provider.calls[-1] == ("sports", 120, ("mlb", "nfl"), ())
    values = env.data.snapshot().values
    assert values["scoreboards"]["mlb"] == ({"id": 1},) and "nfl" in values["scoreboards"]
    env.provider.calls.clear()
    env.clock.advance(3600)
    assert env.service.refresh(screens) == {}  # daily interval, no live games, same date
    monkeypatch.setattr(feeds, "scoreboard_date_for_league", lambda league, now=None: "tomorrow")
    assert env.service.refresh(screens) == {"scoreboards": True}
    monkeypatch.setattr(feeds, "scoreboards_in_live_window", lambda scoreboards, now=None: True)
    env.clock.advance(LIVE_REFRESH_SECONDS)
    env.service.refresh(screens)
    assert env.provider.calls[-1] == ("sports", 0, ("mlb", "nfl"), ("nfl",))


# ── Revisions ───────────────────────────────────────────────────────────────


def test_data_revisions_follow_only_the_screens_feeds(env):
    env.service.refresh({"weather1", "cubs last", "MLB Scoreboard"})
    weather = env.service.data_revision("weather1")
    cubs = env.service.data_revision("cubs last")
    scoreboard = env.service.data_revision("MLB Scoreboard")
    assert env.service.data_revision("date") is None
    env.clock.advance(1800)
    env.service.refresh({"weather1"})
    assert env.service.data_revision("weather1") != weather
    assert env.service.data_revision("weather quad") != weather  # weather + air quality
    assert env.service.data_revision("cubs last") == cubs
    assert env.service.data_revision("MLB Scoreboard") == scoreboard


def test_feed_update_rerenders_only_dependent_screens(env, tmp_path):
    from remote_display.artifact_store import ArtifactStore
    from remote_display.models import ClientCapabilities, ClientDemand, PackageCapabilities
    from remote_display.registry import ClientRegistry
    from remote_display.render_coordinator import RenderCoordinator, RenderOutput
    from remote_display.server_rendering import ServerRendering, StyleRevision

    rendering = ServerRendering(env.data, StyleRevision([tmp_path / "none.json"]), feeds=env.service)
    rendered = []

    def renderer(key):
        rendered.append(key.screen_id)
        preset = PROFILE_PRESETS[key.render_profile]
        return RenderOutput(image=Image.new(preset.color_mode, (preset.width, preset.height), len(rendered)),
                            refresh_seconds=10_000)

    class Now:
        now = 1_800_000_000.0

        def __call__(self):
            return self.now

    now = Now()
    registry = ClientRegistry(lease_seconds=100_000, clock=now)
    preset = PROFILE_PRESETS["hyperpixel4"]
    registry.register(
        ClientCapabilities(protocol_version=1, client_software_version="0.2", client_id="office",
                           display_profile="hyperpixel4", logical_width=preset.width,
                           logical_height=preset.height, image_formats=("PNG",),
                           color_modes=(preset.color_mode,), render_package_versions=(1,)),
        ClientDemand(client_id="office", playlist_revision="pl-1",
                     required_screens=("weather1", "cubs last", "air quality"),
                     package_capabilities=PackageCapabilities(render_package_versions=(1,), image_formats=("PNG",)),
                     sync_interval_seconds=30),
    )
    store = ArtifactStore(tmp_path / "artifacts", clock=now)

    class Inline:
        def submit(self, fn):
            from concurrent.futures import Future

            future = Future()
            try:
                future.set_result(fn())
            except Exception as exc:  # noqa: BLE001
                future.set_exception(exc)
            return future

    coordinator = RenderCoordinator(registry, store, renderer, rendering.revisions, executor=Inline(), clock=now)
    env.service.refresh({"weather1", "cubs last", "air quality"})
    coordinator.tick()
    assert sorted(rendered) == ["air quality", "cubs last", "weather1"]
    rendered.clear()
    # Only weather changes now: only the weather screen rerenders.
    env.clock.advance(feeds.FEED_REFRESH_INTERVALS["weather"])
    env.service.refresh({"weather1"})
    now.now += 60
    coordinator.tick()
    assert rendered == ["weather1"]


# ── Failures and health ─────────────────────────────────────────────────────


def test_failed_refresh_keeps_last_good_data(env):
    env.service.refresh({"weather1"})
    good = env.data.snapshot().values["weather"]
    revision = env.service.data_revision("weather1")
    env.provider.weather = None
    env.clock.advance(1800)
    assert env.service.refresh({"weather1"}) == {"weather": False}
    assert env.data.snapshot().values["weather"] == good
    assert env.service.data_revision("weather1") == revision
    health = env.service.health()["weather"]
    assert health["consecutive_failures"] == 1 and "no data" in health["last_error"]
    # A failing feed retries on its interval from the last attempt, not in a tight loop.
    assert env.service.refresh({"weather1"}) == {}
    env.provider.weather = {"current": {"temp": 72}}
    env.clock.advance(feeds.FEED_REFRESH_INTERVALS["weather"])
    assert env.service.refresh({"weather1"}) == {"weather": True}
    assert env.service.health()["weather"]["consecutive_failures"] == 0


def test_one_failing_feed_does_not_stop_others(env):
    def boom(*args, **kwargs):
        raise RuntimeError("provider down")

    env.provider.read_weather = boom
    assert env.service.refresh({"weather1", "bears stand1"}) == {"weather": False, "bears": True}


def test_air_quality_history_is_kept_and_persisted(env):
    env.service.refresh({"air quality"})
    env.clock.advance(feeds.FEED_REFRESH_INTERVALS["air_quality"])
    env.service.refresh({"air quality"})
    report = env.data.snapshot().values["air_quality"]
    assert [sample[1] for sample in report.component_history] == [11, 12]
    persisted = json.loads((env.tmp / "aq.json").read_text(encoding="utf-8"))["history"]
    assert len(persisted) == 2


def test_air_quality_requires_coordinates(env):
    env.service.settings.AIR_QUALITY_LATITUDE = None
    assert env.service.refresh({"air quality"}) == {"air_quality": False}


def test_health_reports_staleness(env):
    assert env.service.health()["weather"]["stale"] is True
    env.service.refresh({"weather1"})
    assert env.service.health()["weather"]["stale"] is False
    env.clock.advance(feeds.FEED_REFRESH_INTERVALS["weather"] * 2 + 1)
    assert env.service.health()["weather"]["stale"] is True


# ── Standalone parity ───────────────────────────────────────────────────────


def test_standalone_main_uses_the_shared_catalog():
    import main

    assert main._FEED_DEPENDENCIES is feeds.FEED_DEPENDENCIES
    assert main._FEED_REFRESH_INTERVALS is feeds.FEED_REFRESH_INTERVALS
    assert main._SCOREBOARD_SCREEN_TO_LEAGUES is feeds.SCOREBOARD_SCREEN_TO_LEAGUES
    assert main._STARTUP_CRITICAL_FEEDS == feeds.STARTUP_CRITICAL_FEEDS
    assert main._scoreboards_in_live_window is feeds.scoreboards_in_live_window
    assert set(main.cache) == set(feeds.default_cache())


def test_every_feed_has_a_server_refresher(env):
    for feed in feeds.FEED_DEPENDENCIES:
        env.clock.advance(100_000)
        screen = next(iter(feeds.FEED_DEPENDENCIES[feed]))
        assert env.service.refresh({screen}).get(feed) is True, feed
