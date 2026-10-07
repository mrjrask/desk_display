"""Tests for the remote temp monitor screen: MMM-RemoteTempMonitor parsing, caching and drawing."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from screens import draw_remote_temps
from services import feeds, remote_temps
from services.data_coordinator import DataCoordinator
from services.server_feeds import ServerFeedService

FIXTURE = Path(__file__).parent / "fixtures" / "remote_temp_monitor_snapshot.json"
NOW = 1_000_000.0


def raw_snapshot():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def payload(raw=None, fetched_at=NOW):
    return remote_temps.parse_snapshot(raw_snapshot() if raw is None else raw, fetched_at=fetched_at)


@pytest.fixture(autouse=True)
def _empty_cache():
    remote_temps.clear_cache()
    yield
    remote_temps.clear_cache()


# ── Endpoint ────────────────────────────────────────────────────────────────


def test_endpoint_defaults_to_the_magicmirror_aggregate_port():
    assert remote_temps.endpoint_url({}) == "http://192.168.1.201:9877/temps"


def test_endpoint_host_and_port_are_configurable():
    env = {"REMOTE_TEMP_MONITOR_HOST": "mirror.lan", "REMOTE_TEMP_MONITOR_PORT": "9900"}
    assert remote_temps.endpoint_url(env) == "http://mirror.lan:9900/temps"
    assert remote_temps.endpoint_url({"REMOTE_TEMP_MONITOR_PORT": "nope"}) == "http://192.168.1.201:9877/temps"
    assert remote_temps.endpoint_url({"REMOTE_TEMP_MONITOR_HOST": "fd00::5"}) == "http://[fd00::5]:9877/temps"


# ── Parsing ─────────────────────────────────────────────────────────────────


def test_snapshot_devices_are_parsed():
    data = payload()
    assert len(data["devices"]) == 6
    square = next(device for device in data["devices"] if device["hostname"] == "square")
    assert square == {"id": "192.168.1.42:square", "hostname": "square", "celsius": 61.3, "fahrenheit": 142.3,
                      "pi_model": "5", "pi_ram": "8GB", "ip": "192.168.1.42"}
    assert data["updated_at"] == "2026-10-07T18:00:00.000Z"


def test_aliases_and_missing_fahrenheit_are_accepted():
    raw = {"temps": [{"name": "alias", "temperature": {"celsius": 50}}, {"hostname": "c", "temp_c": "40.5"}]}
    devices = payload(raw)["devices"]
    assert [(d["hostname"], d["celsius"], d["fahrenheit"]) for d in devices] == [
        ("alias", 50.0, 122.0), ("c", 40.5, 104.9)]


def test_unusable_devices_are_skipped():
    raw = {"devices": [{"hostname": "ok", "celsius": 40}, {"hostname": "nan", "celsius": "NaN"},
                       {"celsius": 40}, {"hostname": "hot", "celsius": 400}, "junk", {"hostname": "t", "celsius": True}]}
    assert [d["hostname"] for d in payload(raw)["devices"]] == ["ok"]


def test_an_empty_snapshot_is_good_data():
    data = payload({"type": "temperature_snapshot", "count": 0, "updatedAt": None, "devices": []})
    assert data["devices"] == [] and remote_temps.select(data, now=NOW)["rows"] == []


@pytest.mark.parametrize("raw", [None, [], "nope", {}, {"devices": "x"}, {"error": "not_found"}])
def test_malformed_snapshots_are_rejected(raw):
    with pytest.raises(remote_temps.RemoteTempsError):
        remote_temps.parse_snapshot(raw)


# ── Presentation ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("celsius,level", [
    (42.0, "normal"), (59.9, "normal"), (60.0, "warm"), (69.9, "warm"), (70.0, "hot"),
    (80.0, "very_hot"), (84.9, "very_hot"), (85.0, "critical"), (99.0, "critical"),
])
def test_levels_follow_the_modules_default_thresholds(celsius, level):
    assert remote_temps.level_of(celsius) == level


def test_rows_are_hottest_first_and_named_like_the_module():
    report = remote_temps.select(payload(), now=NOW)
    assert [row["name"] for row in report["rows"]] == [
        "adsb (3B+ | 1GB)", "magicmirror (4 | 2GB)", "square (5 | 8GB)", "office-mac",
        "hyper (4 | 4GB)", "pihole (Zero 2 W)"]
    assert [row["level"] for row in report["rows"]] == [
        "critical", "hot", "warm", "normal", "normal", "normal"]
    assert not report["stale"]


def test_old_data_is_marked_cached():
    stale = remote_temps.select(payload(), now=NOW + remote_temps.STALE_AFTER_SECONDS + 120)
    assert stale["stale"] and stale["age_minutes"] == 5
    assert draw_remote_temps.freshness_text(stale) == "Cached · 5 min old"


# ── Cache ───────────────────────────────────────────────────────────────────


class _Clock:
    def __init__(self):
        self.now = 500.0

    def __call__(self):
        return self.now


def test_one_fetch_serves_every_caller_until_the_refresh_interval():
    clock, calls = _Clock(), []

    def download():
        calls.append(1)
        return raw_snapshot()

    first = remote_temps.fetch_snapshot(download=download, clock=clock)
    clock.now += remote_temps.REFRESH_SECONDS - 1
    assert remote_temps.fetch_snapshot(download=download, clock=clock) is first
    clock.now += 1
    remote_temps.fetch_snapshot(download=download, clock=clock)
    assert len(calls) == 2


def test_an_unreachable_magicmirror_keeps_the_last_good_snapshot():
    clock = _Clock()
    good = remote_temps.fetch_snapshot(download=raw_snapshot, clock=clock)
    clock.now += remote_temps.REFRESH_SECONDS

    def offline():
        raise OSError("connection refused")

    with pytest.raises(remote_temps.RemoteTempsError):
        remote_temps.fetch_snapshot(download=offline, clock=clock)
    assert remote_temps.get_snapshot(download=offline, clock=clock) is good
    assert remote_temps.get_snapshot(download=lambda: {"error": "not_found"}, clock=clock) is good
    remote_temps.clear_cache()
    assert remote_temps.get_snapshot(download=offline, clock=clock) is None


# ── Server feed ─────────────────────────────────────────────────────────────


def test_the_server_fetches_temperatures_once_a_minute_for_all_displays():
    assert feeds.feeds_for_screen("remote temp monitor", feeds.SERVER_FEED_DEPENDENCIES) == {"remote_temps"}
    assert feeds.SERVER_FEED_REFRESH_INTERVALS["remote_temps"] == 60
    clock, results = _Clock(), [payload(), OSError("offline")]

    def fetch(*, force=False):
        value = results.pop(0)
        if isinstance(value, Exception):
            raise remote_temps.RemoteTempsError(str(value))
        return value

    service = ServerFeedService(
        DataCoordinator(SimpleNamespace()), SimpleNamespace(), fetch_air_quality=lambda *a, **k: None,
        settings=SimpleNamespace(ENABLE_WEATHER=False, ENABLE_AIR_QUALITY=False),
        standings_fetchers={}, history_path="/nonexistent/aq.json", clock=clock, wall_clock=clock,
        fetch_traffic=lambda **k: None, fetch_remote_temps=fetch,
    )
    assert service.refresh({"remote temp monitor"}) == {"remote_temps": True}
    good = service.data.snapshot().values["remote_temps"]
    assert service.refresh({"remote temp monitor"}) == {}
    clock.now += 60
    assert service.refresh({"remote temp monitor"}) == {"remote_temps": False}
    assert service.data.snapshot().values["remote_temps"] == good


# ── Drawing ─────────────────────────────────────────────────────────────────

SIZES = [(320, 240), (240, 135), (800, 480), (720, 720), (1920, 1080), (128, 64), (1280, 720)]


@pytest.mark.parametrize("width,height", SIZES)
def test_every_display_size_draws(width, height):
    image = draw_remote_temps.compose_remote_temps_image(payload(), width=width, height=height, now=NOW)
    assert image.size == (width, height)
    colours = {colour for _count, colour in image.getcolors(width * height)}
    assert draw_remote_temps.LEVEL_COLORS["normal"] in colours
    assert draw_remote_temps.LEVEL_COLORS["critical"] in colours


@pytest.mark.parametrize("width,height", SIZES)
def test_empty_unavailable_stale_and_crowded_states_draw(width, height):
    for data, now in ((None, NOW), (payload({"devices": []}), NOW), (payload(), NOW + 3600)):
        image = draw_remote_temps.compose_remote_temps_image(data, width=width, height=height, now=now)
        assert image.size == (width, height) and image.getbbox() is not None
    crowded = {"devices": [{"hostname": f"pi-{n:02d}", "celsius": 40 + n} for n in range(40)]}
    image = draw_remote_temps.compose_remote_temps_image(payload(crowded), width=width, height=height, now=NOW)
    assert image.size == (width, height)


def test_server_renders_from_the_snapshot_without_fetching(monkeypatch):
    from display_profiles import DISPLAY_PROFILE_HYPERPIXEL4, PROFILE_PRESETS
    from rendering.screen_renderer import ScreenRenderer, ServerPreferenceSnapshot

    monkeypatch.setattr(remote_temps, "_download", lambda *a, **k: pytest.fail("fetched upstream"))
    seen = []
    original = draw_remote_temps.compose_remote_temps_image

    def spy(data, **kwargs):
        seen.append(data)
        return original(data, **kwargs)

    monkeypatch.setattr(draw_remote_temps, "compose_remote_temps_image", spy)
    data = payload()
    snapshot = DataCoordinator().publish("remote_temps", data)
    artifact = ScreenRenderer().render(
        "remote temp monitor", PROFILE_PRESETS[DISPLAY_PROFILE_HYPERPIXEL4], ServerPreferenceSnapshot(revision=1),
        snapshot,
    )
    assert artifact.image.size == (800, 480)
    assert seen and seen[-1]["devices"] == data["devices"]
