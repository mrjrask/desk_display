"""Tests for the traffic screen: Travel Midwest parsing, direction selection, caching and drawing."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from screens import draw_traffic
from services import feeds, traffic
from services.data_coordinator import DataCoordinator
from services.server_feeds import ServerFeedService

FIXTURE = Path(__file__).parent / "fixtures" / "travel_midwest_quick_traffic.json"
NOW = 1_000_000.0
INBOUND_KEYS = ["edens_lakecook_jane_byrne", "edens_lakecook_montrose",
                "kennedy_reversible_inbound", "kennedy_montrose_jane_byrne"]
OUTBOUND_KEYS = ["edens_jane_byrne_lakecook", "edens_montrose_lakecook",
                 "kennedy_reversible_outbound", "kennedy_jane_byrne_montrose"]


def raw_report():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def payload(raw=None, fetched_at=NOW):
    return traffic.parse_report(raw_report() if raw is None else raw, fetched_at=fetched_at)


def keys(report):
    return [row["key"] for group in report["groups"] for row in group["rows"]]


@pytest.fixture(autouse=True)
def _empty_cache():
    traffic.clear_cache()
    yield
    traffic.clear_cache()


# ── Matching ────────────────────────────────────────────────────────────────


def test_all_eight_segments_match_by_exact_description():
    data = payload()
    assert set(data["segments"]) == {segment.key for segment in traffic.SEGMENTS}
    assert data["missing"] == []
    assert data["age_minutes"] == 3
    edens = data["segments"]["edens_lakecook_montrose"]
    assert (edens["travel_time"], edens["speed"], edens["over"]) == (60, 15, True)
    assert edens["ids"] == ["IL-TESTTSC-167"] and edens["id"] == "IL-TESTTSC-167"
    assert edens["short_description"] == "IB Lake Cook to Montrose"


def test_matching_ignores_array_position_and_lookalike_descriptions():
    raw = raw_report()
    for group in raw[1]:
        group["rows"].reverse()
    raw[1].reverse()
    assert payload(raw)["segments"] == payload()["segments"]
    # "Inbound Kennedy from Ohio to I-290..." and the Deerfield Edens row are not selected.
    assert all(row["description"] in traffic.SEGMENTS_BY_DESCRIPTION for row in payload()["segments"].values())


def test_closed_reversible_and_missing_speed_are_not_available():
    data = payload()["segments"]
    closed = data["kennedy_reversible_inbound"]  # travelTime -16, speed 0
    assert closed["travel_time"] is None and closed["speed"] is None
    open_ = data["kennedy_reversible_outbound"]  # travelTime 28, speed 0
    assert open_["travel_time"] == 28 and open_["speed"] is None


def test_a_missing_segment_is_kept_as_not_available():
    raw = raw_report()
    raw[1][1]["rows"] = [r for r in raw[1][1]["rows"] if r["description"] != "Inbound Edens from Lake Cook to Montrose"]
    data = payload(raw)
    assert data["missing"] == ["edens_lakecook_montrose"]
    report = traffic.select(data, traffic.INBOUND, now=NOW)
    assert keys(report) == INBOUND_KEYS
    assert report["groups"][0]["rows"][1]["status"] == traffic.UNAVAILABLE


@pytest.mark.parametrize("raw", [
    None, {}, "nope", [], [{"ageInMinutes": "1"}], [{"ageInMinutes": "1"}, {"rows": []}],
    [{"ageInMinutes": "1"}, [{"caption": "Other", "rows": [{"description": "Inbound Ike", "travelTime": 5}]}]],
])
def test_malformed_reports_are_rejected(raw):
    with pytest.raises(traffic.TrafficFeedError):
        traffic.parse_report(raw)


# ── Direction selection ─────────────────────────────────────────────────────


def test_inbound_and_outbound_select_their_four_segments_grouped_by_road():
    data = payload()
    inbound = traffic.select(data, traffic.INBOUND, now=NOW)
    outbound = traffic.select(data, traffic.OUTBOUND, now=NOW)
    assert keys(inbound) == INBOUND_KEYS
    assert keys(outbound) == OUTBOUND_KEYS
    assert [group["road"] for group in inbound["groups"]] == ["Edens", "Kennedy"]
    assert [group["road"] for group in outbound["groups"]] == ["Edens", "Kennedy"]
    assert inbound["direction"] == "inbound" and outbound["direction"] == "outbound"


@pytest.mark.parametrize("client_id,direction", [
    ("hyper", "outbound"), ("Hyper", "outbound"), ("hyper-panel", "outbound"),
    ("square", "inbound"), ("square-panel", "inbound"), ("hyperion", "inbound"), ("office", "inbound"),
])
def test_only_hyper_shows_outbound(client_id, direction):
    assert traffic.direction_for_display(client_id, env={}) == direction


def test_outbound_displays_can_be_configured():
    env = {"TRAFFIC_OUTBOUND_DISPLAYS": "office, den"}
    assert traffic.direction_for_display("den", env) == "outbound"
    assert traffic.direction_for_display("hyper", env) == "inbound"


def test_only_an_outbound_display_gets_the_outbound_render_scope():
    screens = ["date", "traffic", "weather1"]
    assert traffic.screen_scopes("hyper", screens, env={}) == {"traffic": traffic.OUTBOUND_SCOPE}
    assert traffic.screen_scopes("square-panel", screens, env={}) == {}
    assert traffic.screen_scopes("hyper", ["date"], env={}) == {}
    assert traffic.direction_for_scope(traffic.OUTBOUND_SCOPE) == "outbound"
    assert traffic.direction_for_scope(None) == "inbound"


def test_standalone_direction_follows_setting_then_host_name():
    assert traffic.local_direction({"TRAFFIC_DIRECTION": "outbound"}, hostname="square") == "outbound"
    assert traffic.local_direction({"TRAFFIC_DIRECTION": "inbound"}, hostname="hyper") == "inbound"
    assert traffic.local_direction({}, hostname="hyper") == "outbound"
    assert traffic.local_direction({"TRAFFIC_DIRECTION": "auto"}, hostname="square") == "inbound"


# ── Status ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("row,status", [
    ({"travel_time": 20, "over": False}, traffic.NORMAL),
    ({"travel_time": 20, "over": True}, traffic.HEAVY),
    ({"travel_time": 20, "over": False, "congestion": "Medium"}, traffic.ELEVATED),
    ({"travel_time": 20, "over": False, "congestion": "heavy"}, traffic.HEAVY),
    ({"travel_time": None, "over": True}, traffic.UNAVAILABLE),
])
def test_status_is_led_by_travel_midwests_over_flag(row, status):
    assert traffic.status_of(row) == status


# ── Cache and stale data ────────────────────────────────────────────────────


class _Clock:
    def __init__(self):
        self.now = 500.0

    def __call__(self):
        return self.now


def test_one_fetch_serves_every_caller_until_the_refresh_interval():
    clock, calls = _Clock(), []

    def download():
        calls.append(1)
        return raw_report()

    first = traffic.fetch_report(download=download, clock=clock)
    clock.now += traffic.REFRESH_SECONDS - 1
    assert traffic.fetch_report(download=download, clock=clock) is first
    assert len(calls) == 1
    clock.now += 1
    traffic.fetch_report(download=download, clock=clock)
    assert len(calls) == 2


def test_failed_or_malformed_refresh_keeps_the_last_good_report():
    clock = _Clock()
    good = traffic.fetch_report(download=raw_report, clock=clock)
    clock.now += traffic.REFRESH_SECONDS

    def broken():
        raise OSError("offline")

    with pytest.raises(traffic.TrafficFeedError):
        traffic.fetch_report(download=broken, clock=clock)
    assert traffic.get_report(download=broken, clock=clock) is good
    assert traffic.get_report(download=lambda: {"error": "bad"}, clock=clock) is good


def test_no_report_yet_gives_none():
    assert traffic.get_report(download=lambda: None, clock=_Clock()) is None
    assert traffic.select(None, traffic.INBOUND) is None


def test_cached_data_reports_its_age_and_staleness():
    data = payload(fetched_at=NOW)
    fresh = traffic.select(data, traffic.INBOUND, now=NOW + 60)
    assert fresh["age_minutes"] == 4 and not fresh["stale"]
    stale = traffic.select(data, traffic.INBOUND, now=NOW + traffic.STALE_AFTER_SECONDS + 120)
    assert stale["stale"] and stale["age_minutes"] == 3 + 12
    assert draw_traffic.freshness_text(stale).startswith("Cached · Travel Midwest")
    assert "15 min old" in draw_traffic.freshness_text(stale)


# ── Server feed ─────────────────────────────────────────────────────────────


def _service(fetch, clock):
    return ServerFeedService(
        DataCoordinator(SimpleNamespace()), SimpleNamespace(), fetch_air_quality=lambda *a, **k: None,
        settings=SimpleNamespace(ENABLE_WEATHER=False, ENABLE_AIR_QUALITY=False),
        standings_fetchers={}, history_path="/nonexistent/aq.json", clock=clock, wall_clock=clock,
        fetch_traffic=fetch,
    )


def test_the_server_fetches_traffic_once_for_all_displays_every_five_minutes():
    assert feeds.feeds_for_screen("traffic", feeds.SERVER_FEED_DEPENDENCIES) == {"traffic"}
    assert feeds.SERVER_FEED_REFRESH_INTERVALS["traffic"] == 300
    clock, calls = _Clock(), []

    def fetch(*, force=False):
        calls.append(force)
        return payload()

    service = _service(fetch, clock)
    assert service.refresh({"traffic"}) == {"traffic": True}
    assert service.refresh({"traffic"}) == {}
    clock.now += 300
    assert service.refresh({"traffic"}) == {"traffic": True}
    assert len(calls) == 2
    assert service.data_revision("traffic").startswith("f-")


def test_a_failed_server_refresh_keeps_the_last_good_report():
    clock, results = _Clock(), [payload()]

    def fetch(*, force=False):
        value = results.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    service = _service(fetch, clock)
    service.refresh({"traffic"})
    good = service.data.snapshot().values["traffic"]
    results.append(traffic.TrafficFeedError("malformed"))
    clock.now += 300
    assert service.refresh({"traffic"}) == {"traffic": False}
    assert service.data.snapshot().values["traffic"] == good


# ── Drawing ─────────────────────────────────────────────────────────────────

SIZES = [(320, 240), (240, 135), (800, 480), (720, 720), (1920, 1080), (128, 64), (1280, 720)]


@pytest.mark.parametrize("width,height", SIZES)
@pytest.mark.parametrize("direction", traffic.DIRECTIONS)
def test_every_display_size_draws_both_directions(width, height, direction):
    image = draw_traffic.compose_traffic_image(payload(), direction, width=width, height=height, now=NOW)
    assert image.size == (width, height)
    assert image.getbbox() is not None


@pytest.mark.parametrize("width,height", SIZES)
def test_unavailable_and_stale_states_draw(width, height):
    blank = draw_traffic.compose_traffic_image(None, "inbound", width=width, height=height)
    assert blank.size == (width, height) and blank.getbbox() is not None
    stale = draw_traffic.compose_traffic_image(payload(), "inbound", width=width, height=height,
                                               now=NOW + 3600)
    assert stale.size == (width, height)


def test_status_colours_reach_the_screen():
    image = draw_traffic.compose_traffic_image(payload(), "inbound", width=800, height=480, now=NOW)
    colours = {colour for _count, colour in image.getcolors(800 * 480)}
    assert draw_traffic.STATUS_COLORS[traffic.HEAVY] in colours  # Lake Cook → Montrose is over
    assert draw_traffic.STATUS_COLORS[traffic.UNAVAILABLE] in colours  # closed reversible
    assert draw_traffic.STATUS_COLORS[traffic.NORMAL] in colours


def test_server_renders_from_the_snapshot_without_fetching(monkeypatch):
    from display_profiles import DISPLAY_PROFILE_HYPERPIXEL4, PROFILE_PRESETS
    from rendering.screen_renderer import ScreenRenderer, ServerPreferenceSnapshot

    monkeypatch.setattr(traffic, "_download", lambda *a, **k: pytest.fail("fetched upstream"))
    seen = []
    original = draw_traffic.compose_traffic_image

    def spy(data, direction, **kwargs):
        seen.append((data, direction))
        return original(data, direction, **kwargs)

    monkeypatch.setattr(draw_traffic, "compose_traffic_image", spy)
    data = payload()
    coordinator = DataCoordinator()
    coordinator.publish("traffic", data)
    snapshot = coordinator.publish("traffic_direction", "outbound")
    artifact = ScreenRenderer().render(
        "traffic", PROFILE_PRESETS[DISPLAY_PROFILE_HYPERPIXEL4], ServerPreferenceSnapshot(revision=1), snapshot,
    )
    assert artifact.image.size == (800, 480)
    assert seen and seen[-1][1] == "outbound" and seen[-1][0]["segments"] == data["segments"]


@pytest.mark.parametrize("scope,direction", [(None, "inbound"), (traffic.OUTBOUND_SCOPE, "outbound")])
def test_compose_screen_gives_the_render_its_scopes_direction(monkeypatch, scope, direction):
    from remote_display import server_rendering

    captured = {}

    class _Renderer:
        def render(self, screen_id, profile, preferences, snapshot, record_frames=False):
            captured["direction"] = snapshot.values.get("traffic_direction")
            from PIL import Image

            return SimpleNamespace(image=Image.new("RGB", (profile.width, profile.height)), metadata={})

    import rendering.screen_renderer as screen_renderer

    monkeypatch.setattr(screen_renderer, "ScreenRenderer", _Renderer)
    monkeypatch.setattr("rendering.packaging.build_package", lambda *a, **k: None)
    key = SimpleNamespace(screen_id="traffic", client_scope=scope)
    from display_profiles import DISPLAY_PROFILE_DISPLAY_HAT_MINI, PROFILE_PRESETS

    snapshot = DataCoordinator().publish("traffic", copy.deepcopy(payload()))
    server_rendering.compose_screen(key, PROFILE_PRESETS[DISPLAY_PROFILE_DISPLAY_HAT_MINI], snapshot,
                                    SimpleNamespace(for_size=lambda w, h: {}))
    assert captured["direction"] == direction


def test_a_downtown_located_display_shows_outbound_whatever_its_id():
    hyper_site = SimpleNamespace(latitude=41.9037, longitude=-87.6357)
    home = SimpleNamespace(latitude=42.1373, longitude=-87.8446)
    assert traffic.direction_for_display("den", {}, hyper_site) == "outbound"
    assert traffic.direction_for_display("den", {}, home) == "inbound"
    assert traffic.direction_for_display("den", {}, None) == "inbound"
    assert traffic.screen_scopes("den", ["traffic"], env={}, location=hyper_site) == {
        "traffic": traffic.OUTBOUND_SCOPE}
