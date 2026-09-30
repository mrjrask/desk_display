"""Per-display weather locations: storage, feeds, render keys, manifests and drawing."""
from __future__ import annotations

import datetime
from concurrent.futures import Future
from types import MappingProxyType

import pytest

from display_profiles import PROFILE_PRESETS
from remote_display import locations
from remote_display.locations import Location, LocationError
from remote_display.models import ScreenRevisions

NEW_YORK = Location(40.7128, -74.006)


# ── Locations ───────────────────────────────────────────────────────────────


def test_location_parsing_and_scopes():
    assert Location.parse(None, "") is None
    parsed = Location.parse("40.71284", " -74.00601 ")
    assert parsed == NEW_YORK
    assert parsed.scope == "loc-40.7128_-74.0060"
    assert Location.from_scope(parsed.scope) == parsed
    assert Location.from_scope("office") is None and Location.from_scope(None) is None
    # -0.0 and 0.0 are the same place.
    assert Location.parse("-0.00001", 0).scope == Location.parse(0, 0).scope == "loc-0.0000_0.0000"
    for latitude, longitude, field in ((91, 0, "latitude"), (0, -181, "longitude"), ("north", 0, "latitude"),
                                       (True, 0, "latitude"), (float("nan"), 0, "latitude"),
                                       (40, "", "longitude"), ("", -74, "latitude")):
        with pytest.raises(LocationError) as caught:
            Location.parse(latitude, longitude)
        assert caught.value.field == field


def test_location_screens_are_the_weather_and_air_quality_screens():
    screens = locations.location_screens()
    assert {"weather1", "weather radar", "weather quad", "astronomical", "air quality"} <= screens
    assert "date" not in screens and "cubs last" not in screens
    assert locations.screen_scopes(None, ["weather1"]) == {}
    assert locations.screen_scopes(NEW_YORK, ["weather1", "date", "astronomical"]) == {
        "weather1": NEW_YORK.scope, "astronomical": NEW_YORK.scope}


def test_scoped_values_swap_in_the_locations_data():
    values = MappingProxyType({
        "weather": MappingProxyType({"current": {"temp": 70}}),
        f"weather@{NEW_YORK.scope}": MappingProxyType({"current": {"temp": 50}}),
        "air_quality": "chicago aqi",
        "cubs": "cubs",
    })
    scoped = locations.scoped_values(values, NEW_YORK.scope)
    assert scoped["weather"]["current"] == {"temp": 50}
    assert scoped["weather"]["location"] == {"latitude": 40.7128, "longitude": -74.006}
    assert scoped["air_quality"] is None  # not fetched yet: never another place's
    assert scoped["cubs"] == "cubs"
    assert locations.scoped_values(values, None)["weather"]["current"] == {"temp": 70}


# ── Store ───────────────────────────────────────────────────────────────────


def test_store_keeps_a_displays_own_location(tmp_path):
    from remote_display.playlist_store import PlaylistStore, PlaylistValidationError

    store = PlaylistStore(tmp_path / "playlists.json")
    assert store.location("hyper") is None and store.client_locations() == {}
    assert store.set_location("hyper", "40.7128", "-74.006", actor="jason") == NEW_YORK
    store.set_vertical_speed_adjustment("hyper", 0.5, actor="jason")
    assert store.location("hyper") == NEW_YORK
    assert store.client_locations() == {"hyper": NEW_YORK}
    assert store.snapshot()["clients"]["hyper"]["location"] == {"latitude": 40.7128, "longitude": -74.006}
    with pytest.raises(PlaylistValidationError):
        store.set_location("hyper", 100, 0, actor="jason")
    assert store.set_location("hyper", None, "", actor="jason") is None
    assert store.snapshot()["clients"]["hyper"] == {"vertical_speed_adjustment": 0.5}
    assert store.snapshot()["audit"][-1]["action"] == "set_location"


# ── Server: render keys, manifests and heartbeats ───────────────────────────


class Inline:
    def submit(self, fn):
        future = Future()
        try:
            future.set_result(fn())
        except Exception as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future


def test_displays_at_one_location_share_its_weather_renders(tmp_path):
    pytest.importorskip("flask")
    from PIL import Image

    import display_server
    from remote_display.registry import Assignment
    from remote_display.render_coordinator import RenderOutput
    from tests.test_display_server import ADMIN_TOKEN, SERVER_TOKEN, Clock, bearer, caps, demand, registered, status

    clock = Clock()
    rendered = []

    def renderer(key):
        rendered.append((key.screen_id, key.client_scope))
        preset = PROFILE_PRESETS[key.render_profile]
        color = 1 if key.client_scope else 0
        return RenderOutput(image=Image.new(preset.color_mode, (preset.width, preset.height), color),
                            refresh_seconds=10_000)

    def revisions(screens):
        return {s: ScreenRevisions("s1", "d-own", "r1") for s in screens}

    def scoped_revisions(pairs):
        return {pair: ScreenRevisions("s1", f"d-{pair[1]}", "r1") for pair in pairs}

    places = {"office": NEW_YORK, "den": NEW_YORK}
    config = display_server.DisplayServerConfig(enrollment="shared", auth_token=SERVER_TOKEN,
                                                admin_token=ADMIN_TOKEN, artifact_dir=tmp_path / "a")
    app = display_server.create_app(
        config, assignments=lambda _c: Assignment("default", "rev-9", ("date", "weather1")), clock=clock, renderer=renderer, revisions=revisions,
        scoped_revisions=scoped_revisions, render_executor=Inline(), client_locations=lambda: dict(places),
        display_status=lambda: {"weather": {"temp_f": 70}},
        located_display_status=lambda location: {"weather": {"temp_f": 50, "at": location.scope}},
    )
    api = app.test_client()
    credentials = {
        client: registered(api, client, capabilities=caps(client),
                           demand=demand(client, screens=("date", "weather1")))
        for client in ("office", "den", "north")
    }
    app.extensions["desk_display_render_coordinator"].tick()
    # Once for the server's location, once for New York (shared by two displays).
    assert sorted(rendered, key=str) == sorted(
        [("date", None), ("weather1", None), ("weather1", NEW_YORK.scope)], key=str)

    def weather_entry(client):
        manifest = api.get(f"/api/v1/clients/{client}/manifest", headers=bearer(credentials[client])).get_json()
        assert manifest["cache_complete"] is True
        return next(a for a in manifest["artifacts"] if a["screen_id"] == "weather1")

    assert weather_entry("office")["sha256"] == weather_entry("den")["sha256"] != weather_entry("north")["sha256"]

    beat = api.post("/api/v1/clients/office/heartbeat", json={"status": status("office")},
                    headers=bearer(credentials["office"])).get_json()
    assert beat["display_status"]["weather"] == {"temp_f": 50, "at": NEW_YORK.scope}
    beat = api.post("/api/v1/clients/north/heartbeat", json={"status": status("north")},
                    headers=bearer(credentials["north"])).get_json()
    assert beat["display_status"]["weather"] == {"temp_f": 70}

    # Clearing the location moves the display back to the shared render.
    del places["den"]
    assert weather_entry("den")["sha256"] == weather_entry("north")["sha256"]


def test_feed_demand_is_split_by_location():
    from display_server import demand_by_location
    from remote_display.models import ClientCapabilities, ClientDemand, PackageCapabilities
    from remote_display.registry import ClientRegistry

    registry = ClientRegistry(lease_seconds=1_000)
    preset = PROFILE_PRESETS["hyperpixel4"]
    for client, screens in (("hyper", ("date", "weather1", "astronomical")), ("square", ("weather1", "cubs last"))):
        registry.register(
            ClientCapabilities(protocol_version=1, client_software_version="0.2", client_id=client,
                               display_profile="hyperpixel4", logical_width=preset.width,
                               logical_height=preset.height, image_formats=("PNG",),
                               color_modes=(preset.color_mode,), render_package_versions=(1,)),
            ClientDemand(client_id=client, playlist_revision="pl-1", required_screens=screens,
                         package_capabilities=PackageCapabilities(render_package_versions=(1,),
                                                                  image_formats=("PNG",)),
                         sync_interval_seconds=30),
        )
    own, places = demand_by_location(registry, {"hyper": NEW_YORK}.get)
    assert own() == {"date", "weather1", "cubs last"}
    assert places() == {NEW_YORK: {"weather1", "astronomical"}}


def test_scoped_renders_use_the_locations_weather(monkeypatch):
    from remote_display import server_rendering
    from remote_display.models import RenderKey
    from services.data_coordinator import DataSnapshot

    seen = {}

    class Renderer:
        def render(self, screen, profile, preferences, data, record_frames=False):
            seen["weather"] = data.values["weather"]
            seen["fetched_at"] = preferences.values["weather_fetched_at"]
            raise RuntimeError("stop")

    import rendering.screen_renderer as screen_renderer

    monkeypatch.setattr(screen_renderer, "ScreenRenderer", Renderer)
    snapshot = DataSnapshot(revision=3, created_at=datetime.datetime.now(datetime.timezone.utc),
                            values=MappingProxyType({"weather": {"current": {"temp": 70}},
                                                     f"weather@{NEW_YORK.scope}": {"current": {"temp": 50}}}),
                            source_revisions=MappingProxyType({}))
    key = RenderKey.for_screen("weather1", "hyperpixel4", ScreenRevisions("s", "d", "r"), client_id=NEW_YORK.scope)

    class Logos:
        def for_size(self, width, height):
            return {}

    with pytest.raises(RuntimeError, match="stop"):
        server_rendering.compose_screen(key, PROFILE_PRESETS["hyperpixel4"], snapshot, Logos(), "then")
    assert seen["weather"]["current"] == {"temp": 50}
    assert seen["weather"]["location"] == {"latitude": 40.7128, "longitude": -74.006}


def test_scoped_revisions_follow_the_locations_feed():
    from remote_display.server_rendering import ServerRendering, StyleRevision
    from services.data_coordinator import DataCoordinator

    class Feeds:
        def data_revision(self, screen, source_revisions, scope=None):
            return f"f-{screen}-{scope}"

    rendering = ServerRendering(DataCoordinator(object()), StyleRevision([]), feeds=Feeds(), preferences="p",
                                logos=object())
    result = rendering.scoped_revisions([("weather1", NEW_YORK.scope)])
    assert result[("weather1", NEW_YORK.scope)].data_revision == f"f-weather1-{NEW_YORK.scope}"
    assert rendering.revisions(["weather1"])["weather1"].data_revision == "f-weather1-None"


# ── Fetching another place's weather ────────────────────────────────────────


def test_location_weather_has_its_own_cache_and_histories(monkeypatch, tmp_path):
    import data_fetch

    calls = []

    def weatherkit(now, location=None):
        calls.append(location)
        data_fetch._update_pressure_trend(now.timestamp(), 1000.0 if location else 1020.0)
        return {"current": {"temp": 50 if location else 70}, "source": "WeatherKit"}

    monkeypatch.setattr(data_fetch, "_weatherkit_configured", lambda: True)
    monkeypatch.setattr(data_fetch, "_fetch_weatherkit", weatherkit)
    monkeypatch.setattr(data_fetch, "_fetch_openweathermap", lambda now, location=None: None)
    monkeypatch.setattr(data_fetch, "_location_weather", {})
    monkeypatch.setattr(data_fetch, "_weather_cache", None)
    monkeypatch.setattr(data_fetch, "_weather_cache_fetched_at", None)
    monkeypatch.setattr(data_fetch, "_weather_last_attempt_at", None)
    monkeypatch.setattr(data_fetch, "_PRESSURE_HISTORY", data_fetch.deque())
    monkeypatch.setattr(data_fetch, "_PRESSURE_HISTORY_LOADED", True)
    monkeypatch.setattr(data_fetch, "_PRESSURE_HISTORY_PATH", str(tmp_path / "pressure.json"))
    monkeypatch.setattr(data_fetch, "_PRESSURE_HISTORY_LAST_SAVE", 0.0)

    place = (40.7128, -74.006)
    assert data_fetch.fetch_weather(force_refresh=True, location=place)["current"]["temp"] == 50
    assert data_fetch.fetch_weather(force_refresh=True)["current"]["temp"] == 70
    assert calls == [place, None]
    # The cached copy is reused within the refresh interval.
    assert data_fetch.fetch_weather(location=place)["current"]["temp"] == 50
    assert len(calls) == 2
    assert data_fetch.get_weather_cache_timestamp(place) is not None
    # Each place keeps its own pressure readings, and its own file.
    assert [p for _, p in data_fetch._PRESSURE_HISTORY] == [1020.0]
    assert (tmp_path / "pressure.40.7128_-74.0060.json").exists()


# ── Drawing ─────────────────────────────────────────────────────────────────


def test_radar_centres_on_a_displays_own_location(monkeypatch):
    from PIL import Image

    from screens import draw_weather

    requested = []

    def frames(zoom, **where):
        requested.append(where)
        return [draw_weather.RadarFrame(Image.new("RGBA", (draw_weather.WIDTH, draw_weather.HEIGHT)), None)]

    monkeypatch.setattr(draw_weather, "_fetch_radar_frames", frames)
    monkeypatch.setattr(draw_weather, "_fetch_base_map", lambda zoom, **where: requested.append(where))

    class Display:
        pass

    draw_weather.draw_weather_radar(Display(), {"hourly": []})
    draw_weather.draw_weather_radar(Display(), {"location": NEW_YORK.as_dict()})
    assert requested == [{}, {}, {"center": (40.7128, -74.006)}, {"center": (40.7128, -74.006)}]

    view = draw_weather._radar_view(7, (40.7128, -74.006))
    x_tile, y_tile, x_frac, y_frac = draw_weather._latlon_to_tile(40.7128, -74.006, 7)
    assert (view.center_x, view.center_y) == ((x_tile + x_frac) * 256, (y_tile + y_frac) * 256)
    assert draw_weather._radar_view(7) != view  # the server's view stays on Chicago's tile


# ── The one-time location seed ──────────────────────────────────────────────


def test_location_seed_sets_hyper_and_every_other_display_once(tmp_path):
    from remote_display import location_seed
    from remote_display.playlist_store import PlaylistStore

    store = PlaylistStore(tmp_path / "playlists.json")
    store.set_friendly_name("den", "Den", actor="jason")
    store.set_location("office", 1, 2, actor="jason")  # already set: kept
    clients = location_seed.known_clients(store, {"clients": {"hyper": {}, "square-panel": {}}},
                                          provisioned=["mini"], static=["lobby"])
    assert clients == {"hyper", "square-panel", "mini", "lobby", "den", "office"}
    changed = location_seed.apply(store, clients)
    assert sorted(changed) == ["den", "hyper", "lobby", "mini", "square-panel"]
    places = store.client_locations()
    assert places["hyper"] == Location(41.9037, -87.6357)
    assert places["square-panel"] == places["den"] == Location(42.1373, -87.8446)
    assert places["office"] == Location(1, 2)
    assert store.snapshot()["clients"]["den"]["friendly_name"] == "Den"
    assert location_seed.location_for("hyper-panel") == location_seed.HYPER_LOCATION
    assert location_seed.location_for("hyperpixel") == location_seed.DEFAULT_LOCATION

    # Never again: later Clients page edits stick.
    store.set_location("hyper", None, None, actor="jason")
    revision = store.snapshot()["store_revision"]
    assert location_seed.apply(store, clients | {"new"}) == []
    assert store.location("hyper") is None and store.location("new") is None
    assert store.snapshot()["store_revision"] == revision


def test_a_display_at_the_servers_location_shares_the_servers_renders(tmp_path, monkeypatch):
    pytest.importorskip("flask")
    import display_server

    monkeypatch.setenv("WEATHER_LATITUDE", "42.1373")
    monkeypatch.setenv("WEATHER_LONGITUDE", "-87.8446")
    config = display_server.DisplayServerConfig(enrollment="shared", auth_token="t" * 40,
                                                admin_token="a" * 40, artifact_dir=tmp_path / "a")
    places = {"square-panel": Location(42.1373, -87.8446), "hyper": Location(41.9037, -87.6357)}
    app = display_server.create_app(config, assignments=lambda _c: None, client_locations=lambda: places)
    location_of = app.extensions["desk_display_location_of"]
    assert location_of("square-panel") is None
    assert location_of("hyper") == places["hyper"]


def test_server_start_applies_the_seed_to_registered_displays(tmp_path):
    pytest.importorskip("flask")
    import json

    import display_server
    from remote_display.playlist_store import PlaylistStore

    snapshot = tmp_path / "clients.json"
    snapshot.write_text(json.dumps({"schema_version": 1, "clients": {"hyper": {}, "square-panel": {}}}))
    config = display_server.DisplayServerConfig(enrollment="shared", auth_token="t" * 40, admin_token="a" * 40,
                                                artifact_dir=tmp_path / "a",
                                                playlist_store_path=tmp_path / "playlists.json",
                                                registry_snapshot_path=snapshot)
    display_server._apply_location_seed(config)
    places = PlaylistStore(tmp_path / "playlists.json").client_locations()
    assert places == {"hyper": Location(41.9037, -87.6357), "square-panel": Location(42.1373, -87.8446)}
