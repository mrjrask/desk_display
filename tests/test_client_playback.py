"""Phase 13: every client runs its assigned playlist independently."""
from __future__ import annotations

import zlib
from collections import Counter
from concurrent.futures import Future

import pytest

pytest.importorskip("flask")

import display_client  # noqa: E402
import display_server  # noqa: E402
from display_profiles import PROFILE_PRESETS  # noqa: E402
from remote_display.models import ScreenRevisions  # noqa: E402
from remote_display.playlist_store import PlaylistStore  # noqa: E402
from remote_display.render_coordinator import RenderOutput  # noqa: E402
from schedule import build_scheduler  # noqa: E402

TOKEN = "server-token-" + "s" * 32
# Deliberately not alphabetical: clients must play screens in saved order.
SHARED = {"screens": {"date": 1, "weather1": 1, "news headlines": 1}, "sequence": []}


class Inline:
    def submit(self, fn):
        future = Future()
        try:
            future.set_result(fn())
        except Exception as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future


class Transport:
    def __init__(self, app):
        self.api = app.test_client()
        self.down = False

    def __call__(self, method, path, *, headers=None, json_body=None, max_bytes=1 << 24):
        from remote_display.client_sync import Response, TransportError

        if self.down:
            raise TransportError("offline")
        response = self.api.open(path, method=method, headers=headers or {}, json=json_body)
        return Response(response.status_code, response.get_data(), dict(response.headers))


class Presenter:
    def present(self, image):
        return image


class Clock:
    now = 1_800_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def world(tmp_path):
    store_path = tmp_path / "playlists.json"
    store = PlaylistStore(store_path)
    rendered = Counter()

    def renderer(key):
        rendered[(key.screen_id, key.render_profile)] += 1
        preset = PROFILE_PRESETS[key.render_profile]
        from PIL import Image

        color = zlib.crc32(key.screen_id.encode()) % 250
        return RenderOutput(image=Image.new(preset.color_mode, (preset.width, preset.height), color),
                            refresh_seconds=100_000)

    def revisions(screens):
        return {s: ScreenRevisions("s1", "d1", "r1") for s in screens}

    config = display_server.DisplayServerConfig(enrollment="shared",
        auth_token=TOKEN, lease_seconds=100_000, artifact_dir=tmp_path / "artifacts",
        playlist_store_path=store_path, render_min_interval_seconds=0,
    )
    app = display_server.create_app(config, renderer=renderer, revisions=revisions,
                                    render_executor=Inline(), clock=Clock())
    coordinator = app.extensions["desk_display_render_coordinator"]
    transports = {}

    def client(client_id, profile="hyperpixel4", **settings):
        transport = transports.setdefault(client_id, Transport(app))
        values = {
            "DESK_DISPLAY_PROFILE": profile, "DESK_DISPLAY_CLIENT_ID": client_id,
            "DESK_DISPLAY_SERVER_URL": "https://render.lan", "DESK_DISPLAY_CLIENT_TOKEN": TOKEN,
            "DESK_DISPLAY_CLIENT_CACHE_DIR": str(tmp_path / client_id),
        }
        values.update(settings)
        return display_client.build_client(values, presenter=Presenter(), transport=transport)

    def assign(client_id, document):
        playlist = store.create(client_id, document, actor="test")
        current = store.snapshot()["assignments"].get(client_id)
        store.assign(client_id, playlist["id"], expected_playlist_id=current["playlist_id"] if current else None,
                     actor="test")
        return playlist

    def settle(*clients):
        """Sync until every client has registered demand and activated its playlist."""

        for _ in range(3):
            for c in clients:
                c.sync.sync_once()
            coordinator.tick()
        for c in clients:
            c.sync.sync_once()
            assert c.sync.active().playlist is not None, c.sync.client_id

    class World:
        pass

    result = World()
    result.__dict__.update(store=store, rendered=rendered, coordinator=coordinator, client=client,
                           assign=assign, settle=settle, transports=transports)
    return result


def shown(client, count):
    return [client.step()[0] for _ in range(count)]


def expected_order(document, count):
    return build_scheduler(document).preview_scheduled_ids(count)


def test_clients_play_independently_and_share_renders(world):
    playlist = world.store.create("Shared", SHARED, actor="test")
    for client_id in ("office", "lobby"):
        world.store.assign(client_id, playlist["id"], expected_playlist_id=None, actor="test")
    office, lobby = world.client("office"), world.client("lobby")
    world.settle(office, lobby)
    assert all(count == 1 for count in world.rendered.values())
    assert set(world.rendered) == {(s, "hyperpixel4") for s in SHARED["screens"]}

    order = expected_order(SHARED, 6)
    assert shown(office, 2) == order[:2]
    assert shown(lobby, 1) == order[:1]
    office.on_button("B")  # skip affects only the office display
    assert shown(office, 1) == order[2:3]
    assert shown(lobby, 1) == order[1:2]
    assert office.playback.history != lobby.playback.history

    # Advancing playback and reporting it never causes a rerender.
    before = dict(world.rendered)
    for c in (office, lobby):
        c.sync.sync_once()
    world.coordinator.tick()
    assert dict(world.rendered) == before


def test_same_playlist_on_different_profiles_renders_once_per_profile(world):
    playlist = world.store.create("Shared", SHARED, actor="test")
    for client_id in ("office", "tv", "den"):
        world.store.assign(client_id, playlist["id"], expected_playlist_id=None, actor="test")
    clients = [world.client("office"), world.client("tv", "hdmi_1080p"), world.client("den", "hyperpixel4")]
    world.settle(*clients)
    assert all(count == 1 for count in world.rendered.values())
    assert {profile for _screen, profile in world.rendered} == {"hyperpixel4", "hdmi_1080p"}


def test_partially_shared_playlists_share_common_screens(world):
    world.assign("office", {"screens": {"date": 1, "weather1": 1}, "sequence": []})
    world.assign("lobby", {"screens": {"date": 1, "news headlines": 1}, "sequence": []})
    office, lobby = world.client("office"), world.client("lobby")
    world.settle(office, lobby)
    assert world.rendered[("date", "hyperpixel4")] == 1
    assert set(world.rendered) == {("date", "hyperpixel4"), ("weather1", "hyperpixel4"),
                                   ("news headlines", "hyperpixel4")}


def test_different_rotations_share_renders_and_keep_own_position(world):
    playlist = world.store.create("Shared", SHARED, actor="test")
    for client_id in ("upright", "sideways"):
        world.store.assign(client_id, playlist["id"], expected_playlist_id=None, actor="test")
    upright, sideways = world.client("upright"), world.client("sideways", DISPLAY_ROTATION=90)
    world.settle(upright, sideways)
    assert all(count == 1 for count in world.rendered.values())
    shown(upright, 2)
    assert shown(sideways, 1) == expected_order(SHARED, 1)


def test_restart_resumes_the_clients_own_position(world):
    world.assign("office", SHARED)
    office = world.client("office")
    world.settle(office)
    order = expected_order(SHARED, 8)
    assert shown(office, 4) == order[:4]
    world.transports["office"].down = True
    restarted = world.client("office")  # offline restart from cache
    assert shown(restarted, 3) == order[4:7]
    assert restarted.playback.history[-7:] == order[:7]


def test_playlist_update_continues_after_the_current_screen(world):
    playlist = world.assign("office", SHARED)
    office = world.client("office")
    world.settle(office)
    first = shown(office, 2)
    updated = {"screens": {**SHARED["screens"], "cubs last": 1}, "sequence": []}
    world.store.update(playlist["id"], updated, expected_revision=playlist["revision"], actor="test")
    world.settle(office)
    assert office.sync.active().playlist.playlist_revision != playlist["revision"]
    new_order = build_scheduler(updated).preview_scheduled_ids(len(updated["screens"]))
    following = new_order[new_order.index(first[-1]) + 1]
    assert office.step()[0] == following


def test_outage_on_one_client_does_not_affect_another(world):
    playlist = world.store.create("Shared", SHARED, actor="test")
    for client_id in ("office", "lobby"):
        world.store.assign(client_id, playlist["id"], expected_playlist_id=None, actor="test")
    office, lobby = world.client("office"), world.client("lobby")
    world.settle(office, lobby)
    world.transports["office"].down = True
    office.sync.step()
    order = expected_order(SHARED, 4)
    assert shown(office, 2) == order[:2] and office.report.playback_state == "offline"
    assert shown(lobby, 1) == order[:1] and lobby.report.playback_state == "playing"
    world.transports["office"].down = False
    office.sync.step()
    assert office.sync.connected and shown(office, 1) == order[2:3]


# ── Scheduler position persistence ─────────────────────────────────────────


ALT = {"screens": {"date": 1, "weather1": {"frequency": 1, "alt": {"screen": "weather2", "frequency": 2}},
                   "news headlines": 2}, "sequence": []}


def test_scheduler_state_round_trip_continues_the_sequence():
    reference = build_scheduler(ALT)
    expected = reference.preview_scheduled_ids(20)
    running = build_scheduler(ALT)
    for _ in range(7):
        running._next_scheduled_id()
    resumed = build_scheduler(ALT)
    assert resumed.restore_state(running.export_state())
    assert resumed.preview_scheduled_ids(13) == expected[7:]


def test_scheduler_state_for_another_playlist_is_ignored():
    state = build_scheduler(ALT).export_state()
    other = build_scheduler(SHARED)
    assert not other.restore_state(state)
    assert not other.restore_state({"signature": other.export_state()["signature"], "cycle": "x"})
    assert other.preview_scheduled_ids(3) == build_scheduler(SHARED).preview_scheduled_ids(3)


def test_seek_after_continues_from_the_next_slot():
    scheduler = build_scheduler(SHARED)
    order = build_scheduler(SHARED).preview_scheduled_ids(3)
    assert scheduler.seek_after(order[0])
    assert scheduler.preview_scheduled_ids(2) == order[1:3]
    assert not build_scheduler(SHARED).seek_after("cubs last")
