"""Phase 20a: end-to-end parity and resilience of the server/client system.

Unlike the per-phase suites, nothing here is faked between the provider data
and the pixels a client shows. One real render server (production renderer,
render coordinator, playlist store and per-client provisioning) serves several
real display clients over the wire protocol, and every picture is compared
with what the standalone display draws from the same fixture data, profile,
screen, style and time.

Provider data is a fixed fixture and every upstream request is blocked, so
the suite is deterministic and offline.
"""
from __future__ import annotations

import datetime as dt
import io
import json
from concurrent.futures import Future
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("flask")

import display_client  # noqa: E402
import display_server  # noqa: E402
import install_modes  # noqa: E402
import screens.mlb_league_standings as mlb  # noqa: E402
from display.rotation import rotate_frame, to_logical, to_physical  # noqa: E402
from display_profiles import PROFILE_PRESETS  # noqa: E402
from playback.package_player import PackagePlayback  # noqa: E402
from remote_display.client_sync import Response, TransportError  # noqa: E402
from remote_display.models import RenderKey, ScreenRevisions  # noqa: E402
from remote_display.playlist_store import PlaylistStore  # noqa: E402
from remote_display.render_package import validate_package  # noqa: E402
from remote_display.server_rendering import ServerRendering  # noqa: E402
from rendering.logos import IMAGES_DIR  # noqa: E402
from rendering.screen_renderer import _thaw_legacy_data  # noqa: E402
from services.data_coordinator import DataCoordinator  # noqa: E402
from utils import ScreenImage  # noqa: E402

ADMIN = "admin-token-" + "a" * 32
START = 1_800_000_000.0

_TEAMS = {
    "East": [("NYY", "Yankees", 95, 67), ("BOS", "Red Sox", 88, 74), ("TOR", "Blue Jays", 80, 82),
             ("TB", "Rays", 77, 85), ("BAL", "Orioles", 70, 92)],
    "Central": [("CLE", "Guardians", 90, 72), ("KC", "Royals", 85, 77), ("DET", "Tigers", 84, 78),
                ("MIN", "Twins", 75, 87), ("CWS", "White Sox", 60, 102)],
    "West": [("HOU", "Astros", 89, 73), ("SEA", "Mariners", 86, 76), ("TEX", "Rangers", 78, 84),
             ("LAA", "Angels", 70, 92), ("OAK", "Athletics", 69, 93)],
}


def standings(bump: int = 0) -> dict:
    """The fixture provider data; *bump* changes the leader's record."""

    def rows(division):
        result = []
        for i, (abbr, name, wins, losses) in enumerate(_TEAMS[division]):
            wins += bump if i == 0 else 0
            result.append({"abbr": abbr, "team_name": name, "wins": wins, "losses": losses,
                           "gb": "-", "wc_gb": "-", "pct": f".{int(1000 * wins / (wins + losses))}"})
        return result

    return {league: {division: rows(division) for division in _TEAMS}
            for league in (mlb.AL_LEAGUE_ID, mlb.NL_LEAGUE_ID)}


# Screens that play in the rotation. The standings scroll on every profile,
# the overview is a still and ``date`` is a clock the client draws itself.
PLAYLIST_A = {"screens": {"MLB AL Standings": 1, "date": 1, "MLB NL Standings": 1}, "sequence": []}
PLAYLIST_B = {"screens": {"AL Overview": 1, "date": 1}, "sequence": []}


@pytest.fixture(autouse=True)
def offline_fixture_data(monkeypatch):
    """Serve the fixture instead of the provider; refuse any real request."""

    import requests

    def refuse(*_args, **_kwargs):
        raise requests.ConnectionError("upstream requests are disabled in the end-to-end suite")

    monkeypatch.setattr(requests.Session, "request", refuse)
    current = {"value": standings()}
    monkeypatch.setattr(mlb, "_fetch_league_standings", lambda **_kwargs: current["value"])
    return current


# ── The standalone reference ────────────────────────────────────────────────


class StandaloneDisplay:
    """What a screen sees on a standalone device, minus the panel.

    It keeps every frame written. For a scrolling screen the first frame is
    the still, and skipping after it (as the buttons do) keeps the scroll off
    the wall clock; its full content comes back from the screen itself.
    """

    def __init__(self, width, height, *, skip_after_first=False):
        self.width, self.height = width, height
        self.frames: list[Image.Image] = []
        self.skip_after_first = skip_after_first

    def clear(self):
        pass

    def image(self, image):
        self.frames.append(image.copy())

    def show(self):
        pass

    def skip_requested(self):
        return self.skip_after_first and bool(self.frames)


def standalone(screen_id, profile, data):
    """Render *screen_id* the way main.py does; return (frames, full content)."""

    from screens.registry import ScreenContext, build_screen_registry

    from rendering.screen_classes import CLASSIFICATIONS, SCROLLING_CANVAS

    display = StandaloneDisplay(profile.width, profile.height,
                                skip_after_first=CLASSIFICATIONS[screen_id].kind == SCROLLING_CANVAS)
    now = dt.datetime.now(dt.timezone.utc)
    rendering = ServerRendering(data, preferences="p-e2e")
    context = ScreenContext(
        display=display, cache=_thaw_legacy_data(data.snapshot().values),
        logos=rendering.logos.for_size(profile.width, profile.height), image_dir=IMAGES_DIR,
        now=now, now_utc=now, offline=False, weather_fetched_at=None, skip_scoreboards=False,
        render_profile=profile,
    )
    import time

    real_sleep, time.sleep = time.sleep, lambda _seconds: None  # frames, not pacing, are compared
    try:
        result = build_screen_registry(context)[0][screen_id].render()
    finally:
        time.sleep = real_sleep
    content = result.image if isinstance(result, ScreenImage) else None
    return display.frames, content


def standalone_still(screen_id, profile, data):
    """The picture a standalone display settles on for *screen_id*.

    A scrolling screen is judged by its first frame (the rest is in the
    package); an animation by the frame it holds at the end.
    """

    from rendering.screen_classes import CLASSIFICATIONS, FINITE_ANIMATION

    frames, _ = standalone(screen_id, profile, data)
    animated = CLASSIFICATIONS[screen_id].kind == FINITE_ANIMATION
    return as_profile(frames[-1] if animated else frames[0], profile)


def as_profile(image, profile):
    """A standalone frame as the panel receives it (see screens.registry._ProfileDisplay)."""

    if image.size != (profile.width, profile.height):
        image = image.resize((profile.width, profile.height), Image.Resampling.LANCZOS)
    return image.convert(profile.color_mode)


def same(a, b):
    return a.mode == b.mode and a.size == b.size and a.tobytes() == b.tobytes()


def data_with(value=None):
    data = DataCoordinator()
    data.publish("mlb_league_standings", value or standings())
    return data


# ── Parity ──────────────────────────────────────────────────────────────────


def _parity_cases():
    cases = [("MLB AL Standings", profile) for profile in PROFILE_PRESETS]
    cases += [(screen, profile) for screen in ("AL Overview", "MLB NLWC Standings")
              for profile in ("hyperpixel4_square", "display_hat_mini", "waveshare_oled_128x64")]
    return cases


@pytest.mark.parametrize(("screen_id", "profile_id"), _parity_cases())
def test_server_render_matches_the_standalone_display(screen_id, profile_id):
    profile = PROFILE_PRESETS[profile_id]
    data = data_with()
    frames, content = standalone(screen_id, profile, data)
    key = RenderKey.for_screen(screen_id, profile_id, ScreenRevisions("s", "d", "r"))
    output = ServerRendering(data, preferences="p-e2e").render(key)

    assert len(output.image.getcolors(1 << 24) or ()) > 1, "the fixture must draw something"
    assert same(output.image, standalone_still(screen_id, profile, data))

    if output.package is None:
        return
    validate_package(output.package, key=key)
    if output.package["kind"] == "animation":
        # Every frame the client plays is one the standalone display showed.
        shown = {as_profile(f, profile).tobytes() for f in frames}
        playback = PackagePlayback(output.package, profile, hold_seconds=0)
        for index in range(len(output.package["animation"]["frames"])):
            t = sum(f["duration_ms"] for f in output.package["animation"]["frames"][:index]) / 1000
            assert playback.frame_at(t).tobytes() in shown, f"animation frame {index} was never shown"
        return
    assert output.package["kind"] == "scroll"
    playback = PackagePlayback(output.package, profile, hold_seconds=0)
    max_offset = content.height - profile.height
    offsets = set()
    t = 0.0
    while t <= playback.motion_seconds + playback.frame_seconds:
        offset = playback.key_at(t)[1]
        offsets.add(offset)
        expected = as_profile(content.crop((0, offset, profile.width, offset + profile.height)), profile)
        assert same(playback.frame_at(t), expected), f"frame at offset {offset} differs"
        t += playback.frame_seconds
    assert {0, max_offset} <= offsets


# ── One server, many clients ────────────────────────────────────────────────


class Inline:
    def submit(self, fn):
        future = Future()
        try:
            future.set_result(fn())
        except Exception as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future


class Clock:
    def __init__(self):
        self.now = START

    def __call__(self):
        return self.now


class Wire:
    """One client's network path to whichever server is currently up."""

    def __init__(self, system):
        self.system = system
        self.down = False

    def __call__(self, method, path, *, headers=None, json_body=None, max_bytes=1 << 24):
        if self.down or self.system.app is None:
            raise TransportError("connection refused")
        response = self.system.app.test_client().open(path, method=method, headers=headers or {},
                                                      json=json_body)
        return Response(response.status_code, response.get_data(), dict(response.headers))


class Panel:
    """A panel mounted at *rotation*: stores the physical frames it is sent."""

    def __init__(self, rotation=0):
        self.rotation = rotation
        self.frames: list[Image.Image] = []
        self.logical: list[Image.Image] = []

    def present(self, image):
        self.logical.append(image)
        self.frames.append(rotate_frame(image, self.rotation))
        return image


class System:
    """A server installation in a project directory, and its clients."""

    def __init__(self, root: Path, data: DataCoordinator):
        self.root = root
        self.data = data
        self.clock = Clock()
        self.renders: list[tuple[str, str]] = []
        self.fail: set[str] = set()
        self.app = None
        self.wires: dict[str, Wire] = {}
        self.credentials: dict[str, str] = {}
        self.store_path = root / ".runtime" / "server" / "playlists.json"
        self.start()

    def start(self):
        rendering = ServerRendering(self.data, preferences="p-e2e")

        def renderer(key):
            if key.screen_id in self.fail:
                raise RuntimeError(f"provider layout changed for {key.screen_id}")
            self.renders.append((key.screen_id, key.render_profile))
            return rendering.render(key)

        server = self.root / ".runtime" / "server"
        config = display_server.DisplayServerConfig(
            admin_token=ADMIN, lease_seconds=600, artifact_dir=self.root / "cache" / "artifacts",
            playlist_store_path=self.store_path, clients_path=server / "provisioned_clients.json",
            registry_snapshot_path=server / "clients.json", render_min_interval_seconds=0,
        )
        self.app = display_server.create_app(config, renderer=renderer, revisions=rendering.revisions,
                                             data_health=rendering.health, render_executor=Inline(),
                                             clock=self.clock)
        self.store = PlaylistStore(self.store_path)

    def stop(self):
        self.app = None

    def admin(self, method, path, body=None):
        response = self.app.test_client().open(f"/api/v1/admin/{path}", method=method, json=body,
                                               headers={"Authorization": f"Bearer {ADMIN}"})
        return response.status_code, response.get_json()

    def playlist(self, name, document):
        return self.store.create(name, document, actor="e2e")["id"]

    def provision(self, client_id, profile, playlist_id):
        status, body = self.admin("POST", "clients", {"client_id": client_id, "display_profile": profile,
                                                      "playlist_id": playlist_id})
        assert status == 201, body
        self.credentials[client_id] = body["client_credential"]
        return body

    def client(self, client_id, profile, *, rotation=0, **settings):
        values = {
            "DESK_DISPLAY_PROFILE": profile, "DESK_DISPLAY_CLIENT_ID": client_id,
            "DESK_DISPLAY_SERVER_URL": "https://render.lan:8765",
            "DESK_DISPLAY_CLIENT_TOKEN": self.credentials[client_id],
            "DESK_DISPLAY_CLIENT_CACHE_DIR": str(self.root / "clients" / client_id),
            "DISPLAY_ROTATION": rotation // 90,
        }
        values.update(settings)
        wire = self.wires.setdefault(client_id, Wire(self))
        return display_client.build_client(values, presenter=Panel(rotation), transport=wire)

    def tick(self):
        self.app.extensions["desk_display_render_coordinator"].tick()

    def settle(self, *clients):
        """Sync until every client has registered demand and plays its playlist."""

        for _ in range(3):
            for c in clients:
                c.sync.sync_once()
            self.tick()
        for c in clients:
            c.sync.sync_once()
            assert c.sync.active().playlist is not None, c.sync.client_id

    def status(self, client_id):
        _, body = self.admin("GET", "status")
        return next(c for c in body["clients"] if c["client_id"] == client_id)


@pytest.fixture
def system(tmp_path):
    return System(tmp_path, data_with())


def play(client, count):
    """Show *count* screens; return (screen, logical frame) for each."""

    shown = []
    for _ in range(count):
        screen, _seconds = client.step()
        shown.append((screen, client.presenter.logical[-1]))
    return shown


def expected_still(system, screen, profile_id):
    profile = PROFILE_PRESETS[profile_id]
    return standalone_still(screen, profile, system.data)


def assert_standalone_shows(system, screen, profile_id, frame):
    """*frame*, the first a client shows for *screen*, is the standalone's.

    Animations start from their first frame on a client, so any frame the
    standalone display showed qualifies; everything else must be the still.
    """

    from rendering.screen_classes import CLASSIFICATIONS, FINITE_ANIMATION

    profile = PROFILE_PRESETS[profile_id]
    if CLASSIFICATIONS[screen].kind == FINITE_ANIMATION:
        frames, _ = standalone(screen, profile, system.data)
        assert frame.tobytes() in {as_profile(f, profile).tobytes() for f in frames}, screen
    else:
        assert same(frame, expected_still(system, screen, profile_id)), screen


def test_one_server_serves_shared_and_different_playlists_and_profiles(system):
    a = system.playlist("Standings", PLAYLIST_A)
    b = system.playlist("Overview", PLAYLIST_B)
    layout = {"office": ("hyperpixel4", a), "den": ("hyperpixel4", a),
              "kitchen": ("display_hat_mini", a), "shelf": ("waveshare_oled_128x64", b)}
    for client_id, (profile, playlist) in layout.items():
        system.provision(client_id, profile, playlist)
    clients = {cid: system.client(cid, profile) for cid, (profile, _) in layout.items()}
    system.settle(*clients.values())

    # Equivalent clients share render work: one render per screen and profile.
    assert len(system.renders) == len(set(system.renders))
    assert set(system.renders) == {
        *((s, "hyperpixel4") for s in PLAYLIST_A["screens"]),
        *((s, "display_hat_mini") for s in PLAYLIST_A["screens"]),
        *((s, "waveshare_oled_128x64") for s in PLAYLIST_B["screens"]),
    }

    # Every client shows exactly what a standalone display of its profile shows.
    for client_id, client in clients.items():
        profile_id, _ = layout[client_id]
        document = PLAYLIST_A if layout[client_id][1] == a else PLAYLIST_B
        seen = play(client, len(document["screens"]))
        assert [s for s, _ in seen] == list(document["screens"])
        for screen, frame in seen:
            if screen == "date":
                assert frame.size == (PROFILE_PRESETS[profile_id].width, PROFILE_PRESETS[profile_id].height)
                continue
            assert_standalone_shows(system, screen, profile_id, frame.convert(PROFILE_PRESETS[profile_id].color_mode))

    # Playback is independent: moving one client leaves the others alone.
    before = {cid: c.playback.current_screen for cid, c in clients.items()}
    clients["office"].on_button("next")
    play(clients["office"], 2)
    assert {cid: c.playback.current_screen for cid, c in clients.items() if cid != "office"} == \
        {cid: s for cid, s in before.items() if cid != "office"}

    # The server learns each client's position and nothing secret.
    for client_id, client in clients.items():
        client.sync.sync_once()
        status = system.status(client_id)
        assert status["lease_state"] == "active"
        assert status["status"]["current_screen"] == client.playback.current_screen
    _, body = system.admin("GET", "status")
    text = json.dumps(body)
    assert not any(credential in text for credential in system.credentials.values())


def test_playlist_edits_reach_clients_and_are_acknowledged(system):
    a = system.playlist("Standings", PLAYLIST_A)
    for client_id in ("office", "den"):
        system.provision(client_id, "hyperpixel4", a)
    office, den = system.client("office", "hyperpixel4"), system.client("den", "hyperpixel4")
    system.settle(office, den)
    first = system.store.get(a)["revision"]
    for c in (office, den):
        c.sync.sync_once()
        assert system.status(c.sync.client_id)["status"]["accepted_revisions"]["playlist_revision"] == first

    edited = {"screens": {"MLB NL Standings": 1, "MLB AL Standings": 1}, "sequence": []}
    system.store.update(a, edited, expected_revision=first, actor="e2e")
    second = system.store.get(a)["revision"]
    assert second != first
    # Not acknowledged until the client has actually taken the new revision.
    assert system.status("office")["status"]["accepted_revisions"]["playlist_revision"] == first
    system.settle(office, den)
    for c in (office, den):
        c.sync.sync_once()
        assert system.status(c.sync.client_id)["status"]["accepted_revisions"]["playlist_revision"] == second
        shown = {screen for screen, _ in play(c, 4)}
        assert shown == set(edited["screens"])


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_every_rotation_shows_the_same_logical_picture(system, rotation):
    a = system.playlist("Standings", {"screens": {"MLB AL Standings": 1}, "sequence": []})
    system.provision("wall", "hyperpixel4", a)
    client = system.client("wall", "hyperpixel4", rotation=rotation)
    system.settle(client)
    (screen, logical), = play(client, 1)
    profile = PROFILE_PRESETS["hyperpixel4"]
    assert same(logical, expected_still(system, screen, "hyperpixel4"))

    physical = client.presenter.frames[-1]
    for x, y in [(0, 0), (profile.width - 1, 0), (0, profile.height - 1), (123, 45)]:
        px, py = to_physical(x, y, profile.width, profile.height, rotation)
        assert physical.getpixel((px, py)) == logical.getpixel((x, y))
        assert to_logical(px, py, profile.width, profile.height, rotation) == (x, y)
    # The server never renders per rotation; the client reports what it applied.
    assert set(system.renders) == {("MLB AL Standings", "hyperpixel4")}
    client.sync.sync_once()
    assert system.status("wall")["status"]["physical_rotation"] == rotation


# ── Restarts and outages ────────────────────────────────────────────────────


def _ready(system, client_id="office", profile="hyperpixel4", document=PLAYLIST_A, **settings):
    playlist = system.playlist(client_id, document)
    system.provision(client_id, profile, playlist)
    client = system.client(client_id, profile, **settings)
    system.settle(client)
    return client


def test_server_restart_is_invisible_to_clients(system):
    client = _ready(system)
    shown = [s for s, _ in play(client, 2)]

    system.stop()
    with pytest.raises(Exception, match="refused"):
        client.sync.sync_once()
    shown += [s for s, _ in play(client, 2)]  # keeps playing from its cache

    system.start()  # same project directory: playlists, credentials, artifacts
    client.sync.sync_once()  # its lease is unknown to the new process: it enrolls again
    assert client.sync.connected
    shown += [s for s, _ in play(client, 2)]
    order = list(PLAYLIST_A["screens"])
    assert shown == [order[i % len(order)] for i in range(6)]
    assert system.status("office")["lease_state"] == "active"


def test_client_restart_resumes_where_it_was_without_the_server(system):
    client = _ready(system)
    play(client, 2)
    position = client.playback.current_screen
    system.stop()

    restarted = system.client("office", "hyperpixel4")
    (screen, frame), = play(restarted, 1)
    order = list(PLAYLIST_A["screens"])
    assert screen == order[(order.index(position) + 1) % len(order)]
    if screen != "date":
        assert same(frame, expected_still(system, screen, "hyperpixel4"))


def test_long_outage_keeps_playing_then_expires_then_recovers(system):
    client = _ready(system, DESK_DISPLAY_OFFLINE_MAX_AGE_HOURS=24)
    system.wires["office"].down = True
    for hours in (1, 12, 23):
        client.sync._clock = lambda h=hours: system.clock.now + h * 3600
        with pytest.raises(Exception, match="refused"):
            client.sync.sync_once()
        assert client.step()[0] is not None
        assert client.report.playback_state == "offline"
    client.sync._clock = lambda: system.clock.now + 30 * 3600
    assert client.step()[0] is None  # shows "offline; cached content expired"

    system.wires["office"].down = False
    system.clock.now += 30 * 3600
    client.sync._clock = system.clock
    system.settle(client)
    assert client.step()[0] is not None and client.report.playback_state == "playing"


def test_stale_data_is_flagged_and_still_shown(system):
    client = _ready(system, document={"screens": {"MLB AL Standings": 1}, "sequence": []})
    (_, fresh), = play(client, 1)
    system.clock.now += 3 * 3600  # far past the refresh deadline, provider silent
    system.fail.add("MLB AL Standings")
    system.tick()
    client.sync.sync_once()
    entry = next(e for e in client.sync.active().manifest["artifacts"] if e["screen_id"] == "MLB AL Standings")
    assert entry["stale"] is True and entry["state"] in {"stale", "fallback"}
    (_, shown), = play(client, 1)
    assert same(shown, fresh)


def test_a_failed_render_keeps_the_last_good_picture(system, offline_fixture_data):
    client = _ready(system, document={"screens": {"MLB AL Standings": 1}, "sequence": []})
    (_, good), = play(client, 1)

    system.fail.add("MLB AL Standings")
    system.data.publish("mlb_league_standings", standings(bump=5))  # new data, render fails
    system.settle(client)
    entry = next(e for e in client.sync.active().manifest["artifacts"] if e["screen_id"] == "MLB AL Standings")
    assert entry["state"] == "fallback" and entry["failure"]
    (_, shown), = play(client, 1)
    assert same(shown, good)

    system.fail.clear()  # the provider recovers: the new data reaches the panel
    offline_fixture_data["value"] = standings(bump=5)
    system.clock.now += 3600
    system.settle(client)
    (_, updated), = play(client, 1)
    assert not same(updated, good)
    assert same(updated, expected_still(system, "MLB AL Standings", "hyperpixel4"))


def test_a_corrupt_client_cache_recovers_from_the_server(system):
    client = _ready(system, document={"screens": {"MLB AL Standings": 1}, "sequence": []})
    (_, good), = play(client, 1)
    objects = list((system.root / "clients" / "office" / "artifacts").iterdir())
    assert objects
    for path in objects:
        path.write_bytes(b"not a png")

    restarted = system.client("office", "hyperpixel4")
    assert restarted.step()[0] is None  # nothing usable: a diagnostic, never garbage
    system.settle(restarted)
    (_, shown), = play(restarted, 1)
    assert same(shown, good)


# ── Security and compatibility ──────────────────────────────────────────────


def test_a_revoked_client_is_cut_off_but_its_panel_stays_up(system):
    client = _ready(system)
    status, body = system.admin("POST", "clients/office/revoke")
    assert status == 200 and body["state"] == "revoked"

    with pytest.raises(Exception) as failure:
        client.sync.sync_once()
    assert getattr(failure.value, "code", "") in {"unauthorized", "http_401"}
    assert client.step()[0] is not None  # cached content, no new content

    # Neither the old lease nor the enrollment credential works any more.
    api = system.app.test_client()
    credential = client.sync._credential
    assert credential is None or api.get("/api/v1/clients/office/manifest",
                                         headers={"Authorization": f"Bearer {credential}"}).status_code == 401
    wire = display_client.capabilities_for("office", PROFILE_PRESETS["hyperpixel4"]).to_wire()
    response = api.post("/api/v1/register", json={"capabilities": wire},
                        headers={"Authorization": f"Bearer {system.credentials['office']}"})
    assert response.status_code == 401
    assert system.credentials["office"] not in response.get_data(as_text=True)


def test_an_incompatible_client_is_refused_with_a_clear_error(system):
    playlist = system.playlist("Standings", PLAYLIST_A)
    system.provision("future", "hyperpixel4", playlist)
    wire = display_client.capabilities_for("future", PROFILE_PRESETS["hyperpixel4"]).to_wire()
    wire["protocol_version"] = 99
    response = system.app.test_client().post(
        "/api/v1/register", json={"capabilities": wire},
        headers={"Authorization": f"Bearer {system.credentials['future']}"})
    assert response.status_code == 409
    body = response.get_json()
    assert body["error"] and system.credentials["future"] not in json.dumps(body)

    # An old manifest schema is refused by the client, which keeps its cache.
    client = _ready(system, "office")
    original = client.sync.transport

    def future_manifests(method, path, **kwargs):
        response = original(method, path, **kwargs)
        if path.endswith("/manifest") and response.status == 200:
            manifest = response.json()
            manifest["manifest_schema_version"] = 99
            return Response(200, json.dumps(manifest).encode(), response.headers)
        return response

    client.sync.transport = future_manifests
    client.sync._fetched = None
    with pytest.raises(Exception, match="unsupported schema"):
        client.sync.sync_once()
    assert client.step()[0] is not None


# ── Backup and restore ──────────────────────────────────────────────────────


def test_server_backup_restores_playlists_credentials_and_clients(system):
    office = _ready(system)
    playlist_id = system.store.assignment_for("office").playlist_id
    snapshot = install_modes.snapshot("server", system.root, home=system.root)
    assert snapshot is not None

    # Disaster: the server's state is lost, then restored from the snapshot.
    system.stop()
    for name in ("playlists.json", "provisioned_clients.json", "clients.json"):
        (system.root / ".runtime" / "server" / name).unlink(missing_ok=True)
    restored = install_modes.restore("server", system.root, snapshot, home=system.root)
    assert ".runtime/server/playlists.json" in restored
    assert ".runtime/server/provisioned_clients.json" in restored
    system.start()

    # The client re-enrolls with the credential it already holds.
    system.settle(office)
    assert system.store.assignment_for("office").playlist_id == playlist_id
    assert system.status("office")["lease_state"] == "active"
    assert [s for s, _ in play(office, 3)] == list(PLAYLIST_A["screens"])


def test_the_render_server_never_writes_provider_credentials_to_clients(system, monkeypatch):
    monkeypatch.setenv("OWM_API_KEY", "owm-secret-value-123")
    client = _ready(system)
    cache = system.root / "clients" / "office"
    for path in cache.rglob("*"):
        if path.is_file():
            assert b"owm-secret-value-123" not in path.read_bytes(), path
    for path in cache.rglob("*.json"):
        text = path.read_text(encoding="utf-8", errors="replace")
        assert ADMIN not in text
    assert client.sync.status()["client_id"] == "office"


def test_artifacts_on_the_wire_are_the_pixels_the_standalone_shows(system):
    client = _ready(system, profile="hyperpixel4_square", document={"screens": {"MLB AL Standings": 1},
                                                                    "sequence": []})
    entry = next(e for e in client.sync.active().manifest["artifacts"] if e["screen_id"] == "MLB AL Standings")
    data = client.artifacts.read(entry)
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        assert same(image.convert("RGB"), expected_still(system, "MLB AL Standings", "hyperpixel4_square"))
