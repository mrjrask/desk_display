"""Deterministic tests for demand-aware bounded rendering."""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future

import pytest
from PIL import Image

from display_profiles import PROFILE_PRESETS
from remote_display.artifact_store import ArtifactStore
from remote_display.models import (
    AcceptedRevisions,
    ClientCapabilities,
    ClientDemand,
    ClientStatus,
    PackageCapabilities,
    ScreenRevisions,
)
from remote_display.registry import Assignment, ClientRegistry
from remote_display.render_coordinator import RenderCoordinator, RenderOutput


class Clock:
    def __init__(self):
        self.now = 1_800_000_000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class ManualExecutor:
    """Runs submitted renders only when the test says so."""

    def __init__(self):
        self.pending: list[tuple] = []

    def submit(self, fn):
        future = Future()
        self.pending.append((fn, future))
        return future

    def run_one(self, index=0):
        fn, future = self.pending.pop(index)
        try:
            future.set_result(fn())
        except Exception as exc:  # noqa: BLE001 - delivered through the future
            future.set_exception(exc)

    def run_all(self):
        while self.pending:
            self.run_one()


class FakeRenderer:
    def __init__(self):
        self.calls = []
        self.fail = set()
        self.bad = set()

    def __call__(self, key):
        self.calls.append((key.screen_id, key.render_profile))
        if key.screen_id in self.fail:
            raise RuntimeError(f"{key.screen_id} exploded")
        preset = PROFILE_PRESETS[key.render_profile]
        size = (10, 10) if key.screen_id in self.bad else (preset.width, preset.height)
        color = len(self.calls) % 250
        return RenderOutput(image=Image.new(preset.color_mode, size, color), refresh_seconds=300)


class Revisions:
    def __init__(self):
        self.data = {}

    def __call__(self, screens):
        return {s: ScreenRevisions("s1", self.data.get(s, "d1"), "r1") for s in screens}


def caps(client_id, profile="hyperpixel4"):
    preset = PROFILE_PRESETS[profile]
    return ClientCapabilities(
        protocol_version=1, client_software_version="0.2", client_id=client_id, display_profile=profile,
        logical_width=preset.width, logical_height=preset.height, image_formats=("PNG",),
        color_modes=(preset.color_mode,), render_package_versions=(1,),
    )


def demand(client_id, screens):
    return ClientDemand(
        client_id=client_id, playlist_revision="pl-1", required_screens=tuple(screens),
        package_capabilities=PackageCapabilities(render_package_versions=(1,), image_formats=("PNG",)),
        sync_interval_seconds=30,
    )


@pytest.fixture
def env(tmp_path):
    clock = Clock()
    assignments = {"lobby": Assignment("pl-lobby", "rev-3", ("date",))}
    registry = ClientRegistry(lease_seconds=3600, static_clients={"lobby": "hdmi_1080p"},
                              assignments=assignments.get, clock=clock)
    store = ArtifactStore(tmp_path / "artifacts", grace_seconds=3600, clock=clock)
    renderer, revisions, executor = FakeRenderer(), Revisions(), ManualExecutor()
    coordinator = RenderCoordinator(registry, store, renderer, revisions, workers=2, timeout_seconds=30,
                                    min_interval_seconds=30, executor=executor, clock=clock,
                                    data_health=lambda: {"revision": 7})
    return type("Env", (), dict(clock=clock, registry=registry, store=store, renderer=renderer,
                                revisions=revisions, executor=executor, coordinator=coordinator))


def register(env, client_id, screens, profile="hyperpixel4"):
    return env.registry.register(caps(client_id, profile), demand(client_id, screens))


def drain(env):
    """Tick and run renders until nothing is left to do."""

    for _ in range(20):
        env.coordinator.tick()
        if not env.executor.pending:
            return
        env.executor.run_all()


# ── Deduplication and invalidation ──────────────────────────────────────────


def test_equivalent_demand_renders_once(env):
    register(env, "office", ["date", "weather1"])
    register(env, "den", ["weather1", "date"])
    drain(env)
    assert sorted(env.renderer.calls) == [
        ("date", "hdmi_1080p"), ("date", "hyperpixel4"), ("weather1", "hyperpixel4"),
    ]
    drain(env)
    assert len(env.renderer.calls) == 3  # nothing changed, nothing rerendered
    for screen in ("date", "weather1"):
        assert env.store.resolve(screen, "hyperpixel4").state == "fresh"


def test_queued_and_in_flight_work_is_not_duplicated(env):
    register(env, "office", ["date"])
    env.coordinator.tick()
    env.coordinator.tick()
    assert len(env.executor.pending) == 2  # date for hyperpixel4 and for the static lobby
    register(env, "den", ["date"])
    env.coordinator.tick()
    assert len(env.executor.pending) == 2


def test_only_screens_with_changed_revisions_rerender(env):
    register(env, "office", ["date", "weather1"])
    drain(env)
    env.renderer.calls.clear()
    env.revisions.data["weather1"] = "d2"
    env.clock.advance(31)
    env.coordinator.tick()
    # The old output stays served but is marked stale while the new render runs.
    assert env.store.resolve("weather1", "hyperpixel4").state == "stale"
    env.executor.run_all()
    assert env.renderer.calls == [("weather1", "hyperpixel4")]
    assert env.store.resolve("weather1", "hyperpixel4").state == "fresh"


def test_refresh_deadline_and_minimum_interval(env):
    env.coordinator.screen_min_intervals = {"date": 600}
    register(env, "office", ["date"])
    drain(env)
    env.renderer.calls.clear()
    env.revisions.data["date"] = "d2"
    env.clock.advance(60)
    env.coordinator.tick()
    assert env.executor.pending == []  # changed, but inside date's 600 s minimum interval
    env.clock.advance(541)
    env.coordinator.tick()
    env.executor.run_all()
    assert ("date", "hyperpixel4") in env.renderer.calls
    env.renderer.calls.clear()
    env.revisions.data["date"] = "d2"
    env.clock.advance(301)  # past the refresh deadline, but not the minimum interval
    env.coordinator.tick()
    assert env.executor.pending == []
    env.clock.advance(300)
    env.coordinator.tick()
    assert len(env.executor.pending) >= 1


# ── Priority and concurrency ────────────────────────────────────────────────


def test_missing_output_before_refresh_and_connected_before_static(env):
    env.coordinator.workers = 1
    env.registry.set_disabled("lobby", True)
    registration = register(env, "office", ["weather1"])
    drain(env)
    env.registry.set_disabled("lobby", False)
    env.revisions.data["weather1"] = "d2"
    env.registry.heartbeat(  # the client now also wants "inside"
        "office", registration.credential,
        ClientStatus(client_id="office", playback_state="playing", accepted_revisions=AcceptedRevisions()),
        demand("office", ["weather1", "inside"]),
    )
    env.clock.advance(31)
    env.admin = env.registry.add_prerender("warm", "waveshare_oled_128x64", ["date"])
    env.renderer.calls.clear()
    env.coordinator.tick()
    queue = [(q["screen_id"], q["render_profile"], q["reason"]) for q in env.coordinator.status()["queue"]]
    running = env.coordinator.status()["in_flight"][0]
    assert (running["screen_id"], running["render_profile"]) == ("inside", "hyperpixel4")
    assert queue == [
        ("date", "hdmi_1080p", "missing"),               # static client
        ("date", "waveshare_oled_128x64", "missing"),    # pre-render entry
        ("weather1", "hyperpixel4", "refresh"),
    ]


def test_worker_pool_is_bounded(env):
    register(env, "office", ["date", "weather1", "inside", "quad", "nixie"])
    env.coordinator.tick()
    assert len(env.executor.pending) == 2
    env.executor.run_one()
    assert len(env.executor.pending) == 2  # a finished render frees a slot for the next
    drain(env)
    assert len(env.renderer.calls) == 6


def test_real_thread_pool_never_exceeds_worker_limit(tmp_path):
    clock = time.time
    registry = ClientRegistry(lease_seconds=60, clock=clock)
    store = ArtifactStore(tmp_path / "artifacts", clock=clock)
    active, peak, lock = [0], [0], threading.Lock()

    def slow(key):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.02)
        with lock:
            active[0] -= 1
        preset = PROFILE_PRESETS[key.render_profile]
        return RenderOutput(image=Image.new(preset.color_mode, (preset.width, preset.height)))

    coordinator = RenderCoordinator(registry, store, slow, Revisions(), workers=2)
    registry.register(caps("office"), demand("office", ["date", "weather1", "inside", "quad", "nixie", "inside"]))
    deadline = time.time() + 10
    while time.time() < deadline:
        coordinator.tick()
        if all(store.resolve(s, "hyperpixel4").state == "fresh" for s in ("date", "weather1", "inside", "quad", "nixie")):
            break
        time.sleep(0.01)
    coordinator.stop()
    assert peak[0] <= 2
    assert all(store.resolve(s, "hyperpixel4").record for s in ("date", "weather1", "inside", "quad", "nixie"))


# ── Lease expiry and cancellation ───────────────────────────────────────────


def test_lease_expiry_cancels_queued_and_discards_in_flight(env):
    env.coordinator.workers = 1
    register(env, "office", ["weather1", "inside"])
    env.registry.set_disabled("lobby", True)
    env.coordinator.tick()
    assert len(env.executor.pending) == 1 and len(env.coordinator.status()["queue"]) == 1
    env.clock.advance(3601)  # office's lease lapses
    env.coordinator.tick()
    assert env.coordinator.status()["queue"] == []
    assert env.coordinator.status()["in_flight"][0]["cancelled"] is True
    env.executor.run_all()
    assert env.store.resolve("inside", "hyperpixel4").record is None
    assert env.store.resolve("weather1", "hyperpixel4").record is None
    env.coordinator.tick()
    assert env.executor.pending == []  # an inactive client consumes nothing


def test_unreferenced_output_is_collected_after_demand_ends(env):
    env.store.previous_revisions = 0
    register(env, "office", ["weather1"])
    drain(env)
    first = env.store.resolve("weather1", "hyperpixel4").record
    env.revisions.data["weather1"] = "d2"
    env.clock.advance(31)
    drain(env)
    env.clock.advance(3601)
    assert first.name in env.store.collect_garbage()


# ── Failures ────────────────────────────────────────────────────────────────


def test_failure_keeps_last_good_output_and_backs_off(env):
    register(env, "office", ["weather1"])
    drain(env)
    good = env.store.resolve("weather1", "hyperpixel4").record
    env.renderer.fail.add("weather1")
    env.revisions.data["weather1"] = "d2"
    env.clock.advance(31)
    env.renderer.calls.clear()
    drain(env)
    resolved = env.store.resolve("weather1", "hyperpixel4")
    assert resolved.state == "fallback" and resolved.record == good
    assert resolved.failure["code"] == "render_error"
    assert env.renderer.calls == [("weather1", "hyperpixel4")]
    env.clock.advance(29)
    env.coordinator.tick()
    assert env.executor.pending == []  # backing off
    env.clock.advance(2)
    env.coordinator.tick()
    env.executor.run_all()  # second failure doubles the back-off to 60 s
    env.clock.advance(59)
    env.coordinator.tick()
    assert env.executor.pending == []
    env.renderer.fail.clear()
    env.clock.advance(2)
    drain(env)
    assert env.store.resolve("weather1", "hyperpixel4").state == "fresh"
    lineage = next(l for l in env.coordinator.status()["lineages"] if l["screen_id"] == "weather1")
    assert lineage["failures"] == 2 and lineage["consecutive_failures"] == 0 and lineage["renders"] == 2


def test_invalid_output_is_rejected_and_counted(env):
    register(env, "office", ["weather1"])
    env.renderer.bad.add("weather1")
    drain(env)
    resolved = env.store.resolve("weather1", "hyperpixel4")
    assert resolved.state == "missing" and resolved.failure["code"] == "wrong_dimensions"
    lineage = next(l for l in env.coordinator.status()["lineages"] if l["screen_id"] == "weather1")
    assert lineage["last_error"] == "wrong_dimensions"


def test_timeout_is_a_failure_and_late_results_are_discarded(env):
    register(env, "office", ["weather1"])
    env.registry.set_disabled("lobby", True)
    env.coordinator.tick()
    env.clock.advance(31)
    env.coordinator.tick()
    status = env.coordinator.status()
    assert status["in_flight"][0]["timed_out"] is True
    assert env.store.resolve("weather1", "hyperpixel4").failure["code"] == "timeout"
    env.executor.run_all()
    assert env.store.resolve("weather1", "hyperpixel4").record is None


def test_invalid_demand_does_not_block_other_clients(env):
    register(env, "office", ["date"])
    register(env, "den", ["weather1"])
    original = env.revisions.__call__

    class Partial(Revisions):
        def __call__(self, screens):
            return {s: r for s, r in original(screens).items() if s != "weather1"}

    env.coordinator.revisions = Partial()
    drain(env)
    assert ("date", "hyperpixel4") in env.renderer.calls
    invalid = env.coordinator.status()["invalid_demand"]
    assert [i["client_id"] for i in invalid] == ["den"]


# ── Status ──────────────────────────────────────────────────────────────────


def test_status_reports_demand_health_and_playlist_delivery(env):
    registration = register(env, "office", ["date"])
    env.registry.mark_delivered("office", None)
    env.registry.heartbeat("office", registration.credential, ClientStatus(
        client_id="office", playback_state="playing",
        accepted_revisions=AcceptedRevisions(playlist_revision=None),
    ))
    env.registry.mark_delivered("lobby", "rev-3")
    drain(env)
    status = env.coordinator.status()
    assert status["workers"] == 2 and status["render_keys"] == 2
    assert status["profiles"] == {"hdmi_1080p": 1, "hyperpixel4": 1}
    assert status["clients"] == {"active": 1, "static": 1}
    assert status["artifact_states"] == {"fresh": 2}
    assert status["data_health"] == {"revision": 7}
    assert status["artifacts"]["objects"] == 2
    lineage = status["lineages"][0]
    assert lineage["renders"] == 1 and lineage["last_success"] and lineage["average_duration_ms"] == 0
    playlists = {row["client_id"]: row for row in status["playlists"]}
    assert playlists["office"]["state"] == "unassigned"
    assert playlists["lobby"]["state"] == "pending_acknowledgment"
    assert playlists["lobby"]["delivered_revision"] == "rev-3"


# ── Server integration ──────────────────────────────────────────────────────


def test_admin_render_status_endpoint(tmp_path):
    pytest.importorskip("flask")
    import display_server

    config = display_server.DisplayServerConfig(
        auth_token="server-token-" + "s" * 32, admin_token="admin-token-" + "a" * 32,
        artifact_dir=tmp_path / "artifacts", render_workers=1,
    )
    executor = ManualExecutor()
    app = display_server.create_app(config, renderer=FakeRenderer(), revisions=Revisions(), render_executor=executor)
    api = app.test_client()
    admin = {"Authorization": "Bearer " + config.admin_token}
    assert api.get("/api/v1/admin/render-status").status_code == 401
    response = api.post("/api/v1/register", headers={"Authorization": "Bearer " + config.auth_token},
                        json={"capabilities": caps("office").to_wire(), "demand": demand("office", ["date"]).to_wire()})
    assert response.status_code == 201
    coordinator = app.extensions["desk_display_render_coordinator"]
    coordinator.tick()
    executor.run_all()
    status = api.get("/api/v1/admin/render-status", headers=admin).get_json()
    assert status["artifact_states"] == {"fresh": 1}
    credential = response.get_json()["client_credential"]
    manifest = api.get("/api/v1/clients/office/manifest", headers={"Authorization": "Bearer " + credential}).get_json()
    assert manifest["state"] == "fresh" and manifest["artifacts"][0]["screen_id"] == "date"

    plain = display_server.create_app(config)
    assert plain.test_client().get("/api/v1/admin/render-status", headers=admin).status_code == 404


# ── Production adapter ──────────────────────────────────────────────────────


def test_server_rendering_revisions_follow_style_and_data(tmp_path):
    from remote_display.server_rendering import ServerRendering, StyleRevision
    from services.data_coordinator import DataCoordinator

    style = tmp_path / "style.json"
    style.write_text("{}", encoding="utf-8")
    data = DataCoordinator()
    rendering = ServerRendering(data, StyleRevision([style, tmp_path / "missing.json"]))
    first = rendering.revisions(["date"])["date"]
    assert first.data_revision == "d0" and first.renderer_revision.startswith("v")
    data.publish("weather", {"temp": 70})
    assert rendering.revisions(["date"])["date"].data_revision == "d1"
    style.write_text('{"date": {"font": "big"}}', encoding="utf-8")
    assert rendering.revisions(["date"])["date"].style_revision != first.style_revision
    assert rendering.health()["sources"] == {"weather": 1}


def test_server_rendering_renders_through_screen_renderer(monkeypatch):
    import rendering.screen_renderer as screen_renderer
    from remote_display.models import RenderKey
    from remote_display.server_rendering import ServerRendering
    from services.data_coordinator import DataCoordinator

    seen = {}

    class Stub:
        def render(self, screen, profile, preferences, data, **kwargs):
            seen.update(screen=screen, profile=profile.profile_id, revision=data.revision)
            image = Image.new(profile.color_mode, (profile.width, profile.height))
            return type("A", (), {"image": image, "metadata": {"animation": {"frames": 2}, "secret": "x"}})

    monkeypatch.setattr(screen_renderer, "ScreenRenderer", Stub)
    rendering = ServerRendering(DataCoordinator())
    key = RenderKey.for_screen("weather1", "hyperpixel4", ScreenRevisions("s", "d0", "r"))
    output = rendering.render(key)
    assert seen == {"screen": "weather1", "profile": "hyperpixel4", "revision": 0}
    assert output.refresh_seconds == 300 and output.metadata == {"animation": {"frames": 2}}
    assert output.package is None

    # Clocks are drawn by the client; the server never runs the clock screen.
    seen.clear()
    clock = rendering.render(RenderKey.for_screen("date", "hyperpixel4", ScreenRevisions("s", "d0", "r")))
    assert seen == {} and clock.refresh_seconds == 60 and clock.package["kind"] == "clock"
