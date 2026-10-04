"""Thin client tests: sync, cache, validation and cached playback.

The client talks to a real :mod:`display_server` app through a transport
adapter, so these tests exercise the actual wire protocol.
"""
from __future__ import annotations

import hashlib
import json
import random
import threading
import time

import pytest

pytest.importorskip("flask")

import display_client
import display_server
from display_profiles import PROFILE_PRESETS
from remote_display.client_cache import ClientCache
from remote_display.client_sync import (
    ArtifactCache,
    Backoff,
    ClientSync,
    Response,
    TooLargeError,
    TransportError,
)
from remote_display.models import RenderKey, ScreenRevisions
from remote_display.playlist_store import PlaylistStore

TOKEN = "server-token-" + "s" * 32
PROFILE = PROFILE_PRESETS["hyperpixel4"]
DOC = {"screens": {"date": 1, "weather1": 1}, "sequence": []}


class Clock:
    def __init__(self):
        self.now = 1_800_000_000.0

    def __call__(self):
        return self.now


class FlaskTransport:
    """Route client requests into the server app, with fault injection."""

    def __init__(self, app):
        self.api = app.test_client()
        self.down = False
        self.corrupt = False
        self.truncate = False
        self.block: threading.Event | None = None
        self.calls: list[tuple[str, str]] = []

    def __call__(self, method, path, *, headers=None, json_body=None, max_bytes=16 << 20):
        if self.block is not None:
            self.block.wait(5)
        if self.down:
            raise TransportError("connection refused")
        self.calls.append((method, path))
        response = self.api.open(path, method=method, headers=headers or {}, json=json_body)
        body = response.get_data()
        if "/artifacts/" in path:
            if self.corrupt:
                body = body[:-1] + bytes([body[-1] ^ 0xFF])
            if self.truncate:
                body = body[: len(body) // 2]
        if len(body) > max_bytes:
            raise TooLargeError(f"response exceeds {max_bytes} bytes")
        return Response(response.status_code, body, dict(response.headers))


class Presenter:
    def __init__(self):
        self.frames = []

    def present(self, image):
        self.frames.append(image)
        return image


@pytest.fixture
def env(tmp_path):
    server_clock = Clock()
    store_path = tmp_path / "playlists.json"
    store = PlaylistStore(store_path)
    playlist = store.create("Office", DOC, actor="test")
    store.assign("office", playlist["id"], expected_playlist_id=None, actor="test")
    config = display_server.DisplayServerConfig(enrollment="shared",
        auth_token=TOKEN, admin_token="admin-token-" + "a" * 32, lease_seconds=300,
        artifact_dir=tmp_path / "server-artifacts", playlist_store_path=store_path,
    )
    app = display_server.create_app(config, clock=server_clock)
    transport = FlaskTransport(app)

    def publish(screen, color=0):
        key = RenderKey.for_screen(screen, PROFILE.profile_id, ScreenRevisions("s1", f"d{color}", "r1"))
        from PIL import Image

        image = Image.new(PROFILE.color_mode, (PROFILE.width, PROFILE.height), color)
        return app.extensions["desk_display_artifacts"].publish_image(key, image)

    def make_client(**settings):
        values = {
            "DESK_DISPLAY_PROFILE": PROFILE.profile_id,
            "DESK_DISPLAY_CLIENT_ID": "office",
            "DESK_DISPLAY_SERVER_URL": "https://render.lan:8765/base?x=1",
            "DESK_DISPLAY_CLIENT_TOKEN": TOKEN,
            "DESK_DISPLAY_CLIENT_CACHE_DIR": str(tmp_path / "client"),
            "DESK_DISPLAY_CLIENT_CACHE_MAX_MB": 16,
        }
        values.update(settings)
        client = display_client.build_client(values, presenter=Presenter(), transport=transport)
        client.sync.backoff = Backoff(rng=random.Random(1))
        return client

    class Env:
        pass

    result = Env()
    result.__dict__.update(app=app, store=store, playlist=playlist, transport=transport, publish=publish,
                           make_client=make_client, server_clock=server_clock, tmp=tmp_path)
    return result


def synced(env, client, passes=2):
    for _ in range(passes):
        client.sync.sync_once()
    return client


def test_built_client_supplies_its_device_ip_to_clock_screens(env, monkeypatch):
    from services import wifi_utils

    monkeypatch.setattr(wifi_utils, "get_assigned_ipv4", lambda: "192.168.1.44")

    client = env.make_client()

    assert client._ip_text() == "IP: 192.168.1.44"


def test_client_ip_text_has_a_visible_fallback(monkeypatch):
    from services import wifi_utils

    monkeypatch.setattr(wifi_utils, "get_assigned_ipv4", lambda: None)

    assert display_client._client_ip_text() == "IP: --"


# ── Cold start, warm start and outages ─────────────────────────────────────


def test_cold_start_without_server_shows_a_safe_diagnostic(env, monkeypatch):
    env.transport.down = True
    client = env.make_client()
    drawn = []
    real = display_client.diagnostic_image
    monkeypatch.setattr(display_client, "diagnostic_image", lambda profile, lines: drawn.append(lines) or real(profile, lines))
    assert client.sync.step() > 0
    screen, _seconds = client.step()
    assert screen is None
    frame = client.presenter.frames[-1]
    assert frame.size == (PROFILE.width, PROFILE.height)
    text = "\n".join(drawn[-1])
    assert "office" in text and "render.lan:8765" in text and "server unreachable" in text
    assert TOKEN not in text and "base" not in text and "x=1" not in text
    assert client.report.playback_state == "starting"


def test_cold_start_syncs_then_plays_artifacts(env):
    env.publish("date", 10)
    env.publish("weather1", 20)
    client = env.make_client()
    client.step()  # diagnostic before the first sync
    synced(env, client)
    screens = {client.step()[0] for _ in range(4)}
    assert screens == {"date", "weather1"}
    assert len({frame.getpixel((0, 0)) for frame in client.presenter.frames[1:]}) == 2
    assert client.report.playback_state == "playing"


def test_played_screens_feed_the_config_ui_screenshots(env):
    from remote_display.client_screenshots import ClientScreenshots

    env.publish("date", 10)
    env.publish("weather1", 20)
    client = env.make_client()
    client.screenshots = ClientScreenshots(env.tmp / "shots", profile_id=PROFILE.profile_id,
                                           width=PROFILE.width, height=PROFILE.height)
    client.step()  # diagnostic before the first sync: not a screen, so no screenshot
    assert not (env.tmp / "shots" / "current").exists()
    synced(env, client)
    shown = {client.step()[0] for _ in range(4)}
    current = env.tmp / "shots" / "current"
    assert {p.stem for p in current.glob("*.png")} == shown == {"date", "weather1"}
    status = json.loads((current / "display_status.json").read_text())
    assert status["screen_id"] == client.report.current_screen


def test_activation_waits_until_something_is_usable(env):
    """Nothing rendered yet (a brand new server) still waits."""

    client = env.make_client()
    synced(env, client)
    assert client.sync.active().playlist is None
    assert client.sync.errors.summaries()[0].code == "artifacts_unavailable"
    env.publish("date")
    synced(env, client, 1)
    assert client.sync.active().playlist.playlist_revision == env.playlist["revision"]


def test_activation_does_not_wait_for_every_artifact(env):
    """A partly-usable playlist activates: real rotations include screens that
    are legitimately unavailable much of the time (no active weather alert,
    an out-of-season team's "live" screen), not merely not-yet-rendered, and
    ClientPlayer already skips whatever has no cached package."""

    env.publish("date")
    client = env.make_client()
    synced(env, client)
    assert client.sync.active().playlist.playlist_revision == env.playlist["revision"]
    screen, _seconds = client.step()
    assert screen == "date"
    env.publish("weather1")
    synced(env, client, 1)
    screens = {client.step()[0] for _ in range(4)}
    assert screens == {"date", "weather1"}


def test_a_manifest_entry_without_a_verified_local_copy_is_never_played(env):
    """A manifest can advertise a screen with a sha256 while its local copy
    failed to download, is still in flight, or was corrupted on disk. That
    screen must never be selected for playback just because the playlist
    activated on the strength of some other screen: it must be treated as
    unavailable, the same as a screen absent from the manifest altogether,
    rather than shown as a broken frame."""

    env.publish("date", 10)
    env.publish("weather1", 20)
    client = synced(env, env.make_client())
    entry = client.sync.active().entry("weather1")
    artifact_path = client.sync.artifacts.path_for(entry)
    artifact_path.write_bytes(b"garbage")
    client._content_revision = None  # force _rebuild() to reconsider the manifest
    client.step()  # rebuilds the player against the corrupted manifest
    # weather1's slot is skipped like any other unavailable screen (it may
    # show a transient "cached content unavailable" diagnostic rather than
    # a frame), but it must never come back as a screen actually shown.
    screens = {client.step()[0] for _ in range(6)}
    assert "date" in screens
    assert "weather1" not in screens


def test_warm_client_starts_and_plays_without_its_server(env):
    env.publish("date")
    env.publish("weather1")
    synced(env, env.make_client())
    env.transport.down = True
    client = env.make_client()  # restart
    assert client.step()[0] in {"date", "weather1"}
    delay = client.sync.step()
    assert delay > 0 and not client.sync.connected
    client.step()
    assert client.report.playback_state == "offline"
    assert client.sync.status()["recent_errors"][0]["code"] == "server_unreachable"


def test_offline_start_off_waits_for_a_first_sync(env):
    env.publish("date")
    env.publish("weather1")
    synced(env, env.make_client())
    env.transport.down = True
    client = env.make_client(DESK_DISPLAY_OFFLINE_START=False)  # restart without the server
    assert client.step()[0] is None
    assert client.report.playback_state == "starting"
    env.transport.down = False
    client.sync.step()
    assert client.step()[0] in {"date", "weather1"}
    env.transport.down = True  # a later outage keeps playing the cache
    client.sync.step()
    assert client.step()[0] in {"date", "weather1"}


def test_offline_start_off_waits_until_current_content_activates(env):
    env.publish("date")
    env.publish("weather1")
    synced(env, env.make_client())
    # The server moves on to a playlist whose artifacts are not rendered yet.
    second = env.store.create("Den", {"screens": {"news headlines": 1}, "sequence": []}, actor="test")
    env.store.assign("office", second["id"], expected_playlist_id=env.playlist["id"], actor="test")
    client = env.make_client(DESK_DISPLAY_OFFLINE_START="0")
    client.sync.step()
    assert client.sync.last_sync_age() is not None and not client.sync.confirmed
    assert client.step()[0] is None  # the stale cache stays off the panel
    env.publish("news headlines")
    client.sync.sync_once()
    assert client.sync.confirmed
    assert client.step()[0] == "news headlines"


def test_reconnects_after_an_outage(env):
    env.publish("date")
    env.publish("weather1")
    client = synced(env, env.make_client())
    env.transport.down = True
    first = client.sync.step()
    second = client.sync.step()
    assert client.sync.backoff.failures == 2 and second > 0 and first > 0
    env.transport.down = False
    assert client.sync.step() == pytest.approx(client.sync.sync_interval_seconds, abs=1)
    assert client.sync.backoff.failures == 0 and client.sync.connected


def test_sync_never_blocks_display_updates(env):
    env.publish("date")
    env.publish("weather1")
    client = synced(env, env.make_client())
    env.transport.block = threading.Event()
    worker = threading.Thread(target=client.sync.step)
    worker.start()
    try:
        started = time.monotonic()
        assert client.step()[0] in {"date", "weather1"}
        client.on_button("A")
        client.wait(10)  # a button press ends the hold at once
        assert client.step()[0] in {"date", "weather1"}
        assert time.monotonic() - started < 1
    finally:
        env.transport.block.set()
        worker.join()


def test_backoff_is_exponential_with_jitter():
    backoff = Backoff(base=2, maximum=60, rng=random.Random(3))
    delays = [backoff.failure() for _ in range(8)]
    for attempt, delay in enumerate(delays, start=1):
        ceiling = min(60, 2 * 2 ** (attempt - 1))
        assert ceiling / 2 <= delay <= ceiling
    backoff.success()
    assert backoff.failure() <= 2


def test_duplicate_lease_waits_for_retry_after(env):
    env.publish("date")
    env.publish("weather1")
    synced(env, env.make_client())
    (env.tmp / "client" / "client_credential").unlink()
    client = env.make_client()
    delay = client.sync.step()
    assert delay >= 1
    assert client.sync.errors.summaries()[0].code == "client_id_in_use"


# ── Credentials ────────────────────────────────────────────────────────────


def test_restart_renews_with_the_stored_credential(env):
    env.publish("date")
    env.publish("weather1")
    synced(env, env.make_client())
    path = env.tmp / "client" / "client_credential"
    assert path.stat().st_mode & 0o777 == 0o600
    client = env.make_client()
    synced(env, client, 1)
    assert client.sync.connected


def test_expired_credential_registers_again(env):
    env.publish("date")
    env.publish("weather1")
    client = synced(env, env.make_client())
    old = (env.tmp / "client" / "client_credential").read_text()
    env.server_clock.now += 10_000  # lease expires
    env.transport.calls.clear()
    synced(env, client, 1)
    assert ("POST", "/api/v1/register") in env.transport.calls
    assert (env.tmp / "client" / "client_credential").read_text() != old


def test_status_reports_accepted_revisions_without_secrets(env):
    env.publish("date")
    env.publish("weather1")
    client = synced(env, env.make_client(DISPLAY_ROTATION=90), 3)
    status = client.sync.status()
    assert status["accepted_revisions"]["playlist_revision"] == env.playlist["revision"]
    assert status["physical_rotation"] == 90
    credential = (env.tmp / "client" / "client_credential").read_text()
    assert TOKEN not in json.dumps(status) and credential not in json.dumps(status)


def test_client_role_holds_no_upstream_provider_credentials():
    import deployment_config

    client_settings = deployment_config.settings_for_role(deployment_config.Role.CLIENT)
    # The enrollment token, the local feed-upload token and the Screenshots page login;
    # no provider API keys.
    assert {s.name for s in client_settings if s.secret} <= {
        "DESK_DISPLAY_CLIENT_TOKEN", "FEED_UPLOAD_TOKEN", "SCREEN_UI_PASSWORD", "SCREEN_SESSION_SECRET",
    }
    assert not [s.name for s in client_settings if "API_KEY" in s.name]


# ── Download validation ────────────────────────────────────────────────────


def test_corrupt_download_is_rejected(env):
    env.publish("date")
    env.publish("weather1")
    env.transport.corrupt = True
    client = synced(env, env.make_client())
    assert client.sync.active().playlist is None
    codes = {e.code for e in client.sync.errors.summaries()}
    assert "checksum_mismatch" in codes
    assert not list((env.tmp / "client").glob("artifacts/*"))


def test_interrupted_download_is_rejected_and_retried(env):
    env.publish("date")
    env.publish("weather1")
    env.transport.truncate = True
    client = synced(env, env.make_client())
    assert "length_mismatch" in {e.code for e in client.sync.errors.summaries()}
    assert client.sync.active().playlist is None
    env.transport.truncate = False
    synced(env, client, 1)
    assert client.sync.active().playlist is not None


def test_artifact_for_another_profile_is_rejected(tmp_path):
    from PIL import Image

    from remote_display.client_sync import InvalidArtifact, validate_artifact

    image = Image.new("RGB", (10, 10))
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()
    entry = {"artifact_type": "static_image", "media_type": "image/png", "length": len(data),
             "sha256": hashlib.sha256(data).hexdigest(), "width": 10, "height": 10, "color_mode": "RGB"}
    validate_artifact(data, entry, {"logical_width": 10, "logical_height": 10, "color_mode": "RGB"})
    with pytest.raises(InvalidArtifact) as excinfo:
        validate_artifact(data, entry, {"logical_width": 20, "logical_height": 10, "color_mode": "RGB"})
    assert excinfo.value.code == "wrong_dimensions"
    with pytest.raises(InvalidArtifact) as excinfo:
        validate_artifact(data, {**entry, "artifact_type": "render_package"}, {})
    assert excinfo.value.code == "unsupported_artifact"


def test_oversized_download_is_refused(env):
    env.publish("date")
    env.publish("weather1")
    client = env.make_client()
    real = env.transport.__call__

    def tiny(method, path, **kwargs):
        if "/artifacts/" in path:
            kwargs["max_bytes"] = 10
        return real(method, path, **kwargs)

    client.sync.transport = tiny
    client.sync.sync_once()
    assert "too_large" in {e.code for e in client.sync.errors.summaries()}


# ── Rollback, eviction and rotation ────────────────────────────────────────


def test_corrupt_active_cache_rolls_back_to_the_previous_copy(env):
    env.publish("date")
    env.publish("weather1")
    first = synced(env, env.make_client()).sync.active()
    updated = dict(DOC, screens={"date": 1})
    env.store.update(env.playlist["id"], updated, expected_revision=env.playlist["revision"], actor="test")
    second = synced(env, env.make_client()).sync.active()
    assert second.playlist.playlist_revision != first.playlist.playlist_revision
    (env.tmp / "client" / "playlist" / "current.json").write_text("{", encoding="utf-8")
    (env.tmp / "client" / "manifests" / "0.json").write_text("{", encoding="utf-8")
    env.transport.down = True
    restarted = env.make_client()
    active = restarted.sync.active()
    assert active.playlist.playlist_revision == first.playlist.playlist_revision
    assert restarted.step()[0] in {"date", "weather1"}


def test_eviction_never_removes_active_content(env):
    env.publish("date")
    env.publish("weather1")
    client = synced(env, env.make_client())
    artifacts = client.sync.artifacts
    objects = env.tmp / "client" / "artifacts"
    for index in range(5):
        (objects / f"{index:064x}.png").write_bytes(b"x" * 4096)
    artifacts.max_bytes = 0
    removed = artifacts.evict()
    assert len(removed) == 5
    active = client.sync.active()
    for screen in ("date", "weather1"):
        assert artifacts.has(active.entry(screen))
    assert client.step()[0] in {"date", "weather1"}


def test_eviction_keeps_prior_manifests_within_the_reserve(env):
    env.publish("date", 1)
    env.publish("weather1", 5)
    client = synced(env, env.make_client())
    old = client.sync.active().entry("date")
    env.publish("date", 2)
    synced(env, client, 2)
    assert client.sync.active().entry("date")["sha256"] != old["sha256"]
    client.sync.artifacts.max_bytes = 0
    client.sync.artifacts.evict()
    assert client.sync.artifacts.has(old)  # previous manifest is last-known-good
    client.sync.artifacts.keep_manifests = 1
    client.sync.artifacts.evict()
    assert not client.sync.artifacts.has(old)


def test_rotation_is_left_to_final_presentation(env):
    env.publish("date")
    env.publish("weather1")
    client = synced(env, env.make_client(DISPLAY_ROTATION=3))
    client.step()
    assert client.physical_rotation == 270
    assert client.presenter.frames[-1].size == (PROFILE.width, PROFILE.height)


def test_corrupt_cached_artifact_falls_back_to_diagnostic(env):
    env.publish("date")
    env.publish("weather1")
    client = synced(env, env.make_client())
    for path in (env.tmp / "client" / "artifacts").iterdir():
        path.write_bytes(b"garbage")
    env.transport.down = True
    screen, _ = client.step()
    assert screen is None
    assert client.report.playback_state == "error"


def test_playback_state_survives_restart(env):
    env.publish("date")
    env.publish("weather1")
    client = synced(env, env.make_client())
    shown = [client.step()[0] for _ in range(3)]
    env.transport.down = True
    restarted = env.make_client()
    restarted.step()
    assert restarted.playback.history[: len(shown)] == shown


def test_unassigned_client_keeps_its_cached_playlist(env):
    env.publish("date")
    env.publish("weather1")
    client = synced(env, env.make_client())
    env.store.assign("office", None, expected_playlist_id=env.playlist["id"], actor="test")
    synced(env, client, 1)
    assert client.sync.active().playlist.playlist_id == env.playlist["id"]
    assert ClientCache(env.tmp / "client").load().playlist_id == env.playlist["id"]
    assert client.sync.unassigned
    assert not [e for e in client.sync.errors.summaries() if e.code == "artifacts_unavailable"]


def test_unassigned_client_stops_server_demand_and_new_artifacts(env):
    env.publish("date")
    env.publish("weather1")
    client = synced(env, env.make_client())
    registry = env.app.extensions["desk_display_registry"]
    assert [entry.client_id for entry in registry.demand_entries()] == ["office"]
    env.store.assign("office", None, expected_playlist_id=env.playlist["id"], actor="test")
    active = client.sync.active()
    synced(env, client, 1)
    assert registry.demand_entries() == []
    assert registry.render_plan({s: ScreenRevisions("s1", "d9", "r1") for s in ("date", "weather1")}) == {}
    env.publish("date", 99)  # new server content is no longer delivered
    synced(env, client, 1)
    assert client.sync.active().manifest == active.manifest
    # Reassigning resumes demand.
    env.store.assign("office", env.playlist["id"], expected_playlist_id=None, actor="test")
    synced(env, client, 1)
    assert not client.sync.unassigned
    assert [entry.client_id for entry in registry.demand_entries()] == ["office"]


def test_same_length_corruption_is_repaired_by_the_next_sync(env):
    env.publish("date")
    env.publish("weather1")
    client = synced(env, env.make_client())
    client.sync.artifacts._verified.clear()  # as after a restart
    entry = client.sync.active().entry("date")
    path = client.sync.artifacts.path_for(entry)
    path.write_bytes(bytes(len(path.read_bytes())))  # same length, wrong bytes
    assert not client.sync.artifacts.has(entry)
    synced(env, client, 1)
    assert client.sync.artifacts.read(entry) is not None
    assert client.step()[0] in {"date", "weather1"}


def test_rewritten_artifact_is_verified_again(tmp_path):
    cache = ArtifactCache(tmp_path, max_bytes=1 << 20)
    data = b"x" * 10
    entry = {"sha256": hashlib.sha256(data).hexdigest(), "media_type": "image/png", "length": len(data)}
    path = cache.path_for(entry)
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    assert cache.has(entry)
    path.write_bytes(b"y" * 10)
    import os

    os.utime(path, ns=(1, 1))
    assert not cache.has(entry) and not path.exists()


def test_missing_manifest_revision_is_not_paired_with_another(tmp_path):
    cache = ArtifactCache(tmp_path, max_bytes=1 << 20)
    cache.activate_manifest({"manifest_revision": "m-1", "artifacts": []})
    assert cache.manifest("m-1")["manifest_revision"] == "m-1"
    assert cache.manifest("m-gone") is None
    assert cache.manifest()["manifest_revision"] == "m-1"


def test_boot_falls_back_to_the_previous_matched_pair(env):
    env.publish("date", 1)
    env.publish("weather1", 1)
    client = synced(env, env.make_client())
    first = client.sync.active()
    # A second playlist activates with its own manifest.
    second = env.store.create("Den", {"screens": {"date": 1}, "sequence": []}, actor="test")
    env.store.assign("office", second["id"], expected_playlist_id=env.playlist["id"], actor="test")
    env.publish("date", 2)
    synced(env, client, 1)
    current = client.sync.active()
    assert current.playlist.playlist_id == second["id"]
    # Crash window: the newest manifest never reached the disk.
    manifests = client.sync.artifacts.manifests()
    kept = [m for m in manifests if m["manifest_revision"] != current.manifest["manifest_revision"]]
    for index, manifest in enumerate(kept):
        (env.tmp / "client" / "manifests" / f"{index}.json").write_text(json.dumps(manifest))
    for index in range(len(kept), len(manifests)):
        (env.tmp / "client" / "manifests" / f"{index}.json").unlink()
    env.transport.down = True
    restarted = env.make_client()
    active = restarted.sync.active()
    assert active.playlist.playlist_id == env.playlist["id"]
    assert active.manifest["manifest_revision"] == first.manifest["manifest_revision"]
    # What the heartbeat acknowledges is the pair that plays.
    accepted = restarted.sync.status()["accepted_revisions"]
    assert accepted["playlist_revision"] == first.playlist.playlist_revision
    assert accepted["manifest_revision"] == first.manifest["manifest_revision"]
    # Once the server is back, the newer playlist is offered and activated again.
    env.transport.down = False
    synced(env, restarted, 1)
    assert restarted.sync.active().playlist.playlist_id == second["id"]


def test_retry_after_header_is_honoured_without_a_json_body(tmp_path):
    clock = Clock()
    sync = ClientSync(display_client.capabilities_for("office", PROFILE), None, ClientCache(tmp_path),
                      ArtifactCache(tmp_path, max_bytes=0), clock=clock)
    assert sync._fail(Response(429, b"", {"Retry-After": "120"}), "x").retry_after == 120
    body = json.dumps({"error": "rate_limited", "retry_after_seconds": 5}).encode()
    assert sync._fail(Response(429, body, {"retry-after": "30"}), "x").retry_after == 30
    assert sync._fail(Response(429, body, {}), "x").retry_after == 5
    from email.utils import formatdate

    dated = sync._fail(Response(503, b"", {"Retry-After": formatdate(clock.now + 60, usegmt=True)}), "x")
    assert 59 <= dated.retry_after <= 60
    assert sync._fail(Response(429, b"", {"Retry-After": "99999999"}), "x").retry_after == 3600
    assert sync._fail(Response(429, b"", {"Retry-After": "soon"}), "x").retry_after is None


def test_rate_limited_sync_waits_for_the_header(env):
    client = env.make_client()
    client.sync.transport = lambda *a, **k: Response(429, b"", {"Retry-After": "90"})
    assert client.sync.step() >= 90


def test_client_follows_the_server_heartbeat_cadence(env):
    env.publish("date")
    env.publish("weather1")
    client = env.make_client(DESK_DISPLAY_SYNC_INTERVAL_SECONDS=600, DESK_DISPLAY_HEARTBEAT_INTERVAL_SECONDS=600)
    clock = Clock()
    client.sync._clock = clock
    # The server's lease is 300 s, so it advertises a 100 s heartbeat and a 30 s sync.
    assert client.sync.step() == 30
    assert client.sync.effective_heartbeat_interval() == 100
    assert client.sync.effective_sync_interval() == 30


def test_slow_passes_do_not_delay_the_next_heartbeat(env):
    env.publish("date")
    env.publish("weather1")
    client = env.make_client(DESK_DISPLAY_SYNC_INTERVAL_SECONDS=30, DESK_DISPLAY_HEARTBEAT_INTERVAL_SECONDS=10)
    clock = Clock()
    client.sync._clock = clock
    real = env.transport.__call__

    def slow(*args, **kwargs):
        clock.now += 2  # every request takes 2 s
        return real(*args, **kwargs)

    client.sync.transport = slow
    started = clock.now
    delay = client.sync.step()
    assert clock.now + delay == started + 10


def test_heartbeat_only_passes_renew_the_lease_between_syncs(env):
    env.publish("date")
    env.publish("weather1")
    client = env.make_client(DESK_DISPLAY_SYNC_INTERVAL_SECONDS=30, DESK_DISPLAY_HEARTBEAT_INTERVAL_SECONDS=10)
    clock = Clock()
    client.sync._clock = clock
    assert client.sync.step() == 10  # full sync, then a heartbeat is due first
    env.transport.calls.clear()
    clock.now += 10
    assert client.sync.step() == 10
    assert [path.rsplit("/", 1)[-1] for _, path in env.transport.calls] == ["heartbeat"]
    # When the server's manifest moves on, the heartbeat pulls a full sync.
    env.publish("date", 42)
    env.transport.calls.clear()
    clock.now += 10
    client.sync.step()
    assert any(path.endswith("/manifest") for _, path in env.transport.calls)


def test_heartbeats_report_delivery_telemetry(env):
    env.publish("date", 10)
    env.publish("weather1", 20)
    client = env.make_client()
    ticks = iter(range(0, 10_000, 5))
    client.sync._timer = lambda: next(ticks) / 1000  # every reading 5 ms after the last
    synced(env, client, 1)
    registry = env.app.extensions["desk_display_registry"]
    # Registration advertised telemetry, so even the first heartbeat carries it
    # (before any download has been timed).
    assert registry.snapshot()["clients"]["office"]["telemetry"]["download_count"] == 0
    timings = client.sync.telemetry()
    assert timings["download_count"] == 2 and timings["download_bytes"] > 0
    assert timings["download_ms"] > 0 and timings["manifest_fetch_ms"] == 5.0
    assert timings["last_sync_duration_ms"] > timings["download_ms"]
    client.step()  # plays a screen, so the content age is known
    synced(env, client, 1)
    reported = registry.snapshot()["clients"]["office"]["telemetry"]
    assert reported["heartbeat_rtt_ms"] == 5.0
    assert reported["download_count"] == 2
    assert reported["displayed_content_age_seconds"] is not None
    # Nothing new on the next pass: no downloads, and the report says so.
    synced(env, client, 1)
    assert client.sync.telemetry()["download_count"] == 0


def test_client_runs_update_and_restart_from_the_clients_page(env, monkeypatch):
    import subprocess

    from remote_display import client_commands
    from remote_display.client_commands import CommandStore

    commands_file = env.tmp / "commands.json"
    config = display_server.DisplayServerConfig(enrollment="shared", auth_token=TOKEN,
                                                artifact_dir=env.tmp / "server-artifacts",
                                                commands_path=commands_file)
    app = display_server.create_app(config, clock=env.server_clock)
    env.transport.api = app.test_client()
    store = CommandStore(commands_file, clock=env.server_clock)
    client = env.make_client()
    runs = []

    def git(argv, **kwargs):
        runs.append(argv[3:])
        output = "abc1234\n" if argv[3] == "rev-parse" else "Already up to date."
        return subprocess.CompletedProcess(argv, 0, output, "")

    runner = client.sync.commands
    runner._run, runner._background = git, False
    update = store.queue("office", "update", actor="jason")
    synced(env, client, 1)  # registers, then the heartbeat collects the command and runs it
    assert ["pull", "--ff-only"] in runs
    assert store.for_client("office")[0]["state"] == "delivered"
    client.sync.heartbeat_once()  # reports the result
    assert store.for_client("office")[0]["state"] == "succeeded"
    assert "Already up to date (abc1234)." in store.for_client("office")[0]["output"]

    restart = store.queue("office", "restart", actor="jason")
    client.sync.heartbeat_once()
    assert client.restart_requested and client._stop.is_set()
    client.sync.heartbeat_once()
    assert {c["id"]: c["state"] for c in store.for_client("office")}[restart["id"]] == "delivered"
    # The process systemd starts next reports that the restart happened.
    monkeypatch.setattr(client_commands.os, "getpid", lambda: -1)
    restarted = env.make_client()
    synced(env, restarted, 1)
    states = {c["id"]: c["state"] for c in store.for_client("office")}
    assert states == {update["id"]: "succeeded", restart["id"]: "succeeded"}


def test_heartbeats_report_client_resources_for_the_stats_page(env):
    env.publish("date", 10)
    client = env.make_client()
    synced(env, client, 1)
    synced(env, client, 1)
    reported = env.app.extensions["desk_display_registry"].snapshot()["clients"]["office"]["resources"]
    assert reported["type"] == "client_resources"
    # Everything the client exchanged with the server before this heartbeat.
    assert 0 < reported["bytes_received"] <= client.sync.bytes_received
    assert 0 < reported["bytes_sent"] <= client.sync.bytes_sent
    assert reported["cache_limit_bytes"] == client.sync.artifacts.max_bytes
    assert reported["uptime_seconds"] >= 0


def test_no_resources_are_sent_to_a_server_that_does_not_advertise_them(env):
    client = env.make_client()
    client.sync._note_cadence({"client_telemetry_versions": [1]})
    assert client.sync._resources_accepted is False
    client.sync._note_cadence({"client_resource_versions": [2]})
    assert client.sync._resources_accepted is False
    client.sync._note_cadence({"client_resource_versions": [1]})
    assert client.sync._resources_accepted is True


def test_no_telemetry_is_sent_to_a_server_that_does_not_advertise_it(env):
    client = env.make_client()
    client.sync._note_cadence({"heartbeat_interval_seconds": 20})
    assert client.sync._telemetry_accepted is False
    client.sync._note_cadence({"client_telemetry_versions": [2]})
    assert client.sync._telemetry_accepted is False
    client.sync._note_cadence({"client_telemetry_versions": [1]})
    assert client.sync._telemetry_accepted is True


def test_telemetry_counts_failed_passes_until_a_success(env):
    env.publish("date")
    client = env.make_client()
    synced(env, client, 1)
    env.transport.down = True
    client.sync.step()
    client.sync.step()
    assert client.sync.telemetry()["consecutive_failures"] == 2
    env.transport.down = False
    client.sync.step()
    assert client.sync.consecutive_failures == 0
    reported = env.app.extensions["desk_display_registry"].snapshot()["clients"]["office"]["telemetry"]
    assert reported["consecutive_failures"] == 2


def test_artifact_cache_rejects_invalid_hashes(tmp_path):
    cache = ArtifactCache(tmp_path, max_bytes=0)
    assert not cache.has({"sha256": "../../etc/passwd", "media_type": "image/png", "length": 1})


def test_sync_rejects_manifest_for_another_client(env):
    env.publish("date")
    env.publish("weather1")
    client = env.make_client()
    real = env.transport.__call__

    def swap(method, path, **kwargs):
        response = real(method, path, **kwargs)
        if path.endswith("/manifest") and response.status == 200:
            body = json.loads(response.body)
            body["client_id"] = "lobby"
            return Response(200, json.dumps(body).encode(), {})
        return response

    client.sync.transport = swap
    client.sync.step()
    assert client.sync.errors.summaries()[0].code == "invalid_manifest"


def test_client_sync_constructs_without_hardware(tmp_path):
    cache = ClientCache(tmp_path)
    sync = ClientSync(display_client.capabilities_for("office", PROFILE), lambda *a, **k: None, cache,
                      ArtifactCache(tmp_path, max_bytes=1))
    assert sync.active().playlist is None and sync.status()["playback_state"] == "starting"


def test_offline_max_age_stops_showing_expired_content(env):
    env.publish("date")
    env.publish("weather1")
    synced(env, env.make_client())
    env.transport.down = True
    client = env.make_client(DESK_DISPLAY_OFFLINE_MAX_AGE_HOURS=1)
    assert client.step()[0] is not None
    client.sync._clock = lambda: env.server_clock.now + 7200
    assert client.step()[0] is None


# ── Notification LED and border ─────────────────────────────────────────────


class LedPresenter(Presenter):
    def __init__(self):
        super().__init__()
        self.leds = []

    def set_led(self, color):
        self.leds.append(color)

    def apply_indicator_border(self, image):
        from PIL import ImageDraw

        bordered = image.convert("RGB")
        ImageDraw.Draw(bordered).rectangle([(0, 0), (bordered.width - 1, bordered.height - 1)],
                                           outline=(255, 0, 0), width=2)
        return bordered


def test_screen_led_color_reaches_the_client_like_v01(env):
    env.publish("date", 10)
    key = RenderKey.for_screen("weather1", PROFILE.profile_id, ScreenRevisions("s1", "d20", "r1"))
    from PIL import Image

    env.app.extensions["desk_display_artifacts"].publish_image(
        key, Image.new(PROFILE.color_mode, (PROFILE.width, PROFILE.height), 20), metadata={"led": [1.0, 0.0, 0.0]})
    client = env.make_client()
    client.presenter = LedPresenter()
    synced(env, client)
    seen = {}
    for _ in range(4):
        screen, _ = client.step()
        seen[screen] = client._led
    assert seen == {"date": None, "weather1": (1.0, 0.0, 0.0)}
    # Lit once per change, and handed back to the update status after it.
    assert client.presenter.leds[0] == (1.0, 0.0, 0.0)
    assert None in client.presenter.leds
    assert all(a != b for a, b in zip(client.presenter.leds, client.presenter.leds[1:]))


def test_client_screenshots_carry_the_indicator_border(env):
    from remote_display.client_screenshots import ClientScreenshots

    env.publish("date", 10)
    env.publish("weather1", 20)
    client = env.make_client()
    client.presenter = LedPresenter()
    client.screenshots = ClientScreenshots(env.tmp / "shots", profile_id=PROFILE.profile_id,
                                           width=PROFILE.width, height=PROFILE.height)
    synced(env, client)
    screen, _ = client.step()
    from PIL import Image

    with Image.open(env.tmp / "shots" / "current" / f"{screen}.png") as shot:
        assert shot.convert("RGB").getpixel((0, 0)) == (255, 0, 0)


def test_clock_screens_start_the_update_check_at_most_once_a_minute(env):
    calls = []
    done = threading.Event()
    env.publish("date", 10)
    env.publish("weather1", 20)
    client = env.make_client()
    client._update_check = lambda: (calls.append(1), done.set())
    synced(env, client)
    for _ in range(6):
        client.step()
    assert done.wait(2)
    client._update_check_thread.join(2)
    assert calls == [1]
    client._monotonic = lambda: time.monotonic() + display_client.UPDATE_CHECK_SECONDS + 1
    for _ in range(2):
        client.step()
    client._update_check_thread.join(2)
    assert calls == [1, 1]


def test_built_client_only_checks_updates_on_real_hardware(env):
    assert env.make_client()._update_check is None


class ButtonPresenter(Presenter):
    def __init__(self):
        super().__init__()
        self.backlight = []
        self.indicator = True

    def set_backlight(self, level):
        self.backlight.append(level)
        return level

    def toggle_update_indicator(self):
        self.indicator = not self.indicator
        return self.indicator


def test_buttons_do_what_they_did_in_v01(env):
    env.publish("date")
    env.publish("weather1")
    client = synced(env, env.make_client())
    client.presenter = presenter = ButtonPresenter()
    restarts = []
    client._restart_service = lambda: restarts.append(True)

    first = client.step()[0]
    client.on_button("A")  # next screen
    assert client._controls_pending()
    second = client.step()[0]
    assert {first, second} == {"date", "weather1"}

    client.on_button("B")  # display off: blank, backlight off, held until B again
    assert client.step() == (None, display_client.DARK_POLL_SECONDS)
    assert presenter.backlight[-1] == 0.0
    assert presenter.frames[-1].getbbox() is None
    assert client.report.playback_state == "paused"
    client.on_button("A")  # other buttons don't wake it
    assert client.step()[0] is None
    client.on_button("B")
    assert client.step()[0] in {"date", "weather1"}
    assert presenter.backlight[-1] == 1.0

    client.on_button("X")  # update indicator toggle
    client.step()
    assert presenter.indicator is False

    client.on_button("Y")  # restart the service
    client.step()
    assert restarts == [True]


def test_screens_hold_for_v01s_screen_delay(env):
    env.publish("date")
    env.publish("weather1")
    client = synced(env, env.make_client())
    assert display_client.DEFAULT_SCREEN_SECONDS == 4.0
    assert client.step()[1] == pytest.approx(4.0)


class FakeWifi:
    def __init__(self, eligible=True):
        self.eligible = eligible
        self.started = []

    def should_monitor_wifi(self):
        return self.eligible

    def start_monitor(self, allow_recovery=True):
        self.started.append(allow_recovery)


def test_client_starts_the_wifi_monitor_like_v01(monkeypatch):
    monkeypatch.delenv("ENABLE_WIFI_MONITOR", raising=False)
    monkeypatch.delenv("ENABLE_WIFI_RECOVERY", raising=False)
    wifi = FakeWifi()
    assert display_client.start_wifi_monitor({"ENABLE_WIFI_MONITOR": True, "ENABLE_WIFI_RECOVERY": True}, wifi)
    assert wifi.started == [True]

    wifi = FakeWifi()
    settings = {"ENABLE_WIFI_MONITOR": True, "ENABLE_WIFI_RECOVERY": False}
    assert display_client.start_wifi_monitor(settings, wifi) and wifi.started == [False]

    wifi = FakeWifi()
    assert not display_client.start_wifi_monitor({"ENABLE_WIFI_MONITOR": False}, wifi)
    assert not display_client.start_wifi_monitor({}, FakeWifi(eligible=False))
    assert wifi.started == []

    # Low-power panels leave it off unless it is asked for explicitly, as v0.1 did.
    low = {"DESK_DISPLAY_LOW_POWER": True, "ENABLE_WIFI_MONITOR": True}
    assert not display_client.start_wifi_monitor(low, FakeWifi())
    monkeypatch.setenv("ENABLE_WIFI_MONITOR", "1")
    assert display_client.start_wifi_monitor(low, FakeWifi())


def test_offline_client_skips_live_screens_past_their_refresh_deadline(env):
    from datetime import datetime, timedelta, timezone

    second = env.store.create("Scores", {"screens": {"date": 1, "NFL Scoreboard": 1}, "sequence": []}, actor="test")
    env.store.assign("office", second["id"], expected_playlist_id=env.playlist["id"], actor="test")
    env.publish("date", 10)
    env.publish("NFL Scoreboard", 20)
    assert "NFL Scoreboard" in display_client.LIVE_SCREENS and "date" not in display_client.LIVE_SCREENS
    client = env.make_client()
    synced(env, client)
    now = [datetime.fromtimestamp(env.server_clock.now, timezone.utc)]  # when the server rendered
    client._clock = lambda: now[0]

    # Online: both play.
    assert {client.step()[0] for _ in range(4)} == {"date", "NFL Scoreboard"}

    # Offline but still before the refresh deadline: the score is recent enough.
    env.transport.down = True
    client.sync.step()
    assert not client.sync.connected
    assert {client.step()[0] for _ in range(4)} == {"date", "NFL Scoreboard"}

    # Offline past the deadline: the frozen score is skipped, the date still plays.
    now[0] += timedelta(hours=1)
    assert [client.step()[0] for _ in range(4)] == ["date"] * 4

    # Back online: the scoreboard returns.
    env.transport.down = False
    synced(env, client)
    assert "NFL Scoreboard" in {client.step()[0] for _ in range(4)}


def test_a_permanently_rejected_artifact_is_not_downloaded_again(env, monkeypatch):
    from remote_display import client_sync

    env.publish("date", color=1)
    env.publish("weather1", color=2)
    original = client_sync.ArtifactCache.store

    def store(self, entry, data, profile):
        if entry.get("screen_id") == "date":
            raise client_sync.InvalidArtifact("wrong_dimensions", "not for this display")
        return original(self, entry, data, profile)

    monkeypatch.setattr(client_sync.ArtifactCache, "store", store)
    client = synced(env, env.make_client())
    downloads = [p for m, p in env.transport.calls if "/artifacts/" in p]
    assert len(downloads) == 2  # each image once: the rejected one is not fetched on the second pass
    synced(env, client, 1)
    assert [p for m, p in env.transport.calls if "/artifacts/" in p] == downloads
    assert "wrong_dimensions" in {e.code for e in client.sync.errors.summaries()}


# ── Screens drawn from the client's own sensor ─────────────────────────────


def _with_inside(env):
    doc = {"screens": {"date": 1, "inside": 1, "weather1": 1}, "sequence": []}
    playlist = env.store.update(env.playlist["id"], doc, expected_revision=env.playlist["revision"], actor="test")
    env.publish("date", 10)
    env.publish("weather1", 20)
    return playlist


def _sensor(available=True, color=(1, 2, 3)):
    from PIL import Image

    from playback.local_screens import SensorScreen

    return SensorScreen(probe=lambda: available,
                        render=lambda: Image.new("RGB", (PROFILE.width, PROFILE.height), color))


def test_client_draws_the_inside_screen_from_its_own_sensor(env):
    _with_inside(env)
    client = env.make_client()
    sensor = _sensor()
    client.local_screens = {"inside": sensor}
    synced(env, client)
    client.step()  # activates the playlist, which starts the sensor probe
    assert sensor.wait(5)
    shown = {}
    for _ in range(6):
        screen, _seconds = client.step()
        shown[screen] = client.presenter.frames[-1]
    assert set(shown) == {"date", "inside", "weather1"}
    assert shown["inside"].getpixel((0, 0)) == (1, 2, 3)
    assert client.report.playback_state == "playing"


def test_client_without_a_sensor_skips_the_inside_screen(env):
    _with_inside(env)
    client = env.make_client()
    sensor = _sensor(available=False)
    client.local_screens = {"inside": sensor}
    synced(env, client)
    client.step()
    assert sensor.wait(5)
    assert {client.step()[0] for _ in range(6)} == {"date", "weather1"}


def test_inside_screen_plays_while_the_server_is_away(env):
    _with_inside(env)
    client = env.make_client(DESK_DISPLAY_OFFLINE_MAX_AGE_HOURS=0.001)
    sensor = _sensor()
    client.local_screens = {"inside": sensor}
    synced(env, client)
    client.step()
    assert sensor.wait(5)
    env.transport.down = True
    client.sync.step()
    client.sync.cache_age = lambda: 3600.0
    assert client._too_old()  # cached server content has expired; the sensor has not
    assert "inside" in {client.step()[0] for _ in range(6)}


def test_server_never_renders_or_misses_the_inside_screen(env):
    _with_inside(env)
    client = env.make_client()
    synced(env, client)
    manifest = client.sync.active().manifest
    assert "inside" not in manifest["requested_screens"]
    assert "inside" not in manifest["missing_screens"]
    assert all(entry["screen_id"] != "inside" for entry in manifest["artifacts"])
    assert manifest["cache_complete"] is True


def test_sigterm_stops_a_running_client_promptly(env):
    import os
    import signal

    client = env.make_client()
    forced = []
    stopper = display_client.StopOnSignal(grace_seconds=30, force_exit=forced.append)
    previous = signal.getsignal(signal.SIGTERM)
    try:
        stopper.install()
        stopper.target = client.stop
        threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGTERM)).start()
        started = time.monotonic()
        client.run()
        elapsed = time.monotonic() - started
    finally:
        signal.signal(signal.SIGTERM, previous)
        client.stop()
    assert stopper.received and client._stop.is_set()
    assert elapsed < 5
    assert forced == []


def test_sigterm_during_startup_exits():
    stopper = display_client.StopOnSignal(grace_seconds=30, force_exit=lambda code: None)
    import signal

    with pytest.raises(SystemExit):
        stopper.handle(signal.SIGTERM)
    assert stopper.received
    stopper.handle(signal.SIGTERM)  # a second signal is ignored


def test_stalled_stop_after_sigterm_exits_by_force():
    import signal

    forced = threading.Event()
    stopper = display_client.StopOnSignal(grace_seconds=0.05, force_exit=lambda code: forced.set())
    stopper.target = lambda: None  # a stop that never ends the process
    stopper.handle(signal.SIGTERM)
    assert forced.wait(2)


@pytest.mark.parametrize("value, fps", [(None, 0.0), ("", 0.0), ("0", 0.0), ("15", 15.0), (12.5, 12.5),
                                        ("fast", 0.0), ("-3", 0.0)])
def test_max_fps_setting(value, fps):
    settings = {} if value is None else {"DESK_DISPLAY_CLIENT_MAX_FPS": value}
    assert display_client.client_max_fps(settings) == fps
