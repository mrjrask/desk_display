"""Display clients on another network upload screenshots to the server for the collector."""
from __future__ import annotations

import io
import json
import os

import pytest
from PIL import Image

pytest.importorskip("flask")

import display_client  # noqa: E402
import display_server  # noqa: E402
from display_profiles import PROFILE_PRESETS  # noqa: E402
from remote_display.client_cache import ClientCache  # noqa: E402
from remote_display.client_screenshots import ClientScreenshots  # noqa: E402
from remote_display.client_sync import ArtifactCache, ClientSync, Response  # noqa: E402
from remote_display.screenshot_uploads import (  # noqa: E402
    ScreenshotInbox,
    UploadQueue,
    UploadRejected,
    file_name,
    upload_dir,
)

TOKEN = "server-token-" + "s" * 32
PROFILE = PROFILE_PRESETS["hyperpixel4"]


class Clock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now


def png(color=(255, 0, 0), size=(8, 8)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


# ── Server store ────────────────────────────────────────────────────────────


def test_inbox_keeps_only_the_latest_image_per_screen(tmp_path):
    clock = Clock()
    inbox = ScreenshotInbox(tmp_path, clock=clock)
    inbox.save("hyper", "NCAA Mens BB Scoreboard", png((1, 1, 1)))
    inbox.save("hyper", "date", png((2, 2, 2)))
    clock.now += 60
    inbox.save("hyper", "NCAA Mens BB Scoreboard", png((3, 3, 3)))
    screens = inbox.screens("hyper")
    # Upload order (a re-upload moves to the end), one file per screen.
    assert [s["screen"] for s in screens] == ["date", "NCAA Mens BB Scoreboard"]
    assert sorted(p.name for p in (tmp_path / "hyper").iterdir()) == sorted(
        ["index.json", file_name("date"), file_name("NCAA Mens BB Scoreboard")])
    path = inbox.image_path("hyper", screens[1]["file"])
    assert Image.open(path).getpixel((0, 0)) == (3, 3, 3)


def test_inbox_rejects_bad_uploads(tmp_path):
    inbox = ScreenshotInbox(tmp_path, max_image_bytes=200)
    with pytest.raises(UploadRejected) as exc:
        inbox.save("hyper", "date", b"GIF89a" + b"0" * 20)
    assert exc.value.status == 415
    with pytest.raises(UploadRejected):
        inbox.save("hyper", "date", b"\x89PNG\r\n\x1a\n" + b"garbage")
    with pytest.raises(UploadRejected) as exc:
        inbox.save("hyper", "date", png(size=(64, 64)) + b"\0" * 300)
    assert exc.value.status == 413
    with pytest.raises(UploadRejected):
        inbox.save("../etc", "date", png())
    with pytest.raises(UploadRejected):
        inbox.save("hyper", "bad\nscreen", png())
    assert inbox.image_path("hyper", "../index.json") is None
    assert not (tmp_path / "hyper").exists()


def test_inbox_prunes_by_age_and_total_size(tmp_path):
    clock = Clock()
    image = png()
    inbox = ScreenshotInbox(tmp_path, max_total_bytes=len(image) * 3, retention_seconds=3600, clock=clock)
    inbox.save("old", "date", image)
    clock.now += 4000
    removed = inbox.prune()
    assert removed == ["old/date"] and not (tmp_path / "old").exists()
    for index, screen in enumerate(["a", "b", "c", "d"]):
        clock.now += 1
        inbox.save("hyper" if index % 2 else "mini", screen, image)
    # Over the cap, the oldest upload goes first.
    remaining = {s["screen"] for c in inbox.clients() for s in inbox.screens(c)}
    assert remaining == {"b", "c", "d"}
    (tmp_path / "hyper" / "stray.png").write_bytes(b"x")
    inbox.prune()
    assert not (tmp_path / "hyper" / "stray.png").exists()


def test_upload_dir_setting(tmp_path):
    assert upload_dir({"DESK_DISPLAY_SCREENSHOT_UPLOAD_DIR": str(tmp_path)}) == tmp_path
    assert upload_dir({}).parts[-3:] == (".runtime", "server", "client_screenshots")


# ── Server API ──────────────────────────────────────────────────────────────


def make_server(tmp_path, clock, **overrides):
    overrides.setdefault("screenshot_upload_dir", tmp_path / "uploads")
    config = display_server.DisplayServerConfig(
        enrollment="shared", auth_token=TOKEN, artifact_dir=tmp_path / "artifacts", **overrides)
    app = display_server.create_app(config, clock=clock)
    app.config["TESTING"] = True
    return app


def register(api, client_id="hyper"):
    caps = display_client.capabilities_for(client_id, PROFILE).to_wire()
    response = api.post("/api/v1/register", json={"capabilities": caps},
                        headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 201, response.get_json()
    return response.get_json()


def put(api, credential, data, *, screen="date", client_id="hyper", content_type="image/png"):
    return api.put(f"/api/v1/clients/{client_id}/screenshots", query_string={"screen": screen},
                   data=data, headers={"Authorization": f"Bearer {credential}", "Content-Type": content_type})


def test_server_stores_uploads_larger_than_its_json_limit(tmp_path):
    app = make_server(tmp_path, Clock())
    api = app.test_client()
    body = register(api)
    assert body["client_screenshot_upload_versions"] == [1]
    assert body["screenshot_upload_max_bytes"] == 2 * 1024 * 1024
    noisy = Image.frombytes("RGB", (200, 200), os.urandom(200 * 200 * 3))
    buffer = io.BytesIO()
    noisy.save(buffer, format="PNG")
    assert len(buffer.getvalue()) > display_server.MAX_REQUEST_BYTES
    response = put(api, body["client_credential"], buffer.getvalue(), screen="NCAA Mens BB Scoreboard")
    assert response.status_code == 201, response.get_json()
    inbox = app.extensions["desk_display_screenshot_inbox"]
    assert [s["screen"] for s in inbox.screens("hyper")] == ["NCAA Mens BB Scoreboard"]


def test_server_refuses_bad_or_unauthenticated_uploads(tmp_path):
    app = make_server(tmp_path, Clock(), screenshot_upload_max_bytes=1024)
    api = app.test_client()
    credential = register(api)["client_credential"]
    other = register(api, "mini")["client_credential"]
    assert put(api, "wrong", png()).status_code == 401
    assert put(api, other, png()).status_code == 401  # another client's credential
    assert put(api, credential, png(), content_type="image/jpeg").status_code == 415
    assert put(api, credential, b"\x89PNG\r\n\x1a\n" + b"\0" * 2000).status_code == 413
    assert put(api, credential, png(), screen="").status_code == 400
    assert not (tmp_path / "uploads" / "hyper").exists()


def test_uploads_can_be_turned_off(tmp_path):
    app = make_server(tmp_path, Clock(), screenshot_upload_dir=None)
    api = app.test_client()
    body = register(api)
    assert "client_screenshot_upload_versions" not in body
    assert put(api, body["client_credential"], png()).status_code == 404


def test_server_settings(tmp_path):
    config = display_server.DisplayServerConfig.from_env({
        "DESK_DISPLAY_ROLE": "server", "DESK_DISPLAY_SERVER_AUTH_TOKEN": TOKEN,
        "DESK_DISPLAY_SCREENSHOT_UPLOAD_DIR": str(tmp_path), "DESK_DISPLAY_SCREENSHOT_UPLOAD_MAX_KB": "512",
        "DESK_DISPLAY_SCREENSHOT_UPLOAD_MAX_MB": "16", "DESK_DISPLAY_SCREENSHOT_UPLOAD_RETENTION_DAYS": "2",
    })
    assert config.screenshot_upload_dir == tmp_path
    assert config.screenshot_upload_max_bytes == 512 * 1024
    assert config.screenshot_upload_max_total_bytes == 16 * 1024 * 1024
    assert config.screenshot_upload_retention_seconds == 2 * 86400
    off = display_server.DisplayServerConfig.from_env({
        "DESK_DISPLAY_ROLE": "server", "DESK_DISPLAY_SERVER_AUTH_TOKEN": TOKEN,
        "DESK_DISPLAY_SCREENSHOT_UPLOADS": "0"})
    assert off.screenshot_upload_dir is None


# ── Client ──────────────────────────────────────────────────────────────────


class FlaskTransport:
    def __init__(self, app):
        self.api = app.test_client()
        self.calls: list[tuple[str, str]] = []

    def __call__(self, method, path, *, headers=None, json_body=None, max_bytes=16 << 20, data=None):
        self.calls.append((method, path))
        response = self.api.open(path, method=method, headers=headers or {},
                                 **({"data": data} if data is not None else {"json": json_body}))
        return Response(response.status_code, response.get_data(), dict(response.headers))


def test_upload_queue_paces_each_screen(tmp_path):
    clock = Clock()
    queue = UploadQueue(600, clock=clock)
    queue.offer("date", tmp_path / "date.png", clock.now)
    queue.offer("weather1", tmp_path / "weather1.png", clock.now)
    assert [item.screen for item in queue.due()] == ["date", "weather1"]
    assert len(queue.due(limit=1)) == 1
    for item in queue.due():
        queue.mark_uploaded(item)
    clock.now += 60
    queue.offer("date", tmp_path / "date.png", clock.now)
    assert queue.due() == []
    clock.now += 600
    assert [item.screen for item in queue.due()] == ["date"]  # weather1 has nothing newer
    assert UploadQueue(1).interval_seconds == 60  # floor


def test_client_uploads_saved_screenshots_after_a_heartbeat(tmp_path):
    clock = Clock()
    app = make_server(tmp_path, clock)
    transport = FlaskTransport(app)
    queue = UploadQueue(600, clock=clock)
    shots = ClientScreenshots(tmp_path / "shots", profile_id=PROFILE.profile_id, width=PROFILE.width,
                              height=PROFILE.height)
    shots.uploads = queue
    sync = ClientSync(display_client.capabilities_for("hyper", PROFILE), transport,
                      ClientCache(tmp_path / "client"), ArtifactCache(tmp_path / "client", max_bytes=1 << 20),
                      enrollment_token=TOKEN, clock=clock, screenshot_uploads=queue)
    shots.record("date", Image.new("RGB", (PROFILE.width, PROFILE.height), (9, 9, 9)))
    shots.record("NCAA Mens BB Scoreboard", Image.new("RGB", (PROFILE.width, PROFILE.height), (5, 5, 5)))
    sync.step()
    puts = [path for method, path in transport.calls if method == "PUT"]
    assert len(puts) == 2 and all("/api/v1/clients/hyper/screenshots?screen=" in p for p in puts)
    inbox = app.extensions["desk_display_screenshot_inbox"]
    screens = {s["screen"]: s for s in inbox.screens("hyper")}
    assert set(screens) == {"date", "NCAA Mens BB Scoreboard"}
    assert screens["date"]["width"] == PROFILE.width
    # Paced: the next pass uploads nothing until the interval passes.
    transport.calls.clear()
    shots.record("date", Image.new("RGB", (PROFILE.width, PROFILE.height), (1, 1, 1)))
    sync.step()
    assert not [c for c in transport.calls if c[0] == "PUT"]
    assert sync.connected


def test_client_does_not_upload_to_a_server_without_uploads(tmp_path):
    clock = Clock()
    app = make_server(tmp_path, clock, screenshot_upload_dir=None)
    transport = FlaskTransport(app)
    queue = UploadQueue(600, clock=clock)
    path = tmp_path / "date.png"
    path.write_bytes(png())
    queue.offer("date", path, clock.now)
    sync = ClientSync(display_client.capabilities_for("hyper", PROFILE), transport,
                      ClientCache(tmp_path / "client"), ArtifactCache(tmp_path / "client", max_bytes=1 << 20),
                      enrollment_token=TOKEN, clock=clock, screenshot_uploads=queue)
    sync.step()
    assert not [c for c in transport.calls if c[0] == "PUT"]
    assert sync.upload_screenshots() == 0


def test_build_client_wires_uploads_only_when_enabled(tmp_path, monkeypatch):
    settings = {"DESK_DISPLAY_PROFILE": "hyperpixel4", "DESK_DISPLAY_CLIENT_ID": "hyper",
                "DESK_DISPLAY_SERVER_URL": "http://square:8765", "DESK_DISPLAY_CLIENT_CACHE_DIR": str(tmp_path)}

    class Presenter:
        def present(self, image):
            pass

    def build(**extra):
        shots = ClientScreenshots(tmp_path / "shots", profile_id="hyperpixel4", width=800, height=480)
        client = display_client.build_client({**settings, **extra}, presenter=Presenter(),
                                             transport=lambda *a, **k: None, screenshots=shots)
        return client, shots

    client, shots = build()
    assert shots.uploads is None and client.sync.screenshot_uploads is None
    client, shots = build(DESK_DISPLAY_CLIENT_UPLOAD_SCREENSHOTS="1", DESK_DISPLAY_CLIENT_SCREENSHOT_UPLOAD_MINUTES="5")
    assert shots.uploads is client.sync.screenshot_uploads
    assert shots.uploads.interval_seconds == 300


# ── Config UI ───────────────────────────────────────────────────────────────


def test_config_ui_lists_and_serves_uploaded_screenshots(tmp_path):
    import config_ui

    app = config_ui.app
    old = app.extensions["desk_display_uploaded_screenshots"]
    inbox = ScreenshotInbox(tmp_path)
    inbox.save("hyper", "date", png((7, 7, 7)))
    app.extensions["desk_display_uploaded_screenshots"] = inbox
    app.config["TESTING"] = True
    try:
        web = app.test_client()
        listing = web.get("/api/clients/hyper/uploaded-screenshots").get_json()
        [entry] = listing["screens"]
        assert entry["screen"] == "date" and entry["url"].endswith(entry["file"])
        image = web.get(entry["url"])
        assert image.status_code == 200 and image.mimetype == "image/png"
        assert Image.open(io.BytesIO(image.data)).getpixel((0, 0)) == (7, 7, 7)
        assert web.get("/api/clients/mini/uploaded-screenshots").get_json()["screens"] == []
        assert web.get("/api/clients/hyper/uploaded-screenshots/index.json").status_code == 404
        assert web.get("/api/clients/hyper/uploaded-screenshots/x-0000000000.png").status_code == 404
    finally:
        app.extensions["desk_display_uploaded_screenshots"] = old


def test_index_file_is_plain_json(tmp_path):
    inbox = ScreenshotInbox(tmp_path)
    inbox.save("hyper", "date", png())
    index = json.loads((tmp_path / "hyper" / "index.json").read_text())
    assert index["version"] == 1 and set(index["screens"]["date"]) >= {"file", "captured_at", "received_at"}
