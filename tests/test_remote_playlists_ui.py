"""API and browser-level tests for the Playlists and Clients pages."""
from __future__ import annotations

import json
import os
import threading

import pytest

pytest.importorskip("flask")

import config_ui  # noqa: E402
import remote_playlists_ui as ui  # noqa: E402
from display_profiles import PROFILE_PRESETS  # noqa: E402
from remote_display.models import AcceptedRevisions, ClientCapabilities, ClientStatus  # noqa: E402
from remote_display.playlist_store import PlaylistStore  # noqa: E402
from remote_display.registry import ClientRegistry  # noqa: E402

CSRF = {ui.CSRF_HEADER: ui.CSRF_VALUE}
DOC = {
    "screens": {"date": 1, "weather radar": 1, "quad": 1, "weather1": {"frequency": 1, "alt": {"screen": "weather2", "frequency": 2}}},
    "sequence": [],
}


class Clock:
    now = 1_800_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("SCREEN_UI_PASSWORD", raising=False)
    monkeypatch.delenv("SCREEN_AUTH_ENABLED", raising=False)
    monkeypatch.setenv("DESK_DISPLAY_STATIC_CLIENTS", "lobby:hdmi_1080p")
    app = config_ui.app
    old = (app.extensions["desk_display_playlist_store"], app.extensions["desk_display_registry_snapshot"])
    store = PlaylistStore(tmp_path / "playlists.json")
    app.extensions["desk_display_playlist_store"] = store
    app.extensions["desk_display_registry_snapshot"] = tmp_path / "clients.json"
    app.config["TESTING"] = True
    yield {"app": app, "store": store, "snapshot": tmp_path / "clients.json", "tmp": tmp_path}
    app.extensions["desk_display_playlist_store"], app.extensions["desk_display_registry_snapshot"] = old


@pytest.fixture
def web(env):
    return env["app"].test_client()


def caps(client_id, profile="hyperpixel4", **extra):
    preset = PROFILE_PRESETS[profile]
    return ClientCapabilities(
        protocol_version=1, client_software_version="0.2", client_id=client_id, display_profile=profile,
        logical_width=preset.width, logical_height=preset.height, image_formats=("PNG",),
        color_modes=(preset.color_mode,), render_package_versions=(1,), **extra,
    )


def publish_registry(env, clients, *, delivered=None, acknowledged=None, clock=None):
    """Write the snapshot the render server would publish."""

    import time

    clock = clock or time.time
    registry = ClientRegistry(lease_seconds=60, static_clients={"lobby": "hdmi_1080p"}, clock=clock)
    for capabilities in clients:
        registration = registry.register(capabilities)
        status = ClientStatus(
            client_id=capabilities.client_id, playback_state="playing",
            accepted_revisions=AcceptedRevisions(playlist_revision=(acknowledged or {}).get(capabilities.client_id)),
            current_screen="date", physical_rotation=180, cache_age_seconds=42,
        )
        registry.heartbeat(capabilities.client_id, registration.credential, status)
        registry.mark_delivered(capabilities.client_id, (delivered or {}).get(capabilities.client_id))
    registry.write_snapshot(env["snapshot"])
    return registry


def create(web, name="Kitchen", document=DOC):
    response = web.post("/api/playlists", json={"name": name, "document": document}, headers=CSRF)
    assert response.status_code == 201, response.get_json()
    return response.get_json()


# ── CRUD, ordering, validation ──────────────────────────────────────────────


def test_playlist_crud(web):
    playlist = create(web)
    listing = web.get("/api/playlists").get_json()
    assert [p["name"] for p in listing["playlists"]] == ["Kitchen"]
    assert listing["playlists"][0]["screen_count"] == 4

    fetched = web.get(f"/api/playlists/{playlist['id']}").get_json()
    assert fetched["document"]["screens"]["weather1"]["alt"] == {"screen": "weather2", "frequency": 2}

    edited = json.loads(json.dumps(DOC))
    edited["screens"]["date"] = 3
    response = web.put(f"/api/playlists/{playlist['id']}", json={"document": edited, "expected_revision": playlist["revision"]}, headers=CSRF)
    assert response.status_code == 200
    updated = response.get_json()
    assert updated["revision"] != playlist["revision"]

    response = web.post(f"/api/playlists/{playlist['id']}/rename", json={"name": "Kitchen 2", "expected_revision": updated["revision"]}, headers=CSRF)
    assert response.get_json()["name"] == "Kitchen 2"

    clone = web.post(f"/api/playlists/{playlist['id']}/clone", json={}, headers=CSRF).get_json()
    assert clone["name"] == "Kitchen 2 (copy)" and clone["id"] != playlist["id"]

    response = web.delete(f"/api/playlists/{clone['id']}?expected_revision={clone['revision']}", headers=CSRF)
    assert response.status_code == 204
    assert web.get(f"/api/playlists/{clone['id']}").status_code == 404


def test_create_defaults_to_current_rotation(web):
    response = web.post("/api/playlists", json={"name": "From rotation"}, headers=CSRF)
    assert response.status_code == 201
    assert response.get_json()["document"]["screens"]


def test_reorder_sequence(web):
    doc = {"screens": {"date": 1, "inside": 1},
           "playlists": {"a": {"label": "a", "steps": [{"screen": "date"}]}, "b": {"label": "b", "steps": [{"screen": "inside"}]}},
           "sequence": [{"playlist": "a"}, {"playlist": "b"}]}
    playlist = create(web, document=doc)
    response = web.post(f"/api/playlists/{playlist['id']}/reorder", json={"order": [1, 0], "expected_revision": playlist["revision"]}, headers=CSRF)
    assert response.status_code == 200
    assert response.get_json()["document"]["sequence"] == [{"playlist": "b"}, {"playlist": "a"}]
    bad = web.post(f"/api/playlists/{playlist['id']}/reorder", json={"order": [0, 0], "expected_revision": response.get_json()["revision"]}, headers=CSRF)
    assert bad.status_code == 400


def test_validate_endpoint(web):
    ok = web.post("/api/playlists/validate", json={"document": DOC}, headers=CSRF).get_json()
    assert ok["valid"] and "weather2" in ok["alternate_screens"]
    bad = web.post("/api/playlists/validate", json={"document": {"screens": {"nope": 1}}}, headers=CSRF).get_json()
    assert bad["valid"] is False and "unknown screen" in bad["error"]
    response = web.post("/api/playlists", json={"name": "Bad", "document": {"screens": {"nope": 1}}}, headers=CSRF)
    assert response.status_code == 400 and response.get_json()["error"] == "invalid_playlist"


def test_revision_conflict(web):
    playlist = create(web)
    edit_a = json.loads(json.dumps(DOC))
    edit_a["screens"]["date"] = 2
    edit_b = json.loads(json.dumps(DOC))
    edit_b["screens"]["date"] = 9
    first = web.put(f"/api/playlists/{playlist['id']}", json={"document": edit_a, "expected_revision": playlist["revision"]}, headers=CSRF)
    second = web.put(f"/api/playlists/{playlist['id']}", json={"document": edit_b, "expected_revision": playlist["revision"]}, headers=CSRF)
    assert first.status_code == 200 and second.status_code == 409
    body = second.get_json()
    assert body["error"] == "revision_conflict" and body["current_revision"] == first.get_json()["revision"]
    assert web.get(f"/api/playlists/{playlist['id']}").get_json()["document"]["screens"]["date"] == 2


# ── Assignment, sharing, deletion safeguards ────────────────────────────────


def test_assignment_and_sharing(web, env):
    shared = create(web, "Shared")
    for client in ("office", "den"):
        response = web.put(f"/api/clients/{client}/assignment", json={"playlist_id": shared["id"], "expected_playlist_id": None}, headers=CSRF)
        assert response.status_code == 200
    detail = web.get(f"/api/playlists/{shared['id']}").get_json()
    assert detail["clients"] == ["den", "office"]
    assert web.get("/api/playlists").get_json()["playlists"][0]["clients"] == ["den", "office"]

    stale = web.put("/api/clients/office/assignment", json={"playlist_id": None, "expected_playlist_id": None}, headers=CSRF)
    assert stale.status_code == 409
    missing = web.put("/api/clients/office/assignment", json={"playlist_id": None}, headers=CSRF)
    assert missing.status_code == 400

    fork = web.post("/api/clients/office/fork", json={"expected_playlist_id": shared["id"]}, headers=CSRF)
    assert fork.status_code == 201
    assert env["store"].snapshot()["assignments"]["office"]["playlist_id"] == fork.get_json()["id"]
    assert env["store"].clients_using(shared["id"]) == ["den"]


def test_deletion_safeguard(web):
    playlist = create(web)
    web.put("/api/clients/office/assignment", json={"playlist_id": playlist["id"], "expected_playlist_id": None}, headers=CSRF)
    response = web.delete(f"/api/playlists/{playlist['id']}?expected_revision={playlist['revision']}", headers=CSRF)
    assert response.status_code == 409 and response.get_json()["clients"] == ["office"]
    stale = web.delete(f"/api/playlists/{playlist['id']}?expected_revision=r-old", headers=CSRF)
    assert stale.status_code == 409 and stale.get_json()["error"] == "revision_conflict"


# ── Import / export ─────────────────────────────────────────────────────────


def test_import_export_round_trip_without_secrets(web, monkeypatch):
    monkeypatch.setenv("AIRNOW_API_KEY", "airnow-secret-4242")
    doc = json.loads(json.dumps(DOC))
    doc["screens"]["date"] = {"frequency": 1, "note": "key airnow-secret-4242"}
    playlist = create(web, document=doc)
    response = web.get(f"/api/playlists/{playlist['id']}/export")
    assert response.status_code == 200
    assert "attachment" in response.headers["Content-Disposition"]
    assert b"airnow-secret-4242" not in response.data
    exported = response.get_json()
    imported = web.post("/api/playlists/import", json={"export": exported, "name": "Copy"}, headers=CSRF)
    assert imported.status_code == 201 and imported.get_json()["name"] == "Copy"
    detail = web.get(f"/api/playlists/{playlist['id']}")
    assert b"airnow-secret-4242" not in detail.data
    bad = web.post("/api/playlists/import", json={"export": {"format": "zip"}}, headers=CSRF)
    assert bad.status_code == 400


# ── Client registry, revisions, warnings ────────────────────────────────────


def test_client_registry_and_acknowledgment_status(web, env):
    playlist = create(web)
    for client in ("office", "den", "kitchen"):
        web.put(f"/api/clients/{client}/assignment", json={"playlist_id": playlist["id"], "expected_playlist_id": None}, headers=CSRF)
    rev = playlist["revision"]
    publish_registry(
        env,
        [caps("office", supports_animation=True, has_touch=True, buttons=("A", "B")), caps("den"), caps("kitchen")],
        delivered={"office": rev, "den": rev, "kitchen": "r-old"},
        acknowledged={"office": rev, "den": "r-old"},
    )
    web.put("/api/clients/office/name", json={"friendly_name": "Office desk"}, headers=CSRF)
    rows = {row["client_id"]: row for row in web.get("/api/clients").get_json()["clients"]}

    office = rows["office"]
    assert office["friendly_name"] == "Office desk" and office["kind"] == "dynamic"
    assert office["state"] == "online"
    assert office["display_profile"] == "hyperpixel4" and office["dimensions"] == "800x480"
    assert office["physical_rotation"] == 180 and office["software_version"] == "0.2"
    assert office["current_screen"] == "date" and office["cache_age_seconds"] == 42
    assert office["capabilities"]["has_touch"] is True
    assert office["assignment"]["playlist_id"] == playlist["id"]
    assert (office["saved_revision"], office["delivered_revision"], office["acknowledged_revision"]) == (rev, rev, rev)
    assert office["revision_state"] == "in_sync"
    assert rows["den"]["revision_state"] == "pending_acknowledgment"
    assert rows["kitchen"]["revision_state"] == "pending_delivery"

    lobby = rows["lobby"]
    assert lobby["kind"] == "static" and lobby["state"] == "never_connected"
    assert lobby["revision_state"] == "unassigned"


def test_client_states(env):
    heartbeat = 20
    now = 1_800_000_000.0
    iso = lambda t: ui._iso_now() if t is None else __import__("remote_display.registry", fromlist=["_iso"])._iso(t)  # noqa: E731
    assert ui.client_state({"disabled": True}, heartbeat, now) == "disabled"
    assert ui.client_state({"static": True, "lease_expires_at": None}, heartbeat, now) == "never_connected"
    assert ui.client_state({"lease_expires_at": iso(now - 1), "last_seen": iso(now - 70)}, heartbeat, now) == "expired"
    assert ui.client_state({"lease_expires_at": iso(now + 30), "last_seen": iso(now - 45)}, heartbeat, now) == "stale"
    assert ui.client_state({"lease_expires_at": iso(now + 50), "last_seen": iso(now - 5)}, heartbeat, now) == "online"


def test_capability_warnings_and_demand_preview(web, env):
    playlist = create(web)
    for client in ("office", "oled"):
        web.put(f"/api/clients/{client}/assignment", json={"playlist_id": playlist["id"], "expected_playlist_id": None}, headers=CSRF)
    publish_registry(env, [caps("office", supports_animation=True, has_touch=True), caps("oled", "waveshare_oled_128x64")])
    preview = web.get(f"/api/playlists/{playlist['id']}/preview").get_json()
    assert preview["required_screens"] == ["date", "quad", "weather radar", "weather1"]
    assert preview["alternate_screens"] == ["weather2"]
    by_client = {c["client_id"]: {w["code"] for w in c["warnings"]} for c in preview["clients"]}
    assert by_client["office"] == set()
    assert by_client["oled"] >= {"no_animation", "no_touch", "monochrome", "small_display"}
    assert {d["display_profile"] for d in preview["render_demand"]} == {"hyperpixel4", "waveshare_oled_128x64"}
    ad_hoc = web.get(f"/api/playlists/{playlist['id']}/preview?client=ghost").get_json()
    assert ad_hoc["clients"][0]["warnings"][0]["code"] == "unknown_client"
    rows = {r["client_id"]: r for r in web.get("/api/clients").get_json()["clients"]}
    assert {w["code"] for w in rows["oled"]["warnings"]} >= {"monochrome"}


def test_registry_snapshot_contains_no_credentials(env):
    registry = publish_registry(env, [caps("office")])
    text = env["snapshot"].read_text(encoding="utf-8")
    assert "credential" not in text
    for record in registry.records():
        if record.credential_hash:
            assert record.credential_hash not in text


# ── Authorization and CSRF ──────────────────────────────────────────────────


def test_mutations_require_csrf_header(web):
    response = web.post("/api/playlists", json={"name": "x", "document": DOC})
    assert response.status_code == 403 and response.get_json()["error"] == "csrf_check_failed"
    assert web.put("/api/clients/office/assignment", json={"playlist_id": None, "expected_playlist_id": None}).status_code == 403


def test_authentication_required_when_enabled(web, monkeypatch):
    monkeypatch.setenv("SCREEN_UI_PASSWORD", "correct horse battery")
    for method, url in (("get", "/api/playlists"), ("post", "/api/playlists"), ("get", "/api/clients"),
                        ("put", "/api/clients/office/assignment"), ("get", "/api/playlists/audit")):
        response = getattr(web, method)(url, json={}, headers=CSRF)
        assert response.status_code == 401, url
    page = web.get("/playlists")
    assert page.status_code == 302 and "/login" in page.headers["Location"]
    web.post("/login", data={"password": "correct horse battery"})
    playlist = create(web)
    audit = web.get("/api/playlists/audit").get_json()["entries"]
    assert audit[0]["target"] == playlist["id"]
    assert audit[0]["actor"] != "unauthenticated"


def test_browser_responses_exclude_secrets(web, env, monkeypatch):
    monkeypatch.setenv("OWM_API_KEY", "owm-secret-value-123")
    monkeypatch.setenv("DESK_DISPLAY_SERVER_AUTH_TOKEN", "server-token-" + "z" * 32)
    doc = json.loads(json.dumps(DOC))
    doc["screens"]["date"] = {"frequency": 1, "note": "owm-secret-value-123"}
    playlist = create(web, document=doc)
    web.put("/api/clients/office/assignment", json={"playlist_id": playlist["id"], "expected_playlist_id": None}, headers=CSRF)
    publish_registry(env, [caps("office")])
    for url in ("/api/playlists", f"/api/playlists/{playlist['id']}", f"/api/playlists/{playlist['id']}/export",
                f"/api/playlists/{playlist['id']}/preview", "/api/clients", "/api/playlists/audit", "/playlists", "/clients"):
        body = web.get(url).get_data(as_text=True)
        assert "owm-secret-value-123" not in body, url
        assert "server-token-" not in body, url


# ── Pages ───────────────────────────────────────────────────────────────────


def test_pages_render_and_are_linked(web):
    for path, marker in (("/playlists", "playlist-list"), ("/clients", 'id="clients"')):
        response = web.get(path)
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert marker in html and ui.CSRF_VALUE in html
    assert 'href="/playlists"' in web.get("/").get_data(as_text=True)


# ── Browser-level (Playwright, when installed) ──────────────────────────────


@pytest.fixture
def live_server(env):
    from werkzeug.serving import make_server

    server = make_server("127.0.0.1", 0, env["app"], threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture
def browser():
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as playwright:
        kwargs = {}
        executable = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
        bundled = os.path.join(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", ""), "chromium")
        if not executable and os.environ.get("PLAYWRIGHT_BROWSERS_PATH") and os.path.isfile(bundled):
            executable = bundled
        if executable:
            kwargs["executable_path"] = executable
        try:
            instance = playwright.chromium.launch(**kwargs)
        except Exception as exc:  # pragma: no cover - environment dependent
            pytest.skip(f"Chromium is not available: {exc}")
        yield instance
        instance.close()


def test_browser_playlist_workflow(live_server, browser, env):
    page = browser.new_page()
    page.goto(f"{live_server}/playlists")
    page.fill("#new-name", "Browser playlist")
    page.click("#create-btn")
    page.wait_for_selector("#editor:not(.hidden)")
    assert page.input_value("#edit-name") == "Browser playlist"
    playlist_id = page.inner_text("#edit-id")
    first_revision = page.inner_text("#edit-revision")

    # Someone else edits the playlist behind this page's back.
    store = env["store"]
    other = json.loads(json.dumps(store.get(playlist_id)["document"]))
    first_screen = next(iter(other["screens"]))
    other["screens"][first_screen] = 7 if other["screens"][first_screen] != 7 else 8
    store.update(playlist_id, other, expected_revision=first_revision, actor="someone-else")

    page.click("#save-btn")
    page.wait_for_selector("#notice.error")
    assert "Someone else changed this playlist" in page.inner_text("#notice")

    page.click(f"li[data-id='{playlist_id}']")
    page.wait_for_function(f"document.querySelector('#edit-revision').textContent !== '{first_revision}'")
    page.click("#save-btn")
    page.wait_for_selector("#notice.ok")
    assert "Saved revision" in page.inner_text("#notice")


def test_browser_client_assignment(live_server, browser, env):
    playlist = env["store"].create("Shared", DOC, actor="test")
    publish_registry(env, [caps("office")], delivered={"office": playlist["revision"]})
    page = browser.new_page()
    page.goto(f"{live_server}/clients")
    page.wait_for_selector("tr[data-client-id='office']")
    page.select_option("tr[data-client-id='office'] select", playlist["id"])
    page.wait_for_selector("#notice.ok")
    assert env["store"].snapshot()["assignments"]["office"]["playlist_id"] == playlist["id"]
    page.wait_for_function("document.querySelector(\"tr[data-client-id='office']\").textContent.includes('pending acknowledgment')")
    row_text = page.inner_text("tr[data-client-id='office']")
    assert "saved " + playlist["revision"] in row_text
    assert "delivered " + playlist["revision"] in row_text
    assert "acknowledged —" in row_text
    assert "lobby" in page.inner_text("#clients")
