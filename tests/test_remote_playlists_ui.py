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
from remote_display.models import (  # noqa: E402
    AcceptedRevisions,
    ClientCapabilities,
    ClientStatus,
    ClientTelemetry,
)
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
    from remote_display.provisioning import ProvisioningStore

    from remote_display.client_commands import CommandStore

    keys = ("desk_display_playlist_store", "desk_display_registry_snapshot", "desk_display_provisioning",
            "desk_display_client_commands")
    old = tuple(app.extensions[key] for key in keys)
    store = PlaylistStore(tmp_path / "playlists.json")
    app.extensions["desk_display_playlist_store"] = store
    app.extensions["desk_display_registry_snapshot"] = tmp_path / "clients.json"
    app.extensions["desk_display_provisioning"] = ProvisioningStore(tmp_path / "provisioned.json")
    app.extensions["desk_display_client_commands"] = CommandStore(tmp_path / "commands.json")
    app.config["TESTING"] = True
    yield {"app": app, "store": store, "snapshot": tmp_path / "clients.json", "tmp": tmp_path}
    for key, value in zip(keys, old):
        app.extensions[key] = value


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


def publish_registry(env, clients, *, delivered=None, acknowledged=None, clock=None, telemetry=None,
                     addresses=None):
    """Write the snapshot the render server would publish."""

    import time

    clock = clock or time.time
    registry = ClientRegistry(lease_seconds=60, static_clients={"lobby": "hdmi_1080p"}, clock=clock)
    for capabilities in clients:
        registration = registry.register(capabilities,
                                         address=(addresses or {}).get(capabilities.client_id))
        status = ClientStatus(
            client_id=capabilities.client_id, playback_state="playing",
            accepted_revisions=AcceptedRevisions(playlist_revision=(acknowledged or {}).get(capabilities.client_id)),
            current_screen="date", physical_rotation=180, cache_age_seconds=42, last_sync_age_seconds=12,
        )
        registry.heartbeat(capabilities.client_id, registration.credential, status,
                           telemetry=(telemetry or {}).get(capabilities.client_id))
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


def _editor_payload(view, **changes):
    """Rows, groups and sequence as the Playlists page editor would send them."""

    rows = {row["id"]: dict(row) for row in view["screens"]}
    for screen_id, fields in changes.items():
        rows[screen_id].update(fields)
    return list(rows.values())


def test_editor_view_matches_the_rotation_editor(web):
    doc = {**DOC, "scroll": {"speed": 1.5, "smoothness": 1.0, "vertical_speed_adjustment": 0.25},
           "playlists": {"wx": {"label": "Weather", "steps": [{"screen": "weather1"}]}},
           "sequence": [{"playlist": "wx"}]}
    playlist = create(web, document=doc)
    response = web.get(f"/api/playlists/{playlist['id']}/editor")
    assert response.status_code == 200
    data = response.get_json()
    assert data["revision"] == playlist["revision"] and data["document"] == playlist["document"]
    view = data["editor"]
    rows = {row["id"]: row for row in view["screens"]}
    assert rows["weather1"]["alt_screen"] == "weather2" and rows["weather1"]["alt_frequency"] == 2
    assert rows["date"]["frequency"] == 1
    # Every catalog screen is offered, disabled, so it can be switched on.
    assert rows["nixie"]["frequency"] == 0
    assert view["playlists"] == [{"id": "wx", "name": "Weather"}]
    assert view["playlist_assignments"] == {"weather1": "wx"}
    assert "weather2" in view["screen_ids"]


def test_editor_save_round_trip(web):
    from remote_display.playlist_store import canonicalize_screen_ids, document_revision

    doc = {**DOC, "scroll": {"speed": 1.5, "smoothness": 1.0, "vertical_speed_adjustment": 0.25}}
    playlist = create(web, document=doc)
    view = web.get(f"/api/playlists/{playlist['id']}/editor").get_json()["editor"]
    screens = _editor_payload(
        view,
        date={"frequency": 3, "extra_seconds": 5},
        nixie={"frequency": 2, "hide_after_enabled": True, "hide_after_at": "2030-01-02T03:04"},
        quad={"alt_screen": "date", "alt_frequency": "4"},
    )
    # The page sends rows in on-screen order: the "Late" group plays last.
    order = ["nixie", "date", "weather radar", "quad", "weather1"]
    screens.sort(key=lambda row: order.index(row["id"]) if row["id"] in order else len(order))
    payload = {
        "screens": screens,
        "playlists": {"late": {"label": "Late", "steps": [{"screen": "weather1"}]},
                      "early": {"label": "Early", "steps": [{"screen": "quad"}]}},
        "sequence": [{"playlist": "early"}, {"playlist": "late"}],
        "expected_revision": playlist["revision"],
    }
    response = web.put(f"/api/playlists/{playlist['id']}/editor", json=payload, headers=CSRF)
    assert response.status_code == 200, response.get_json()
    saved = response.get_json()
    document = saved["document"]
    assert list(document["screens"]) == order
    assert document["screens"]["date"] == {"frequency": 3, "extra_seconds": 5}
    assert document["screens"]["nixie"] == {"frequency": 2, "hide_after_enabled": True, "hide_after_at": "2030-01-02T03:04"}
    assert document["screens"]["quad"] == {"frequency": 1, "alt": {"screen": "date", "frequency": 4}}
    assert document["screens"]["weather1"] == {"frequency": 1, "alt": {"screen": "weather2", "frequency": 2}}
    # Disabled catalog screens the playlist never named are not added to it.
    assert "on this day" not in document["screens"]
    assert document["playlists"]["early"]["steps"] == [{"screen": "quad"}]
    assert document["sequence"] == [{"playlist": "early"}, {"playlist": "late"}]
    assert document["scroll"] == doc["scroll"]
    # Clients canonicalize before checking the revision, so it must match.
    assert saved["revision"] == document_revision(canonicalize_screen_ids(document))
    assert saved["editor"]["playlists"] == [{"id": "early", "name": "Early"}, {"id": "late", "name": "Late"}]
    stored = web.get(f"/api/playlists/{playlist['id']}").get_json()
    assert stored["revision"] == saved["revision"] and stored["document"] == document

    # Saving the editor again unchanged keeps the same revision.
    again = web.put(f"/api/playlists/{playlist['id']}/editor", json={
        "screens": saved["editor"]["screens"],
        "playlists": document["playlists"],
        "sequence": document["sequence"],
        "expected_revision": saved["revision"],
    }, headers=CSRF)
    assert again.status_code == 200 and again.get_json()["revision"] == saved["revision"]


def test_editor_save_canonicalizes_renamed_screens(web):
    playlist = create(web)
    view = web.get(f"/api/playlists/{playlist['id']}/editor").get_json()["editor"]
    screens = [row for row in view["screens"] if row["id"] != "NCAA Mens BB Scoreboard"]
    screens.append({"id": "NCAAM Scoreboard", "frequency": 2})
    payload = {"screens": screens, "playlists": {"p": {"label": "P", "steps": [{"screen": "NCAAM Scoreboard"}]}},
               "sequence": [{"playlist": "p"}], "expected_revision": playlist["revision"]}
    response = web.put(f"/api/playlists/{playlist['id']}/editor", json=payload, headers=CSRF)
    assert response.status_code == 200, response.get_json()
    document = response.get_json()["document"]
    assert document["screens"]["NCAA Mens BB Scoreboard"] == 2 and "NCAAM Scoreboard" not in document["screens"]
    assert document["playlists"]["p"]["steps"] == [{"screen": "NCAA Mens BB Scoreboard"}]


def test_editor_save_rejects_conflicts_and_bad_rows(web):
    playlist = create(web)
    view = web.get(f"/api/playlists/{playlist['id']}/editor").get_json()["editor"]
    url = f"/api/playlists/{playlist['id']}/editor"
    stale = web.put(url, json={"screens": view["screens"], "expected_revision": "r-stale"}, headers=CSRF)
    assert stale.status_code == 409 and stale.get_json()["error"] == "revision_conflict"
    unknown = web.put(url, json={"screens": [{"id": "nope", "frequency": 1}], "expected_revision": playlist["revision"]}, headers=CSRF)
    assert unknown.status_code == 400 and "Unknown screen id" in unknown.get_json()["message"]
    bad_alt = web.put(url, json={"screens": [{"id": "date", "frequency": 1, "alt_screen": "nope"}],
                                 "expected_revision": playlist["revision"]}, headers=CSRF)
    assert bad_alt.status_code == 400
    no_csrf = web.put(url, json={"screens": view["screens"], "expected_revision": playlist["revision"]})
    assert no_csrf.status_code == 403
    assert web.get(f"/api/playlists/{playlist['id']}").get_json()["revision"] == playlist["revision"]


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


def test_client_rows_carry_delivery_telemetry(web, env):
    timings = ClientTelemetry(heartbeat_rtt_ms=84.5, manifest_fetch_ms=120.0, last_sync_duration_ms=900.0,
                              download_count=2, download_bytes=40960, download_ms=610.0,
                              displayed_content_age_seconds=35.0, consecutive_failures=1)
    publish_registry(env, [caps("office"), caps("den")], telemetry={"office": timings})
    rows = {row["client_id"]: row for row in web.get("/api/clients").get_json()["clients"]}
    assert rows["office"]["telemetry"] == {
        "heartbeat_rtt_ms": 84.5, "manifest_fetch_ms": 120.0, "last_sync_duration_ms": 900.0,
        "download_count": 2, "download_bytes": 40960, "download_ms": 610.0,
        "displayed_content_age_seconds": 35.0, "consecutive_failures": 1,
    }
    # A client that predates telemetry (or has not reported yet) has none.
    assert rows["den"]["telemetry"] is None
    assert "last_sync_age_seconds" in rows["den"]


def test_client_rows_carry_the_address_the_display_connected_from(web, env):
    publish_registry(env, [caps("office"), caps("den")], addresses={"office": "10.0.0.7"})
    rows = {row["client_id"]: row for row in web.get("/api/clients").get_json()["clients"]}
    assert rows["office"]["address"] == "10.0.0.7"
    assert rows["den"]["address"] is None
    assert rows["lobby"]["address"] is None  # static, never connected


def test_update_and_restart_buttons_queue_commands(web, env):
    publish_registry(env, [caps("office")])
    url = "/api/clients/office/commands"
    assert web.post(url, json={"action": "update"}).status_code == 403  # CSRF guard
    response = web.post(url, json={"action": "update"}, headers=CSRF)
    assert response.status_code == 202
    command = response.get_json()["command"]
    assert command["action"] == "update" and command["state"] == "pending"
    assert web.post(url, json={"action": "update"}, headers=CSRF).status_code == 409
    assert web.post(url, json={"action": "restart"}, headers=CSRF).status_code == 202
    assert web.post(url, json={"action": "reboot"}, headers=CSRF).status_code == 400
    assert web.post("/api/clients/ghost/commands", json={"action": "update"}, headers=CSRF).status_code == 404

    store = env["app"].extensions["desk_display_client_commands"]
    store.take_pending("office")
    store.record_results("office", [{"id": command["id"], "status": "failed", "exit_code": 1,
                                     "output": "fatal: Not possible to fast-forward"}])
    rows = {row["client_id"]: row for row in web.get("/api/clients").get_json()["clients"]}
    latest = {c["action"]: c for c in rows["office"]["commands"]}
    assert latest["update"]["state"] == "failed" and "fast-forward" in latest["update"]["output"]
    assert latest["update"]["label"] == "Update code (git pull)"
    assert latest["update"]["finished_at"].endswith("Z")
    assert latest["restart"]["state"] == "delivered"
    assert rows["lobby"]["commands"] == []


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
                        ("put", "/api/clients/office/assignment"), ("get", "/api/playlists/audit"),
                        ("post", "/api/clients/office/commands")):
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


GROUPED_DOC = {
    "screens": {"weather radar": 1, "nixie": 1, "date": 1, "weather1": {"frequency": 1, "alt": {"screen": "weather2", "frequency": 2}}},
    "playlists": {"a": {"label": "Morning", "steps": [{"screen": "date"}]},
                  "b": {"label": "Weather", "steps": [{"screen": "weather1"}]}},
    "sequence": [{"playlist": "a"}, {"playlist": "b"}],
}


def _row(screen_id):
    return f".screen-row[data-screen-id='{screen_id}']"


def test_browser_playlist_rotation_editor(live_server, browser, env):
    playlist = env["store"].create("Grouped", GROUPED_DOC, actor="test")
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    page.on("dialog", lambda dialog: dialog.accept())
    page.goto(f"{live_server}/playlists")
    page.click(f"li[data-id='{playlist['id']}']")
    page.wait_for_selector(_row("date"))
    assert page.inner_text(".playlist-row[data-playlist-id='a'] .name") == "Morning"
    assert not page.is_visible(".table-header .speed-col")
    assert page.inner_text("#dirty-state") == "No unsaved changes."

    # Reorder playlists and screens, and set per-screen options.
    page.click(".playlist-row[data-playlist-id='b'] >> text=Move up")
    page.drag_and_drop(f"{_row('nixie')} .handle", f"{_row('weather radar')} .handle")
    page.fill(f"{_row('date')} .freq-input", "4")
    page.fill(f"{_row('date')} .extra-seconds-input", "6")
    page.select_option(f"{_row('weather radar')} .alt-screen-input", "date")
    page.fill(f"{_row('weather radar')} .alt-frequency-input", "3")
    page.select_option(f"{_row('quad')} .playlist-select", "a")
    page.wait_for_function("document.querySelector(\"" + _row("quad") + " .playlist-select\").value === 'a'")
    assert page.inner_text("#dirty-state") == "Unsaved changes."

    page.click("#collapseAllPlaylistsBtn")
    assert not page.is_visible(_row("date")) and not page.is_visible(_row("weather1"))
    page.click("#expandAllPlaylistsBtn")
    assert page.is_visible(_row("date")) and page.is_visible(_row("weather1"))

    page.click("#save-btn")
    page.wait_for_selector("#notice.ok")
    assert "Saved revision" in page.inner_text("#notice")
    assert page.inner_text("#dirty-state") == "No unsaved changes."
    saved = env["store"].get(playlist["id"])
    document = saved["document"]
    assert page.inner_text("#edit-revision") == saved["revision"]
    assert document["sequence"] == [{"playlist": "b"}, {"playlist": "a"}]
    assert list(document["screens"])[:2] == ["nixie", "weather radar"]
    assert document["screens"]["date"] == {"frequency": 4, "extra_seconds": 6}
    assert document["screens"]["weather radar"] == {"frequency": 1, "alt": {"screen": "date", "frequency": 3}}
    assert document["screens"]["weather1"] == GROUPED_DOC["screens"]["weather1"]
    assert document["playlists"]["a"]["steps"] == [{"screen": "date"}, {"screen": "quad"}]
    assert page.inner_text(".playlist-row[data-playlist-id='a'] .count") == "2 screens"


def test_browser_playlist_editor_keeps_json_editing(live_server, browser, env):
    playlist = env["store"].create("Json", GROUPED_DOC, actor="test")
    page = browser.new_page()
    page.goto(f"{live_server}/playlists")
    page.click(f"li[data-id='{playlist['id']}']")
    page.wait_for_selector(_row("date"))
    page.click("#advanced summary")
    edited = json.loads(page.input_value("#document"))
    edited["screens"]["date"] = 9
    page.fill("#document", json.dumps(edited))
    page.click("#save-json-btn")
    page.wait_for_selector("#notice.ok")
    page.wait_for_function("document.querySelector(\"" + _row("date") + " .freq-input\").value === '9'")
    assert env["store"].get(playlist["id"])["document"]["screens"]["date"] == 9


def test_browser_rotation_config_page_still_saves(live_server, browser, env, monkeypatch, tmp_path):
    config_path = tmp_path / "screens_config.json"
    config_path.write_text(json.dumps(GROUPED_DOC), encoding="utf-8")
    monkeypatch.setattr(config_ui, "LOCAL_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(config_ui, "LAYOUTS_CONFIG_PATH", str(tmp_path / "screens_layouts.json"))
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    page.goto(f"{live_server}/")
    page.wait_for_selector(_row("date"))
    assert page.is_visible(".table-header .speed-col")
    page.click(".playlist-row[data-playlist-id='b'] >> text=Move up")
    page.fill(f"{_row('date')} .freq-input", "5")
    page.click("#collapseAllPlaylistsBtn")
    assert not page.is_visible(_row("date"))
    page.click("#expandAllPlaylistsBtn")
    page.click("#saveBtn")
    page.wait_for_function("!document.getElementById('saveBtn').disabled && !sessionStorage.getItem('screen-config-draft')")
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["sequence"] == [{"playlist": "b"}, {"playlist": "a"}]
    assert saved["screens"]["date"] == 5
    assert saved["playlists"]["a"] == {"label": "Morning", "steps": [{"screen": "date"}]}


def test_browser_client_assignment(live_server, browser, env):
    playlist = env["store"].create("Shared", DOC, actor="test")
    publish_registry(env, [caps("office")], delivered={"office": playlist["revision"]})
    page = browser.new_page()
    page.goto(f"{live_server}/clients")
    page.wait_for_selector("[data-client-id='office']")
    page.select_option("[data-client-id='office'] select", playlist["id"])
    page.wait_for_selector("#notice.ok")
    assert env["store"].snapshot()["assignments"]["office"]["playlist_id"] == playlist["id"]
    page.wait_for_function("document.querySelector(\"[data-client-id='office']\").textContent.includes('pending acknowledgment')")
    page.click("[data-client-id='office'] button[data-panel='delivery']")
    row_text = page.inner_text("[data-client-id='office']")
    assert "saved " + playlist["revision"] in row_text
    assert "delivered " + playlist["revision"] in row_text
    assert "acknowledged —" in row_text
    assert "lobby" in page.inner_text("#clients")


def test_browser_clients_page_shows_delivery_timings(live_server, browser, env):
    timings = ClientTelemetry(heartbeat_rtt_ms=84.5, manifest_fetch_ms=1250.0, last_sync_duration_ms=2400.0,
                              download_count=3, download_bytes=2 * 1048576, download_ms=1800.0,
                              displayed_content_age_seconds=35.0, consecutive_failures=2)
    publish_registry(env, [caps("office"), caps("den")], telemetry={"office": timings})
    page = browser.new_page(viewport={"width": 1600, "height": 700})
    page.goto(f"{live_server}/clients")
    page.wait_for_selector("[data-client-id='office']")
    for client_id in ("office", "den"):
        page.click(f"[data-client-id='{client_id}'] button[data-panel='delivery']")
    office = page.inner_text("[data-client-id='office']")
    assert "last sync 12s ago" in office and "on screen rendered 35s ago" in office
    assert "heartbeat 85 ms · manifest 1.3 s · sync 2.4 s" in office
    assert "last download 3 files, 2.0 MB in 1.8 s" in office
    assert "2 failed syncs before the last success" in office
    assert "no timings reported" in page.inner_text("[data-client-id='den']")


def test_browser_clients_page_sets_vertical_scroll(live_server, browser, env):
    publish_registry(env, [caps("office"), caps("den")])
    page = browser.new_page(viewport={"width": 1600, "height": 700})
    page.goto(f"{live_server}/clients")
    field = "[data-client-id='office'] input[data-vertical-scroll]"
    page.wait_for_selector("[data-client-id='den']")
    for client_id in ("office", "den"):
        page.click(f"[data-client-id='{client_id}'] button[data-panel='settings']")
    page.wait_for_selector(field)
    page.fill(field, "0.5")
    page.click("[data-client-id='office'] .vertical-scroll button:text('Save')")
    page.wait_for_selector("#notice.ok")
    assert env["store"].vertical_speed_adjustment("office") == 0.5
    page.wait_for_function("document.querySelector(\"[data-client-id='office']\").textContent.includes('own +0.50')")
    assert "global +0.00" in page.inner_text("[data-client-id='den']")
    page.click("[data-client-id='office'] button:text('Use global')")
    page.wait_for_function("!document.querySelector(\"[data-client-id='office']\").textContent.includes('own +0.50')")
    assert env["store"].vertical_speed_adjustment("office") is None


def test_browser_clients_page_sets_a_weather_location(live_server, browser, env, monkeypatch):
    monkeypatch.setenv("WEATHER_LATITUDE", "41.8781")
    monkeypatch.setenv("WEATHER_LONGITUDE", "-87.6298")
    publish_registry(env, [caps("office"), caps("den")])
    page = browser.new_page(viewport={"width": 1600, "height": 700})
    page.goto(f"{live_server}/clients")
    row = "[data-client-id='office']"
    page.wait_for_selector("[data-client-id='den']")
    for client_id in ("office", "den"):
        page.click(f"[data-client-id='{client_id}'] button[data-panel='settings']")
    page.wait_for_selector(f"{row} input[data-location='latitude']")
    assert page.get_attribute(f"{row} input[data-location='latitude']", "placeholder") == "41.8781"
    page.fill(f"{row} input[data-location='latitude']", "40.7128")
    page.fill(f"{row} input[data-location='longitude']", "-74.006")
    page.click(f"{row} .weather-location button:text('Save')")
    page.wait_for_selector("#notice.ok")
    assert env["store"].location("office").scope == "loc-40.7128_-74.0060"
    page.wait_for_function(f"document.querySelector(\"{row}\").textContent.includes('weather at 40.7128, -74.0060')")
    assert "weather at server location 41.8781, -87.6298" in page.inner_text("[data-client-id='den']")
    page.click(f"{row} button:text(\"Use server's\")")
    page.wait_for_function(f"!document.querySelector(\"{row}\").textContent.includes('weather at 40.7128')")
    assert env["store"].location("office") is None


# ── Provisioning (Phase 16) ────────────────────────────────────────────────


def test_provisioning_from_the_clients_page(env, web):
    playlist = env["store"].create("Office", DOC, actor="test")
    assert web.post("/api/clients/provision", json={"client_id": "den", "display_profile": "hyperpixel4"}).status_code == 403
    created = web.post("/api/clients/provision", headers=CSRF, json={
        "client_id": "den", "display_profile": "hyperpixel4", "playlist_id": playlist["id"]})
    assert created.status_code == 201 and created.headers["Cache-Control"] == "no-store"
    issued = created.get_json()
    token = next(line.split("=", 1)[1] for line in issued["client_env"].splitlines()
                 if line.startswith("DESK_DISPLAY_CLIENT_TOKEN="))
    assert env["store"].assignment_for("den").playlist_id == playlist["id"]

    listing = web.get("/api/clients").get_json()
    assert token not in json.dumps(listing)
    row = {r["client_id"]: r for r in listing["clients"]}["den"]
    assert row["state"] == "never_connected" and row["credential"]["state"] == "active"

    rotated = web.post("/api/clients/den/credential/rotate", headers=CSRF, json={}).get_json()
    assert token not in rotated["client_env"]
    assert web.post("/api/clients/den/credential/disable", headers=CSRF, json={}).get_json()["state"] == "disabled"
    row = {r["client_id"]: r for r in web.get("/api/clients").get_json()["clients"]}["den"]
    assert row["state"] == "disabled" and row["assignment"]["playlist_id"] == playlist["id"]
    assert web.post("/api/clients/den/credential/revoke", headers=CSRF, json={}).get_json()["state"] == "revoked"
    assert web.post("/api/clients/den/credential/nope", headers=CSRF, json={}).status_code == 404


def test_clients_page_offers_provisioning(web):
    page = web.get("/clients").get_data(as_text=True)
    assert 'id="add-display"' in page and "/clients/add" in page
    wizard = web.get("/clients/add").get_data(as_text=True)
    assert 'id="step-1"' in wizard and 'id="install-command"' in wizard and ui.CSRF_VALUE in wizard


# ── Guided registration wizard ─────────────────────────────────────────────


def test_registration_checks_flag_a_loopback_server(web, monkeypatch):
    monkeypatch.setenv("DESK_DISPLAY_SERVER_HOST", "127.0.0.1")
    monkeypatch.delenv("DESK_DISPLAY_SERVER_PUBLIC_URL", raising=False)
    monkeypatch.setenv("DESK_DISPLAY_SERVER_ADMIN_TOKEN", "admin-" + "x" * 40)
    data = web.get("/api/clients/registration", headers={"Host": "square.local:5002"}).get_json()
    checks = {c["code"]: c for c in data["checks"]}
    assert data["ready"] is False
    assert checks["bind"]["status"] == "error" and "DESK_DISPLAY_SERVER_HOST=0.0.0.0" in checks["bind"]["fix"]
    assert checks["public_url"]["status"] == "warning"
    assert data["server_url"] == "http://square.local:8765"
    assert data["insecure_transport"] is True
    assert "lobby" in data["client_ids"]
    assert {p["id"] for p in data["profiles"]} == set(PROFILE_PRESETS) - {"waveshare_oled_128x64"}
    assert "x" * 40 not in json.dumps(data)


def test_registration_checks_pass_when_reachable(web, monkeypatch):
    from remote_display import registration

    probed = []
    monkeypatch.setenv("DESK_DISPLAY_SERVER_HOST", "0.0.0.0")
    monkeypatch.setenv("DESK_DISPLAY_SERVER_PUBLIC_URL", "http://square.local:8765")
    monkeypatch.setattr(registration, "lan_address", lambda: "192.168.1.20")
    monkeypatch.setattr(registration, "probe", lambda host, port: probed.append((host, port)) or True)
    data = web.get("/api/clients/registration").get_json()
    assert data["ready"] is True and probed == [("192.168.1.20", 8765)]
    assert {c["code"]: c["status"] for c in data["checks"]} == {
        "bind": "ok", "public_url": "ok", "reachable": "ok", "transport": "warning"}


def test_wizard_provisioning_writes_a_complete_client_env(env, web):
    playlist = env["store"].create("Office", DOC, actor="test")
    created = web.post("/api/clients/provision", headers=CSRF, json={
        "client_id": "office-mini", "display_profile": "display_hat_mini", "playlist_id": playlist["id"],
        "friendly_name": "  Office   mini ", "server_url": "square.local:8765",
        "allow_insecure_transport": True})
    assert created.status_code == 201
    issued = created.get_json()
    lines = issued["client_env"].splitlines()
    assert "DESK_DISPLAY_SERVER_URL=http://square.local:8765" in lines
    assert "DESK_DISPLAY_ALLOW_INSECURE_TRANSPORT=1" in lines
    token = next(line.split("=", 1)[1] for line in lines if line.startswith("DESK_DISPLAY_CLIENT_TOKEN="))
    assert token in issued["install_command"]
    assert "--mode client --credentials ~/office-mini.env.client display_hat_mini" in issued["install_command"]
    assert issued["credentials_filename"] == "office-mini.env.client"
    row = {r["client_id"]: r for r in web.get("/api/clients").get_json()["clients"]}["office-mini"]
    assert row["friendly_name"] == "Office mini" and row["assignment"]["playlist_id"] == playlist["id"]


def test_clients_page_sets_a_displays_own_vertical_scroll(env, web, monkeypatch):
    playlist = env["store"].create("Office", DOC, actor="test")
    env["store"].assign("office", playlist["id"], expected_playlist_id=None, actor="test")
    publish_registry(env, [caps("office")])
    listing = web.get("/api/clients").get_json()
    assert {r["client_id"]: r for r in listing["clients"]}["office"]["vertical_speed_adjustment"] is None
    assert isinstance(listing["global_vertical_speed_adjustment"], float)

    saved = web.put("/api/clients/office/scroll", json={"vertical_speed_adjustment": 0.3}, headers=CSRF)
    assert saved.status_code == 200 and saved.get_json()["vertical_speed_adjustment"] == 0.3
    rows = {r["client_id"]: r for r in web.get("/api/clients").get_json()["clients"]}
    assert rows["office"]["vertical_speed_adjustment"] == 0.3
    assert env["store"].vertical_speed_adjustment("office") == 0.3

    bad = web.put("/api/clients/office/scroll", json={"vertical_speed_adjustment": 9}, headers=CSRF)
    assert bad.status_code == 400
    unknown = web.put("/api/clients/nobody/scroll", json={"vertical_speed_adjustment": 0.1}, headers=CSRF)
    assert unknown.status_code == 404
    cleared = web.put("/api/clients/office/scroll", json={"vertical_speed_adjustment": None}, headers=CSRF)
    assert cleared.get_json()["vertical_speed_adjustment"] is None
    assert env["store"].vertical_speed_adjustment("office") is None


def test_clients_page_sets_a_displays_own_weather_location(env, web, monkeypatch):
    monkeypatch.setenv("WEATHER_LATITUDE", "41.8781")
    monkeypatch.setenv("WEATHER_LONGITUDE", "-87.6298")
    publish_registry(env, [caps("hyper")])
    listing = web.get("/api/clients").get_json()
    assert {r["client_id"]: r for r in listing["clients"]}["hyper"]["location"] is None
    assert listing["global_location"] == {"latitude": 41.8781, "longitude": -87.6298}

    saved = web.put("/api/clients/hyper/location", json={"latitude": "40.7128", "longitude": -74.006},
                    headers=CSRF)
    assert saved.status_code == 200
    assert saved.get_json()["location"] == {"latitude": 40.7128, "longitude": -74.006}
    rows = {r["client_id"]: r for r in web.get("/api/clients").get_json()["clients"]}
    assert rows["hyper"]["location"] == {"latitude": 40.7128, "longitude": -74.006}

    bad = web.put("/api/clients/hyper/location", json={"latitude": 95, "longitude": 0}, headers=CSRF)
    assert bad.status_code == 400
    half = web.put("/api/clients/hyper/location", json={"latitude": 40, "longitude": None}, headers=CSRF)
    assert half.status_code == 400
    unknown = web.put("/api/clients/nobody/location", json={"latitude": 1, "longitude": 1}, headers=CSRF)
    assert unknown.status_code == 404
    cleared = web.put("/api/clients/hyper/location", json={"latitude": None, "longitude": None}, headers=CSRF)
    assert cleared.get_json()["location"] is None
    assert env["store"].location("hyper") is None


def test_wizard_join_flow_never_shows_the_credential(env, web):
    created = web.post("/api/clients/provision", headers=CSRF, json={
        "client_id": "den", "display_profile": "display_hat_mini", "server_url": "http://square.local:8765",
        "allow_insecure_transport": True, "join": True})
    assert created.status_code == 201 and created.headers["Cache-Control"] == "no-store"
    body = created.get_json()
    assert "client_env" not in body and "ddc_" not in json.dumps(body)
    assert body["join_command"].endswith('http://square.local:8765/api/v1/join)"')
    assert body["join_expires_in_seconds"] == 1800 and body["allow_insecure_transport"] is True
    provisioning = env["app"].extensions["desk_display_provisioning"]
    assert provisioning.join_pending("den") is not None

    again = web.post("/api/clients/den/join", headers=CSRF, json={"allow_insecure_transport": False,
                                                                  "server_url": "http://square.local:8765"})
    assert again.status_code == 200 and again.get_json()["join_command"] != body["join_command"]
    assert "refuse" in again.get_json()["warnings"][0]
    assert web.post("/api/clients/nobody/join", headers=CSRF, json={}).status_code == 404
    assert web.post("/api/clients/den/join", json={}).status_code == 403


def test_wizard_rejects_bad_input_before_issuing_a_credential(env, web):
    for payload in ({"server_url": "ftp://square.local"}, {"server_url": "http://u:p@square.local:8765"},
                    {"server_url": "http://square.local:8765/path"}, {"friendly_name": "x" * 81}):
        response = web.post("/api/clients/provision", headers=CSRF, json={
            "client_id": "den", "display_profile": "hyperpixel4", **payload})
        assert response.status_code == 400, payload
    assert env["app"].extensions["desk_display_provisioning"].get("den") is None


def test_wizard_provisioning_without_choices_keeps_defaults(env, web, monkeypatch):
    monkeypatch.setenv("DESK_DISPLAY_SERVER_PUBLIC_URL", "https://square.lan:8765")
    issued = web.post("/api/clients/provision", headers=CSRF, json={
        "client_id": "den", "display_profile": "hyperpixel4"}).get_json()
    assert "DESK_DISPLAY_SERVER_URL=https://square.lan:8765" in issued["client_env"]
    assert "ALLOW_INSECURE" not in issued["client_env"] and issued["warnings"] == []


def test_browser_registration_wizard(live_server, browser, env, monkeypatch):
    playlist = env["store"].create("Office", DOC, actor="test")
    page = browser.new_page()
    page.goto(f"{live_server}/clients/add")
    page.wait_for_selector(".check[data-code='bind'] .badge.error")
    page.click("#to-step-2")
    page.check("input[name='display_profile'][value='hyperpixel4_square']")
    page.fill("#friendly-name", "Kitchen Square")
    assert page.input_value("#client-id") == "kitchen-square"
    assert page.input_value("#playlist") == playlist["id"]
    page.fill("#server-url", "http://square.local:8765")
    assert page.is_visible("#insecure-row")
    page.fill("#client-id", "lobby")
    assert "already in use" in page.inner_text("#client-id-hint")
    page.fill("#client-id", "kitchen-square")
    page.click("#create")
    page.wait_for_selector("#step-3:not(.hidden)")
    join = page.inner_text("#join-command")
    assert join.startswith('bash -c "$(curl -fsS -d code=') and "http://square.local:8765/api/v1/join" in join
    assert "Works once" in page.inner_text("#join-expiry")
    provisioning = env["app"].extensions["desk_display_provisioning"]
    assert provisioning.get("kitchen-square")["display_profile"] == "hyperpixel4_square"
    assert provisioning.join_pending("kitchen-square") is not None
    assert "ddc_" not in page.content()  # no credential on the page until asked for

    # By hand instead: the credential appears once and the join code stops working.
    page.on("dialog", lambda dialog: dialog.accept())
    page.click("#manual summary")
    page.click("#show-manual")
    page.wait_for_selector("#manual-body:not(.hidden)")
    command = page.inner_text("#install-command")
    assert "HYPERPIXEL_PANEL=hyperpixel4sq bash Installers/install.sh --mode client" in command
    assert "DESK_DISPLAY_ALLOW_INSECURE_TRANSPORT=1" in command
    assert provisioning.join_pending("kitchen-square") is None

    # The display registers: the wizard notices on its own.
    publish_registry(env, [caps("kitchen-square", "hyperpixel4_square")])
    page.evaluate("poll()")
    page.wait_for_selector("#step-4:not(.hidden)")
    page.wait_for_function("document.querySelector('#connect-status').textContent === 'Connected'")
    assert "Kitchen Square is connected" in page.inner_text("#notice")


def test_browser_screen_config_collapse_and_expand_all_playlists(live_server, browser, monkeypatch):
    monkeypatch.setattr(
        config_ui,
        "_load_active_config",
        lambda: {
            "screens": {"date": 1, "inside": 1, "quad": 1},
            "playlists": {
                "morning": {"label": "Morning", "steps": [{"screen": "date"}]},
                "evening": {"label": "Evening", "steps": [{"screen": "inside"}]},
            },
            "sequence": [{"playlist": "morning"}, {"playlist": "evening"}, {"screen": "quad"}],
        },
    )
    monkeypatch.setattr(config_ui, "_load_active_style_config", lambda: {"screens": {}})
    page = browser.new_page()
    page.goto(f"{live_server}/")
    page.wait_for_selector(".playlist-row")
    visible_rows = "[...document.querySelectorAll('.screen-row')].filter((r) => r.style.display !== 'none').length"
    total_rows = page.locator(".screen-row").count()
    assert total_rows >= 3 and page.evaluate(visible_rows) == total_rows

    page.click("#collapseAllPlaylistsBtn")
    assert page.evaluate(visible_rows) == 0
    assert page.locator(".playlist-row.is-collapsed").count() == page.locator(".playlist-row").count() == 3
    assert page.locator(".screen-row").count() == total_rows  # hidden, not dropped

    # Playlists can still be reordered while collapsed, and the state survives a reload.
    page.click(".playlist-row:has-text('Evening') >> text=Move up")
    names = page.locator(".playlist-row .name").all_inner_texts()
    assert names == ["Ungrouped", "Evening", "Morning"]
    page.reload()
    page.wait_for_selector(".playlist-row")
    assert page.evaluate(visible_rows) == 0

    page.click("#expandAllPlaylistsBtn")
    assert page.evaluate(visible_rows) == total_rows
    assert page.locator(".playlist-row.is-collapsed").count() == 0


def test_browser_update_and_restart_buttons(live_server, browser, env):
    publish_registry(env, [caps("office")])
    page = browser.new_page()
    page.on("dialog", lambda dialog: dialog.accept())
    page.goto(f"{live_server}/clients")
    update = "[data-client-id='office'] button[data-command='update']"
    page.click("[data-client-id='office'] button[data-panel='maintenance']")
    page.wait_for_selector(update)
    page.click(update)
    page.wait_for_selector("#notice.ok")
    assert "update queued" in page.inner_text("#notice")
    page.wait_for_selector(f"{update}[disabled]")
    assert "waiting for the display" in page.inner_text("[data-client-id='office']")
    store = env["app"].extensions["desk_display_client_commands"]
    [command] = store.for_client("office")
    assert command["action"] == "update" and command["state"] == "pending"
