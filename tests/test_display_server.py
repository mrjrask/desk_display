"""API tests for the render server's registration, heartbeat and demand endpoints."""
from __future__ import annotations

import pytest

pytest.importorskip("flask")

import display_server  # noqa: E402
from display_profiles import PROFILE_PRESETS  # noqa: E402
from remote_display.models import ScreenRevisions  # noqa: E402
from remote_display.registry import Assignment, ClientRegistry  # noqa: E402

SERVER_TOKEN = "server-token-" + "s" * 32
ADMIN_TOKEN = "admin-token-" + "a" * 32


class Clock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def caps(client_id="office", profile="hyperpixel4", **overrides):
    preset = PROFILE_PRESETS[profile]
    wire = {
        "type": "client_capabilities",
        "version": 1,
        "protocol_version": 1,
        "client_software_version": "0.1",
        "client_id": client_id,
        "display_profile": profile,
        "logical_width": preset.width,
        "logical_height": preset.height,
        "image_formats": ["PNG"],
        "color_modes": [preset.color_mode],
        "render_package_versions": [1],
    }
    wire.update(overrides)
    return wire


def demand(client_id="office", screens=("date", "weather1")):
    return {
        "type": "client_demand",
        "version": 1,
        "client_id": client_id,
        "playlist_revision": "pl-1",
        "required_screens": list(screens),
        "package_capabilities": {"render_package_versions": [1], "image_formats": ["PNG"]},
        "sync_interval_seconds": 30,
    }


def status(client_id="office", **overrides):
    wire = {
        "type": "client_status",
        "version": 1,
        "client_id": client_id,
        "playback_state": "playing",
        "accepted_revisions": {"manifest_revision": "m-1", "playlist_revision": "pl-1"},
        "current_screen": "date",
        "physical_rotation": 90,
    }
    wire.update(overrides)
    return wire


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def server(tmp_path, clock):
    config = display_server.DisplayServerConfig(
        auth_token=SERVER_TOKEN,
        admin_token=ADMIN_TOKEN,
        lease_seconds=60,
        static_clients={"lobby": "hdmi_1080p"},
        artifact_dir=tmp_path / "artifacts",
    )
    assignments = {
        "lobby": Assignment("lobby-loop", "rev-3", ("date", "news headlines")),
        "office": Assignment("default", "rev-9", ("date",)),
    }
    app = display_server.create_app(config, assignments=assignments.get, clock=clock)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def api(server):
    return server.test_client()


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def register(api, client_id="office", *, token=SERVER_TOKEN, **body):
    payload = {"capabilities": caps(client_id), **body}
    return api.post("/api/v1/register", json=payload, headers=bearer(token))


def registered(api, client_id="office", **body):
    response = register(api, client_id, **body)
    assert response.status_code == 201, response.get_json()
    return response.get_json()["client_credential"]


def admin_status(api):
    response = api.get("/api/v1/admin/status", headers=bearer(ADMIN_TOKEN))
    assert response.status_code == 200
    return response.get_json()


# ── Registration ────────────────────────────────────────────────────────────


def test_registration_success(api):
    response = register(api, demand=demand())
    assert response.status_code == 201
    body = response.get_json()
    assert body["accepted"] is True
    assert body["accepted_protocol_versions"] == [1]
    assert body["server_software_version"]
    assert body["assignment_state"] == "assigned"
    assert body["assigned_playlist"] == {"playlist_id": "default", "playlist_revision": "rev-9"}
    assert body["manifest_revision"].startswith("m-")
    assert body["lease_seconds"] == 60
    assert body["lease_expires_at"].endswith("Z")
    assert body["heartbeat_interval_seconds"] == 20
    assert body["sync_interval_seconds"] == 30
    assert len(body["client_credential"]) >= 40
    assert body["renewed"] is False


def test_unassigned_client(api):
    body = register(api, "kitchen").get_json()
    assert body["assignment_state"] == "unassigned" and body["assigned_playlist"] is None


@pytest.mark.parametrize("token", ["", "wrong", SERVER_TOKEN + "x", ADMIN_TOKEN, "ünïcode"])
def test_registration_bad_credentials(api, token):
    response = api.post("/api/v1/register", json={"capabilities": caps()}, headers=bearer(token))
    assert response.status_code == 401
    response = api.post("/api/v1/register", json={"capabilities": caps()})
    assert response.status_code == 401


def test_unauthenticated_mode_still_issues_client_credentials(tmp_path, clock):
    config = display_server.DisplayServerConfig(allow_unauthenticated=True, artifact_dir=tmp_path)
    api = display_server.create_app(config, clock=clock).test_client()
    response = api.post("/api/v1/register", json={"capabilities": caps()})
    assert response.status_code == 201
    assert api.get("/api/v1/clients/office/config").status_code == 401


@pytest.mark.parametrize(
    "payload, field",
    [
        ({"capabilities": caps(display_profile="crt")}, "capabilities.display_profile"),
        ({"capabilities": caps(logical_width=480, logical_height=800)}, "capabilities.logical_width"),
        ({"capabilities": caps(client_id="../../etc")}, "capabilities.client_id"),
        ({"capabilities": caps(), "demand": demand(screens=["not a screen"])}, "demand.required_screens[0]"),
        ({"capabilities": caps(), "extra": 1}, "extra"),
        ({}, "capabilities"),
    ],
)
def test_registration_malformed_payloads(api, payload, field):
    response = api.post("/api/v1/register", json=payload, headers=bearer(SERVER_TOKEN))
    assert response.status_code == 400
    assert response.get_json()["field"] == field


def test_registration_rejects_non_json_and_oversized_bodies(api):
    response = api.post("/api/v1/register", data="capabilities", headers=bearer(SERVER_TOKEN))
    assert response.status_code == 400
    response = api.post("/api/v1/register", json=["capabilities"], headers=bearer(SERVER_TOKEN))
    assert response.status_code == 400
    big = {"capabilities": caps(), "demand": "x" * (display_server.MAX_REQUEST_BYTES + 1)}
    response = api.post("/api/v1/register", json=big, headers=bearer(SERVER_TOKEN))
    assert response.status_code == 413
    assert response.get_json()["error"] == "request_entity_too_large"


def test_unsupported_versions(api):
    response = api.post("/api/v1/register", json={"capabilities": caps(protocol_version=99)},
                        headers=bearer(SERVER_TOKEN))
    assert response.status_code == 409
    assert response.get_json()["error"] == "incompatible_protocol_version"
    response = api.post("/api/v1/register", json={"capabilities": caps(render_package_versions=[7])},
                        headers=bearer(SERVER_TOKEN))
    assert response.status_code == 409
    assert response.get_json()["error"] == "unsupported_capabilities"
    wire = caps()
    wire["version"] = 2
    response = api.post("/api/v1/register", json={"capabilities": wire}, headers=bearer(SERVER_TOKEN))
    assert response.status_code == 400 and response.get_json()["field"] == "capabilities.version"


def test_demand_must_match_registering_client(api):
    response = register(api, demand=demand("someone-else"))
    assert response.status_code == 400


# ── Duplicate IDs ───────────────────────────────────────────────────────────


def test_duplicate_id_is_refused_while_lease_is_active(api, clock):
    first = registered(api)
    response = register(api)
    assert response.status_code == 409
    body = response.get_json()
    assert body["error"] == "client_id_in_use" and body["retry_after_seconds"] == 60
    assert "client_credential" not in body
    # The original holder is unaffected.
    assert api.get("/api/v1/clients/office/config", headers=bearer(first)).status_code == 200
    # A wrong credential does not help.
    assert register(api, client_credential="guess").status_code == 409


def test_holder_can_renew_registration(api):
    first = registered(api)
    response = register(api, client_credential=first)
    assert response.status_code == 200 and response.get_json()["renewed"] is True
    second = response.get_json()["client_credential"]
    assert second != first
    assert api.get("/api/v1/clients/office/config", headers=bearer(first)).status_code == 401
    assert api.get("/api/v1/clients/office/config", headers=bearer(second)).status_code == 200


def test_duplicate_id_can_register_after_expiry(api, clock):
    first = registered(api)
    clock.advance(61)
    second = registered(api)
    assert api.get("/api/v1/clients/office/config", headers=bearer(first)).status_code == 401
    assert api.get("/api/v1/clients/office/config", headers=bearer(second)).status_code == 200


# ── Cross-client access and traversal ───────────────────────────────────────


@pytest.mark.parametrize("endpoint", ["config", "manifest", "assets/x.png"])
def test_client_cannot_read_another_client(api, endpoint):
    office = registered(api, "office")
    registered(api, "kitchen")
    assert api.get(f"/api/v1/clients/kitchen/{endpoint}", headers=bearer(office)).status_code == 401
    assert api.get(f"/api/v1/clients/kitchen/{endpoint}").status_code == 401
    assert api.get(f"/api/v1/clients/kitchen/{endpoint}", headers=bearer(SERVER_TOKEN)).status_code == 401


def test_client_cannot_heartbeat_as_another_client(api):
    office = registered(api, "office")
    registered(api, "kitchen")
    response = api.post("/api/v1/clients/kitchen/heartbeat", json={"status": status("kitchen")},
                        headers=bearer(office))
    assert response.status_code == 401
    response = api.post("/api/v1/clients/office/heartbeat", json={"status": status("kitchen")},
                        headers=bearer(office))
    assert response.status_code == 400


def test_client_endpoints_do_not_expose_credentials(api):
    credential = registered(api)
    for endpoint in ("config", "manifest"):
        text = api.get(f"/api/v1/clients/office/{endpoint}", headers=bearer(credential)).get_data(as_text=True)
        assert credential not in text and SERVER_TOKEN not in text and ADMIN_TOKEN not in text
    text = api.get("/api/v1/admin/status", headers=bearer(ADMIN_TOKEN)).get_data(as_text=True)
    assert credential not in text and "credential" not in text


def test_asset_serving_and_traversal(api, server):
    root = server.extensions["desk_display_config"].artifact_dir
    (root / "ab").mkdir(parents=True)
    (root / "ab" / "abc123.png").write_bytes(b"\x89PNG artifact")
    (root.parent / "secret.txt").write_text("private", encoding="utf-8")
    credential = registered(api)
    headers = bearer(credential)
    response = api.get("/api/v1/clients/office/assets/ab/abc123.png", headers=headers)
    assert response.status_code == 200 and response.data == b"\x89PNG artifact"
    assert response.headers["ETag"]
    again = api.get("/api/v1/clients/office/assets/ab/abc123.png",
                    headers={**headers, "If-None-Match": response.headers["ETag"]})
    assert again.status_code == 304
    for path in ("../secret.txt", "ab/../../secret.txt", "..%2Fsecret.txt", "%2e%2e/secret.txt",
                 ".hidden", "ab/missing.png", "a/b/c/d/e.png", "ab"):
        assert api.get(f"/api/v1/clients/office/assets/{path}", headers=headers).status_code == 404, path


def test_invalid_client_id_in_path(api):
    credential = registered(api)
    assert api.get("/api/v1/clients/..%2Foffice/config", headers=bearer(credential)).status_code in {401, 404}
    assert api.get("/api/v1/clients/bad id/config", headers=bearer(credential)).status_code == 401


# ── Heartbeats and lease lifecycle ──────────────────────────────────────────


def test_heartbeat_renews_lease_and_records_status(api, clock):
    credential = registered(api)
    clock.advance(50)
    response = api.post("/api/v1/clients/office/heartbeat",
                        json={"status": status(), "demand": demand(screens=("inside",))},
                        headers=bearer(credential))
    assert response.status_code == 200
    body = response.get_json()
    assert body["lease_expires_at"] and body["heartbeat_interval_seconds"] == 20
    clock.advance(50)  # 100s after registration, 50s after the heartbeat
    assert api.get("/api/v1/clients/office/config", headers=bearer(credential)).status_code == 200
    office = next(c for c in admin_status(api)["clients"] if c["client_id"] == "office")
    assert office["lease_state"] == "active"
    assert office["status"]["physical_rotation"] == 90
    assert office["demand"]["required_screens"] == ["inside"]


def test_malformed_heartbeat(api):
    credential = registered(api)
    for body in ({}, {"status": {"type": "client_status"}}, {"status": status(physical_rotation=45)},
                 {"status": status(), "other": 1}):
        response = api.post("/api/v1/clients/office/heartbeat", json=body, headers=bearer(credential))
        assert response.status_code == 400, body


def test_missed_heartbeats_expire_dynamic_demand(api, clock):
    credential = registered(api, demand=demand())
    assert "office" in {d["client_id"] for d in admin_status(api)["demand"]}
    clock.advance(61)
    status_body = admin_status(api)
    assert "office" not in {d["client_id"] for d in status_body["demand"]}
    office = next(c for c in status_body["clients"] if c["client_id"] == "office")
    assert office["lease_state"] == "expired" and office["demand"] is None
    response = api.post("/api/v1/clients/office/heartbeat", json={"status": status()},
                        headers=bearer(credential))
    assert response.status_code == 401
    assert response.get_json()["error"] == "invalid_client_credential"


def test_lease_boundary(api, clock):
    credential = registered(api)
    clock.advance(59.9)
    assert api.get("/api/v1/clients/office/config", headers=bearer(credential)).status_code == 200
    clock.advance(0.1)
    assert api.get("/api/v1/clients/office/config", headers=bearer(credential)).status_code == 401


# ── Static and pre-render demand ────────────────────────────────────────────


def test_static_demand_exists_without_connection(api):
    body = admin_status(api)
    lobby = next(c for c in body["clients"] if c["client_id"] == "lobby")
    assert lobby["static"] is True and lobby["lease_state"] == "static"
    assert {"source": "static", "client_id": "lobby", "display_profile": "hdmi_1080p",
            "screens": ["date", "news headlines"]} in body["demand"]


def test_static_and_dynamic_clients_coexist(api, clock):
    credential = registered(api, "lobby", capabilities=caps("lobby", "hdmi_1080p"))
    registered(api, "office", demand=demand())
    sources = {d["client_id"]: d["source"] for d in admin_status(api)["demand"]}
    assert sources == {"lobby": "dynamic", "office": "dynamic"}
    clock.advance(61)
    sources = {d["client_id"]: d["source"] for d in admin_status(api)["demand"]}
    assert sources == {"lobby": "static"}
    # The static client may register again after its lease lapsed.
    assert api.get("/api/v1/clients/lobby/config", headers=bearer(credential)).status_code == 401
    assert register(api, "lobby", capabilities=caps("lobby", "hdmi_1080p")).status_code == 201


def test_static_client_profile_is_enforced(api):
    response = register(api, "lobby")  # registers as hyperpixel4, configured as hdmi_1080p
    assert response.status_code == 409
    assert response.get_json()["error"] == "static_profile_mismatch"


def test_prerender_demand_merges_with_clients(api, server):
    headers = bearer(ADMIN_TOKEN)
    response = api.put("/api/v1/admin/prerender/warm-oled",
                       json={"display_profile": "waveshare_oled_128x64", "screens": ["date", "time"]},
                       headers=headers)
    assert response.status_code == 200
    assert response.get_json()["screens"] == ["date", "nixie"]
    registered(api, "office", demand=demand())
    entries = admin_status(api)["demand"]
    assert {e["source"] for e in entries} == {"static", "dynamic", "prerender"}

    registry: ClientRegistry = server.extensions["desk_display_registry"]
    revisions = {s: ScreenRevisions("s1", "d1", "r1") for s in ("date", "weather1", "news headlines", "nixie")}
    plan = registry.render_plan(revisions)
    by_profile = {}
    for key in plan:
        by_profile.setdefault(key.render_profile, set()).add(key.screen_id)
    assert by_profile == {
        "hyperpixel4": {"date", "weather1"},
        "hdmi_1080p": {"date", "news headlines"},
        "waveshare_oled_128x64": {"date", "nixie"},
    }

    assert api.delete("/api/v1/admin/prerender/warm-oled", headers=headers).status_code == 204
    assert api.delete("/api/v1/admin/prerender/warm-oled", headers=headers).status_code == 404
    bad = api.put("/api/v1/admin/prerender/x", json={"display_profile": "crt", "screens": []}, headers=headers)
    assert bad.status_code == 400


def test_equivalent_clients_share_render_work(api, server):
    registered(api, "office", demand=demand("office", ("date", "weather1")))
    registered(api, "den", capabilities=caps("den"), demand=demand("den", ("weather1", "date")))
    registry: ClientRegistry = server.extensions["desk_display_registry"]
    revisions = {s: ScreenRevisions("s1", "d1", "r1") for s in ("date", "weather1", "news headlines")}
    plan = {k: v for k, v in registry.render_plan(revisions).items() if k.render_profile == "hyperpixel4"}
    assert len(plan) == 2 and all(v == {"office", "den"} for v in plan.values())


# ── Health and admin authorization ──────────────────────────────────────────


def test_health_is_public_and_minimal(api):
    registered(api)
    response = api.get("/api/v1/health")
    assert response.status_code == 200 and response.get_json() == {"status": "ok"}


@pytest.mark.parametrize("token", [None, "wrong", SERVER_TOKEN])
def test_admin_requires_admin_token(api, token):
    headers = {} if token is None else bearer(token)
    assert api.get("/api/v1/admin/status", headers=headers).status_code == 401
    assert api.put("/api/v1/admin/prerender/x", json={"display_profile": "hdmi_1080p", "screens": []},
                   headers=headers).status_code == 401
    assert api.post("/api/v1/admin/clients/lobby/disable", headers=headers).status_code == 401


def test_admin_rejects_client_credentials(api):
    credential = registered(api)
    assert api.get("/api/v1/admin/status", headers=bearer(credential)).status_code == 401


def test_admin_disabled_without_token(tmp_path, clock):
    config = display_server.DisplayServerConfig(auth_token=SERVER_TOKEN, artifact_dir=tmp_path)
    api = display_server.create_app(config, clock=clock).test_client()
    response = api.get("/api/v1/admin/status", headers=bearer(SERVER_TOKEN))
    assert response.status_code == 403 and response.get_json()["error"] == "admin_disabled"


def test_admin_can_disable_and_enable_clients(api):
    credential = registered(api)
    headers = bearer(ADMIN_TOKEN)
    assert api.post("/api/v1/admin/clients/office/disable", headers=headers).get_json()["disabled"] is True
    assert api.get("/api/v1/clients/office/config", headers=bearer(credential)).status_code == 401
    response = register(api)
    assert response.status_code == 403 and response.get_json()["error"] == "client_disabled"
    assert api.post("/api/v1/admin/clients/office/enable", headers=headers).status_code == 200
    assert register(api).status_code == 201
    assert api.post("/api/v1/admin/clients/nobody/disable", headers=headers).status_code == 404
    assert api.post("/api/v1/admin/clients/office/delete", headers=headers).status_code == 404


def test_config_from_env_and_role_check(monkeypatch):
    config = display_server.DisplayServerConfig.from_env({
        "DESK_DISPLAY_SERVER_AUTH_TOKEN": SERVER_TOKEN,
        "DESK_DISPLAY_STATIC_CLIENTS": "lobby:hdmi_1080p",
        "DESK_DISPLAY_CLIENT_LEASE_SECONDS": "120",
    })
    assert config.static_clients == {"lobby": "hdmi_1080p"} and config.lease_seconds == 120
    assert config.admin_token is None
    monkeypatch.setenv("DESK_DISPLAY_ROLE", "standalone")
    with pytest.raises(SystemExit, match="DESK_DISPLAY_ROLE=server"):
        display_server.run_display_server()


# ── Feed server stays backward compatible ───────────────────────────────────


def test_feed_server_routes_are_unchanged():
    import feed_server

    rules = {(rule.rule, tuple(sorted(rule.methods - {"HEAD", "OPTIONS"})))
             for rule in feed_server.app.url_map.iter_rules()}
    assert ("/api/feed/<source>/status", ("POST",)) in rules
    assert not any(rule.startswith("/api/v1") for rule, _ in rules)
