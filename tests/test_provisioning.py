"""Phase 16: per-client provisioning, rotation, revocation and rate limits."""
from __future__ import annotations

import json
import stat

import pytest

import deployment_config as dc
from remote_display.provisioning import (
    AlreadyProvisionedError,
    ProvisioningStore,
    client_env,
    transport_warnings,
)
from remote_display.rate_limit import Limit, RateLimiter

pytest.importorskip("flask")

import display_server  # noqa: E402
from tests.test_display_server import ADMIN_TOKEN, Clock, caps, status  # noqa: E402

SERVER_TOKEN = "server-token-" + "s" * 32
PROVIDER_SECRET = "provider-" + "p" * 30


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


# ── The store ──────────────────────────────────────────────────────────────


def test_credentials_are_stored_as_hashes_in_a_private_file(tmp_path):
    store = ProvisioningStore(tmp_path / "clients.json")
    issued = store.provision("office", "hyperpixel4")
    text = (tmp_path / "clients.json").read_text()
    assert issued.credential not in text
    assert stat.S_IMODE((tmp_path / "clients.json").stat().st_mode) == 0o600
    assert "credential_hash" not in json.dumps(store.records())
    assert store.verify("office", issued.credential) == issued.credential_id
    assert store.verify("office", issued.credential + "x") is None
    assert store.verify("kitchen", issued.credential) is None


def test_existing_ids_are_never_reissued_silently(tmp_path):
    store = ProvisioningStore(tmp_path / "clients.json")
    store.provision("office", "hyperpixel4")
    with pytest.raises(AlreadyProvisionedError):
        store.provision("office", "hyperpixel4")


def test_rotation_revocation_and_disabling_keep_history(tmp_path):
    store = ProvisioningStore(tmp_path / "clients.json")
    first = store.provision("office", "hyperpixel4")
    second = store.rotate("office")
    assert store.verify("office", first.credential) is None
    assert store.verify("office", second.credential) == second.credential_id
    store.set_disabled("office", True)
    assert store.verify("office", second.credential) is None
    store.set_disabled("office", False)
    assert store.verify("office", second.credential) == second.credential_id
    store.revoke("office")
    assert store.verify("office", second.credential) is None
    third = store.rotate("office")  # a revoked client comes back only with a new credential
    assert store.verify("office", third.credential)
    actions = [event["action"] for event in store.get("office")["history"]]
    assert actions == ["provisioned", "rotated", "disabled", "enabled", "revoked", "rotated"]


def test_a_second_process_sees_changes(tmp_path):
    ui = ProvisioningStore(tmp_path / "clients.json")
    server = ProvisioningStore(tmp_path / "clients.json")
    issued = ui.provision("office", "hyperpixel4")
    assert server.verify("office", issued.credential)
    ui.revoke("office")
    assert server.current_credential_id("office") is None


def test_client_env_carries_only_the_clients_own_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DESK_DISPLAY_SERVER_AUTH_TOKEN", SERVER_TOKEN)
    monkeypatch.setenv("OWM_API_KEY", PROVIDER_SECRET)
    issued = ProvisioningStore(tmp_path / "c.json").provision("office", "hyperpixel4")
    text, warnings = client_env(issued, "https://render.lan:8765")
    (tmp_path / ".env.client").write_text(text)
    env = dc.parse_env_file(tmp_path / ".env.client")
    assert env["DESK_DISPLAY_CLIENT_TOKEN"] == issued.credential and warnings == []
    assert SERVER_TOKEN not in text and PROVIDER_SECRET not in text
    assert dc.validate(dc.Role.CLIENT, env).ok


def test_plaintext_http_beyond_loopback_is_flagged():
    assert transport_warnings("http://render.lan:8765")
    assert transport_warnings("http://127.0.0.1:8765") == []
    assert transport_warnings("https://render.lan:8765") == []


def test_cli_prints_the_credential_once(tmp_path, capsys):
    from remote_display.provisioning import _cli

    store = str(tmp_path / "c.json")
    assert _cli(["--store", store, "provision", "office", "--profile", "hyperpixel4",
                 "--server-url", "http://render.lan"]) == 0
    out, err = capsys.readouterr()
    assert "DESK_DISPLAY_CLIENT_TOKEN=ddc_" in out and "plain HTTP" in err
    assert _cli(["--store", store, "list"]) == 0
    listed = capsys.readouterr().out
    assert "office\tactive" in listed and "ddc_" not in listed


# ── The server ─────────────────────────────────────────────────────────────


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def env(tmp_path, clock):
    config = display_server.DisplayServerConfig(
        admin_token=ADMIN_TOKEN, auth_token=SERVER_TOKEN, lease_seconds=60,
        artifact_dir=tmp_path / "artifacts", clients_path=tmp_path / "clients.json",
        playlist_store_path=tmp_path / "playlists.json",
    )
    app = display_server.create_app(config, clock=clock)
    app.config["TESTING"] = True
    return app, app.test_client()


def provision(api, client_id="office", **extra):
    response = api.post("/api/v1/admin/clients", headers=bearer(ADMIN_TOKEN),
                        json={"client_id": client_id, "display_profile": "hyperpixel4", **extra})
    assert response.status_code == 201, response.get_json()
    return response.get_json()


def register(api, token, client_id="office"):
    return api.post("/api/v1/register", headers=bearer(token), json={"capabilities": caps(client_id)})


def test_provisioned_credentials_are_the_default_and_the_shared_token_is_refused(env):
    _app, api = env
    assert register(api, SERVER_TOKEN).status_code == 401
    issued = provision(api)
    assert register(api, issued["client_credential"]).status_code == 201


def test_the_credential_is_shown_once(env):
    _app, api = env
    issued = provision(api)
    assert issued["client_credential"] in issued["client_env"]
    assert issued["client_credential"] not in json.dumps(
        api.get("/api/v1/admin/clients", headers=bearer(ADMIN_TOKEN)).get_json())
    assert issued["client_credential"] not in json.dumps(
        api.get("/api/v1/admin/status", headers=bearer(ADMIN_TOKEN)).get_json())
    assert api.post("/api/v1/admin/clients", headers=bearer(ADMIN_TOKEN),
                    json={"client_id": "office", "display_profile": "hyperpixel4"}).status_code == 409


def test_provisioning_can_assign_a_playlist(env, tmp_path):
    from remote_display.playlist_store import PlaylistStore

    app, api = env
    playlist = PlaylistStore(tmp_path / "playlists.json").create(
        "Office", {"screens": {"date": 1}, "sequence": []}, actor="test")
    provision(api, playlist_id=playlist["id"])
    assert PlaylistStore(tmp_path / "playlists.json").assignment_for("office").playlist_id == playlist["id"]
    response = api.post("/api/v1/admin/clients", headers=bearer(ADMIN_TOKEN),
                        json={"client_id": "den", "display_profile": "hyperpixel4", "playlist_id": "pl-nope"})
    assert response.status_code == 400


def test_a_credential_only_registers_its_own_client(env):
    _app, api = env
    office = provision(api)["client_credential"]
    provision(api, "kitchen")
    assert register(api, office, "kitchen").status_code == 401


def test_cross_client_requests_are_refused(env):
    _app, api = env
    office = register(api, provision(api)["client_credential"]).get_json()["client_credential"]
    register(api, provision(api, "kitchen")["client_credential"], "kitchen")
    for path in ("/api/v1/clients/kitchen/config", "/api/v1/clients/kitchen/manifest"):
        assert api.get(path, headers=bearer(office)).status_code == 401
    assert api.post("/api/v1/clients/kitchen/heartbeat", headers=bearer(office),
                    json={"status": status("kitchen")}).status_code == 401


def test_rotation_ends_the_old_lease_and_leaves_other_clients_alone(env):
    _app, api = env
    first = provision(api)["client_credential"]
    lease = register(api, first).get_json()["client_credential"]
    other = register(api, provision(api, "kitchen")["client_credential"], "kitchen").get_json()["client_credential"]
    rotated = api.post("/api/v1/admin/clients/office/rotate", headers=bearer(ADMIN_TOKEN)).get_json()
    assert api.get("/api/v1/clients/office/config", headers=bearer(lease)).status_code == 401
    assert register(api, first).status_code == 401
    assert register(api, rotated["client_credential"]).status_code == 201  # no wait for the old lease
    assert api.get("/api/v1/clients/kitchen/config", headers=bearer(other)).status_code == 200


def test_revocation_is_immediate_and_rotation_restores(env):
    _app, api = env
    token = provision(api)["client_credential"]
    lease = register(api, token).get_json()["client_credential"]
    assert api.post("/api/v1/admin/clients/office/revoke", headers=bearer(ADMIN_TOKEN)).get_json()["state"] == "revoked"
    assert api.get("/api/v1/clients/office/manifest", headers=bearer(lease)).status_code == 401
    assert register(api, token).status_code == 401
    fresh = api.post("/api/v1/admin/clients/office/rotate", headers=bearer(ADMIN_TOKEN)).get_json()
    assert register(api, fresh["client_credential"]).status_code == 201


def test_disabled_clients_keep_their_assignment_and_come_back(env, tmp_path):
    from remote_display.playlist_store import PlaylistStore

    _app, api = env
    playlist = PlaylistStore(tmp_path / "playlists.json").create(
        "Office", {"screens": {"date": 1}, "sequence": []}, actor="test")
    token = provision(api, playlist_id=playlist["id"])["client_credential"]
    register(api, token)
    assert api.post("/api/v1/admin/clients/office/disable", headers=bearer(ADMIN_TOKEN)).status_code == 200
    refused = register(api, token)
    assert refused.status_code == 403 and refused.get_json()["error"] == "client_disabled"
    assert PlaylistStore(tmp_path / "playlists.json").assignment_for("office") is not None
    api.post("/api/v1/admin/clients/office/enable", headers=bearer(ADMIN_TOKEN))
    assert register(api, token).status_code == 201


def test_admin_endpoints_need_the_admin_token(env):
    _app, api = env
    token = provision(api)["client_credential"]
    lease = register(api, token).get_json()["client_credential"]
    for credential in (None, token, lease, SERVER_TOKEN):
        headers = {} if credential is None else bearer(credential)
        assert api.get("/api/v1/admin/clients", headers=headers).status_code == 401
        assert api.post("/api/v1/admin/clients/office/rotate", headers=headers).status_code == 401
        assert api.post("/api/v1/admin/clients", headers=headers,
                        json={"client_id": "x", "display_profile": "hyperpixel4"}).status_code == 401


def test_provisioning_response_warns_about_plaintext(tmp_path, clock):
    config = display_server.DisplayServerConfig(
        admin_token=ADMIN_TOKEN, artifact_dir=tmp_path / "a", clients_path=tmp_path / "c.json",
        public_url="http://render.lan:8765", rate_limits=False)
    api = display_server.create_app(config, clock=clock).test_client()
    body = provision(api)
    assert body["warnings"] and "plain HTTP" in body["warnings"][0]


def test_shared_enrollment_is_opt_in(tmp_path, clock):
    config = display_server.DisplayServerConfig(enrollment="shared", auth_token=SERVER_TOKEN,
                                                artifact_dir=tmp_path / "a", clients_path=tmp_path / "c.json")
    api = display_server.create_app(config, clock=clock).test_client()
    assert register(api, SERVER_TOKEN).status_code == 201


# ── Rate limits ────────────────────────────────────────────────────────────


def test_token_bucket_refills():
    now = [0.0]
    limiter = RateLimiter({"x": Limit(burst=2, per_second=1)}, clock=lambda: now[0])
    assert limiter.hit("x", "a") == 0 and limiter.hit("x", "a") == 0
    assert limiter.hit("x", "a") == pytest.approx(1)
    assert limiter.hit("x", "b") == 0  # buckets are per key
    now[0] += 1
    assert limiter.hit("x", "a") == 0


def test_guessing_credentials_locks_out_the_address(env):
    _app, api = env
    token = provision(api)["client_credential"]
    codes = [register(api, "ddc_wrong").status_code for _ in range(25)]
    assert codes[0] == 401 and codes[-1] == 429
    locked = register(api, token)
    assert locked.status_code == 429 and int(locked.headers["Retry-After"]) >= 1


def test_a_runaway_client_is_throttled_without_affecting_others(env):
    _app, api = env
    office = register(api, provision(api)["client_credential"]).get_json()["client_credential"]
    kitchen = register(api, provision(api, "kitchen")["client_credential"], "kitchen").get_json()["client_credential"]
    codes = [api.post("/api/v1/clients/office/heartbeat", headers=bearer(office),
                      json={"status": status("office")}).status_code for _ in range(40)]
    assert codes[0] == 200 and codes[-1] == 429
    assert api.post("/api/v1/clients/kitchen/heartbeat", headers=bearer(kitchen),
                    json={"status": status("kitchen")}).status_code == 200
