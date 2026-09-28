"""The guided "Add a display" flow: server checks, join codes and setup scripts."""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from display_profiles import PROFILE_PRESETS
from remote_display import registration
from remote_display.provisioning import InvalidJoinCodeError, ProvisioningError, ProvisioningStore

SERVER = {"DESK_DISPLAY_SERVER_HOST": "0.0.0.0", "DESK_DISPLAY_SERVER_PORT": 8765,
          "DESK_DISPLAY_SERVER_PUBLIC_URL": "http://square.local:8765"}


class Clock:
    now = 1_800_000_000.0

    def __call__(self):
        return self.now


# ── Server address ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(("raw", "expected"), [
    ("square.local:8765", "http://square.local:8765"),
    (" https://Square.LAN:8765/ ", "https://square.lan:8765"),
    ("http://192.168.1.20:8765", "http://192.168.1.20:8765"),
    ("http://[fe80::1]:8765", "http://[fe80::1]:8765"),
])
def test_server_urls_are_normalized(raw, expected):
    assert registration.normalize_server_url(raw) == expected


@pytest.mark.parametrize("raw", [
    "", None, "ftp://square.local", "http://user:pw@square.local:8765", "http://square.local:8765/api",
    "http://square.local:8765?x=1", "http://square.local:99999", "http://",
])
def test_bad_server_urls_are_refused(raw):
    with pytest.raises(registration.ServerUrlError):
        registration.normalize_server_url(raw)


def test_suggested_url_prefers_the_configured_one_then_the_browser_host(monkeypatch):
    assert registration.suggested_server_url(SERVER, "anything") == "http://square.local:8765"
    unset = {**SERVER, "DESK_DISPLAY_SERVER_PUBLIC_URL": None}
    assert registration.suggested_server_url(unset, "192.168.1.20") == "http://192.168.1.20:8765"
    tls = {**unset, "DESK_DISPLAY_SERVER_TLS_CERT": "/etc/cert.pem"}
    assert registration.suggested_server_url(tls, "square.local") == "https://square.local:8765"
    monkeypatch.setattr(registration.socket, "gethostname", lambda: "square")
    assert registration.suggested_server_url(unset, "127.0.0.1") == "http://square.local:8765"


# ── Readiness ──────────────────────────────────────────────────────────────


def codes(result):
    return {c["code"]: c["status"] for c in result["checks"]}


def test_a_loopback_server_is_not_ready_and_says_how_to_fix_it():
    probed = []
    result = registration.readiness({**SERVER, "DESK_DISPLAY_SERVER_HOST": "127.0.0.1",
                                     "DESK_DISPLAY_SERVER_PUBLIC_URL": None},
                                    browser_host="square.local", prober=lambda *a: probed.append(a))
    assert result["ready"] is False and probed == []
    bind = next(c for c in result["checks"] if c["code"] == "bind")
    assert "DESK_DISPLAY_SERVER_HOST=0.0.0.0" in bind["fix"]
    public = next(c for c in result["checks"] if c["code"] == "public_url")
    assert public["status"] == "warning"
    assert "DESK_DISPLAY_SERVER_PUBLIC_URL=http://square.local:8765" in public["fix"]


def test_a_reachable_lan_server_is_ready():
    result = registration.readiness(SERVER, prober=lambda host, port: (host, port) == ("192.168.1.20", 8765),
                                    lan=lambda: "192.168.1.20")
    assert result["ready"] is True and result["reachable"] is True
    assert codes(result) == {"bind": "ok", "public_url": "ok", "reachable": "ok", "transport": "warning"}
    assert result["insecure_transport"] is True


def test_an_unreachable_server_shared_enrollment_and_loopback_url_are_errors():
    result = registration.readiness(
        {**SERVER, "DESK_DISPLAY_SERVER_ENROLLMENT": "shared",
         "DESK_DISPLAY_SERVER_PUBLIC_URL": "http://localhost:8765"},
        prober=lambda *a: False, lan=lambda: "192.168.1.20")
    assert result["ready"] is False
    assert codes(result) == {"enrollment": "error", "bind": "ok", "public_url": "ok",
                            "public_url_loopback": "error", "reachable": "error"}


def test_https_needs_no_transport_warning():
    result = registration.readiness({**SERVER, "DESK_DISPLAY_SERVER_PUBLIC_URL": "https://square.lan:8765"},
                                    prober=lambda *a: True, lan=lambda: "10.0.0.2")
    assert "transport" not in codes(result) and result["insecure_transport"] is False


def test_every_profile_has_a_readable_choice():
    choices = registration.profile_choices()
    assert {c["id"] for c in choices} == set(PROFILE_PRESETS)
    assert choices[0]["id"] == "display_hat_mini"
    assert all(c["label"] and c["size"] for c in choices)


def test_repository_url_drops_embedded_credentials(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "remote", "add", "origin",
                    "https://user:ghp_secret@github.com/mrjrask/desk_display.git"], check=True)
    assert registration.repository_url(tmp_path) == "https://github.com/mrjrask/desk_display.git"
    subprocess.run(["git", "-C", str(tmp_path), "remote", "set-url", "origin",
                    "git@github.com:mrjrask/desk_display.git"], check=True)
    assert registration.repository_url(tmp_path) == "git@github.com:mrjrask/desk_display.git"
    assert registration.repository_url(tmp_path / "missing") is None


# ── Join tickets ───────────────────────────────────────────────────────────


def test_a_join_code_works_once_and_issues_the_credential_then(tmp_path):
    clock = Clock()
    store = ProvisioningStore(tmp_path / "clients.json", clock=clock)
    first = store.provision("office", "display_hat_mini")
    code, expires = store.create_join_ticket("office", server_url="http://square.local:8765",
                                             allow_insecure_transport=True)
    assert expires == clock.now + 1800 and store.join_pending("office") == expires
    assert code not in (tmp_path / "clients.json").read_text()
    issued, ticket = store.redeem_join_ticket(code)
    assert ticket["server_url"] == "http://square.local:8765" and ticket["allow_insecure_transport"] is True
    assert store.verify("office", issued.credential) == issued.credential_id
    assert store.verify("office", first.credential) is None
    assert store.join_pending("office") is None
    with pytest.raises(InvalidJoinCodeError):
        store.redeem_join_ticket(code)
    assert [e["action"] for e in store.get("office")["history"]] == ["provisioned", "join_code_created", "joined"]


def test_join_codes_expire_and_are_replaced_or_cancelled(tmp_path):
    clock = Clock()
    store = ProvisioningStore(tmp_path / "clients.json", clock=clock)
    store.provision("office", "display_hat_mini")
    old, _ = store.create_join_ticket("office", server_url="http://s:1")
    new, _ = store.create_join_ticket("office", server_url="http://s:1")
    with pytest.raises(InvalidJoinCodeError):
        store.redeem_join_ticket(old)
    clock.now += 1801
    with pytest.raises(InvalidJoinCodeError):
        store.redeem_join_ticket(new)
    for cancel in (lambda: store.rotate("office"), lambda: store.set_disabled("office", True)):
        store.set_disabled("office", False)
        code, _ = store.create_join_ticket("office", server_url="http://s:1")
        cancel()
        with pytest.raises(InvalidJoinCodeError):
            store.redeem_join_ticket(code)
    with pytest.raises(ProvisioningError):
        store.create_join_ticket("office", server_url="http://s:1")  # disabled
    for bad in (None, "", "x" * 200):
        with pytest.raises(InvalidJoinCodeError):
            store.redeem_join_ticket(bad)


# ── Commands and the setup script ──────────────────────────────────────────


def test_join_command_posts_the_code():
    command = registration.join_command("square.local:8765", "abc_DEF-123")
    assert command == 'bash -c "$(curl -fsS -d code=abc_DEF-123 http://square.local:8765/api/v1/join)"'


ENV_TEXT = ("DESK_DISPLAY_ROLE=client\nDESK_DISPLAY_SERVER_URL=http://square.local:8765\n"
            "DESK_DISPLAY_CLIENT_ID=kitchen\nDESK_DISPLAY_CLIENT_TOKEN=ddc_token\n"
            "DESK_DISPLAY_PROFILE=hyperpixel4_square\nDESK_DISPLAY_ALLOW_INSECURE_TRANSPORT=1\n")


def fake_home(tmp_path, exit_code=0):
    home = tmp_path / "home"
    installers = home / "desk_display" / "Installers"
    installers.mkdir(parents=True)
    (installers / "install.sh").write_text(
        "#!/usr/bin/env bash\n"
        'echo "$HYPERPIXEL_PANEL $*" > "$HOME/installer-args"\n'
        'cp "$4" "$HOME/installer-credentials"\n'
        f"exit {exit_code}\n")
    return home


def run(script, home):
    env = {"HOME": str(home), "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, check=False, timeout=30)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_join_script_writes_private_settings_and_runs_the_installer(tmp_path):
    home = fake_home(tmp_path)
    (home / "desk_display" / ".env.client").write_text("DESK_DISPLAY_CLIENT_ID=old\n")
    script = registration.join_script(ENV_TEXT, "kitchen", "hyperpixel4_square")
    result = run(script, home)
    assert result.returncode == 0, result.stderr
    assert (home / "installer-args").read_text().split() == [
        "hyperpixel4sq", "--mode", "client", "--credentials", str(home / "kitchen.env.client"), "hyperpixel"]
    assert (home / "installer-credentials").read_text() == ENV_TEXT
    assert not (home / "kitchen.env.client").exists()  # removed after a successful install
    backups = list((home / "desk_display").glob(".env.client.before-join-*"))
    assert len(backups) == 1 and "Done" in result.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_failed_install_keeps_the_settings_for_a_rerun(tmp_path):
    home = fake_home(tmp_path, exit_code=3)
    result = run(registration.join_script(ENV_TEXT, "kitchen", "display_hat_mini"), home)
    assert result.returncode == 3 and "Done" not in result.stdout
    path = home / "kitchen.env.client"
    assert path.read_text() == ENV_TEXT and stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_manual_install_command_runs_as_pasted(tmp_path):
    home = fake_home(tmp_path)
    command = registration.install_command(ENV_TEXT, "kitchen", "display_hat_mini")
    result = run(command, home)
    assert result.returncode == 0, result.stderr
    assert (home / "installer-args").read_text().split()[-1] == "display_hat_mini"
    assert not (home / "kitchen.env.client").exists()


def test_the_join_script_clones_when_the_project_is_missing(tmp_path):
    script = registration.join_script(ENV_TEXT, "kitchen", "fallback_default", "https://github.com/o/r.git")
    assert "git clone https://github.com/o/r.git ~/desk_display" in script
    assert "--credentials ~/kitchen.env.client\n" in script  # the installer asks which panel


# ── The render server's join endpoint ──────────────────────────────────────


pytest.importorskip("flask")


@pytest.fixture
def server(tmp_path):
    import display_server

    clock = Clock()
    config = display_server.DisplayServerConfig(
        admin_token="admin-" + "a" * 40, lease_seconds=60, artifact_dir=tmp_path / "artifacts",
        clients_path=tmp_path / "clients.json", playlist_store_path=tmp_path / "playlists.json")
    app = display_server.create_app(config, clock=clock)
    app.config["TESTING"] = True
    return app.test_client(), ProvisioningStore(tmp_path / "clients.json", clock=clock)


def test_join_endpoint_returns_a_setup_script_once(server):
    api, store = server
    store.provision("kitchen", "display_hat_mini")
    code, _ = store.create_join_ticket("kitchen", server_url="http://square.local:8765",
                                       allow_insecure_transport=True)
    response = api.post("/api/v1/join", data={"code": code})
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert response.content_type.startswith("text/x-shellscript")
    script = response.get_data(as_text=True)
    token = next(line.split("=", 1)[1] for line in script.splitlines()
                 if line.startswith("DESK_DISPLAY_CLIENT_TOKEN="))
    assert store.verify("kitchen", token)
    assert "DESK_DISPLAY_SERVER_URL=http://square.local:8765" in script
    assert "DESK_DISPLAY_ALLOW_INSECURE_TRANSPORT=1" in script
    assert "admin-" not in script
    again = api.post("/api/v1/join", data={"code": code})
    assert again.status_code == 401 and again.get_json()["error"] == "invalid_join_code"
    assert api.get(f"/api/v1/join?code={code}").status_code == 405


def test_join_endpoint_counts_bad_codes_as_auth_failures(server):
    api, _store = server
    statuses = [api.post("/api/v1/join", data={"code": f"guess{i}"}).status_code for i in range(30)]
    assert statuses[0] == 401 and 429 in statuses


def test_readme_paths_exist():
    # The wizard's "fix" commands name these; keep them real.
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "restart_services.sh").is_file()
    assert (root / "Installers" / "install.sh").is_file()
