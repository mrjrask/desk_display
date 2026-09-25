"""Phase 15: service definitions and lifecycle for each installation mode."""
from __future__ import annotations

import random
from pathlib import Path

import pytest

import deployment_config as dc
import service_units as su

ROOT = Path(__file__).resolve().parents[1]


def units(mode):
    rendered = su.render_units(mode, project_dir="/opt/dd", python="/opt/dd/venv/bin/python", user="pi")
    return {name: su.parse_unit(text) for name, text in rendered.items()}


# ── Service definitions ────────────────────────────────────────────────────


@pytest.mark.parametrize("mode, expected", [
    ("standalone", {su.STANDALONE_SERVICE, su.CONFIG_UI_SERVICE}),
    ("server", {su.SERVER_SERVICE, su.CONFIG_UI_SERVICE}),
    ("client", {su.CLIENT_SERVICE}),
    ("combined", {su.SERVER_SERVICE, su.CLIENT_SERVICE, su.CONFIG_UI_SERVICE}),
])
def test_each_mode_installs_its_services_and_disables_the_rest(mode, expected):
    assert set(su.services_for(mode)) == expected
    assert set(su.disabled_for(mode)) == set(su.ALL_SERVICES) - expected


def test_only_the_standalone_mode_runs_main_py():
    for mode in su.Mode:
        scripts = {section["Service"]["ExecStart"][0].split()[-1] for section in units(mode).values()}
        assert ("/opt/dd/main.py" in scripts) == (mode is su.Mode.STANDALONE)


def test_combined_panel_is_an_ordinary_client_of_its_own_server():
    combined = units("combined")
    client, server = combined[su.CLIENT_SERVICE], combined[su.SERVER_SERVICE]
    assert client == units("client")[su.CLIENT_SERVICE]  # same unit as any remote client
    assert client["Service"]["ExecStart"] == ["/opt/dd/venv/bin/python /opt/dd/display_client.py"]
    assert client["Service"]["EnvironmentFile"] == ["-/opt/dd/.env.client"]
    assert server["Service"]["EnvironmentFile"] == ["-/opt/dd/.env"]
    assert client["Service"]["Environment"] == ["DESK_DISPLAY_ROLE=client"]
    assert server["Service"]["Environment"] == ["DESK_DISPLAY_ROLE=server"]


def test_server_restart_never_stops_or_delays_the_local_client():
    client = units("combined")[su.CLIENT_SERVICE]["Unit"]
    for key in su.COUPLING_KEYS:
        assert su.SERVER_SERVICE not in " ".join(client.get(key, []))
    # Not ordered after the server either: the panel starts from its cache first.
    assert su.SERVER_SERVICE not in " ".join(client.get("After", []))


def test_the_standalone_renderer_and_a_client_never_share_the_panel():
    assert su.CLIENT_SERVICE in units("standalone")[su.STANDALONE_SERVICE]["Unit"]["Conflicts"]
    assert su.STANDALONE_SERVICE in units("client")[su.CLIENT_SERVICE]["Unit"]["Conflicts"]


def test_units_keep_the_graceful_shutdown_contract():
    for mode in su.Mode:
        for unit in units(mode).values():
            service = unit["Service"]
            assert service["KillSignal"] == ["SIGTERM"] and service["Restart"] == ["always"]
            assert "ExecStop" not in service


def test_cli_writes_the_unit_files(tmp_path, capsys):
    assert su._cli(["--mode", "combined", "--output", str(tmp_path), "--project-dir", "/opt/dd"]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(su.services_for("combined"))
    assert f"disable: {su.STANDALONE_SERVICE}" in capsys.readouterr().out


def test_restart_script_knows_the_split_services_in_dependency_order():
    script = (ROOT / "scripts" / "restart_services.sh").read_text(encoding="utf-8")
    block = script.split("ORDERED_SERVICES=(", 1)[1].split(")", 1)[0].split()
    assert block.index(su.SERVER_SERVICE) < block.index(su.CLIENT_SERVICE)


# ── Roles ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("role, runs", [("server", "display_server.py"), ("client", "display_client.py")])
def test_main_py_refuses_split_roles(role, runs):
    with pytest.raises(SystemExit) as info:
        dc.require_role("main.py", dc.Role.STANDALONE, {"DESK_DISPLAY_ROLE": role})
    assert runs in str(info.value)
    dc.require_role("main.py", dc.Role.STANDALONE, {})


def test_main_checks_its_role_before_touching_the_display(monkeypatch):
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    body = source.split("def main(argv", 1)[1]
    assert body.index("require_role(") < body.index("init_runtime()")


def test_loopback_client_needs_no_tls():
    env = {
        "DESK_DISPLAY_ROLE": "client",
        "DESK_DISPLAY_SERVER_URL": "http://127.0.0.1:8765",
        "DESK_DISPLAY_CLIENT_ID": "desk-panel",
        "DESK_DISPLAY_CLIENT_TOKEN": "server-token-" + "s" * 32,
        "DESK_DISPLAY_PROFILE": "hyperpixel4",
    }
    report = dc.validate(dc.Role.CLIENT, env)
    assert report.ok, [issue.message for issue in report.errors]


# ── Lifecycle of the combined panel ────────────────────────────────────────


pytest.importorskip("flask")

import display_client  # noqa: E402
import display_server  # noqa: E402
from remote_display.client_sync import Backoff  # noqa: E402
from remote_display.playlist_store import PlaylistStore  # noqa: E402
from tests.test_display_client import PROFILE, TOKEN, Clock, FlaskTransport, Presenter  # noqa: E402


class Loopback:
    """The panel's transport to whichever server process is running."""

    def __init__(self):
        self.target = None

    def __call__(self, method, path, **kwargs):
        if self.target is None:
            from remote_display.client_sync import TransportError

            raise TransportError("connection refused")
        return self.target(method, path, **kwargs)


@pytest.fixture
def combined(tmp_path):
    store_path = tmp_path / "playlists.json"
    store = PlaylistStore(store_path)
    playlist = store.create("Desk", {"screens": {"date": 1, "weather1": 1}, "sequence": []}, actor="test")
    store.assign("desk-panel", playlist["id"], expected_playlist_id=None, actor="test")
    clock = Clock()
    # The panel is provisioned like any other client: its own credential.
    from remote_display.provisioning import ProvisioningStore

    clients_path = tmp_path / "provisioned_clients.json"
    panel_token = ProvisioningStore(clients_path).provision("desk-panel", PROFILE.profile_id).credential

    def start_server():
        config = display_server.DisplayServerConfig(
            admin_token="admin-token-" + "a" * 32, lease_seconds=300, clients_path=clients_path,
            artifact_dir=tmp_path / "server-artifacts", playlist_store_path=store_path,
        )
        app = display_server.create_app(config, clock=clock)
        return app, FlaskTransport(app)

    loopback = Loopback()

    def start_client():
        settings = {
            "DESK_DISPLAY_PROFILE": PROFILE.profile_id,
            "DESK_DISPLAY_CLIENT_ID": "desk-panel",
            "DESK_DISPLAY_SERVER_URL": "http://127.0.0.1:8765",
            "DESK_DISPLAY_CLIENT_TOKEN": panel_token,
            "DESK_DISPLAY_CLIENT_CACHE_DIR": str(tmp_path / "client"),
        }
        client = display_client.build_client(settings, presenter=Presenter(), transport=loopback)
        client.sync.backoff = Backoff(rng=random.Random(1))
        return client

    def publish(app):
        from PIL import Image

        from remote_display.models import RenderKey, ScreenRevisions

        for index, screen in enumerate(("date", "weather1")):
            key = RenderKey.for_screen(screen, PROFILE.profile_id, ScreenRevisions("s1", "d1", "r1"))
            app.extensions["desk_display_artifacts"].publish_image(
                key, Image.new(PROFILE.color_mode, (PROFILE.width, PROFILE.height), 10 + index))

    return start_server, start_client, loopback, publish


def _plays(client):
    return client.step()[0] in {"date", "weather1"}


def test_panel_starts_from_cache_before_its_server(combined):
    start_server, start_client, loopback, publish = combined
    app, loopback.target = start_server()
    publish(app)
    client = start_client()
    for _ in range(2):
        client.sync.sync_once()
    assert _plays(client)

    loopback.target = None  # reboot: the client comes up first
    client = start_client()
    assert _plays(client)
    assert client.sync.step() > 0 and not client.sync.connected
    assert _plays(client) and client.report.playback_state == "offline"

    app, loopback.target = start_server()  # the server finishes starting later
    client.sync.step()
    assert client.sync.connected and _plays(client)


def test_server_restart_does_not_blank_the_panel(combined):
    start_server, start_client, loopback, publish = combined
    app, loopback.target = start_server()
    publish(app)
    client = start_client()
    for _ in range(2):
        client.sync.sync_once()
    frames_before = len(client.presenter.frames)

    loopback.target = None  # server stopping
    client.sync.step()
    shown = [client.step()[0] for _ in range(4)]
    assert set(shown) <= {"date", "weather1"}  # never the diagnostic screen
    assert all(frame.getpixel((0, 0))[0] in (10, 11) for frame in client.presenter.frames[frames_before:])

    app, loopback.target = start_server()  # the same store, a new process
    client.sync.step()
    assert client.sync.connected
    assert client.sync.active().playlist is not None and _plays(client)
