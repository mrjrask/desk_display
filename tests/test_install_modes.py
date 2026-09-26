"""Phase 19: installers, dependency sets, upgrades and uninstall scope per mode."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import deployment_config as dc
import install_modes as im
import service_units as su
from service_units import Mode

ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "requirements"

PROVIDER_LIBS = {"yfinance", "pyjwt", "cairosvg"}
WEB_LIBS = {"flask", "waitress"}
GPIO_LIBS = {"adafruit-blinka", "spidev", "smbus", "lgpio", "gpiozero", "rpi.gpio"}
OUTPUTS = ("displayhatmini", "minipitft", "framebuffer", "kernel", "window", "headless")


def names(path) -> set[str]:
    return im.requirement_names(ROOT / path)


def digest_tree(root: Path, paths) -> dict[str, str]:
    result = {}
    for rel in paths:
        path = root / rel
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for file in files:
            if file.exists():
                result[str(file.relative_to(root))] = hashlib.sha256(file.read_bytes()).hexdigest()
    return result


# ── Dependency sets ────────────────────────────────────────────────────────


@pytest.mark.parametrize("output", OUTPUTS)
def test_clients_install_no_provider_or_web_libraries(output):
    installed = names(im.requirements_file("client", output))
    assert {"requests", "pillow", "pytz"} <= installed
    assert not installed & (PROVIDER_LIBS | WEB_LIBS)


def test_servers_install_the_application_but_no_panel_stack():
    installed = names(im.requirements_file("server"))
    assert installed >= names("requirements/base.txt") and {"flask", "yfinance"} <= installed
    assert not installed & (GPIO_LIBS | {"pygame", "displayhatmini"})


@pytest.mark.parametrize("output, drivers", [
    ("displayhatmini", {"displayhatmini", "pygame", *GPIO_LIBS}),
    ("minipitft", {"adafruit-circuitpython-rgb-display", "numpy", *GPIO_LIBS}),
    ("framebuffer", {"numpy", *GPIO_LIBS}),
    ("kernel", {"pygame", "numpy", *GPIO_LIBS}),
    ("window", {"pygame"}),
])
def test_panel_modes_get_their_drivers(output, drivers):
    for mode in ("client", "combined", "standalone"):
        assert drivers <= names(im.requirements_file(mode, output)), mode
    assert names(im.requirements_file("combined", output)) >= names(im.requirements_file("server"))


def test_standalone_dependency_sets_are_unchanged():
    legacy = {
        "displayhatmini": {"displayhatmini", "pygame", *GPIO_LIBS},
        "framebuffer": set(GPIO_LIBS),
        "kernel": {"pygame", *GPIO_LIBS},
        "minipitft": {"adafruit-circuitpython-rgb-display", *GPIO_LIBS},
    }
    base = names("requirements/base.txt")
    for output, extra in legacy.items():
        assert names(f"requirements/{output}.txt") == base | extra


def test_the_client_process_imports_no_provider_or_web_library():
    script = textwrap.dedent("""
        import importlib.abc, sys
        BLOCKED = {"jwt", "yfinance", "cairosvg", "flask", "waitress", "cryptography", "numpy"}
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError(f"client imported {name}")
        sys.meta_path.insert(0, Block())
        import display_client  # noqa: F401
    """)
    env = {**os.environ, "CONFIG_LOAD_DOTENV": "0"}
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]


def test_unknown_outputs_are_refused():
    with pytest.raises(ValueError):
        im.requirements_file("client", "hdmi-please")


# ── Services ───────────────────────────────────────────────────────────────


def written_units(tmp_path, mode, output=None, **kw):
    paths = im.write_units(mode, tmp_path / mode, project_dir="/opt/dd", python="/opt/dd/venv/bin/python",
                           user="pi", output=output, **kw)
    return {p.name: su.parse_unit(p.read_text()) for p in paths}, paths


@pytest.mark.parametrize("mode", [m.value for m in Mode])
def test_units_restart_on_failure_and_are_world_readable(tmp_path, mode):
    units, paths = written_units(tmp_path, mode, "kernel")
    assert set(units) == set(su.services_for(mode))
    for path in paths:
        assert stat.S_IMODE(path.stat().st_mode) == 0o644
    for unit in units.values():
        assert unit["Service"]["Restart"] == ["always"] and unit["Install"]["WantedBy"] == ["multi-user.target"]


def test_panel_hooks_go_only_to_the_unit_that_drives_the_panel(tmp_path):
    units, _ = written_units(tmp_path, "combined", "kernel", environment={"DISPLAY_ROTATION": "90"})
    client, server = units[su.CLIENT_SERVICE], units[su.SERVER_SERVICE]
    assert "graphical.target" in client["Unit"]["After"]
    assert any("prepare_kernel_session_env.sh" in line for line in client["Service"]["ExecStartPre"])
    assert "DISPLAY_ROTATION=90" in client["Service"]["Environment"]
    assert "DESK_DISPLAY_OUTPUT=kernel" in client["Service"]["Environment"]
    assert "ExecStartPre" not in server["Service"] and server["Service"]["Environment"] == ["DESK_DISPLAY_ROLE=server"]
    fb, _ = written_units(tmp_path, "client", "framebuffer")
    service = fb[su.CLIENT_SERVICE]["Service"]
    assert any("framebuffer_service.sh start" in line for line in service["ExecStartPre"])
    assert any("framebuffer_service.sh stop" in line for line in service["ExecStopPost"])


def test_server_units_carry_no_panel_settings(tmp_path):
    units, _ = written_units(tmp_path, "server", "kernel", environment={"DISPLAY_ROTATION": "90"})
    text = json.dumps(units)
    assert "DISPLAY_ROTATION" not in text and "DESK_DISPLAY_OUTPUT" not in text


def test_combined_starts_in_a_cache_friendly_order(tmp_path):
    order = im.start_order("combined")
    assert order.index(su.SERVER_SERVICE) < order.index(su.CLIENT_SERVICE)
    units, _ = written_units(tmp_path, "combined", "kernel")
    client = units[su.CLIENT_SERVICE]["Unit"]
    # The panel never waits on, or stops with, the server: it plays its cache.
    for key in (*su.COUPLING_KEYS, "After"):
        assert su.SERVER_SERVICE not in " ".join(client.get(key, []))


def test_the_marker_records_how_units_were_written(tmp_path):
    im.write_marker(tmp_path, "client", output="kernel", user="kiosk", environment={"DISPLAY_ROTATION": "180"})
    marked = im.read_marker(tmp_path)
    assert marked == im.Installed(Mode.CLIENT, "kernel", "kiosk", (("DISPLAY_ROTATION", "180"),))
    (tmp_path / im.MODE_MARKER).write_text("server\n")
    assert im.read_marker(tmp_path) == im.Installed(Mode.SERVER)


@pytest.mark.parametrize("installed, expected", [
    ((), Mode.STANDALONE),
    ((su.SERVER_SERVICE, su.CONFIG_UI_SERVICE), Mode.SERVER),
    ((su.CLIENT_SERVICE,), Mode.CLIENT),
    ((su.SERVER_SERVICE, su.CLIENT_SERVICE), Mode.COMBINED),
])
def test_mode_is_detected_from_installed_units(tmp_path, installed, expected):
    systemd = tmp_path / "systemd"
    systemd.mkdir()
    for name in installed:
        (systemd / name).write_text("")
    assert im.detect_mode(tmp_path, systemd) is expected
    im.write_marker(tmp_path, "client")
    assert im.detect_mode(tmp_path, systemd) is Mode.CLIENT  # the installer's record wins


# ── Data: upgrades keep it, uninstall backs it up ──────────────────────────


def populate(project: Path, home: Path) -> None:
    files = {
        ".env": "DESK_DISPLAY_ROLE=server\nOWM_API_KEY=owm-" + "k" * 28 + "\n",
        ".env.client": "DESK_DISPLAY_CLIENT_ID=office\nDESK_DISPLAY_CLIENT_TOKEN=ddc_" + "c" * 40 + "\n",
        "screens_config.local.json": "{}",
        ".runtime/server/playlists.json": '{"assignments": {"office": "p1"}}',
        ".runtime/server/provisioned_clients.json": '{"clients": {}}',
        ".runtime/server/clients.json": "{}",
        ".runtime/server/migrations/standalone-1.json": "{}",
        ".runtime/server/backups/upgrade-old/playlists.json": "{}",
        "cache/artifacts/office/a.png": "png",
        "cache/client/manifest.json": "{}",
        "cache/weather.json": "{}",
    }
    for rel, text in files.items():
        path = project / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (home / "keys").mkdir(parents=True)
    (home / "keys" / "AuthKey.p8").write_text("key")


@pytest.mark.parametrize("mode", [m.value for m in Mode])
def test_every_mode_documents_its_data(mode):
    paths = im.preserved_on_upgrade(mode)
    assert (".env.client" in paths) == (mode in ("client", "combined"))
    assert ("cache/client" in paths) == (mode in ("client", "combined"))
    assert (".runtime/server/playlists.json" in paths) == (mode in ("server", "combined"))
    backed_up = {item.path for item in im.uninstall_backups(mode)}
    assert "cache/artifacts" not in backed_up and "cache/client" not in backed_up


def test_server_snapshot_copies_state_privately(tmp_path):
    project, home = tmp_path / "dd", tmp_path / "home"
    populate(project, home)
    dest = im.snapshot("server", project, home=home, now=0)
    copied = {p.name for p in dest.iterdir()}
    assert {".runtime__server__playlists.json", ".runtime__server__provisioned_clients.json", ".env"} <= copied
    assert not any("backups" in name or "migrations" in name for name in copied)
    assert stat.S_IMODE(dest.stat().st_mode) == 0o700
    assert stat.S_IMODE((dest / ".env").stat().st_mode) == 0o600
    assert im.snapshot("client", project) is None
    assert im.snapshot("server", project, home=home, now=0) != dest  # never overwrites a snapshot


@pytest.mark.parametrize("mode", ["client", "server", "combined"])
def test_uninstall_backs_up_exactly_the_documented_data(tmp_path, mode):
    project, home = tmp_path / "dd", tmp_path / "home"
    populate(project, home)
    copied = dict(im.backup_for_uninstall(mode, project, tmp_path / "backup", home=home))
    assert set(copied) == {item.path for item in im.uninstall_backups(mode)
                           if im.resolve(item.path, project, home).exists()}
    names_ = {p.name for p in (tmp_path / "backup").iterdir()}
    if mode == "client":
        assert names_ == {"dot.env.client"}
    else:
        assert {"dot.env", "keys", "dot.runtime__server__playlists.json"} <= names_
        assert not any("artifacts" in n or n == "cache" for n in names_)
    for name in names_ & {"dot.env", "dot.env.client"}:
        assert stat.S_IMODE((tmp_path / "backup" / name).stat().st_mode) == 0o600


# ── Env preparation ────────────────────────────────────────────────────────

STANDALONE_ENV = """# my display
OWM_API_KEY=owm-{key}
WEATHER_LATITUDE=41.9
WEATHER_LONGITUDE=-87.6
HYPERPIXEL_PANEL=hyperpixel4sq
DISPLAY_ROTATION=180
SCREEN_CONFIG_HOST=127.0.0.1
DESK_DISPLAY_SERVER_CLIENTS_PATH={store}
""".replace("{key}", "k" * 28)


def test_client_env_starts_from_the_panel_settings_and_holds_no_credentials(tmp_path):
    (tmp_path / ".env").write_text(STANDALONE_ENV.replace("{store}", str(tmp_path / "store.json")))
    creds = tmp_path / "office.env.client"
    creds.write_text("DESK_DISPLAY_SERVER_URL=https://render.lan:8765\nDESK_DISPLAY_CLIENT_ID=office\n"
                     "DESK_DISPLAY_CLIENT_TOKEN=ddc_" + "c" * 40 + "\n")
    notes = im.prepare_env("client", tmp_path, install_profile="hyperpixel", credentials=creds)
    env = dc.parse_env_file(tmp_path / ".env.client")
    assert env["DESK_DISPLAY_PROFILE"] == "hyperpixel4_square" and env["DISPLAY_ROTATION"] == "180"
    assert "OWM_API_KEY" not in env and "WEATHER_LATITUDE" not in env
    assert dc.validate(dc.Role.CLIENT, env).ok
    assert stat.S_IMODE((tmp_path / ".env.client").stat().st_mode) == 0o600
    assert "c" * 40 not in " ".join(notes)
    assert not (tmp_path / ".runtime" / "install" / "env.client.tmp").exists()


def test_an_existing_client_identity_is_never_replaced(tmp_path):
    (tmp_path / ".env.client").write_text("DESK_DISPLAY_CLIENT_ID=office\n")
    before = (tmp_path / ".env.client").read_bytes()
    assert im.prepare_env("client", tmp_path) == ["kept .env.client (client identity and credential unchanged)"]
    assert (tmp_path / ".env.client").read_bytes() == before


def test_combined_provisions_its_own_panel_and_converts_the_server_env(tmp_path):
    store = tmp_path / "store.json"
    (tmp_path / ".env").write_text(STANDALONE_ENV.replace("{store}", str(store)))
    notes = im.prepare_env("combined", tmp_path, install_profile="hyperpixel", client_id="desk-panel")
    client = dc.parse_env_file(tmp_path / ".env.client")
    assert client["DESK_DISPLAY_SERVER_URL"] == "http://127.0.0.1:8765"
    assert client["DESK_DISPLAY_CLIENT_ID"] == "desk-panel" and client["DESK_DISPLAY_CLIENT_TOKEN"]
    assert dc.validate(dc.Role.CLIENT, client).ok
    from remote_display.provisioning import ProvisioningStore

    assert ProvisioningStore(store).verify("desk-panel", client["DESK_DISPLAY_CLIENT_TOKEN"])
    server = dc.parse_env_file(tmp_path / ".env")
    assert server[dc.ROLE_ENV] == "server" and "DISPLAY_ROTATION" not in server
    assert list(tmp_path.glob(".env.bak-*"))  # the standalone original is kept
    assert client["DESK_DISPLAY_CLIENT_TOKEN"] not in " ".join(notes)
    # Rerunning keeps both files as they are.
    before = {p: (tmp_path / p).read_bytes() for p in (".env", ".env.client")}
    im.prepare_env("combined", tmp_path, install_profile="hyperpixel")
    assert {p: (tmp_path / p).read_bytes() for p in before} == before


def test_combined_without_a_known_profile_asks_for_one(tmp_path):
    (tmp_path / ".env").write_text("DISPLAY_ROTATION=0\n")
    with pytest.raises(ValueError, match="DESK_DISPLAY_PROFILE"):
        im.prepare_env("combined", tmp_path, install_profile="kernel")
    assert not (tmp_path / ".env.client").exists()


# ── Shell workflows ────────────────────────────────────────────────────────

SCRIPTS = ["Installers/install.sh", "Installers/uninstall.sh", "scripts/helpers/base_setup.sh",
           "scripts/upgrade.sh", "scripts/update_dependencies.sh", "scripts/update_services.sh",
           "scripts/cleanup.sh", "scripts/restart_services.sh"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_parse(script):
    assert subprocess.run(["bash", "-n", str(ROOT / script)]).returncode == 0


@pytest.mark.parametrize("args", [["--mode", "bogus"], ["--mode", "server", "--credentials", "x.env"],
                                  ["--mode", "client", "pi_window"]])
def test_installer_refuses_bad_mode_combinations_before_changing_anything(args, tmp_path):
    env = {**os.environ, "PROJECT_DIR": str(tmp_path)}
    result = subprocess.run(["bash", str(ROOT / "Installers/install.sh"), *args], env=env,
                            stdin=subprocess.DEVNULL, capture_output=True, text=True)
    assert result.returncode == 1 and "[ERROR]" in result.stderr
    assert not list(tmp_path.iterdir())


def test_uninstall_and_maintenance_scripts_know_every_service():
    uninstall = (ROOT / "Installers/uninstall.sh").read_text()
    for name in su.ALL_SERVICES:
        assert name in uninstall or name.split(".")[0].upper() in uninstall
    assert "display_client.py" in uninstall and "display_server.py" in uninstall
    for script in ("scripts/update_services.sh", "scripts/restart_services.sh"):
        text = (ROOT / script).read_text()
        assert su.SERVER_SERVICE in text and su.CLIENT_SERVICE in text


def fake_project(tmp_path: Path, mode: str) -> tuple[Path, Path, Path]:
    """A project whose dependency and restart steps only log what they were asked."""

    project = tmp_path / "dd"
    (project / "scripts" / "helpers").mkdir(parents=True)
    for rel in ("install_modes.py", "service_units.py", "scripts/upgrade.sh", "scripts/helpers/common.sh"):
        shutil.copy2(ROOT / rel, project / rel)
    log = tmp_path / "calls.log"
    for rel in ("scripts/update_dependencies.sh", "scripts/restart_services.sh", "scripts/update_services.sh"):
        path = project / rel
        path.write_text(f'#!/usr/bin/env bash\necho "{rel} $* panel=${{DESK_DISPLAY_PANEL_ENV_FILE:-}}" >> {log}\n')
        path.chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "systemctl").write_text(f'#!/usr/bin/env bash\necho "systemctl $*" >> {log}\n')
    (bin_dir / "systemctl").chmod(0o755)
    home = tmp_path / "home"
    populate(project, home)
    im.write_marker(project, mode, output="kernel", user="kiosk", environment={"DISPLAY_ROTATION": "90"})
    return project, log, bin_dir


@pytest.mark.parametrize("mode", ["client", "server", "combined"])
def test_upgrade_preserves_the_documented_data(tmp_path, mode):
    project, log, bin_dir = fake_project(tmp_path, mode)
    preserved = [p for p in im.preserved_on_upgrade(mode) if not p.startswith("~/")]
    before = digest_tree(project, preserved)
    systemd = tmp_path / "systemd"
    systemd.mkdir()
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "SUDO": "", "PYTHON": sys.executable,
           "SYSTEMD_UNIT_DIR": str(systemd), "PROJECT_DIR": str(project), "HOME": str(tmp_path / "home")}
    result = subprocess.run(["bash", str(project / "scripts/upgrade.sh"), "--no-pull"], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    after = digest_tree(project, preserved)
    new = set(after) - set(before)
    assert {k: after[k] for k in before} == before  # nothing existing changed or vanished
    assert all(k.startswith(".runtime/server/backups/upgrade-") for k in new)  # only the snapshot is new
    assert bool(new) == (mode != "client")
    calls = log.read_text()
    assert f"--requirements {im.requirements_file(mode, 'kernel')}" in calls
    assert ("panel=none" in calls) == (mode == "server")
    assert "systemctl daemon-reload" in calls and "restart_services.sh" in calls
    assert {p.name for p in systemd.iterdir()} == set(su.services_for(mode))
    if mode != "server":
        client = su.parse_unit((systemd / su.CLIENT_SERVICE).read_text())["Service"]
        assert client["User"] == ["kiosk"] and "DISPLAY_ROTATION=90" in client["Environment"]


def test_combined_with_shared_enrollment_uses_the_shared_token(tmp_path):
    token = "shared-" + "t" * 40
    (tmp_path / ".env").write_text(STANDALONE_ENV.replace("{store}", str(tmp_path / "store.json"))
                                   + f"DESK_DISPLAY_SERVER_ENROLLMENT=shared\nDESK_DISPLAY_SERVER_AUTH_TOKEN={token}\n")
    notes = im.prepare_env("combined", tmp_path, install_profile="hyperpixel", client_id="desk-panel")
    client = dc.parse_env_file(tmp_path / ".env.client")
    assert client["DESK_DISPLAY_CLIENT_TOKEN"] == token and not (tmp_path / "store.json").exists()
    assert token not in " ".join(notes)
