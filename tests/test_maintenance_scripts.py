"""The operator scripts in scripts/ know about the server/client install modes."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import install_modes as im
import service_units as su

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _script_files() -> list[Path]:
    return sorted(p for p in SCRIPTS.rglob("*") if p.is_file() and p.suffix in {".sh", ".py"}
                  and "__pycache__" not in p.parts)


@pytest.mark.parametrize("path", _script_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_every_script_is_executable_with_a_shebang(path):
    listed = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-s", str(path.relative_to(ROOT))],
                            capture_output=True, text=True, check=True).stdout
    if listed:  # tracked: the mode git checks out on the Pi
        assert listed.startswith("100755"), listed
    assert os.access(path, os.X_OK)
    assert path.read_text(encoding="utf-8").startswith("#!")


# ── update_services.sh ─────────────────────────────────────────────────────


def fake_systemctl(tmp_path: Path, *, enabled: tuple[str, ...] = ()) -> tuple[Path, Path]:
    log = tmp_path / "systemctl.log"
    path = tmp_path / "bin" / "systemctl"
    path.parent.mkdir()
    cases = "".join(f'  "is-enabled {name}") echo enabled; exit 0 ;;\n' for name in enabled)
    path.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$*" >> {log}\n'
        'case "$1 $2" in\n'
        f"{cases}"
        '  is-enabled*) echo disabled; exit 1 ;;\n'
        '  is-active*) echo inactive; exit 3 ;;\n'
        "esac\n"
    )
    path.chmod(0o755)
    return path, log


def project_copy(tmp_path: Path) -> Path:
    project = tmp_path / "dd"
    (project / "scripts" / "helpers").mkdir(parents=True)
    for rel in ("install_modes.py", "service_units.py", "deployment_config.py", "scripts/update_services.sh",
                "scripts/helpers/common.sh", "scripts/cleanup.sh", "main.py", "config_ui.py",
                "display_client.py", "display_server.py"):
        shutil.copy2(ROOT / rel, project / rel)
    return project


def run_update_services(project: Path, systemd: Path, systemctl: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "SUDO": "", "SYSTEMCTL": str(systemctl), "SYSTEMD_UNIT_DIR": str(systemd),
           "PROJECT_DIR": str(project), "PYTHON": sys.executable}
    result = subprocess.run(["bash", str(project / "scripts/update_services.sh"), *args], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


OLD_CLIENT_UNIT = """[Unit]
Description=Desk Display - display client

[Service]
Environment=DESK_DISPLAY_ROLE=client
Environment=DISPLAY_ROTATION=180
Environment=DESK_DISPLAY_OUTPUT=framebuffer
ExecStart=/opt/dd/venv/bin/python /opt/dd/display_client.py
User=kiosk
"""


def test_update_services_brings_an_unrecorded_combined_install_up_to_date(tmp_path):
    project = project_copy(tmp_path)
    systemd = tmp_path / "systemd"
    systemd.mkdir()
    (systemd / su.CLIENT_SERVICE).write_text(OLD_CLIENT_UNIT)
    (systemd / su.SERVER_SERVICE).write_text(OLD_CLIENT_UNIT.replace("client", "server"))
    (systemd / su.STANDALONE_SERVICE).write_text("[Service]\nExecStart=/opt/dd/main.py\n")
    systemctl, log = fake_systemctl(tmp_path, enabled=(su.STANDALONE_SERVICE,))

    result = run_update_services(project, systemd, systemctl)

    assert "Installed mode: combined" in result.stdout
    for name in su.services_for("combined"):
        assert (systemd / name).exists(), name
    client = su.parse_unit((systemd / su.CLIENT_SERVICE).read_text())
    assert client["Service"]["User"] == ["kiosk"]
    assert "DISPLAY_ROTATION=180" in client["Service"]["Environment"]
    assert "DESK_DISPLAY_OUTPUT=framebuffer" in client["Service"]["Environment"]
    assert client["Unit"]["Conflicts"] == [su.STANDALONE_SERVICE]
    assert client["Service"]["ExecStart"] == [f"{project}/venv/bin/python {project}/display_client.py"]
    marker = im.read_marker(project)
    assert marker is not None and marker.mode is im.Mode.COMBINED and marker.user == "kiosk"
    assert marker.output == "framebuffer" and dict(marker.environment) == {"DISPLAY_ROTATION": "180"}
    calls = log.read_text()
    assert f"disable --now {su.STANDALONE_SERVICE}" in calls  # it drives the same panel
    assert "daemon-reload" in calls
    assert f"restart {su.CLIENT_SERVICE}" in calls and f"restart {su.STANDALONE_SERVICE}" not in calls
    assert "Project units on this device:" in result.stdout

    # A second run finds nothing to change.
    log.write_text("")
    again = run_update_services(project, systemd, systemctl)
    assert "Every installed unit is already current." in again.stdout
    assert "daemon-reload" not in log.read_text() and "restart" not in log.read_text()


def test_update_services_rewrites_a_recorded_client_install(tmp_path):
    project = project_copy(tmp_path)
    im.write_marker(project, "client", output="kernel", user="pi", environment={"DISPLAY_ROTATION": "90"})
    systemd = tmp_path / "systemd"
    systemd.mkdir()
    (systemd / su.CLIENT_SERVICE).write_text(OLD_CLIENT_UNIT)
    systemctl, log = fake_systemctl(tmp_path)

    run_update_services(project, systemd, systemctl, "--no-restart")

    assert {p.name for p in systemd.iterdir()} == {su.CLIENT_SERVICE}
    client = su.parse_unit((systemd / su.CLIENT_SERVICE).read_text())["Service"]
    assert client["User"] == ["pi"] and "DISPLAY_ROTATION=90" in client["Environment"]
    assert any("prepare_kernel_session_env.sh" in line for line in client["ExecStartPre"])
    calls = log.read_text()
    assert f"enable {su.CLIENT_SERVICE}" in calls and "restart" not in calls


STANDALONE_UNIT = """[Unit]
Description=Desk Display Service - main
After=network-online.target

[Service]
Environment=DISPLAY_ROTATION=270
ExecStart=/home/pi/desk_display/venv/bin/python /home/pi/desk_display/main.py
ExecStop=/bin/bash -lc '/home/pi/desk_display/tools/maintenance/cleanup.sh'
TimeoutStopSec=90
User=kiosk

[Install]
WantedBy=multi-user.target
"""


def test_update_services_patches_a_standalone_unit_in_place(tmp_path):
    project = project_copy(tmp_path)
    systemd = tmp_path / "systemd"
    systemd.mkdir()
    (systemd / su.STANDALONE_SERVICE).write_text(STANDALONE_UNIT)
    systemctl, log = fake_systemctl(tmp_path)

    dry = run_update_services(project, systemd, systemctl, "--dry-run")
    assert "[dry-run]" in dry.stdout
    assert (systemd / su.STANDALONE_SERVICE).read_text() == STANDALONE_UNIT
    assert not (systemd / su.CONFIG_UI_SERVICE).exists()
    queries = log.read_text().splitlines() if log.exists() else []
    assert all(line.split()[0] in ("is-enabled", "is-active") for line in queries)  # read-only

    run_update_services(project, systemd, systemctl)
    unit = su.parse_unit((systemd / su.STANDALONE_SERVICE).read_text())
    assert unit["Unit"]["Conflicts"] == [su.CLIENT_SERVICE]
    assert unit["Service"]["Environment"] == ["DISPLAY_ROTATION=270"]  # the profile is left alone
    assert unit["Service"]["User"] == ["kiosk"] and "ExecStop" not in unit["Service"]
    assert unit["Service"]["TimeoutStopSec"] == ["10"]
    config_ui = su.parse_unit((systemd / su.CONFIG_UI_SERVICE).read_text())["Service"]
    assert config_ui["User"] == ["kiosk"]
    assert not (project / im.MODE_MARKER).exists()


def test_upgrade_delegates_units_to_update_services():
    upgrade = (SCRIPTS / "upgrade.sh").read_text()
    assert 'update_services.sh" --mode "$mode" --no-restart' in upgrade
    assert "modes units" not in upgrade


# ── update_dependencies.sh ─────────────────────────────────────────────────


@pytest.mark.parametrize("mode", [m.value for m in im.Mode])
def test_update_dependencies_installs_the_recorded_modes_requirements(tmp_path, mode):
    project = tmp_path / "dd"
    (project / "scripts" / "helpers").mkdir(parents=True)
    for rel in ("install_modes.py", "service_units.py", "deployment_config.py",
                "scripts/update_dependencies.sh", "scripts/helpers/common.sh"):
        shutil.copy2(ROOT / rel, project / rel)
    im.write_marker(project, mode, output=None if mode == "server" else "kernel")
    env = {k: v for k, v in os.environ.items() if k not in ("DESK_DISPLAY_OUTPUT", "DESK_DISPLAY_INSTALL_MODE")}
    result = subprocess.run(["bash", str(project / "scripts/update_dependencies.sh"), "--print-requirements"],
                            env={**env, "PROJECT_DIR": str(project)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == im.requirements_file(mode, "kernel")
    assert not (project / "venv").exists()


# ── reset_screenshots.sh ───────────────────────────────────────────────────


def reset_project(tmp_path: Path) -> Path:
    project = tmp_path / "dd"
    (project / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPTS / "reset_screenshots.sh", project / "scripts")
    return project


def run_reset(project: Path, **extra: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("SCREENSHOT_DIR", "SCREENSHOT_ARCHIVE_BASE")}
    return subprocess.run(["bash", str(project / "scripts/reset_screenshots.sh")], env={**env, **extra},
                          capture_output=True, text=True)


def test_reset_screenshots_follows_the_configured_folders(tmp_path):
    project = reset_project(tmp_path)
    (project / ".env").write_text('SCREENSHOT_DIR="./shots"  # moved\n')
    for rel in ("shots/current/a.png", "screenshot_archive/x/b.png", "screenshots/keep.png"):
        (project / rel).parent.mkdir(parents=True, exist_ok=True)
        (project / rel).write_text("x")

    result = run_reset(project)

    assert result.returncode == 0, result.stdout + result.stderr
    assert list((project / "shots").iterdir()) == [] and list((project / "screenshot_archive").iterdir()) == []
    assert (project / "screenshots/keep.png").exists()  # not the configured folder any more


def test_reset_screenshots_refuses_a_folder_outside_the_project_before_clearing_anything(tmp_path):
    project = reset_project(tmp_path)
    (project / "screenshots").mkdir()
    (project / "screenshots/a.png").write_text("x")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "b.png").write_text("x")

    result = run_reset(project, SCREENSHOT_ARCHIVE_BASE=str(outside))

    assert result.returncode == 1 and "outside project root" in result.stdout
    assert (project / "screenshots/a.png").exists() and (outside / "b.png").exists()


# ── validate_required_files.py and the panel-service helpers ───────────────


def test_validate_required_files_passes_on_the_repository():
    result = subprocess.run([sys.executable, str(SCRIPTS / "validate_required_files.py")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout
    assert "display_server" in result.stdout and "display_client" in result.stdout


def test_optional_modules_still_exist():
    from scripts import validate_required_files

    for name in validate_required_files.OPTIONAL_MODULES:
        assert (ROOT / (name.replace(".", "/") + ".py")).exists(), name


@pytest.mark.parametrize("script", ["check_hyperpixel_setup.sh", "check_waveshare_setup.sh"])
def test_setup_checks_look_at_the_modes_panel_service(script):
    text = (SCRIPTS / script).read_text()
    assert "install_modes.py\" detect" in text and su.CLIENT_SERVICE in text
    assert 'status "$PANEL_SERVICE"' in text and ".env.client" in text


def test_led_check_warns_about_either_panel_service():
    from scripts import test_led

    assert set(test_led.PANEL_SERVICES) == set(su.PANEL_SERVICES)


def test_update_services_moves_a_converted_standalone_install_onto_the_server_units(tmp_path):
    """After scripts/convert_env.py --role server, main.py refuses to start; the units must follow."""

    project = project_copy(tmp_path)
    (project / ".env").write_text("DESK_DISPLAY_ROLE=server\n")
    systemd = tmp_path / "systemd"
    systemd.mkdir()
    (systemd / su.STANDALONE_SERVICE).write_text(STANDALONE_UNIT)
    systemctl, log = fake_systemctl(tmp_path, enabled=(su.STANDALONE_SERVICE,))

    result = run_update_services(project, systemd, systemctl)

    assert "Installed mode: server" in result.stdout
    assert (systemd / su.SERVER_SERVICE).exists() and (systemd / su.CONFIG_UI_SERVICE).exists()
    server = su.parse_unit((systemd / su.SERVER_SERVICE).read_text())["Service"]
    assert server["User"] == ["kiosk"]
    assert server["ExecStart"] == [f"{project}/venv/bin/python {project}/display_server.py"]
    calls = log.read_text()
    assert f"disable --now {su.STANDALONE_SERVICE}" in calls
    assert f"enable {su.SERVER_SERVICE}" in calls and f"restart {su.SERVER_SERVICE}" in calls
    assert f"restart {su.STANDALONE_SERVICE}" not in calls
    marker = im.read_marker(project)
    assert marker is not None and marker.mode is im.Mode.SERVER and marker.user == "kiosk"


def test_uninstall_only_stops_services_this_device_has():
    uninstall = (ROOT / "Installers/uninstall.sh").read_text()
    loop = uninstall.split('for managed_service in "${MANAGED_SYSTEM_SERVICES[@]}"; do', 1)[1].split("done", 1)[0]
    assert loop.index("list-unit-files") < loop.index("systemctl stop")
