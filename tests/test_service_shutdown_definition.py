"""Regression coverage for the desk_display systemd shutdown contract."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SETUP = ROOT / "scripts" / "helpers" / "base_setup.sh"
UPDATE_SERVICES = ROOT / "scripts" / "update_services.sh"


def test_generated_service_uses_main_process_graceful_shutdown():
    setup = BASE_SETUP.read_text(encoding="utf-8")
    service = setup.split("<<SERVICE\n", 1)[1].split("\nSERVICE", 1)[0]

    assert "ExecStart=$VENV_DIR/bin/python $PROJECT_DIR/main.py" in service
    assert "TimeoutStopSec=10" in service
    assert "KillSignal=SIGTERM" in service
    assert "ExecStop=" not in service
    assert "Restart=always" in service


def test_service_updater_migrates_existing_cleanup_execstop():
    updater = UPDATE_SERVICES.read_text(encoding="utf-8")

    assert 'unit_name" == "desk_display.service' in updater
    assert "ExecStop=.*cleanup\\.sh" in updater
    assert 'print "TimeoutStopSec=10"' in updater
    assert 'print "KillSignal=SIGTERM"' in updater


def test_service_updater_rewrites_installed_shutdown_settings(tmp_path):
    unit_dir = tmp_path / "systemd"
    unit_dir.mkdir()
    unit_path = unit_dir / "desk_display.service"
    unit_path.write_text(
        """[Unit]
Description=Desk Display

[Service]
ExecStart=/home/pi/desk_display/venv/bin/python /home/pi/desk_display/main.py
ExecStop=/bin/bash -lc '/home/pi/desk_display/scripts/cleanup.sh'
TimeoutStopSec=90
Restart=always

[Install]
WantedBy=multi-user.target
""",
        encoding="utf-8",
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "systemctl").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (fake_bin / "systemctl").chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PROJECT_DIR": str(ROOT),
            "SYSTEMD_UNIT_DIR": str(unit_dir),
        }
    )
    subprocess.run([str(UPDATE_SERVICES)], check=True, env=env, capture_output=True, text=True)

    updated = unit_path.read_text(encoding="utf-8")
    assert "ExecStop=" not in updated
    assert updated.count("TimeoutStopSec=10") == 1
    assert updated.count("KillSignal=SIGTERM") == 1
    assert "Restart=always" in updated
