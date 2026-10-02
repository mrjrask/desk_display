"""Tests for scripts/wait_for_display_ready.sh and the Wayland socket lookup."""

from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path

import pytest

import utils

REPO = Path(__file__).resolve().parents[1]
WAIT_SCRIPT = REPO / "scripts" / "wait_for_display_ready.sh"
COMMON_SCRIPT = REPO / "scripts" / "helpers" / "common.sh"


def _run_wait(drm_root: Path, timeout: int = 2) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(
        DESK_DISPLAY_DRM_SYSFS_ROOT=str(drm_root),
        DESK_DISPLAY_STARTUP_TIMEOUT_SECONDS=str(timeout),
        DESK_DISPLAY_STARTUP_POLL_INTERVAL_SECONDS="1",
        DESK_DISPLAY_STARTUP_STABLE_POLLS="2",
    )
    env.pop("WAYLAND_DISPLAY", None)
    env.pop("DISPLAY", None)
    return subprocess.run(
        ["bash", str(WAIT_SCRIPT)], capture_output=True, text=True, env=env, timeout=30
    )


def _connector(root: Path, name: str, status: str, modes: str = "") -> None:
    path = root / name
    path.mkdir(parents=True)
    (path / "status").write_text(status + "\n")
    (path / "modes").write_text(modes)


def test_no_connected_connector_times_out_without_unbound_variable(tmp_path):
    # Pi 5 layout: card0 is v3d (no connectors), HDMI lives on card1.
    _connector(tmp_path, "card1-HDMI-A-1", "disconnected")
    _connector(tmp_path, "card1-HDMI-A-2", "disconnected")

    result = _run_wait(tmp_path)

    assert result.returncode == 0
    assert "unbound variable" not in result.stderr
    assert "no connected mode detected" in result.stderr
    assert "DRM connector card1-HDMI-A-1: status=disconnected modes=0" in result.stderr


def test_missing_drm_connectors_names_kms_overlay(tmp_path):
    result = _run_wait(tmp_path)

    assert result.returncode == 0
    assert "unbound variable" not in result.stderr
    assert "vc4-kms-v3d" in result.stderr


def test_pi5_hdmi_connector_on_card1_is_detected(tmp_path):
    _connector(tmp_path, "card1-HDMI-A-1", "connected", "1920x1080\n1280x720\n")
    _connector(tmp_path, "card1-HDMI-A-2", "disconnected")

    result = _run_wait(tmp_path, timeout=10)

    assert result.returncode == 0
    assert "Display ready" in result.stderr
    assert "card1-HDMI-A-1:1920x1080" in result.stderr


@pytest.fixture
def wayland_runtime(tmp_path):
    # AF_UNIX paths are short; keep the socket dir near the filesystem root.
    runtime = Path(os.environ.get("TMPDIR", "/tmp")) / f"wl-{os.getpid()}"
    runtime.mkdir(exist_ok=True)
    (runtime / "wayland-1.lock").write_text("")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(runtime / "wayland-1"))
    try:
        yield runtime
    finally:
        sock.close()
        for child in runtime.iterdir():
            child.unlink()
        runtime.rmdir()


def test_find_wayland_socket_accepts_wayland_1(wayland_runtime):
    assert utils._find_wayland_socket(str(wayland_runtime)) == "wayland-1"


def test_find_wayland_socket_none_when_missing(tmp_path):
    assert utils._find_wayland_socket(str(tmp_path)) is None


def test_shell_find_wayland_socket_accepts_wayland_1(wayland_runtime):
    result = subprocess.run(
        ["bash", "-c", f'source "{COMMON_SCRIPT}"; find_wayland_socket "$1"', "_", str(wayland_runtime)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "wayland-1"
