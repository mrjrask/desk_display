"""Tests for framebuffer device auto-selection in runtime display startup."""

from pathlib import Path

import utils


def test_detect_framebuffer_device_prefers_matching_mode(monkeypatch):
    monkeypatch.setattr(
        utils.Path,
        "glob",
        lambda self, pattern: [Path("/dev/fb0"), Path("/dev/fb1")],
    )
    monkeypatch.setattr(
        utils.Path,
        "exists",
        lambda self: str(self) in {"/dev/fb0", "/dev/fb1"},
    )
    monkeypatch.setattr(
        utils,
        "_read_framebuffer_mode_size",
        lambda device: (1024, 600) if device == "/dev/fb0" else (800, 480),
    )
    monkeypatch.setattr(utils, "_read_sysfs_value", lambda _path: None)

    detected = utils._detect_framebuffer_device("/dev/fb9", (800, 480))

    assert detected == "/dev/fb1"


def test_detect_framebuffer_device_falls_back_to_fb0_when_no_devices(monkeypatch):
    monkeypatch.setattr(utils.Path, "glob", lambda self, pattern: [])
    monkeypatch.setattr(utils.Path, "exists", lambda self: False)
    monkeypatch.setattr(utils, "_read_framebuffer_mode_size", lambda _device: None)
    monkeypatch.setattr(utils, "_read_sysfs_value", lambda _path: None)

    detected = utils._detect_framebuffer_device("/dev/fb7", (800, 480))

    assert detected == "/dev/fb0"
