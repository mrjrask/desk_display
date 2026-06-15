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


def test_detect_exact_framebuffer_device_requires_exact_match(monkeypatch):
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
        lambda device: (1920, 1080) if device == "/dev/fb0" else (320, 240),
    )
    monkeypatch.setattr(utils, "_read_sysfs_value", lambda _path: None)

    detected = utils._detect_exact_framebuffer_device("/dev/fb0", (320, 240))

    assert detected == "/dev/fb1"


def test_detect_exact_framebuffer_device_returns_none_without_match(monkeypatch):
    monkeypatch.setattr(
        utils.Path,
        "glob",
        lambda self, pattern: [Path("/dev/fb0")],
    )
    monkeypatch.setattr(utils.Path, "exists", lambda self: str(self) == "/dev/fb0")
    monkeypatch.setattr(utils, "_read_framebuffer_mode_size", lambda _device: (800, 480))
    monkeypatch.setattr(utils, "_read_sysfs_value", lambda _path: None)

    detected = utils._detect_exact_framebuffer_device("/dev/fb0", (320, 240))

    assert detected is None


def test_detect_framebuffer_device_uses_fbset_when_mode_unavailable(monkeypatch):
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
    monkeypatch.setattr(utils, "_read_framebuffer_mode_size", lambda _device: None)
    monkeypatch.setattr(
        utils,
        "_read_framebuffer_fbset_size",
        lambda device: (320, 240) if device == "/dev/fb0" else None,
    )
    monkeypatch.setattr(utils, "_read_sysfs_value", lambda _path: None)

    detected = utils._detect_framebuffer_device("/dev/fb1", (320, 240))

    assert detected == "/dev/fb0"


def test_detect_exact_framebuffer_device_uses_fbset_when_mode_unavailable(monkeypatch):
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
    monkeypatch.setattr(utils, "_read_framebuffer_mode_size", lambda _device: None)
    monkeypatch.setattr(
        utils,
        "_read_framebuffer_fbset_size",
        lambda device: (320, 240) if device == "/dev/fb0" else None,
    )
    monkeypatch.setattr(utils, "_read_sysfs_value", lambda _path: None)

    detected = utils._detect_exact_framebuffer_device("/dev/fb1", (320, 240))

    assert detected == "/dev/fb0"


def test_init_framebuffer_output_retries_fb0_when_preferred_fails(monkeypatch):
    init_calls = []

    class _FakeFramebuffer:
        def __init__(self, device_path):
            init_calls.append(device_path)
            self.device_path = device_path
            self._fd = None if device_path == "/dev/fb1" else object()
            self.width = 320
            self.height = 240
            self.bpp = 16

    monkeypatch.setattr(utils, "_detect_framebuffer_device", lambda *_args, **_kwargs: "/dev/fb1")
    monkeypatch.setattr(utils, "_FrameBufferDevice", _FakeFramebuffer)

    framebuffer = utils._init_framebuffer_output(
        requested_size=(320, 240),
        configured_device="/dev/fb1",
    )

    assert framebuffer is not None
    assert framebuffer.device_path == "/dev/fb0"
    assert init_calls == ["/dev/fb1", "/dev/fb0"]


def test_init_framebuffer_output_returns_none_when_all_devices_fail(monkeypatch):
    class _FakeFramebuffer:
        def __init__(self, device_path):
            self.device_path = device_path
            self._fd = None
            self.width = 320
            self.height = 240
            self.bpp = 16

    monkeypatch.setattr(utils, "_detect_framebuffer_device", lambda *_args, **_kwargs: "/dev/fb2")
    monkeypatch.setattr(utils, "_FrameBufferDevice", _FakeFramebuffer)

    framebuffer = utils._init_framebuffer_output(
        requested_size=(320, 240),
        configured_device="/dev/fb2",
    )

    assert framebuffer is None


def test_hide_framebuffer_console_cursor_switches_tty_to_graphics(monkeypatch):
    writes = []
    ioctls = []
    closed = []

    monkeypatch.setattr(utils, "_FRAMEBUFFER_HIDE_CONSOLE_CURSOR", True)
    monkeypatch.setattr(utils, "_FRAMEBUFFER_CONSOLE_GRAPHICS", True)
    monkeypatch.setattr(utils, "_open_framebuffer_console", lambda: 42)
    monkeypatch.setattr(utils.os, "write", lambda fd, data: writes.append((fd, data)) or len(data))
    monkeypatch.setattr(utils.fcntl, "ioctl", lambda fd, req, arg: ioctls.append((fd, req, arg)))
    monkeypatch.setattr(utils.os, "close", lambda fd: closed.append(fd))

    tty_fd = utils._hide_framebuffer_console_cursor()
    utils._restore_framebuffer_console_cursor(tty_fd)

    assert tty_fd == 42
    assert writes == [(42, b"\033[?25l"), (42, b"\033[?25h")]
    assert ioctls == [
        (42, utils._KDSETMODE, utils._KD_GRAPHICS),
        (42, utils._KDSETMODE, utils._KD_TEXT),
    ]
    assert closed == [42]


def test_hide_framebuffer_console_cursor_respects_disable_flag(monkeypatch):
    opened = []

    monkeypatch.setattr(utils, "_FRAMEBUFFER_HIDE_CONSOLE_CURSOR", False)
    monkeypatch.setattr(utils, "_open_framebuffer_console", lambda: opened.append(True) or 42)

    assert utils._hide_framebuffer_console_cursor() is None
    assert opened == []
