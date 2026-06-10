"""Tests for framebuffer stride padding buffer reuse."""

import os

import utils


def test_framebuffer_write_image_reuses_padded_stride_buffer(monkeypatch):
    writes = []

    monkeypatch.setattr(utils, "_resolve_framebuffer_info", lambda _path: (2, 2, 24, 8))
    monkeypatch.setattr(utils, "_resolve_framebuffer_pixel_settings", lambda *_args: ("rgb888", "rgb"))
    monkeypatch.setattr(utils, "_disable_framebuffer_cursor", lambda: None)
    monkeypatch.setattr(os, "open", lambda *_args: 99)
    monkeypatch.setattr(os, "lseek", lambda *_args: 0)

    def _write(_fd, data):
        writes.append((id(data), bytes(data)))
        return len(data)

    monkeypatch.setattr(os, "write", _write)

    device = utils._FrameBufferDevice("/dev/fb-test")
    first = utils.Image.new("RGB", (2, 2), "red")
    second = utils.Image.new("RGB", (2, 2), "blue")

    device.write_image(first)
    device.write_image(second)

    assert writes[0][0] == writes[1][0]
    assert writes[0][1] == b"\xff\x00\x00\xff\x00\x00\x00\x00" * 2
    assert writes[1][1] == b"\x00\x00\xff\x00\x00\xff\x00\x00" * 2
