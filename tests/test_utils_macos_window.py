"""Tests for macOS-friendly window behavior in utils."""

import utils


def test_sdl_driver_candidates_try_default_driver_first(monkeypatch):
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    monkeypatch.delenv("DESK_DISPLAY_SDL_DRIVERS", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    candidates = utils._sdl_driver_candidates()

    assert candidates[0] is None


def test_check_apt_updates_skips_when_apt_get_unavailable(monkeypatch):
    monkeypatch.setattr(utils.shutil, "which", lambda _name: None)
    monkeypatch.setattr(utils, "_set_update_status", lambda **_kwargs: None)

    called = {"run": False}

    def _should_not_run(*_args, **_kwargs):
        called["run"] = True
        raise AssertionError("subprocess.run should not be called when apt-get is unavailable")

    monkeypatch.setattr(utils.subprocess, "run", _should_not_run)

    assert utils.check_apt_updates() is False
    assert called["run"] is False


def test_window_output_failure_does_not_fallback_to_framebuffer(monkeypatch):
    monkeypatch.setattr(utils, "_FORCE_HEADLESS", False)
    monkeypatch.setattr(utils, "_DISPLAY_OUTPUT", "window")
    monkeypatch.setattr(utils, "_normalize_display_output", lambda _value: "window")

    def _raise_kernel_display(*_args, **_kwargs):
        raise RuntimeError("SDL init failed")

    fallback_called = {"called": False}

    def _fallback(*_args, **_kwargs):
        fallback_called["called"] = True
        return object()

    monkeypatch.setattr(utils, "_KernelDisplay", _raise_kernel_display)
    monkeypatch.setattr(utils, "_init_framebuffer_output", _fallback)

    display = utils.Display()

    assert fallback_called["called"] is False
    assert display._output_strategy == "headless"
