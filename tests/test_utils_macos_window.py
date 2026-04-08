"""Tests for macOS-friendly window behavior in utils."""

from types import SimpleNamespace

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


class _FakeSurface:
    def __init__(self, size):
        self._size = size

    def get_size(self):
        return self._size

    def blit(self, _surface, _coords):
        return None


class _FakePygameDisplay:
    def __init__(self):
        self.last_set_mode = None

    def quit(self):
        return None

    def init(self):
        return None

    def set_mode(self, size, flags):
        self.last_set_mode = (size, flags)
        return _FakeSurface(size)

    def set_caption(self, _caption):
        return None

    def get_driver(self):
        return "fake"

    def flip(self):
        return None


class _FakePygame:
    FULLSCREEN = 1
    SCALED = 2
    RESIZABLE = 4

    def __init__(self):
        self.display = _FakePygameDisplay()
        self.mouse = SimpleNamespace(set_visible=lambda _visible: None)
        self.event = SimpleNamespace(pump=lambda: None)
        self.transform = SimpleNamespace(smoothscale=lambda surface, _size: surface)
        self.image = SimpleNamespace(frombuffer=lambda _bytes, size, _mode: _FakeSurface(size))


def test_window_mode_defaults_to_unscaled_render_size(monkeypatch):
    fake_pygame = _FakePygame()
    monkeypatch.setattr(utils, "_load_pygame", lambda: fake_pygame)
    monkeypatch.setattr(utils, "_sdl_driver_candidates", lambda: [None])
    monkeypatch.setattr(utils, "_maybe_configure_desktop_env", lambda: None)
    monkeypatch.setattr(utils, "_park_mouse_cursor", lambda _pygame: None)
    monkeypatch.setattr(utils, "_wiggle_mouse_cursor", lambda _pygame: None)
    monkeypatch.setattr(utils, "_schedule_mouse_cursor_wiggle", lambda *_args, **_kwargs: None)
    monkeypatch.delenv("DESK_DISPLAY_WINDOW_SCALE", raising=False)
    monkeypatch.delenv("DESK_DISPLAY_SDL_FULLSCREEN", raising=False)

    utils._KernelDisplay(800, 480, window_mode=True)

    assert fake_pygame.display.last_set_mode[0] == (800, 480)


def test_window_mode_scales_to_resized_display_surface(monkeypatch):
    fake_pygame = _FakePygame()
    monkeypatch.setattr(utils, "_load_pygame", lambda: fake_pygame)
    monkeypatch.setattr(utils, "_sdl_driver_candidates", lambda: [None])
    monkeypatch.setattr(utils, "_maybe_configure_desktop_env", lambda: None)
    monkeypatch.setattr(utils, "_park_mouse_cursor", lambda _pygame: None)
    monkeypatch.setattr(utils, "_wiggle_mouse_cursor", lambda _pygame: None)
    monkeypatch.setattr(utils, "_schedule_mouse_cursor_wiggle", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("DESK_DISPLAY_WINDOW_SCALE", "1")
    monkeypatch.setenv("DESK_DISPLAY_SDL_FULLSCREEN", "0")

    display = utils._KernelDisplay(800, 480, window_mode=True)
    display._screen = _FakeSurface((1440, 900))
    display.screen_width, display.screen_height = display._screen.get_size()
    display._scale_to_screen = True

    called = {"size": None}

    def _smoothscale(_surface, size):
        called["size"] = size
        return _FakeSurface(size)

    fake_pygame.transform = SimpleNamespace(smoothscale=_smoothscale)

    display.write_image(utils.Image.new("RGB", (800, 480), "black"))

    assert called["size"] == (1440, 900)


def test_window_mode_scale_env_expands_requested_window_size(monkeypatch):
    fake_pygame = _FakePygame()
    monkeypatch.setattr(utils, "_load_pygame", lambda: fake_pygame)
    monkeypatch.setattr(utils, "_sdl_driver_candidates", lambda: [None])
    monkeypatch.setattr(utils, "_maybe_configure_desktop_env", lambda: None)
    monkeypatch.setattr(utils, "_park_mouse_cursor", lambda _pygame: None)
    monkeypatch.setattr(utils, "_wiggle_mouse_cursor", lambda _pygame: None)
    monkeypatch.setattr(utils, "_schedule_mouse_cursor_wiggle", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("DESK_DISPLAY_WINDOW_SCALE", "2")
    monkeypatch.setenv("DESK_DISPLAY_SDL_FULLSCREEN", "0")

    utils._KernelDisplay(800, 480, window_mode=True)

    assert fake_pygame.display.last_set_mode[0] == (1600, 960)


def test_kernel_display_skips_worker_thread_render_on_macos(monkeypatch):
    fake_pygame = _FakePygame()
    monkeypatch.setattr(utils, "_load_pygame", lambda: fake_pygame)
    monkeypatch.setattr(utils, "_sdl_driver_candidates", lambda: [None])
    monkeypatch.setattr(utils, "_maybe_configure_desktop_env", lambda: None)
    monkeypatch.setattr(utils, "_park_mouse_cursor", lambda _pygame: None)
    monkeypatch.setattr(utils, "_wiggle_mouse_cursor", lambda _pygame: None)
    monkeypatch.setattr(utils, "_schedule_mouse_cursor_wiggle", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("DESK_DISPLAY_SDL_FULLSCREEN", "0")
    monkeypatch.setattr(utils.sys, "platform", "darwin")

    display = utils._KernelDisplay(800, 480, window_mode=True)

    main_thread = object()
    worker_thread = object()
    monkeypatch.setattr(utils.threading, "main_thread", lambda: main_thread)
    monkeypatch.setattr(utils.threading, "current_thread", lambda: worker_thread)

    called = {"frombuffer": False, "flip": False, "pump": False}

    fake_pygame.image = SimpleNamespace(
        frombuffer=lambda *_args, **_kwargs: called.__setitem__("frombuffer", True)
    )
    fake_pygame.display.flip = lambda: called.__setitem__("flip", True)
    fake_pygame.event.pump = lambda: called.__setitem__("pump", True)

    display.write_image(utils.Image.new("RGB", (800, 480), "black"))

    assert called == {"frombuffer": False, "flip": False, "pump": False}
