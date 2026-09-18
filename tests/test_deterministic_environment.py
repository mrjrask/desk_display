import importlib

from conftest import DETERMINISTIC_ENV, HOST_ONLY_ENV, establish_deterministic_environment

import config


def test_hostile_host_display_values_are_replaced_before_imports():
    environ = {
        "CONFIG_LOAD_DOTENV": "1",
        "DESK_DISPLAY_OUTPUT": "kernel",
        "DISPLAY": ":99",
        "DISPLAY_FB_DEVICE": "/dev/host-framebuffer",
        "DISPLAY_HEIGHT": "720",
        "DISPLAY_ROTATION": "270",
        "DISPLAY_WIDTH": "1280",
        "HYPERPIXEL_PANEL": "hyperpixel4",
        "SDL_VIDEODRIVER": "wayland",
        "WAYLAND_DISPLAY": "wayland-host",
    }

    establish_deterministic_environment(environ)

    assert all(environ[name] == value for name, value in DETERMINISTIC_ENV.items())
    assert HOST_ONLY_ENV.isdisjoint(environ)
    assert (config.WIDTH, config.HEIGHT, config.DISPLAY_ROTATION) == (320, 240, 0)
    assert config.get_display_profile_id() == "display_hat_mini"


def test_explicit_display_profile_simulation_can_override_defaults(monkeypatch):
    with monkeypatch.context() as platform_environment:
        platform_environment.setenv("DESK_DISPLAY_OUTPUT", "kernel")
        platform_environment.setenv("DISPLAY_WIDTH", "800")
        platform_environment.setenv("DISPLAY_HEIGHT", "480")
        platform_environment.setenv("HYPERPIXEL_PANEL", "hyperpixel4")

        module = importlib.reload(config)

        assert module.WIDTH == 800
        assert module.HEIGHT == 480
        assert module.get_display_profile_id() == "hyperpixel4"

    module = importlib.reload(config)
    assert (module.WIDTH, module.HEIGHT) == (320, 240)
