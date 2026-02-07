import importlib
from pathlib import Path


def _reload_config(monkeypatch, **env):
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    import config  # local import to ensure module exists before reload
    return importlib.reload(config)


def test_enable_screenshots_obeys_env(monkeypatch):
    module = _reload_config(monkeypatch, ENABLE_SCREENSHOTS="0")
    assert module.ENABLE_SCREENSHOTS is False

    module = _reload_config(monkeypatch, ENABLE_SCREENSHOTS="TRUE")
    assert module.ENABLE_SCREENSHOTS is True

    module = _reload_config(monkeypatch, ENABLE_SCREENSHOTS=None)
    assert module.ENABLE_SCREENSHOTS is True


def test_other_feature_flags_use_bool_parser(monkeypatch):
    module = _reload_config(monkeypatch, ENABLE_VIDEO="yes", ENABLE_WIFI_MONITOR="off")
    assert module.ENABLE_VIDEO is True
    assert module.ENABLE_WIFI_MONITOR is False

    module = _reload_config(monkeypatch, ENABLE_VIDEO=None, ENABLE_WIFI_MONITOR=None)
    assert module.ENABLE_VIDEO is False
    assert module.ENABLE_WIFI_MONITOR is True


def test_kernel_portrait_mode_is_normalized_to_landscape(monkeypatch):
    module = _reload_config(
        monkeypatch,
        DISPLAY_WIDTH="480",
        DISPLAY_HEIGHT="800",
        DESK_DISPLAY_OUTPUT="kernel",
        HYPERPIXEL_PANEL=None,
    )

    assert module.WIDTH == 800
    assert module.HEIGHT == 480


def test_display_rotation_defaults_to_0(monkeypatch):
    module = _reload_config(monkeypatch, DISPLAY_ROTATION=None)
    assert module.DISPLAY_ROTATION == 0


def test_display_rotation_env_overrides_kernel_overlay(monkeypatch):
    config_text = "dtoverlay=vc4-kms-dpi-hyperpixel4,rotate=270\n"

    original_read_text = Path.read_text

    def fake_read_text(path_obj, *args, **kwargs):
        if str(path_obj) in {"/boot/firmware/config.txt", "/boot/config.txt"}:
            return config_text
        return original_read_text(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    module = _reload_config(
        monkeypatch,
        DESK_DISPLAY_OUTPUT="kernel",
        HYPERPIXEL_PANEL="hyperpixel4",
        DISPLAY_ROTATION="180",
    )

    assert module.DISPLAY_ROTATION == 180


def test_display_rotation_uses_kernel_overlay_when_env_missing(monkeypatch):
    config_text = "dtoverlay=vc4-kms-dpi-hyperpixel4,rotate=270\n"

    original_read_text = Path.read_text

    def fake_read_text(path_obj, *args, **kwargs):
        if str(path_obj) in {"/boot/firmware/config.txt", "/boot/config.txt"}:
            return config_text
        return original_read_text(path_obj, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    module = _reload_config(
        monkeypatch,
        DESK_DISPLAY_OUTPUT="kernel",
        HYPERPIXEL_PANEL="hyperpixel4",
        DISPLAY_ROTATION=None,
    )

    assert module.DISPLAY_ROTATION == 270
