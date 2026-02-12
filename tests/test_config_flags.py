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


def test_display_rotation_env_is_used_even_when_kernel_overlay_exists(monkeypatch):
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


def test_display_rotation_defaults_to_0_when_env_missing_even_if_overlay_exists(monkeypatch):
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

    assert module.DISPLAY_ROTATION == 0


def test_display_rotation_shorthand_values_expand_to_degrees(monkeypatch):
    module = _reload_config(monkeypatch, DISPLAY_ROTATION="2")
    assert module.DISPLAY_ROTATION == 180


def test_kernel_overlay_rotate_shorthand_is_parsed_for_logging(monkeypatch):
    config_text = "dtoverlay=vc4-kms-dpi-hyperpixel4,rotate=2\n"

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

    assert module._kernel_overlay_rotation == 180
    assert module.DISPLAY_ROTATION == 0


def test_hyperpixel_layout_includes_hd_widescreen_dimensions(monkeypatch):
    module = _reload_config(monkeypatch, DISPLAY_WIDTH="1920", DISPLAY_HEIGHT="1080")

    assert module.is_hyperpixel_next_layout() is True
    assert module.is_hyperpixel_next_layout(1280, 720) is True


def test_hyperpixel_layout_excludes_non_widescreen_hd_dimensions(monkeypatch):
    module = _reload_config(monkeypatch, DISPLAY_WIDTH="1024", DISPLAY_HEIGHT="768")

    assert module.is_hyperpixel_next_layout() is False
    assert module.is_hyperpixel_next_layout(1024, 768) is False
