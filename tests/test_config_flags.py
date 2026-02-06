import importlib


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


def test_hyperpixel_overlay_rotation_from_boot_config(monkeypatch):
    def fake_read_text(path_obj, encoding="utf-8"):
        if str(path_obj) == "/boot/firmware/config.txt":
            return "dtoverlay=vc4-kms-dpi-hyperpixel4,rotate=90\n"
        raise OSError

    monkeypatch.setattr("pathlib.Path.read_text", fake_read_text)

    module = _reload_config(
        monkeypatch,
        DISPLAY_ROTATION="180",
        DESK_DISPLAY_OUTPUT="kernel",
        HYPERPIXEL_PANEL="hyperpixel4",
    )

    assert module.DISPLAY_ROTATION == 90


def test_display_rotation_falls_back_to_env_without_overlay(monkeypatch):
    def fake_read_text(path_obj, encoding="utf-8"):
        raise OSError

    monkeypatch.setattr("pathlib.Path.read_text", fake_read_text)

    module = _reload_config(
        monkeypatch,
        DISPLAY_ROTATION="180",
        DESK_DISPLAY_OUTPUT="kernel",
        HYPERPIXEL_PANEL="hyperpixel4",
    )

    assert module.DISPLAY_ROTATION == 180
