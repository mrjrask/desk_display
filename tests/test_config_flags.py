import importlib
import platform
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


def test_enable_screenshots_defaults_off_for_macos_window_output(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    module = _reload_config(
        monkeypatch,
        ENABLE_SCREENSHOTS=None,
        DESK_DISPLAY_OUTPUT="window",
    )
    assert module.ENABLE_SCREENSHOTS is False


def test_enable_screenshots_default_unchanged_for_non_macos(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    module = _reload_config(
        monkeypatch,
        ENABLE_SCREENSHOTS=None,
        DESK_DISPLAY_OUTPUT="window",
    )
    assert module.ENABLE_SCREENSHOTS is True


def test_enable_screenshots_explicit_env_override_wins_for_macos_window(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    module = _reload_config(
        monkeypatch,
        ENABLE_SCREENSHOTS="1",
        DESK_DISPLAY_OUTPUT="window",
    )
    assert module.ENABLE_SCREENSHOTS is True

    module = _reload_config(
        monkeypatch,
        ENABLE_SCREENSHOTS="0",
        DESK_DISPLAY_OUTPUT="window",
    )
    assert module.ENABLE_SCREENSHOTS is False


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


def test_display_rotation_env_is_zeroed_when_strict_and_kernel_overlay_exists(monkeypatch):
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
    module.initialise_runtime_probes()

    assert module.DISPLAY_ROTATION == 0


def test_display_rotation_env_is_used_when_strict_disabled_and_overlay_exists(monkeypatch):
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
        DISPLAY_ROTATION_STRICT="0",
    )

    assert module.DISPLAY_ROTATION == 180


def test_display_rotation_strict_defaults_to_kernel_mode(monkeypatch):
    module = _reload_config(monkeypatch, DESK_DISPLAY_OUTPUT="kernel", DISPLAY_ROTATION_STRICT=None)
    assert module.DISPLAY_ROTATION_STRICT is True

    module = _reload_config(monkeypatch, DESK_DISPLAY_OUTPUT="headless", DISPLAY_ROTATION_STRICT=None)
    assert module.DISPLAY_ROTATION_STRICT is False


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
    module.initialise_runtime_probes()

    assert module._kernel_overlay_rotation == 180
    assert module.DISPLAY_ROTATION == 0


def test_display_hat_mini_reinit_seconds_default_and_env(monkeypatch):
    module = _reload_config(monkeypatch, DISPLAY_HAT_MINI_REINIT_SECONDS=None)
    assert module.DISPLAY_HAT_MINI_REINIT_SECONDS == 1800

    module = _reload_config(monkeypatch, DISPLAY_HAT_MINI_REINIT_SECONDS="600")
    assert module.DISPLAY_HAT_MINI_REINIT_SECONDS == 600


def test_display_hat_mini_reinit_seconds_invalid_and_negative(monkeypatch):
    module = _reload_config(monkeypatch, DISPLAY_HAT_MINI_REINIT_SECONDS="bad")
    assert module.DISPLAY_HAT_MINI_REINIT_SECONDS == 1800

    module = _reload_config(monkeypatch, DISPLAY_HAT_MINI_REINIT_SECONDS="-5")
    assert module.DISPLAY_HAT_MINI_REINIT_SECONDS == 0


def test_scoreboard_scroll_step_doubles_only_on_1080p(monkeypatch):
    module = _reload_config(monkeypatch, DISPLAY_WIDTH="1920", DISPLAY_HEIGHT="1080")
    assert module.SCOREBOARD_SCROLL_STEP == 2

    module = _reload_config(monkeypatch, DISPLAY_WIDTH="1080", DISPLAY_HEIGHT="1920")
    assert module.SCOREBOARD_SCROLL_STEP == 2

    module = _reload_config(monkeypatch, DISPLAY_WIDTH="2560", DISPLAY_HEIGHT="1440")
    assert module.SCOREBOARD_SCROLL_STEP == 1

    module = _reload_config(monkeypatch, DISPLAY_WIDTH="1280", DISPLAY_HEIGHT="720")
    assert module.SCOREBOARD_SCROLL_STEP == 1

    module = _reload_config(monkeypatch, DISPLAY_WIDTH="320", DISPLAY_HEIGHT="240")
    assert module.SCOREBOARD_SCROLL_STEP == 1


def test_display_profile_id_resolution(monkeypatch):
    module = _reload_config(monkeypatch, DISPLAY_WIDTH="320", DISPLAY_HEIGHT="240")
    assert module.get_display_profile_id() == "display_hat_mini"

    module = _reload_config(monkeypatch, DISPLAY_WIDTH="240", DISPLAY_HEIGHT="135")
    assert module.get_display_profile_id() == "adafruit_minipitft_114"

    module = _reload_config(monkeypatch, DISPLAY_WIDTH="800", DISPLAY_HEIGHT="480")
    assert module.get_display_profile_id() == "hyperpixel4"

    module = _reload_config(monkeypatch, DISPLAY_WIDTH="720", DISPLAY_HEIGHT="720")
    assert module.get_display_profile_id() == "hyperpixel4_square"

    module = _reload_config(monkeypatch, DISPLAY_WIDTH="1080", DISPLAY_HEIGHT="1920")
    assert module.get_display_profile_id() == "hdmi_1080p"


def test_display_profile_presets_drive_scroll_defaults(monkeypatch):
    module = _reload_config(monkeypatch, DISPLAY_WIDTH="1920", DISPLAY_HEIGHT="1080")
    assert module.SCOREBOARD_SCROLL_STEP == module.ACTIVE_DISPLAY_PROFILE.scoreboard_scroll_step

    module = _reload_config(monkeypatch, DISPLAY_WIDTH="1280", DISPLAY_HEIGHT="720")
    assert module.SCOREBOARD_SCROLL_STEP == module.ACTIVE_DISPLAY_PROFILE.scoreboard_scroll_step


def test_hyperpixel_fade_steps_default_to_zero(monkeypatch):
    module = _reload_config(
        monkeypatch,
        DISPLAY_WIDTH="800",
        DISPLAY_HEIGHT="480",
        DISPLAY_FADE_IN_HYPERPIXEL_STEPS=None,
    )

    assert module.get_display_profile_id() == "hyperpixel4"
    assert module.DISPLAY_FADE_IN_HYPERPIXEL_STEPS == 0
    assert module.DISPLAY_FADE_IN_STEPS_BY_PROFILE["hyperpixel4"] == 0


def test_hyperpixel_fade_steps_can_be_overridden(monkeypatch):
    module = _reload_config(
        monkeypatch,
        DISPLAY_WIDTH="800",
        DISPLAY_HEIGHT="480",
        DISPLAY_FADE_IN_HYPERPIXEL_STEPS="4",
    )

    assert module.DISPLAY_FADE_IN_HYPERPIXEL_STEPS == 4
    assert module.DISPLAY_FADE_IN_STEPS_BY_PROFILE["hyperpixel4"] == 4


def test_load_env_file_strips_inline_comments_for_unquoted_values(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "INSIDE_SENSOR=pimoroni_bme68x # Pimoroni BME688\n"
        "OTHER_VALUE='abc # not a comment'\n",
        encoding="utf-8",
    )

    import config

    monkeypatch.delenv("INSIDE_SENSOR", raising=False)
    monkeypatch.delenv("OTHER_VALUE", raising=False)

    config._load_env_file(str(env_path))

    assert config.os.environ["INSIDE_SENSOR"] == "pimoroni_bme68x"
    assert config.os.environ["OTHER_VALUE"] == "abc # not a comment"


def test_load_env_file_accepts_export_prefix(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("export WEATHERKIT_TEAM_ID=team_123\n", encoding="utf-8")

    import config

    monkeypatch.delenv("WEATHERKIT_TEAM_ID", raising=False)
    config._load_env_file(str(env_path))

    assert config.os.environ["WEATHERKIT_TEAM_ID"] == "team_123"


def test_config_import_loads_dotenv_before_weather_constants(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "WEATHER_LATITUDE=41.1\n"
        "WEATHER_LONGITUDE=-87.2\n"
        "WEATHERKIT_TEAM_ID=team_123\n"
        "WEATHERKIT_KEY_ID=key_123\n"
        "WEATHERKIT_SERVICE_ID=service_123\n"
        "WEATHERKIT_KEY_PATH=/tmp/AuthKey_key_123.p8\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    for key in (
        "WEATHER_LATITUDE",
        "WEATHER_LONGITUDE",
        "WEATHERKIT_TEAM_ID",
        "WEATHERKIT_KEY_ID",
        "WEATHERKIT_SERVICE_ID",
        "WEATHERKIT_KEY_PATH",
        "WEATHERKIT_PRIVATE_KEY",
        "OWM_API_KEY",
        "OWM_API_KEY_DEFAULT",
        "OWM_API_KEY_WIFFY",
        "OWM_API_KEY_VERANO",
    ):
        monkeypatch.delenv(key, raising=False)

    module = _reload_config(monkeypatch)

    assert module.WEATHERKIT_TEAM_ID == "team_123"
    assert module.WEATHERKIT_KEY_ID == "key_123"
    assert module.WEATHERKIT_SERVICE_ID == "service_123"
    assert module.WEATHERKIT_KEY_PATH == "/tmp/AuthKey_key_123.p8"
    assert module.ENABLE_WEATHER is True


def test_ip_with_time_flag_obeys_env(monkeypatch):
    module = _reload_config(monkeypatch, IP_WITH_TIME="false")
    assert module.IP_WITH_TIME is False

    module = _reload_config(monkeypatch, IP_WITH_TIME="true")
    assert module.IP_WITH_TIME is True

    module = _reload_config(monkeypatch, IP_WITH_TIME=None)
    assert module.IP_WITH_TIME is True
