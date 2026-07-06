"""Tests for best-effort Waveshare OLED cleanup gating in cleanup.sh."""

from __future__ import annotations

import sys
import types
from pathlib import Path


CLEANUP_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "maintenance" / "cleanup.sh"


def _cleanup_python_block() -> str:
    text = CLEANUP_SCRIPT.read_text(encoding="utf-8")
    return text.split('"${python_bin}" - <<\'PY\'\n', 1)[1].split("\nPY\n", 1)[0]


def _install_display_fakes(monkeypatch):
    image_mod = types.SimpleNamespace(new=lambda *_args, **_kwargs: object())
    monkeypatch.setitem(sys.modules, "PIL", types.SimpleNamespace(Image=image_mod))
    monkeypatch.setitem(sys.modules, "PIL.Image", image_mod)

    class _Display:
        width = 320
        height = 240

        def image(self, *_args, **_kwargs):
            return None

        def set_led(self, *_args, **_kwargs):
            return None

        def show(self):
            return None

        def clear(self):
            return None

        def set_backlight(self, *_args, **_kwargs):
            return None

    monkeypatch.setitem(
        sys.modules,
        "utils",
        types.SimpleNamespace(
            Display=_Display,
            clear_display=lambda *_args, **_kwargs: None,
            clear_update_indicator=lambda *_args, **_kwargs: None,
        ),
    )


def _clear_waveshare_env(monkeypatch):
    for name in list(sys.modules["os"].environ):
        if name.startswith("WAVESHARE_OLED"):
            monkeypatch.delenv(name, raising=False)


def test_cleanup_skips_waveshare_oled_when_profile_is_not_configured(monkeypatch, tmp_path):
    _install_display_fakes(monkeypatch)
    _clear_waveshare_env(monkeypatch)
    monkeypatch.chdir(tmp_path)

    def _fail_import(name, *args, **kwargs):
        if name in {"smbus", "smbus2", "waveshare_oled_status"}:
            raise AssertionError(f"unexpected Waveshare cleanup import: {name}")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", _fail_import)

    exec(_cleanup_python_block(), {})


def test_cleanup_skips_waveshare_oled_when_i2c_device_is_missing(monkeypatch, tmp_path):
    _install_display_fakes(monkeypatch)
    _clear_waveshare_env(monkeypatch)
    monkeypatch.setenv("WAVESHARE_OLED_MAX_VALUE_FONT_SIZE", "26")
    monkeypatch.chdir(tmp_path)

    def _unexpected_bus(*_args, **_kwargs):
        raise AssertionError("SMBus should not open when /dev/i2c-1 is missing")

    monkeypatch.setitem(sys.modules, "smbus", types.SimpleNamespace(SMBus=_unexpected_bus))
    monkeypatch.setitem(
        sys.modules,
        "waveshare_oled_status",
        types.SimpleNamespace(
            I2C_BUS=1,
            TEMP_ADDR=0x3C,
            TIME_ADDR=0x3D,
            OLED_WIDTH=128,
            OLED_HEIGHT=64,
            SSD1306Display=object,
        ),
    )

    exec(_cleanup_python_block(), {})


def test_cleanup_skips_waveshare_oled_when_hyperpixel_profile_is_configured(monkeypatch, tmp_path):
    _install_display_fakes(monkeypatch)
    _clear_waveshare_env(monkeypatch)
    monkeypatch.setenv("WAVESHARE_OLED_MAX_VALUE_FONT_SIZE", "26")
    monkeypatch.setenv("HYPERPIXEL_PANEL", "hyperpixel4")
    monkeypatch.chdir(tmp_path)

    def _fail_import(name, *args, **kwargs):
        if name in {"smbus", "smbus2", "waveshare_oled_status"}:
            raise AssertionError(f"unexpected Waveshare cleanup import: {name}")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", _fail_import)

    exec(_cleanup_python_block(), {})


def test_cleanup_skips_waveshare_oled_when_pimoroni_profile_is_configured(monkeypatch, tmp_path):
    _install_display_fakes(monkeypatch)
    _clear_waveshare_env(monkeypatch)
    monkeypatch.setenv("WAVESHARE_OLED_MAX_VALUE_FONT_SIZE", "26")
    monkeypatch.setenv("DESK_DISPLAY_OUTPUT", "displayhatmini")
    monkeypatch.chdir(tmp_path)

    def _fail_import(name, *args, **kwargs):
        if name in {"smbus", "smbus2", "waveshare_oled_status"}:
            raise AssertionError(f"unexpected Waveshare cleanup import: {name}")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", _fail_import)

    exec(_cleanup_python_block(), {})
