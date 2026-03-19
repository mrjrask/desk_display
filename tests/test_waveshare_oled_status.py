"""Tests for Waveshare OLED helper resiliency and shutdown behavior."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "waveshare_oled_status.py"


class _FakeSMBus:
    def __init__(self, *_args, **_kwargs):
        self.closed = False

    def write_byte_data(self, *_args, **_kwargs):
        return None

    def write_i2c_block_data(self, *_args, **_kwargs):
        return None

    def close(self):
        self.closed = True


def _load_module(*, use_smbus2: bool = False):
    sys.modules.pop("waveshare_oled_status", None)
    sys.modules.pop("smbus", None)
    sys.modules.pop("smbus2", None)

    if use_smbus2:
        fake_smbus2 = types.ModuleType("smbus2")
        fake_smbus2.SMBus = _FakeSMBus
        sys.modules["smbus2"] = fake_smbus2
    else:
        fake_smbus = types.ModuleType("smbus")
        fake_smbus.SMBus = _FakeSMBus
        sys.modules["smbus"] = fake_smbus

    spec = importlib.util.spec_from_file_location("waveshare_oled_status", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_request_stop_sets_stop_event():
    mod = _load_module()
    mod._STOP_EVENT.clear()

    mod._request_stop(15, None)

    assert mod._STOP_EVENT.is_set() is True


def test_safe_render_handles_render_failures():
    mod = _load_module()

    class _Display:
        pass

    mod.fade_transition = lambda _d, _i: (_ for _ in ()).throw(RuntimeError("i2c write failed"))

    ok = mod._safe_render(_Display(), object(), "left")

    assert ok is False


def test_main_clears_oleds_and_closes_bus_on_stop(monkeypatch):
    mod = _load_module()

    bus = _FakeSMBus()
    created_displays = []

    class _Display:
        def __init__(self, bus_obj, _addr, _width, _height):
            self.bus = bus_obj
            self.cleared = 0
            created_displays.append(self)

        def initialize(self):
            return None

        def clear(self):
            self.cleared += 1

    monkeypatch.setattr(mod, "SMBus", lambda *_args, **_kwargs: bus)
    monkeypatch.setattr(mod, "SSD1306Display", _Display)
    monkeypatch.setattr(mod, "fade_transition", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "read_temperature", lambda: "72°F")
    monkeypatch.setattr(mod, "current_time_12h", lambda: "8:00 PM")
    monkeypatch.setattr(mod, "render_centered_text", lambda *_args, **_kwargs: object())

    mod._STOP_EVENT.set()
    rc = mod.main()

    assert rc == 0
    assert len(created_displays) == 2
    assert created_displays[0].cleared >= 2
    assert created_displays[1].cleared >= 2
    assert bus.closed is True


def test_import_uses_smbus2_when_smbus_missing():
    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name == "smbus":
            raise ImportError("smbus unavailable")
        return real_import(name, *args, **kwargs)

    import builtins

    original_import = builtins.__import__
    builtins.__import__ = _fake_import
    try:
        mod = _load_module(use_smbus2=True)
    finally:
        builtins.__import__ = original_import

    assert mod.SMBus is _FakeSMBus


def test_read_weather1_temp_uses_fetch_weather(monkeypatch):
    mod = _load_module()

    fake = types.ModuleType("data_fetch")
    fake.fetch_weather = lambda: {"current": {"temp": 68.4}}
    monkeypatch.setitem(sys.modules, "data_fetch", fake)

    assert mod._read_weather1_temp_f() == 68.4


def test_read_weather1_temp_supports_legacy_get_weather_data(monkeypatch):
    mod = _load_module()

    fake = types.ModuleType("data_fetch")
    fake.get_weather_data = lambda: {"current": {"temp": "71"}}
    monkeypatch.setitem(sys.modules, "data_fetch", fake)

    assert mod._read_weather1_temp_f() == 71.0
