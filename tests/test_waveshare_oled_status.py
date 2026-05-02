"""Tests for Waveshare OLED helper resiliency and shutdown behavior."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
import json


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


def test_read_weather1_temp_prefers_cached_weather_before_force_refresh(monkeypatch):
    mod = _load_module()
    calls = []

    fake = types.ModuleType("data_fetch")

    def _fetch_weather(*, force_refresh=False):
        calls.append(force_refresh)
        return {"current": {"temp": 67.9}}

    fake.fetch_weather = _fetch_weather
    monkeypatch.setitem(sys.modules, "data_fetch", fake)

    assert mod._read_weather1_temp_f() == 67.9
    assert calls == [False]


def test_read_weather1_temp_returns_last_known_value_when_fetch_fails(monkeypatch):
    mod = _load_module()
    mod._LAST_WEATHER_TEMP_F = 72.2

    fake = types.ModuleType("data_fetch")
    fake.fetch_weather = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setitem(sys.modules, "data_fetch", fake)

    assert mod._read_weather1_temp_f() == 72.2


def test_read_temperature_uses_last_known_weather_value(monkeypatch):
    mod = _load_module()

    monkeypatch.setattr(mod, "TEMP_SOURCE", "weather1")
    monkeypatch.setattr(mod, "_read_weather1_temp_f", lambda: 70.4)

    assert mod.read_temperature() == "70°F"


def test_weather2_gate_uses_status_path_override(monkeypatch, tmp_path):
    mod = _load_module()
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps({"screen_id": "weather2"}), encoding="utf-8")

    monkeypatch.setattr(mod, "_WEATHER2_RENDERED", False)
    monkeypatch.setattr(mod, "TEMP_SOURCE", "weather1")
    monkeypatch.setattr(mod, "WAIT_FOR_WEATHER2", True)
    monkeypatch.setenv("WAVESHARE_OLED_DISPLAY_STATUS_PATH", str(status_path))

    assert mod._weather2_screen_has_rendered() is True
    assert mod._WEATHER2_RENDERED is True


def test_weather2_gate_waits_until_weather2(monkeypatch, tmp_path):
    mod = _load_module()
    screenshot_dir = tmp_path / "shots"
    status_path = screenshot_dir / "current" / "display_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps({"screen_id": "weather1"}), encoding="utf-8")

    monkeypatch.setattr(mod, "_WEATHER2_RENDERED", False)
    monkeypatch.setattr(mod, "TEMP_SOURCE", "weather1")
    monkeypatch.setattr(mod, "WAIT_FOR_WEATHER2", True)
    monkeypatch.setenv("SCREENSHOT_DIR", str(screenshot_dir))
    monkeypatch.delenv("WAVESHARE_OLED_DISPLAY_STATUS_PATH", raising=False)

    assert mod._weather2_screen_has_rendered() is False


def test_weather2_gate_disabled_for_non_weather_sources(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(mod, "_WEATHER2_RENDERED", False)
    monkeypatch.setattr(mod, "TEMP_SOURCE", "cpu")
    monkeypatch.setattr(mod, "WAIT_FOR_WEATHER2", True)

    assert mod._weather2_screen_has_rendered() is True


def test_best_time_font_size_accounts_for_meridiem_width():
    mod = _load_module()

    full_time_size = mod._best_time_font_size(128, 64, "12:54 PM", 12)
    base_time_size = mod._best_value_font_size(128, 64, "12:54", 12)

    assert full_time_size <= base_time_size


def test_font_size_respects_configured_maximums(monkeypatch):
    monkeypatch.setenv("WAVESHARE_OLED_MAX_VALUE_FONT_SIZE", "12")
    monkeypatch.setenv("WAVESHARE_OLED_MAX_TIME_FONT_SIZE", "10")
    mod = _load_module()

    value_size = mod._best_value_font_size(128, 64, "11/18/26", 12)
    time_size = mod._best_time_font_size(128, 64, "10:54 PM", 12)

    assert value_size <= 12
    assert time_size <= 10


def test_github_updates_available_returns_false_when_not_git_repo(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(mod, "_LAST_GITHUB_UPDATE_CHECK_AT", 0.0)
    monkeypatch.setattr(mod, "_LAST_GITHUB_UPDATE_AVAILABLE", True)
    monkeypatch.setattr(mod.Path, "exists", lambda _self: False)

    assert mod._github_updates_available(force=True) is False


def test_main_inverts_frames_when_github_update_available(monkeypatch):
    mod = _load_module()

    bus = _FakeSMBus()
    rendered = []

    class _Display:
        def __init__(self, *_args, **_kwargs):
            pass

        def initialize(self):
            return None

        def clear(self):
            return None

    def _capture_render(_display, image, _name):
        rendered.append(image)
        return True

    monkeypatch.setattr(mod, "SMBus", lambda *_args, **_kwargs: bus)
    monkeypatch.setattr(mod, "SSD1306Display", _Display)
    monkeypatch.setattr(mod, "_safe_render", _capture_render)
    monkeypatch.setattr(mod, "current_time_12h", lambda: "10:54 PM")
    monkeypatch.setattr(mod, "current_date_mdy", lambda: "11/18/26")
    monkeypatch.setattr(mod, "random_swap_interval_seconds", lambda: 60)
    monkeypatch.setattr(mod, "_best_time_font_size", lambda *_args, **_kwargs: 20)
    monkeypatch.setattr(mod, "_best_value_font_size", lambda *_args, **_kwargs: 16)
    monkeypatch.setattr(mod, "_github_updates_available", lambda: True)
    monkeypatch.setattr(mod._STOP_EVENT, "wait", lambda _seconds: mod._STOP_EVENT.set())

    def _solid_white(*_args, **_kwargs):
        return mod.Image.new("1", (mod.OLED_WIDTH, mod.OLED_HEIGHT), 1)

    monkeypatch.setattr(mod, "render_centered_text", _solid_white)
    monkeypatch.setattr(mod, "render_centered_time_text", _solid_white)

    mod._STOP_EVENT.clear()
    rc = mod.main()

    assert rc == 0
    assert len(rendered) >= 2
    assert rendered[0].getpixel((0, 0)) == 0
    assert rendered[1].getpixel((0, 0)) == 0


def test_main_uses_independent_font_sizes_for_date_and_time(monkeypatch):
    mod = _load_module()

    bus = _FakeSMBus()
    font_sizes = []

    class _Display:
        def __init__(self, *_args, **_kwargs):
            pass

        def initialize(self):
            return None

        def clear(self):
            return None

    def _render_centered_text(_w, _h, _text, *, title=None, value_font_size=None):
        font_sizes.append((title, value_font_size))
        return object()

    monkeypatch.setattr(mod, "SMBus", lambda *_args, **_kwargs: bus)
    monkeypatch.setattr(mod, "SSD1306Display", _Display)
    monkeypatch.setattr(mod, "fade_transition", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "current_time_12h", lambda: "10:54 PM")
    monkeypatch.setattr(mod, "current_date_mdy", lambda: "11/18/26")
    monkeypatch.setattr(mod, "random_swap_interval_seconds", lambda: 60)
    monkeypatch.setattr(mod, "_best_time_font_size", lambda *_args, **_kwargs: 20)
    monkeypatch.setattr(mod, "_best_value_font_size", lambda *_args, **_kwargs: 16)
    monkeypatch.setattr(mod, "render_centered_text", _render_centered_text)
    monkeypatch.setattr(mod, "render_centered_time_text", _render_centered_text)
    monkeypatch.setattr(mod._STOP_EVENT, "wait", lambda _seconds: mod._STOP_EVENT.set())

    mod._STOP_EVENT.clear()
    rc = mod.main()

    assert rc == 0
    assert ("Date", 16) in font_sizes
    assert ("Time", 20) in font_sizes


def test_cubs_oled_frames_prefers_live_game(monkeypatch):
    mod = _load_module()
    rendered = []
    game = {
        "gamePk": 123,
        "status": {"abstractGameState": "Live", "detailedState": "In Progress", "statusCode": "I"},
        "teams": {
            "away": {"team": {"id": 112, "name": "Chicago Cubs", "abbreviation": "CHC"}, "score": 3},
            "home": {"team": {"id": 121, "name": "New York Mets", "abbreviation": "NYM"}, "score": 2},
        },
        "linescore": {"inningState": "Bottom", "currentInningOrdinal": "3rd", "outs": 2},
    }

    monkeypatch.setattr(mod, "_read_display_status_payload", lambda: {"cubs": {"live_game": game}})
    monkeypatch.setattr(mod, "_render_score_panel", lambda *_args, **kwargs: rendered.append(kwargs) or object())
    monkeypatch.setattr(mod, "_resolve_mlb_abbreviation", lambda: (lambda text: "NYM" if "Mets" in text else "CUBS"))
    monkeypatch.setattr(mod, "_CUBS_FINAL_GAME_PK", None)
    monkeypatch.setattr(mod, "_CUBS_FINAL_HOLD_UNTIL_EPOCH", 0.0)
    monkeypatch.setattr(mod, "_load_cubs_final_state", lambda: (None, 0.0))
    monkeypatch.setattr(mod, "_persist_cubs_final_state", lambda *_args, **_kwargs: None)

    frames = mod._cubs_oled_frames()

    assert frames is not None
    assert len(rendered) == 2
    assert rendered[0]["team"] == "CUBS"
    assert rendered[0]["footer"] == ""
    assert rendered[1]["team"] == "NYM"
    assert rendered[1]["footer"] == "Bottom 3rd • 2 Outs"


def test_cubs_oled_frames_holds_final_for_90_minutes(monkeypatch):
    mod = _load_module()
    game = {
        "gamePk": 999,
        "status": {"abstractGameState": "Final", "detailedState": "Final", "statusCode": "F"},
        "teams": {
            "away": {"team": {"id": 112, "name": "Chicago Cubs", "abbreviation": "CHC"}, "score": 7},
            "home": {"team": {"id": 111, "name": "Boston Red Sox", "abbreviation": "BOS"}, "score": 4},
        },
        "linescore": {},
    }

    now = [1000.0]
    monkeypatch.setattr(mod, "time", types.SimpleNamespace(time=lambda: now[0]))
    monkeypatch.setattr(mod, "_read_display_status_payload", lambda: {"cubs": {"last_game": game}})
    monkeypatch.setattr(mod, "_render_score_panel", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(mod, "_CUBS_FINAL_GAME_PK", None)
    monkeypatch.setattr(mod, "_CUBS_FINAL_HOLD_UNTIL_EPOCH", 0.0)
    monkeypatch.setattr(mod, "_load_cubs_final_state", lambda: (None, 0.0))
    monkeypatch.setattr(mod, "_persist_cubs_final_state", lambda *_args, **_kwargs: None)

    assert mod._cubs_oled_frames() is not None
    now[0] += (90 * 60) - 1
    assert mod._cubs_oled_frames() is not None
    now[0] += 2
    assert mod._cubs_oled_frames() is None
