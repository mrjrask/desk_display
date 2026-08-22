"""Tests for Waveshare OLED helper resiliency and shutdown behavior."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

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


def test_read_weather1_payload_reads_display_status_file(monkeypatch):
    mod = _load_module()

    monkeypatch.setattr(
        mod,
        "_read_display_status_payload",
        lambda: {"weather": {"temp_f": 68.4, "condition": "clear sky"}},
    )

    assert mod._read_weather1_temp_f() == 68.4
    assert mod._LAST_WEATHER_CONDITION == "Clear Sky"
    assert mod.read_weather_condition() == "Clear Sky"


def test_read_weather1_payload_does_not_call_data_fetch(monkeypatch):
    """The OLED helper must never make its own weather API call; it only
    reads the summary main.py already wrote to display_status.json."""

    mod = _load_module()

    def _boom(*_args, **_kwargs):
        raise AssertionError("OLED helper must not call data_fetch itself")

    fake = types.ModuleType("data_fetch")
    fake.fetch_weather = _boom
    monkeypatch.setitem(sys.modules, "data_fetch", fake)
    monkeypatch.setattr(
        mod,
        "_read_display_status_payload",
        lambda: {"weather": {"temp_f": 70.0, "condition": "windy"}},
    )

    assert mod._read_weather1_temp_f() == 70.0
    assert mod.read_weather_condition() == "Windy"


def test_read_weather1_temp_returns_last_known_value_when_status_missing_weather(monkeypatch):
    mod = _load_module()
    mod._LAST_WEATHER_TEMP_F = 72.2
    mod._LAST_WEATHER_CONDITION = "Sunny"

    monkeypatch.setattr(mod, "_read_display_status_payload", lambda: {})

    assert mod._read_weather1_temp_f() == 72.2
    assert mod.read_weather_condition() == "Sunny"


def test_read_temperature_uses_last_known_weather_value(monkeypatch):
    mod = _load_module()

    monkeypatch.setattr(mod, "TEMP_SOURCE", "weather1")
    monkeypatch.setattr(mod, "_read_weather1_temp_f", lambda: 70.4)

    assert mod.read_temperature() == "70°F"


def test_read_weather_condition_returns_cached_value(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(
        mod, "_read_weather1_payload", lambda: (68.4, "Sunny")
    )

    assert mod.read_weather_condition() == "Sunny"


def test_read_weather_condition_defaults_to_empty_string(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(mod, "_read_weather1_payload", lambda: (None, None))

    assert mod.read_weather_condition() == ""


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


def test_render_idle_panel_dispatches_by_content_id(monkeypatch):
    mod = _load_module()

    monkeypatch.setattr(mod, "current_date_mdy", lambda: "11/18/26")
    monkeypatch.setattr(mod, "current_time_12h", lambda: "10:54 PM")
    monkeypatch.setattr(mod, "read_temperature", lambda: "72°F")
    monkeypatch.setattr(mod, "read_weather_condition", lambda: "Sunny")

    calls = {}

    def _capture_centered_text(_w, _h, text, *, title=None, value_font_size=None):
        calls[title] = text
        return object()

    monkeypatch.setattr(
        mod,
        "render_centered_text",
        _capture_centered_text,
    )
    monkeypatch.setattr(
        mod,
        "render_centered_time_text",
        lambda *_a, **kw: calls.setdefault("Time", kw.get("title")) or object(),
    )

    mod._render_idle_panel("date")
    mod._render_idle_panel("time")
    mod._render_idle_panel("temp")
    mod._render_idle_panel("condition")

    assert calls["Date"] == "11/18/26"
    assert calls["Time"] == "Time"
    assert calls["Temp"] == "72°F"
    assert calls["Now"] == "Sunny"


def test_render_idle_panel_falls_back_to_placeholder_when_weather_missing(monkeypatch):
    mod = _load_module()

    monkeypatch.setattr(mod, "read_temperature", lambda: "")
    monkeypatch.setattr(mod, "read_weather_condition", lambda: "")

    calls = {}

    def _capture_centered_text(_w, _h, text, *, title=None, value_font_size=None):
        calls[title] = text
        return object()

    monkeypatch.setattr(mod, "render_centered_text", _capture_centered_text)

    mod._render_idle_panel("temp")
    mod._render_idle_panel("condition")

    assert calls["Temp"] == "--"
    assert calls["Now"] == "--"


def test_main_rotates_between_date_time_and_temp_condition_pairs(monkeypatch):
    mod = _load_module()

    bus = _FakeSMBus()
    seen_pairs = []

    class _Display:
        def __init__(self, *_args, **_kwargs):
            pass

        def initialize(self):
            return None

        def clear(self):
            return None

    monkeypatch.setattr(mod, "SMBus", lambda *_args, **_kwargs: bus)
    monkeypatch.setattr(mod, "SSD1306Display", _Display)
    monkeypatch.setattr(mod, "fade_transition", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "_cubs_oled_frames", lambda: None)
    monkeypatch.setattr(mod, "_hawks_oled_frames", lambda: None)
    monkeypatch.setattr(mod, "_github_updates_available", lambda: False)
    monkeypatch.setattr(mod, "_save_oled_screenshot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "random_swap_interval_seconds", lambda: 0)

    def _render_idle_panel(content_id):
        return content_id

    monkeypatch.setattr(mod, "_render_idle_panel", _render_idle_panel)

    remaining_iterations = [3]

    def _capture_render(_display, image, name):
        if name == "left":
            seen_pairs.append([image])
        else:
            seen_pairs[-1].append(image)
        return True

    monkeypatch.setattr(mod, "_safe_render", _capture_render)

    def _fake_wait(_seconds):
        remaining_iterations[0] -= 1
        if remaining_iterations[0] <= 0:
            mod._STOP_EVENT.set()

    monkeypatch.setattr(mod._STOP_EVENT, "wait", _fake_wait)

    mod._STOP_EVENT.clear()
    rc = mod.main()

    assert rc == 0
    assert seen_pairs == [
        ["date", "time"],
        ["temp", "condition"],
        ["date", "time"],
    ]


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
    monkeypatch.setattr(mod, "_save_oled_screenshot", lambda *_args, **_kwargs: None)
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
    assert rendered[0]["footer"] == "Bottom 3rd"
    assert rendered[1]["team"] == "NYM"
    assert rendered[1]["footer"] == "2 Outs"


def test_cubs_oled_frames_footer_layout_is_consistent_regardless_of_batter(monkeypatch):
    """The inning always renders on the away panel and outs on the home panel,
    regardless of which team is currently batting."""
    mod = _load_module()
    rendered = []
    game = {
        "gamePk": 321,
        "status": {"abstractGameState": "Live", "detailedState": "In Progress", "statusCode": "I"},
        "teams": {
            "away": {"team": {"id": 112, "name": "Chicago Cubs", "abbreviation": "CHC"}, "score": 1},
            "home": {"team": {"id": 121, "name": "New York Mets", "abbreviation": "NYM"}, "score": 0},
        },
        "linescore": {"inningState": "Top", "currentInningOrdinal": "5th", "outs": 1},
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
    assert rendered[0]["footer"] == "Top 5th"
    assert rendered[1]["footer"] == "1 Out"


def test_cubs_oled_frames_final_hides_outs(monkeypatch):
    mod = _load_module()
    rendered = []
    game = {
        "gamePk": 777,
        "status": {"abstractGameState": "Final", "detailedState": "Final", "statusCode": "F"},
        "teams": {
            "away": {"team": {"id": 112, "name": "Chicago Cubs", "abbreviation": "CHC"}, "score": 5},
            "home": {"team": {"id": 121, "name": "New York Mets", "abbreviation": "NYM"}, "score": 3},
        },
        "linescore": {"inningState": "Bottom", "currentInningOrdinal": "9th", "outs": 2},
    }

    monkeypatch.setattr(mod, "time", types.SimpleNamespace(time=lambda: 1000.0))
    monkeypatch.setattr(mod, "_read_display_status_payload", lambda: {"cubs": {"last_game": game}})
    monkeypatch.setattr(mod, "_render_score_panel", lambda *_args, **kwargs: rendered.append(kwargs) or object())
    monkeypatch.setattr(mod, "_CUBS_FINAL_GAME_PK", None)
    monkeypatch.setattr(mod, "_CUBS_FINAL_HOLD_UNTIL_EPOCH", 0.0)
    monkeypatch.setattr(mod, "_load_cubs_final_state", lambda: (None, 0.0))
    monkeypatch.setattr(mod, "_persist_cubs_final_state", lambda *_args, **_kwargs: None)

    frames = mod._cubs_oled_frames()

    assert frames is not None
    assert len(rendered) == 2
    assert rendered[0]["footer"] == ""
    assert rendered[1]["footer"] == "Final"

@pytest.mark.parametrize(
    "detailed, expected_status",
    [
        ("Warmup", "Warmup"),
        ("Pre-Game Warmup", "Warmup"),
        ("Delayed", "Delayed"),
        ("Rain Delay", "Rain Delay"),
        ("Suspended", "Suspended"),
        ("Postponed", "Postponed"),
        ("Canceled", "Canceled"),
    ],
)
def test_cubs_oled_frames_shows_non_inning_statuses(monkeypatch, detailed, expected_status):
    mod = _load_module()
    rendered = []
    game = {
        "gamePk": 246,
        "status": {"abstractGameState": "Preview", "detailedState": detailed, "statusCode": ""},
        "teams": {
            "away": {"team": {"id": 112, "name": "Chicago Cubs", "abbreviation": "CHC"}, "score": 0},
            "home": {"team": {"id": 121, "name": "New York Mets", "abbreviation": "NYM"}, "score": 0},
        },
        "linescore": {},
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
    assert rendered[0]["footer"] == expected_status
    assert rendered[1]["footer"] == ""


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


def test_hawks_oled_frames_prefers_live_game(monkeypatch):
    mod = _load_module()
    rendered = []
    game = {
        "id": 555,
        "gameState": "LIVE",
        "awayTeam": {"id": 16, "abbrev": "CHI"},
        "homeTeam": {"id": 10, "abbrev": "TOR"},
    }
    feed = {
        "awayScore": 2,
        "homeScore": 1,
        "perOrdinal": "2",
        "clock": "12:34",
        "clockState": "",
    }

    monkeypatch.setattr(
        mod,
        "_read_display_status_payload",
        lambda: {"hawks": {"live_game": game, "live_feed": feed}},
    )
    monkeypatch.setattr(mod, "_render_score_panel", lambda *_args, **kwargs: rendered.append(kwargs) or object())
    monkeypatch.setattr(mod, "_HAWKS_FINAL_GAME_PK", None)
    monkeypatch.setattr(mod, "_HAWKS_FINAL_HOLD_UNTIL_EPOCH", 0.0)
    monkeypatch.setattr(mod, "_load_hawks_final_state", lambda: (None, 0.0))
    monkeypatch.setattr(mod, "_persist_hawks_final_state", lambda *_args, **_kwargs: None)

    frames = mod._hawks_oled_frames()

    assert frames is not None
    assert len(rendered) == 2
    assert rendered[0]["team"] == "HAWKS"
    assert rendered[0]["score"] == "2"
    assert rendered[0]["footer"] == "2nd Period"
    assert rendered[1]["team"] == "TOR"
    assert rendered[1]["score"] == "1"
    assert rendered[1]["footer"] == "12:34"


def test_hawks_oled_frames_final_hides_clock(monkeypatch):
    mod = _load_module()
    rendered = []
    game = {
        "id": 556,
        "gameState": "OFF",
        "awayTeam": {"id": 16, "abbrev": "CHI", "score": 4},
        "homeTeam": {"id": 10, "abbrev": "TOR", "score": 3},
    }

    monkeypatch.setattr(mod, "time", types.SimpleNamespace(time=lambda: 1000.0))
    monkeypatch.setattr(mod, "_read_display_status_payload", lambda: {"hawks": {"last_game": game}})
    monkeypatch.setattr(mod, "_render_score_panel", lambda *_args, **kwargs: rendered.append(kwargs) or object())
    monkeypatch.setattr(mod, "_HAWKS_FINAL_GAME_PK", None)
    monkeypatch.setattr(mod, "_HAWKS_FINAL_HOLD_UNTIL_EPOCH", 0.0)
    monkeypatch.setattr(mod, "_load_hawks_final_state", lambda: (None, 0.0))
    monkeypatch.setattr(mod, "_persist_hawks_final_state", lambda *_args, **_kwargs: None)

    frames = mod._hawks_oled_frames()

    assert frames is not None
    assert len(rendered) == 2
    assert rendered[0]["score"] == "4"
    assert rendered[0]["footer"] == ""
    assert rendered[1]["score"] == "3"
    assert rendered[1]["footer"] == "Final"


@pytest.mark.parametrize(
    "game_state, expected_status",
    [
        ("PRE", "Pre-Game"),
        ("PREGAME", "Pre-Game"),
        ("POSTP", "Postponed"),
        ("PPD", "Postponed"),
        ("SUSP", "Suspended"),
    ],
)
def test_hawks_oled_frames_shows_non_period_statuses(monkeypatch, game_state, expected_status):
    mod = _load_module()
    rendered = []
    game = {
        "id": 559,
        "gameState": game_state,
        "awayTeam": {"id": 16, "abbrev": "CHI"},
        "homeTeam": {"id": 10, "abbrev": "TOR"},
    }

    monkeypatch.setattr(mod, "_read_display_status_payload", lambda: {"hawks": {"live_game": game}})
    monkeypatch.setattr(mod, "_render_score_panel", lambda *_args, **kwargs: rendered.append(kwargs) or object())
    monkeypatch.setattr(mod, "_HAWKS_FINAL_GAME_PK", None)
    monkeypatch.setattr(mod, "_HAWKS_FINAL_HOLD_UNTIL_EPOCH", 0.0)
    monkeypatch.setattr(mod, "_load_hawks_final_state", lambda: (None, 0.0))
    monkeypatch.setattr(mod, "_persist_hawks_final_state", lambda *_args, **_kwargs: None)

    frames = mod._hawks_oled_frames()

    assert frames is not None
    assert len(rendered) == 2
    assert rendered[0]["team"] == "HAWKS"
    assert rendered[0]["footer"] == expected_status
    assert rendered[1]["footer"] == ""


def test_hawks_oled_frames_holds_final_for_90_minutes(monkeypatch):
    mod = _load_module()
    game = {
        "id": 557,
        "gameState": "OFF",
        "awayTeam": {"id": 16, "abbrev": "CHI", "score": 5},
        "homeTeam": {"id": 10, "abbrev": "TOR", "score": 2},
    }

    now = [1000.0]
    monkeypatch.setattr(mod, "time", types.SimpleNamespace(time=lambda: now[0]))
    monkeypatch.setattr(mod, "_read_display_status_payload", lambda: {"hawks": {"last_game": game}})
    monkeypatch.setattr(mod, "_render_score_panel", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(mod, "_HAWKS_FINAL_GAME_PK", None)
    monkeypatch.setattr(mod, "_HAWKS_FINAL_HOLD_UNTIL_EPOCH", 0.0)
    monkeypatch.setattr(mod, "_load_hawks_final_state", lambda: (None, 0.0))
    monkeypatch.setattr(mod, "_persist_hawks_final_state", lambda *_args, **_kwargs: None)

    assert mod._hawks_oled_frames() is not None
    now[0] += (90 * 60) - 1
    assert mod._hawks_oled_frames() is not None
    now[0] += 2
    assert mod._hawks_oled_frames() is None


def test_hawks_oled_frames_falls_back_when_cubs_not_playing(monkeypatch):
    mod = _load_module()
    hawks_game = {
        "id": 558,
        "gameState": "LIVE",
        "awayTeam": {"id": 16, "abbrev": "CHI"},
        "homeTeam": {"id": 10, "abbrev": "TOR"},
    }

    monkeypatch.setattr(mod, "_read_display_status_payload", lambda: {"hawks": {"live_game": hawks_game}})
    monkeypatch.setattr(mod, "_render_score_panel", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(mod, "_HAWKS_FINAL_GAME_PK", None)
    monkeypatch.setattr(mod, "_HAWKS_FINAL_HOLD_UNTIL_EPOCH", 0.0)
    monkeypatch.setattr(mod, "_load_hawks_final_state", lambda: (None, 0.0))
    monkeypatch.setattr(mod, "_persist_hawks_final_state", lambda *_args, **_kwargs: None)

    assert mod._cubs_oled_frames() is None
    assert mod._hawks_oled_frames() is not None


def test_save_oled_screenshot_writes_current_png(monkeypatch, tmp_path):
    mod = _load_module()
    monkeypatch.setenv("WAVESHARE_OLED_SCREENSHOT_DIR", str(tmp_path))

    image = mod.Image.new("1", (mod.OLED_WIDTH, mod.OLED_HEIGHT), 1)
    mod._save_oled_screenshot("oled_left", image)

    saved_path = tmp_path / "oled_left.png"
    assert saved_path.exists()
    assert not (tmp_path / "oled_left.png.tmp").exists()


def test_save_oled_screenshot_swallows_errors(monkeypatch, tmp_path):
    mod = _load_module()
    monkeypatch.setenv("WAVESHARE_OLED_SCREENSHOT_DIR", str(tmp_path))

    class _Unsavable:
        def save(self, _path):
            raise RuntimeError("boom")

    # Should not raise.
    mod._save_oled_screenshot("oled_left", _Unsavable())
