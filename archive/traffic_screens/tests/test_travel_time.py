import datetime as dt
import importlib
import os

os.environ.setdefault("GOOGLE_MAPS_API_KEY", "test")

travel = importlib.import_module("screens.draw_travel_time")


def test_get_travel_active_window_none(monkeypatch):
    monkeypatch.setattr(travel, "TRAVEL_ACTIVE_WINDOW", None)
    assert travel.get_travel_active_window() is None
    assert travel.is_travel_screen_active(now=dt.time(12, 0))


def test_get_travel_active_window_string_parsing(monkeypatch):
    monkeypatch.setattr(travel, "TRAVEL_ACTIVE_WINDOW", ("2:30 PM", "7:00 PM"))
    window = travel.get_travel_active_window()
    assert window is not None
    start, end = window
    assert start.hour == 14 and start.minute == 30
    assert end.hour == 19 and end.minute == 0
    assert travel.is_travel_screen_active(now=dt.time(15, 0))
    assert not travel.is_travel_screen_active(now=dt.time(8, 0))


def test_get_travel_active_window_equal_times(monkeypatch):
    same_time = dt.time(6, 0)
    monkeypatch.setattr(travel, "TRAVEL_ACTIVE_WINDOW", (same_time, same_time))
    window = travel.get_travel_active_window()
    assert window == (same_time, same_time)
    assert travel.is_travel_screen_active(now=dt.time(5, 0))
    assert travel.is_travel_screen_active(now=dt.time(23, 0))


def test_invalid_window_defaults_to_active(monkeypatch):
    monkeypatch.setattr(travel, "TRAVEL_ACTIVE_WINDOW", ("invalid", "value"))
    assert travel.get_travel_active_window() is None
    assert travel.is_travel_screen_active(now=dt.time(10, 0))
