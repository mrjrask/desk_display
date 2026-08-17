import config
from screens.draw_adsb_stats import (
    _hour_range_label,
    draw_adsb_stats_screen,
)
from services.adsb import DailyStats, FurthestCatch
from utils import ScreenImage


def _stats(**overrides) -> DailyStats:
    base = {
        "day": "2026-08-17",
        "total_combined": 12,
        "total_by_device": {"Receiver 1": 8, "Receiver 2": 6},
        "furthest": FurthestCatch(
            hex="abc123",
            callsign="UAL123",
            device="Receiver 1",
            distance_nm=42.7,
            seen_at=1755000000.0,
        ),
        "busiest_hour_combined": (15, 9),
        "busiest_hour_by_device": {"Receiver 1": (15, 6)},
        "highest_altitude_ft": 38000,
        "messages_today_by_device": {"Receiver 1": 128500},
        "currently_tracked_combined": 4,
        "currently_tracked_by_device": {"Receiver 1": 3, "Receiver 2": 1},
        "device_online": {"Receiver 1": True, "Receiver 2": True},
        "all_time_furthest": None,
        "device_errors": {},
    }
    base.update(overrides)
    return DailyStats(**base)


def test_hour_range_label_compact_when_same_period():
    assert _hour_range_label(15) == "3–4 PM"


def test_hour_range_label_expands_across_noon():
    assert _hour_range_label(11) == "11 AM–12 PM"


def test_hour_range_label_expands_across_midnight():
    assert _hour_range_label(23) == "11 PM–12 AM"


def test_draw_with_stats_returns_screen_image():
    result = draw_adsb_stats_screen(None, stats=_stats())
    assert isinstance(result, ScreenImage)
    assert result.image.size == (config.WIDTH, config.HEIGHT)


def test_draw_no_data_state_when_total_is_zero():
    empty_stats = _stats(
        total_combined=0,
        total_by_device={},
        furthest=None,
        busiest_hour_combined=None,
        busiest_hour_by_device={},
        highest_altitude_ft=None,
        messages_today_by_device={},
        currently_tracked_combined=0,
        currently_tracked_by_device={},
    )
    result = draw_adsb_stats_screen(None, stats=empty_stats)
    assert isinstance(result, ScreenImage)
    assert result.image.size == (config.WIDTH, config.HEIGHT)


def test_draw_no_data_state_without_any_stats_object(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_ADSB", False)
    result = draw_adsb_stats_screen(None, stats=None)
    assert isinstance(result, ScreenImage)
    assert result.image.size == (config.WIDTH, config.HEIGHT)


def test_draw_renders_all_time_best_row_when_different_from_todays_furthest():
    stats = _stats(
        all_time_furthest=FurthestCatch(
            hex="def456",
            callsign="DAL456",
            device="Receiver 2",
            distance_nm=61.2,
            seen_at=1740000000.0,
        )
    )
    result = draw_adsb_stats_screen(None, stats=stats)
    assert isinstance(result, ScreenImage)
