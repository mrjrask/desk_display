from PIL import Image, ImageDraw

import config
import screens.draw_adsb_stats as draw_adsb_stats_module
from screens.draw_adsb_stats import (
    _activity_color,
    _build_tiles,
    _fit_text,
    _hour_range_label,
    _live_now_breakdown_lines,
    _status_text,
    _tile_grid_cells,
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
        "hourly_counts_combined": {15: 9, 14: 3, 10: 2},
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


def _empty_stats(**overrides) -> DailyStats:
    empty = {
        "total_combined": 0,
        "total_by_device": {},
        "furthest": None,
        "busiest_hour_combined": None,
        "busiest_hour_by_device": {},
        "hourly_counts_combined": {},
        "highest_altitude_ft": None,
        "messages_today_by_device": {},
        "currently_tracked_combined": 0,
        "currently_tracked_by_device": {},
    }
    empty.update(overrides)
    return _stats(**empty)


def test_hour_range_label_compact_when_same_period():
    assert _hour_range_label(15) == "3–4 PM"


def test_hour_range_label_expands_across_noon():
    assert _hour_range_label(11) == "11 AM–12 PM"


def test_hour_range_label_expands_across_midnight():
    assert _hour_range_label(23) == "11 PM–12 AM"


def test_activity_color_warms_up_as_traffic_increases():
    quiet = _activity_color(0)
    busy = _activity_color(500)
    assert quiet != busy
    # Warmer (busier) should shift toward more red/green, less blue.
    assert busy[0] >= quiet[0]
    assert busy[2] <= quiet[2]


def test_activity_color_is_stable_for_repeated_calls():
    assert _activity_color(42) == _activity_color(42)


def test_status_text_single_device():
    assert _status_text(_stats(device_online={"Receiver 1": True})) == "ONLINE"
    assert _status_text(_stats(device_online={"Receiver 1": False})) == "OFFLINE"


def test_status_text_multiple_devices():
    assert (
        _status_text(_stats(device_online={"Receiver 1": True, "Receiver 2": True}))
        == "2 ONLINE"
    )
    assert (
        _status_text(_stats(device_online={"Receiver 1": True, "Receiver 2": False}))
        == "1 OF 2 ONLINE"
    )


def test_status_text_empty_when_no_devices_known(monkeypatch):
    monkeypatch.setattr(config, "ADSB_DEVICES", [])
    assert _status_text(_stats(device_online={})) == ""


def test_build_tiles_leads_with_furthest_and_by_receiver():
    tiles = _build_tiles(_stats())
    assert [tile["label"] for tile in tiles[:2]] == ["Furthest", "By Receiver"]
    assert tiles[0]["value"] == "42.7 nm"
    assert "UAL123" in tiles[0]["caption"]


def test_build_tiles_shows_no_position_message_without_furthest():
    tiles = _build_tiles(_stats(furthest=None))
    assert tiles[0]["value"] == "--"
    assert tiles[0]["caption"] == "No position data yet"


def test_build_tiles_caps_extras_at_two():
    tiles = _build_tiles(
        _stats(
            all_time_furthest=FurthestCatch(
                hex="def456",
                callsign="DAL456",
                device="Receiver 2",
                distance_nm=61.2,
                seen_at=1740000000.0,
            )
        )
    )
    assert len(tiles) == 4
    labels = [tile["label"] for tile in tiles]
    assert "All-Time Best" in labels
    assert "Live Now" in labels  # next-highest priority extra


def test_live_now_breakdown_lines_sorts_and_folds_overflow_into_other():
    counts = {"B738": 5, "A320": 8, "E170": 2, "CRJ7": 1, "C172": 1}
    lines = _live_now_breakdown_lines(counts, max_lines=3)
    assert lines == ["A320: 8", "B738: 5", "Other: 4"]  # E170 + CRJ7 + C172


def test_live_now_breakdown_lines_empty_when_no_models():
    assert _live_now_breakdown_lines({}) == []


def test_build_tiles_live_now_toggles_between_count_and_breakdown(monkeypatch):
    stats = _stats(
        currently_tracked_combined=5, currently_tracked_by_model={"B738": 3, "A320": 2}
    )

    monkeypatch.setattr(draw_adsb_stats_module.time, "time", lambda: 0.0)
    count_tile = next(t for t in _build_tiles(stats) if t["label"] == "Live Now")
    assert count_tile["value"] == "5"
    assert count_tile["caption"] == "aircraft in range"

    monkeypatch.setattr(
        draw_adsb_stats_module.time, "time", lambda: draw_adsb_stats_module._LIVE_NOW_CYCLE_SECONDS
    )
    breakdown_tile = next(t for t in _build_tiles(stats) if t["label"] == "Live Now")
    assert breakdown_tile["value"] == "B738: 3\nA320: 2"
    assert breakdown_tile["caption"] == "by model"


def test_build_tiles_live_now_stays_on_count_when_no_model_data(monkeypatch):
    stats = _stats(currently_tracked_by_model={})
    monkeypatch.setattr(
        draw_adsb_stats_module.time, "time", lambda: draw_adsb_stats_module._LIVE_NOW_CYCLE_SECONDS
    )
    tile = next(t for t in _build_tiles(stats) if t["label"] == "Live Now")
    assert tile["value"] == str(stats.currently_tracked_combined)
    assert tile["caption"] == "aircraft in range"


def test_build_tiles_omits_all_time_when_same_as_todays_furthest():
    todays = FurthestCatch(
        hex="abc123", callsign="UAL123", device="Receiver 1", distance_nm=42.7, seen_at=1755000000.0
    )
    tiles = _build_tiles(_stats(furthest=todays, all_time_furthest=todays))
    assert "All-Time Best" not in [tile["label"] for tile in tiles]


def test_build_tiles_falls_back_to_receiver_status_when_no_extras_available():
    tiles = _build_tiles(
        _empty_stats(
            total_combined=3,
            total_by_device={"Receiver 1": 3},
            furthest=FurthestCatch(
                hex="abc123", callsign=None, device="Receiver 1", distance_nm=5.0, seen_at=100.0
            ),
            device_online={"Receiver 1": True, "Receiver 2": False},
        )
    )
    labels = [tile["label"] for tile in tiles]
    assert "Receivers" in labels
    receivers_tile = next(tile for tile in tiles if tile["label"] == "Receivers")
    assert receivers_tile["value"] == "1/2"


def test_tile_grid_cells_covers_rect_without_overlap_for_various_counts():
    rect = (0, 0, 200, 100)
    for count in (1, 2, 3, 4):
        cells = _tile_grid_cells(rect, count)
        assert len(cells) == count
        for left, top, right, bottom in cells:
            assert rect[0] <= left < right <= rect[2]
            assert rect[1] <= top < bottom <= rect[3]


def test_fit_text_truncates_with_ellipsis_when_too_long():
    img = Image.new("RGB", (50, 20))
    draw = ImageDraw.Draw(img)
    long_text = "A" * 200
    fitted = _fit_text(draw, long_text, config.FONT_WEATHER_DETAILS_TINY, 40)
    assert fitted.endswith("…")
    assert len(fitted) < len(long_text)


def test_draw_with_stats_returns_screen_image():
    result = draw_adsb_stats_screen(None, stats=_stats())
    assert isinstance(result, ScreenImage)
    assert result.image.size == (config.WIDTH, config.HEIGHT)


def test_draw_no_data_state_when_total_is_zero():
    result = draw_adsb_stats_screen(None, stats=_empty_stats())
    assert isinstance(result, ScreenImage)
    assert result.image.size == (config.WIDTH, config.HEIGHT)


def test_draw_no_data_state_without_any_stats_object(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_ADSB", False)
    result = draw_adsb_stats_screen(None, stats=None)
    assert isinstance(result, ScreenImage)
    assert result.image.size == (config.WIDTH, config.HEIGHT)


def test_draw_no_data_state_shows_device_error_detail():
    result = draw_adsb_stats_screen(
        None,
        stats=_empty_stats(device_errors={"Receiver 1": "HTTP 404 (tried 3 known paths)"}),
    )
    assert isinstance(result, ScreenImage)


def test_draw_renders_all_time_best_tile_when_different_from_todays_furthest():
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


def test_draw_no_data_state_still_shows_all_time_best_tile():
    result = draw_adsb_stats_screen(
        None,
        stats=_empty_stats(
            all_time_furthest=FurthestCatch(
                hex="def456",
                callsign="DAL456",
                device="Receiver 2",
                distance_nm=61.2,
                seen_at=1740000000.0,
            )
        ),
    )
    assert isinstance(result, ScreenImage)


def test_draw_handles_extremely_long_callsign_and_device_labels_without_error():
    """Regression guard for the wrapping bug: absurdly long strings must
    shrink/truncate to a single line rather than raise or overflow."""

    stats = _stats(
        furthest=FurthestCatch(
            hex="abc123",
            callsign="SUPERLONGCALLSIGN123456789",
            device="A Very Long Receiver Label Indeed",
            distance_nm=1234.5,
            seen_at=1755000000.0,
        ),
        total_by_device={
            "A Very Long Receiver Label Indeed": 999,
            "Another Extremely Long Receiver Label": 888,
        },
    )
    result = draw_adsb_stats_screen(None, stats=stats)
    assert isinstance(result, ScreenImage)
    assert result.image.size == (config.WIDTH, config.HEIGHT)
