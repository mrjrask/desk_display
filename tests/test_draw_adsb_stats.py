from PIL import Image, ImageDraw

import config
import screens.draw_adsb_stats as draw_adsb_stats_module
from screens.draw_adsb_stats import (
    _activity_color,
    _airline_logo,
    _airline_logo_path,
    _build_tiles,
    _fit_text,
    _hour_range_label,
    _live_now_breakdown_lines,
    _status_text,
    _tile_grid_cells,
    _top_breakdown_items,
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


def test_build_tiles_leads_with_furthest_and_by_receiver_on_every_variant():
    for variant in ("best", "live", "live airlines"):
        tiles = _build_tiles(_stats(), variant)
        assert [tile["label"] for tile in tiles[:2]] == ["Furthest", "By Receiver"]
    tiles = _build_tiles(_stats(), "best")
    assert tiles[0]["value"] == "42.7 nm"
    assert "UAL123" in tiles[0]["caption"]


def test_build_tiles_by_receiver_carries_online_status_per_line():
    tiles = _build_tiles(
        _stats(device_online={"Receiver 1": True, "Receiver 2": False}), "best"
    )
    by_receiver = next(tile for tile in tiles if tile["label"] == "By Receiver")
    assert by_receiver["value"] == "Receiver 1: 8\nReceiver 2: 6"
    assert by_receiver["line_online"] == [True, False]


def test_build_tiles_shows_no_position_message_without_furthest():
    tiles = _build_tiles(_stats(furthest=None), "best")
    assert tiles[0]["value"] == "--"
    assert tiles[0]["caption"] == "No position data yet"


def test_build_tiles_best_variant_shows_all_time_and_messages():
    tiles = _build_tiles(
        _stats(
            all_time_furthest=FurthestCatch(
                hex="def456",
                callsign="DAL456",
                device="Receiver 2",
                distance_nm=61.2,
                seen_at=1740000000.0,
            )
        ),
        "best",
    )
    assert len(tiles) == 4
    labels = [tile["label"] for tile in tiles]
    assert labels == ["Furthest", "By Receiver", "All-Time Best", "Messages"]


def test_build_tiles_best_variant_omits_all_time_when_same_as_todays_furthest():
    todays = FurthestCatch(
        hex="abc123", callsign="UAL123", device="Receiver 1", distance_nm=42.7, seen_at=1755000000.0
    )
    tiles = _build_tiles(_stats(furthest=todays, all_time_furthest=todays), "best")
    assert "All-Time Best" not in [tile["label"] for tile in tiles]


def test_build_tiles_best_variant_can_render_with_only_two_tiles():
    """No all-time-best distinct from today's furthest and no messages
    today: the grid should just show Furthest + By Receiver, never a
    fallback 'Receivers' tile."""

    tiles = _build_tiles(_empty_stats(messages_today_by_device={}), "best")
    labels = [tile["label"] for tile in tiles]
    assert labels == ["Furthest", "By Receiver"]
    assert "Receivers" not in labels


def test_build_tiles_live_variant_shows_total_and_by_aircraft():
    stats = _stats(
        currently_tracked_combined=5, currently_tracked_by_model={"B738": 3, "A320": 2}
    )
    tiles = _build_tiles(stats, "live")
    labels = [tile["label"] for tile in tiles]
    assert labels == ["Furthest", "By Receiver", "Live Now", "Live Now"]
    total_tile, by_aircraft_tile = tiles[2], tiles[3]
    assert total_tile["value"] == "5"
    assert total_tile["caption"] == "aircraft in range"
    assert by_aircraft_tile["value"] == "B738: 3\nA320: 2"
    assert by_aircraft_tile["caption"] == "by aircraft"


def test_build_tiles_live_variant_omits_live_tiles_when_nothing_tracked():
    tiles = _build_tiles(_stats(currently_tracked_combined=0), "live")
    assert [tile["label"] for tile in tiles] == ["Furthest", "By Receiver"]


def test_build_tiles_live_airlines_variant_shows_by_aircraft_and_by_airline():
    stats = _stats(
        currently_tracked_combined=3,
        currently_tracked_by_model={"B738": 2, "A320": 1},
        currently_tracked_by_airline={"UAL": 2, "DAL": 1},
    )
    tiles = _build_tiles(stats, "live airlines")
    labels = [tile["label"] for tile in tiles]
    assert labels == ["Furthest", "By Receiver", "Live Now", "Live Now"]
    by_aircraft_tile, by_airline_tile = tiles[2], tiles[3]
    assert by_aircraft_tile["caption"] == "by aircraft"
    assert by_airline_tile["caption"] == "by airline"
    assert by_airline_tile["airline_rows"] == [("UAL", 2), ("DAL", 1)]


def test_build_tiles_live_airlines_variant_omitted_when_no_airline_data():
    stats = _stats(currently_tracked_combined=3, currently_tracked_by_model={"B738": 3})
    tiles = _build_tiles(stats, "live airlines")
    labels = [tile["label"] for tile in tiles]
    assert labels == ["Furthest", "By Receiver", "Live Now"]


def test_top_breakdown_items_sorts_and_folds_overflow_into_other():
    counts = {"B738": 5, "A320": 8, "E170": 2, "CRJ7": 1, "C172": 1}
    items = _top_breakdown_items(counts, max_lines=3)
    assert items == [("A320", 8), ("B738", 5), ("Other", 4)]  # E170 + CRJ7 + C172


def test_top_breakdown_items_empty_when_no_data():
    assert _top_breakdown_items({}) == []


def test_live_now_breakdown_lines_sorts_and_folds_overflow_into_other():
    counts = {"B738": 5, "A320": 8, "E170": 2, "CRJ7": 1, "C172": 1}
    lines = _live_now_breakdown_lines(counts, max_lines=3)
    assert lines == ["A320: 8", "B738: 5", "Other: 4"]  # E170 + CRJ7 + C172


def test_live_now_breakdown_lines_empty_when_no_models():
    assert _live_now_breakdown_lines({}) == []


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


def test_airline_logo_path_matches_case_insensitively(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AIR_IMAGES_DIR", str(tmp_path))
    (tmp_path / "UAL.png").write_bytes(b"placeholder")
    assert _airline_logo_path("ual") == str(tmp_path / "UAL.png")
    assert _airline_logo_path("dal") is None
    assert _airline_logo_path("") is None


def test_airline_logo_loads_and_resizes_when_file_present(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AIR_IMAGES_DIR", str(tmp_path))
    draw_adsb_stats_module._AIRLINE_LOGO_CACHE.clear()
    Image.new("RGBA", (40, 20), (10, 20, 30, 255)).save(tmp_path / "swa.png")
    logo = _airline_logo("SWA", 10)
    assert logo is not None
    assert logo.height == 10


def test_airline_logo_returns_none_when_no_file_present(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AIR_IMAGES_DIR", str(tmp_path))
    draw_adsb_stats_module._AIRLINE_LOGO_CACHE.clear()
    assert _airline_logo("ZZZ", 10) is None


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


def test_draw_live_variant_returns_screen_image():
    stats = _stats(
        currently_tracked_combined=5, currently_tracked_by_model={"B738": 3, "A320": 2}
    )
    result = draw_adsb_stats_screen(None, stats=stats, variant="live")
    assert isinstance(result, ScreenImage)
    assert result.image.size == (config.WIDTH, config.HEIGHT)


def test_draw_live_airlines_variant_falls_back_to_text_without_logo_files(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "AIR_IMAGES_DIR", str(tmp_path))
    draw_adsb_stats_module._AIRLINE_LOGO_CACHE.clear()
    stats = _stats(
        currently_tracked_combined=3,
        currently_tracked_by_model={"B738": 2, "A320": 1},
        currently_tracked_by_airline={"UAL": 2, "DAL": 1},
    )
    result = draw_adsb_stats_screen(None, stats=stats, variant="live airlines")
    assert isinstance(result, ScreenImage)
    assert result.image.size == (config.WIDTH, config.HEIGHT)


def test_draw_live_airlines_variant_pastes_logo_when_file_present(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AIR_IMAGES_DIR", str(tmp_path))
    draw_adsb_stats_module._AIRLINE_LOGO_CACHE.clear()
    Image.new("RGBA", (40, 20), (200, 30, 30, 255)).save(tmp_path / "UAL.png")
    stats = _stats(
        currently_tracked_combined=3,
        currently_tracked_by_model={"B738": 2, "A320": 1},
        currently_tracked_by_airline={"UAL": 2, "DAL": 1},
    )
    result = draw_adsb_stats_screen(None, stats=stats, variant="live airlines")
    assert isinstance(result, ScreenImage)
    assert result.image.size == (config.WIDTH, config.HEIGHT)
