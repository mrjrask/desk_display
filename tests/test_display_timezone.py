import datetime as dt
import os
import time

import pytest

import config
import config_ui
from screens import draw_date_time, draw_nixie, draw_wolves_schedule


@pytest.fixture(autouse=True)
def non_central_host_timezone(monkeypatch):
    """Make accidental parameterless local-time conversions visible."""
    original_tz = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    if hasattr(time, "tzset"):
        time.tzset()
    yield
    if hasattr(time, "tzset"):
        if original_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original_tz
        time.tzset()


def test_display_datetime_handles_central_dst_boundary():
    before = config.display_datetime(
        dt.datetime(2025, 3, 9, 7, 30, tzinfo=dt.timezone.utc)
    )
    after = config.display_datetime(
        dt.datetime(2025, 3, 9, 8, 30, tzinfo=dt.timezone.utc)
    )

    assert (before.strftime("%H:%M"), before.utcoffset()) == (
        "01:30",
        dt.timedelta(hours=-6),
    )
    assert (after.strftime("%H:%M"), after.utcoffset()) == (
        "03:30",
        dt.timedelta(hours=-5),
    )


def test_date_time_frame_uses_central_date_across_midnight(monkeypatch):
    instant = dt.datetime(
        2025, 1, 2, 5, 30, tzinfo=dt.timezone.utc
    )  # Jan 1, 11:30 PM CT
    seen = {}
    monkeypatch.setattr(
        draw_date_time, "display_datetime", lambda: config.display_datetime(instant)
    )
    monkeypatch.setattr(
        draw_date_time,
        "date_strings",
        lambda value: seen.setdefault("date", value) and ("Wednesday", "Jan 1"),
    )
    monkeypatch.setattr(
        draw_date_time,
        "time_strings",
        lambda value: seen.setdefault("time", value) and ("11:30", "PM"),
    )

    draw_date_time._compose_frame(
        "date_time", (255, 0, 0), (0, 255, 0), False, "date"
    )

    assert seen["date"].strftime("%Y-%m-%d %H:%M %Z") == "2025-01-01 23:30 CST"
    assert seen["time"] is seen["date"]


def test_nixie_injected_instant_is_rendered_in_central_time():
    instant = dt.datetime(
        2025, 11, 2, 7, 30, 45, tzinfo=dt.timezone.utc
    )  # 1:30:45 CST
    expected = dt.datetime(2025, 11, 2, 1, 30, 45, tzinfo=config.CENTRAL_TIME)

    assert draw_nixie.nixie_frame(instant).tobytes() == draw_nixie.nixie_frame(
        expected
    ).tobytes()


def test_wolves_labels_and_game_time_use_central_at_midnight(monkeypatch):
    now = dt.datetime(
        2025, 1, 2, 6, 15, tzinfo=dt.timezone.utc
    )  # Jan 2, 12:15 AM CT
    monkeypatch.setattr(
        draw_wolves_schedule, "display_datetime", lambda: config.display_datetime(now)
    )
    monkeypatch.setattr(draw_wolves_schedule, "_MLB_FORMAT_GAME_LABEL", None)

    assert (
        draw_wolves_schedule._format_last_date_bottom("2025-01-02T05:30:00Z")
        == "Yesterday"
    )
    assert (
        draw_wolves_schedule._format_next_bottom("", "2025-01-03T05:30:00Z")
        == "Today • 11:30 PM"
    )
    assert (
        draw_wolves_schedule._format_next_bottom("", "2025-01-03T06:30:00Z")
        == "Tomorrow • 12:30 AM"
    )


def test_config_ui_timestamp_uses_central_timezone_at_dst_boundary():
    timestamp = dt.datetime(
        2025, 3, 9, 8, 30, tzinfo=dt.timezone.utc
    ).timestamp()
    assert config_ui._format_timestamp(timestamp) == "2025-03-09 03:30:00"
