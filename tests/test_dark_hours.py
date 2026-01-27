import datetime

import pytest

import config


def test_parse_dark_hours_spec_handles_overnight_and_weekend():
    spec = "Mon-Thu 19:00-07:00; Fri-Sun 00:00-24:00"
    segments = config._parse_dark_hours_spec(spec)  # type: ignore[attr-defined]

    # Monday evening through midnight.
    assert config.DarkHoursSegment(0, 19 * 60, 24 * 60) in segments
    # Tuesday early morning wrap from the overnight entry.
    assert config.DarkHoursSegment(1, 0, 7 * 60) in segments
    # Saturday should be a full-day shutdown window.
    assert config.DarkHoursSegment(5, 0, 24 * 60) in segments


@pytest.mark.parametrize(
    "dt_value, expected",
    [
        (datetime.datetime(2024, 5, 6, 21, 30), True),   # Monday 9:30 PM
        (datetime.datetime(2024, 5, 7, 6, 45), True),    # Tuesday 6:45 AM
        (datetime.datetime(2024, 5, 8, 12, 0), False),   # Wednesday noon
        (datetime.datetime(2024, 5, 10, 15, 0), True),   # Friday afternoon
        (datetime.datetime(2024, 5, 12, 9, 0), True),    # Sunday morning
    ],
)
def test_is_within_dark_hours(monkeypatch, dt_value, expected):
    spec = "Mon-Thu 19:00-07:00; Fri-Sun 00:00-24:00"
    segments = config._parse_dark_hours_spec(spec)  # type: ignore[attr-defined]

    monkeypatch.setattr(config, "DARK_HOURS_SEGMENTS", segments)
    monkeypatch.setattr(config, "DARK_HOURS_ENABLED", bool(segments))

    tz_aware = config.CENTRAL_TIME.localize(dt_value)
    assert config.is_within_dark_hours(tz_aware) is expected

    # Also ensure naive datetimes are assumed to be Central Time.
    assert config.is_within_dark_hours(dt_value) is expected
