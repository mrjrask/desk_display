"""DESK_DISPLAY_CONTENT_TIMEZONE drives dates, clocks and dark hours."""
from __future__ import annotations

import datetime as dt

import pytest

import display_time
from display_time import CENTRAL_TIME, content_timezone_name


@pytest.fixture
def zone(monkeypatch):
    def use(name):
        if name is None:
            monkeypatch.delenv("DESK_DISPLAY_CONTENT_TIMEZONE", raising=False)
        else:
            monkeypatch.setenv("DESK_DISPLAY_CONTENT_TIMEZONE", name)
        CENTRAL_TIME.reset()
        return CENTRAL_TIME

    yield use
    monkeypatch.delenv("DESK_DISPLAY_CONTENT_TIMEZONE", raising=False)
    CENTRAL_TIME.reset()


def test_default_and_invalid_names_fall_back_to_chicago():
    assert content_timezone_name({}) == "America/Chicago"
    assert content_timezone_name({"DESK_DISPLAY_CONTENT_TIMEZONE": " "}) == "America/Chicago"
    assert content_timezone_name({"DESK_DISPLAY_CONTENT_TIMEZONE": "Mars/Base"}) == "America/Chicago"
    assert content_timezone_name({"DESK_DISPLAY_CONTENT_TIMEZONE": "Europe/London"}) == "Europe/London"


@pytest.mark.parametrize("name, expected_hour", [("America/Chicago", 7), ("Europe/London", 13),
                                                 ("Asia/Tokyo", 21)])
def test_content_zone_follows_the_setting(zone, name, expected_hour):
    tz = zone(name)
    moment = dt.datetime(2026, 7, 1, 12, tzinfo=dt.timezone.utc)
    assert moment.astimezone(tz).hour == expected_hour
    assert tz.key == name
    assert display_time.CONTENT_TIME is tz


def test_content_zone_handles_dst(zone):
    tz = zone("America/New_York")
    assert tz.localize(dt.datetime(2026, 1, 15, 12)).utcoffset() == dt.timedelta(hours=-5)
    assert tz.localize(dt.datetime(2026, 7, 15, 12)).utcoffset() == dt.timedelta(hours=-4)
    # Across the spring-forward boundary (02:00 local on 2026-03-08).
    before = dt.datetime(2026, 3, 8, 6, 59, tzinfo=dt.timezone.utc).astimezone(tz)
    after = dt.datetime(2026, 3, 8, 7, 0, tzinfo=dt.timezone.utc).astimezone(tz)
    assert (before.hour, after.hour) == (1, 3)


def test_clock_packages_carry_the_configured_zone(zone):
    from display_profiles import PROFILE_PRESETS
    from rendering.clock_faces import clock_layout

    zone("Europe/Berlin")
    assert clock_layout("date", PROFILE_PRESETS["hyperpixel4"])["time_zone"] == "Europe/Berlin"


def test_standalone_dark_hours_use_the_content_zone(zone, monkeypatch):
    import config

    segments = config._parse_dark_hours_spec("Mon-Sun 22:00-06:00")
    monkeypatch.setattr(config, "DARK_HOURS_SEGMENTS", segments)
    moment = dt.datetime(2026, 7, 1, 4, tzinfo=dt.timezone.utc)  # 23:00 Chicago, 05:00 London
    zone("America/Chicago")
    assert config.is_within_dark_hours(moment)
    zone("Asia/Tokyo")  # 13:00
    assert not config.is_within_dark_hours(moment)


def test_render_preferences_change_with_the_zone():
    from remote_display.server_rendering import preferences_revision

    chicago = preferences_revision({})
    assert preferences_revision({"DESK_DISPLAY_CONTENT_TIMEZONE": "America/Chicago"}) == chicago
    assert preferences_revision({"DESK_DISPLAY_CONTENT_TIMEZONE": "Europe/London"}) != chicago


def test_zone_survives_pickling(zone):
    import pickle

    tz = zone("Europe/Paris")
    assert pickle.loads(pickle.dumps(tz)).key == "Europe/Paris"
