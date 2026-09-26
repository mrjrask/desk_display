"""A display client applies its own dark hours and backlight settings."""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("flask")

from display_client import DARK_POLL_SECONDS, DarkHours  # noqa: E402
import test_display_client  # noqa: E402

env = test_display_client.env  # the shared client/server fixture
synced = test_display_client.synced

CHICAGO = ZoneInfo("America/Chicago")


class Now:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value

    def local(self, *args, zone=CHICAGO):
        self.value = dt.datetime(*args, tzinfo=zone)


class Backlit:
    def __init__(self, presenter):
        self.inner = presenter
        self.levels = []
        self.frames = presenter.frames

    def present(self, image):
        return self.inner.present(image)

    def set_backlight(self, level):
        self.levels.append(level)
        return level


def client_with(env, dark):
    env.publish("date")
    env.publish("weather1")
    client = synced(env, env.make_client())
    client.presenter = Backlit(client.presenter)
    client.dark_hours = dark
    return client


def test_off_mode_blanks_the_panel_and_reports_dark(env):
    now = Now(None)
    now.local(2026, 7, 1, 23, 0)  # Wednesday night
    client = client_with(env, DarkHours("Mon-Sun 22:00-06:30", mode="off", level=80, zone=CHICAGO, now=now))
    screen, seconds = client.step()
    assert screen is None and seconds == DARK_POLL_SECONDS
    assert client.report.playback_state == "dark" and client.report.current_screen is None
    assert client.presenter.levels == [0.0]
    frame = client.presenter.frames[-1]
    assert frame.getbbox() is None  # all black
    shown = len(client.presenter.frames)
    client.step()
    assert len(client.presenter.frames) == shown  # blanked once, not every pass
    # Across midnight it stays dark, then lifts in the morning.
    now.local(2026, 7, 2, 6, 29)
    assert client.step()[0] is None
    now.local(2026, 7, 2, 6, 30)
    assert client.step()[0] in {"date", "weather1"}
    assert client.presenter.levels == [0.0, 0.8]
    assert client.report.playback_state == "playing"


def test_dim_mode_keeps_playing_at_the_dark_level(env):
    now = Now(None)
    now.local(2026, 7, 1, 12, 0)
    client = client_with(env, DarkHours("Wed 11:00-13:00", mode="dim", level=100, dark_level=15,
                                        zone=CHICAGO, now=now))
    assert client.step()[0] in {"date", "weather1"}
    now.local(2026, 7, 1, 13, 0)
    client.step()
    assert client.presenter.levels == [0.15, 1.0]


def test_without_dark_hours_the_configured_level_applies_once(env):
    client = client_with(env, DarkHours(level=60))
    client.step()
    client.step()
    assert client.presenter.levels == [0.6]


def test_dark_hours_follow_wall_clock_across_dst():
    # Spring forward in Chicago: 2026-03-08 02:00 -> 03:00.
    now = Now(None)
    dark = DarkHours("Sun 01:00-03:30", zone=CHICAGO, now=now)
    now.value = dt.datetime(2026, 3, 8, 8, 15, tzinfo=dt.timezone.utc)  # 03:15 CDT
    assert dark.state() == "dark"
    now.value = dt.datetime(2026, 3, 8, 8, 30, tzinfo=dt.timezone.utc)  # 03:30 CDT
    assert dark.state() == "normal"
    # Fall back: 2026-11-01 02:00 CDT -> 01:00 CST; 01:30 happens twice.
    dark = DarkHours("Sun 01:00-02:00", zone=CHICAGO, now=now)
    for utc_hour in (6, 7):
        now.value = dt.datetime(2026, 11, 1, utc_hour, 30, tzinfo=dt.timezone.utc)
        assert dark.state() == "dark"


def test_from_settings_reads_the_client_settings():
    now = Now(dt.datetime(2026, 7, 1, 4, 0, tzinfo=dt.timezone.utc))  # 05:00 London, 23:00 Chicago
    settings = {
        "DARK_HOURS": "Mon-Sun 22:00-06:00",
        "DESK_DISPLAY_DARK_HOURS_MODE": "dim",
        "DESK_DISPLAY_BACKLIGHT_LEVEL": 0,
        "DESK_DISPLAY_DARK_HOURS_BACKLIGHT_LEVEL": 5,
        "DESK_DISPLAY_CONTENT_TIMEZONE": "Europe/London",
    }
    dark = DarkHours.from_settings(settings, now=now)
    assert (dark.mode, dark.level, dark.dark_level, dark.zone.key) == ("dim", 0, 5, "Europe/London")
    assert dark.state() == "dim" and dark.backlight("dim") == 0.05 and dark.backlight("normal") == 0.0
    now.value = dt.datetime(2026, 7, 1, 12, 0, tzinfo=dt.timezone.utc)
    assert dark.state() == "normal"
    assert DarkHours.from_settings({}).state() == "normal"


def test_a_dark_hours_boundary_ends_a_long_hold(env):
    now = Now(None)
    now.local(2026, 7, 1, 21, 59)
    client = client_with(env, DarkHours("Mon-Sun 22:00-06:00", zone=CHICAGO, now=now))
    ticks = iter(range(10_000))
    client._monotonic = lambda: float(next(ticks))
    client._stop.wait = lambda _seconds: now.local(2026, 7, 1, 22, 0)  # time passes during the hold
    assert client.step()[0] in {"date", "weather1"}
    client.wait(3600)
    assert next(ticks) < 10  # returned at the boundary, not after an hour
    assert client.step()[0] is None and client.report.playback_state == "dark"
