import datetime

from data_fetch import _is_night_time_hourly
from config import CENTRAL_TIME


def _to_ts(year: int, month: int, day: int, hour: int) -> int:
    local_dt = CENTRAL_TIME.localize(datetime.datetime(year, month, day, hour, 0, 0))
    return int(local_dt.timestamp())


def test_hourly_night_cutover_during_daylight_saving_time():
    # July in US Central Time is DST.
    assert _is_night_time_hourly(_to_ts(2026, 7, 1, 17), None, None, increment_seconds=3600) is False
    assert _is_night_time_hourly(_to_ts(2026, 7, 1, 18), None, None, increment_seconds=3600) is True


def test_hourly_night_cutover_during_standard_time():
    # January in US Central Time is standard time.
    assert _is_night_time_hourly(_to_ts(2026, 1, 10, 16), None, None, increment_seconds=3600) is False
    assert _is_night_time_hourly(_to_ts(2026, 1, 10, 17), None, None, increment_seconds=3600) is True


def test_hourly_switches_back_to_daytime_after_5am():
    assert _is_night_time_hourly(_to_ts(2026, 1, 10, 4), None, None, increment_seconds=3600) is True
    assert _is_night_time_hourly(_to_ts(2026, 1, 10, 5), None, None, increment_seconds=3600) is False
