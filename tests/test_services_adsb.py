import datetime

from services.adsb import (
    AdsbDevice,
    AdsbStore,
    AircraftSighting,
    PollResult,
    _extract_messages_total,
    _parse_aircraft,
    haversine_distance,
    today_key,
)

UTC = datetime.UTC


def _store(tmp_path) -> AdsbStore:
    return AdsbStore(db_path=str(tmp_path / "adsb_test.db"))


def test_haversine_distance_zero_for_same_point():
    assert haversine_distance(41.0, -87.0, 41.0, -87.0) == 0


def test_haversine_distance_nm_vs_mi():
    nm = haversine_distance(41.0, -87.0, 41.5, -87.5, unit="nm")
    mi = haversine_distance(41.0, -87.0, 41.5, -87.5, unit="mi")
    assert mi > nm  # a mile is shorter than a nautical mile, so more of them fit


def test_parse_aircraft_computes_distance_and_skips_missing_hex():
    raw = [
        {"hex": "abc123", "flight": " UAL123  ", "alt_baro": 35000, "lat": 41.9, "lon": -87.6},
        {"flight": "NOHEX", "lat": 41.9, "lon": -87.6},
        {"hex": "def456", "alt_baro": "ground"},
    ]
    sightings = _parse_aircraft(raw, home_lat=41.88, home_lon=-87.63, unit="nm")
    assert len(sightings) == 2
    first = sightings[0]
    assert first.hex == "abc123"
    assert first.callsign == "UAL123"
    assert first.altitude_ft == 35000
    assert first.distance_nm is not None and first.distance_nm > 0

    second = sightings[1]
    assert second.hex == "def456"
    assert second.altitude_ft == 0
    assert second.distance_nm is None  # no lat/lon reported


def test_extract_messages_total_prefers_total_section():
    assert _extract_messages_total({"total": {"messages": 500}, "messages": 10}) == 500
    assert _extract_messages_total({"messages": 10}) == 10
    assert _extract_messages_total({}) is None
    assert _extract_messages_total(None) is None


def test_today_key_uses_provided_timezone():
    tz = datetime.timezone(datetime.timedelta(hours=-5))
    ts = datetime.datetime(2026, 8, 17, 23, 30, tzinfo=tz).timestamp()
    assert today_key(tz, now=ts) == "2026-08-17"


def test_record_poll_and_daily_stats_dedupe_across_devices(tmp_path):
    store = _store(tmp_path)
    day = "2026-08-17"
    device_a = AdsbDevice(host="1.2.3.4", label="Receiver 1")
    device_b = AdsbDevice(host="1.2.3.5", label="Receiver 2")

    shared = AircraftSighting(hex="aaa111", callsign="SHARED1", distance_nm=10.0, altitude_ft=30000)
    only_a = AircraftSighting(hex="bbb222", callsign="ONLYA", distance_nm=5.0, altitude_ft=10000)

    store.record_poll(
        PollResult(device=device_a, ok=True, error=None, sightings=(shared, only_a)),
        now=1000.0,
        day=day,
    )
    store.record_poll(
        PollResult(device=device_b, ok=True, error=None, sightings=(shared,)),
        now=1001.0,
        day=day,
    )

    stats = store.compute_daily_stats(day=day, tz=UTC)
    assert stats.total_by_device == {"Receiver 1": 2, "Receiver 2": 1}
    assert stats.total_combined == 2  # shared aircraft counted once combined


def test_furthest_catch_tracks_max_distance_and_time(tmp_path):
    store = _store(tmp_path)
    day = "2026-08-17"
    device = AdsbDevice(host="1.2.3.4", label="Receiver 1")

    near = AircraftSighting(hex="aaa111", callsign="NEAR1", distance_nm=10.0, altitude_ft=20000)
    far = AircraftSighting(hex="bbb222", callsign="FAR1", distance_nm=99.5, altitude_ft=40000)

    store.record_poll(
        PollResult(device=device, ok=True, error=None, sightings=(near,)), now=1000.0, day=day
    )
    store.record_poll(
        PollResult(device=device, ok=True, error=None, sightings=(far,)), now=2000.0, day=day
    )

    stats = store.compute_daily_stats(day=day, tz=UTC)
    assert stats.furthest is not None
    assert stats.furthest.hex == "bbb222"
    assert stats.furthest.callsign == "FAR1"
    assert stats.furthest.distance_nm == 99.5
    assert stats.furthest.seen_at == 2000.0
    assert stats.highest_altitude_ft == 40000


def test_all_time_furthest_persists_across_days_and_only_grows(tmp_path):
    store = _store(tmp_path)
    device = AdsbDevice(host="1.2.3.4", label="Receiver 1")

    day1 = AircraftSighting(hex="aaa111", callsign="DAY1", distance_nm=50.0, altitude_ft=30000)
    store.record_poll(
        PollResult(device=device, ok=True, error=None, sightings=(day1,)),
        now=1000.0,
        day="2026-08-16",
    )
    stats_day1 = store.compute_daily_stats(day="2026-08-16", tz=UTC)
    assert stats_day1.all_time_furthest.distance_nm == 50.0

    # A smaller catch the next day should not overwrite the all-time record.
    day2_small = AircraftSighting(
        hex="bbb222", callsign="DAY2SMALL", distance_nm=20.0, altitude_ft=15000
    )
    store.record_poll(
        PollResult(device=device, ok=True, error=None, sightings=(day2_small,)),
        now=90000.0,
        day="2026-08-17",
    )
    stats_day2 = store.compute_daily_stats(day="2026-08-17", tz=UTC)
    assert stats_day2.all_time_furthest.distance_nm == 50.0
    assert stats_day2.all_time_furthest.hex == "aaa111"

    # A bigger catch should overwrite it.
    day2_big = AircraftSighting(
        hex="ccc333", callsign="DAY2BIG", distance_nm=75.0, altitude_ft=35000
    )
    store.record_poll(
        PollResult(device=device, ok=True, error=None, sightings=(day2_big,)),
        now=91000.0,
        day="2026-08-17",
    )
    stats_day2b = store.compute_daily_stats(day="2026-08-17", tz=UTC)
    assert stats_day2b.all_time_furthest.distance_nm == 75.0
    assert stats_day2b.all_time_furthest.hex == "ccc333"


def test_busiest_hour_counts_unique_aircraft_first_seen_per_hour(tmp_path):
    store = _store(tmp_path)
    day = "2026-08-17"
    device = AdsbDevice(host="1.2.3.4", label="Receiver 1")

    hour_15 = datetime.datetime(2026, 8, 17, 15, 0, tzinfo=UTC).timestamp()
    hour_16 = datetime.datetime(2026, 8, 17, 16, 0, tzinfo=UTC).timestamp()

    for i in range(3):
        sighting = AircraftSighting(
            hex=f"h15{i}", callsign=None, distance_nm=None, altitude_ft=None
        )
        store.record_poll(
            PollResult(device=device, ok=True, error=None, sightings=(sighting,)),
            now=hour_15 + i,
            day=day,
        )
    sighting = AircraftSighting(hex="h16a", callsign=None, distance_nm=None, altitude_ft=None)
    store.record_poll(
        PollResult(device=device, ok=True, error=None, sightings=(sighting,)), now=hour_16, day=day
    )

    stats = store.compute_daily_stats(day=day, tz=UTC)
    assert stats.busiest_hour_combined == (15, 3)
    assert stats.busiest_hour_by_device["Receiver 1"] == (15, 3)


def test_failed_poll_marks_device_offline_without_crashing(tmp_path):
    store = _store(tmp_path)
    device = AdsbDevice(host="1.2.3.4", label="Receiver 1")

    store.record_poll(
        PollResult(device=device, ok=False, error="timeout"), now=1000.0, day="2026-08-17"
    )

    stats = store.compute_daily_stats(day="2026-08-17", tz=UTC)
    assert stats.device_online == {"Receiver 1": False}
    assert stats.device_errors == {"Receiver 1": "timeout"}
    assert stats.total_combined == 0


def test_messages_today_uses_first_poll_of_day_as_baseline(tmp_path):
    store = _store(tmp_path)
    device = AdsbDevice(host="1.2.3.4", label="Receiver 1")
    day = "2026-08-17"

    store.record_poll(
        PollResult(device=device, ok=True, error=None, sightings=(), messages_total=1000),
        now=1000.0,
        day=day,
    )
    store.record_poll(
        PollResult(device=device, ok=True, error=None, sightings=(), messages_total=1500),
        now=2000.0,
        day=day,
    )

    stats = store.compute_daily_stats(day=day, tz=UTC)
    assert stats.messages_today_by_device == {"Receiver 1": 500}


def test_currently_tracked_reflects_latest_poll_snapshot(tmp_path):
    store = _store(tmp_path)
    device_a = AdsbDevice(host="1.2.3.4", label="Receiver 1")
    device_b = AdsbDevice(host="1.2.3.5", label="Receiver 2")
    day = "2026-08-17"

    shared = AircraftSighting(hex="aaa111", callsign=None, distance_nm=None, altitude_ft=None)
    only_b = AircraftSighting(hex="bbb222", callsign=None, distance_nm=None, altitude_ft=None)

    store.record_poll(
        PollResult(device=device_a, ok=True, error=None, sightings=(shared,)), now=1000.0, day=day
    )
    store.record_poll(
        PollResult(device=device_b, ok=True, error=None, sightings=(shared, only_b)),
        now=1001.0,
        day=day,
    )

    stats = store.compute_daily_stats(day=day, tz=UTC)
    assert stats.currently_tracked_by_device == {"Receiver 1": 1, "Receiver 2": 2}
    assert stats.currently_tracked_combined == 2  # deduped across devices


def test_prune_removes_old_sightings_but_keeps_all_time_record(tmp_path):
    store = _store(tmp_path)
    device = AdsbDevice(host="1.2.3.4", label="Receiver 1")

    old_sighting = AircraftSighting(hex="old1", callsign="OLD", distance_nm=80.0, altitude_ft=30000)
    store.record_poll(
        PollResult(device=device, ok=True, error=None, sightings=(old_sighting,)),
        now=1000.0,
        day="2026-08-01",
    )

    removed = store.prune(7, now_day=datetime.date(2026, 8, 17))
    assert removed == 1

    old_day_stats = store.compute_daily_stats(day="2026-08-01", tz=UTC)
    assert old_day_stats.total_combined == 0
    assert old_day_stats.all_time_furthest is not None
    assert old_day_stats.all_time_furthest.distance_nm == 80.0
