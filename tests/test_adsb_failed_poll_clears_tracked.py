import datetime

from services.adsb import AdsbDevice, AdsbStore, AircraftSighting, PollResult

UTC = datetime.UTC


def _store(tmp_path) -> AdsbStore:
    return AdsbStore(db_path=str(tmp_path / "adsb_test.db"))


def test_failed_poll_clears_currently_tracked(tmp_path):
    # Regression: a failed poll (ok=False) only updated `online`/`error` and
    # left tracked_hexes/tracked_types/tracked_callsigns at whatever the last
    # successful poll wrote, so compute_daily_stats kept reporting the same
    # aircraft as "currently tracked" indefinitely once a receiver went
    # offline.
    store = _store(tmp_path)
    day = "2026-08-17"
    device = AdsbDevice(host="1.2.3.4", label="Receiver 1")

    seen = AircraftSighting(hex="aaa111", callsign="UAL1", distance_nm=10.0, altitude_ft=30000)
    store.record_poll(
        PollResult(device=device, ok=True, error=None, sightings=(seen,)), now=1000.0, day=day
    )

    stats = store.compute_daily_stats(day=day, tz=UTC)
    assert stats.currently_tracked_by_device["Receiver 1"] == 1

    store.record_poll(
        PollResult(device=device, ok=False, error="timeout", sightings=()), now=1010.0, day=day
    )

    stats = store.compute_daily_stats(day=day, tz=UTC)
    assert stats.currently_tracked_by_device["Receiver 1"] == 0
    assert stats.device_online["Receiver 1"] is False
    assert stats.device_errors["Receiver 1"] == "timeout"
