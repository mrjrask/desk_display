import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "adsb_collector", ROOT / "scripts" / "adsb_collector.py"
)
adsb_collector = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("adsb_collector", adsb_collector)
_SPEC.loader.exec_module(adsb_collector)

from services.adsb import AdsbDevice, AdsbStore, PollResult  # noqa: E402


def test_configured_devices_reads_from_config(monkeypatch):
    monkeypatch.setattr(
        adsb_collector.config,
        "ADSB_DEVICES",
        [{"host": "1.2.3.4", "label": "Attic"}, {"host": "1.2.3.5", "label": "Garage"}],
    )
    devices = adsb_collector.configured_devices()
    assert devices == [
        AdsbDevice(host="1.2.3.4", label="Attic"),
        AdsbDevice(host="1.2.3.5", label="Garage"),
    ]


def test_run_once_polls_every_device_and_records_results(tmp_path, monkeypatch):
    devices = [
        AdsbDevice(host="1.2.3.4", label="Attic"),
        AdsbDevice(host="1.2.3.5", label="Garage"),
    ]
    store = AdsbStore(db_path=str(tmp_path / "test.db"))
    recorded = []

    def _fake_poll_device(device, *, home_lat, home_lon, unit, timeout):
        if device.label == "Garage":
            return PollResult(device=device, ok=False, error="timeout")
        return PollResult(device=device, ok=True, error=None, sightings=(), messages_total=100)

    monkeypatch.setattr(adsb_collector, "poll_device", _fake_poll_device)
    original_record_poll = store.record_poll

    def _tracking_record_poll(result, *, now, day):
        recorded.append((result.device.label, result.ok))
        return original_record_poll(result, now=now, day=day)

    monkeypatch.setattr(store, "record_poll", _tracking_record_poll)

    adsb_collector.run_once(store, devices)

    assert recorded == [("Attic", True), ("Garage", False)]
    store.close()


def test_main_returns_error_when_no_devices_configured(monkeypatch):
    monkeypatch.setattr(adsb_collector.config, "ADSB_DEVICES", [])
    exit_code = adsb_collector.main([])
    assert exit_code == 1


def test_main_once_polls_a_single_cycle_and_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(
        adsb_collector.config,
        "ADSB_DEVICES",
        [{"host": "1.2.3.4", "label": "Attic"}],
    )
    monkeypatch.setattr(
        adsb_collector, "AdsbStore", lambda: AdsbStore(db_path=str(tmp_path / "test.db"))
    )

    calls = []

    def _fake_poll_device(device, *, home_lat, home_lon, unit, timeout):
        calls.append(device.label)
        return PollResult(device=device, ok=True, error=None, sightings=())

    monkeypatch.setattr(adsb_collector, "poll_device", _fake_poll_device)

    exit_code = adsb_collector.main(["--once"])

    assert exit_code == 0
    assert calls == ["Attic"]
