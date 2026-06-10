from pathlib import Path
from types import SimpleNamespace

from services import system_health


def test_read_system_health_omits_missing_pi_metrics(monkeypatch, tmp_path):
    thermal_path = tmp_path / "missing-temp"
    meminfo_path = tmp_path / "missing-meminfo"

    monkeypatch.setattr(system_health.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        system_health.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=1000, used=250, free=750),
    )

    health = system_health.read_system_health(
        tmp_path,
        thermal_path=thermal_path,
        meminfo_path=meminfo_path,
    )

    assert "vcgencmd_get_throttled" not in health
    assert "cpu_temperature_c" not in health
    assert "memory" not in health
    assert health["disk"] == {
        "path": str(tmp_path),
        "total_bytes": 1000,
        "used_bytes": 250,
        "free_bytes": 750,
    }


def test_read_system_health_collects_available_metrics(monkeypatch, tmp_path):
    thermal_path = tmp_path / "temp"
    thermal_path.write_text("45678\n", encoding="utf-8")
    meminfo_path = tmp_path / "meminfo"
    meminfo_path.write_text(
        "MemTotal:       1024 kB\nMemFree:         128 kB\nMemAvailable:    512 kB\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(system_health.shutil, "which", lambda _name: "/usr/bin/vcgencmd")
    monkeypatch.setattr(
        system_health.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="throttled=0x0\n"),
    )
    monkeypatch.setattr(
        system_health.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=2000, used=500, free=1500),
    )

    health = system_health.read_system_health(
        tmp_path,
        thermal_path=thermal_path,
        meminfo_path=meminfo_path,
    )

    assert health["vcgencmd_get_throttled"] == {"raw": "throttled=0x0", "flags_hex": "0x0"}
    assert health["cpu_temperature_c"] == 45.7
    assert health["memory"] == {
        "total_bytes": 1024 * 1024,
        "free_bytes": 128 * 1024,
        "available_bytes": 512 * 1024,
    }
    assert health["disk"]["free_bytes"] == 1500


def test_get_system_health_caches_reads(monkeypatch, tmp_path):
    system_health.clear_system_health_cache()
    calls = []

    def _read(path, **kwargs):
        calls.append(Path(path))
        return {"disk": {"path": str(path), "free_bytes": len(calls)}}

    monkeypatch.setattr(system_health, "read_system_health", _read)
    monkeypatch.setattr(system_health.time, "monotonic", lambda: 100.0)

    first = system_health.get_system_health(tmp_path, ttl_seconds=45)
    second = system_health.get_system_health(tmp_path, ttl_seconds=45)

    assert first == second
    assert len(calls) == 1

    monkeypatch.setattr(system_health.time, "monotonic", lambda: 200.0)
    third = system_health.get_system_health(tmp_path, ttl_seconds=45)

    assert third["disk"]["free_bytes"] == 2
    assert len(calls) == 2
