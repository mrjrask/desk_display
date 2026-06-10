"""Best-effort system health snapshots for display heartbeats.

The helpers in this module are intentionally defensive: Raspberry Pi specific
files and commands are optional, and failures simply omit the unavailable metric
so callers can keep writing heartbeat data on non-Pi development machines.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_CACHE_TTL_SECONDS = 45.0
_THERMAL_ZONE0_TEMP = Path("/sys/class/thermal/thermal_zone0/temp")
_MEMINFO = Path("/proc/meminfo")

_cache_key: Optional[str] = None
_cache_timestamp: float = 0.0
_cache_value: Dict[str, Any] = {}


def _read_cpu_temperature_c(thermal_path: Path = _THERMAL_ZONE0_TEMP) -> Optional[float]:
    """Return CPU temperature in Celsius when the Linux thermal file exists."""

    try:
        raw = thermal_path.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        return round(int(raw) / 1000.0, 1)
    except (OSError, TypeError, ValueError):
        return None


def _parse_meminfo_bytes(meminfo_path: Path = _MEMINFO) -> Dict[str, int]:
    """Return selected /proc/meminfo values in bytes when available."""

    keys = {
        "MemTotal": "total_bytes",
        "MemAvailable": "available_bytes",
        "MemFree": "free_bytes",
    }
    values: Dict[str, int] = {}
    try:
        lines = meminfo_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values

    for line in lines:
        name, separator, rest = line.partition(":")
        if not separator or name not in keys:
            continue
        parts = rest.strip().split()
        if not parts:
            continue
        try:
            amount = int(parts[0])
        except ValueError:
            continue
        multiplier = 1024 if len(parts) < 2 or parts[1].lower() == "kb" else 1
        values[keys[name]] = amount * multiplier
    return values


def _read_vcgencmd_throttled() -> Optional[Dict[str, Any]]:
    """Run vcgencmd get_throttled when vcgencmd is installed."""

    executable = shutil.which("vcgencmd")
    if not executable:
        return None

    try:
        result = subprocess.run(
            [executable, "get_throttled"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    raw = result.stdout.strip()
    if not raw:
        return None

    status: Dict[str, Any] = {"raw": raw}
    prefix = "throttled="
    if raw.startswith(prefix):
        status["flags_hex"] = raw[len(prefix) :]
    return status


def _read_disk_usage(storage_path: Path) -> Optional[Dict[str, Any]]:
    """Return disk usage for the configured storage path."""

    try:
        usage = shutil.disk_usage(storage_path)
    except OSError:
        return None

    return {
        "path": str(storage_path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def read_system_health(
    storage_path: str | Path,
    *,
    thermal_path: Path = _THERMAL_ZONE0_TEMP,
    meminfo_path: Path = _MEMINFO,
) -> Dict[str, Any]:
    """Read currently available system health metrics without raising."""

    health: Dict[str, Any] = {}
    storage = Path(storage_path)

    throttled = _read_vcgencmd_throttled()
    if throttled:
        health["vcgencmd_get_throttled"] = throttled

    cpu_temperature_c = _read_cpu_temperature_c(thermal_path)
    if cpu_temperature_c is not None:
        health["cpu_temperature_c"] = cpu_temperature_c

    memory = _parse_meminfo_bytes(meminfo_path)
    if memory:
        health["memory"] = memory

    disk = _read_disk_usage(storage)
    if disk:
        health["disk"] = disk

    return health


def get_system_health(
    storage_path: str | Path,
    *,
    ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
    thermal_path: Path = _THERMAL_ZONE0_TEMP,
    meminfo_path: Path = _MEMINFO,
) -> Dict[str, Any]:
    """Return a cached best-effort system health snapshot."""

    global _cache_key, _cache_timestamp, _cache_value

    now = time.monotonic()
    cache_key = str(Path(storage_path))
    if (
        _cache_key == cache_key
        and ttl_seconds > 0
        and now - _cache_timestamp < ttl_seconds
    ):
        return dict(_cache_value)

    health = read_system_health(
        storage_path,
        thermal_path=thermal_path,
        meminfo_path=meminfo_path,
    )
    _cache_key = cache_key
    _cache_timestamp = now
    _cache_value = dict(health)
    return health


def clear_system_health_cache() -> None:
    """Clear cached health data, primarily for tests."""

    global _cache_key, _cache_timestamp, _cache_value

    _cache_key = None
    _cache_timestamp = 0.0
    _cache_value = {}
