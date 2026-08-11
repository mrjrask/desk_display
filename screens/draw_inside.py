#!/usr/bin/env python3
"""
draw_inside.py (RGB, 320x240)

Universal environmental sensor screen with a calmer, data-forward layout:
  • Title area with automatic sensor attribution
  • Soft temperature card with contextual descriptor
  • Responsive grid of metric cards driven entirely by the available readings
Everything is dynamically sized to stay legible on the configured canvas.
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw

import config
from paths import resolve_cache_file_path
from utils import (
    clear_display,
    clone_font,
    fit_font,
    format_voc_ohms,
    measure_text,
    temperature_color,
)

# Optional HW libs (import lazily in _probe_sensor)
try:
    import board
    import busio  # type: ignore
except Exception:  # allows non-Pi dev boxes
    board = None
    busio = None

W, H = config.WIDTH, config.HEIGHT

SensorReadings = dict[str, Optional[float]]
SensorProbeResult = tuple[str, Callable[[], SensorReadings]]
SensorProbeFn = Callable[[Any, set[int]], Optional[SensorProbeResult]]
SensorProbeName = str

_sensor_probe_cache_lock = threading.Lock()
_sensor_probe_cache: Optional[tuple[Optional[str], Optional[Callable[[], SensorReadings]]]] = None
_KNOWN_SENSOR_I2C_ADDRESSES: set[int] = {0x44, 0x45, 0x76, 0x77}
_MAX_REASONABLE_I2C_HITS = 16
_HISTORY_LIMIT = 60
_HISTORY_MAX_AGE_SECONDS = 6 * 60 * 60
_HISTORY_PATH = str(resolve_cache_file_path("INSIDE_HISTORY_PATH", "inside_history.json"))
_inside_history: dict[str, list[tuple[float, float]]] = {}
_inside_history_loaded = False
_inside_history_lock = threading.RLock()


def _parse_i2c_bus_candidates() -> tuple[int, ...]:
    """Return preferred Linux I2C bus numbers for fallback probing."""

    # Keep a universal default candidate set that covers common Pi setups:
    # - 1/2 for standard headers and legacy overlays
    # - 10/11/13/14/15 for HyperPixel accessory headers.
    raw = os.environ.get("INSIDE_I2C_BUSES", "1,2,10,11,13,14,15")
    buses: list[int] = []
    seen: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            bus_num = int(token)
        except ValueError:
            logging.debug("draw_inside: ignoring non-numeric INSIDE_I2C_BUSES token %r", token)
            continue
        if bus_num < 0 or bus_num in seen:
            continue
        seen.add(bus_num)
        buses.append(bus_num)

    if not buses:
        return (1, 2, 10, 11, 13, 14, 15)
    return tuple(buses)


def _i2cdetect_bus_has_known_sensor(bus_num: int) -> bool:
    """Return True when `i2cdetect` output indicates a supported sensor address."""

    try:
        result = subprocess.run(
            ["i2cdetect", "-y", str(bus_num)],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return False

    if result.returncode != 0:
        return False

    addresses = _parse_i2cdetect_addresses(result.stdout)
    if len(addresses) > _MAX_REASONABLE_I2C_HITS:
        logging.debug(
            "draw_inside: ignoring noisy i2cdetect bus %s (%s responding addresses)",
            bus_num,
            len(addresses),
        )
        return False

    for addr in addresses:
        if addr in _KNOWN_SENSOR_I2C_ADDRESSES:
            return True
    return False


def _parse_i2cdetect_addresses(output: str) -> set[int]:
    """Extract responding I2C addresses from `i2cdetect` table output."""

    addresses: set[int] = set()
    for line in output.splitlines():
        match = re.match(r"^\s*([0-7][0-9a-fA-F]):\s+(.*)$", line)
        if not match:
            continue
        row_base = int(match.group(1), 16)
        cells = match.group(2).split()
        for idx, cell in enumerate(cells):
            token = cell.strip().lower()
            if token in {"--", "uu"}:
                if token == "uu":
                    addresses.add(row_base + idx)
                continue
            if re.fullmatch(r"[0-9a-f]{2}", token):
                addresses.add(int(token, 16))
    return addresses


def _rank_i2c_buses(configured_buses: Sequence[int]) -> tuple[int, ...]:
    """Return configured buses ordered by detected supported sensor addresses."""

    detected_sensor_buses = [bus for bus in configured_buses if _i2cdetect_bus_has_known_sensor(bus)]
    ranked: list[int] = list(detected_sensor_buses)
    for bus_num in configured_buses:
        if bus_num not in ranked:
            ranked.append(bus_num)
    return tuple(ranked)


def _resolve_i2c_bus_number(i2c: Any) -> Optional[int]:
    """Best-effort lookup for the Linux bus number behind a Blinka I2C object."""

    for attr in ("busnum", "_busnum", "_i2c_bus"):
        value = getattr(i2c, attr, None)
        if isinstance(value, int):
            return value

    inner_i2c = getattr(i2c, "i2c", None)
    if inner_i2c is not None:
        for attr in ("busnum", "_busnum", "_i2c_bus"):
            value = getattr(inner_i2c, attr, None)
            if isinstance(value, int):
                return value

    return None

def _get_smbus_candidates(i2c: Any) -> tuple[int, ...]:
    """Return ordered Linux SMBus numbers to probe for Pimoroni drivers."""

    candidates: list[int] = []
    primary_bus = _resolve_i2c_bus_number(i2c) if i2c is not None else None
    if primary_bus is not None:
        candidates.append(primary_bus)

    for bus_num in _rank_i2c_buses(_parse_i2c_bus_candidates()):
        if bus_num not in candidates:
            candidates.append(bus_num)

    if not candidates:
        return (1,)
    return tuple(candidates)


def _prepend_vendor_sensor_drivers():
    """Prefer vendored Pimoroni sensor drivers when available."""

    repo_root = Path(__file__).resolve().parents[1]
    vendor_paths = (
        repo_root / "vendor" / "pimoroni-bme280",
        repo_root / "vendor" / "pimoroni-bme680",
        repo_root / "vendor" / "bme68x",
    )
    for vendor_path in vendor_paths:
        if vendor_path.exists():
            path_str = str(vendor_path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


_prepend_vendor_sensor_drivers()


def _normalize_sensor_name(raw: str) -> str:
    normalized = raw.strip().lower().replace("-", " ").replace("_", " ")
    tokens = [part for part in normalized.split() if part]
    return "_".join(tokens)


def _get_sensor_env_override() -> tuple[Optional[SensorProbeName], Optional[str]]:
    """Return the requested sensor driver from the environment, if provided."""

    aliases = {
        "pim_sensor_stick": "pim_sensor_stick",
        "adafruit_bme680": "adafruit_bme680",
        "pimoroni_bme680": "pimoroni_bme680",
        "pimoroni_bme68x": "pimoroni_bme68x",
        "pimoroni_bme688": "pimoroni_bme68x",
        "adafruit_sht41": "adafruit_sht41",
        "adafruit_sht4x": "adafruit_sht41",
        "pimoroni_bme280": "pimoroni_bme280",
        "adafruit_bme280": "adafruit_bme280",
    }

    raw_value = None
    for env_name in ("INSIDE_SENSOR", "INDOOR_SENSOR"):
        candidate = os.environ.get(env_name)
        if candidate:
            raw_value = candidate
            break

    if not raw_value:
        return None, None

    normalized_input = raw_value.strip()
    if "#" in normalized_input and not normalized_input.startswith(("'", '"')):
        normalized_input = normalized_input.split("#", 1)[0].strip()

    normalized = _normalize_sensor_name(normalized_input)
    resolved = aliases.get(normalized)
    return resolved, raw_value


def _extract_field(data: Any, key: str) -> Optional[float]:
    if hasattr(data, key):
        value = getattr(data, key)
    elif isinstance(data, dict):
        value = data.get(key)
    else:
        value = None
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _normalize_pressure(pres_raw: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    """Return (pressure_hpa, pressure_inhg) for a raw sensor reading."""

    if pres_raw is None:
        return None, None

    try:
        pres_value = float(pres_raw)
    except Exception:
        return None, None

    # Many drivers report Pascals while others provide hectopascals directly.
    # Treat anything that looks like a Pascal reading (>2,000) as Pa and
    # convert down to hPa before deriving inches of mercury.
    pres_hpa = pres_value / 100.0 if pres_value > 2000 else pres_value
    pres_inhg = pres_hpa * 0.02953 if pres_hpa is not None else None
    return pres_hpa, pres_inhg


def _read_chip_id(i2c: Any, addr: int, register: int = 0xD0) -> Optional[int]:
    """Best-effort helper to read a chip ID register over I2C.

    Used to guard against BME680/BME68x drivers latching onto a BME280 at the
    same address. Returns ``None`` if the register cannot be read cleanly.
    """

    if not hasattr(i2c, "writeto_then_readfrom"):
        return None

    buf = bytearray(1)
    locked = False
    try:
        if hasattr(i2c, "try_lock"):
            for _ in range(3):
                try:
                    locked = i2c.try_lock()
                except Exception:
                    locked = False
                if locked:
                    break
                time.sleep(0.005)
        if not locked and hasattr(i2c, "try_lock"):
            return None
        try:
            i2c.writeto_then_readfrom(addr, bytes([register]), buf)
        except Exception:
            return None
        return buf[0]
    finally:
        if locked and hasattr(i2c, "unlock"):
            try:
                i2c.unlock()
            except Exception:
                pass



def _suppress_i2c_error_output():
    """Context manager that silences noisy stderr output from native drivers."""

    class _Suppressor:
        def __enter__(self):
            try:
                self._fd = sys.stderr.fileno()
            except (AttributeError, ValueError, OSError):
                self._fd = None
                return self

            try:
                sys.stderr.flush()
            except Exception:
                pass

            self._saved = os.dup(self._fd)
            self._devnull = open(os.devnull, "wb")  # pylint: disable=consider-using-with
            os.dup2(self._devnull.fileno(), self._fd)
            return self

        def __exit__(self, exc_type, exc, tb):
            if getattr(self, "_fd", None) is None:
                return False

            try:
                sys.stderr.flush()
            except Exception:
                pass

            os.dup2(self._saved, self._fd)
            os.close(self._saved)
            self._devnull.close()
            return False

    return _Suppressor()


def _import_smbus_class() -> Any:
    """Return an SMBus class from the available Python I2C bindings.

    Raspberry Pi OS commonly ships the ``smbus`` package, while many Python
    projects depend on ``smbus2``.  The Pimoroni pure-Python drivers only need
    an object with the SMBus read/write methods, so accept either package.
    """

    import_errors: list[str] = []
    for module_name in ("smbus2", "smbus"):
        try:
            module = __import__(module_name, fromlist=["SMBus"])
            return module.SMBus
        except Exception as exc:  # pragma: no cover - depends on host packages
            import_errors.append(f"{module_name}: {exc}")

    raise ModuleNotFoundError(
        "No SMBus Python binding available; tried " + "; ".join(import_errors)
    )


def _probe_adafruit_bme680(i2c: Any, addresses: set[int]) -> Optional[SensorProbeResult]:
    if addresses and not addresses.intersection({0x76, 0x77}):
        return None

    import adafruit_bme680  # type: ignore

    expected_chip_id = getattr(adafruit_bme680, "_BME680_CHIPID", 0x61)

    candidate_addresses: Sequence[int]
    if addresses:
        candidate_addresses = tuple(sorted(addresses.intersection({0x76, 0x77})))
    else:
        candidate_addresses = (0x77, 0x76)

    dev = None
    last_error: Optional[Exception] = None
    for addr in candidate_addresses:
        chip_id = _read_chip_id(i2c, addr)
        if chip_id is not None and chip_id != expected_chip_id:
            logging.debug(
                "draw_inside: skipping Adafruit BME680 probe at 0x%02X due to chip ID 0x%02X",
                addr,
                chip_id,
            )
            continue
        try:
            dev = adafruit_bme680.Adafruit_BME680_I2C(i2c, address=addr)
            break
        except Exception as exc:  # pragma: no cover - relies on hardware
            last_error = exc

    if dev is None:
        if last_error is not None:
            raise last_error
        return None

    def read() -> SensorReadings:
        temp_f = float(dev.temperature) * 9 / 5 + 32
        hum = float(dev.humidity)
        pres_raw = getattr(dev, "pressure", None)
        pres_hpa, pres = _normalize_pressure(pres_raw)
        if pres_hpa is not None and not 300 <= pres_hpa <= 1100:
            raise RuntimeError(f"BME680 pressure sanity check failed: {pres_hpa:.1f} hPa")
        gas = getattr(dev, "gas", None)
        voc = float(gas) if gas not in (None, 0) else None
        return dict(
            temp_f=temp_f,
            humidity=hum,
            pressure_inhg=pres,
            pressure_hpa=pres_hpa,
            voc_ohms=voc,
        )

    return "Adafruit BME680", read


def _should_try_bme68x_helper(_i2c: Any) -> bool:
    """Return True when the Pimoroni bme68x helper can address the configured bus."""

    # The pi3g/Pimoroni bme68x C extension used by the helper opens
    # /dev/i2c-1 internally.  When the indoor sensor is explicitly configured
    # on another Linux bus (for example HyperPixel STEMMA/QT bus 15), trying
    # the helper only produces a noisy startup traceback before the working
    # pure-Python fallback runs.
    primary_bus = _resolve_i2c_bus_number(_i2c) if _i2c is not None else None
    if primary_bus is not None:
        return primary_bus == 1
    return 1 in _parse_i2c_bus_candidates()


def _summarize_bme68x_helper_error(message: str) -> str:
    """Collapse helper stderr tracebacks to a single useful log line."""

    lines = [line.strip() for line in message.splitlines() if line.strip()]
    if not lines:
        return "BME68X helper failed"

    for line in reversed(lines):
        if "Error:" in line or line.startswith(("ModuleNotFoundError", "ImportError")):
            return line
    return lines[-1]


def _probe_pimoroni_bme68x(_i2c: Any, addresses: set[int]) -> Optional[SensorProbeResult]:
    if addresses and not addresses.intersection({0x76, 0x77}):
        return None

    if not _should_try_bme68x_helper(_i2c):
        logging.debug(
            "draw_inside: skipping Pimoroni bme68x helper because it only supports "
            "/dev/i2c-1; using pimoroni_bme680 fallback"
        )
        return _probe_pimoroni_bme680(_i2c, addresses)

    def _read_bme68x_via_subprocess() -> dict[str, Any]:
        repo_root = Path(__file__).resolve().parents[1]
        bme68x_vendor_path = repo_root / "vendor" / "bme68x"
        helper_env = os.environ.copy()
        if bme68x_vendor_path.exists():
            existing_pythonpath = helper_env.get("PYTHONPATH")
            helper_env["PYTHONPATH"] = (
                str(bme68x_vendor_path)
                if not existing_pythonpath
                else f"{bme68x_vendor_path}{os.pathsep}{existing_pythonpath}"
            )

        script = """
import json
import sys
from importlib import import_module

sys.path.insert(0, __BME68X_VENDOR_PATH__)

import bme68x

try:
    const = import_module('bme68xConstants')
except Exception:
    const = None

addr_low = getattr(bme68x, 'BME68X_I2C_ADDR_LOW', 0x76)
addr_high = getattr(bme68x, 'BME68X_I2C_ADDR_HIGH', 0x77)
addresses = (addr_low, addr_high)

sensor = None
selected_addr = None
last_error = None
for addr in addresses:
    try:
        sensor = bme68x.BME68X(addr)
        selected_addr = addr
        break
    except Exception as exc:
        last_error = exc

if sensor is None:
    if last_error is not None:
        raise last_error
    raise RuntimeError('BME68X sensor not found')

data = sensor.get_data()
if isinstance(data, (list, tuple)):
    data = data[0] if data else None
if data is None:
    raise RuntimeError('BME68X returned no data')

def extract_field(payload, key):
    if hasattr(payload, key):
        value = getattr(payload, key)
    elif isinstance(payload, dict):
        value = payload.get(key)
    else:
        value = None
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None

bsec_data = getattr(data, 'bsec_data', None)
if bsec_data is None and isinstance(data, dict):
    bsec_data = data.get('bsec_data')

voc_index = None
if isinstance(bsec_data, dict):
    voc_index = extract_field(bsec_data, 'breath_voc_equivalent')
if voc_index is None:
    voc_index = extract_field(data, 'breath_voc_equivalent')

variant_id = getattr(sensor, 'variant_id', None)
gas_high = getattr(const, 'BME68X_VARIANT_GAS_HIGH', None) if const else None
provider = 'Pimoroni BME688' if variant_id == gas_high else 'Pimoroni BME68X'

print(json.dumps({
    'provider': provider,
    'address': selected_addr,
    'temperature': extract_field(data, 'temperature'),
    'humidity': extract_field(data, 'humidity'),
    'pressure': extract_field(data, 'pressure'),
    'gas_resistance': extract_field(data, 'gas_resistance'),
    'voc_index': voc_index,
}))
""".replace("__BME68X_VENDOR_PATH__", repr(str(bme68x_vendor_path)))
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=6,
            check=False,
            env=helper_env,
        )
        if result.returncode != 0:
            if result.returncode < 0:
                signal_num = -result.returncode
                raise RuntimeError(f"BME68X helper exited by signal {signal_num}")
            stderr = _summarize_bme68x_helper_error((result.stderr or "").strip())
            raise RuntimeError(stderr)

        stdout = (result.stdout or "").strip()
        if not stdout:
            raise RuntimeError("BME68X helper returned empty output")

        return json.loads(stdout)

    try:
        payload = _read_bme68x_via_subprocess()
    except Exception as exc:
        if isinstance(exc, RuntimeError) and "exited by signal" in str(exc):
            raise
        logging.info(
            "draw_inside: Pimoroni bme68x helper unavailable (%s); falling back to pimoroni_bme680",
            exc,
        )
        # The vendored bme68x C-extension is hard-wired to /dev/i2c-1.
        # On HyperPixel 4 STEMMA/QT setups the sensor is often wired on
        # auxiliary Linux buses (10/11/13/14/15), so reuse the pure-Python
        # Pimoroni driver that can bind an explicit SMBus instance.
        return _probe_pimoroni_bme680(_i2c, addresses)

    provider = str(payload.get("provider") or "Pimoroni BME68X")

    def read() -> SensorReadings:
        sample = _read_bme68x_via_subprocess()
        temp_c = _extract_field(sample, "temperature")
        hum = _extract_field(sample, "humidity")
        pres_raw = _extract_field(sample, "pressure")
        voc_raw = _extract_field(sample, "gas_resistance")
        voc_index = _extract_field(sample, "voc_index")

        temp_f = temp_c * 9 / 5 + 32 if temp_c is not None else None
        pres_hpa, pres = _normalize_pressure(pres_raw)
        if pres_hpa is not None and not 300 <= pres_hpa <= 1100:
            raise RuntimeError(f"BME68X pressure sanity check failed: {pres_hpa:.1f} hPa")

        voc = voc_raw if voc_raw not in (None, 0) else None
        voc_index = voc_index if voc_index not in (None, 0) else None

        if temp_f is None:
            raise RuntimeError("BME68X temperature reading missing")

        return dict(
            temp_f=temp_f,
            humidity=hum,
            pressure_inhg=pres,
            pressure_hpa=pres_hpa,
            voc_ohms=voc,
            voc_index=voc_index,
        )

    return provider, read


def _probe_pimoroni_bme680(_i2c: Any, addresses: set[int]) -> Optional[SensorProbeResult]:
    if addresses and not addresses.intersection({0x76, 0x77}):
        return None

    from importlib import import_module

    module = None
    last_import_error: Optional[Exception] = None
    for name in ("pimoroni_bme680", "bme680"):
        try:
            module = import_module(name)  # type: ignore[assignment]
            break
        except ModuleNotFoundError as exc:
            last_import_error = exc
        except Exception as exc:  # pragma: no cover - depends on environment
            logging.debug("draw_inside: error importing %s: %s", name, exc)
            last_import_error = exc

    if module is None:
        if last_import_error is not None:
            raise last_import_error
        raise RuntimeError("Pimoroni BME680 driver not available")

    candidate_addresses: Sequence[int]
    if addresses:
        candidate_addresses = tuple(sorted(addresses.intersection({0x76, 0x77})))
    else:
        candidate_addresses = (
            getattr(module, "I2C_ADDR_PRIMARY", 0x76),
            getattr(module, "I2C_ADDR_SECONDARY", 0x77),
        )

    bus_candidates = list(_get_smbus_candidates(_i2c))
    primary_bus = _resolve_i2c_bus_number(_i2c) if _i2c is not None else None

    sensor = None
    sensor_bus: Optional[int] = None
    last_error: Optional[Exception] = None
    provider_label = "Pimoroni BME680"
    expected_chip_id = getattr(module, "CHIP_ID", 0x61)
    variant_high = getattr(module, "VARIANT_HIGH", None)
    variant_low = getattr(module, "VARIANT_LOW", None)
    try:
        SMBus = _import_smbus_class()
    except Exception as exc:
        last_error = exc
        SMBus = None  # type: ignore[assignment]

    for bus_num in bus_candidates:
        if SMBus is None:
            break
        try:
            bus = SMBus(bus_num)
        except Exception as exc:
            last_error = exc
            continue

        for addr in candidate_addresses:
            chip_id = None
            if _i2c is not None and primary_bus is not None and bus_num == primary_bus:
                chip_id = _read_chip_id(_i2c, addr)
            if chip_id is None:
                try:
                    chip_id = int(bus.read_byte_data(addr, 0xD0))
                except Exception:
                    chip_id = None
            if chip_id is not None and chip_id != expected_chip_id:
                logging.debug(
                    "draw_inside: skipping Pimoroni BME680 probe at bus %s addr 0x%02X due to chip ID 0x%02X",
                    bus_num,
                    addr,
                    chip_id,
                )
                continue

            try:
                sensor = module.BME680(addr, i2c_device=bus)  # type: ignore[arg-type]
                sensor_bus = bus_num
                variant = getattr(sensor, "_variant", None)
                if variant is not None:
                    if variant_high is not None and variant == variant_high:
                        provider_label = f"Pimoroni BME688 (bus {bus_num}, 0x{addr:02X})"
                    elif variant_low is not None and variant == variant_low:
                        provider_label = f"Pimoroni BME680 (bus {bus_num}, 0x{addr:02X})"
                    else:
                        provider_label = f"Pimoroni BME68x (bus {bus_num}, 0x{addr:02X})"
                else:
                    provider_label = f"Pimoroni BME680 (bus {bus_num}, 0x{addr:02X})"
                break
            except Exception as exc:  # pragma: no cover - relies on hardware
                last_error = exc
        if sensor is not None:
            break

    if sensor is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("BME680 sensor not found")

    if sensor_bus is not None and primary_bus is not None and sensor_bus != primary_bus:
        logging.info(
            "draw_inside: selected Pimoroni BME68x on fallback bus %s (Blinka bus was %s)",
            sensor_bus,
            primary_bus,
        )

    for method, value in (
        ("set_humidity_oversample", getattr(module, "OS_2X", None)),
        ("set_pressure_oversample", getattr(module, "OS_4X", None)),
        ("set_temperature_oversample", getattr(module, "OS_8X", None)),
        ("set_filter", getattr(module, "FILTER_SIZE_3", None)),
        ("set_gas_status", getattr(module, "ENABLE_GAS_MEAS", None)),
    ):
        fn = getattr(sensor, method, None)
        if callable(fn) and value is not None:
            try:
                fn(value)
            except Exception:
                pass

    gas_temp = getattr(
        module,
        "DEFAULT_GAS_HEATER_TEMPERATURE",
        getattr(module, "GAS_HEATER_TEMP", None),
    )
    gas_dur = getattr(
        module,
        "DEFAULT_GAS_HEATER_DURATION",
        getattr(module, "GAS_HEATER_DURATION", None),
    )
    fn_temp = getattr(sensor, "set_gas_heater_temperature", None)
    fn_dur = getattr(sensor, "set_gas_heater_duration", None)
    if callable(fn_temp) and gas_temp is not None:
        try:
            fn_temp(gas_temp)
        except Exception:
            pass
    if callable(fn_dur) and gas_dur is not None:
        try:
            fn_dur(gas_dur)
        except Exception:
            pass

    def read() -> SensorReadings:
        if not getattr(sensor, "get_sensor_data", lambda: False)():
            raise RuntimeError("BME680 has no fresh data")
        data = getattr(sensor, "data", None)
        if data is None:
            raise RuntimeError("BME680 returned no data")

        temp_c = getattr(data, "temperature", None)
        hum = getattr(data, "humidity", None)
        pres_raw = getattr(data, "pressure", None)
        gas = getattr(data, "gas_resistance", None)
        heat_stable = getattr(data, "heat_stable", True)

        temp_f = float(temp_c) * 9 / 5 + 32 if temp_c is not None else None
        pres_hpa, pres = _normalize_pressure(pres_raw)
        if pres_hpa is not None and not 300 <= pres_hpa <= 1100:
            raise RuntimeError(f"BME680 pressure sanity check failed: {pres_hpa:.1f} hPa")
        voc_raw = float(gas) if gas is not None else None
        if voc_raw is not None and voc_raw <= 0:
            voc_raw = None
        if voc_raw is not None and not heat_stable:
            logging.debug("draw_inside: using BME680 gas reading before heater stability")
        voc = voc_raw
        hum_val = float(hum) if hum is not None else None

        if temp_f is None:
            raise RuntimeError("BME680 temperature reading missing")

        return dict(
            temp_f=temp_f,
            humidity=hum_val,
            pressure_inhg=pres,
            pressure_hpa=pres_hpa,
            voc_ohms=voc,
        )

    return provider_label, read


def _probe_pimoroni_bme280(i2c: Any, addresses: set[int]) -> Optional[SensorProbeResult]:
    if addresses and not addresses.intersection({0x76, 0x77}):
        return None

    from importlib import import_module

    module = None
    last_import_error: Optional[Exception] = None
    for name in ("pimoroni_bme280", "bme280"):
        try:
            module = import_module(name)  # type: ignore[assignment]
            break
        except ModuleNotFoundError as exc:
            last_import_error = exc
        except Exception as exc:  # pragma: no cover - depends on environment
            logging.debug("draw_inside: error importing %s: %s", name, exc)
            last_import_error = exc

    if module is None:
        if last_import_error is not None:
            raise last_import_error
        raise RuntimeError("Pimoroni BME280 driver not available")

    sensor_cls = getattr(module, "BME280", None)
    if sensor_cls is None:
        raise RuntimeError(f"{module.__name__} is missing the BME280 class")

    expected_chip_id = 0x60

    try:
        SMBus = _import_smbus_class()
    except Exception as exc:
        logging.warning("draw_inside: failed to import an SMBus binding: %s", exc)
        raise

    bus_candidates = _get_smbus_candidates(i2c)

    # Prefer the addresses we actually saw on the bus so we don't try the
    # wrong default. Fallback to the library defaults if we could not scan.
    candidate_addresses: Sequence[int]
    if addresses:
        candidate_addresses = tuple(sorted(addresses.intersection({0x76, 0x77})))
    else:
        candidate_addresses = (0x76, 0x77)

    dev = None
    successful_addr: Optional[int] = None
    successful_bus: Optional[int] = None
    last_error: Optional[Exception] = None
    primary_bus = _resolve_i2c_bus_number(i2c) if i2c is not None else None
    for bus_num in bus_candidates:
        try:
            bus = SMBus(bus_num)
        except Exception as exc:
            last_error = exc
            logging.debug("draw_inside: failed to initialize SMBus(%s): %s", bus_num, exc)
            continue

        for addr in candidate_addresses:
            chip_id = None
            if i2c is not None and primary_bus is not None and bus_num == primary_bus:
                chip_id = _read_chip_id(i2c, addr)
            if chip_id is None:
                try:
                    chip_id = int(bus.read_byte_data(addr, 0xD0))
                except Exception:
                    chip_id = None
            if chip_id is not None and chip_id != expected_chip_id:
                logging.debug(
                    "draw_inside: skipping Pimoroni BME280 probe at bus %s addr 0x%02X due to chip ID 0x%02X",
                    bus_num,
                    addr,
                    chip_id,
                )
                continue

            try:
                candidate = sensor_cls(i2c_addr=addr, i2c_dev=bus)  # type: ignore[call-arg]
                # Force an initial reading to validate connectivity. The Pimoroni
                # driver raises a RuntimeError with a helpful message if the bus
                # is not responding.
                _ = float(candidate.get_temperature())
                dev = candidate
                successful_addr = addr
                successful_bus = bus_num
                break
            except Exception as exc:  # pragma: no cover - relies on hardware
                last_error = exc
        if dev is not None:
            break

    adafruit_dev: Optional[Any] = None
    if dev is None and i2c is not None:
        try:
            import adafruit_bme280  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            last_error = exc
        else:
            for addr in candidate_addresses:
                try:
                    candidate = adafruit_bme280.Adafruit_BME280_I2C(i2c, address=addr)
                    # Trigger a measurement; attribute access will perform I2C IO
                    _ = float(candidate.temperature)
                except Exception as exc:  # pragma: no cover - relies on hardware
                    last_error = exc
                    continue
                adafruit_dev = candidate
                successful_addr = addr
                logging.debug(
                    "draw_inside: falling back to Adafruit BME280 driver for Pimoroni sensor at 0x%02X",
                    addr,
                )
                break

    if dev is None and adafruit_dev is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Pimoroni BME280 sensor not found")

    addr_for_label = successful_addr if successful_addr is not None else candidate_addresses[0]
    if successful_bus is not None:
        label = f"Pimoroni BME280 (bus {successful_bus}, 0x{addr_for_label:02X})"
    else:
        label = f"Pimoroni BME280 (0x{addr_for_label:02X})"

    fallback_dev: Optional[Any] = adafruit_dev
    fallback_error: Optional[Exception] = None

    def read_with_fallback() -> Optional[SensorReadings]:
        nonlocal fallback_dev, fallback_error

        if fallback_dev is None and fallback_error is None:
            try:
                import adafruit_bme280  # type: ignore

                fallback_dev = adafruit_bme280.Adafruit_BME280_I2C(
                    i2c, address=addr_for_label
                )
            except ModuleNotFoundError:
                fallback_error = ModuleNotFoundError("Adafruit BME280 driver missing")
            except Exception as exc:  # pragma: no cover - relies on hardware
                fallback_error = exc

        if fallback_dev is None:
            if fallback_error is not None:
                logging.debug(
                    "draw_inside: unable to use Adafruit fallback BME280 driver: %s",
                    fallback_error,
                )
            return None

        temp_f = float(fallback_dev.temperature) * 9 / 5 + 32
        hum_raw = getattr(fallback_dev, "humidity", None)
        pres_raw = getattr(fallback_dev, "pressure", None)
        pres_hpa, pres_inhg = _normalize_pressure(pres_raw)
        hum = float(hum_raw) if hum_raw is not None else None

        if pres_hpa is None or not 300 <= pres_hpa <= 1100:
            logging.debug(
                "draw_inside: Adafruit fallback BME280 pressure sanity check failed: %s",
                pres_hpa,
            )
            return None

        if hum is not None and not 0 <= hum <= 100:
            logging.debug(
                "draw_inside: Adafruit fallback BME280 humidity sanity check failed: %s",
                hum,
            )
            return None

        return dict(
            temp_f=temp_f,
            humidity=hum,
            pressure_inhg=pres_inhg,
            pressure_hpa=pres_hpa,
            voc_ohms=None,
        )

    if dev is not None:
        if successful_bus is not None and primary_bus is not None and successful_bus != primary_bus:
            logging.info(
                "draw_inside: selected Pimoroni BME280 on fallback bus %s (Blinka bus was %s)",
                successful_bus,
                primary_bus,
            )

        def read() -> SensorReadings:
            temp_f = float(dev.get_temperature()) * 9 / 5 + 32
            hum = float(dev.get_humidity())
            pres_raw = dev.get_pressure()
            pres_hpa, pres_inhg = _normalize_pressure(pres_raw)

            logging.info(
                "draw_inside: Pimoroni BME280 raw pressure: %s -> %.2f hPa = %.2f inHg",
                pres_raw,
                pres_hpa if pres_hpa is not None else float("nan"),
                pres_inhg if pres_inhg is not None else float("nan"),
            )

            if pres_hpa is not None and not 300 <= pres_hpa <= 1100:
                logging.warning(
                    "draw_inside: discarding Pimoroni BME280 reading with out-of-range pressure %.1f hPa",
                    pres_hpa,
                )
                fallback = read_with_fallback()
                if fallback is not None:
                    return fallback

                raise RuntimeError(
                    f"Pimoroni BME280 pressure sanity check failed: {pres_hpa:.1f} hPa"
                )

            if hum is not None and not 0 <= hum <= 100:
                logging.warning(
                    "draw_inside: discarding Pimoroni BME280 reading with out-of-range humidity %.1f%%",
                    hum,
                )
                fallback = read_with_fallback()
                if fallback is not None:
                    return fallback

                raise RuntimeError(
                    f"Pimoroni BME280 humidity sanity check failed: {hum:.1f}%"
                )

            return dict(
                temp_f=temp_f,
                humidity=hum,
                pressure_inhg=pres_inhg,
                pressure_hpa=pres_hpa,
                voc_ohms=None,
            )

        return label, read

    assert fallback_dev is not None

    def read() -> SensorReadings:
        temp_c = float(fallback_dev.temperature)
        hum_raw = getattr(fallback_dev, "humidity", None)
        pres_raw = getattr(fallback_dev, "pressure", None)
        pres_hpa, pres = _normalize_pressure(pres_raw)
        hum = float(hum_raw) if hum_raw is not None else None
        if pres_hpa is not None:
            logging.info(
                "draw_inside: Pimoroni BME280 (fallback) raw pressure: %s -> %.2f hPa = %.2f inHg",
                pres_raw,
                pres_hpa,
                pres if pres is not None else float("nan"),
            )

        if pres_hpa is not None and not 300 <= pres_hpa <= 1100:
            logging.warning(
                "draw_inside: discarding Pimoroni BME280 (fallback) reading with out-of-range pressure %.1f hPa",
                pres_hpa,
            )
            raise RuntimeError(
                f"Pimoroni BME280 (fallback) pressure sanity check failed: {pres_hpa:.1f} hPa"
            )

        if hum is not None and not 0 <= hum <= 100:
            logging.warning(
                "draw_inside: discarding Pimoroni BME280 (fallback) reading with out-of-range humidity %.1f%%",
                hum,
            )
            raise RuntimeError(
                f"Pimoroni BME280 (fallback) humidity sanity check failed: {hum:.1f}%"
            )

        temp_f = temp_c * 9 / 5 + 32
        return dict(
            temp_f=temp_f,
            humidity=hum,
            pressure_inhg=pres,
            pressure_hpa=pres_hpa,
            voc_ohms=None,
        )

    return label, read


def _probe_adafruit_bme280(i2c: Any, addresses: set[int]) -> Optional[SensorProbeResult]:
    if addresses and not addresses.intersection({0x76, 0x77}):
        return None

    import adafruit_bme280  # type: ignore

    dev = adafruit_bme280.Adafruit_BME280_I2C(i2c)

    expected_chip_id = 0x60

    chip_id = _read_chip_id(i2c, getattr(dev, "address", 0x76))
    if chip_id is not None and chip_id != expected_chip_id:
        logging.debug(
            "draw_inside: skipping Adafruit BME280 probe at 0x%02X due to chip ID 0x%02X",
            getattr(dev, "address", 0x76),
            chip_id,
        )
        return None

    def read() -> SensorReadings:
        temp_f = float(dev.temperature) * 9 / 5 + 32
        hum = float(dev.humidity)
        pres_raw = getattr(dev, "pressure", None)
        pres_hpa, pres = _normalize_pressure(pres_raw)

        logging.info(
            "draw_inside: Adafruit BME280 raw pressure: %s -> %.2f hPa = %.2f inHg",
            pres_raw,
            pres_hpa if pres_hpa is not None else float("nan"),
            pres if pres is not None else float("nan"),
        )

        if pres_hpa is not None and not 300 <= pres_hpa <= 1100:
            logging.warning(
                "draw_inside: discarding Adafruit BME280 reading with out-of-range pressure %.1f hPa",
                pres_hpa,
            )
            raise RuntimeError(
                f"Adafruit BME280 pressure sanity check failed: {pres_hpa:.1f} hPa"
            )

        if hum is not None and not 0 <= hum <= 100:
            logging.warning(
                "draw_inside: discarding Adafruit BME280 reading with out-of-range humidity %.1f%%",
                hum,
            )
            raise RuntimeError(
                f"Adafruit BME280 humidity sanity check failed: {hum:.1f}%"
            )

        return dict(
            temp_f=temp_f,
            humidity=hum,
            pressure_inhg=pres,
            pressure_hpa=pres_hpa,
            voc_ohms=None,
        )

    return "Adafruit BME280", read


def _probe_adafruit_sht4x(i2c: Any, addresses: set[int]) -> Optional[SensorProbeResult]:
    if addresses and not addresses.intersection({0x44, 0x45}):
        return None

    import adafruit_sht4x  # type: ignore

    dev = adafruit_sht4x.SHT4x(i2c)
    try:
        mode = getattr(adafruit_sht4x, "Mode", None)
        if mode is not None and hasattr(mode, "NOHEAT_HIGHPRECISION"):
            dev.mode = mode.NOHEAT_HIGHPRECISION
    except Exception:
        pass

    def read() -> SensorReadings:
        temp_c, hum = dev.measurements
        temp_f = float(temp_c) * 9 / 5 + 32
        hum_val = float(hum)
        return dict(temp_f=temp_f, humidity=hum_val, pressure_inhg=None, voc_ohms=None)

    return "Adafruit SHT41", read


def _get_probe_order(preference: Optional[SensorProbeName]) -> tuple[tuple[SensorProbeName, SensorProbeFn], ...]:
    probers: tuple[tuple[SensorProbeName, SensorProbeFn], ...] = (
        ("pimoroni_bme280", _probe_pimoroni_bme280),
        ("adafruit_bme280", _probe_adafruit_bme280),
        ("pimoroni_bme680", _probe_pimoroni_bme680),
        ("pimoroni_bme68x", _probe_pimoroni_bme68x),
        ("adafruit_bme680", _probe_adafruit_bme680),
        ("adafruit_sht41", _probe_adafruit_sht4x),
    )

    if not preference:
        return probers

    if preference == "pim_sensor_stick":
        return (
            ("pimoroni_bme280", _probe_pimoroni_bme280),
            ("adafruit_bme280", _probe_adafruit_bme280),
        )

    filtered = tuple((name, fn) for name, fn in probers if name == preference)
    if filtered:
        return filtered
    return probers


def _scan_i2c_addresses(i2c: Any) -> set[int]:
    addresses: set[int] = set()

    if not hasattr(i2c, "scan"):
        return addresses

    locked = False
    try:
        if hasattr(i2c, "try_lock"):
            for _ in range(5):
                try:
                    locked = i2c.try_lock()
                except Exception:
                    locked = False
                if locked:
                    break
                time.sleep(0.01)
        if locked or not hasattr(i2c, "try_lock"):
            try:
                addresses = set(i2c.scan())  # type: ignore[arg-type]
            except Exception as exc:
                logging.debug("draw_inside: I2C scan failed: %s", exc, exc_info=True)
        else:
            logging.debug("draw_inside: could not lock I2C bus for scanning")
    finally:
        if locked and hasattr(i2c, "unlock"):
            try:
                i2c.unlock()
            except Exception:
                pass

    return addresses


def _iter_board_i2c_pin_pairs() -> tuple[tuple[str, str], ...]:
    """Return candidate ``(scl, sda)`` board pin names for Blinka I2C init."""

    return (
        ("SCL", "SDA"),
        ("SCL1", "SDA1"),
        ("D3", "D2"),
        ("D45", "D44"),
        ("GP3", "GP2"),
        ("GP45", "GP44"),
    )


def _try_init_blinka_i2c() -> Optional[Any]:
    """Try common Blinka pin aliases before falling back to ExtendedI2C."""

    if board is None or busio is None:
        return None

    for scl_name, sda_name in _iter_board_i2c_pin_pairs():
        scl_pin = getattr(board, scl_name, None)
        sda_pin = getattr(board, sda_name, None)
        if scl_pin is None or sda_pin is None:
            continue
        try:
            i2c = busio.I2C(scl_pin, sda_pin)
            if (scl_name, sda_name) != ("SCL", "SDA"):
                logging.info(
                    "draw_inside: using alternate I2C pins (%s, %s)",
                    scl_name,
                    sda_name,
                )
            return i2c
        except Exception as exc:
            logging.debug(
                "draw_inside: failed to initialise I2C on (%s, %s): %s",
                scl_name,
                sda_name,
                exc,
            )

    return None


def _probe_sensor() -> tuple[Optional[str], Optional[Callable[[], SensorReadings]]]:
    """Try the available sensor drivers and return the first match."""

    if platform.system() != "Linux":
        logging.info(
            "draw_inside: skipping indoor sensor probe on unsupported platform %s",
            platform.system(),
        )
        return None, None

    preference, raw_preference = _get_sensor_env_override()
    if preference:
        logging.info("draw_inside: INSIDE_SENSOR set to %s; restricting probe order", preference)
    elif raw_preference:
        logging.warning(
            "draw_inside: INSIDE_SENSOR value %r not recognized; falling back to auto-detect",
            raw_preference,
        )

    # Pimoroni's bme68x Python driver talks to SMBus directly and does not
    # require Blinka/ExtendedI2C. When this driver is explicitly selected,
    # probe it first so missing Blinka dependencies do not generate misleading
    # warnings.
    if preference == "pimoroni_bme68x":
        try:
            result = _probe_pimoroni_bme68x(None, set())
        except ModuleNotFoundError as exc:
            logging.debug(
                "draw_inside: probe %s skipped (module missing): %s", preference, exc
            )
        except Exception as exc:  # pragma: no cover - relies on hardware
            logging.debug("draw_inside: probe %s failed: %s", preference, exc, exc_info=True)
        else:
            if result:
                provider, reader = result
                logging.info("draw_inside: detected %s", provider)
                return provider, reader

    i2c = None
    if board is None or busio is None:
        logging.warning(
            "draw_inside: Blinka I2C libs unavailable; trying SMBus-capable sensor probes only"
        )
    else:
        i2c = _try_init_blinka_i2c()
        if i2c is None:
            logging.warning("draw_inside: failed to initialise Blinka I2C on known pin mappings")

        if i2c is None:
            bus_candidates = _rank_i2c_buses(_parse_i2c_bus_candidates())
            try:
                from adafruit_extended_bus import ExtendedI2C  # type: ignore
            except Exception as exc:
                logging.warning(
                    "draw_inside: adafruit_extended_bus unavailable; continuing without Blinka I2C on fallback buses %s: %s",
                    bus_candidates,
                    exc,
                )
            else:
                for bus_num in bus_candidates:
                    try:
                        i2c = ExtendedI2C(bus_num)
                        logging.info("draw_inside: using fallback I2C bus %s", bus_num)
                        break
                    except Exception as exc:
                        logging.debug(
                            "draw_inside: failed to initialise fallback I2C bus %s: %s",
                            bus_num,
                            exc,
                            exc_info=True,
                        )

    if i2c is None:
        logging.warning(
            "draw_inside: no usable Blinka I2C bus available; continuing with SMBus-only probes"
        )

    addresses: set[int] = set()
    if i2c is not None:
        addresses = _scan_i2c_addresses(i2c)
        if addresses:
            formatted = ", ".join(f"0x{addr:02X}" for addr in sorted(addresses))
            logging.debug("draw_inside: detected I2C addresses: %s", formatted)
        else:
            logging.debug("draw_inside: no I2C addresses detected during scan")

    # Prefer BME280 variants before BME680/BME68x. Some BME680 drivers can
    # incorrectly initialise against a BME280 at the same address and return
    # garbage pressure values (~660 hPa instead of ~997 hPa). Trying the
    # BME280-specific probers first keeps the readings aligned with the
    # standalone BME280 CLI script. When INSIDE_SENSOR is set, only the
    # requested probe will run.
    probe_passes: list[tuple[str, tuple[tuple[SensorProbeName, SensorProbeFn], ...]]] = [
        ("preferred", _get_probe_order(preference)),
    ]
    if preference:
        probe_passes.append(("fallback", _get_probe_order(None)))

    for pass_name, probe_plan in probe_passes:
        for probe_name, probe in probe_plan:
            try:
                result = probe(i2c, addresses)
            except ModuleNotFoundError as exc:
                logging.debug(
                    "draw_inside: probe %s skipped (module missing): %s", probe_name, exc
                )
                continue
            except Exception as exc:  # pragma: no cover - relies on hardware
                logging.debug("draw_inside: probe %s failed: %s", probe_name, exc, exc_info=True)
                continue
            if result:
                provider, reader = result
                logging.info("draw_inside: detected %s", provider)
                return provider, reader

        if pass_name == "preferred" and preference:
            logging.warning(
                "draw_inside: preferred sensor probe %r failed; falling back to auto-detect",
                preference,
            )

    logging.warning("No supported indoor environmental sensor detected.")
    return None, None


def _probe_sensor_cached(
    *,
    force_refresh: bool = False,
) -> tuple[Optional[str], Optional[Callable[[], SensorReadings]]]:
    """Return a cached sensor probe result to avoid repeated full hardware scans."""

    global _sensor_probe_cache

    if force_refresh:
        with _sensor_probe_cache_lock:
            _sensor_probe_cache = None

    with _sensor_probe_cache_lock:
        if _sensor_probe_cache is not None:
            return _sensor_probe_cache

    result = _probe_sensor()

    with _sensor_probe_cache_lock:
        _sensor_probe_cache = result

    return result


def is_inside_sensor_available(*, force_refresh: bool = False) -> bool:
    """Return ``True`` only when an indoor sensor can be probed successfully."""

    provider, read_fn = _probe_sensor_cached(force_refresh=force_refresh)
    if provider and read_fn:
        return True

    preference, raw_preference = _get_sensor_env_override()
    if preference or raw_preference:
        logging.warning(
            "draw_inside: explicitly configured indoor sensor was not detected; inside screen will be skipped"
        )
    else:
        logging.info(
            "draw_inside: no indoor sensor configured or auto-detected; inside screen will be skipped"
        )

    return False


def _log_sensor_data(provider: Optional[str], data: dict[str, Optional[float]]) -> None:
    """Log sensor readings to a file in the user's home directory."""
    try:
        home_dir = Path.home()
        log_file = home_dir / "sensor_data.log"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Format the sensor readings
        readings = []
        if data:
            for key, value in sorted(data.items()):
                if value is not None:
                    readings.append(f"{key}={value:.2f}")

        readings_str = ", ".join(readings) if readings else "no data"
        log_line = f"{timestamp} | {provider or 'Unknown Sensor'} | {readings_str}\n"

        # Append to log file
        with open(log_file, "a") as f:
            f.write(log_line)

    except Exception as exc:
        logging.debug("Failed to log sensor data: %s", exc)


# ── Layout helpers ───────────────────────────────────────────────────────────
def _mix_color(color: tuple[int, int, int], target: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    factor = max(0.0, min(1.0, factor))
    return tuple(int(round(color[idx] * (1 - factor) + target[idx] * factor)) for idx in range(3))


def _interpolate_color(
    stops: Sequence[tuple[float, tuple[int, int, int]]],
    value: float,
) -> tuple[int, int, int]:
    """Linearly interpolate *value* across a gradient defined by *stops*.

    ``stops`` should contain ``(position, color)`` pairs sorted by position in the
    inclusive range ``[0.0, 1.0]``. Values outside the range are clamped to the
    nearest stop.
    """

    if not stops:
        return (0, 0, 0)

    value = max(0.0, min(1.0, value))

    previous_pos, previous_color = stops[0]
    for pos, color in stops[1:]:
        if value <= pos:
            span = pos - previous_pos or 1e-6
            alpha = (value - previous_pos) / span
            return _mix_color(previous_color, color, alpha)
        previous_pos, previous_color = pos, color

    return stops[-1][1]


def _draw_temperature_panel(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    temp_f: float,
    temp_text: str,
    descriptor: str,
    temp_base,
    label_base,
) -> None:
    x0, y0, x1, y1 = rect
    color = temperature_color(temp_f)
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)

    radius = max(14, min(26, min(width, height) // 5))
    bg = _mix_color(color, config.INSIDE_COL_BG, 0.4)
    outline = _mix_color(color, config.INSIDE_COL_BG, 0.25)
    draw.rounded_rectangle(rect, radius=radius, fill=bg, outline=outline, width=1)

    padding_x = max(16, width // 12)
    padding_y = max(12, height // 10)
    label_text = "Temperature"

    label_base_size = getattr(label_base, "size", 18)
    label_font = fit_font(
        draw,
        label_text,
        label_base,
        max_width=width - 2 * padding_x,
        max_height=max(14, int(height * 0.18)),
        min_pt=min(label_base_size, 10),
        max_pt=label_base_size,
    )
    _, label_h = measure_text(draw, label_text, label_font)
    label_x = x0 + padding_x
    label_y = y0 + padding_y

    descriptor = descriptor.strip()
    has_descriptor = bool(descriptor)
    if has_descriptor:
        descriptor_base_size = getattr(label_base, "size", 18)
        desc_font = fit_font(
            draw,
            descriptor,
            label_base,
            max_width=width - 2 * padding_x,
            max_height=max(14, int(height * 0.2)),
            min_pt=min(descriptor_base_size, 12),
            max_pt=descriptor_base_size,
        )
        _, desc_h = measure_text(draw, descriptor, desc_font)
        desc_x = x0 + padding_x
        desc_y = y1 - padding_y - desc_h
    else:
        desc_font = None
        desc_h = 0
        desc_x = x0 + padding_x
        desc_y = y1 - padding_y

    value_gap = max(10, height // 14)
    value_top = label_y + label_h + value_gap
    value_bottom = desc_y - value_gap if has_descriptor else y1 - padding_y
    value_max_height = max(32, value_bottom - value_top)
    temp_base_size = getattr(temp_base, "size", 48)

    safe_margin = max(4, width // 28)
    inner_left = x0 + padding_x
    inner_right = x1 - padding_x - safe_margin
    if inner_right <= inner_left:
        # Fall back to the widest area available without letting the value escape
        safe_margin = max(0, (width - 2 * padding_x - 1) // 2)
        inner_left = x0 + padding_x + safe_margin
        inner_right = max(inner_left + 1, x1 - padding_x - safe_margin)

    value_region_width = max(1, inner_right - inner_left)

    temp_font = fit_font(
        draw,
        temp_text,
        temp_base,
        max_width=value_region_width,
        max_height=value_max_height,
        min_pt=min(temp_base_size, 20),
        max_pt=temp_base_size,
    )

    # Re-check the rendered bounds to ensure the glyphs stay within the tile
    temp_bbox = draw.textbbox((0, 0), temp_text, font=temp_font)
    temp_w = temp_bbox[2] - temp_bbox[0]
    temp_h = temp_bbox[3] - temp_bbox[1]
    while temp_w > value_region_width and getattr(temp_font, "size", 0) > 12:
        next_size = getattr(temp_font, "size", 0) - 1
        temp_font = clone_font(temp_font, next_size)
        temp_bbox = draw.textbbox((0, 0), temp_text, font=temp_font)
        temp_w = temp_bbox[2] - temp_bbox[0]
        temp_h = temp_bbox[3] - temp_bbox[1]

    temp_x = inner_left
    temp_y = value_top

    if has_descriptor:
        if temp_y + temp_h > desc_y - value_gap:
            temp_y = max(label_y + label_h + value_gap, desc_y - value_gap - temp_h)
    else:
        max_temp_y = y1 - padding_y - temp_h
        if temp_y > max_temp_y:
            temp_y = max_temp_y

    draw.text(
        (label_x, label_y),
        label_text,
        font=label_font,
        fill=_mix_color(color, config.INSIDE_COL_TEXT, 0.2),
    )
    draw.text((temp_x, temp_y), temp_text, font=temp_font, fill=config.INSIDE_COL_TEXT)
    if has_descriptor:
        draw.text(
            (desc_x, desc_y),
            descriptor,
            font=desc_font,
            fill=_mix_color(color, config.INSIDE_COL_TEXT, 0.35),
        )


def _draw_metric_row(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    label: str,
    value: str,
    accent: tuple[int, int, int],
    label_base,
    value_base,
) -> None:
    x0, y0, x1, y1 = rect
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    radius = max(8, min(20, min(width, height) // 4))
    bg = _mix_color(accent, config.INSIDE_COL_BG, 0.3)
    outline = _mix_color(accent, config.INSIDE_COL_BG, 0.18)
    draw.rounded_rectangle(rect, radius=radius, fill=bg, outline=outline, width=1)

    padding_x = max(10, width // 10)
    padding_y = max(6, height // 8)

    available_width = max(1, width - 2 * padding_x)
    available_height = max(1, height - 2 * padding_y)

    label_base_size = getattr(label_base, "size", 18)
    label_min_pt = min(label_base_size, 8 if width < 120 else 10)
    label_font = fit_font(
        draw,
        label,
        label_base,
        max_width=available_width,
        max_height=max(12, int(height * 0.38)),
        min_pt=label_min_pt,
        max_pt=label_base_size,
    )
    label_w, label_h = measure_text(draw, label, label_font)

    value_base_size = getattr(value_base, "size", 24)
    value_min_pt = min(value_base_size, 10 if width < 120 else 12)
    value_max_height = max(18, available_height - label_h - max(6, height // 12))
    value_font = fit_font(
        draw,
        value,
        value_base,
        max_width=available_width,
        max_height=value_max_height,
        min_pt=value_min_pt,
        max_pt=value_base_size,
    )
    value_w, value_h = measure_text(draw, value, value_font)

    def _shrink_font(
        text: str,
        base,
        current,
        current_size: int,
        min_size: int,
    ) -> tuple[Any, tuple[int, int], int]:
        """Reduce *current* font size until the text fits or *min_size* reached."""

        width_limit = available_width
        height_limit = available_height
        width, height = measure_text(draw, text, current)
        while (width > width_limit or height > height_limit) and current_size > min_size:
            next_size = current_size - 1
            new_font = clone_font(base, next_size)
            new_size = getattr(new_font, "size", current_size)
            if new_size >= current_size:
                break
            current = new_font
            current_size = new_size
            width, height = measure_text(draw, text, current)
        return current, (width, height), current_size

    label_size = getattr(label_font, "size", label_base_size)
    value_size = getattr(value_font, "size", value_base_size)

    label_font, (label_w, label_h), label_size = _shrink_font(
        label,
        label_base,
        label_font,
        label_size,
        label_min_pt,
    )

    value_font, (value_w, value_h), value_size = _shrink_font(
        value,
        value_base,
        value_font,
        value_size,
        value_min_pt,
    )

    min_gap = max(6, height // 12)
    total_needed = label_h + min_gap + value_h
    while total_needed > available_height and (label_size > label_min_pt or value_size > value_min_pt):
        shrink_label = label_size > label_min_pt and (
            label_h >= value_h or value_size <= value_min_pt
        )
        if shrink_label:
            next_size = max(label_min_pt, label_size - 1)
            if next_size == label_size:
                break
            label_font = clone_font(label_base, next_size)
            new_size = getattr(label_font, "size", label_size)
            if new_size >= label_size:
                break
            label_size = new_size
            label_w, label_h = measure_text(draw, label, label_font)
        else:
            next_size = max(value_min_pt, value_size - 1)
            if next_size == value_size:
                break
            value_font = clone_font(value_base, next_size)
            new_size = getattr(value_font, "size", value_size)
            if new_size >= value_size:
                break
            value_size = new_size
            value_w, value_h = measure_text(draw, value, value_font)
        total_needed = label_h + min_gap + value_h

    label_w = min(label_w, available_width)
    value_w = min(value_w, available_width)

    label_x = x0 + padding_x
    label_y = y0 + padding_y
    value_indent = max(0, width // 14)
    value_x = x0 + padding_x + value_indent
    min_gap = max(6, height // 12)
    value_y = label_y + label_h + min_gap
    max_value_y = y1 - padding_y - value_h
    if value_y > max_value_y:
        value_y = max_value_y

    label_color = _mix_color(accent, config.INSIDE_COL_TEXT, 0.25)
    value_color = config.INSIDE_COL_TEXT

    draw.text((label_x, label_y), label, font=label_font, fill=label_color)
    draw.text((value_x, value_y), value, font=value_font, fill=value_color)


def _draw_voc_tile(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    label: str,
    value: str,
    descriptor: str,
    score: float,
    label_base,
    value_base,
) -> None:
    x0, y0, x1, y1 = rect
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    radius = max(10, min(20, min(width, height) // 4))

    bg = _voc_quality_color(score)
    outline = _mix_color(bg, config.INSIDE_COL_BG, 0.25)
    draw.rounded_rectangle(rect, radius=radius, fill=bg, outline=outline, width=1)

    padding_x = max(12, width // 12)
    padding_y = max(8, height // 10)

    descriptor = descriptor.strip()
    has_descriptor = bool(descriptor)

    # Horizontal layout: reserve a left text column for VOC + descriptor,
    # then draw the metric value on the right so it stays clear of both labels.
    content_w = max(1, width - 2 * padding_x)
    side_gap = max(10, width // 18)
    left_w = max(1, int(content_w * 0.52))
    right_w = max(1, content_w - left_w - side_gap)
    if right_w < max(72, width // 4):
        right_w = max(1, min(content_w, max(72, width // 4)))
        left_w = max(1, content_w - right_w - side_gap)

    left_x = x0 + padding_x
    right_x = left_x + left_w + side_gap

    label_base_size = getattr(label_base, "size", 18)
    label_line = f"{label} · {descriptor}" if has_descriptor else label
    label_font = fit_font(
        draw,
        label_line,
        label_base,
        max_width=left_w,
        max_height=max(12, int(height * 0.26)),
        min_pt=min(label_base_size, 10),
        max_pt=label_base_size,
    )
    label_w, label_h = measure_text(draw, label_line, label_font)

    stack_h = label_h
    stack_top = y0 + (height - stack_h) // 2
    min_top = y0 + padding_y
    max_top = y1 - padding_y - stack_h
    if max_top < min_top:
        stack_top = min_top
    else:
        stack_top = max(min_top, min(stack_top, max_top))

    label_x = left_x
    label_y = stack_top
    value_base_size = getattr(value_base, "size", 24)
    value_font = fit_font(
        draw,
        value,
        value_base,
        max_width=right_w,
        max_height=max(18, height - 2 * padding_y),
        min_pt=min(value_base_size, 14),
        max_pt=value_base_size,
    )
    value_w, value_h = measure_text(draw, value, value_font)
    value_x = right_x + max(0, right_w - value_w)
    value_y = y0 + (height - value_h) // 2
    label_color = _mix_color(bg, config.INSIDE_COL_TEXT, 0.32)
    value_color = config.INSIDE_COL_TEXT

    draw.text((label_x, label_y), label_line, font=label_font, fill=label_color)
    draw.text((value_x, value_y), value, font=value_font, fill=value_color)


def _metric_grid_dimensions(count: int) -> tuple[int, int]:
    if count <= 0:
        return 0, 0
    if count <= 2:
        columns = count
    elif count <= 6:
        columns = 2
    else:
        columns = 3
    columns = max(1, columns)
    rows = int(math.ceil(count / columns))
    return columns, rows


def _metric_grid_cells(
    rect: tuple[int, int, int, int], count: int
) -> list[tuple[int, int, int, int]]:
    x0, y0, x1, y1 = rect
    width = max(0, x1 - x0)
    height = max(0, y1 - y0)
    if count <= 0 or width <= 0 or height <= 0:
        return []

    columns, rows = _metric_grid_dimensions(count)
    if columns <= 0 or rows <= 0:
        return []

    if columns > 1:
        desired_h_gap = max(8, width // 30)
        max_h_gap = max(0, (width - columns) // (columns - 1))
        h_gap = min(desired_h_gap, max_h_gap)
    else:
        h_gap = 0
    if rows > 1:
        desired_v_gap = max(8, height // 30)
        max_v_gap = max(0, (height - rows) // (rows - 1))
        v_gap = min(desired_v_gap, max_v_gap)
    else:
        v_gap = 0

    total_h_gap = h_gap * (columns - 1)
    total_v_gap = v_gap * (rows - 1)

    available_width = max(columns, width - total_h_gap)
    available_height = max(rows, height - total_v_gap)

    cell_width = max(72, available_width // columns)
    if cell_width * columns + total_h_gap > width:
        cell_width = max(1, available_width // columns)
    cell_height = max(44, available_height // rows)
    if cell_height * rows + total_v_gap > height:
        cell_height = max(1, available_height // rows)

    grid_width = min(width, cell_width * columns + total_h_gap)
    grid_height = min(height, cell_height * rows + total_v_gap)
    start_x = x0 + max(0, (width - grid_width) // 2)
    start_y = y0 + max(0, (height - grid_height) // 2)

    cells: list[tuple[int, int, int, int]] = []
    for index in range(count):
        row = index // columns
        col = index % columns
        left = start_x + col * (cell_width + h_gap)
        top = start_y + row * (cell_height + v_gap)
        right = min(x1, left + cell_width)
        bottom = min(y1, top + cell_height)
        if right <= left or bottom <= top:
            continue
        cells.append((left, top, right, bottom))

    return cells


def _draw_metric_rows(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    metrics: Sequence[dict[str, Any]],
    label_base,
    value_base,
    *,
    cells: Optional[Sequence[tuple[int, int, int, int]]] = None,
) -> None:
    count = len(metrics)
    cell_rects = list(cells) if cells is not None else _metric_grid_cells(rect, count)

    for metric, cell_rect in zip(metrics, cell_rects):
        _draw_metric_row(
            draw,
            cell_rect,
            metric["label"],
            metric["value"],
            metric["color"],
            label_base,
            value_base,
        )


def _prettify_metric_label(key: str) -> str:
    key = key.replace("_", " ").strip()
    if not key:
        return "Value"
    replacements = {
        "voc": "VOC",
        "co2": "CO₂",
        "co": "CO",
        "pm25": "PM2.5",
        "pm10": "PM10",
        "iaq": "IAQ",
    }
    parts = []
    for token in key.split():
        lower = token.lower()
        if lower in replacements:
            parts.append(replacements[lower])
        elif len(token) <= 2:
            parts.append(token.upper())
        else:
            parts.append(token.capitalize())
    return " ".join(parts)


def _format_generic_metric_value(key: str, value: float) -> str:
    key_lower = key.lower()
    if key_lower.endswith("_ohms"):
        return format_voc_ohms(value)
    if key_lower.endswith("_f"):
        return f"{value:.1f}°F"
    if key_lower.endswith("_c"):
        return f"{value:.1f}°C"
    if key_lower.endswith("_ppm"):
        return f"{value:.0f} ppm"
    if key_lower.endswith("_ppb"):
        return f"{value:.0f} ppb"
    if key_lower.endswith("_percent") or key_lower.endswith("_pct"):
        return f"{value:.1f}%"
    if key_lower.endswith("_inhg"):
        return f"{value:.2f} inHg"
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _voc_quality_score(value: Optional[float], scale: str) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None

    if scale == "index":
        normalized = 1.0 - max(0.0, min(numeric, 500.0)) / 500.0
    else:
        clean_min = 5_000.0
        clean_max = 800_000.0
        numeric = max(1.0, numeric)
        normalized = (
            math.log10(numeric) - math.log10(clean_min)
        ) / (math.log10(clean_max) - math.log10(clean_min))

    return max(0.0, min(1.0, normalized))


def _voc_quality_color(score: float) -> tuple[int, int, int]:
    gradient = (
        (0.0, (190, 38, 44)),
        (0.25, (225, 118, 32)),
        (0.5, (230, 198, 64)),
        (0.75, (38, 184, 132)),
        (1.0, (64, 156, 255)),
    )
    return _interpolate_color(gradient, score)


def _describe_voc(score: float) -> str:
    if score >= 0.82:
        return "Excellent air"
    if score >= 0.64:
        return "Good air"
    if score >= 0.46:
        return "Fair air"
    if score >= 0.28:
        return "Poor air"
    return "Very poor"

# ── Main render ──────────────────────────────────────────────────────────────
def _clean_metric(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _build_metric_entries(data: dict[str, Optional[float]]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    used_groups: set[str] = set()

    palette: list[tuple[int, int, int]] = [
        config.INSIDE_CHIP_BLUE,
        config.INSIDE_CHIP_AMBER,
        config.INSIDE_CHIP_PURPLE,
        _mix_color(config.INSIDE_CHIP_BLUE, config.INSIDE_CHIP_AMBER, 0.45),
        _mix_color(config.INSIDE_CHIP_PURPLE, config.INSIDE_CHIP_BLUE, 0.4),
        _mix_color(config.INSIDE_CHIP_PURPLE, config.INSIDE_COL_BG, 0.35),
    ]

    Spec = tuple[str, str, Callable[[float], str], tuple[int, int, int], Optional[str]]
    known_specs: Sequence[Spec] = (
        ("humidity", "Humidity", lambda v: f"{v:.1f}%", config.INSIDE_CHIP_BLUE, "humidity"),
        ("dew_point_f", "Dew Point", lambda v: f"{v:.1f}°F", config.INSIDE_CHIP_BLUE, "dew_point"),
        ("dew_point_c", "Dew Point", lambda v: f"{v:.1f}°C", config.INSIDE_CHIP_BLUE, "dew_point"),
        # Prefer inHg for consistency with the standalone Pimoroni BME280 CLI
        # script; fall back to metric units if necessary.
        ("pressure_inhg", "Pressure", lambda v: f"{v:.2f} inHg", config.INSIDE_CHIP_AMBER, "pressure"),
        ("pressure_pa", "Pressure", lambda v: f"{v:.0f} Pa", config.INSIDE_CHIP_AMBER, "pressure"),
        ("voc_ohms", "VOC", format_voc_ohms, config.INSIDE_CHIP_PURPLE, "voc"),
        ("voc_index", "VOC Index", lambda v: f"{v:.0f}", config.INSIDE_CHIP_PURPLE, "voc"),
        ("iaq", "IAQ", lambda v: f"{v:.0f}", config.INSIDE_CHIP_PURPLE, "iaq"),
        ("co2_ppm", "CO₂", lambda v: f"{v:.0f} ppm", _mix_color(config.INSIDE_CHIP_BLUE, config.INSIDE_CHIP_AMBER, 0.35), "co2"),
    )

    for key, label, formatter, color, group in known_specs:
        if group and group in used_groups:
            continue
        value = _clean_metric(data.get(key))
        if value is None:
            continue
        metrics.append(dict(label=label, value=formatter(value), color=color))
        used_keys.add(key)
        if group:
            used_groups.add(group)

    skip_keys = {"temp", "temperature", "pressure_hpa"}
    extra_palette_index = 0
    for key in sorted(data.keys()):
        if key in used_keys or key == "temp_f":
            continue
        if any(key.lower().startswith(prefix) for prefix in skip_keys):
            continue
        value = _clean_metric(data.get(key))
        if value is None:
            continue
        color = palette[(len(metrics) + extra_palette_index) % len(palette)]
        extra_palette_index += 1
        metrics.append(
            dict(
                label=_prettify_metric_label(key),
                value=_format_generic_metric_value(key, value),
                color=color,
            )
        )

    return metrics


def _build_voc_tile(data: dict[str, Optional[float]], provider: Optional[str]) -> Optional[dict[str, Any]]:
    voc_index = data.get("voc_index")
    voc_ohms = data.get("voc_ohms")

    scale = "index" if voc_index is not None else "ohms"
    value = voc_index if voc_index is not None else voc_ohms
    if value is None:
        return None

    score = _voc_quality_score(value, scale)
    if score is None:
        return None

    descriptor = _describe_voc(score)
    label = "VOC Index" if scale == "index" else "VOC"
    display_value = f"{value:.0f}" if scale == "index" else format_voc_ohms(value)

    return dict(label=label, value=display_value, descriptor=descriptor, score=score)


def _history_values(data: dict[str, Optional[float]]) -> dict[str, float]:
    """Return the canonical readings that can be plotted on the inside screen."""

    candidates = (
        ("Temperature", "temp_f"),
        ("Humidity", "humidity"),
        ("Pressure", "pressure_inhg"),
        ("VOC Index", "voc_index"),
        ("VOC", "voc_ohms"),
        ("IAQ", "iaq"),
        ("CO₂", "co2_ppm"),
    )
    values: dict[str, float] = {}
    for label, key in candidates:
        value = _clean_metric(data.get(key))
        if value is not None and label not in values:
            values[label] = value
    return values


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
    """Ellipsize text so values and sensor names stay inside their columns."""

    if measure_text(draw, text, font)[0] <= max_width:
        return text
    shortened = text
    while shortened and measure_text(draw, shortened + "…", font)[0] > max_width:
        shortened = shortened[:-1]
    return shortened + "…" if shortened else "…"


def _record_inside_history(data: dict[str, Optional[float]], timestamp: Optional[float] = None) -> None:
    """Store and persist a bounded history of readings for the mini charts."""

    recorded_at = time.time() if timestamp is None else timestamp
    with _inside_history_lock:
        _load_inside_history(recorded_at)
        for label, value in _history_values(data).items():
            points = _inside_history.setdefault(label, [])
            points.append((recorded_at, value))
            points[:] = [
                point
                for point in points
                if 0 <= recorded_at - point[0] <= _HISTORY_MAX_AGE_SECONDS
            ]
            points.sort(key=lambda point: point[0])
            del points[:-_HISTORY_LIMIT]
        _save_inside_history()


def _load_inside_history(now: Optional[float] = None) -> None:
    """Load recent chart samples once, tolerating missing or corrupt state."""

    global _inside_history_loaded
    with _inside_history_lock:
        if _inside_history_loaded:
            return
        _inside_history_loaded = True
        current_time = time.time() if now is None else now
        cutoff = current_time - _HISTORY_MAX_AGE_SECONDS
        path = Path(os.path.expandvars(_HISTORY_PATH)).expanduser()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
            history = payload.get("history")
            if not isinstance(history, dict):
                raise ValueError("history must be an object")
            for label, raw_points in history.items():
                if not isinstance(label, str) or not isinstance(raw_points, list):
                    continue
                points: list[tuple[float, float]] = []
                for point in raw_points:
                    if not isinstance(point, (list, tuple)) or len(point) != 2:
                        continue
                    try:
                        stamp, value = float(point[0]), float(point[1])
                    except (TypeError, ValueError):
                        continue
                    if cutoff <= stamp <= current_time and math.isfinite(stamp) and math.isfinite(value):
                        points.append((stamp, value))
                if points:
                    _inside_history[label] = sorted(points)[-_HISTORY_LIMIT:]
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logging.warning("draw_inside: unable to load chart history from %s: %s", path, exc)


def _save_inside_history() -> None:
    """Atomically save chart samples so a restart does not empty the graphs."""

    path = Path(os.path.expandvars(_HISTORY_PATH)).expanduser()
    temp_name: Optional[str] = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f"{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temp_name = handle.name
            json.dump({"history": _inside_history}, handle)
        os.replace(temp_name, path)
    except OSError as exc:
        logging.warning("draw_inside: unable to save chart history to %s: %s", path, exc)
        if temp_name:
            try:
                os.remove(temp_name)
            except OSError:
                pass


def _draw_history_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    points: Sequence[tuple[float, float]],
    line_color: tuple[int, int, int],
) -> None:
    """Draw an AQI-style sparkline, including its subtle grid and frame."""

    x0, y0, x1, y1 = box
    if x1 - x0 < 12 or y1 - y0 < 8:
        return
    draw.rounded_rectangle(box, radius=max(2, min(5, (y1 - y0) // 3)), outline=(28, 64, 88))
    ix0, iy0, ix1, iy1 = x0 + 2, y0 + 2, x1 - 2, y1 - 2
    width, height = max(1, ix1 - ix0), max(1, iy1 - iy0)
    for tick in range(1, 4):
        tick_x = ix0 + round(width * tick / 4)
        draw.line((tick_x, iy0, tick_x, iy1), fill=(43, 88, 116))
    draw.line((ix0, iy0 + height // 2, ix1, iy0 + height // 2), fill=(68, 105, 130))
    if len(points) < 2:
        return
    start, end = points[0][0], points[-1][0]
    if end <= start:
        end = start + 1
    values = [value for _, value in points]
    low, high = min(values), max(values)
    if low == high:
        padding = max(1.0, abs(high) * 0.05)
        low, high = low - padding, high + padding
    coordinates = [
        (ix0 + round((stamp - start) / (end - start) * width), iy1 - round((value - low) / (high - low) * height))
        for stamp, value in points
    ]
    draw.line(coordinates, fill=line_color, width=2 if y1 - y0 >= 14 else 1)
    for point in coordinates[-12:]:
        draw.point(point, fill=(245, 250, 255))


def _render_inside(data: dict[str, Optional[float]], provider: Optional[str], sensor_error: Optional[str]) -> Image.Image:
    """Render indoor readings with the same badge-and-chart language as AQI."""

    img = Image.new("RGB", (W, H), config.INSIDE_COL_BG)
    draw = ImageDraw.Draw(img)
    margin = max(4, W // 32)
    header_font = config.FONT_WEATHER_DETAILS_SMALL_BOLD
    value_font = config.FONT_WEATHER_DETAILS_SMALL
    label_font = config.FONT_WEATHER_DETAILS_TINY
    title = "INSIDE"
    draw.text((margin, margin), title, font=header_font, fill=(235, 235, 235))
    title_w, title_h = measure_text(draw, title, header_font)
    source = provider or sensor_error or "Indoor sensor"
    source = _fit_text(draw, source, label_font, W - title_w - margin * 3)
    source_w, source_h = measure_text(draw, source, label_font)
    draw.text(
        (W - margin - source_w, margin + (title_h - source_h) // 2),
        source,
        font=label_font,
        fill=(185, 200, 215),
    )

    badge_top = margin + title_h + max(4, H // 40)
    badge_h = max(40, H // 4)
    temp_f = _clean_metric(data.get("temp_f"))
    badge_color = temperature_color(temp_f if temp_f is not None else 65.0)
    badge_fill = _mix_color(badge_color, config.INSIDE_COL_BG, 0.25)
    draw.rounded_rectangle((margin, badge_top, W - margin, badge_top + badge_h), radius=8, fill=badge_fill)
    temp_text = f"{temp_f:.1f}°F" if temp_f is not None else "--.-°F"
    badge_padding = max(8, W // 32)
    temp_left = margin + badge_padding
    chart_left = max(W * 9 // 16, temp_left + 1)
    chart_right = W - margin - badge_padding
    chart_gap = max(6, W // 64)
    temp_max_width = max(1, chart_left - chart_gap - temp_left)
    temp_min_pt = min(24, max(10, W // 10))
    temp_font = fit_font(
        draw,
        temp_text,
        config.FONT_WEATHER_DETAILS_SMALL_BOLD,
        max_width=temp_max_width,
        max_height=max(24, badge_h - 12),
        min_pt=temp_min_pt,
        max_pt=max(40, H // 5),
    )
    temp_w, temp_h = measure_text(draw, temp_text, temp_font)
    temp_y = badge_top + (badge_h - temp_h) // 2
    draw.text((temp_left, temp_y), temp_text, font=temp_font, fill=(255, 255, 255))

    _load_inside_history()
    with _inside_history_lock:
        histories = {key: tuple(value) for key, value in _inside_history.items()}
    temperature_history = histories.get("Temperature")
    temperature_fits = temp_left + temp_w + chart_gap <= chart_left
    if temperature_history and temperature_fits and chart_right - chart_left >= 12:
        chart_padding_y = max(7, badge_h // 6)
        _draw_history_chart(
            draw,
            (
                chart_left,
                badge_top + chart_padding_y,
                chart_right,
                badge_top + badge_h - chart_padding_y,
            ),
            temperature_history,
            badge_color,
        )

    card_top = badge_top + badge_h + max(4, H // 40)
    card_bottom = H - margin
    draw.rounded_rectangle((margin, card_top, W - margin, card_bottom), radius=8, fill=(12, 28, 42), outline=(34, 70, 98))
    metrics = _build_metric_entries(data)[:4]
    if not metrics:
        metrics = [{"label": "Status", "value": sensor_error or "Waiting for readings", "color": config.INSIDE_CHIP_BLUE}]
    content_x, right_edge = margin + 6, W - margin - 6
    label_w = max(measure_text(draw, metric["label"].upper(), label_font)[0] for metric in metrics) + 10
    value_x = content_x + label_w
    chart_x = max(W * 5 // 8, value_x + 38)
    charts_enabled = chart_x < right_edge - 24
    row_h = max(1, (card_bottom - card_top - 8) // len(metrics))
    for index, metric in enumerate(metrics):
        y0 = card_top + 4 + index * row_h
        y1 = card_top + 4 + (index + 1) * row_h if index < len(metrics) - 1 else card_bottom - 4
        if index:
            draw.line((content_x, y0, right_edge, y0), fill=(26, 54, 76))
        label = metric["label"]
        label_y = y0 + (y1 - y0 - measure_text(draw, label, label_font)[1]) // 2
        value_max_x = chart_x - 6 if charts_enabled and label in histories else right_edge
        value = _fit_text(draw, metric["value"], value_font, max(1, value_max_x - value_x))
        value_y = y0 + (y1 - y0 - measure_text(draw, value, value_font)[1]) // 2
        draw.text((content_x, label_y), label.upper(), font=label_font, fill=(165, 185, 205))
        draw.text((value_x, value_y), value, font=value_font, fill=(235, 242, 248))
        if charts_enabled and label in histories:
            chart_h = max(8, min(y1 - y0 - 6, H // 18))
            chart_y = y0 + (y1 - y0 - chart_h) // 2
            _draw_history_chart(draw, (chart_x, chart_y, right_edge, chart_y + chart_h), histories[label], metric["color"])
    return img


def draw_inside(display, transition: bool=False):
    provider, read_fn = _probe_sensor_cached()
    sensor_error: Optional[str] = None
    if not read_fn:
        logging.warning("draw_inside: sensor not available")
        sensor_error = "Sensor unavailable"
        cleaned = {}
        temp_f = None
    else:
        try:
            data = read_fn()
            cleaned: dict[str, Optional[float]] = {}
            if isinstance(data, dict):
                cleaned = {key: _clean_metric(value) for key, value in data.items()}
            else:
                logging.debug("draw_inside: unexpected data payload type %s", type(data))
                cleaned = {}
            temp_f = cleaned.get("temp_f")

            # Log the sensor data to file
            _log_sensor_data(provider, cleaned)
            _record_inside_history(cleaned)

        except Exception as e:
            logging.warning(f"draw_inside: sensor read failed: {e}")
            cleaned = {}
            temp_f = None
            sensor_error = "Read failed"

    if temp_f is None and sensor_error is None:
        logging.warning("draw_inside: temperature missing from sensor data")
        sensor_error = "No temperature"

    img = _render_inside(cleaned, provider, sensor_error)

    if transition:
        return img

    clear_display(display)
    display.image(img)
    display.show()
    time.sleep(5)
    return None


if __name__ == "__main__":
    try:
        preview = draw_inside(None, transition=True)
        if preview:
            preview.show()
    except Exception:
        pass
