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
import time
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
from datetime import datetime

from PIL import Image, ImageDraw
import config
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
    import board, busio  # type: ignore
except Exception:  # allows non-Pi dev boxes
    board = None
    busio = None

W, H = config.WIDTH, config.HEIGHT

SensorReadings = Dict[str, Optional[float]]
SensorProbeResult = Tuple[str, Callable[[], SensorReadings]]
SensorProbeFn = Callable[[Any, Set[int]], Optional[SensorProbeResult]]
SensorProbeName = str


def _prepend_vendor_sensor_drivers():
    """Prefer vendored Pimoroni sensor drivers when available."""

    repo_root = Path(__file__).resolve().parents[1]
    vendor_paths = (
        repo_root / "vendor" / "pimoroni-bme280",
        repo_root / "vendor" / "pimoroni-bme680",
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


def _get_sensor_env_override() -> Tuple[Optional[SensorProbeName], Optional[str]]:
    """Return the requested sensor driver from the environment, if provided."""

    aliases = {
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

    normalized = _normalize_sensor_name(raw_value)
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


def _normalize_pressure(pres_raw: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
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


def _probe_adafruit_bme680(i2c: Any, addresses: Set[int]) -> Optional[SensorProbeResult]:
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


def _probe_pimoroni_bme68x(_i2c: Any, addresses: Set[int]) -> Optional[SensorProbeResult]:
    if addresses and not addresses.intersection({0x76, 0x77}):
        return None

    from importlib import import_module

    import bme68x  # type: ignore

    try:
        I2C_ADDR_LOW = getattr(bme68x, "BME68X_I2C_ADDR_LOW")
        I2C_ADDR_HIGH = getattr(bme68x, "BME68X_I2C_ADDR_HIGH")
    except AttributeError:
        const = import_module("bme68xConstants")  # type: ignore
        I2C_ADDR_LOW = getattr(const, "BME68X_I2C_ADDR_LOW", 0x76)
        I2C_ADDR_HIGH = getattr(const, "BME68X_I2C_ADDR_HIGH", 0x77)

    sensor = None
    last_error: Optional[Exception] = None
    for addr in (I2C_ADDR_LOW, I2C_ADDR_HIGH):
        chip_id = _read_chip_id(_i2c, addr)
        expected_id = getattr(bme68x, "BME68X_CHIP_ID", 0x61)
        if chip_id is not None and chip_id != expected_id:
            logging.debug(
                "draw_inside: skipping BME68X probe at 0x%02X due to chip ID 0x%02X",
                addr,
                chip_id,
            )
            continue
        try:
            with _suppress_i2c_error_output():
                sensor = bme68x.BME68X(addr)  # type: ignore
            break
        except Exception as exc:  # pragma: no cover - relies on hardware
            last_error = exc
    if sensor is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("BME68X sensor not found")

    variant_id = getattr(sensor, "variant_id", None)
    const_module = import_module("bme68xConstants")  # type: ignore
    gas_low = getattr(const_module, "BME68X_VARIANT_GAS_LOW", None)
    gas_high = getattr(const_module, "BME68X_VARIANT_GAS_HIGH", None)
    if variant_id == gas_high:
        provider = "Pimoroni BME688"
    else:
        provider = "Pimoroni BME68X"

    def read() -> SensorReadings:
        data = sensor.get_data()
        if isinstance(data, (list, tuple)):
            data = data[0] if data else None
        if data is None:
            raise RuntimeError("BME68X returned no data")

        temp_c = _extract_field(data, "temperature")
        hum = _extract_field(data, "humidity")
        pres_raw = _extract_field(data, "pressure")

        voc_index = None
        bsec_data = getattr(data, "bsec_data", None)
        if bsec_data is None and isinstance(data, dict):
            bsec_data = data.get("bsec_data")

        if isinstance(bsec_data, dict):
            voc_index = _extract_field(bsec_data, "breath_voc_equivalent")

        if voc_index is None:
            voc_index = _extract_field(data, "breath_voc_equivalent")

        voc_raw = _extract_field(data, "gas_resistance")

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


def _probe_pimoroni_bme680(_i2c: Any, addresses: Set[int]) -> Optional[SensorProbeResult]:
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

    sensor = None
    last_error: Optional[Exception] = None
    provider_label = "Pimoroni BME680"
    expected_chip_id = getattr(module, "CHIP_ID", 0x61)
    variant_high = getattr(module, "VARIANT_HIGH", None)
    variant_low = getattr(module, "VARIANT_LOW", None)
    for addr in candidate_addresses:
        chip_id = _read_chip_id(_i2c, addr)
        if chip_id is not None and chip_id != expected_chip_id:
            logging.debug(
                "draw_inside: skipping Pimoroni BME680 probe at 0x%02X due to chip ID 0x%02X",
                addr,
                chip_id,
            )
            continue
        try:
            sensor = module.BME680(addr)  # type: ignore[arg-type]
            variant = getattr(sensor, "_variant", None)
            if variant is not None:
                if variant_high is not None and variant == variant_high:
                    provider_label = f"Pimoroni BME688 (0x{addr:02X})"
                elif variant_low is not None and variant == variant_low:
                    provider_label = f"Pimoroni BME680 (0x{addr:02X})"
                else:
                    provider_label = f"Pimoroni BME68x (0x{addr:02X})"
            else:
                provider_label = f"Pimoroni BME680 (0x{addr:02X})"
            break
        except Exception as exc:  # pragma: no cover - relies on hardware
            last_error = exc
    if sensor is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("BME680 sensor not found")

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


def _probe_pimoroni_bme280(i2c: Any, addresses: Set[int]) -> Optional[SensorProbeResult]:
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

    # Import SMBus for proper sensor initialization
    try:
        from smbus2 import SMBus
        bus = SMBus(1)
    except Exception as exc:
        logging.warning("draw_inside: failed to initialize SMBus: %s", exc)
        raise

    # Prefer the addresses we actually saw on the bus so we don't try the
    # wrong default. Fallback to the library defaults if we could not scan.
    candidate_addresses: Sequence[int]
    if addresses:
        candidate_addresses = tuple(sorted(addresses.intersection({0x76, 0x77})))
    else:
        candidate_addresses = (0x76, 0x77)

    dev = None
    successful_addr: Optional[int] = None
    last_error: Optional[Exception] = None
    for addr in candidate_addresses:
        chip_id = _read_chip_id(i2c, addr)
        if chip_id is not None and chip_id != expected_chip_id:
            logging.debug(
                "draw_inside: skipping Pimoroni BME280 probe at 0x%02X due to chip ID 0x%02X",
                addr,
                chip_id,
            )
            continue
        try:
            candidate = sensor_cls(i2c_addr=addr, i2c_dev=bus)  # type: ignore[call-arg]
            try:
                # Force an initial reading to validate connectivity. The Pimoroni
                # driver raises a RuntimeError with a helpful message if the bus
                # is not responding.
                _ = float(candidate.get_temperature())
            except Exception as exc:  # pragma: no cover - relies on hardware
                last_error = exc
                continue
            dev = candidate
            successful_addr = addr
            break
        except Exception as exc:  # pragma: no cover - relies on hardware
            last_error = exc

    fallback_dev = None
    if dev is None:
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
                fallback_dev = candidate
                successful_addr = addr
                logging.debug(
                    "draw_inside: falling back to Adafruit BME280 driver for Pimoroni sensor at 0x%02X",
                    addr,
                )
                break

    if dev is None and fallback_dev is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Pimoroni BME280 sensor not found")

    addr_for_label = successful_addr if successful_addr is not None else candidate_addresses[0]
    label = f"Pimoroni BME280 (0x{addr_for_label:02X})"

    fallback_dev: Optional[Any] = None
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


def _probe_adafruit_bme280(i2c: Any, addresses: Set[int]) -> Optional[SensorProbeResult]:
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


def _probe_adafruit_sht4x(i2c: Any, addresses: Set[int]) -> Optional[SensorProbeResult]:
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


def _get_probe_order(preference: Optional[SensorProbeName]) -> Tuple[Tuple[SensorProbeName, SensorProbeFn], ...]:
    probers: Tuple[Tuple[SensorProbeName, SensorProbeFn], ...] = (
        ("pimoroni_bme280", _probe_pimoroni_bme280),
        ("adafruit_bme280", _probe_adafruit_bme280),
        ("pimoroni_bme680", _probe_pimoroni_bme680),
        ("pimoroni_bme68x", _probe_pimoroni_bme68x),
        ("adafruit_bme680", _probe_adafruit_bme680),
        ("adafruit_sht41", _probe_adafruit_sht4x),
    )

    if not preference:
        return probers

    filtered = tuple((name, fn) for name, fn in probers if name == preference)
    if filtered:
        return filtered
    return probers


def _scan_i2c_addresses(i2c: Any) -> Set[int]:
    addresses: Set[int] = set()

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


def _probe_sensor() -> Tuple[Optional[str], Optional[Callable[[], SensorReadings]]]:
    """Try the available sensor drivers and return the first match."""

    if board is None or busio is None:
        logging.warning("BME* libs not available on this host; skipping sensor probe")
        return None, None

    try:
        i2c = busio.I2C(getattr(board, "SCL"), getattr(board, "SDA"))
    except Exception as exc:
        logging.warning("draw_inside: failed to initialise I2C bus: %s", exc)
        return None, None

    addresses = _scan_i2c_addresses(i2c)
    if addresses:
        formatted = ", ".join(f"0x{addr:02X}" for addr in sorted(addresses))
        logging.debug("draw_inside: detected I2C addresses: %s", formatted)
    else:
        logging.debug("draw_inside: no I2C addresses detected during scan")

    preference, raw_preference = _get_sensor_env_override()
    if preference:
        logging.info("draw_inside: INSIDE_SENSOR set to %s; restricting probe order", preference)
    elif raw_preference:
        logging.warning(
            "draw_inside: INSIDE_SENSOR value %r not recognized; falling back to auto-detect",
            raw_preference,
        )

    # Prefer BME280 variants before BME680/BME68x. Some BME680 drivers can
    # incorrectly initialise against a BME280 at the same address and return
    # garbage pressure values (~660 hPa instead of ~997 hPa). Trying the
    # BME280-specific probers first keeps the readings aligned with the
    # standalone BME280 CLI script. When INSIDE_SENSOR is set, only the
    # requested probe will run.
    probe_plan = _get_probe_order(preference)

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

    logging.warning("No supported indoor environmental sensor detected.")
    return None, None


def _log_sensor_data(provider: Optional[str], data: Dict[str, Optional[float]]) -> None:
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
def _mix_color(color: Tuple[int, int, int], target: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    factor = max(0.0, min(1.0, factor))
    return tuple(int(round(color[idx] * (1 - factor) + target[idx] * factor)) for idx in range(3))


def _interpolate_color(
    stops: Sequence[Tuple[float, Tuple[int, int, int]]],
    value: float,
) -> Tuple[int, int, int]:
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
    rect: Tuple[int, int, int, int],
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
    rect: Tuple[int, int, int, int],
    label: str,
    value: str,
    accent: Tuple[int, int, int],
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
    ) -> Tuple[Any, Tuple[int, int], int]:
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
    value_x = x0 + padding_x
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
    rect: Tuple[int, int, int, int],
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

    label_base_size = getattr(label_base, "size", 18)
    label_font = fit_font(
        draw,
        label,
        label_base,
        max_width=width - 2 * padding_x,
        max_height=max(12, int(height * 0.24)),
        min_pt=min(label_base_size, 10),
        max_pt=label_base_size,
    )
    label_w, label_h = measure_text(draw, label, label_font)
    label_x = x0 + padding_x
    label_y = y0 + padding_y

    descriptor = descriptor.strip()
    has_descriptor = bool(descriptor)
    if has_descriptor:
        desc_font = fit_font(
            draw,
            descriptor,
            label_base,
            max_width=width - 2 * padding_x,
            max_height=max(12, int(height * 0.22)),
            min_pt=min(label_base_size, 10),
            max_pt=label_base_size,
        )
        desc_w, desc_h = measure_text(draw, descriptor, desc_font)
        desc_x = x0 + padding_x
        desc_y = y1 - padding_y - desc_h
    else:
        desc_font = None
        desc_w, desc_h = 0, 0
        desc_x = x0 + padding_x
        desc_y = y1 - padding_y

    available_value_height = max(24, height - (label_h + desc_h + 3 * padding_y))
    value_base_size = getattr(value_base, "size", 24)
    value_font = fit_font(
        draw,
        value,
        value_base,
        max_width=width - 2 * padding_x,
        max_height=available_value_height,
        min_pt=min(value_base_size, 14),
        max_pt=value_base_size,
    )
    value_w, value_h = measure_text(draw, value, value_font)
    value_x = x0 + padding_x
    value_gap = max(8, height // 14)
    desc_gap = max(8, height // 16)
    top_limit = label_y + label_h + value_gap
    bottom_limit = desc_y - desc_gap if has_descriptor else y1 - padding_y
    available_space = bottom_limit - top_limit
    if available_space >= value_h:
        value_y = top_limit + (available_space - value_h) // 2
    else:
        value_y = min(top_limit, bottom_limit - value_h)

    label_color = _mix_color(bg, config.INSIDE_COL_TEXT, 0.3)
    value_color = config.INSIDE_COL_TEXT
    desc_color = _mix_color(bg, config.INSIDE_COL_TEXT, 0.32)

    draw.text((label_x, label_y), label, font=label_font, fill=label_color)
    draw.text((value_x, value_y), value, font=value_font, fill=value_color)
    if has_descriptor and desc_font:
        draw.text((desc_x, desc_y), descriptor, font=desc_font, fill=desc_color)


def _metric_grid_dimensions(count: int) -> Tuple[int, int]:
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
    rect: Tuple[int, int, int, int], count: int
) -> List[Tuple[int, int, int, int]]:
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

    cells: List[Tuple[int, int, int, int]] = []
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
    rect: Tuple[int, int, int, int],
    metrics: Sequence[Dict[str, Any]],
    label_base,
    value_base,
    *,
    cells: Optional[Sequence[Tuple[int, int, int, int]]] = None,
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


def _voc_quality_color(score: float) -> Tuple[int, int, int]:
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


def _build_metric_entries(data: Dict[str, Optional[float]]) -> List[Dict[str, Any]]:
    metrics: List[Dict[str, Any]] = []
    used_keys: Set[str] = set()
    used_groups: Set[str] = set()

    palette: List[Tuple[int, int, int]] = [
        config.INSIDE_CHIP_BLUE,
        config.INSIDE_CHIP_AMBER,
        config.INSIDE_CHIP_PURPLE,
        _mix_color(config.INSIDE_CHIP_BLUE, config.INSIDE_CHIP_AMBER, 0.45),
        _mix_color(config.INSIDE_CHIP_PURPLE, config.INSIDE_CHIP_BLUE, 0.4),
        _mix_color(config.INSIDE_CHIP_PURPLE, config.INSIDE_COL_BG, 0.35),
    ]

    Spec = Tuple[str, str, Callable[[float], str], Tuple[int, int, int], Optional[str]]
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


def _build_voc_tile(data: Dict[str, Optional[float]], provider: Optional[str]) -> Optional[Dict[str, Any]]:
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


def draw_inside(display, transition: bool=False):
    provider, read_fn = _probe_sensor()
    if not read_fn:
        logging.warning("draw_inside: sensor not available")
        return None

    try:
        data = read_fn()
        cleaned: Dict[str, Optional[float]] = {}
        if isinstance(data, dict):
            cleaned = {key: _clean_metric(value) for key, value in data.items()}
        else:
            logging.debug("draw_inside: unexpected data payload type %s", type(data))
            cleaned = {}
        temp_f = cleaned.get("temp_f")

        # Log the sensor data to file
        _log_sensor_data(provider, cleaned)

    except Exception as e:
        logging.warning(f"draw_inside: sensor read failed: {e}")
        return None

    if temp_f is None:
        logging.warning("draw_inside: temperature missing from sensor data")
        return None

    metrics = _build_metric_entries(cleaned)
    voc_tile = _build_voc_tile(cleaned, provider)
    if voc_tile:
        metrics = [m for m in metrics if not m["label"].lower().startswith("voc")]

    # Title text
    title = "Inside"
    subtitle = provider or ""

    # Compose canvas
    img  = Image.new("RGB", (W, H), config.INSIDE_COL_BG)
    draw = ImageDraw.Draw(img)

    # Fonts (with fallbacks)
    default_title_font = config.FONT_TITLE_SPORTS
    title_base = getattr(config, "FONT_TITLE_INSIDE", None)
    if title_base is None or getattr(title_base, "size", 0) < getattr(default_title_font, "size", 0):
        title_base = default_title_font

    subtitle_base = getattr(config, "FONT_INSIDE_SUBTITLE", None)
    default_subtitle_font = getattr(config, "FONT_DATE_SPORTS", default_title_font)
    if subtitle_base is None or getattr(subtitle_base, "size", 0) < getattr(default_subtitle_font, "size", 0):
        subtitle_base = default_subtitle_font

    temp_base  = getattr(config, "FONT_TIME",        default_title_font)
    label_base = getattr(config, "FONT_INSIDE_LABEL", getattr(config, "FONT_DATE_SPORTS", default_title_font))
    value_base = getattr(config, "FONT_INSIDE_VALUE", getattr(config, "FONT_DATE_SPORTS", default_title_font))

    # --- Title (auto-fit to width without shrinking below the standard size)
    title_side_pad = 8
    title_base_size = getattr(title_base, "size", 30)
    title_sample_h = measure_text(draw, "Hg", title_base)[1]
    title_max_h = max(1, title_sample_h)
    t_font = fit_font(
        draw,
        title,
        title_base,
        max_width=W - 2 * title_side_pad,
        max_height=title_max_h,
        min_pt=min(title_base_size, 12),
        max_pt=title_base_size,
    )
    tw, th = measure_text(draw, title, t_font)
    title_y = 0
    draw.text(((W - tw)//2, title_y), title, font=t_font, fill=config.INSIDE_COL_TITLE)

    subtitle_gap = 6
    if subtitle:
        subtitle_base_size = getattr(subtitle_base, "size", getattr(default_subtitle_font, "size", 24))
        subtitle_sample_h = measure_text(draw, "Hg", subtitle_base)[1]
        subtitle_max_h = max(1, subtitle_sample_h)
        sub_font = fit_font(
            draw,
            subtitle,
            subtitle_base,
            max_width=W - 2 * title_side_pad,
            max_height=subtitle_max_h,
            min_pt=min(subtitle_base_size, 12),
            max_pt=subtitle_base_size,
        )
        sw, sh = measure_text(draw, subtitle, sub_font)
        subtitle_y = title_y + th + subtitle_gap
        draw.text(((W - sw)//2, subtitle_y), subtitle, font=sub_font, fill=config.INSIDE_COL_TITLE)
    else:
        sub_font = t_font
        sw, sh = 0, 0
        subtitle_y = title_y + th

    title_block_h = subtitle_y + (sh if subtitle else 0)

    # --- Temperature panel --------------------------------------------------
    temp_value = f"{temp_f:.1f}°F"
    descriptor = ""

    content_top = title_block_h + 12
    bottom_margin = 12
    side_pad = 12
    content_bottom = H - bottom_margin
    content_height = max(1, content_bottom - content_top)

    metric_count = len(metrics)
    _, grid_rows = _metric_grid_dimensions(metric_count)
    total_rows = grid_rows + (1 if voc_tile else 0)
    if metric_count or voc_tile:
        temp_ratio = max(0.42, 0.58 - 0.03 * min(metric_count, 6))
        min_temp = max(84, 118 - 8 * min(metric_count, 6))
    else:
        temp_ratio = 0.82
        min_temp = 128

    temp_height = min(content_height, max(min_temp, int(content_height * temp_ratio)))
    metric_block_gap = 12 if (metric_count or voc_tile) else 0
    if metric_count or voc_tile:
        min_metric_row_height = 44
        min_metric_gap = 10 if total_rows > 1 else 0
        target_metrics_height = (
            total_rows * min_metric_row_height + max(0, total_rows - 1) * min_metric_gap
        )
        preferred_temp_cap = content_height - (target_metrics_height + metric_block_gap)
        min_temp_floor = min(54, content_height)
        preferred_temp_cap = max(min_temp_floor, preferred_temp_cap)
        temp_height = min(temp_height, preferred_temp_cap)
        temp_height = max(min_temp_floor, min(temp_height, content_height))
    else:
        metric_block_gap = 0
    temp_rect = (
        side_pad,
        content_top,
        W - side_pad,
        min(content_bottom, content_top + temp_height),
    )

    _draw_temperature_panel(
        img,
        draw,
        temp_rect,
        temp_f,
        temp_value,
        descriptor,
        temp_base,
        label_base,
    )

    if metric_count or voc_tile:
        metrics_rect = (
            side_pad,
            min(content_bottom, temp_rect[3] + metric_block_gap),
            W - side_pad,
            content_bottom,
        )
        metrics_x0, metrics_y0, metrics_x1, metrics_y1 = metrics_rect
        metrics_height = max(0, metrics_y1 - metrics_y0)
        total_rows = grid_rows + (1 if voc_tile else 0)
        if total_rows > 1:
            desired_v_gap = max(8, metrics_height // 30)
            max_v_gap = max(0, (metrics_height - total_rows) // (total_rows - 1))
            v_gap = min(desired_v_gap, max_v_gap)
        else:
            v_gap = 0
        available_height = max(total_rows, metrics_height - v_gap * (total_rows - 1))
        row_height = max(44, available_height // total_rows) if total_rows else 0
        if total_rows and row_height * total_rows + v_gap * (total_rows - 1) > metrics_height:
            row_height = max(1, available_height // total_rows)
        grid_height = min(metrics_height, row_height * total_rows + v_gap * (total_rows - 1)) if total_rows else 0
        start_y = metrics_y0 + max(0, (metrics_height - grid_height) // 2)

        if metric_count and grid_rows:
            metrics_area_height = row_height * grid_rows + v_gap * max(0, grid_rows - 1)
            metrics_area_rect = (
                metrics_x0,
                start_y,
                metrics_x1,
                min(metrics_y1, start_y + metrics_area_height),
            )
            metric_cells = _metric_grid_cells(metrics_area_rect, metric_count)
            if metric_cells:
                _draw_metric_rows(
                    draw,
                    metrics_area_rect,
                    metrics,
                    label_base,
                    value_base,
                    cells=metric_cells,
                )

        if voc_tile and total_rows:
            voc_top = start_y + (row_height * grid_rows) + (v_gap * max(0, grid_rows))
            voc_rect = (
                metrics_x0,
                min(metrics_y1, voc_top),
                metrics_x1,
                min(metrics_y1, voc_top + row_height),
            )
            if voc_rect[3] > voc_rect[1]:
                _draw_voc_tile(
                    draw,
                    voc_rect,
                    voc_tile["label"],
                    voc_tile["value"],
                    voc_tile["descriptor"],
                    voc_tile["score"],
                    label_base,
                    value_base,
                )

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
