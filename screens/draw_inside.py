#!/usr/bin/env python3
"""
draw_inside.py (RGB, 320x240)

Universal environmental sensor screen with a streamlined layout:
  • Title (detects and names: Adafruit/Pimoroni BME680, BME688, BME280, SHT41)
  • Highlighted temperature panel with contextual descriptor
  • Linear stat strip showing humidity / pressure / VOC (if available)
Everything is dynamically sized to stay legible on the configured canvas.
"""

from __future__ import annotations
import time
import logging
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image, ImageDraw
import config
from utils import (
    clear_display,
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

    dev = adafruit_bme680.Adafruit_BME680_I2C(i2c)

    def read() -> SensorReadings:
        temp_f = float(dev.temperature) * 9 / 5 + 32
        hum = float(dev.humidity)
        pres = float(dev.pressure) * 0.02953
        gas = getattr(dev, "gas", None)
        voc = float(gas) if gas not in (None, 0) else None
        return dict(temp_f=temp_f, humidity=hum, pressure_inhg=pres, voc_ohms=voc)

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
        voc_raw = _extract_field(data, "gas_resistance")

        temp_f = temp_c * 9 / 5 + 32 if temp_c is not None else None
        pres = None
        if pres_raw is not None:
            pres_hpa = pres_raw / 100.0 if pres_raw > 2000 else pres_raw
            pres = pres_hpa * 0.02953

        voc = voc_raw if voc_raw not in (None, 0) else None

        if temp_f is None:
            raise RuntimeError("BME68X temperature reading missing")

        return dict(temp_f=temp_f, humidity=hum, pressure_inhg=pres, voc_ohms=voc)

    return provider, read


def _probe_pimoroni_bme680(_i2c: Any, addresses: Set[int]) -> Optional[SensorProbeResult]:
    if addresses and not addresses.intersection({0x76, 0x77}):
        return None

    from importlib import import_module

    try:
        bme680 = import_module("pimoroni_bme680")  # type: ignore
    except ModuleNotFoundError:
        bme680 = import_module("bme680")  # type: ignore

    try:
        sensor = bme680.BME680(getattr(bme680, "I2C_ADDR_PRIMARY", 0x76))  # type: ignore
    except Exception:
        sensor = bme680.BME680()  # type: ignore

    for method, value in (
        ("set_humidity_oversample", getattr(bme680, "OS_2X", None)),
        ("set_pressure_oversample", getattr(bme680, "OS_4X", None)),
        ("set_temperature_oversample", getattr(bme680, "OS_8X", None)),
        ("set_filter", getattr(bme680, "FILTER_SIZE_3", None)),
        ("set_gas_status", getattr(bme680, "ENABLE_GAS_MEAS", None)),
    ):
        fn = getattr(sensor, method, None)
        if callable(fn) and value is not None:
            try:
                fn(value)
            except Exception:
                pass

    gas_temp = getattr(bme680, "DEFAULT_GAS_HEATER_TEMPERATURE", getattr(bme680, "GAS_HEATER_TEMP", None))
    gas_dur = getattr(bme680, "DEFAULT_GAS_HEATER_DURATION", getattr(bme680, "GAS_HEATER_DURATION", None))
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
        pres = float(pres_raw) * 0.02953 if pres_raw is not None else None
        voc = float(gas) if gas not in (None, 0) and heat_stable else None
        hum_val = float(hum) if hum is not None else None

        if temp_f is None:
            raise RuntimeError("BME680 temperature reading missing")

        return dict(temp_f=temp_f, humidity=hum_val, pressure_inhg=pres, voc_ohms=voc)

    return "Pimoroni BME68X", read


def _probe_pimoroni_bme280(_i2c: Any, addresses: Set[int]) -> Optional[SensorProbeResult]:
    if addresses and not addresses.intersection({0x76, 0x77}):
        return None

    import bme280  # type: ignore

    dev = bme280.BME280()

    def read() -> SensorReadings:
        temp_f = float(dev.get_temperature()) * 9 / 5 + 32
        hum = float(dev.get_humidity())
        pres = float(dev.get_pressure()) * 0.02953
        return dict(temp_f=temp_f, humidity=hum, pressure_inhg=pres, voc_ohms=None)

    return "Pimoroni BME280", read


def _probe_adafruit_bme280(i2c: Any, addresses: Set[int]) -> Optional[SensorProbeResult]:
    if addresses and not addresses.intersection({0x76, 0x77}):
        return None

    import adafruit_bme280  # type: ignore

    dev = adafruit_bme280.Adafruit_BME280_I2C(i2c)

    def read() -> SensorReadings:
        temp_f = float(dev.temperature) * 9 / 5 + 32
        hum = float(dev.humidity)
        pres = float(dev.pressure) * 0.02953
        return dict(temp_f=temp_f, humidity=hum, pressure_inhg=pres, voc_ohms=None)

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

    probers: Tuple[SensorProbeFn, ...] = (
        _probe_adafruit_bme680,
        _probe_pimoroni_bme68x,
        _probe_pimoroni_bme680,
        _probe_adafruit_sht4x,
        _probe_pimoroni_bme280,
        _probe_adafruit_bme280,
    )

    for probe in probers:
        try:
            result = probe(i2c, addresses)
        except ModuleNotFoundError as exc:
            logging.debug("draw_inside: probe %s skipped (module missing): %s", probe.__name__, exc)
            continue
        except Exception as exc:  # pragma: no cover - relies on hardware
            logging.debug("draw_inside: probe %s failed: %s", probe.__name__, exc, exc_info=True)
            continue
        if result:
            provider, reader = result
            logging.info("draw_inside: detected %s", provider)
            return provider, reader

    logging.warning("No supported indoor environmental sensor detected.")
    return None, None

# ── Layout helpers ───────────────────────────────────────────────────────────
def _mix_color(color: Tuple[int, int, int], target: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    factor = max(0.0, min(1.0, factor))
    return tuple(int(round(color[idx] * (1 - factor) + target[idx] * factor)) for idx in range(3))


def _lighten(color: Tuple[int, int, int], amount: float) -> Tuple[int, int, int]:
    return _mix_color(color, (255, 255, 255), amount)


def _darken(color: Tuple[int, int, int], amount: float) -> Tuple[int, int, int]:
    return _mix_color(color, (0, 0, 0), amount)


def _draw_gradient_panel(
    img: Image.Image,
    rect: Tuple[int, int, int, int],
    start_color: Tuple[int, int, int],
    end_color: Tuple[int, int, int],
    radius: int,
    outline: Optional[Tuple[int, int, int]] = None,
) -> None:
    x0, y0, x1, y1 = rect
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)

    gradient = Image.new("RGB", (width, height))
    grad_draw = ImageDraw.Draw(gradient)
    for y in range(height):
        blend = y / (height - 1) if height > 1 else 0
        color = _mix_color(start_color, end_color, blend)
        grad_draw.line((0, y, width, y), fill=color)

    mask = Image.new("L", (width, height), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=radius, fill=255)
    img.paste(gradient, (x0, y0), mask)

    if outline is not None:
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle(rect, radius=radius, outline=outline, width=1)


def _describe_temperature(temp_f: float) -> str:
    if temp_f < 60:
        return "Chilly"
    if temp_f < 68:
        return "Cool"
    if temp_f < 75:
        return "Comfortable"
    if temp_f < 82:
        return "Warm"
    if temp_f < 90:
        return "Toasty"
    return "Hot"


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
    start = _lighten(color, 0.72)
    end = _darken(color, 0.22)
    outline = _mix_color(color, (0, 0, 0), 0.35)
    radius = max(14, min(26, (y1 - y0) // 4))
    _draw_gradient_panel(img, rect, start, end, radius=radius, outline=outline)

    width = max(1, x1 - x0)
    height = max(1, y1 - y0)

    label_base_size = getattr(label_base, "size", 18)
    label_font = fit_font(
        draw,
        "Temperature",
        label_base,
        max_width=width - 32,
        max_height=max(14, int(height * 0.22)),
        min_pt=min(label_base_size, 10),
        max_pt=label_base_size,
    )
    label_w, label_h = measure_text(draw, "Temperature", label_font)
    label_pos = (x0 + 18, y0 + max(10, height // 12))

    temp_base_size = getattr(temp_base, "size", 48)
    temp_font = fit_font(
        draw,
        temp_text,
        temp_base,
        max_width=width - 40,
        max_height=max(32, int(height * 0.6)),
        min_pt=min(temp_base_size, 20),
        max_pt=temp_base_size,
    )
    temp_w, temp_h = measure_text(draw, temp_text, temp_font)
    temp_pos = (
        x0 + (width - temp_w) // 2,
        y0 + max(label_h + 18, (height - temp_h) // 2),
    )

    desc_font = fit_font(
        draw,
        descriptor,
        label_base,
        max_width=width - 40,
        max_height=max(12, int(height * 0.22)),
        min_pt=min(label_base_size, 12),
        max_pt=label_base_size,
    )
    desc_w, desc_h = measure_text(draw, descriptor, desc_font)
    desc_pos = (x0 + (width - desc_w) // 2, y1 - desc_h - max(14, height // 12))

    label_color = _mix_color(start, (0, 0, 0), 0.65)
    temp_color = config.INSIDE_COL_TEXT
    desc_color = _mix_color(start, (0, 0, 0), 0.45)

    draw.text(label_pos, "Temperature", font=label_font, fill=label_color)
    draw.text(temp_pos, temp_text, font=temp_font, fill=temp_color)
    draw.text(desc_pos, descriptor, font=desc_font, fill=desc_color)

    accent_y = desc_pos[1] - 8
    accent_start = x0 + 18
    accent_end = x1 - 18
    if accent_end > accent_start and accent_y > y0 + label_h + 8:
        draw.line(
            (accent_start, accent_y, accent_end, accent_y),
            fill=_mix_color(start, (255, 255, 255), 0.25),
            width=1,
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
    height = max(1, y1 - y0)
    radius = max(6, min(14, height // 2))
    bg = _lighten(accent, 0.82)
    draw.rounded_rectangle(rect, radius=radius, fill=bg)

    accent_w = max(4, min(8, int(max(1, x1 - x0) * 0.08)))
    draw.rectangle((x0, y0, x0 + accent_w, y1), fill=_mix_color(accent, (0, 0, 0), 0.15))

    inner_left = x0 + accent_w + 10
    inner_right = x1 - 10
    inner_width = max(1, inner_right - inner_left)

    label_base_size = getattr(label_base, "size", 18)
    value_base_size = getattr(value_base, "size", 24)

    label_font = fit_font(
        draw,
        label,
        label_base,
        max_width=int(inner_width * 0.6),
        max_height=max(12, int(height * 0.48)),
        min_pt=min(label_base_size, 10),
        max_pt=label_base_size,
    )
    value_font = fit_font(
        draw,
        value,
        value_base,
        max_width=inner_width,
        max_height=max(16, int(height * 0.62)),
        min_pt=min(value_base_size, 12),
        max_pt=value_base_size,
    )

    lw, lh = measure_text(draw, label, label_font)
    vw, vh = measure_text(draw, value, value_font)

    label_x = inner_left
    label_y = y0 + (height - lh) // 2
    value_x = inner_right - vw
    value_y = y0 + (height - vh) // 2

    midpoint = y0 + height // 2
    line_start = label_x + lw + 8
    line_end = value_x - 12
    if line_end > line_start:
        draw.line(
            (line_start, midpoint, line_end, midpoint),
            fill=_mix_color(accent, (255, 255, 255), 0.55),
            width=1,
        )

    label_color = _mix_color(accent, (0, 0, 0), 0.45)
    value_color = config.INSIDE_COL_TEXT

    draw.text((label_x, label_y), label, font=label_font, fill=label_color)
    draw.text((value_x, value_y), value, font=value_font, fill=value_color)


def _draw_metric_rows(
    draw: ImageDraw.ImageDraw,
    rect: Tuple[int, int, int, int],
    metrics: Sequence[Dict[str, Any]],
    label_base,
    value_base,
) -> None:
    x0, y0, x1, y1 = rect
    height = max(0, y1 - y0)
    count = len(metrics)
    if count <= 0 or height <= 0:
        return
    columns = 1 if count <= 3 else 2
    gap = 6
    column_gap = 10 if columns > 1 else 0
    rows = int(math.ceil(count / columns))
    total_vertical_gap = gap * (rows - 1)
    row_h = 0
    if rows > 0:
        available_height = max(1, height - total_vertical_gap)
        row_h = max(22, available_height // rows)

    available_width = max(1, x1 - x0)
    total_horizontal_gap = column_gap * (columns - 1)
    col_w = max(40, (available_width - total_horizontal_gap) // columns)

    for idx, metric in enumerate(metrics):
        row = idx // columns
        col = idx % columns
        left = x0 + col * (col_w + column_gap)
        top = y0 + row * (row_h + gap)
        if top >= y1:
            break
        bottom = min(y1, top + row_h)
        right = min(x1, left + col_w)
        if bottom <= top or right <= left:
            continue
        _draw_metric_row(
            draw,
            (left, top, right, bottom),
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
    if key_lower.endswith("_hpa"):
        return f"{value:.1f} hPa"
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"

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
        ("pressure_inhg", "Pressure", lambda v: f"{v:.2f} inHg", config.INSIDE_CHIP_AMBER, "pressure"),
        ("pressure_hpa", "Pressure", lambda v: f"{v:.1f} hPa", config.INSIDE_CHIP_AMBER, "pressure"),
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

    skip_keys = {"temp", "temperature"}
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
    except Exception as e:
        logging.warning(f"draw_inside: sensor read failed: {e}")
        return None

    if temp_f is None:
        logging.warning("draw_inside: temperature missing from sensor data")
        return None

    metrics = _build_metric_entries(cleaned)

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
    descriptor = _describe_temperature(temp_f)

    content_top = title_block_h + 12
    bottom_margin = 12
    side_pad = 12
    content_bottom = H - bottom_margin
    content_height = max(1, content_bottom - content_top)

    metric_count = len(metrics)
    if metric_count:
        if metric_count <= 2:
            temp_ratio = 0.56
            min_temp = 104
        elif metric_count <= 3:
            temp_ratio = 0.52
            min_temp = 96
        else:
            temp_ratio = 0.48
            min_temp = 90
    else:
        temp_ratio = 0.86
        min_temp = 118

    temp_height = min(content_height, max(min_temp, int(content_height * temp_ratio)))
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

    if metrics:
        metrics_rect = (
            side_pad,
            min(content_bottom, temp_rect[3] + 12),
            W - side_pad,
            content_bottom,
        )
        _draw_metric_rows(draw, metrics_rect, metrics, label_base, value_base)

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
