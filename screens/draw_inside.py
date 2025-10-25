#!/usr/bin/env python3
"""
draw_inside.py (RGB, 320x240)

Universal environmental sensor screen with a compact, modern layout:
  • Title (detects and names: Adafruit/Pimoroni BME680, BME688, BME280, SHT41)
  • Metric tiles that auto-fit the screen:
      Temperature / Humidity / Pressure (inHg) / VOC (only if data available)
Everything is dynamically sized to fit the configured canvas without clipping.
"""

from __future__ import annotations
import time
import logging
import math
import os
import sys
from typing import Any, Callable, Dict, Optional, Set, Tuple

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

# ── Tile drawing (LABEL top-left | VALUE centered) ───────────────────────────
def _draw_tile(
    draw: ImageDraw.ImageDraw,
    rect: Tuple[int, int, int, int],
    bg: Tuple[int, int, int],
    label: str,
    value: str,
    label_base,
    value_base,
    *,
    label_color=config.INSIDE_COL_TEXT,
    value_color=config.INSIDE_COL_TEXT,
    value_min_pt: int = 12,
):
    x0, y0, x1, y1 = rect
    radius = max(10, min(18, (y1 - y0) // 3))
    draw.rounded_rectangle(rect, radius=radius, fill=bg, outline=config.INSIDE_COL_STROKE)

    pad_x, pad_y = 10, 8
    inner_w = max(0, (x1 - x0) - 2 * pad_x)
    inner_h = max(0, (y1 - y0) - 2 * pad_y)

    label_max_h = max(12, int(inner_h * 0.35))
    value_max_h = max(value_min_pt, int(inner_h * 0.78))

    lf = fit_font(
        draw,
        label,
        label_base,
        max_width=inner_w,
        max_height=label_max_h,
        min_pt=8,
        max_pt=label_max_h + 2,
    )
    vf = fit_font(
        draw,
        value,
        value_base,
        max_width=inner_w,
        max_height=value_max_h,
        min_pt=value_min_pt,
        max_pt=value_max_h + 4,
    )

    lw, lh = measure_text(draw, label, lf)
    vw, vh = measure_text(draw, value, vf)

    label_x = x0 + pad_x
    label_y = y0 + pad_y
    value_x = x0 + ((x1 - x0) - vw) // 2
    value_y = y1 - pad_y - vh

    draw.text((label_x, label_y), label, font=lf, fill=label_color)
    draw.text((value_x, value_y), value, font=vf, fill=value_color)

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


def draw_inside(display, transition: bool=False):
    provider, read_fn = _probe_sensor()
    if not read_fn:
        logging.warning("draw_inside: sensor not available")
        return None

    try:
        data = read_fn()
        temp_f = _clean_metric(data.get("temp_f"))
        hum = _clean_metric(data.get("humidity"))
        pres = _clean_metric(data.get("pressure_inhg"))
        voc = _clean_metric(data.get("voc_ohms"))
    except Exception as e:
        logging.warning(f"draw_inside: sensor read failed: {e}")
        return None

    if temp_f is None:
        logging.warning("draw_inside: temperature missing from sensor data")
        return None

    # Title text
    title = "Inside"
    subtitle = provider or ""

    # Compose canvas
    img  = Image.new("RGB", (W, H), config.INSIDE_COL_BG)
    draw = ImageDraw.Draw(img)

    # Fonts (with fallbacks)
    title_base = getattr(config, "FONT_TITLE_INSIDE", config.FONT_TITLE_SPORTS)
    temp_base  = getattr(config, "FONT_TIME",        config.FONT_TITLE_SPORTS)
    label_base = getattr(config, "FONT_INSIDE_LABEL", getattr(config, "FONT_DATE_SPORTS", config.FONT_TITLE_SPORTS))
    value_base = getattr(config, "FONT_INSIDE_VALUE", getattr(config, "FONT_DATE_SPORTS", config.FONT_TITLE_SPORTS))

    # --- Title (auto-fit to width, compact height)
    title_max_h = 12
    t_font = fit_font(
        draw,
        title,
        title_base,
        max_width=W - 8,
        max_height=title_max_h,
        min_pt=9,
        max_pt=title_max_h + 2,
    )
    tw, th = measure_text(draw, title, t_font)
    title_y = 0
    draw.text(((W - tw)//2, title_y), title, font=t_font, fill=config.INSIDE_COL_TITLE)

    subtitle_gap = 4
    subtitle_max_h = 10
    if subtitle:
        sub_font = fit_font(
            draw,
            subtitle,
            title_base,
            max_width=W - 8,
            max_height=subtitle_max_h,
            min_pt=8,
            max_pt=subtitle_max_h + 1,
        )
        sw, sh = measure_text(draw, subtitle, sub_font)
        draw.text(((W - sw)//2, title_y + th + subtitle_gap), subtitle, font=sub_font, fill=config.INSIDE_COL_TITLE)
    else:
        sub_font = t_font
        sw, sh = 0, 0

    title_block_h = th + (subtitle_gap + sh if subtitle else 0)

    # --- Metric tiles -------------------------------------------------------
    temp_bg = tuple(min(255, int(c * 0.8 + 40)) for c in temperature_color(temp_f))
    temperature_card = dict(
        label="Temperature",
        value=f"{temp_f:.1f}°F",
        bg=temp_bg,
        value_min_pt=30,
    )

    metric_cards = []
    if hum is not None:
        metric_cards.append(
            dict(
                label="Humidity",
                value=f"{hum:.1f}%",
                bg=config.INSIDE_CHIP_BLUE,
            )
        )
    if pres is not None:
        metric_cards.append(
            dict(
                label="Pressure",
                value=f"{pres:.2f} inHg",
                bg=config.INSIDE_CHIP_AMBER,
            )
        )
    if voc is not None:
        metric_cards.append(
            dict(
                label="VOC",
                value=format_voc_ohms(voc),
                bg=config.INSIDE_CHIP_PURPLE,
            )
        )

    tiles_top = title_block_h + 10
    bottom_margin = 8
    tiles_gap = 6
    side_pad = 8
    tiles_h_avail = max(30, H - bottom_margin - tiles_top)

    # Temperature tile gets priority real estate at the top.
    extra_gap = tiles_gap if metric_cards else 0
    if metric_cards:
        temp_height_ratio = 0.5 if len(metric_cards) <= 1 else 0.56
    else:
        temp_height_ratio = 0.85
    max_temp_height = max(1, tiles_h_avail - extra_gap)
    temp_tile_h = max(70, int(tiles_h_avail * temp_height_ratio))
    temp_tile_h = min(temp_tile_h, max_temp_height)
    min_target = 60 if max_temp_height >= 60 else max_temp_height
    temp_tile_h = max(temp_tile_h, min_target)

    temp_rect = (
        side_pad,
        tiles_top,
        W - side_pad,
        tiles_top + temp_tile_h,
    )

    _draw_tile(
        draw,
        temp_rect,
        temperature_card["bg"],
        temperature_card["label"],
        temperature_card["value"],
        label_base,
        temp_base,
        value_min_pt=temperature_card["value_min_pt"],
    )

    if metric_cards:
        metrics_top = temp_rect[3] + extra_gap
        metrics_h_avail = max(30, H - bottom_margin - metrics_top)
        metric_count = len(metric_cards)
        cols = 1 if metric_count == 1 else 2
        rows = math.ceil(metric_count / cols)

        tile_w = max(80, (W - 2 * side_pad - (cols - 1) * tiles_gap) // cols)
        tile_h = max(50, (metrics_h_avail - (rows - 1) * tiles_gap) // rows)

        total_height = rows * tile_h + (rows - 1) * tiles_gap
        y_offset = metrics_top + max(0, (metrics_h_avail - total_height) // 2)

        for idx, card in enumerate(metric_cards):
            row = idx // cols
            col = idx % cols

            x0 = side_pad + col * (tile_w + tiles_gap)
            y0 = y_offset + row * (tile_h + tiles_gap)
            rect = (x0, y0, x0 + tile_w, y0 + tile_h)

            _draw_tile(
                draw,
                rect,
                card["bg"],
                card["label"],
                card["value"],
                label_base,
                value_base,
                value_min_pt=card.get("value_min_pt", 12),
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
