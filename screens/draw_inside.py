#!/usr/bin/env python3
"""
draw_inside.py (RGB, 320x240)

Universal environmental sensor screen with a compact, modern layout:
  • Title (detects and names: Adafruit/Pimoroni BME680, BME688, BME280, SHT41)
  • Temperature (auto-fit)
  • Three rounded "chips": Humidity / Pressure (inHg) / VOC (or N/A if missing)
Everything is dynamically sized to fit the configured canvas without clipping.
"""

from __future__ import annotations
import time
import logging
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

# ── Chip drawing (LABEL left | VALUE right) ──────────────────────────────────
def _chip_lr(draw: ImageDraw.ImageDraw, rect: Tuple[int,int,int,int],
             bg: Tuple[int,int,int], label: str, value: str,
             label_base, value_base, center_gap_min=10,
             pad_x=8, pad_y=4):
    x0,y0,x1,y1 = rect
    # Slightly smaller radius to suit the shorter chips
    radius = max(7, min(12, (y1 - y0)//2))
    draw.rounded_rectangle(rect, radius=radius, fill=bg, outline=config.INSIDE_COL_STROKE)

    ix0, iy0 = x0 + pad_x, y0 + pad_y
    ix1, iy1 = x1 - pad_x, y1 - pad_y
    iw, ih   = max(0, ix1 - ix0), max(0, iy1 - iy0)

    # Split into left/right areas with a guaranteed center gap
    left_w  = int(iw * 0.46)
    right_w = int(iw * 0.46)
    gap = iw - (left_w + right_w)
    if gap < center_gap_min:
        shrink = (center_gap_min - gap + 1) // 2
        left_w  = max(16, left_w  - shrink)
        right_w = max(16, right_w - shrink)

    lx0, lx1 = ix0, ix0 + left_w
    rx1, rx0 = ix1, ix1 - right_w  # (rx0 < rx1)

    lw_max, lh_max = max(8, lx1 - lx0), max(10, ih)
    vw_max, vh_max = max(8, rx1 - rx0), max(10, ih)

    # Fit fonts tighter to shorter chips
    lf = fit_font(draw, label, label_base, lw_max, lh_max, min_pt=8,  max_pt=int(ih*0.62))
    vf = fit_font(draw, value, value_base, vw_max, vh_max, min_pt=9,  max_pt=int(ih*0.68))

    lw, lh = measure_text(draw, label, lf)
    vw, vh = measure_text(draw, value, vf)

    ly = iy0 + (ih - lh)//2
    vy = iy0 + (ih - vh)//2
    draw.text((lx0,      ly), label, font=lf, fill=config.INSIDE_COL_TEXT)
    draw.text((rx1 - vw, vy), value, font=vf, fill=config.INSIDE_COL_TEXT)

# ── Main render ──────────────────────────────────────────────────────────────
def draw_inside(display, transition: bool=False):
    provider, read_fn = _probe_sensor()
    if not read_fn:
        logging.warning("draw_inside: sensor not available")
        return None

    try:
        data = read_fn()
        temp_f = float(data["temp_f"])
        hum_raw = data.get("humidity")
        hum = float(hum_raw) if hum_raw is not None else None
        pres_raw = data.get("pressure_inhg")
        pres = float(pres_raw) if pres_raw is not None else None
        voc_raw = data.get("voc_ohms", None)
        voc = float(voc_raw) if voc_raw is not None else None
    except Exception as e:
        logging.warning(f"draw_inside: sensor read failed: {e}")
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
    draw.text(((W - tw)//2, 0), title, font=t_font, fill=config.INSIDE_COL_TITLE)

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
        draw.text(((W - sw)//2, th), subtitle, font=sub_font, fill=config.INSIDE_COL_TITLE)
    else:
        sub_font = t_font
        sw, sh = 0, 0

    # --- Temperature (auto-fit into a bounded block; slightly smaller)
    temp_txt     = f"{temp_f:.1f}°F"
    temp_area_h  = 22  # ↓ was 24; frees more space for shorter chips
    T_font = fit_font(
        draw,
        temp_txt,
        temp_base,
        max_width=W - 10,
        max_height=temp_area_h,
        min_pt=12,
        max_pt=temp_area_h + 2,
    )
    ttw, tth = measure_text(draw, temp_txt, T_font)
    title_block_h = th + (sh if subtitle else 0)
    t_y = title_block_h + 2 + max(0, (temp_area_h - tth)//2)
    draw.text(((W - ttw)//2, t_y), temp_txt, font=T_font, fill=temperature_color(temp_f))

    # --- Chips region (shorter chips, tighter gaps, smaller bottom margin)
    top_after_temp = title_block_h + 2 + temp_area_h
    bottom_margin  = 2      # ↓ was 3
    gap            = 2      # ↓ was 3
    chips_h_avail  = max(18, H - bottom_margin - top_after_temp)
    chip_h         = (chips_h_avail - 2*gap) // 3
    chip_h         = max(18, min(22, chip_h))  # ↓ clamp to 18–22px
    chips_total    = 3*chip_h + 2*gap
    chips_top      = H - bottom_margin - chips_total

    # Values
    hum_val = f"{hum:.1f}%" if hum is not None else "N/A"
    prs_val = f"{pres:.2f} inHg" if pres is not None else "N/A"
    voc_val = format_voc_ohms(voc)

    chips = [
        ("Humidity", hum_val, config.INSIDE_CHIP_BLUE),
        ("Pressure", prs_val, config.INSIDE_CHIP_AMBER),
        ("VOC",      voc_val, config.INSIDE_CHIP_PURPLE),
    ]

    side_pad = 6
    inner_w  = W - 2*side_pad
    y = chips_top
    for label, value, bg in chips:
        rect = (side_pad, y, side_pad + inner_w, y + chip_h)
        _chip_lr(draw, rect, bg, label, value, label_base, value_base,
                 center_gap_min=10, pad_x=8, pad_y=4)
        y += chip_h + gap

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
