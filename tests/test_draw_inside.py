import math

from screens.draw_inside import (
    _build_metric_entries,
    _build_voc_tile,
    _iter_board_i2c_pin_pairs,
    _get_probe_order,
    _get_sensor_env_override,
    _normalize_sensor_name,
    _normalize_pressure,
    _parse_i2c_bus_candidates,
)


def test_normalize_pressure_returns_hpa_and_inhg():
    pres_hpa, pres_inhg = _normalize_pressure(101325)
    assert math.isclose(pres_hpa, 1013.25, rel_tol=1e-4)
    assert math.isclose(pres_inhg, 29.92, rel_tol=1e-3)


def test_build_metric_entries_prefers_inhg():
    data = {
        "pressure_hpa": 1013.2,
        "pressure_inhg": 29.92,
    }

    metrics = _build_metric_entries(data)
    assert metrics, "Expected at least one metric entry"
    first_metric = metrics[0]
    assert first_metric["label"] == "Pressure"
    assert "inHg" in first_metric["value"]


def test_build_metric_entries_omits_pressure_hpa():
    data = {"pressure_hpa": 1013.2}

    metrics = _build_metric_entries(data)

    assert metrics == []


def test_build_voc_tile_includes_bme680_providers():
    data = {"voc_ohms": 12_000.0}

    voc_tile = _build_voc_tile(data, "Adafruit BME680")

    assert voc_tile, "Expected VOC tile to be built when VOC data is present"
    assert voc_tile["label"] == "VOC"
    assert "kΩ" in voc_tile["value"]


def test_build_voc_tile_uses_bsec_voc_index():
    data = {"voc_index": 125.0}

    voc_tile = _build_voc_tile(data, "Pimoroni BME688")

    assert voc_tile, "Expected VOC tile to render from BSEC VOC index"
    assert voc_tile["label"] == "VOC Index"
    assert voc_tile["value"].startswith("125")


def test_normalize_sensor_env_value_handles_spacing_and_case():
    assert _normalize_sensor_name(" Pimoroni-BME680 ") == "pimoroni_bme680"


def test_sensor_env_override_supports_aliases(monkeypatch):
    monkeypatch.setenv("INSIDE_SENSOR", "Adafruit-SHT4X")
    preference, raw = _get_sensor_env_override()
    assert raw == "Adafruit-SHT4X"
    assert preference == "adafruit_sht41"




def test_sensor_env_override_ignores_inline_comments(monkeypatch):
    monkeypatch.setenv("INSIDE_SENSOR", "pimoroni_bme68x # Pimoroni breakout")
    preference, raw = _get_sensor_env_override()
    assert raw == "pimoroni_bme68x # Pimoroni breakout"
    assert preference == "pimoroni_bme68x"

def test_probe_order_restricts_to_preference():
    plan = _get_probe_order("pimoroni_bme280")
    assert plan and plan[0][0] == "pimoroni_bme280"
    assert all(name == "pimoroni_bme280" for name, _ in plan)


def test_i2c_pin_pairs_include_hyperpixel_bus_10_mapping():
    pairs = _iter_board_i2c_pin_pairs()
    assert pairs[0] == ("SCL", "SDA")
    assert ("D45", "D44") in pairs


def test_parse_i2c_bus_candidates_defaults_include_hyperpixel_buses(monkeypatch):
    monkeypatch.delenv("INSIDE_I2C_BUSES", raising=False)
    assert _parse_i2c_bus_candidates() == (10, 11, 1, 0)


def test_draw_inside_returns_placeholder_image_when_sensor_unavailable(monkeypatch):
    import screens.draw_inside as draw_inside_module

    monkeypatch.setattr(draw_inside_module, "_probe_sensor", lambda: (None, None))

    image = draw_inside_module.draw_inside(None, transition=True)

    assert image is not None
    assert image.size == (draw_inside_module.W, draw_inside_module.H)
