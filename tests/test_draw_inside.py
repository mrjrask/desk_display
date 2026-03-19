import math
import sys
import types

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
    assert _parse_i2c_bus_candidates() == (13, 14, 15)


def test_probe_sensor_uses_pimoroni_bme68x_without_blinka(monkeypatch):
    import screens.draw_inside as draw_inside_module

    monkeypatch.setenv("INSIDE_SENSOR", "pimoroni_bme68x")
    monkeypatch.setattr(draw_inside_module, "board", None)
    monkeypatch.setattr(draw_inside_module, "busio", None)

    reader = lambda: {"temp_f": 70.0}
    monkeypatch.setattr(
        draw_inside_module,
        "_probe_pimoroni_bme68x",
        lambda _i2c, _addresses: ("Pimoroni BME688", reader),
    )

    provider, probe_reader = draw_inside_module._probe_sensor()

    assert provider == "Pimoroni BME688"
    assert probe_reader is reader




def test_probe_sensor_falls_back_to_auto_detect_when_preference_fails(monkeypatch):
    import screens.draw_inside as draw_inside_module

    monkeypatch.setenv("INSIDE_SENSOR", "pimoroni_bme68x")
    monkeypatch.setattr(draw_inside_module, "board", None)
    monkeypatch.setattr(draw_inside_module, "busio", None)

    reader = lambda: {"temp_f": 70.0}

    def fake_get_probe_order(preference):
        if preference == "pimoroni_bme68x":
            return (("pimoroni_bme68x", lambda _i2c, _addresses: None),)
        return (("pimoroni_bme280", lambda _i2c, _addresses: ("Pimoroni BME280", reader)),)

    monkeypatch.setattr(draw_inside_module, "_get_probe_order", fake_get_probe_order)

    provider, probe_reader = draw_inside_module._probe_sensor()

    assert provider == "Pimoroni BME280"
    assert probe_reader is reader

def test_probe_sensor_attempts_smbus_probes_without_blinka(monkeypatch):
    import screens.draw_inside as draw_inside_module

    monkeypatch.delenv("INSIDE_SENSOR", raising=False)
    monkeypatch.setattr(draw_inside_module, "board", None)
    monkeypatch.setattr(draw_inside_module, "busio", None)

    reader = lambda: {"temp_f": 70.0}

    monkeypatch.setattr(
        draw_inside_module,
        "_get_probe_order",
        lambda _preference: (("pimoroni_bme680", lambda _i2c, _addresses: ("Pimoroni BME680", reader)),),
    )

    provider, probe_reader = draw_inside_module._probe_sensor()

    assert provider == "Pimoroni BME680"
    assert probe_reader is reader


def test_probe_pimoroni_bme680_reads_chip_id_from_each_bus(monkeypatch):
    import importlib
    import screens.draw_inside as draw_inside_module

    class FakeBus:
        def __init__(self, bus_num):
            self.bus_num = bus_num

        def read_byte_data(self, _addr, _register):
            if self.bus_num == 15:
                return 0x61
            return 0x58

    class FakeSensor:
        def __init__(self, _addr, i2c_device):
            self._i2c_device = i2c_device
            self._variant = None

        def get_sensor_data(self):
            return False

    fake_driver = types.SimpleNamespace(
        CHIP_ID=0x61,
        I2C_ADDR_PRIMARY=0x76,
        I2C_ADDR_SECONDARY=0x77,
        BME680=FakeSensor,
    )

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_driver if name in ("pimoroni_bme680", "bme680") else None,
    )
    monkeypatch.setattr(draw_inside_module, "_resolve_i2c_bus_number", lambda _i2c: 13)
    monkeypatch.setattr(draw_inside_module, "_parse_i2c_bus_candidates", lambda: (13, 15))
    monkeypatch.setattr(draw_inside_module, "_read_chip_id", lambda _i2c, _addr: 0x58)
    monkeypatch.setitem(sys.modules, "smbus2", types.SimpleNamespace(SMBus=FakeBus))

    provider, _reader = draw_inside_module._probe_pimoroni_bme680(object(), {0x76})

    assert provider == "Pimoroni BME680 (bus 15, 0x76)"


def test_draw_inside_returns_placeholder_image_when_sensor_unavailable(monkeypatch):
    import screens.draw_inside as draw_inside_module

    monkeypatch.setattr(draw_inside_module, "_probe_sensor", lambda: (None, None))

    image = draw_inside_module.draw_inside(None, transition=True)

    assert image is not None
    assert image.size == (draw_inside_module.W, draw_inside_module.H)


def test_probe_sensor_cached_reuses_result_until_forced(monkeypatch):
    import screens.draw_inside as draw_inside_module

    calls = {"count": 0}

    def fake_probe_sensor():
        calls["count"] += 1
        return "Fake Sensor", lambda: {"temp_f": 70.0}

    monkeypatch.setattr(draw_inside_module, "_probe_sensor", fake_probe_sensor)
    monkeypatch.setattr(draw_inside_module, "_sensor_probe_cache", None)

    first = draw_inside_module._probe_sensor_cached()
    second = draw_inside_module._probe_sensor_cached()
    refreshed = draw_inside_module._probe_sensor_cached(force_refresh=True)

    assert first[0] == "Fake Sensor"
    assert second[0] == "Fake Sensor"
    assert refreshed[0] == "Fake Sensor"
    assert calls["count"] == 2


def test_is_inside_sensor_available_reflects_probe_result(monkeypatch):
    import screens.draw_inside as draw_inside_module

    monkeypatch.setattr(
        draw_inside_module,
        "_probe_sensor_cached",
        lambda force_refresh=False: ("Pimoroni BME280", lambda: {}),
    )
    assert draw_inside_module.is_inside_sensor_available() is True

    monkeypatch.setattr(
        draw_inside_module,
        "_probe_sensor_cached",
        lambda force_refresh=False: (None, None),
    )
    assert draw_inside_module.is_inside_sensor_available() is False


def test_probe_pimoroni_bme68x_survives_child_segfault(monkeypatch):
    import subprocess
    import screens.draw_inside as draw_inside_module

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=[], returncode=-11, stdout="", stderr="")

    monkeypatch.setattr(draw_inside_module.subprocess, "run", fake_run)

    try:
        draw_inside_module._probe_pimoroni_bme68x(None, set())
    except RuntimeError as exc:
        assert "signal 11" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when helper process segfaults")
