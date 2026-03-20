import screens.draw_sensors as draw_sensors


def test_parse_i2c_bus_candidates_prefers_explicit_sensor_stick_bus(monkeypatch):
    monkeypatch.setenv("PIM_SENSOR_STICK_I2C_BUS", "10")
    monkeypatch.setenv("INSIDE_I2C_BUSES", "1,2")
    assert draw_sensors._parse_i2c_bus_candidates() == (10,)


def test_parse_i2c_bus_candidates_uses_inside_i2c_buses(monkeypatch):
    monkeypatch.delenv("PIM_SENSOR_STICK_I2C_BUS", raising=False)
    monkeypatch.setenv("INSIDE_I2C_BUSES", "10,11,11,bad")
    assert draw_sensors._parse_i2c_bus_candidates() == (10, 11)


def test_parse_i2c_bus_candidates_defaults_include_hyperpixel_buses(monkeypatch):
    monkeypatch.delenv("PIM_SENSOR_STICK_I2C_BUS", raising=False)
    monkeypatch.delenv("INSIDE_I2C_BUSES", raising=False)
    assert draw_sensors._parse_i2c_bus_candidates() == (1, 2, 10, 11, 13, 14, 15)


def test_ltr559_reader_falls_back_to_module_level_api(monkeypatch):
    class FakeLTR:
        @staticmethod
        def get_lux():
            return 123.4

        @staticmethod
        def get_proximity():
            return 56

    monkeypatch.setattr(draw_sensors, "ltr559", FakeLTR)
    monkeypatch.setattr(draw_sensors, "_build_device", lambda *args, **kwargs: (None, None))

    reader = draw_sensors.LTR559Reader()
    assert reader.ok is True
    assert reader.sample() == (123.4, 56)
