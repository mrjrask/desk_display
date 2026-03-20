import screens.draw_sensors as draw_sensors


def test_parse_i2c_bus_candidates_prefers_explicit_sensor_stick_bus(monkeypatch):
    monkeypatch.setenv("PIM_SENSOR_STICK_I2C_BUS", "10")
    monkeypatch.setenv("INSIDE_I2C_BUSES", "1,2")
    assert draw_sensors._parse_i2c_bus_candidates() == (10,)


def test_parse_i2c_bus_candidates_uses_inside_i2c_buses(monkeypatch):
    monkeypatch.delenv("PIM_SENSOR_STICK_I2C_BUS", raising=False)
    monkeypatch.setenv("INSIDE_I2C_BUSES", "10,11,11,bad")
    assert draw_sensors._parse_i2c_bus_candidates() == (10, 11)

