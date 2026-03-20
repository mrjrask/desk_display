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


def test_ltr559_reader_supports_alternate_method_names(monkeypatch):
    class FakeLTR:
        @staticmethod
        def get_als():
            return 42.5

        @staticmethod
        def get_ps():
            return 9

    monkeypatch.setattr(draw_sensors, "ltr559", FakeLTR)
    monkeypatch.setattr(draw_sensors, "_build_device", lambda *args, **kwargs: (None, None))

    reader = draw_sensors.LTR559Reader()
    assert reader.ok is True
    assert reader.sample() == (42.5, 9)


def test_imu_reader_supports_getter_style_api(monkeypatch):
    class FakeIMUDevice:
        def get_acceleration(self):
            return (0.0, 0.0, 1.0)

        def get_gyroscope(self):
            return (1.0, 2.0, 3.0)

    monkeypatch.setattr(draw_sensors, "_IMU", object())
    monkeypatch.setattr(draw_sensors, "_build_device", lambda *args, **kwargs: (FakeIMUDevice(), 10))

    reader = draw_sensors.IMUReader()
    assert reader.ok is True
    accel_mag, rot_z = reader.sample()
    assert accel_mag == 1.0
    assert rot_z == 3.0
