from screens_catalog import RAW_SCREEN_IDS


def test_raw_screen_ids_are_unique():
    assert len(RAW_SCREEN_IDS) == len(set(RAW_SCREEN_IDS))


def test_weather_quad_screen_id_is_listed():
    assert "weather quad" in RAW_SCREEN_IDS


def test_mlb_schedule_quad_screen_ids_are_listed():
    assert "cubs schedule quad" in RAW_SCREEN_IDS
    assert "sox schedule quad" in RAW_SCREEN_IDS
