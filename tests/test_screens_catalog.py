from screens_catalog import RAW_SCREEN_IDS


def test_raw_screen_ids_are_unique():
    assert len(RAW_SCREEN_IDS) == len(set(RAW_SCREEN_IDS))


def test_weather_quad_screen_id_is_listed():
    assert "weather quad" in RAW_SCREEN_IDS


def test_astronomical_screen_is_listed_after_weather_daily():
    assert "astronomical" in RAW_SCREEN_IDS
    assert RAW_SCREEN_IDS.index("astronomical") == RAW_SCREEN_IDS.index("weather daily") + 1


def test_mlb_schedule_quad_screen_ids_are_listed():
    assert "cubs schedule quad" in RAW_SCREEN_IDS
    assert "sox schedule quad" in RAW_SCREEN_IDS


def test_nba_nhl_schedule_quad_screen_ids_are_listed():
    assert "bulls schedule quad" in RAW_SCREEN_IDS
    assert "hawks schedule quad" in RAW_SCREEN_IDS
