from screens_catalog import RAW_SCREEN_IDS


def test_raw_screen_ids_are_unique():
    assert len(RAW_SCREEN_IDS) == len(set(RAW_SCREEN_IDS))
