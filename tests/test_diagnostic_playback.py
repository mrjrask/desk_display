import diagnostic_playback


def test_save_load_and_clear_diagnostic_screen(tmp_path, monkeypatch):
    path = tmp_path / "diagnostic.json"
    monkeypatch.setenv("DESK_DISPLAY_DIAGNOSTIC_CONTROL_PATH", str(path))

    assert diagnostic_playback.save_diagnostic_screen("NEWS HEADLINES") == "news headlines"
    assert diagnostic_playback.load_diagnostic_screen() == "news headlines"
    assert diagnostic_playback.save_diagnostic_screen(None) is None
    assert diagnostic_playback.load_diagnostic_screen() is None


def test_rejects_unknown_diagnostic_screen(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "DESK_DISPLAY_DIAGNOSTIC_CONTROL_PATH", str(tmp_path / "diagnostic.json")
    )

    try:
        diagnostic_playback.save_diagnostic_screen("not a screen")
    except ValueError as exc:
        assert "Unknown screen" in str(exc)
    else:
        raise AssertionError("unknown screen should be rejected")
