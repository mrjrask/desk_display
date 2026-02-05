import os
from datetime import datetime

import config_ui


def test_build_screenshot_entries_marks_stale(monkeypatch, tmp_path):
    stale_file = tmp_path / "date.png"
    stale_file.write_bytes(b"x")
    stale_timestamp = datetime.now().timestamp() - (2 * 60 * 60 + 5)
    fresh_timestamp = datetime.now().timestamp() - 60

    fresh_file = tmp_path / "weather.png"
    fresh_file.write_bytes(b"x")

    monkeypatch.setattr(config_ui, "_current_screenshot_dir", lambda: tmp_path)
    monkeypatch.setattr(config_ui, "_load_active_config", lambda: {"screens": {"date": {}, "weather": {}}})

    os.utime(stale_file, (stale_timestamp, stale_timestamp))
    os.utime(fresh_file, (fresh_timestamp, fresh_timestamp))

    entries = config_ui._build_screenshot_entries()
    entry_map = {entry["id"]: entry for entry in entries}

    assert entry_map["date"]["is_stale"] is True
    assert entry_map["weather"]["is_stale"] is False


def test_screenshots_template_adds_stale_class(monkeypatch):
    monkeypatch.setattr(
        config_ui,
        "_build_screenshot_entries",
        lambda: [
            {"id": "date", "filename": "date.png", "timestamp": "2025-01-01 00:00:00", "is_stale": True},
            {"id": "weather", "filename": "weather.png", "timestamp": "2025-01-01 01:30:00", "is_stale": False},
        ],
    )

    client = config_ui.app.test_client()
    response = client.get("/screenshots")

    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'class="timestamp is-stale"' in html
