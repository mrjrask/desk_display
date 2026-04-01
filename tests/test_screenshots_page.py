import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import config_ui


def test_build_screenshot_entries_marks_stale(monkeypatch, tmp_path):
    current_dir = tmp_path / "current"
    current_dir.mkdir()

    stale_file = current_dir / "date.png"
    stale_file.write_bytes(b"x")
    stale_timestamp = datetime.now().timestamp() - (2 * 60 * 60 + 5)
    fresh_timestamp = datetime.now().timestamp() - 60

    fresh_file = current_dir / "weather.png"
    fresh_file.write_bytes(b"x")

    monkeypatch.setattr(
        config_ui,
        "resolve_storage_paths",
        lambda **kwargs: SimpleNamespace(screenshot_dir=tmp_path, current_screenshot_dir=current_dir),
    )
    monkeypatch.setattr(config_ui, "_load_active_config", lambda: {"screens": {"date": {}, "weather": {}}})

    os.utime(stale_file, (stale_timestamp, stale_timestamp))
    os.utime(fresh_file, (fresh_timestamp, fresh_timestamp))

    entries = config_ui._build_screenshot_entries()
    entry_map = {entry["id"]: entry for entry in entries}

    assert entry_map["date"]["is_stale"] is True
    assert entry_map["weather"]["is_stale"] is False
    assert entry_map["date"]["elapsed"] is not None
    assert entry_map["weather"]["version"] is not None


def test_screenshots_template_adds_stale_class(monkeypatch):
    monkeypatch.setattr(
        config_ui,
        "_build_screenshot_entries",
        lambda: [
            {
                "id": "date",
                "path": "current/date.png",
                "timestamp": "2025-01-01 00:00:00",
                "elapsed": "1d 2h 3m 4s ago",
                "version": 1735689600,
                "is_stale": True,
            },
            {
                "id": "weather",
                "path": "current/weather.png",
                "timestamp": "2025-01-01 01:30:00",
                "elapsed": "0d 0h 10m 0s ago",
                "version": 1735695000,
                "is_stale": False,
            },
        ],
    )
    monkeypatch.setattr(
        config_ui,
        "_load_display_status",
        lambda: {"screen_id": "date", "loop_iteration": 7, "screen_play_counts": {"date": 5}, "is_stale": False},
    )

    client = config_ui.app.test_client()
    response = client.get("/screenshots")

    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'class="timestamp is-stale"' in html
    assert "1d 2h 3m 4s ago" in html
    assert 'id="hideMissingScreens" checked' in html
    assert 'data-screen-play-counter="date">Plays 5<' in html


def test_screenshots_api_returns_entries(monkeypatch):
    monkeypatch.setattr(
        config_ui,
        "_build_screenshot_entries",
        lambda: [{"id": "date", "path": "current/date.png", "timestamp": "2025-01-01 00:00:00", "elapsed": "0d 0h 0m 5s ago", "version": 1, "is_stale": False}],
    )
    monkeypatch.setattr(config_ui, "_load_display_status", lambda: {"screen_id": "date", "is_stale": False})

    client = config_ui.app.test_client()
    response = client.get("/api/screenshots")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload == {
        "screens": [
            {
                "id": "date",
                "path": "current/date.png",
                "timestamp": "2025-01-01 00:00:00",
                "elapsed": "0d 0h 0m 5s ago",
                "version": 1,
                "is_stale": False,
            }
        ],
        "display_status": {"screen_id": "date", "is_stale": False},
    }


def test_layout_editor_routes_removed():
    client = config_ui.app.test_client()

    assert client.get("/layouts").status_code == 404
    assert client.get("/api/layouts").status_code == 404


def test_screenshots_template_removes_layout_editor_nav_link(monkeypatch):
    monkeypatch.setattr(config_ui, "_build_screenshot_entries", lambda: [])

    client = config_ui.app.test_client()
    response = client.get("/screenshots")

    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Layout Editor" not in html


def test_build_screenshot_entries_falls_back_to_latest_screen_folder(monkeypatch, tmp_path):
    current_dir = tmp_path / "current"
    current_dir.mkdir()
    screen_dir = tmp_path / "travel map"
    screen_dir.mkdir()
    latest = screen_dir / "travel_map_20260101_120000.png"
    latest.write_bytes(b"x")

    monkeypatch.setattr(
        config_ui,
        "resolve_storage_paths",
        lambda **kwargs: SimpleNamespace(screenshot_dir=tmp_path, current_screenshot_dir=current_dir),
    )
    monkeypatch.setattr(config_ui, "_load_active_config", lambda: {"screens": {"travel map": {}}})

    entries = config_ui._build_screenshot_entries()
    entry_map = {entry["id"]: entry for entry in entries}

    assert entry_map["travel map"]["path"] == "travel map/travel_map_20260101_120000.png"


def test_load_display_status_reads_heartbeat(monkeypatch, tmp_path):
    current_dir = tmp_path / "current"
    current_dir.mkdir()
    rendered_at = datetime.now(timezone.utc) - timedelta(seconds=30)
    (current_dir / "display_status.json").write_text(
        """{
  "screen_id": "bears next season",
  "loop_iteration": 42,
  "rendered_at": "%s",
  "frame_id": 314,
  "screen_play_counts": {"bears next season": 12}
}
""" % rendered_at.isoformat(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        config_ui,
        "resolve_storage_paths",
        lambda **kwargs: SimpleNamespace(screenshot_dir=tmp_path, current_screenshot_dir=current_dir),
    )

    status = config_ui._load_display_status()

    assert status["screen_id"] == "bears next season"
    assert status["loop_iteration"] == 42
    assert status["frame_id"] == 314
    assert status["screen_play_counts"] == {"bears next season": 12}
    assert status["is_stale"] is False
    assert status["elapsed"] is not None
