import os
from datetime import UTC, datetime, timedelta
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
    monkeypatch.setattr(
        config_ui,
        "_load_service_status",
        lambda: {"unit": "desk_display.service", "summary": "active (running), enabled", "is_active": True, "error": None},
    )

    client = config_ui.app.test_client()
    response = client.get("/screenshots")

    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'class="timestamp is-stale"' in html
    assert "1d 2h 3m 4s ago" in html
    assert 'id="hideMissingScreens" checked' in html
    assert 'const hideMissingStorageKey = "deskDisplay.hideMissingScreens";' in html
    assert 'window.localStorage.setItem(hideMissingStorageKey, shouldHide ? "true" : "false");' in html
    assert 'data-screen-play-counter="date">Plays 5<' in html


def test_screenshots_api_returns_entries(monkeypatch):
    monkeypatch.setattr(
        config_ui,
        "_build_screenshot_entries",
        lambda: [{"id": "date", "path": "current/date.png", "timestamp": "2025-01-01 00:00:00", "elapsed": "0d 0h 0m 5s ago", "version": 1, "is_stale": False}],
    )
    monkeypatch.setattr(config_ui, "_load_display_status", lambda: {"screen_id": "date", "is_stale": False})
    monkeypatch.setattr(
        config_ui,
        "_load_service_status",
        lambda: {"unit": "desk_display.service", "summary": "active (running), enabled", "is_active": True, "error": None},
    )

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
        "service_status": {
            "unit": "desk_display.service",
            "summary": "active (running), enabled",
            "is_active": True,
            "error": None,
        },
    }


def test_layout_editor_routes_removed():
    client = config_ui.app.test_client()

    assert client.get("/layouts").status_code == 404
    assert client.get("/api/layouts").status_code == 404


def test_screenshots_template_removes_layout_editor_nav_link(monkeypatch):
    monkeypatch.setattr(config_ui, "_build_screenshot_entries", list)
    monkeypatch.setattr(
        config_ui,
        "_load_service_status",
        lambda: {"unit": "desk_display.service", "summary": "active (running), enabled", "is_active": True, "error": None},
    )

    client = config_ui.app.test_client()
    response = client.get("/screenshots")

    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Layout Editor" not in html


def test_screen_config_page_renders_service_indicator(monkeypatch):
    monkeypatch.setattr(config_ui, "_load_active_config", lambda: {"screens": {"date": 1}})
    monkeypatch.setattr(config_ui, "_load_active_style_config", lambda: {"screens": {}})
    monkeypatch.setattr(config_ui, "_load_active_layouts_config", lambda: {"screens": {"quad": {}}})
    monkeypatch.setattr(
        config_ui,
        "_build_screen_entries",
        lambda config, style: [{"id": "date", "frequency": 1, "background": "", "alt_screen": "", "alt_frequency": ""}],
    )
    monkeypatch.setattr(config_ui, "_build_playlist_assignments", lambda config: ([], {}))
    monkeypatch.setattr(
        config_ui,
        "_load_service_status",
        lambda: {"unit": "desk_display.service", "summary": "active (running), enabled", "is_active": True, "error": None},
    )

    client = config_ui.app.test_client()
    response = client.get("/")

    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Service <strong>desk_display.service</strong>:" in html
    assert "active (running), enabled" in html


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


def test_screenshot_order_matches_config_page_order(monkeypatch, tmp_path):
    current_dir = tmp_path / "current"
    current_dir.mkdir()

    config = {"screens": {"weather": {}, "date": {}, "news headlines": {}}}

    monkeypatch.setattr(
        config_ui,
        "resolve_storage_paths",
        lambda **kwargs: SimpleNamespace(screenshot_dir=tmp_path, current_screenshot_dir=current_dir),
    )
    monkeypatch.setattr(config_ui, "_load_active_config", lambda: config)

    config_page_order = [
        entry["id"] for entry in config_ui._build_screen_entries(config, {})
    ]
    screenshot_page_order = [
        entry["id"] for entry in config_ui._build_screenshot_entries()
    ]
    visible_screenshot_order = [
        screen_id for screen_id in screenshot_page_order if screen_id in config_page_order
    ]

    assert visible_screenshot_order == config_page_order


def test_screenshot_order_matches_playlist_grouped_config_order(monkeypatch, tmp_path):
    current_dir = tmp_path / "current"
    current_dir.mkdir()

    config = {
        "screens": {
            "date": {},
            "nixie": {},
            "on this day": {},
            "news headlines": {},
            "weather logo": {},
            "weather1": {},
        },
        "playlists": {
            "starter": {"label": "starter", "steps": [{"screen": "date"}, {"screen": "nixie"}]},
            "weather": {
                "label": "weather",
                "steps": [{"screen": "weather logo"}, {"screen": "weather1"}],
            },
            "other": {
                "label": "Other",
                "steps": [{"screen": "on this day"}, {"screen": "news headlines"}],
            },
        },
        "sequence": [
            {"playlist": "starter"},
            {"playlist": "weather"},
            {"playlist": "other"},
        ],
    }

    monkeypatch.setattr(
        config_ui,
        "resolve_storage_paths",
        lambda **kwargs: SimpleNamespace(screenshot_dir=tmp_path, current_screenshot_dir=current_dir),
    )
    monkeypatch.setattr(config_ui, "_load_active_config", lambda: config)

    entries = config_ui._build_screenshot_entries()
    order = [entry["id"] for entry in entries if entry["id"] in config["screens"]]

    # The Config page groups rows by playlist (in sequence order), not raw
    # dict order, so screens assigned to a later playlist ("Other") must
    # appear after screens in earlier playlists ("starter", "weather") even
    # though "on this day" and "news headlines" come earlier in the raw
    # screens dict.
    assert order == ["date", "nixie", "weather logo", "weather1", "on this day", "news headlines"]


def test_build_screenshot_entries_includes_oled_when_present(monkeypatch, tmp_path):
    current_dir = tmp_path / "current"
    current_dir.mkdir()
    (current_dir / "oled_left.png").write_bytes(b"x")
    (current_dir / "oled_right.png").write_bytes(b"x")

    monkeypatch.setattr(
        config_ui,
        "resolve_storage_paths",
        lambda **kwargs: SimpleNamespace(screenshot_dir=tmp_path, current_screenshot_dir=current_dir),
    )
    monkeypatch.setattr(config_ui, "_load_active_config", lambda: {"screens": {"date": {}}})

    entries = config_ui._build_screenshot_entries()
    entry_map = {entry["id"]: entry for entry in entries}

    assert entry_map["oled left"]["path"] == "current/oled_left.png"
    assert entry_map["oled right"]["path"] == "current/oled_right.png"


def test_build_screenshot_entries_hides_oled_when_absent(monkeypatch, tmp_path):
    current_dir = tmp_path / "current"
    current_dir.mkdir()

    monkeypatch.setattr(
        config_ui,
        "resolve_storage_paths",
        lambda **kwargs: SimpleNamespace(screenshot_dir=tmp_path, current_screenshot_dir=current_dir),
    )
    monkeypatch.setattr(config_ui, "_load_active_config", lambda: {"screens": {"date": {}}})

    entries = config_ui._build_screenshot_entries()
    entry_map = {entry["id"]: entry for entry in entries}

    assert entry_map["oled left"]["path"] is None
    assert entry_map["oled right"]["path"] is None


def test_load_display_status_reads_heartbeat(monkeypatch, tmp_path):
    current_dir = tmp_path / "current"
    current_dir.mkdir()
    rendered_at = datetime.now(UTC) - timedelta(seconds=30)
    (current_dir / "display_status.json").write_text(
        """{
  "screen_id": "bears next season",
  "loop_iteration": 42,
  "rendered_at": "%s",
  "frame_id": 314,
  "display": {"profile_id": "hyperpixel4", "width": 800, "height": 480},
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
    assert status["display"] == {"profile_id": "hyperpixel4", "width": 800, "height": 480}
    assert status["screen_play_counts"] == {"bears next season": 12}
    assert status["is_stale"] is False
    assert status["elapsed"] is not None
