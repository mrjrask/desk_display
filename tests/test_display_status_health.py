import json

from PIL import Image

import main


def test_write_display_status_includes_system_health_when_metrics_missing(monkeypatch, tmp_path):
    status_path = tmp_path / "current" / "display_status.json"
    status_path.parent.mkdir()

    monkeypatch.setattr(main, "DISPLAY_STATUS_PATH", str(status_path))
    monkeypatch.setattr(main, "SCREENSHOT_DIR", str(tmp_path))
    monkeypatch.setattr(main, "CURRENT_SCREENSHOT_DIR", str(status_path.parent))
    monkeypatch.setattr(main, "display", object())
    monkeypatch.setattr(main, "cache", {})
    monkeypatch.setattr(main, "get_system_health", lambda _path: {})

    main._write_display_status(
        "date",
        Image.new("RGB", (2, 2), "black"),
        loop_iteration=3,
    )

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["screen_id"] == "date"
    assert payload["loop_iteration"] == 3
    assert payload["system_health"] == {}


def test_write_display_status_survives_system_health_failure(monkeypatch, tmp_path):
    status_path = tmp_path / "current" / "display_status.json"
    status_path.parent.mkdir()

    monkeypatch.setattr(main, "DISPLAY_STATUS_PATH", str(status_path))
    monkeypatch.setattr(main, "SCREENSHOT_DIR", str(tmp_path))
    monkeypatch.setattr(main, "CURRENT_SCREENSHOT_DIR", str(status_path.parent))
    monkeypatch.setattr(main, "display", object())
    monkeypatch.setattr(main, "cache", {})

    def _raise(_path):
        raise OSError("health unavailable")

    monkeypatch.setattr(main, "get_system_health", _raise)

    main._write_display_status(
        "date",
        Image.new("RGB", (2, 2), "white"),
        loop_iteration=4,
    )

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["screen_id"] == "date"
    assert payload["system_health"] == {}
