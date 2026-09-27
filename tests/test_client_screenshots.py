"""A display client writes the screenshots and heartbeat the config UI reads."""
from __future__ import annotations

import datetime
import json

from PIL import Image

from remote_display.client_screenshots import ClientScreenshots


class Ticker:
    def __init__(self):
        self.now = datetime.datetime(2026, 9, 27, 12, 0, 0, tzinfo=datetime.timezone.utc)

    def __call__(self):
        self.now += datetime.timedelta(seconds=1)
        return self.now


def _writer(tmp_path, **kwargs):
    return ClientScreenshots(tmp_path / "shots", profile_id="hyperpixel4_square", width=720, height=720,
                             now=Ticker(), **kwargs)


def test_record_writes_history_current_and_heartbeat(tmp_path):
    shots = _writer(tmp_path)
    shots.record("NFL Standings/AFC", Image.new("RGB", (720, 720), (255, 0, 0)))

    history = list((tmp_path / "shots" / "NFL Standings-AFC").glob("*.png"))
    assert [p.name for p in history] == ["NFL_Standings-AFC_20260927_120001.png"]
    current = tmp_path / "shots" / "current" / "NFL_Standings-AFC.png"
    with Image.open(current) as image:
        assert image.getpixel((0, 0)) == (255, 0, 0)
    status = json.loads((tmp_path / "shots" / "current" / "display_status.json").read_text())
    assert status["screen_id"] == "NFL Standings/AFC"
    assert status["loop_iteration"] == 1
    assert status["display"] == {"profile_id": "hyperpixel4_square", "width": 720, "height": 720}
    assert datetime.datetime.fromisoformat(status["rendered_at"]).tzinfo is not None


def test_history_is_pruned_per_screen(tmp_path):
    shots = _writer(tmp_path, max_per_screen=2)
    for shade in range(4):
        shots.record("date", Image.new("L", (4, 4), shade))
    names = sorted(p.name for p in (tmp_path / "shots" / "date").iterdir())
    assert names == ["date_20260927_120003.png", "date_20260927_120004.png"]


def test_disabled_screenshots_still_write_the_heartbeat(tmp_path):
    shots = _writer(tmp_path, enabled=False)
    shots.record("date", Image.new("L", (4, 4)))
    assert not (tmp_path / "shots" / "date").exists()
    assert not (tmp_path / "shots" / "current" / "date.png").exists()
    assert json.loads((tmp_path / "shots" / "current" / "display_status.json").read_text())["screen_id"] == "date"


def test_a_write_failure_never_reaches_playback(tmp_path):
    blocker = tmp_path / "shots"
    blocker.write_text("not a directory")
    ClientScreenshots(blocker, profile_id="p", width=1, height=1).record("date", Image.new("L", (1, 1)))


def test_from_settings_honours_enable_screenshots(tmp_path, monkeypatch):
    from display_profiles import PROFILE_PRESETS

    monkeypatch.setenv("SCREENSHOT_DIR", str(tmp_path / "shots"))
    monkeypatch.setenv("SCREENSHOT_ARCHIVE_BASE", str(tmp_path / "archive"))
    profile = PROFILE_PRESETS["hyperpixel4"]
    on = ClientScreenshots.from_settings({"ENABLE_SCREENSHOTS": True}, profile)
    off = ClientScreenshots.from_settings({"ENABLE_SCREENSHOTS": "0"}, profile)
    assert on.enabled and not off.enabled
    assert on.screenshot_dir == tmp_path / "shots"
    assert on.display["width"] == profile.width


def test_heartbeat_counts_plays_per_screen(tmp_path):
    shots = _writer(tmp_path)
    for screen in ("date", "weather1", "date"):
        shots.record(screen, Image.new("L", (4, 4)))
    status = json.loads((tmp_path / "shots" / "current" / "display_status.json").read_text())
    assert status["screen_play_counts"] == {"date": 2, "weather1": 1}
    assert status["loop_iteration"] == 3


def test_a_standalone_ticker_sidecar_is_cleared(tmp_path):
    current = tmp_path / "shots" / "current"
    current.mkdir(parents=True)
    (current / "news.ticker.json").write_text('{"headlines": ["old"]}')
    (current / "sports.ticker.json").write_text('{"headlines": ["other"]}')
    _writer(tmp_path).record("news", Image.new("L", (4, 4)))
    assert not (current / "news.ticker.json").exists()
    assert (current / "sports.ticker.json").exists()
