from PIL import Image

import main
from schedule import build_scheduler


class _FakeDisplay:
    def apply_indicator_border(self, img):
        return img


def test_refresh_alt_screenshots_saves_configured_alternate(monkeypatch):
    scheduler = build_scheduler(
        {
            "screens": {
                "date": {
                    "frequency": 1,
                    "alt": {"screen": "nixie", "frequency": 3},
                },
                "nixie": 0,
            }
        }
    )
    monkeypatch.setattr(main, "screen_scheduler", scheduler)
    monkeypatch.setattr(main, "ENABLE_SCREENSHOTS", True)
    monkeypatch.setattr(main, "display", _FakeDisplay())

    frame = Image.new("RGB", (10, 10), "black")
    monkeypatch.setitem(main._ALT_SCREENSHOT_REFRESHERS, "nixie", lambda: frame)

    saved_calls = []

    def _fake_save_screenshot(sid, img):
        saved_calls.append((sid, img))
        return ("nixie", False, 0)

    monkeypatch.setattr(main, "_save_screenshot", _fake_save_screenshot)
    monkeypatch.setattr(main, "maybe_archive_screenshots", lambda folder: None)

    main._refresh_alt_screenshots("date")

    assert saved_calls == [("nixie", frame)]


def test_refresh_alt_screenshots_noop_without_alt_config(monkeypatch):
    scheduler = build_scheduler({"screens": {"date": 1, "inside": 2}})
    monkeypatch.setattr(main, "screen_scheduler", scheduler)
    monkeypatch.setattr(main, "ENABLE_SCREENSHOTS", True)

    calls = []
    monkeypatch.setattr(main, "_save_screenshot", lambda sid, img: calls.append(sid))

    main._refresh_alt_screenshots("date")

    assert calls == []


def test_refresh_alt_screenshots_noop_when_screenshots_disabled(monkeypatch):
    scheduler = build_scheduler(
        {
            "screens": {
                "date": {
                    "frequency": 1,
                    "alt": {"screen": "nixie", "frequency": 3},
                },
                "nixie": 0,
            }
        }
    )
    monkeypatch.setattr(main, "screen_scheduler", scheduler)
    monkeypatch.setattr(main, "ENABLE_SCREENSHOTS", False)

    calls = []
    monkeypatch.setattr(main, "_save_screenshot", lambda sid, img: calls.append(sid))

    main._refresh_alt_screenshots("date")

    assert calls == []


def test_select_screenshot_image_prefers_full_scroll_capture():
    display_frame = Image.new("RGB", (320, 240), "black")
    full_scroll = Image.new("RGB", (320, 900), "blue")

    selected = main._select_screenshot_image(display_frame, full_scroll)

    assert selected is full_scroll
    assert selected.size == (320, 900)


def test_select_screenshot_image_falls_back_to_display_frame():
    display_frame = Image.new("RGB", (320, 240), "black")

    selected = main._select_screenshot_image(display_frame, None)

    assert selected is display_frame
    assert selected.size == (320, 240)
