from PIL import Image

import main
from schedule import build_scheduler
from screens.registry import ScreenDefinition


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


def test_refresh_alt_screenshots_does_not_advance_alternate_schedule(monkeypatch):
    scheduler = build_scheduler(
        {
            "screens": {
                "date": {
                    "frequency": 1,
                    "alt": {"screen": ["nixie", "weather1"], "frequency": 2},
                },
                "nixie": 0,
                "weather1": 0,
            }
        }
    )
    entry = scheduler._entries[0]
    alternate = entry.alternate
    assert alternate is not None
    monkeypatch.setattr(main, "screen_scheduler", scheduler)
    monkeypatch.setattr(main, "ENABLE_SCREENSHOTS", True)
    monkeypatch.setattr(main, "display", _FakeDisplay())
    monkeypatch.setattr(
        main,
        "_ALT_SCREENSHOT_REFRESHERS",
        {
            "nixie": lambda: Image.new("RGB", (10, 10), "black"),
            "weather1": lambda: Image.new("RGB", (10, 10), "blue"),
        },
    )
    monkeypatch.setattr(main, "_save_screenshot", lambda *_args: ("screen", False, 0))

    state_before = (entry.presentation_count, alternate.cursor, scheduler._cycle_number)
    main._refresh_alt_screenshots("date")

    assert (entry.presentation_count, alternate.cursor, scheduler._cycle_number) == state_before


def _registry(*screen_ids):
    colors = {"date": "red", "nixie": "black", "weather1": "blue", "inside": "green"}
    return {
        screen_id: ScreenDefinition(
            id=screen_id,
            render=lambda screen_id=screen_id: Image.new(
                "RGB", (2, 2), colors.get(screen_id, "white")
            ),
        )
        for screen_id in screen_ids
    }


def test_cycle_one_can_capture_every_available_positive_frequency_base(monkeypatch):
    scheduler = build_scheduler(
        {
            "screens": {
                "date": {"frequency": 1, "alt": {"screen": "nixie", "frequency": 1}},
                "nixie": 0,
                "weather1": 3,
                "inside": 2,
            }
        }
    )
    registry = _registry("date", "nixie", "weather1", "inside")
    captures = []
    monkeypatch.setattr(main, "_save_screenshot", lambda sid, image: captures.append((sid, image)))

    for _ in range(3):
        entry = scheduler.next_available(registry)
        assert entry is not None
        main._save_screenshot(entry.id, entry.render())

    assert [screen_id for screen_id, _image in captures] == ["date", "weather1", "inside"]
    assert [image.getpixel((0, 0)) for _screen_id, image in captures] == [
        (255, 0, 0),
        (0, 0, 255),
        (0, 128, 0),
    ]
    assert scheduler._cycle_number == 1


def test_frequency_zero_alternate_does_not_replace_base_during_cycle_one():
    scheduler = build_scheduler(
        {
            "screens": {
                "date": {"frequency": 1, "alt": {"screen": "nixie", "frequency": 1}},
                "nixie": 0,
            }
        }
    )

    first = scheduler.next_available(_registry("date", "nixie"))

    assert first is not None
    assert first.id == "date"
    assert scheduler._entries[0].presentation_count == 1


def test_unavailable_cycle_one_screen_does_not_block_later_screenshot():
    scheduler = build_scheduler({"screens": {"date": 1, "weather1": 1, "inside": 1}})
    registry = _registry("date", "weather1", "inside")
    registry["date"].available = False

    first = scheduler.next_available(registry)
    second = scheduler.next_available(registry)

    assert first is not None and first.id == "weather1"
    assert second is not None and second.id == "inside"
    assert scheduler._cycle_number == 1


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
