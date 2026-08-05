import datetime as dt
import json

from PIL import Image, ImageDraw

import screens.on_this_day as otd
from screens.on_this_day import _build_sections, draw_on_this_day


class DummyDisplay:
    width = 320
    height = 240

    def __init__(self):
        self.frames = []

    def clear(self):
        pass

    def image(self, img):
        self.frames.append(img.copy())


def test_on_this_day_exposes_editable_font_defaults_for_requested_profiles():
    expected_profiles = {
        "hdmi_1080p",
        "fallback_hd",
        "fallback_default",
        "display_hat_mini",
        "adafruit_minipitft_114",
        "hyperpixel4_square",
        "hyperpixel4",
    }

    assert expected_profiles <= set(otd.ON_THIS_DAY_FONT_SIZES_BY_PROFILE)
    for profile in expected_profiles:
        sizes = otd._font_sizes_for_profile(profile)
        assert {"title", "section", "body", "year"} <= set(sizes)
        assert all(isinstance(value, int) and value > 0 for value in sizes.values())

    assert otd._font_sizes_for_profile("adafruit_minipitft_114") == {
        "title": 24,
        "section": 15,
        "body": 12,
        "year": 11,
    }


def test_on_this_day_has_requested_curated_sections_for_july_6(monkeypatch):
    monkeypatch.setattr("screens.on_this_day._wiki_items", lambda *args, **kwargs: [])
    monkeypatch.setattr(otd, "_jewish_holiday_items", lambda *args, **kwargs: [])

    sections = _build_sections(dt.date(2026, 7, 6))

    assert "🇺🇸 American History" in sections
    assert "🏙️ Chicago History" in sections
    assert "🏟️ Sports History" in sections
    assert "🎂 Famous Birthdays" in sections
    assert "💾 Tech & Science" in sections


def test_on_this_day_uses_curated_sections_without_live_feed_calls(monkeypatch):
    def fail_wiki_call(*args, **kwargs):
        raise AssertionError("fallback date should not call Wikimedia feeds")

    def fail_jewish_holiday_call(*args, **kwargs):
        raise AssertionError("fallback date should not call Hebcal feed")

    monkeypatch.setattr("screens.on_this_day._wiki_items", fail_wiki_call)
    monkeypatch.setattr(otd, "_jewish_holiday_items", fail_jewish_holiday_call)

    sections = _build_sections(dt.date(2026, 7, 6))

    assert "🌎 General History" in sections
    assert "💾 Tech & Science" in sections


def test_on_this_day_renderer_produces_frame(monkeypatch):
    monkeypatch.setattr("screens.on_this_day._wiki_items", lambda *args, **kwargs: [])
    monkeypatch.setattr(otd, "_jewish_holiday_items", lambda *args, **kwargs: [])
    display = DummyDisplay()

    screen = draw_on_this_day(display, transition=False, today=dt.date(2026, 7, 6))

    assert screen.image.size == (320, 240)
    assert screen.screenshot_image is not None
    assert screen.screenshot_image.size[0] == 320
    assert screen.screenshot_image.size[1] >= screen.image.size[1]
    assert display.frames


def test_on_this_day_full_image_is_tall_enough_for_curated_july_6(monkeypatch):
    monkeypatch.setattr("screens.on_this_day._wiki_items", lambda *args, **kwargs: [])
    monkeypatch.setattr(otd, "_jewish_holiday_items", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "screens.on_this_day._download_thumbnail", lambda *args, **kwargs: None
    )

    full_img = otd._render_full_image(dt.date(2026, 7, 6))

    assert full_img.height >= 1211


def test_on_this_day_year_items_wrap_within_remaining_card_width(monkeypatch):
    monkeypatch.setattr("screens.on_this_day._wiki_items", lambda *args, **kwargs: [])
    monkeypatch.setattr(otd, "_jewish_holiday_items", lambda *args, **kwargs: [])
    sections = otd._build_sections(dt.date(2026, 7, 6))
    draw = ImageDraw.Draw(Image.new("RGB", (otd.W, otd.H), otd._BG))
    pad = max(8, otd.W // 32)
    thumb_size = 34 if otd.W >= 300 else 0

    for items in sections.values():
        for item in items:
            if item.year is None:
                continue
            _, text_width = otd._item_text_layout(
                draw, item, pad, otd.W - pad, thumb_size
            )
            lines = otd.wrap_text(item.text, otd.BODY_FONT, text_width)

            assert lines
            assert all(
                otd.measure_text(draw, line, otd.BODY_FONT)[0] <= text_width
                for line in lines
            )
            assert text_width <= 210


def test_on_this_day_scroll_uses_smooth_readable_tuning(monkeypatch):
    monkeypatch.setattr("screens.on_this_day._wiki_items", lambda *args, **kwargs: [])
    monkeypatch.setattr(otd, "_jewish_holiday_items", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "screens.on_this_day._download_thumbnail", lambda *args, **kwargs: None
    )
    captured = {}

    def fake_scroll_vertical_content(**kwargs):
        captured.update(kwargs)
        kwargs["render_at_offset"](0)

    monkeypatch.setattr(otd, "scroll_vertical_content", fake_scroll_vertical_content)
    display = DummyDisplay()

    screen = draw_on_this_day(display, transition=True, today=dt.date(2026, 7, 6))

    assert screen.screenshot_image is not None
    assert screen.screenshot_image.height > screen.image.height
    assert captured["base_step"] == 1
    assert captured["min_frame_time"] == 0.030
    assert captured["page_jump_mode"] is False


def test_on_this_day_sections_cache_checks_wikimedia_once_per_date(monkeypatch):
    otd._clear_caches_for_tests()
    calls = []

    def fake_wiki(feed_type, month, day, limit=3):
        calls.append((feed_type, month, day, limit))
        return [otd.DayItem(2000, f"{feed_type} item")]

    monkeypatch.setattr(otd, "_wiki_items", fake_wiki)
    monkeypatch.setattr(otd, "_jewish_holiday_items", lambda *args, **kwargs: [])

    first = otd._build_sections(dt.date(2026, 7, 7))
    second = otd._build_sections(dt.date(2026, 7, 7))
    next_day = otd._build_sections(dt.date(2026, 7, 8))

    assert first == second
    assert next_day
    assert len(calls) == 8
    assert [call[2] for call in calls] == [7, 7, 7, 7, 8, 8, 8, 8]


def test_on_this_day_builds_live_sections_in_parallel(monkeypatch):
    otd._clear_caches_for_tests()
    calls = []

    def fake_wiki(feed_type, month, day, limit=3):
        calls.append((feed_type, month, day, limit))
        return [otd.DayItem(2000, f"{feed_type} item")]

    monkeypatch.setattr(otd, "_wiki_items", fake_wiki)
    monkeypatch.setattr(
        otd,
        "_jewish_holiday_items",
        lambda *args, **kwargs: [otd.DayItem(None, "Holiday")],
    )

    sections = otd._build_sections_uncached(dt.date(2026, 7, 8))

    assert "🌎 General History" in sections
    assert "🎉 Holidays & Culture" in sections
    assert {call[0] for call in calls} == {"events", "births", "deaths", "holidays"}


def test_on_this_day_renders_non_empty_fallback_when_live_feeds_are_empty(monkeypatch):
    otd._clear_caches_for_tests()
    monkeypatch.setattr(otd, "_wiki_items", lambda *args, **kwargs: [])
    monkeypatch.setattr(otd, "_jewish_holiday_items", lambda *args, **kwargs: [])

    sections = otd._build_sections(dt.date(2026, 7, 10))

    assert sections
    assert "🌎 General History" in sections
    assert any(item.text for items in sections.values() for item in items)


def test_on_this_day_thumbnail_downloads_are_opt_in(monkeypatch):
    otd._clear_caches_for_tests()

    def fail_http_get(*args, **kwargs):
        raise AssertionError("thumbnail download should not run by default")

    monkeypatch.setattr(otd, "http_get", fail_http_get)
    monkeypatch.setattr(otd, "_LIVE_THUMBNAILS_ENABLED", False)

    assert otd._download_thumbnail("https://example.com/thumb.jpg", 34) is None


def test_on_this_day_render_cache_reuses_daily_image(monkeypatch):
    otd._clear_caches_for_tests()
    calls = []

    def fake_build_sections(today):
        calls.append(today)
        return {"🌎 General History": [otd.DayItem(2000, "Cached render item")]}

    monkeypatch.setattr(otd, "_build_sections", fake_build_sections)
    monkeypatch.setattr(otd, "_download_thumbnail", lambda *args, **kwargs: None)

    first = otd._render_full_image(dt.date(2026, 7, 7))
    second = otd._render_full_image(dt.date(2026, 7, 7))
    next_day = otd._render_full_image(dt.date(2026, 7, 8))

    assert first.size == second.size
    assert next_day.size
    assert calls == [dt.date(2026, 7, 7), dt.date(2026, 7, 8)]
    assert first is not second


def test_on_this_day_parses_hebrew_calendar_holidays_for_today():
    ics = """BEGIN:VCALENDAR
BEGIN:VEVENT
DTSTART;VALUE=DATE:20250707
SUMMARY:Past Same Gregorian Date
END:VEVENT
BEGIN:VEVENT
DTSTART;VALUE=DATE:20260707
SUMMARY:Tzom Tammuz
END:VEVENT
BEGIN:VEVENT
DTSTART;VALUE=DATE:20260708
SUMMARY:Different Day
END:VEVENT
BEGIN:VEVENT
DTSTART;VALUE=DATE:20270707
SUMMARY:Future Same Gregorian Date
END:VEVENT
END:VCALENDAR
"""

    items = otd._parse_jewish_holidays_ics(ics, dt.date(2026, 7, 7))

    assert items == [otd.DayItem(None, "Jewish holiday: Tzom Tammuz.")]


def test_on_this_day_includes_jewish_holidays_in_holidays_and_culture(monkeypatch):
    otd._clear_caches_for_tests()

    class FakeResponse:
        text = (
            "BEGIN:VCALENDAR\n"
            "BEGIN:VEVENT\n"
            "DTSTART;VALUE=DATE:20260707\n"
            "SUMMARY:Tzom Tammuz\n"
            "END:VEVENT\n"
            "END:VCALENDAR\n"
        )

        def raise_for_status(self):
            pass

    monkeypatch.setattr(otd, "http_get", lambda *args, **kwargs: FakeResponse())
    monkeypatch.setattr(otd, "_wiki_items", lambda *args, **kwargs: [])

    sections = otd._build_sections(dt.date(2026, 7, 7))

    assert sections["🎉 Holidays & Culture"] == [
        otd.DayItem(None, "Jewish holiday: Tzom Tammuz.")
    ]


def test_on_this_day_adds_wikimedia_holiday_descriptions(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "holidays": [
                    {
                        "text": "Test Holiday",
                        "pages": [
                            {
                                "extract": "A day for testing holiday descriptions.",
                                "thumbnail": {
                                    "source": "https://example.com/holiday.jpg"
                                },
                            }
                        ],
                    }
                ]
            }

    monkeypatch.setattr(otd, "http_get", lambda *args, **kwargs: FakeResponse())

    items = otd._wiki_items("holidays", 7, 7, 1)

    assert items == [
        otd.DayItem(
            None,
            "Test Holiday: A day for testing holiday descriptions.",
            "https://example.com/holiday.jpg",
        )
    ]


def test_on_this_day_skips_wikimedia_holiday_descriptions_for_grouped_rows(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "holidays": [
                    {
                        "text": "Holiday One; Holiday Two",
                        "pages": [
                            {
                                "extract": "Description for only the first holiday.",
                                "thumbnail": {
                                    "source": "https://example.com/first.jpg"
                                },
                            },
                            {
                                "extract": "Description for the second holiday.",
                                "thumbnail": {
                                    "source": "https://example.com/second.jpg"
                                },
                            },
                        ],
                    }
                ]
            }

    monkeypatch.setattr(otd, "http_get", lambda *args, **kwargs: FakeResponse())

    items = otd._wiki_items("holidays", 7, 7, 1)

    assert items == [
        otd.DayItem(
            None,
            "Holiday One; Holiday Two",
            "https://example.com/first.jpg",
        )
    ]


def test_on_this_day_parses_hebrew_calendar_holiday_descriptions():
    ics = """BEGIN:VCALENDAR
BEGIN:VEVENT
DTSTART;VALUE=DATE:20260707
SUMMARY:Tzom Tammuz
DESCRIPTION:Fast day commemorating the breach of Jerusalem's walls.
END:VEVENT
END:VCALENDAR
"""

    items = otd._parse_jewish_holidays_ics(ics, dt.date(2026, 7, 7))

    assert items == [
        otd.DayItem(
            None,
            "Jewish holiday: Tzom Tammuz: Fast day commemorating the breach of Jerusalem's walls.",
        )
    ]


def test_on_this_day_retries_after_cached_offline_fallback_expires(monkeypatch):
    otd._clear_caches_for_tests()
    times = iter([100.0, 100.0, 200.0, 1200.0, 1200.0])
    builds = []

    def fake_monotonic():
        return next(times)

    def fake_build_sections_uncached(today):
        builds.append(today)
        if len(builds) == 1:
            return otd._offline_sections_fallback()
        return {"🌎 General History": [otd.DayItem(2000, "Recovered live item")]}

    monkeypatch.setattr(otd.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(otd, "_OFFLINE_FALLBACK_RETRY_SECONDS", 900.0)
    monkeypatch.setattr(otd, "_build_sections_uncached", fake_build_sections_uncached)

    today = dt.date(2026, 7, 14)
    first = otd._build_sections(today)
    second = otd._build_sections(today)
    third = otd._build_sections(today)

    assert first == otd._offline_sections_fallback()
    assert second == first
    assert third == {"🌎 General History": [otd.DayItem(2000, "Recovered live item")]}
    assert builds == [today, today]


def test_on_this_day_retries_and_merges_partial_live_feeds(monkeypatch):
    otd._clear_caches_for_tests()
    times = iter([100.0, 100.0, 200.0, 500.0, 500.0])
    builds = []

    def fake_build_sections_uncached(today):
        builds.append(today)
        if len(builds) == 1:
            return {
                "🌎 General History": [otd.DayItem(2000, "First event")],
                "🎉 Holidays & Culture": [
                    otd.DayItem(None, "Jewish holiday from Hebcal")
                ],
            }
        return {
            "🌎 General History": [otd.DayItem(2001, "Refreshed event")],
            "🎂 Famous Birthdays": [otd.DayItem(1980, "A birthday")],
            "🕯️ Notable Lives": [otd.DayItem(1990, "A notable life")],
            "🎉 Holidays & Culture": [
                otd.DayItem(None, "Public holiday from Wikimedia")
            ],
        }

    monkeypatch.setattr(otd.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(otd, "_INCOMPLETE_FEED_RETRY_SECONDS", 300.0)
    monkeypatch.setattr(otd, "_build_sections_uncached", fake_build_sections_uncached)

    today = dt.date(2026, 7, 15)
    first = otd._build_sections(today)
    cached = otd._build_sections(today)
    recovered = otd._build_sections(today)

    assert cached == first
    assert set(recovered) == {
        "🌎 General History",
        "🎂 Famous Birthdays",
        "🕯️ Notable Lives",
        "🎉 Holidays & Culture",
    }
    assert recovered["🎉 Holidays & Culture"] == [
        otd.DayItem(None, "Jewish holiday from Hebcal"),
        otd.DayItem(None, "Public holiday from Wikimedia"),
    ]
    assert otd._sections_retry_interval(today, recovered) is None
    assert builds == [today, today]


def test_on_this_day_merge_deduplicates_same_holiday_from_retry():
    holiday = otd.DayItem(None, "Shared holiday")

    merged = otd._merge_sections(
        {"🎉 Holidays & Culture": [holiday]},
        {"🎉 Holidays & Culture": [holiday]},
    )

    assert merged["🎉 Holidays & Culture"] == [holiday]


def test_on_this_day_disk_cache_round_trips_sections(tmp_path, monkeypatch):
    cache_path = tmp_path / "on_this_day_cache.json"
    monkeypatch.setattr(otd, "_CACHE_DISK_PATH", str(cache_path))
    today = dt.date(2026, 7, 20)
    sections = {
        "🌎 General History": [otd.DayItem(1999, "Something happened", "https://x/thumb.jpg")]
    }

    otd._save_disk_cache(today, sections, otd.time.time())
    assert cache_path.exists()

    loaded = otd._sections_from_jsonable(
        json.loads(cache_path.read_text())["sections"]
    )
    assert loaded == sections


def test_on_this_day_restores_cache_from_disk_on_fresh_process(tmp_path, monkeypatch):
    cache_path = tmp_path / "on_this_day_cache.json"
    monkeypatch.setattr(otd, "_CACHE_DISK_PATH", str(cache_path))
    today = dt.date(2026, 7, 21)
    sections = {"🌎 General History": [otd.DayItem(2010, "Persisted event")]}
    otd._save_disk_cache(today, sections, otd.time.time())

    # Simulate a process restart: in-memory caches are gone, but the disk
    # cache flag/state must look exactly like process start (not the
    # "already loaded, skip disk" state that _clear_caches_for_tests sets).
    otd._SECTIONS_CACHE_DATE = None
    otd._SECTIONS_CACHE_VALUE = None
    otd._SECTIONS_CACHE_TIME = None
    otd._DISK_CACHE_LOADED = False

    def fail_build(*args, **kwargs):
        raise AssertionError("restored disk cache should avoid a live rebuild")

    monkeypatch.setattr(otd, "_build_sections_uncached", fail_build)

    result = otd._build_sections(today)

    assert result == sections
    otd._clear_caches_for_tests()


def test_on_this_day_ignores_disk_cache_from_a_previous_day(tmp_path, monkeypatch):
    cache_path = tmp_path / "on_this_day_cache.json"
    monkeypatch.setattr(otd, "_CACHE_DISK_PATH", str(cache_path))
    yesterday = dt.date(2026, 7, 21)
    otd._save_disk_cache(
        yesterday, {"🌎 General History": [otd.DayItem(2010, "Old event")]}, otd.time.time()
    )

    otd._SECTIONS_CACHE_DATE = None
    otd._SECTIONS_CACHE_VALUE = None
    otd._SECTIONS_CACHE_TIME = None
    otd._DISK_CACHE_LOADED = False

    today = dt.date(2026, 7, 22)
    otd._load_disk_cache_once(today)

    assert otd._SECTIONS_CACHE_DATE is None
    assert otd._SECTIONS_CACHE_VALUE is None
    otd._clear_caches_for_tests()


def test_on_this_day_does_not_persist_offline_fallback_to_disk(tmp_path, monkeypatch):
    # A process restart shortly after a transient network failure should
    # retry live feeds immediately, not reload a persisted "temporarily
    # unavailable" placeholder from disk and treat it as fresh content.
    otd._clear_caches_for_tests()
    cache_path = tmp_path / "on_this_day_cache.json"
    monkeypatch.setattr(otd, "_CACHE_DISK_PATH", str(cache_path))
    monkeypatch.setattr(
        otd, "_build_sections_uncached", lambda today: otd._offline_sections_fallback()
    )

    today = dt.date(2026, 8, 5)
    sections = otd._build_sections(today)

    assert sections == otd._offline_sections_fallback()
    assert not cache_path.exists()


def test_on_this_day_build_sections_persists_to_disk(tmp_path, monkeypatch):
    otd._clear_caches_for_tests()
    cache_path = tmp_path / "on_this_day_cache.json"
    monkeypatch.setattr(otd, "_CACHE_DISK_PATH", str(cache_path))
    monkeypatch.setattr(otd, "_wiki_items", lambda *args, **kwargs: [])
    monkeypatch.setattr(otd, "_jewish_holiday_items", lambda *args, **kwargs: [])

    today = dt.date(2026, 7, 6)
    sections = otd._build_sections(today)

    assert cache_path.exists()
    payload = json.loads(cache_path.read_text())
    assert payload["date"] == today.isoformat()
    assert otd._sections_from_jsonable(payload["sections"]) == sections
