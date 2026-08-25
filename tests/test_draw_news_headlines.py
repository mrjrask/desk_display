import datetime as dt

import screens.draw_news_headlines as dnh
from services.news_feeds import NewsHeadline, NewsTopic
from services.stock_quotes import StockQuote


class DummyDisplay:
    width = 320
    height = 240
    rotation = 0

    def __init__(self):
        self.frames = []

    def image(self, img):
        self.frames.append(img.copy())

    def skip_requested(self):
        return False


def _topics(ids):
    return [
        NewsTopic(id=t, label=f"{t.title()} News", name=t, url=f"https://example.com/{t}")
        for t in ids
    ]


def _headlines_for(topics, count=2):
    return {
        topic.id: [
            NewsHeadline(topic_id=topic.id, title=f"{topic.label} sample {i}", link=f"https://example.com/{topic.id}/{i}")
            for i in range(1, count + 1)
        ]
        for topic in topics
    }


def test_layout_row_entries_visible_window_and_wraparound():
    widths = [50.0, 60.0, 70.0]

    visible = dnh.layout_row_entries(widths, offset=0.0, lane_width=200.0)

    assert visible == [(-0.0, 50.0, 0), (50.0, 110.0, 1), (110.0, 180.0, 2), (180.0, 230.0, 0)]


def test_layout_row_entries_advances_with_offset():
    widths = [50.0, 60.0, 70.0]

    visible = dnh.layout_row_entries(widths, offset=55.0, lane_width=200.0)

    # total width is 180; offset 55 lands inside entry 1 (50..110), 5px in
    assert visible[0] == (-5.0, 55.0, 1)


def test_layout_row_entries_handles_empty_and_zero_width_lane():
    assert dnh.layout_row_entries([], offset=0.0, lane_width=100.0) == []
    assert dnh.layout_row_entries([10.0], offset=0.0, lane_width=0.0) == []
    assert dnh.layout_row_entries([0.0, 0.0], offset=0.0, lane_width=100.0) == []


def test_speed_multiplier_is_deterministic_and_topic_specific():
    first = dnh._speed_multiplier("local")
    second = dnh._speed_multiplier("local")
    other = dnh._speed_multiplier("sports")

    assert first == second
    assert 0.82 <= first <= 1.37
    assert first != other


def test_theme_for_topic_falls_back_for_unknown_topic():
    assert dnh._theme_for_topic("local") is dnh._ROW_THEMES["local"]
    assert dnh._theme_for_topic("some_custom_topic") is dnh._FALLBACK_THEME


def test_build_rows_skips_topics_with_no_headlines(monkeypatch):
    monkeypatch.setattr(dnh, "_download_thumbnail", lambda *args, **kwargs: None)
    topics = _topics(["local", "sports"])
    headlines = {"local": [NewsHeadline(topic_id="local", title="Only local headline", link="https://example.com/1")]}

    rows = dnh._build_rows(topics, headlines, row_height=40)

    assert [row.topic.id for row in rows] == ["local"]
    assert rows[0].entries[0].headline.title == "Only local headline"


def test_compute_row_layout_divides_height_and_respects_minimum():
    row_height, row_tops = dnh._compute_row_layout(4)

    assert row_height >= dnh._MIN_ROW_HEIGHT
    assert row_tops == [i * row_height for i in range(len(row_tops))]
    assert dnh._compute_row_layout(0) == (0, [])


def test_render_frame_produces_full_size_image_and_hit_rects(monkeypatch):
    monkeypatch.setattr(dnh, "_download_thumbnail", lambda *args, **kwargs: None)
    topics = _topics(["local", "sports"])
    headlines = _headlines_for(topics)
    row_height, row_tops = dnh._compute_row_layout(len(topics))
    rows = dnh._build_rows(topics, headlines, row_height)

    img, hit_rects = dnh._render_frame(rows, row_height, row_tops)

    assert img.size == (dnh.WIDTH, dnh.HEIGHT)
    assert hit_rects
    for x0, y0, x1, y1, headline in hit_rects:
        assert 0 <= x0 < x1 <= dnh.WIDTH
        assert 0 <= y0 < y1 <= dnh.HEIGHT
        assert isinstance(headline, NewsHeadline)


def test_hit_test_returns_headline_for_matching_point_and_none_otherwise():
    headline = NewsHeadline(topic_id="local", title="Hit me", link="https://example.com/1")
    rects = [(10, 20, 110, 60, headline)]

    assert dnh._hit_test(rects, 50, 40) is headline
    assert dnh._hit_test(rects, 5, 40) is None
    assert dnh._hit_test(rects, 50, 100) is None


def test_map_touch_to_render_coords_handles_each_rotation():
    args = {"display_width": 200.0, "display_height": 100.0}

    assert dnh._map_touch_to_render_coords(50, 25, rotation_degrees=0, **args) == (50.0, 25.0)
    # 90 degrees: mapped_x = 1 - y_norm, mapped_y = x_norm (in the 200x100 space)
    mapped_90 = dnh._map_touch_to_render_coords(50, 25, rotation_degrees=90, **args)
    assert mapped_90 == ((1.0 - 25 / 100.0) * 200.0, (50 / 200.0) * 100.0)
    mapped_180 = dnh._map_touch_to_render_coords(50, 25, rotation_degrees=180, **args)
    assert mapped_180 == (150.0, 75.0)


def test_is_touch_capable_detects_finger_or_mouse_support():
    class _NoTouch:
        pass

    class _Finger:
        FINGERDOWN = 1

    class _Mouse:
        MOUSEBUTTONDOWN = 2

    assert dnh._is_touch_capable(None) is False
    assert dnh._is_touch_capable(_NoTouch()) is False
    assert dnh._is_touch_capable(_Finger()) is True
    assert dnh._is_touch_capable(_Mouse()) is True


def test_draw_news_headlines_renders_without_touch(monkeypatch):
    monkeypatch.setattr(dnh, "NEWS_HEADLINES_DISPLAY_SECONDS", 0.05)
    monkeypatch.setattr(dnh, "_pygame_module_for_display", lambda display: None)
    monkeypatch.setattr(dnh, "_download_thumbnail", lambda *args, **kwargs: None)
    monkeypatch.setattr(dnh, "fetch_stock_quotes", lambda *args, **kwargs: [])

    topics = _topics(["local", "sports", "business"])
    headlines = _headlines_for(topics)
    monkeypatch.setattr(dnh, "load_news_feed_config", lambda: (topics, 5, 20))
    monkeypatch.setattr(dnh, "fetch_all_headlines", lambda: headlines)

    display = DummyDisplay()
    screen = dnh.draw_news_headlines(display, transition=True)

    assert screen.displayed is True
    assert screen.image.size == (dnh.WIDTH, dnh.HEIGHT)
    assert display.frames


def test_draw_news_headlines_appends_stock_row_at_bottom(monkeypatch):
    monkeypatch.setattr(dnh, "NEWS_HEADLINES_DISPLAY_SECONDS", 0.05)
    monkeypatch.setattr(dnh, "_pygame_module_for_display", lambda display: None)
    monkeypatch.setattr(dnh, "_download_thumbnail", lambda *args, **kwargs: None)

    topics = _topics(["local"])
    headlines = _headlines_for(topics)
    monkeypatch.setattr(dnh, "load_news_feed_config", lambda: (topics, 5, 20))
    monkeypatch.setattr(dnh, "fetch_all_headlines", lambda: headlines)

    quotes = [StockQuote(symbol="^DJI", label="DJIA", price=44000.0, change=12.5, change_pct=0.03)]
    captured_rows = []
    monkeypatch.setattr(dnh, "fetch_stock_quotes", lambda *args, **kwargs: quotes)
    original_run_ticker = dnh._run_ticker

    def _spy_run_ticker(display, rows, ticker_data=None):
        captured_rows.extend(rows)
        return original_run_ticker(display, rows, ticker_data)

    monkeypatch.setattr(dnh, "_run_ticker", _spy_run_ticker)

    display = DummyDisplay()
    dnh.draw_news_headlines(display, transition=True)

    assert [row.topic.id for row in captured_rows] == ["local", "markets"]


def test_draw_news_headlines_skips_stock_row_when_disabled(monkeypatch):
    monkeypatch.setattr(dnh, "NEWS_HEADLINES_DISPLAY_SECONDS", 0.05)
    monkeypatch.setattr(dnh, "_pygame_module_for_display", lambda display: None)
    monkeypatch.setattr(dnh, "_download_thumbnail", lambda *args, **kwargs: None)
    monkeypatch.setattr(dnh.config, "ENABLE_STOCK_TICKER", False)

    calls = []
    monkeypatch.setattr(dnh, "fetch_stock_quotes", lambda *args, **kwargs: calls.append(1) or [])

    topics = _topics(["local"])
    headlines = _headlines_for(topics)
    monkeypatch.setattr(dnh, "load_news_feed_config", lambda: (topics, 5, 20))
    monkeypatch.setattr(dnh, "fetch_all_headlines", lambda: headlines)

    display = DummyDisplay()
    dnh.draw_news_headlines(display, transition=True)

    assert calls == []


def test_format_stock_entry_text_colors_by_direction():
    up = StockQuote(symbol="AAPL", label="AAPL", price=200.0, change=1.5, change_pct=0.75)
    down = StockQuote(symbol="MSFT", label="MSFT", price=400.0, change=-2.0, change_pct=-0.5)
    flat = StockQuote(symbol="VRNO", label="VRNO", price=1.0, change=0.0, change_pct=0.0)
    unpriced = StockQuote(symbol="NVDA", label="NVDA", price=120.0, change=None, change_pct=None)

    up_text, up_color = dnh._format_stock_entry_text(up)
    _down_text, down_color = dnh._format_stock_entry_text(down)
    _flat_text, flat_color = dnh._format_stock_entry_text(flat)
    unpriced_text, unpriced_color = dnh._format_stock_entry_text(unpriced)

    assert up_color == dnh._STOCK_UP_COLOR
    assert "200.00" in up_text and "+1.50" in up_text
    assert down_color == dnh._STOCK_DOWN_COLOR
    assert flat_color == dnh._STOCK_FLAT_COLOR
    assert unpriced_color == dnh._STOCK_FLAT_COLOR
    assert "N/A" in unpriced_text


def test_load_company_logo_is_case_insensitive(tmp_path, monkeypatch):
    (tmp_path / "aapl.png").write_bytes(_tiny_png_bytes())
    monkeypatch.setattr(dnh.config, "COMPANY_LOGOS_DIR", str(tmp_path))
    dnh._COMPANY_LOGO_INDEX_DIR_STAMP = None
    dnh._COMPANY_LOGO_CACHE.clear()

    logo = dnh._load_company_logo("AAPL", 16)

    assert logo is not None
    assert logo.size == (16, 16)


def test_load_company_logo_missing_logs_warning_once(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(dnh.config, "COMPANY_LOGOS_DIR", str(tmp_path))
    dnh._COMPANY_LOGO_INDEX_DIR_STAMP = None
    dnh._COMPANY_LOGO_CACHE.clear()
    dnh._LOGGED_MISSING_COMPANY_LOGOS.clear()

    with caplog.at_level("WARNING"):
        first = dnh._load_company_logo("ZZZZ", 16)
        second = dnh._load_company_logo("ZZZZ", 16)

    assert first is None
    assert second is None
    warnings = [r for r in caplog.records if "ZZZZ" in r.message]
    assert len(warnings) == 1


def _tiny_png_bytes() -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(buf, format="PNG")
    return buf.getvalue()


def test_build_stock_row_skips_unpriced_quotes_and_none_when_empty():
    quotes = [
        StockQuote(symbol="^DJI", label="DJIA", price=44000.0, change=10.0, change_pct=0.02),
        StockQuote(symbol="XXXX", label="XXXX", price=None, change=None, change_pct=None),
    ]

    row = dnh._build_stock_row(quotes, row_height=20)

    assert row is not None
    assert row.topic.id == "markets"
    assert len(row.entries) == 1
    assert row.entries[0].headline is None

    assert dnh._build_stock_row([], row_height=20) is None
    assert dnh._build_stock_row([StockQuote("X", "X", None, None, None)], row_height=20) is None


def test_render_frame_stock_row_entries_are_not_tappable():
    quotes = [StockQuote(symbol="AAPL", label="AAPL", price=200.0, change=1.5, change_pct=0.75)]
    row = dnh._build_stock_row(quotes, row_height=20)
    row_height, row_tops = dnh._compute_row_layout(1)

    _img, hit_rects = dnh._render_frame([row], row_height, row_tops)

    assert hit_rects == []


def test_draw_news_headlines_shows_empty_state_when_disabled(monkeypatch):
    monkeypatch.setattr(dnh.config, "ENABLE_NEWS_HEADLINES", False)
    display = DummyDisplay()

    screen = dnh.draw_news_headlines(display, transition=True)

    assert screen.image.size == (dnh.WIDTH, dnh.HEIGHT)
    assert display.frames


def test_draw_news_headlines_shows_empty_state_when_no_topics(monkeypatch):
    monkeypatch.setattr(dnh, "load_news_feed_config", lambda: ([], 5, 20))
    display = DummyDisplay()

    screen = dnh.draw_news_headlines(display, transition=True)

    assert screen.image.size == (dnh.WIDTH, dnh.HEIGHT)
    assert display.frames


def test_draw_news_headlines_2_renders_using_its_own_feeds(monkeypatch):
    monkeypatch.setattr(dnh, "NEWS_HEADLINES_DISPLAY_SECONDS", 0.05)
    monkeypatch.setattr(dnh, "_pygame_module_for_display", lambda display: None)
    monkeypatch.setattr(dnh, "_download_thumbnail", lambda *args, **kwargs: None)
    monkeypatch.setattr(dnh, "fetch_stock_quotes", lambda *args, **kwargs: [])

    primary_topics = _topics(["local"])
    secondary_topics = _topics(["cnn", "tribune"])
    monkeypatch.setattr(dnh, "load_news_feed_config", lambda: (primary_topics, 5, 20))
    monkeypatch.setattr(dnh, "fetch_all_headlines", lambda: _headlines_for(primary_topics))
    monkeypatch.setattr(dnh, "load_news_feed_config_2", lambda: (secondary_topics, 5, 20))
    monkeypatch.setattr(dnh, "fetch_all_headlines_2", lambda: _headlines_for(secondary_topics))

    display = DummyDisplay()
    screen = dnh.draw_news_headlines_2(display, transition=True)

    assert screen.displayed is True
    assert screen.image.size == (dnh.WIDTH, dnh.HEIGHT)
    assert display.frames


def test_draw_news_headlines_2_shows_empty_state_when_disabled(monkeypatch):
    monkeypatch.setattr(dnh.config, "ENABLE_NEWS_HEADLINES_2", False)
    display = DummyDisplay()

    screen = dnh.draw_news_headlines_2(display, transition=True)

    assert screen.image.size == (dnh.WIDTH, dnh.HEIGHT)
    assert display.frames


def test_draw_news_headlines_2_shows_empty_state_when_no_topics(monkeypatch):
    monkeypatch.setattr(dnh, "load_news_feed_config_2", lambda: ([], 5, 20))
    display = DummyDisplay()

    screen = dnh.draw_news_headlines_2(display, transition=True)

    assert screen.image.size == (dnh.WIDTH, dnh.HEIGHT)
    assert display.frames


def test_run_ticker_opens_reader_overlay_on_headline_tap(monkeypatch):
    monkeypatch.setattr(dnh, "NEWS_HEADLINES_DISPLAY_SECONDS", 0.3)
    monkeypatch.setattr(dnh, "_download_thumbnail", lambda *args, **kwargs: None)
    topics = _topics(["local"])
    headlines = _headlines_for(topics, count=1)
    row_height, row_tops = dnh._compute_row_layout(len(topics))
    rows = dnh._build_rows(topics, headlines, row_height)

    # Render one frame up front to find a real on-screen point over the
    # single headline entry, rather than guessing a coordinate.
    _first_frame, seed_hit_rects = dnh._render_frame(rows, row_height, row_tops)
    assert seed_hit_rects
    hx0, hy0, hx1, hy1, _headline = seed_hit_rects[0]
    tap_point = ((hx0 + hx1) // 2, (hy0 + hy1) // 2)

    class _FakePygame:
        FINGERDOWN = 1

    overlay_calls = []

    def fake_poll_taps(display, pygame_module):
        # Return a tap once, then stop so the loop can finish naturally.
        if not overlay_calls:
            return [tap_point]
        return []

    def fake_show_overlay(display, headline, pygame_module):
        overlay_calls.append(headline)

    monkeypatch.setattr(dnh, "_pygame_module_for_display", lambda display: _FakePygame())
    monkeypatch.setattr(dnh, "_poll_taps", fake_poll_taps)
    monkeypatch.setattr(dnh, "_show_reader_overlay", fake_show_overlay)

    display = DummyDisplay()
    dnh._run_ticker(display, rows)

    assert len(overlay_calls) == 1
    assert overlay_calls[0].topic_id == "local"


def test_filter_recent_headlines_drops_old_but_keeps_undated():
    now = dt.datetime.now(dt.UTC)
    fresh = NewsHeadline(topic_id="local", title="Fresh", link="https://example.com/1", published=now)
    stale = NewsHeadline(
        topic_id="local",
        title="Stale",
        link="https://example.com/2",
        published=now - dt.timedelta(days=dnh._MAX_HEADLINE_AGE_DAYS + 1),
    )
    undated = NewsHeadline(topic_id="local", title="Undated", link="https://example.com/3")

    result = dnh._filter_recent_headlines([fresh, stale, undated])

    assert result == [fresh, undated]


def test_build_rows_filters_out_stale_headlines(monkeypatch):
    monkeypatch.setattr(dnh, "_download_thumbnail", lambda *args, **kwargs: None)
    topics = _topics(["local"])
    now = dt.datetime.now(dt.UTC)
    stale = NewsHeadline(
        topic_id="local",
        title="Old news",
        link="https://example.com/old",
        published=now - dt.timedelta(days=dnh._MAX_HEADLINE_AGE_DAYS + 5),
    )
    headlines = {"local": [stale]}

    rows = dnh._build_rows(topics, headlines, row_height=40)

    assert rows == []


def test_build_rows_restores_saved_offset(monkeypatch):
    monkeypatch.setattr(dnh, "_download_thumbnail", lambda *args, **kwargs: None)
    dnh._ROW_OFFSETS.clear()
    dnh._ROW_OFFSETS[dnh._offset_key("news_feeds.json", "local")] = 123.5
    topics = _topics(["local"])
    headlines = _headlines_for(topics)

    rows = dnh._build_rows(topics, headlines, row_height=40, screen_key="news_feeds.json")

    assert rows[0].offset == 123.5

    # A different screen key must not see another screen's saved offset.
    other_rows = dnh._build_rows(topics, headlines, row_height=40, screen_key="news_feeds_2.json")
    assert other_rows[0].offset == 0.0


def test_save_row_offsets_persists_per_screen_and_topic():
    dnh._ROW_OFFSETS.clear()
    row = dnh._TickerRow(
        topic=NewsTopic(id="local", label="Local", name="local", url="https://example.com"),
        theme=dnh._FALLBACK_THEME,
        entries=[],
        speed=1.0,
        offset=42.0,
    )

    dnh._save_row_offsets("news_feeds.json", [row])

    assert dnh._saved_offset("news_feeds.json", "local") == 42.0
    assert dnh._saved_offset("news_feeds_2.json", "local") == 0.0


def test_draw_news_headlines_resumes_scroll_position_across_calls(monkeypatch):
    monkeypatch.setattr(dnh, "NEWS_HEADLINES_DISPLAY_SECONDS", 0.05)
    monkeypatch.setattr(dnh, "_pygame_module_for_display", lambda display: None)
    monkeypatch.setattr(dnh, "_download_thumbnail", lambda *args, **kwargs: None)
    monkeypatch.setattr(dnh, "fetch_stock_quotes", lambda *args, **kwargs: [])
    dnh._ROW_OFFSETS.clear()

    topics = _topics(["local"])
    monkeypatch.setattr(dnh, "load_news_feed_config", lambda: (topics, 5, 20))
    monkeypatch.setattr(dnh, "fetch_all_headlines", lambda: _headlines_for(topics))

    display = DummyDisplay()
    dnh.draw_news_headlines(display, transition=True)

    saved_offset = dnh._saved_offset("news_feeds.json", "local")
    assert saved_offset > 0.0

    display2 = DummyDisplay()
    dnh.draw_news_headlines(display2, transition=True)

    assert dnh._saved_offset("news_feeds.json", "local") > saved_offset
