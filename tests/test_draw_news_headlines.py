import screens.draw_news_headlines as dnh
from services.news_feeds import NewsHeadline, NewsTopic


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
    monkeypatch.setattr(dnh, "SCREEN_DELAY", 0.05)
    monkeypatch.setattr(dnh, "_pygame_module_for_display", lambda display: None)
    monkeypatch.setattr(dnh, "_download_thumbnail", lambda *args, **kwargs: None)

    topics = _topics(["local", "sports", "business"])
    headlines = _headlines_for(topics)
    monkeypatch.setattr(dnh, "load_news_feed_config", lambda: (topics, 5, 20))
    monkeypatch.setattr(dnh, "fetch_all_headlines", lambda: headlines)

    display = DummyDisplay()
    screen = dnh.draw_news_headlines(display, transition=True)

    assert screen.displayed is True
    assert screen.image.size == (dnh.WIDTH, dnh.HEIGHT)
    assert display.frames


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


def test_run_ticker_opens_reader_overlay_on_headline_tap(monkeypatch):
    monkeypatch.setattr(dnh, "SCREEN_DELAY", 0.3)
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
