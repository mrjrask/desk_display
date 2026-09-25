"""Tests for the hardware-free rendering boundary."""

from types import SimpleNamespace

from PIL import Image

import screens.mlb_league_standings as mlb_standings
from config import CENTRAL_TIME
from display_profiles import DISPLAY_PROFILE_DISPLAY_HAT_MINI, PROFILE_PRESETS
from rendering.screen_renderer import ScreenRenderer, ServerPreferenceSnapshot
from services.data_coordinator import DataCoordinator
from services.data_provider import DataProvider
from utils import ScreenImage


def test_default_renderer_thaws_snapshot_for_legacy_screens(monkeypatch):
    coordinator = DataCoordinator()
    snapshot = coordinator.publish(
        "weather",
        {
            "hourly": [{"temperature": 72}],
            "alerts": {"active", "watch"},
        },
    )
    received = {}

    def fake_build_screen_registry(context):
        received["cache"] = context.cache
        received["now"] = context.now
        received["now_utc"] = context.now_utc
        context.cache["weather"]["hourly"][0]["temperature"] = 0
        definition = SimpleNamespace(
            available=True,
            metadata={},
            render=lambda: Image.new("RGB", (320, 240)),
        )
        return {"weather_hourly": definition}, None

    monkeypatch.setattr("screens.registry.build_screen_registry", fake_build_screen_registry)

    renderer = ScreenRenderer()
    renderer.render(
        "weather_hourly",
        PROFILE_PRESETS[DISPLAY_PROFILE_DISPLAY_HAT_MINI],
        ServerPreferenceSnapshot(revision=1),
        snapshot,
    )

    weather = received["cache"]["weather"]
    assert isinstance(received["cache"], dict)
    assert isinstance(weather, dict)
    assert isinstance(weather["hourly"], list)
    assert isinstance(weather["hourly"][0], dict)
    assert isinstance(weather["alerts"], set)
    assert snapshot["weather"]["hourly"][0]["temperature"] == 72
    assert received["now"].tzinfo is CENTRAL_TIME
    assert received["now_utc"] == received["now"].astimezone(received["now_utc"].tzinfo)


def test_snapshot_renderer_preserves_mlb_standings_with_numeric_league_ids(monkeypatch):
    standings = {
        mlb_standings.AL_LEAGUE_ID: {
            "East": [{"abbr": "NYY", "team_name": "Yankees", "wins": 1, "losses": 0}]
        }
    }
    monkeypatch.setattr(
        mlb_standings, "_fetch_league_standings", lambda **_kwargs: standings
    )

    coordinator = DataCoordinator(DataProvider())
    coordinator.read_mlb_league_standings()
    snapshot = coordinator.snapshot()
    assert mlb_standings.AL_LEAGUE_ID in snapshot["mlb_league_standings"]

    captured_rows = []
    original_column_layout = mlb_standings._column_layout

    def capture_column_layout(draw, rows):
        captured_rows.extend(rows)
        return original_column_layout(draw, rows)

    monkeypatch.setattr(mlb_standings, "_column_layout", capture_column_layout)
    monkeypatch.setattr(mlb_standings, "_load_logo", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlb_standings, "clear_display", lambda display: None)
    monkeypatch.setattr(mlb_standings, "scroll_vertical_content", lambda **kwargs: None)

    ScreenRenderer().render(
        "MLB AL Standings",
        PROFILE_PRESETS[DISPLAY_PROFILE_DISPLAY_HAT_MINI],
        ServerPreferenceSnapshot(revision=1),
        snapshot,
    )

    assert [row["abbr"] for row in captured_rows] == ["NYY"]


def test_displayed_screen_image_uses_captured_viewport():
    viewport = Image.new("RGB", (320, 240), "red")
    tall_canvas = Image.new("RGB", (320, 480), "blue")

    def registry_factory(capture, profile, preferences, data):
        def render():
            capture.image(viewport)
            return ScreenImage(tall_canvas, displayed=True)

        return {
            "scrolling": SimpleNamespace(available=True, metadata={}, render=render)
        }

    snapshot = DataCoordinator().snapshot()
    artifact = ScreenRenderer(registry_factory).render(
        "scrolling",
        PROFILE_PRESETS[DISPLAY_PROFILE_DISPLAY_HAT_MINI],
        ServerPreferenceSnapshot(revision=1),
        snapshot,
    )

    assert artifact.image.getpixel((0, 0)) == (255, 0, 0)


def _news_setup(monkeypatch):
    import screens.draw_news_headlines as dnh
    from services.news_feeds import NewsHeadline, NewsTopic

    topics = [
        NewsTopic(id=t, label=f"{t} News", name=t, url=f"https://example.com/{t}")
        for t in ("local", "sports")
    ]
    headlines = {
        topic.id: [
            NewsHeadline(topic_id=topic.id, title=f"{topic.label} story", link="https://example.com/a")
        ]
        for topic in topics
    }
    monkeypatch.setattr(dnh, "NEWS_HEADLINES_DISPLAY_SECONDS", 30.0)
    monkeypatch.setattr(dnh, "_pygame_module_for_display", lambda display: None)
    monkeypatch.setattr(dnh, "_download_thumbnail", lambda *args, **kwargs: None)
    monkeypatch.setattr(dnh, "fetch_stock_quotes", lambda *args, **kwargs: [])
    monkeypatch.setattr(dnh, "load_news_feed_config", lambda: (topics, 5, 20))
    monkeypatch.setattr(dnh, "fetch_all_headlines", lambda: headlines)
    return dnh


def test_render_only_capture_skips_news_ticker_playback_interval(monkeypatch):
    import time

    dnh = _news_setup(monkeypatch)

    def registry_factory(capture, profile, preferences, data):
        return {
            "news headlines": SimpleNamespace(
                available=True,
                metadata={},
                render=lambda: dnh.draw_news_headlines(capture, transition=True),
            )
        }

    started = time.monotonic()
    artifact = ScreenRenderer(registry_factory).render(
        "news headlines",
        PROFILE_PRESETS[DISPLAY_PROFILE_DISPLAY_HAT_MINI],
        ServerPreferenceSnapshot(revision=1),
        DataCoordinator().snapshot(),
    )

    assert time.monotonic() - started < 5.0
    assert artifact.image.getbbox() is not None


def test_render_only_capture_runs_frame_bounded_animation_to_final_frame():
    import time

    colors = ["black", "gray", "white"]

    def registry_factory(capture, profile, preferences, data):
        def render():
            for color in colors:
                capture.image(Image.new("RGB", (320, 240), color))
                if capture.wait_for_skip(10.0):
                    break
            return ScreenImage(capture.current_image, displayed=True)

        return {"drop": SimpleNamespace(available=True, metadata={}, render=render)}

    started = time.monotonic()
    artifact = ScreenRenderer(registry_factory).render(
        "drop",
        PROFILE_PRESETS[DISPLAY_PROFILE_DISPLAY_HAT_MINI],
        ServerPreferenceSnapshot(revision=1),
        DataCoordinator().snapshot(),
    )

    assert time.monotonic() - started < 5.0
    assert artifact.image.convert("RGB").getpixel((0, 0)) == (255, 255, 255)
