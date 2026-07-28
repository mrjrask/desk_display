import datetime
from io import BytesIO

import pytest
from PIL import Image

from screens.draw_weather import (
    BASE_MAP_CACHE_TTL_SECONDS,
    RADAR_ANIMATION_LOOPS,
    RADAR_FRAMES_CACHE_TTL_SECONDS,
    RADAR_CENTER_LATITUDE,
    RADAR_CENTER_LONGITUDE,
    RadarFrame,
    _clear_radar_map_caches,
    _fetch_base_map,
    _fetch_radar_frames,
    draw_weather_radar,
)


class _MockResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None



@pytest.fixture(autouse=True)
def _clear_caches_between_tests():
    _clear_radar_map_caches()
    yield
    _clear_radar_map_caches()


def _png_bytes(color=(255, 255, 255)):
    img = Image.new("RGB", (8, 8), color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_fetch_base_map_uses_basic_free_osm(monkeypatch):
    seen = []

    def _mock_get(url, timeout, headers):
        seen.append((url, timeout, headers.get("User-Agent")))
        return _MockResponse(_png_bytes())

    monkeypatch.setattr("screens.draw_weather.http_get", _mock_get)

    result = _fetch_base_map(zoom=7)

    assert result is not None
    assert seen
    assert seen[0][0].startswith("https://tile.openstreetmap.org/")
    assert seen[0][1] == 6
    assert seen[0][2] == "desk-display/weather-radar"


def test_fetch_base_map_falls_back_when_osm_unavailable(monkeypatch):
    seen_urls = []

    def _mock_get(url, timeout, headers):
        seen_urls.append(url)
        if "openstreetmap" in url:
            raise RuntimeError("temporary outage")
        return _MockResponse(_png_bytes(color=(64, 64, 64)))

    monkeypatch.setattr("screens.draw_weather.http_get", _mock_get)

    result = _fetch_base_map(zoom=7)

    assert result is not None
    assert len(seen_urls) == 2
    assert "openstreetmap" in seen_urls[0]
    assert "cartocdn.com/light_all" in seen_urls[1]


def test_fetch_base_map_uses_cache_within_ttl(monkeypatch):
    now = 1_000.0
    calls = []

    def _mock_get(url, timeout, headers):
        calls.append(url)
        return _MockResponse(_png_bytes(color=(10, 20, 30)))

    monkeypatch.setattr("screens.draw_weather.time.monotonic", lambda: now)
    monkeypatch.setattr("screens.draw_weather.http_get", _mock_get)

    first = _fetch_base_map(zoom=7)
    second = _fetch_base_map(zoom=7)

    assert first is not None
    assert second is not None
    assert len(calls) == 1
    assert first is not second


def test_fetch_base_map_refreshes_after_ttl(monkeypatch):
    now = 1_000.0
    calls = []

    def _mock_get(url, timeout, headers):
        calls.append(url)
        return _MockResponse(_png_bytes(color=(len(calls), 20, 30)))

    monkeypatch.setattr("screens.draw_weather.time.monotonic", lambda: now)
    monkeypatch.setattr("screens.draw_weather.http_get", _mock_get)

    assert _fetch_base_map(zoom=7) is not None
    now += BASE_MAP_CACHE_TTL_SECONDS + 1
    assert _fetch_base_map(zoom=7) is not None

    assert len(calls) == 2


def test_fetch_base_map_uses_chicago_center_coordinates(monkeypatch):
    seen_coords = []

    def _mock_latlon_to_tile(lat, lon, zoom):
        seen_coords.append((lat, lon, zoom))
        return (10, 20, 0.0, 0.0)

    def _mock_get(url, timeout, headers):
        return _MockResponse(_png_bytes())

    monkeypatch.setattr("screens.draw_weather._latlon_to_tile", _mock_latlon_to_tile)
    monkeypatch.setattr("screens.draw_weather.http_get", _mock_get)

    result = _fetch_base_map(zoom=7)

    assert result is not None
    assert seen_coords == [(RADAR_CENTER_LATITUDE, RADAR_CENTER_LONGITUDE, 7)]


def test_fetch_radar_frames_uses_cache_within_ttl(monkeypatch):
    now = 2_000.0
    urls_requested = []
    timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    metadata = {
        "host": "https://tilecache.rainviewer.com",
        "radar": {"past": [{"path": "cached", "time": timestamp}]},
    }

    class _JsonResponse(_MockResponse):
        def __init__(self, payload):
            self._payload = payload
            super().__init__(b"")

        def json(self):
            return self._payload

    def _mock_get(url, timeout):
        urls_requested.append(url)
        if "weather-maps.json" in url:
            return _JsonResponse(metadata)
        return _MockResponse(_png_bytes())

    monkeypatch.setattr("screens.draw_weather.time.monotonic", lambda: now)
    monkeypatch.setattr("screens.draw_weather.requests.get", _mock_get)

    first = _fetch_radar_frames(zoom=7, max_frames=6)
    second = _fetch_radar_frames(zoom=7, max_frames=6)

    assert len(first) == 1
    assert len(second) == 1
    assert len(urls_requested) == 2
    assert first[0].image is not second[0].image


def test_fetch_radar_frames_refreshes_after_ttl(monkeypatch):
    now = 2_000.0
    calls = []
    timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp())

    def _mock_rainviewer(zoom, max_frames):
        calls.append((zoom, max_frames))
        return [RadarFrame(Image.new("RGBA", (8, 8), (len(calls), 255, 255, 255)), timestamp)]

    monkeypatch.setattr("screens.draw_weather.time.monotonic", lambda: now)
    monkeypatch.setattr("screens.draw_weather._fetch_rainviewer_frames", _mock_rainviewer)
    monkeypatch.setattr("screens.draw_weather._fetch_iem_radar_fallback_frames", lambda zoom: [])

    assert _fetch_radar_frames(zoom=7, max_frames=6)
    now += RADAR_FRAMES_CACHE_TTL_SECONDS + 1
    assert _fetch_radar_frames(zoom=7, max_frames=6)

    assert calls == [(7, 6), (7, 6)]


def test_fetch_radar_frames_prefers_recent_frames(monkeypatch):
    now_ts = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    stale_ts = now_ts - (6 * 60 * 60)
    fresh_ts = now_ts - (20 * 60)
    sample = Image.new("RGBA", (8, 8), (255, 255, 255, 255))

    monkeypatch.setattr(
        "screens.draw_weather._fetch_rainviewer_frames",
        lambda zoom, max_frames: [
            RadarFrame(sample, stale_ts),
            RadarFrame(sample, fresh_ts),
        ],
    )
    monkeypatch.setattr("screens.draw_weather._fetch_iem_radar_fallback_frames", lambda zoom: [])

    frames = _fetch_radar_frames(zoom=7, max_frames=6)

    assert len(frames) == 1
    assert frames[0].timestamp == fresh_ts


def test_draw_weather_radar_animates_when_transition_enabled(monkeypatch):
    sample = Image.new("RGBA", (8, 8), (255, 255, 255, 255))
    frames = [
        RadarFrame(sample, 1_700_000_000),
        RadarFrame(sample, 1_700_000_060),
    ]

    monkeypatch.setattr("screens.draw_weather._fetch_radar_frames", lambda zoom: frames)
    monkeypatch.setattr("screens.draw_weather._fetch_base_map", lambda zoom: Image.new("RGB", (8, 8), (0, 0, 0)))
    monkeypatch.setattr("screens.draw_weather.time.sleep", lambda _: None)

    class _Display:
        def __init__(self):
            self.frames = []

        def display(self, image):
            self.frames.append(image)

    display = _Display()
    result = draw_weather_radar(display, transition=True)

    assert result.displayed is True
    assert len(display.frames) == len(frames) * RADAR_ANIMATION_LOOPS


def test_fetch_rainviewer_frames_sorts_to_include_latest(monkeypatch):
    timestamps_requested = []
    now_ts = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    metadata = {
        "host": "https://tilecache.rainviewer.com",
        "radar": {
            "past": [
                {"path": "a", "time": now_ts - 600},
                {"path": "b", "time": now_ts - 60},
                {"path": "c", "time": now_ts - 300},
            ]
        },
    }

    class _JsonResponse(_MockResponse):
        def __init__(self, payload):
            self._payload = payload
            super().__init__(b"")

        def json(self):
            return self._payload

    def _mock_get(url, timeout):
        if "weather-maps.json" in url:
            return _JsonResponse(metadata)
        timestamps_requested.append(url.split("/")[3])
        return _MockResponse(_png_bytes())

    monkeypatch.setattr("screens.draw_weather.http_get", _mock_get)

    _fetch_radar_frames(zoom=7, max_frames=2)

    assert timestamps_requested == ["c", "b"]


def test_fetch_rainviewer_frames_tries_alternate_metadata_url(monkeypatch):
    seen_urls = []
    now_ts = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    metadata = {
        "host": "https://tilecache.rainviewer.com",
        "radar": {"past": [{"path": "z", "time": now_ts}]},
    }

    class _JsonResponse(_MockResponse):
        def __init__(self, payload):
            self._payload = payload
            super().__init__(b"")

        def json(self):
            return self._payload

    def _mock_get(url, timeout):
        seen_urls.append(url)
        if "weather-maps.json" in url:
            raise RuntimeError("404")
        if "maps.json" in url:
            return _JsonResponse(metadata)
        return _MockResponse(_png_bytes())

    monkeypatch.setattr("screens.draw_weather.http_get", _mock_get)

    frames = _fetch_radar_frames(zoom=7, max_frames=2)

    assert frames
    assert any("weather-maps.json" in url for url in seen_urls)
    assert any("maps.json" in url for url in seen_urls)


def test_fetch_radar_frames_uses_iem_when_rainviewer_unavailable(monkeypatch):
    sample = Image.new("RGBA", (8, 8), (255, 255, 255, 255))
    monkeypatch.setattr("screens.draw_weather._fetch_rainviewer_frames", lambda zoom, max_frames: [])
    monkeypatch.setattr(
        "screens.draw_weather._fetch_iem_radar_fallback_frames",
        lambda zoom: [RadarFrame(sample, None)],
    )

    frames = _fetch_radar_frames(zoom=7, max_frames=6)

    assert len(frames) == 1
