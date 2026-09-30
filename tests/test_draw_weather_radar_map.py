import datetime
import importlib
from io import BytesIO

import pytest
from PIL import Image

from screens.draw_weather import (
    BASE_MAP_CACHE_TTL_SECONDS,
    RADAR_ANIMATION_LOOPS,
    RADAR_CENTER_LATITUDE,
    RADAR_CENTER_LONGITUDE,
    RADAR_FRAMES_CACHE_TTL_SECONDS,
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
def _clear_caches_between_tests(monkeypatch, tmp_path):
    monkeypatch.setattr("screens.draw_weather.BASE_MAP_TILE_DIR", str(tmp_path / "radar_basemap"))
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
    assert seen[0][2].startswith("desk-display/weather-radar (+https://")


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
    assert result.info["attribution"] == "\u00a9 OpenStreetMap \u00a9 CARTO"
    assert any("openstreetmap" in url for url in seen_urls)
    assert any("cartocdn.com/light_all" in url for url in seen_urls)


def test_fetch_base_map_uses_cache_within_ttl(monkeypatch):
    now = 1_000.0
    calls = []

    def _mock_get(url, timeout, headers):
        calls.append(url)
        return _MockResponse(_png_bytes(color=(10, 20, 30)))

    monkeypatch.setattr("screens.draw_weather.time.monotonic", lambda: now)
    monkeypatch.setattr("screens.draw_weather.http_get", _mock_get)

    first = _fetch_base_map(zoom=7)
    fetched = len(calls)
    second = _fetch_base_map(zoom=7)

    assert first is not None
    assert second is not None
    assert fetched and len(calls) == fetched
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
    fetched = len(calls)
    now += BASE_MAP_CACHE_TTL_SECONDS + 1
    monkeypatch.setattr("screens.draw_weather.BASE_MAP_TILE_MAX_AGE_SECONDS", -1)
    assert _fetch_base_map(zoom=7) is not None

    assert len(calls) == 2 * fetched


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
    timestamp = int(datetime.datetime.now(datetime.UTC).timestamp())
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
    monkeypatch.setattr("screens.draw_weather.http_get", _mock_get)

    first = _fetch_radar_frames(zoom=7, max_frames=6)
    fetched = len(urls_requested)
    second = _fetch_radar_frames(zoom=7, max_frames=6)

    assert len(first) == 1
    assert len(second) == 1
    assert fetched >= 2 and len(urls_requested) == fetched
    assert first[0].image is not second[0].image


def test_fetch_radar_frames_refreshes_after_ttl(monkeypatch):
    now = 2_000.0
    calls = []
    timestamp = int(datetime.datetime.now(datetime.UTC).timestamp())

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
    now_ts = int(datetime.datetime.now(datetime.UTC).timestamp())
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
    now_ts = int(datetime.datetime.now(datetime.UTC).timestamp())
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
        # Tile fetches run concurrently, so record which paths were
        # requested without assuming a particular completion order.
        timestamps_requested.append(url.split("/")[3])
        return _MockResponse(_png_bytes())

    monkeypatch.setattr("screens.draw_weather.http_get", _mock_get)

    frames = _fetch_radar_frames(zoom=7, max_frames=2)

    # The two most recent frames ("c" then "b") should be fetched and
    # returned in chronological order, even though the concurrent fetch
    # may complete the underlying requests in either order.
    assert sorted(set(timestamps_requested)) == ["b", "c"]
    assert [frame.timestamp for frame in frames] == [now_ts - 300, now_ts - 60]


def test_fetch_rainviewer_frames_tries_alternate_metadata_url(monkeypatch):
    seen_urls = []
    now_ts = int(datetime.datetime.now(datetime.UTC).timestamp())
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


def test_low_power_mode_trims_radar_frame_count_and_animation_loops(monkeypatch):
    import config

    monkeypatch.setattr(config, "DESK_DISPLAY_LOW_POWER", True)
    module = importlib.reload(importlib.import_module("screens.draw_weather"))
    try:
        assert module.RADAR_MAX_FRAMES < 6
        assert module.RADAR_ANIMATION_LOOPS < 3
    finally:
        monkeypatch.setattr(config, "DESK_DISPLAY_LOW_POWER", False)
        importlib.reload(module)


def test_fetch_radar_frames_uses_iem_when_rainviewer_unavailable(monkeypatch):
    sample = Image.new("RGBA", (8, 8), (255, 255, 255, 255))
    monkeypatch.setattr("screens.draw_weather._fetch_rainviewer_frames", lambda zoom, max_frames: [])
    monkeypatch.setattr(
        "screens.draw_weather._fetch_iem_radar_fallback_frames",
        lambda zoom: [RadarFrame(sample, None)],
    )

    frames = _fetch_radar_frames(zoom=7, max_frames=6)

    assert len(frames) == 1


def _tile_png(size=256, color=(200, 200, 200)):
    img = Image.new("RGB", (size, size), color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _set_display(monkeypatch, width, height):
    monkeypatch.setattr("screens.draw_weather.WIDTH", width)
    monkeypatch.setattr("screens.draw_weather.HEIGHT", height)


def test_base_map_at_1080p_stitches_detailed_tiles_without_stretching(monkeypatch):
    _set_display(monkeypatch, 1920, 1080)
    urls = []

    def _mock_get(url, timeout, headers):
        urls.append(url)
        return _MockResponse(_tile_png())

    monkeypatch.setattr("screens.draw_weather.http_get", _mock_get)

    result = _fetch_base_map(zoom=7)

    assert result is not None
    assert result.size == (1920, 1080)
    assert result.info["attribution"] == "© OpenStreetMap"
    zooms = {url.split("/")[3] for url in urls}
    assert zooms == {"9"}
    # 1080 display rows cover one zoom-7 tile: 1024 zoom-9 pixels, so the
    # map is shown near its native size instead of a 256px tile blown up 4x.
    assert 20 <= len(urls) <= 40


def test_base_map_on_small_display_keeps_single_zoom7_tile(monkeypatch):
    _set_display(monkeypatch, 240, 240)
    urls = []

    def _mock_get(url, timeout, headers):
        urls.append(url)
        return _MockResponse(_tile_png())

    monkeypatch.setattr("screens.draw_weather.http_get", _mock_get)

    result = _fetch_base_map(zoom=7)

    assert result is not None and result.size == (240, 240)
    assert len(urls) == 1 and "/7/" in urls[0]


def test_base_map_tiles_are_reused_from_disk(monkeypatch):
    _set_display(monkeypatch, 800, 480)
    urls = []

    def _mock_get(url, timeout, headers):
        urls.append(url)
        return _MockResponse(_tile_png())

    monkeypatch.setattr("screens.draw_weather.http_get", _mock_get)

    assert _fetch_base_map(zoom=7) is not None
    fetched = len(urls)
    _clear_radar_map_caches()  # e.g. a restarted render process
    assert _fetch_base_map(zoom=7) is not None

    assert fetched and len(urls) == fetched


def test_base_map_falls_back_when_any_osm_tile_fails(monkeypatch):
    _set_display(monkeypatch, 800, 480)

    failed = []

    def _mock_get(url, timeout, headers):
        if "openstreetmap" in url and not failed:
            failed.append(url)
            raise RuntimeError("one tile missing")
        return _MockResponse(_tile_png())

    monkeypatch.setattr("screens.draw_weather.http_get", _mock_get)

    result = _fetch_base_map(zoom=7)

    assert result is not None
    assert "CARTO" in result.info["attribution"]


def test_rainviewer_frames_use_512px_tiles_sized_to_large_display(monkeypatch):
    _set_display(monkeypatch, 1920, 1080)
    now_ts = int(datetime.datetime.now(datetime.UTC).timestamp())
    metadata = {"host": "https://tilecache.rainviewer.com", "radar": {"past": [{"path": "/p", "time": now_ts}]}}
    tile_urls = []

    class _JsonResponse(_MockResponse):
        def json(self):
            return metadata

    def _mock_get(url, timeout):
        if "maps.json" in url:
            return _JsonResponse(b"")
        tile_urls.append(url)
        return _MockResponse(_tile_png(512, (0, 0, 255)))

    monkeypatch.setattr("screens.draw_weather.http_get", _mock_get)

    frames = _fetch_radar_frames(zoom=7, max_frames=1)

    assert len(frames) == 1
    assert frames[0].image.size == (1920, 1080)
    assert tile_urls and all("/512/7/" in url for url in tile_urls)


def _mercator_px(lat, lon, zoom):
    import math

    n = 256 * 2**zoom
    lat_rad = math.radians(lat)
    return (lon + 180) / 360 * n, (1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n


def _marker_tile_png(zoom, x, y, size, lat, lon):
    from PIL import ImageDraw

    img = Image.new("RGB", (size, size), (0, 0, 0))
    scale = size / 256
    wx, wy = _mercator_px(lat, lon, zoom)
    cx, cy = (wx - x * 256) * scale, (wy - y * 256) * scale
    ImageDraw.Draw(img).ellipse((cx - 3 * scale, cy - 3 * scale, cx + 3 * scale, cy + 3 * scale), fill=(255, 255, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _bright_centroid(image):
    gray = image.convert("L")
    pixels = gray.load()
    xs = ys = count = 0
    for py in range(gray.height):
        for px in range(gray.width):
            if pixels[px, py] > 100:
                xs += px
                ys += py
                count += 1
    assert count, "marker not visible"
    return xs / count, ys / count


@pytest.mark.parametrize("size", [(1920, 1080), (720, 720), (320, 240)])
def test_radar_overlay_lines_up_with_base_map(monkeypatch, size):
    import re

    _set_display(monkeypatch, *size)
    lat, lon = RADAR_CENTER_LATITUDE, RADAR_CENTER_LONGITUDE  # Chicago
    now_ts = int(datetime.datetime.now(datetime.UTC).timestamp())
    metadata = {"host": "https://tilecache.rainviewer.com", "radar": {"past": [{"path": "/p", "time": now_ts}]}}

    class _JsonResponse(_MockResponse):
        def json(self):
            return metadata

    def _mock_get(url, timeout, headers=None):
        if "maps.json" in url:
            return _JsonResponse(b"")
        radar = re.search(r"/(\d+)/(\d+)/(\d+)/(\d+)/2/1_1\.png$", url)
        if radar:
            tile_px, zoom, x, y = map(int, radar.groups())
            return _MockResponse(_marker_tile_png(zoom, x, y, tile_px, lat, lon))
        zoom, x, y = map(int, re.search(r"/(\d+)/(\d+)/(\d+)\.png$", url).groups())
        return _MockResponse(_marker_tile_png(zoom, x, y, 256, lat, lon))

    monkeypatch.setattr("screens.draw_weather.http_get", _mock_get)

    base = _fetch_base_map(zoom=7)
    radar = _fetch_radar_frames(zoom=7, max_frames=1)[0].image
    map_x, map_y = _bright_centroid(base)
    radar_x, radar_y = _bright_centroid(radar)

    # Map (zoom 7-9, 256px) and radar (zoom 7, 512px) tiles must put the same
    # place on the same display pixel.
    assert abs(map_x - radar_x) <= 1.5
    assert abs(map_y - radar_y) <= 1.5
