import datetime
from io import BytesIO

from PIL import Image

from screens.draw_weather import (
    RADAR_ANIMATION_LOOPS,
    RADAR_CENTER_LATITUDE,
    RADAR_CENTER_LONGITUDE,
    RadarFrame,
    _fetch_base_map,
    _fetch_radar_frames,
    draw_weather_radar,
)


class _MockResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None


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

    monkeypatch.setattr("screens.draw_weather.requests.get", _mock_get)

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

    monkeypatch.setattr("screens.draw_weather.requests.get", _mock_get)

    result = _fetch_base_map(zoom=7)

    assert result is not None
    assert len(seen_urls) == 2
    assert "openstreetmap" in seen_urls[0]
    assert "cartocdn.com/light_all" in seen_urls[1]


def test_fetch_base_map_uses_chicago_center_coordinates(monkeypatch):
    seen_coords = []

    def _mock_latlon_to_tile(lat, lon, zoom):
        seen_coords.append((lat, lon, zoom))
        return (10, 20, 0.0, 0.0)

    def _mock_get(url, timeout, headers):
        return _MockResponse(_png_bytes())

    monkeypatch.setattr("screens.draw_weather._latlon_to_tile", _mock_latlon_to_tile)
    monkeypatch.setattr("screens.draw_weather.requests.get", _mock_get)

    result = _fetch_base_map(zoom=7)

    assert result is not None
    assert seen_coords == [(RADAR_CENTER_LATITUDE, RADAR_CENTER_LONGITUDE, 7)]


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

    monkeypatch.setattr("screens.draw_weather.requests.get", _mock_get)

    _fetch_radar_frames(zoom=7, max_frames=2)

    assert timestamps_requested == ["c", "b"]
