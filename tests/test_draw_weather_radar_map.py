from io import BytesIO

from PIL import Image

from screens.draw_weather import _fetch_base_map


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
