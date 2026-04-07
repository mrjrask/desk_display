from __future__ import annotations

import io

from PIL import Image

from screens import draw_wrigley


def test_candidate_urls_prioritize_webcam_like_sources():
    html = """
    <html>
      <body>
        <img src="https://cdn.example.com/brand/square-logo.png" />
        <img src="https://cams.example.com/wrigley/live_snapshot_hd.jpg" />
      </body>
    </html>
    """

    urls = draw_wrigley._candidate_urls_from_html(html, draw_wrigley.WRIGLEY_PAGE_URL)

    assert urls[0] == "https://cams.example.com/wrigley/live_snapshot_hd.jpg"


def test_download_wrigley_frame_skips_square_like_images(monkeypatch):
    html = """
    <html>
      <body>
        <img src="https://cdn.example.com/assets/square-promo.jpg" />
        <img src="https://cams.example.com/wrigley/live.jpg" />
      </body>
    </html>
    """

    class _Resp:
        def __init__(self, text="", content=b"", content_type="text/html"):
            self.text = text
            self.content = content
            self.headers = {"content-type": content_type}

        def raise_for_status(self):
            return None

    def _img_bytes(size):
        img = Image.new("RGB", size, (10, 20, 30))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    def _fake_get(url, timeout=None, headers=None):  # noqa: ARG001
        if url == draw_wrigley.WRIGLEY_PAGE_URL:
            return _Resp(text=html, content_type="text/html")
        if "square-promo" in url:
            return _Resp(content=_img_bytes((300, 300)), content_type="image/jpeg")
        if "live.jpg" in url:
            return _Resp(content=_img_bytes((1280, 720)), content_type="image/jpeg")
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(draw_wrigley._SESSION, "get", _fake_get)

    image, source = draw_wrigley._download_wrigley_frame()

    assert image is not None
    assert source == "https://cams.example.com/wrigley/live.jpg"


def test_is_plausible_webcam_frame_rejects_square_images():
    assert not draw_wrigley._is_plausible_webcam_frame(Image.new("RGB", (300, 300)))
    assert draw_wrigley._is_plausible_webcam_frame(Image.new("RGB", (1280, 720)))
