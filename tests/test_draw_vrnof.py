from PIL import Image
from PIL import ImageDraw
import time

from screens import draw_vrnof


def test_get_logo_casts_scaled_height_to_int(monkeypatch, tmp_path):
    logo_path = tmp_path / "verano.jpg"
    Image.new("RGB", (40, 20), (255, 255, 255)).save(logo_path)

    monkeypatch.setattr(draw_vrnof, "LOGO_PATH", str(logo_path))
    monkeypatch.setattr(draw_vrnof, "LOGO_HEIGHT", 81.5)
    monkeypatch.setattr(draw_vrnof, "_LOGO", None)

    logo = draw_vrnof._get_logo()

    assert logo is not None
    assert logo.height == 82


def test_build_image_places_price_block_below_title(monkeypatch):
    monkeypatch.setattr(draw_vrnof, "WIDTH", 320)
    monkeypatch.setattr(draw_vrnof, "HEIGHT", 180)
    monkeypatch.setattr(draw_vrnof, "_get_logo", lambda: None)
    monkeypatch.setitem(
        draw_vrnof._cache,
        "price",
        1.23,
    )
    monkeypatch.setitem(draw_vrnof._cache, "change_val", -0.1)
    monkeypatch.setitem(draw_vrnof._cache, "change_pct", -5.0)
    monkeypatch.setitem(draw_vrnof._cache, "all_time", "-7.10%")
    monkeypatch.setitem(draw_vrnof._cache, "ts", time.time())

    calls = []
    original_text = ImageDraw.ImageDraw.text

    def _capture_text(self, xy, text, *args, **kwargs):
        calls.append((text, xy))
        return original_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", _capture_text)

    draw_vrnof._build_image("VRNO")

    positions = {text: xy for text, xy in calls}
    title_y = positions["VRNO"][1]
    price_text = next(text for text in positions if text.startswith("$"))
    price_y = positions[price_text][1]

    probe = ImageDraw.Draw(Image.new("RGB", (320, 180), (0, 0, 0)))
    _, title_h = probe.textsize("VRNO", font=draw_vrnof.FONT_STOCK_TITLE)

    assert price_y >= title_y + title_h + draw_vrnof.LOGO_GAP
