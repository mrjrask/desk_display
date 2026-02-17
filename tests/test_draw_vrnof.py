from PIL import Image

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
