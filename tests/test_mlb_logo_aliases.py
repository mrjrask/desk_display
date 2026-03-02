from PIL import Image

from utils import load_team_logo


def _write_logo(path):
    Image.new("RGBA", (8, 6), (255, 0, 0, 255)).save(path)


def test_load_team_logo_supports_wbc_tricode_alias(tmp_path):
    _write_logo(tmp_path / "Italy.png")

    logo = load_team_logo(str(tmp_path), "ITA", box_size=10)

    assert logo is not None


def test_load_team_logo_supports_uppercase_country_name(tmp_path):
    _write_logo(tmp_path / "Dominican Republic.png")

    logo = load_team_logo(str(tmp_path), "DOMINICAN REPUBLIC", box_size=10)

    assert logo is not None
