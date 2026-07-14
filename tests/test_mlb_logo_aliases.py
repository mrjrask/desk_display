from pathlib import Path

from PIL import Image

from utils import get_mlb_abbreviation, load_team_logo


REPO_ROOT = Path(__file__).resolve().parents[1]
MLB_LOGO_DIR = REPO_ROOT / "images" / "mlb"


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


def test_load_team_logo_supports_all_star_league_abbreviations():
    assert get_mlb_abbreviation("American League All-Stars") == "AL"
    assert get_mlb_abbreviation("National League All-Stars") == "NL"

    american_league_logo = load_team_logo(str(MLB_LOGO_DIR), "AL", box_size=10)
    national_league_logo = load_team_logo(str(MLB_LOGO_DIR), "NL", box_size=10)

    assert american_league_logo is not None
    assert national_league_logo is not None
