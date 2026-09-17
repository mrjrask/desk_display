from pathlib import Path

from config import AHL_FALLBACK_LOGO, AHL_IMAGES_DIR
from data_fetch import _AHL_FALLBACK_ONLY_ABBRS, _AHL_TEAM_ABBR_OVERRIDES


def _logo_for_abbreviation(abbreviation: str) -> Path | None:
    """Mirror the case-sensitive lookup order used by the Wolves screen."""

    logo_dir = Path(AHL_IMAGES_DIR)
    for variant in (abbreviation.upper(), abbreviation.lower()):
        candidate = logo_dir / f"{variant}.png"
        if candidate.exists():
            return candidate
    return None


def test_ahl_fallback_logo_exists():
    assert Path(AHL_FALLBACK_LOGO).is_file()


def test_configured_ahl_abbreviations_have_logos_or_documented_fallbacks():
    configured = set(_AHL_TEAM_ABBR_OVERRIDES.values())

    assert set(_AHL_FALLBACK_ONLY_ABBRS) <= configured
    assert all(reason.strip() for reason in _AHL_FALLBACK_ONLY_ABBRS.values())

    missing = {
        abbreviation
        for abbreviation in configured - set(_AHL_FALLBACK_ONLY_ABBRS)
        if _logo_for_abbreviation(abbreviation) is None
    }
    assert not missing, f"Missing AHL logo assets for: {', '.join(sorted(missing))}"
