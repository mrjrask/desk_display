"""Tests for Olympic hockey scoreboard helpers."""

from screens.oly_hockey_scoreboard import _team_fallback_text


def test_team_fallback_text_prefers_country_code():
    team = {"abbreviation": "USA", "displayName": "United States"}
    assert _team_fallback_text(team) == "USA"


def test_team_fallback_text_uses_name_when_code_missing():
    team = {"displayName": "Great Britain"}
    assert _team_fallback_text(team) == "GBR"


def test_team_fallback_text_truncates_display_name_when_unmapped():
    team = {"displayName": "Olympic Athletes"}
    assert _team_fallback_text(team) == "OLY"


def test_team_fallback_text_uses_placeholder_when_blank():
    assert _team_fallback_text({}) == "?"
