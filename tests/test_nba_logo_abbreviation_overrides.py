import pytest

from screens import draw_bulls_schedule, nba_scoreboard


def test_bulls_logo_override_maps_washington_and_brooklyn_to_expected_files():
    overrides = draw_bulls_schedule.LOGO_ABBREVIATION_OVERRIDES

    assert overrides["WAS"] == "WSH"
    assert overrides["BKN"] == "BRK"


@pytest.mark.parametrize(
    ("upstream_abbr", "canonical_logo_abbr"),
    [("SAS", "SA"), ("NYK", "NY"), ("BKN", "BRK"), ("WAS", "WSH")],
)
def test_nba_scoreboard_converts_upstream_feed_codes_to_canonical_logo_codes(
    upstream_abbr, canonical_logo_abbr
):
    upstream_team = {"teamTricode": upstream_abbr}

    assert upstream_team["teamTricode"] == upstream_abbr
    assert nba_scoreboard._team_logo_abbr(upstream_team) == canonical_logo_abbr
