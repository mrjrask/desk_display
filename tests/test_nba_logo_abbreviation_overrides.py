from screens import draw_bulls_schedule, nba_scoreboard


def test_bulls_logo_override_maps_washington_and_brooklyn_to_expected_files():
    overrides = draw_bulls_schedule.LOGO_ABBREVIATION_OVERRIDES

    assert overrides["WAS"] == "WSH"
    assert overrides["BKN"] == "BRK"


def test_nba_scoreboard_logo_abbreviation_override_maps_washington_and_brooklyn():
    team_washington = {"teamTricode": "WAS"}
    team_brooklyn = {"teamTricode": "BKN"}

    assert nba_scoreboard._team_logo_abbr(team_washington) == "WSH"
    assert nba_scoreboard._team_logo_abbr(team_brooklyn) == "BRK"
