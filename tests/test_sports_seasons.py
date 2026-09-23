import datetime

from services.sports.seasons import nba_season_year, nfl_season_year


def test_nba_season_rolls_forward_without_an_annual_constant():
    assert nba_season_year(datetime.date(2026, 6, 30)) == 2026
    assert nba_season_year(datetime.date(2026, 7, 1)) == 2027
    assert nba_season_year(datetime.date(2027, 7, 1)) == 2028


def test_nfl_season_uses_starting_year_and_rolls_forward_automatically():
    assert nfl_season_year(datetime.date(2026, 6, 30)) == 2025
    assert nfl_season_year(datetime.date(2026, 7, 1)) == 2026
    assert nfl_season_year(datetime.date(2027, 7, 1)) == 2027
