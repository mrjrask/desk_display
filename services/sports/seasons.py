"""League season identifiers derived from the date being displayed.

ESPN identifies an NBA season by the calendar year in which it ends (the
2026-27 season is ``2027``), while NFL data uses the year in which the season
starts.  Keeping that convention in one place prevents annual hard-coded
season updates.
"""

from __future__ import annotations

import datetime as dt


def nba_season_year(value: dt.date | dt.datetime) -> int:
    """Return ESPN's ending-year identifier for the NBA season at *value*."""

    # July is safely between the Finals and the following preseason.  Using it
    # as the rollover also lets schedule screens discover the newly published
    # season well before opening night.
    return value.year + 1 if value.month >= 7 else value.year


def nfl_season_year(value: dt.date | dt.datetime) -> int:
    """Return the starting year of the NFL season at *value*."""

    # ESPN and nflverse attach preseason games to the upcoming season.  July
    # avoids showing the previous season once those schedules/records appear.
    return value.year if value.month >= 7 else value.year - 1
