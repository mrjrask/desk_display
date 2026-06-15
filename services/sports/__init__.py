"""Stable sports data-fetch APIs used by providers and screen renderers."""

from .mlb import fetch_scoreboard as fetch_mlb_scoreboard, scoreboard_date as mlb_scoreboard_date
from .nba import fetch_scoreboard as fetch_nba_scoreboard, scoreboard_date as nba_scoreboard_date
from .ncaam import fetch_scoreboard as fetch_ncaam_scoreboard, scoreboard_date as ncaam_scoreboard_date
from .nfl import (
    fetch_next_scoreboard as fetch_nfl_next_scoreboard,
    fetch_scoreboard as fetch_nfl_scoreboard,
    fetch_week_scoreboard as fetch_nfl_week_scoreboard,
)
from .nhl import fetch_scoreboard as fetch_nhl_scoreboard, scoreboard_date as nhl_scoreboard_date
from .world_cup import fetch_scoreboard as fetch_world_cup_scoreboard, scoreboard_date as world_cup_scoreboard_date

__all__ = [
    "fetch_mlb_scoreboard",
    "mlb_scoreboard_date",
    "fetch_nba_scoreboard",
    "nba_scoreboard_date",
    "fetch_ncaam_scoreboard",
    "ncaam_scoreboard_date",
    "fetch_nfl_scoreboard",
    "fetch_nfl_week_scoreboard",
    "fetch_nfl_next_scoreboard",
    "fetch_nhl_scoreboard",
    "nhl_scoreboard_date",
    "fetch_world_cup_scoreboard",
    "world_cup_scoreboard_date",
]
