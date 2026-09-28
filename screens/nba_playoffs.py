#!/usr/bin/env python3
"""Render the NBA playoff bracket with the current round's series below it.

Same design as MLB Playoffs (:mod:`screens.playoff_bracket`): the Western
Conference on the left, the Eastern on the right, the NBA Finals in the
middle, and under the bracket one line per series of the round being played
with the series score and the next game (or who won).
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Optional

from PIL import Image

from screens import playoff_bracket
from screens.playoff_bracket import PlayoffScreen
from services.sports import nba_postseason
from utils import ScreenImage

SCREEN_ID = "NBA Playoffs"
TITLE = "NBA Playoffs"
SPEC = PlayoffScreen(
    screen_id=SCREEN_ID,
    title=TITLE,
    logo_dir=playoff_bracket.images_dir("nba"),
    header_logo="NBA",
    round_labels=("R1", "R2", "CF", "Finals"),
)


def compose_playoffs_image(data: Any, *, now: Optional[datetime.datetime] = None) -> Image.Image:
    """Render the whole screen (header, bracket, series list) as one tall image."""

    # images/nba has no conference logos, so the bracket names each side.
    return playoff_bracket.compose_sixteen_team_image(SPEC, data, nba_postseason.ROUND_NAMES, now=now)


def render_nba_playoffs(display, data: Any = None, transition: bool = False) -> ScreenImage:
    """Draw the NBA Playoffs screen.

    *data* is the ``nba_playoffs`` feed value; ``None`` fetches it here
    (standalone mode, where screens fetch what they draw).
    """

    if data is None:
        try:
            data = nba_postseason.fetch_postseason()
        except Exception as exc:  # noqa: BLE001 - draw the empty state instead
            logging.warning("NBA playoffs fetch failed: %s", exc)
            data = {}
    return playoff_bracket.show(display, compose_playoffs_image(data), transition)


__all__ = ["SCREEN_ID", "compose_playoffs_image", "render_nba_playoffs"]
