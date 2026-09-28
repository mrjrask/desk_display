#!/usr/bin/env python3
"""Render the MLB postseason bracket with the current round's series below it.

The bracket follows mlb.com/postseason: the American League on the left, the
National League on the right, the World Series in the middle, and rounds
running Wild Card → Division Series → LCS toward the centre.  Under it, each
series of the round being played shows both teams, the series score and the
next game (or who won).  The drawing is shared with the NHL and NBA playoff
screens (:mod:`screens.playoff_bracket`).
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Optional

from PIL import Image

from config import CENTRAL_TIME, WIDTH
from screens import playoff_bracket
from screens.playoff_bracket import PlayoffScreen
from services.sports import mlb_postseason
from utils import ScreenImage

SCREEN_ID = "MLB Playoffs"
TITLE = "MLB Playoffs"
SPEC = PlayoffScreen(
    screen_id=SCREEN_ID,
    title=TITLE,
    logo_dir=playoff_bracket.images_dir("mlb"),
    header_logo="MLB",
    round_labels=("WC", "DS", "LCS", "WS"),
)


def series_status(slot: dict, data: dict, now: Optional[datetime.datetime] = None):
    """The line under a series: who won, a live game, or the next game."""

    return playoff_bracket.series_status(slot, (data or {}).get("names") or {}, now)


def _bracket(data: dict) -> dict:
    """:func:`mlb_postseason.build_bracket` in the shared drawing's shape."""

    built = mlb_postseason.build_bracket(data)
    sides = {}
    for side, league in (("left", "AL"), ("right", "NL")):
        rounds = built.get(league) or {}
        sides[side] = [rounds.get("F") or [], rounds.get("D") or [], rounds.get("L") or []]
    return {
        **sides,
        "center": (built.get("W") or [None])[0],
        "seeds": built.get("seeds") or {},
        "mode": "byes",
        "left_logo": "AL",
        "right_logo": "NL",
        "_rounds": built,
    }


def _list_slots(bracket: dict, round_code: str) -> list[dict]:
    built = bracket["_rounds"]
    if round_code == "W":
        return list(built.get("W") or [])
    return [slot for league in mlb_postseason.LEAGUES for slot in (built.get(league) or {}).get(round_code) or []]


def _round_heading(data: dict, round_code: str, projected: bool) -> str:
    name = mlb_postseason.ROUND_NAMES[round_code]
    if projected:
        return f"Projected {name}"
    series = [s for s in data.get("series") or [] if s.get("round") == round_code]
    best_of = max((s.get("best_of") or 0 for s in series), default=0) or mlb_postseason.DEFAULT_BEST_OF[round_code]
    return f"{name} · Best of {best_of}"


def _compose_series_list(width: int, data: dict, bracket: dict, now: datetime.datetime) -> Optional[Image.Image]:
    round_code = mlb_postseason.current_round(data)
    projected = round_code is None
    if projected:
        round_code = "F"
    return playoff_bracket.compose_series_list(
        SPEC, width, _round_heading(data, round_code, projected), _list_slots(bracket, round_code),
        data.get("names") or {}, bracket.get("seeds") or {}, now,
    )


def compose_playoffs_image(data: Any, *, now: Optional[datetime.datetime] = None) -> Image.Image:
    """Render the whole screen (header, bracket, series list) as one tall image."""

    now = now or datetime.datetime.now(CENTRAL_TIME)
    data = data if isinstance(data, dict) else {}
    if not (data.get("series") or data.get("seeds")):
        return playoff_bracket.compose_screen(SPEC, None, None)
    bracket = _bracket(data)
    return playoff_bracket.compose_screen(SPEC, bracket, _compose_series_list(WIDTH, data, bracket, now))


def render_mlb_playoffs(display, data: Any = None, transition: bool = False) -> ScreenImage:
    """Draw the MLB Playoffs screen.

    *data* is the ``mlb_postseason`` feed value; ``None`` fetches it here
    (standalone mode, where screens fetch what they draw).
    """

    if data is None:
        try:
            data = mlb_postseason.fetch_postseason()
        except Exception as exc:  # noqa: BLE001 - draw the empty state instead
            logging.warning("MLB postseason fetch failed: %s", exc)
            data = {}
    return playoff_bracket.show(display, compose_playoffs_image(data), transition)


__all__ = ["SCREEN_ID", "compose_playoffs_image", "render_mlb_playoffs", "series_status"]
