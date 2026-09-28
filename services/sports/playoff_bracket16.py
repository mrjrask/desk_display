"""A 16-team, best-of-seven playoff bracket (NHL and NBA).

The league modules (:mod:`services.sports.nhl_postseason`,
:mod:`services.sports.nba_postseason`) turn their APIs into a list of series::

    {"round": 1..4, "conference": "west" | "east" | "", "position": int | None,
     "teams": [top, bottom], "wins": {abbr: n}, "winner": abbr | None,
     "live": bool, "next_start": iso | None, "next_time_tbd": bool, "next_game": n | None}

``position`` is the series' place in its conference's column, top to bottom
(0-3 in the first round, 0-1 in the second, 0 in the conference final); the
final is round 4 with no conference.  :func:`build_bracket` lays those out
for :mod:`screens.playoff_bracket`: the West on the left, the East on the
right, and slots the playoffs have not reached filled with the winners that
will meet there.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

ROUNDS = (1, 2, 3, 4)
CONFERENCES = ("west", "east")
WINS_NEEDED = 4
BEST_OF = 7


def normalize_conference(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("w"):
        return "west"
    if text.startswith("e"):
        return "east"
    return ""


def has_live_series(data: Any) -> bool:
    # The render server's snapshot holds frozen mappings, not dicts.
    if not isinstance(data, Mapping):
        return False
    return any(isinstance(s, Mapping) and s.get("live") for s in data.get("series") or ())


def _slot(top: Optional[str] = None, bottom: Optional[str] = None) -> dict:
    return {"teams": [top, bottom], "wins": [None, None], "series": None, "winner": None}


def _apply(slot: dict, series: Mapping) -> None:
    teams = list(series.get("teams") or [None, None])[:2]
    expected_top = slot["teams"][0]
    if expected_top and expected_top in teams:
        teams.sort(key=lambda team: team != expected_top)
    wins = series.get("wins") or {}
    slot["teams"] = teams
    if series.get("projected"):
        return
    slot["wins"] = [wins.get(team, 0) if team else None for team in teams]
    slot["series"] = dict(series)
    slot["winner"] = series.get("winner")


def _fill(slots: list[dict], candidates: list[Mapping]) -> None:
    """Place one round's series: by position, then by expected teams, then in order."""

    unmatched = list(candidates)
    for series in list(unmatched):
        position = series.get("position")
        if isinstance(position, int) and 0 <= position < len(slots) and slots[position]["series"] is None \
                and not slots[position].get("_placed"):
            _apply(slots[position], series)
            slots[position]["_placed"] = True
            unmatched.remove(series)
    open_slots = []
    for slot in slots:
        if slot.pop("_placed", False):
            continue
        expected = {team for team in slot["teams"] if team}
        match = next((s for s in unmatched if expected & set(s.get("teams") or ())), None) if expected else None
        if match is None:
            open_slots.append(slot)
            continue
        unmatched.remove(match)
        _apply(slot, match)
    for slot, series in zip(open_slots, unmatched):
        _apply(slot, series)


def _winner(slot: dict) -> Optional[str]:
    return slot.get("winner")


def build_bracket(data: Any) -> dict[str, Any]:
    data = data if isinstance(data, Mapping) else {}
    series = [s for s in data.get("series") or () if isinstance(s, Mapping)]
    sides: dict[str, list[list[dict]]] = {}
    for conference in CONFERENCES:
        columns: list[list[dict]] = []
        previous: Optional[list[dict]] = None
        for round_number, count in ((1, 4), (2, 2), (3, 1)):
            if previous is None:
                slots = [_slot() for _ in range(count)]
            else:
                slots = [_slot(_winner(previous[2 * i]), _winner(previous[2 * i + 1])) for i in range(count)]
            _fill(slots, [s for s in series if s.get("round") == round_number
                          and normalize_conference(s.get("conference")) == conference])
            columns.append(slots)
            previous = slots
        sides[conference] = columns
    final = _slot(_winner(sides["west"][2][0]), _winner(sides["east"][2][0]))
    _fill([final], [s for s in series if s.get("round") == 4])
    return {
        "left": sides["west"],
        "right": sides["east"],
        "center": final,
        "seeds": dict(data.get("seeds") or {}),
        "mode": "pairs",
    }


def round_slots(bracket: Mapping, round_number: int) -> list[dict]:
    """A round's slots, West then East (the final alone for round 4)."""

    if round_number == 4:
        return [bracket["center"]]
    index = round_number - 1
    return list(bracket["left"][index]) + list(bracket["right"][index])


def current_round(data: Any) -> Optional[int]:
    """The earliest round with a series still going, else the last one played."""

    data = data if isinstance(data, Mapping) else {}
    series = [s for s in data.get("series") or () if isinstance(s, Mapping) and not s.get("projected")]
    if not series:
        return None
    rounds = sorted({int(s.get("round") or 0) for s in series if s.get("round") in ROUNDS})
    if not rounds:
        return None
    for round_number in rounds:
        if any(not s.get("winner") for s in series if s.get("round") == round_number):
            return round_number
    return rounds[-1]


def winner_from_wins(teams: list[Optional[str]], wins: Mapping[str, int]) -> Optional[str]:
    for team in teams:
        if team and (wins.get(team) or 0) >= WINS_NEEDED:
            return team
    return None


__all__ = [
    "BEST_OF",
    "build_bracket",
    "current_round",
    "has_live_series",
    "normalize_conference",
    "round_slots",
    "winner_from_wins",
]
