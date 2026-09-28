"""NHL playoff bracket data for the NHL Playoffs screen.

Series come from api-web's ``playoff-bracket/{year}`` (the bracket on
nhl.com/playoffs), with the next game and live games taken from the week's
schedule.  Before the playoffs start the first round is projected from the
standings; in the summer, before a new season's bracket exists, last spring's
bracket is shown with its results.
"""

from __future__ import annotations

import datetime
import logging
import threading
import time
from typing import Any, Optional

from services.http_client import NHL_HEADERS, get_session
from services.sports import playoff_bracket16
from services.sports.playoff_bracket16 import has_live_series

REQUEST_TIMEOUT = 10
CACHE_TTL_SECONDS = 120
LIVE_CACHE_TTL_SECONDS = 30

BRACKET_URL = "https://api-web.nhle.com/v1/playoff-bracket/{year}"
SCHEDULE_URL = "https://api-web.nhle.com/v1/schedule/now"
STANDINGS_URL = "https://api-web.nhle.com/v1/standings/now"

# nhl.com's series letters: A-D East first round (Atlantic A/B, Metropolitan
# C/D), E-H West (Central E/F, Pacific G/H), I/J and K/L the second round,
# M and N the conference finals, O the Stanley Cup Final.
SERIES_LETTERS: dict[str, tuple[int, str, int]] = {
    "A": (1, "east", 0), "B": (1, "east", 1), "C": (1, "east", 2), "D": (1, "east", 3),
    "E": (1, "west", 0), "F": (1, "west", 1), "G": (1, "west", 2), "H": (1, "west", 3),
    "I": (2, "east", 0), "J": (2, "east", 1), "K": (2, "west", 0), "L": (2, "west", 1),
    "M": (3, "east", 0), "N": (3, "west", 0), "O": (4, "", 0),
}
ROUND_NAMES = {1: "First Round", 2: "Second Round", 3: "Conference Finals", 4: "Stanley Cup Final"}
# First-round letters by conference and division (top division first).
_DIVISION_LETTERS = {"east": (("A", "B"), ("C", "D")), "west": (("E", "F"), ("G", "H"))}
_DIVISIONS = {"east": ("A", "M"), "west": ("C", "P")}
LIVE_STATES = {"LIVE", "CRIT"}
UPCOMING_STATES = {"FUT", "PRE"}

_SESSION = get_session()
_CACHE: dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()


def season_end_year(now: Optional[datetime.datetime] = None) -> int:
    """The year the current season's playoffs are played in (seasons start in the fall)."""

    now = now or datetime.datetime.now(datetime.timezone.utc)
    return now.year + 1 if now.month >= 9 else now.year


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("default") or next((v for v in value.values() if isinstance(v, str)), "")
    return str(value or "").strip()


def _team(team: Any) -> tuple[str, str, Optional[int]]:
    """``(abbrev, name, id)`` of an api-web team object."""

    if not isinstance(team, dict):
        return "", "", None
    abbr = _text(team.get("abbrev") or team.get("teamAbbrev") or team.get("triCode")).upper()
    name = _text(team.get("commonName") or team.get("name") or team.get("teamCommonName")) or abbr
    return abbr, name, _as_int(team.get("id"))


def series_from_bracket(payload: Any) -> tuple[list[dict], dict[str, str], dict[str, str]]:
    """Parse ``playoff-bracket``: series, team names and seeds ("D1", "WC2")."""

    items = payload.get("series") if isinstance(payload, dict) else None
    series: list[dict] = []
    names: dict[str, str] = {}
    seeds: dict[str, str] = {}
    for item in items or ():
        if not isinstance(item, dict):
            continue
        letter = str(item.get("seriesLetter") or "").strip().upper()
        place = SERIES_LETTERS.get(letter)
        top, top_name, top_id = _team(item.get("topSeedTeam"))
        bottom, bottom_name, bottom_id = _team(item.get("bottomSeedTeam"))
        if place is None or not (top and bottom):
            continue
        round_number, conference, position = place
        wins = {top: _as_int(item.get("topSeedWins")) or 0, bottom: _as_int(item.get("bottomSeedWins")) or 0}
        winning_id = _as_int(item.get("winningTeamId"))
        winner = {top_id: top, bottom_id: bottom}.get(winning_id) if winning_id else None
        winner = winner or playoff_bracket16.winner_from_wins([top, bottom], wins)
        names.update({top: top_name, bottom: bottom_name})
        for abbr, key in ((top, "topSeedRankAbbrev"), (bottom, "bottomSeedRankAbbrev")):
            seed = _text(item.get(key))
            if seed and round_number == 1:
                seeds[abbr] = seed
        series.append({
            "round": round_number,
            "conference": conference,
            "position": position,
            "letter": letter,
            "teams": [top, bottom],
            "wins": wins,
            "winner": winner,
            "live": False,
            "next_start": None,
            "next_time_tbd": False,
            "next_game": None,
        })
    return series, names, seeds


def _schedule_games(payload: Any) -> list[dict]:
    games: list[dict] = []
    if not isinstance(payload, dict):
        return games
    for day in payload.get("gameWeek") or ():
        for game in (day or {}).get("games") or ():
            if isinstance(game, dict) and str(game.get("gameType")) == "3":
                games.append(game)
    return games


def apply_schedule(series: list[dict], payload: Any) -> None:
    """Mark live series and add each open series' next game from the schedule."""

    games = _schedule_games(payload)
    for item in series:
        if item.get("winner"):
            continue
        pair = set(item["teams"])
        matching = [
            game for game in games
            if {_team(game.get("awayTeam"))[0], _team(game.get("homeTeam"))[0]} == pair
        ]
        item["live"] = any(str(game.get("gameState") or "").upper() in LIVE_STATES for game in matching)
        upcoming = sorted(
            (game for game in matching if str(game.get("gameState") or "").upper() in UPCOMING_STATES
             and game.get("startTimeUTC")),
            key=lambda game: str(game.get("startTimeUTC")),
        )
        if upcoming:
            game = upcoming[0]
            item["next_start"] = game.get("startTimeUTC")
            item["next_time_tbd"] = str(game.get("gameScheduleState") or "").upper() == "TBD"
            status = game.get("seriesStatus") if isinstance(game.get("seriesStatus"), dict) else {}
            item["next_game"] = _as_int(status.get("gameNumberOfSeries")) or (sum(item["wins"].values()) + 1)


def _standings_value(row: dict, *keys: str) -> float:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def projected_from_standings(payload: Any) -> tuple[list[dict], dict[str, str], dict[str, str]]:
    """First-round matchups if the playoffs started today (NHL's division format)."""

    rows = payload.get("standings") if isinstance(payload, dict) else None
    rows = [row for row in rows or () if isinstance(row, dict)]
    if not rows or not any(_standings_value(row, "gamesPlayed") for row in rows):
        return [], {}, {}

    def rank_key(row: dict) -> tuple[float, ...]:
        return (-_standings_value(row, "points"), _standings_value(row, "gamesPlayed"),
                -_standings_value(row, "regulationWins"), -_standings_value(row, "regulationPlusOtWins"),
                -_standings_value(row, "wins"), -_standings_value(row, "goalDifferential"))

    series: list[dict] = []
    names: dict[str, str] = {}
    seeds: dict[str, str] = {}
    for conference, division_codes in _DIVISIONS.items():
        conf_rows = [r for r in rows if playoff_bracket16.normalize_conference(_text(r.get("conferenceAbbrev"))) == conference]
        divisions = []
        for code in division_codes:
            ranked = sorted((r for r in conf_rows if _text(r.get("divisionAbbrev")).upper() == code), key=rank_key)
            if len(ranked) < 3:
                break
            divisions.append(ranked)
        if len(divisions) != 2:
            continue
        top_three = {id(r) for division in divisions for r in division[:3]}
        wild_cards = sorted((r for r in conf_rows if id(r) not in top_three), key=rank_key)[:2]
        if len(wild_cards) < 2:
            continue
        # The better division winner plays the second wild card.
        order = sorted(range(2), key=lambda i: rank_key(divisions[i][0]))
        wild_for = {order[0]: wild_cards[1], order[1]: wild_cards[0]}
        for index, division in enumerate(divisions):
            abbrs = [_text(r.get("teamAbbrev")).upper() for r in division[:3]]
            wild = wild_for[index]
            wild_abbr = _text(wild.get("teamAbbrev")).upper()
            for rank, row in enumerate(division[:3], start=1):
                seeds[abbrs[rank - 1]] = f"D{rank}"
                names[abbrs[rank - 1]] = _text(row.get("teamCommonName") or row.get("teamName"))
            seeds[wild_abbr] = "WC2" if wild is wild_cards[1] else "WC1"
            names[wild_abbr] = _text(wild.get("teamCommonName") or wild.get("teamName"))
            letters = _DIVISION_LETTERS[conference][index]
            for letter, teams in ((letters[0], [abbrs[0], wild_abbr]), (letters[1], [abbrs[1], abbrs[2]])):
                round_number, conf, position = SERIES_LETTERS[letter]
                series.append({"round": round_number, "conference": conf, "position": position, "letter": letter,
                               "teams": teams, "wins": {}, "winner": None, "projected": True})
    return series, names, seeds


def _get_json(url: str) -> Any:
    response = _SESSION.get(url, timeout=REQUEST_TIMEOUT, headers=NHL_HEADERS)
    response.raise_for_status()
    return response.json()


def _try_json(url: str) -> Any:
    try:
        return _get_json(url)
    except Exception as exc:  # noqa: BLE001 - each source is optional
        logging.debug("NHL playoffs request failed (%s): %s", url, exc)
        return None


def _bracket(year: int) -> tuple[list[dict], dict[str, str], dict[str, str]]:
    payload = _try_json(BRACKET_URL.format(year=year))
    return series_from_bracket(payload) if payload is not None else ([], {}, {})


def fetch_postseason(*, force: bool = False, now: Optional[datetime.datetime] = None) -> dict[str, Any]:
    """Return ``{"season", "series", "names", "seeds"}``; raises when nothing answered."""

    with _CACHE_LOCK:
        cached = _CACHE.get("data")
        age = time.monotonic() - float(_CACHE.get("fetched_at", 0.0))
    if cached and not force:
        ttl = LIVE_CACHE_TTL_SECONDS if has_live_series(cached) else CACHE_TTL_SECONDS
        if age < ttl:
            return cached

    now = now or datetime.datetime.now(datetime.timezone.utc)
    year = season_end_year(now)
    series, names, seeds = _bracket(year)
    if series:
        apply_schedule(series, _try_json(SCHEDULE_URL))
    else:
        # Between the start of a season and its playoffs, project the first
        # round; otherwise (the summer, or no games played yet) show last spring's.
        if 10 <= now.month or now.month <= 6:
            series, names, seeds = projected_from_standings(_try_json(STANDINGS_URL))
        if not series:
            year -= 1
            series, names, seeds = _bracket(year)
    if not series:
        raise RuntimeError("NHL playoff bracket could not be fetched")
    data = {"season": year, "series": series, "names": names, "seeds": seeds}
    with _CACHE_LOCK:
        _CACHE["data"] = data
        _CACHE["fetched_at"] = time.monotonic()
    return data


__all__ = [
    "ROUND_NAMES",
    "apply_schedule",
    "fetch_postseason",
    "has_live_series",
    "projected_from_standings",
    "season_end_year",
    "series_from_bracket",
]
