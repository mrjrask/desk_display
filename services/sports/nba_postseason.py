"""NBA playoff bracket data for the NBA Playoffs screen.

Series come from the NBA's live bracket feed (the bracket on nba.com/playoffs),
which carries each series' seeds, wins, winner and next game.  The feed keeps
last season's bracket until the next one is drawn, so the screen shows the
most recent playoffs out of season.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

from services.http_client import get_session
from services.sports import playoff_bracket16
from services.sports.playoff_bracket16 import has_live_series

REQUEST_TIMEOUT = 10
CACHE_TTL_SECONDS = 120
LIVE_CACHE_TTL_SECONDS = 30

BRACKET_URLS = (
    "https://cdn.nba.com/static/json/liveData/playoffbracket/playoffbracket_00.json",
    "https://nba-prod-us-east-1-media.s3.amazonaws.com/json/liveData/playoffbracket/playoffbracket_00.json",
)
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json",
}
ROUND_NAMES = {1: "First Round", 2: "Conference Semifinals", 3: "Conference Finals", 4: "NBA Finals"}
# First-round slot of the higher seed, top to bottom: 1v8, 4v5, 3v6, 2v7.
FIRST_ROUND_POSITION = {1: 0, 4: 1, 3: 2, 2: 3}
# The logo files in images/nba use these codes for a few teams.
LOGO_CODES = {"BKN": "BRK", "GSW": "GS", "NOP": "NO", "NYK": "NY", "PHO": "PHX", "SAS": "SA", "WAS": "WSH"}
LIVE_STATUS = 2

_SESSION = get_session()
_CACHE: dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def team_code(tricode: Any) -> str:
    code = str(tricode or "").strip().upper()
    return LOGO_CODES.get(code, code)


def _series_items(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("playoffBracketSeries", "series"):
        if isinstance(payload.get(key), list):
            return _series_items(payload[key])
    return _series_items(payload.get("bracket"))


def series_from_bracket(payload: Any) -> tuple[list[dict], dict[str, str], dict[str, int]]:
    """Parse the bracket feed: series, team names and conference seeds."""

    series: list[dict] = []
    names: dict[str, str] = {}
    seeds: dict[str, int] = {}
    for item in _series_items(payload):
        round_number = _as_int(item.get("roundNumber"))
        if round_number not in playoff_bracket16.ROUNDS:
            continue  # the play-in tournament
        high, low = team_code(item.get("highSeedTricode")), team_code(item.get("lowSeedTricode"))
        if not (high and low):
            continue
        high_rank, low_rank = _as_int(item.get("highSeedRank")), _as_int(item.get("lowSeedRank"))
        conference = "" if round_number == 4 else playoff_bracket16.normalize_conference(item.get("seriesConference"))
        wins = {high: _as_int(item.get("highSeedSeriesWins")) or 0, low: _as_int(item.get("lowSeedSeriesWins")) or 0}
        winner_id = _as_int(item.get("seriesWinner"))
        winner = {_as_int(item.get("highSeedId")): high, _as_int(item.get("lowSeedId")): low}.get(winner_id) \
            if winner_id else None
        winner = winner or playoff_bracket16.winner_from_wins([high, low], wins)
        for code, key, rank in ((high, "highSeedName", high_rank), (low, "lowSeedName", low_rank)):
            names[code] = str(item.get(key) or code)
            if rank and round_number == 1:
                seeds[code] = rank
        live = not winner and _as_int(item.get("nextGameStatus")) == LIVE_STATUS
        upcoming = not winner and not live and _as_int(item.get("nextGameStatus")) == 1
        series.append({
            "round": round_number,
            "conference": conference,
            "position": FIRST_ROUND_POSITION.get(high_rank or 0) if round_number == 1 else None,
            "teams": [high, low],
            "wins": wins,
            "winner": winner,
            "live": live,
            "next_start": (item.get("nextGameDateTimeUTC") or None) if upcoming else None,
            "next_time_tbd": False,
            "next_game": _as_int(item.get("nextGameNumber")) if upcoming else None,
        })
    return series, names, seeds


def fetch_postseason(*, force: bool = False) -> dict[str, Any]:
    """Return ``{"series", "names", "seeds"}``; raises when the bracket could not be read."""

    with _CACHE_LOCK:
        cached = _CACHE.get("data")
        age = time.monotonic() - float(_CACHE.get("fetched_at", 0.0))
    if cached and not force:
        ttl = LIVE_CACHE_TTL_SECONDS if has_live_series(cached) else CACHE_TTL_SECONDS
        if age < ttl:
            return cached

    series: list[dict] = []
    names: dict[str, str] = {}
    seeds: dict[str, int] = {}
    for url in BRACKET_URLS:
        try:
            response = _SESSION.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)
            response.raise_for_status()
            series, names, seeds = series_from_bracket(response.json())
        except Exception as exc:  # noqa: BLE001 - try the mirror
            logging.debug("NBA playoff bracket request failed (%s): %s", url, exc)
            continue
        if series:
            break
    if not series:
        raise RuntimeError("NBA playoff bracket could not be fetched")
    data = {"series": series, "names": names, "seeds": seeds}
    with _CACHE_LOCK:
        _CACHE["data"] = data
        _CACHE["fetched_at"] = time.monotonic()
    return data


__all__ = ["ROUND_NAMES", "fetch_postseason", "has_live_series", "series_from_bracket", "team_code"]
