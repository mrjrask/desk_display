"""How each screen plays in remote (server/client) mode.

Every screen in :data:`screens_catalog.SCREEN_IDS` has an explicit entry.
The class decides which render package the server produces, so a client can
play motion locally instead of the server streaming frames:

``static``
    One image. The still artifact is the whole screen.
``periodic``
    One image whose data changes often (live games, live aircraft); the
    server refreshes it on a short deadline.
``scrolling_canvas``
    A canvas taller than the screen plus viewport and timing; the client
    scrolls it. A canvas that fits the screen needs no package.
``ticker_overlay``
    A base image plus one looping strip per ticker lane (the news ticker
    sidecar, carried as pixels instead of text).
``finite_animation``
    A bounded animation: a sprite slide (team logos) or at most
    ``MAX_ANIMATION_FRAMES`` frames with durations (radar, standings drops).
``composite``
    A quad of tiles, each with its own bounded frames.
``interactive_focus``
    A composite whose tiles open full screen when tapped; the tile screens
    are declared as interaction-dependency demand.
``client_timed``
    A clock: background plus layout, and the client draws the time, so it
    stays right while offline.
``unsupported``
    Needs hardware on the display itself; not served remotely yet.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

STATIC = "static"
PERIODIC = "periodic"
SCROLLING_CANVAS = "scrolling_canvas"
TICKER_OVERLAY = "ticker_overlay"
FINITE_ANIMATION = "finite_animation"
COMPOSITE = "composite"
INTERACTIVE_FOCUS = "interactive_focus"
CLIENT_TIMED = "client_timed"
UNSUPPORTED = "unsupported"
CLASSES = (STATIC, PERIODIC, SCROLLING_CANVAS, TICKER_OVERLAY, FINITE_ANIMATION, COMPOSITE,
           INTERACTIVE_FOCUS, CLIENT_TIMED, UNSUPPORTED)

# Package kind each class produces (None: the still artifact is enough).
PACKAGE_KINDS: Mapping[str, str | None] = {
    STATIC: None,
    PERIODIC: None,
    SCROLLING_CANVAS: "scroll",
    TICKER_OVERLAY: "ticker",
    FINITE_ANIMATION: "animation",
    COMPOSITE: "composite",
    INTERACTIVE_FOCUS: "composite",
    CLIENT_TIMED: "clock",
    UNSUPPORTED: None,
}
PERIODIC_REFRESH_SECONDS = 60


@dataclass(frozen=True)
class ScreenClass:
    screen_id: str
    kind: str
    note: str = ""

    @property
    def package_kind(self) -> str | None:
        return PACKAGE_KINDS[self.kind]

    @property
    def interactive(self) -> bool:
        return self.kind == INTERACTIVE_FOCUS

    @property
    def remote_supported(self) -> bool:
        return self.kind != UNSUPPORTED


_LOGOS = ("weather", "verano", "bears", "nfl", "nba", "bulls", "hawks", "nhl", "wolves", "cubs", "sox", "mlb")

_TABLE: dict[str, tuple[str, str]] = {
    "date": (CLIENT_TIMED, "date and time; client draws the time"),
    "nixie": (CLIENT_TIMED, "nixie tubes with seconds; client draws the time"),
    "quad": (INTERACTIVE_FOCUS, "configured tile pages; tap a tile to focus it"),
    "weather quad": (INTERACTIVE_FOCUS, "four weather tiles; tap a tile to focus it"),
    "on this day": (SCROLLING_CANVAS, "tall history list"),
    "news headlines": (TICKER_OVERLAY, "news ticker lanes"),
    "news headlines 2": (TICKER_OVERLAY, "second news ticker"),
    "weather1": (STATIC, ""),
    "weather2": (STATIC, ""),
    "air quality": (STATIC, ""),
    "weather alert": (STATIC, "shown only while an alert is active"),
    "weather hourly": (STATIC, ""),
    "weather daily": (STATIC, ""),
    "astronomical": (STATIC, ""),
    "weather radar": (FINITE_ANIMATION, "radar frames loop"),
    "inside": (UNSUPPORTED, "reads a sensor attached to the display itself"),
    "vrnof": (STATIC, ""),
    "bears stand1": (STATIC, ""),
    "bears stand2": (STATIC, ""),
    "bears next": (STATIC, ""),
    "bears next season": (STATIC, ""),
    "bears next season sched": (SCROLLING_CANVAS, "season schedule list"),
    "NFL Scoreboard": (SCROLLING_CANVAS, "scrolls when games overflow"),
    "NFL Scoreboard v2": (SCROLLING_CANVAS, "scrolls when games overflow"),
    "NFL Overview NFC": (FINITE_ANIMATION, "logos drop into place"),
    "NFL Overview AFC": (FINITE_ANIMATION, "logos drop into place"),
    "NFL Standings NFC": (SCROLLING_CANVAS, "division tables"),
    "NFL Standings AFC": (SCROLLING_CANVAS, "division tables"),
    "NBA Scoreboard": (SCROLLING_CANVAS, "scrolls when games overflow"),
    "NBA Scoreboard v2": (SCROLLING_CANVAS, "scrolls when games overflow"),
    "NBA Playoffs": (SCROLLING_CANVAS, "bracket"),
    "NCAAM Scoreboard": (SCROLLING_CANVAS, "scrolls when games overflow"),
    "World Cup Scoreboard": (SCROLLING_CANVAS, "repeating score list"),
    "bulls stand1": (STATIC, ""),
    "bulls last": (STATIC, ""),
    "bulls live": (PERIODIC, "live game"),
    "bulls next": (STATIC, ""),
    "bulls next home": (STATIC, ""),
    "bulls schedule quad": (COMPOSITE, "schedule tiles"),
    "hawks stand1": (STATIC, ""),
    "hawks last": (STATIC, ""),
    "hawks live": (PERIODIC, "live game"),
    "hawks next": (STATIC, ""),
    "hawks next home": (STATIC, ""),
    "hawks schedule quad": (COMPOSITE, "schedule tiles"),
    "NHL Scoreboard": (SCROLLING_CANVAS, "scrolls when games overflow"),
    "NHL Scoreboard v2": (SCROLLING_CANVAS, "scrolls when games overflow"),
    "NHL Playoffs": (SCROLLING_CANVAS, "bracket"),
    "NHL Standings Overview West": (FINITE_ANIMATION, "logos drop into place"),
    "NHL Standings Overview East": (FINITE_ANIMATION, "logos drop into place"),
    "NHL Standings West": (SCROLLING_CANVAS, "division tables"),
    "NHL Standings West v2": (SCROLLING_CANVAS, "division tables"),
    "NHL Standings East": (SCROLLING_CANVAS, "division tables"),
    "NHL Standings East v2": (SCROLLING_CANVAS, "division tables"),
    "wolves last": (STATIC, ""),
    "wolves live": (PERIODIC, "live game"),
    "wolves next": (STATIC, ""),
    "wolves next home": (STATIC, ""),
    "MLB Scoreboard": (SCROLLING_CANVAS, "scrolls when games overflow"),
    "MLB Scoreboard v2": (SCROLLING_CANVAS, "scrolls when games overflow"),
    "NL Overview": (FINITE_ANIMATION, "logos drop into place"),
    "AL Overview": (FINITE_ANIMATION, "logos drop into place"),
    "NL Overview+WC": (FINITE_ANIMATION, "logos drop into place"),
    "AL Overview+WC": (FINITE_ANIMATION, "logos drop into place"),
    "MLB AL Standings": (SCROLLING_CANVAS, "league table"),
    "MLB ALWC Standings": (SCROLLING_CANVAS, "league table"),
    "MLB NL Standings": (SCROLLING_CANVAS, "league table"),
    "MLB NLWC Standings": (SCROLLING_CANVAS, "league table"),
    "adsb stats": (STATIC, ""),
    "adsb live": (PERIODIC, "aircraft overhead now"),
}
for _team in ("cubs", "sox"):
    _TABLE.update({
        f"{_team} stand1": (STATIC, ""),
        f"{_team} stand2": (STATIC, ""),
        f"{_team} stand3": (STATIC, ""),
        f"{_team} last": (STATIC, ""),
        f"{_team} live": (PERIODIC, "live game"),
        f"{_team} no game": (STATIC, ""),
        f"{_team} next": (STATIC, ""),
        f"{_team} next home": (STATIC, ""),
        f"{_team} current series": (STATIC, ""),
        f"{_team} next series": (STATIC, ""),
        f"{_team} next home series": (STATIC, ""),
        f"{_team} schedule quad": (COMPOSITE, "schedule tiles"),
    })
for _logo in _LOGOS:
    _TABLE[f"{_logo} logo"] = (FINITE_ANIMATION, "logo slides across once")

CLASSIFICATIONS: Mapping[str, ScreenClass] = {
    screen: ScreenClass(screen, kind, note) for screen, (kind, note) in _TABLE.items()
}
# Tiles of the fixed interactive quads, as registered in screens/registry.py.
_FIXED_FOCUS_TILES: Mapping[str, tuple[str, ...]] = {
    "weather quad": ("weather1", "air quality", "weather hourly", "weather daily"),
}


def classify(screen_id: str) -> ScreenClass:
    """The class of *screen_id*; raises ``KeyError`` for an unknown screen."""

    return CLASSIFICATIONS[screen_id]


def focus_targets(screen_id: str) -> tuple[str, ...]:
    """Screens a tap on *screen_id*'s tiles can open full screen."""

    if screen_id in _FIXED_FOCUS_TILES:
        return _FIXED_FOCUS_TILES[screen_id]
    if screen_id == "quad":
        from screens.registry import _quad_layout_from_layouts

        _enabled, _speed, pages = _quad_layout_from_layouts()
        tiles = {tile for page in pages for tile in page}
        return tuple(sorted(t for t in tiles if t in CLASSIFICATIONS and t != "quad"))
    return ()


def interaction_targets(screens: Iterable[str]) -> set[str]:
    """Focus targets of every interactive screen in *screens*."""

    targets: set[str] = set()
    for screen in screens:
        entry = CLASSIFICATIONS.get(screen)
        if entry is not None and entry.interactive:
            targets.update(focus_targets(screen))
    return targets


__all__ = [
    "CLASSES",
    "CLASSIFICATIONS",
    "PACKAGE_KINDS",
    "PERIODIC_REFRESH_SECONDS",
    "ScreenClass",
    "classify",
    "focus_targets",
    "interaction_targets",
]
