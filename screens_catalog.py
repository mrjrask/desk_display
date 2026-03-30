# screens_catalog.py
# A single source of truth for all of your possible screen IDs.

LEGACY_SCOREBOARD_V2_SCREEN_MAP = {
    "NFL Scoreboard v2": "NFL Scoreboard",
    "NHL Scoreboard v2": "NHL Scoreboard",
    "NBA Scoreboard v2": "NBA Scoreboard",
    "MLB Scoreboard v2": "MLB Scoreboard",
    "NHL Standings Overview v2 West": "NHL Standings Overview West",
    "NHL Standings Overview v2 East": "NHL Standings Overview East",
}

LEGACY_SCREEN_ID_MAP = {
    # Removed legacy IDs are canonicalized to active equivalents so older
    # persisted configs continue to load after upgrades.
    "time": "nixie",
    "sensors": "inside",
}


def canonical_screen_id(screen_id: str) -> str:
    mapped_id = LEGACY_SCOREBOARD_V2_SCREEN_MAP.get(screen_id, screen_id)
    return LEGACY_SCREEN_ID_MAP.get(mapped_id, mapped_id)


RAW_SCREEN_IDS = [
    "date", "nixie",
    "quad",
    "weather logo", "weather1", "weather2", "weather hourly", "weather radar", "inside",
    "verano logo", "vrnof",
    "weather logo", "weather1", "weather2", "weather hourly", "weather daily", "weather radar", "inside",
    "verano logo", "vrnof",
    "bears logo", "bears stand1", "bears stand2", "bears next", "bears next season", "nfl logo", "NFL Scoreboard", "NFL Overview NFC", "NFL Overview AFC", "NFL Standings NFC", "NFL Standings AFC", "nba logo", "NBA Scoreboard", "NBA Playoffs", "NCAAM Scoreboard", "bulls logo", "bulls stand1", "bulls last", "bulls live", "bulls next", "bulls next home",
    "hawks logo", "hawks stand1", "hawks last", "hawks live", "hawks next", "hawks next home", "nhl logo", "NHL Scoreboard", "NHL Playoffs", "NHL Standings Overview West", "NHL Standings Overview East", "NHL Standings West", "NHL Standings West v2", "NHL Standings East", "NHL Standings East v2",
    "wolves logo", "wolves last", "wolves next", "wolves next home",
    "cubs logo", "cubs stand1", "cubs stand2",
    "cubs last", "cubs result", "cubs live", "cubs next", "cubs next home",
    "sox logo", "sox stand1", "sox stand2",
    "sox last", "sox live", "sox next", "sox next home",
    "mlb logo", "MLB Scoreboard",
    "NL Overview",
    "AL Overview", "MLB AL Standings", "MLB NL Standings",
]

SCREEN_IDS = list(dict.fromkeys(RAW_SCREEN_IDS))
