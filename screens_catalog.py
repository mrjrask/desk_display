# screens_catalog.py
# A single source of truth for all of your possible screen IDs.

RAW_SCREEN_IDS = [
    "date", "time", "nixie",
    "weather logo", "weather1", "weather2", "weather hourly", "weather radar", "inside", "sensors",
    "verano logo", "vrnof", "travel", "travel map", "travel v2", "travel map v2",
    "weather logo", "weather1", "weather2", "weather hourly", "weather daily", "weather radar", "inside", "sensors",
    "verano logo", "vrnof", "travel", "travel map",
    "bears logo", "bears stand1", "bears stand2", "bears next", "bears next season", "nfl logo", "NFL Scoreboard", "NFL Scoreboard v2", "NFL Overview NFC", "NFL Overview AFC", "NFL Standings NFC", "NFL Standings AFC", "nba logo", "NBA Scoreboard", "NBA Scoreboard v2", "bulls logo", "bulls stand1", "bulls stand2", "bulls last", "bulls live", "bulls next", "bulls next home",
    "hawks logo", "hawks stand1", "hawks stand2", "hawks last", "hawks live", "hawks next", "hawks next home", "nhl logo", "NHL Scoreboard", "NHL Scoreboard v2", "NHL Standings Overview West", "NHL Standings Overview East", "NHL Standings Overview v2 West", "NHL Standings Overview v2 East", "NHL Standings Overview v3 West", "NHL Standings Overview v3 East", "NHL Standings West", "NHL Standings West v2", "NHL Standings East", "NHL Standings East v2",
    "wolves logo", "wolves last", "wolves next", "wolves next home",
    "cubs logo", "cubs stand1", "cubs stand2",
    "cubs last", "cubs result", "cubs live", "cubs next", "cubs next home",
    "sox logo", "sox stand1", "sox stand2",
    "sox last", "sox live", "sox next", "sox next home",
    "mlb logo", "MLB Scoreboard", "MLB Scoreboard v2",
    "NL Overview", "NL East", "NL Central", "NL West", "NL Wild Card",
    "AL Overview", "AL East", "AL Central", "AL West", "AL Wild Card",
]

SCREEN_IDS = list(dict.fromkeys(RAW_SCREEN_IDS))
