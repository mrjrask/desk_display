# External API Reference

This project pulls weather, travel, finance, and sports data from third-party endpoints. The app normalizes most payloads into internal structures so screens can share logic regardless of upstream provider shape. This document lists the active endpoints in code and in `scripts/test_api_connections.py`, plus the key fields relied on by renderers.

> **Tip:** Most API keys can be provided via `.env` (with `CONFIG_LOAD_DOTENV=1`) or the environment.

---


## Connectivity test script

Use the single supported diagnostics script below to probe third-party API
reachability and app-level fetch helper behavior:

```bash
python scripts/test_api_connections.py
```

Common CLI usage:

```bash
# Human-readable output
python scripts/test_api_connections.py

# JSON output for automation/CI parsing
python scripts/test_api_connections.py --json
```

Expected statuses and exit-code behavior:
- `OK`: check passed.
- `SKIP`: check intentionally skipped (usually due to missing optional credentials).
- `FAIL`: connectivity, authentication, parsing, or payload-shape validation failed.
- Exit code `0`: all checks are `OK` or `SKIP`.
- Exit code `1`: one or more checks are `FAIL`.

### Endpoint families covered by the diagnostic script
- Weather: WeatherKit, OpenWeatherMap, RainViewer
- Travel/maps: Google Directions + Static Maps, Apple Maps Directions + Snapshot
- Sports: NHL/NBA/NFL/MLB scoreboards + standings, AHL ICS + HockeyTech
- Finance: Yahoo chart API (used by stock screen helpers)
- App-level helper probes for Bears/Bulls/Blackhawks/Cubs/Sox/Wolves fetch functions

---

## Weather

### Apple WeatherKit (primary)
- **Endpoint:** `https://weatherkit.apple.com/api/v1/weather/{language}/{lat}/{lon}`
- **Data sets:** `currentWeather`, `forecastDaily`, `forecastHourly`, `weatherAlerts`
- **Fields used:**
  - `currentWeather`: `temperature`, `temperatureApparent`, `windSpeed`, `windGust`, `windDirection`, `humidity`, `pressure`, `uvIndex`, `cloudCover`, `asOf`, `conditionCode`, `sunrise`, `sunset`
  - `forecastDaily.days`: `temperatureMax`, `temperatureMin`, `sunrise`, `sunset`, `precipitationChance`, `conditionCode`, `forecastStart`
  - `forecastHourly.hours`: `temperature`, `temperatureApparent`, `precipitationChance`, `windSpeed`, `windGust`, `windDirection`, `uvIndex`, `conditionCode`, `forecastStart`
  - `weatherAlerts.alerts`: alert payloads passed through for display
- **Notes:** JWT auth signed via `WEATHERKIT_TEAM_ID`, `WEATHERKIT_KEY_ID`, `WEATHERKIT_SERVICE_ID`, and either `WEATHERKIT_PRIVATE_KEY` (the PEM key contents) or `WEATHERKIT_KEY_PATH` (path to the downloaded `.p8` private key). `WEATHERKIT_KEY_ID` is only the Apple key identifier and does not replace the private key.

### OpenWeatherMap OneCall (fallback)
- **Endpoint:** `https://api.openweathermap.org/data/3.0/onecall`
- **Fields used:**
  - `current`: `temp`, `feels_like`, `wind_speed`, `wind_gust`, `wind_deg`, `humidity`, `pressure`, `uvi`, `sunrise`, `sunset`, `dt`, `clouds`, `weather[].description/icon`
  - `daily[]`: `temp.max`, `temp.min`, `sunrise`, `sunset`, `pop`, `weather[].description/icon`
  - `hourly[]`: `dt`, `temp`, `feels_like`, `pop`, `wind_speed`, `wind_gust`, `wind_deg`, `uvi`, `weather[].description/icon`
  - `alerts`: passed through for display
- **Notes:** Used when WeatherKit is unavailable; normalized into the same structure.

### RainViewer radar + Google Static Maps
- **RainViewer metadata (primary):** `https://api.rainviewer.com/public/weather-maps.json`
- **RainViewer metadata (fallback):** `https://api.rainviewer.com/public/maps.json`
- **RainViewer tiles:** `https://{host}/{path}/256/{zoom}/{x}/{y}/2/1_1.png`
- **Google Static Maps:** `https://maps.googleapis.com/maps/api/staticmap`
- **Fields used:**
  - RainViewer: `host`, `radar.past`, `radar.nowcast` (`path`, `time`)
  - Google Static Maps: map tiles for the radar background (with `center`, `zoom`, `size`, `maptype`, `key`)

---

## Travel & map services

### Google Directions (travel routes)
- **Endpoint:** `https://maps.googleapis.com/maps/api/directions/json`
- **Fields used:** `routes[].summary`, `legs[].duration` / `duration_in_traffic`, `legs[].steps[].html_instructions`
- **Notes:** Enabled when `GOOGLE_MAPS_API_KEY` is configured. Route filtering is performed in-app on normalized summary/step text.

### Apple Maps Directions
- **Endpoint:** `https://maps-api.apple.com/v1/directions`
- **Fields used:** route travel times (`expectedTravelTime`, `staticTravelTime`, `typicalTravelTime`), summary/name fields, and route steps/polyline data when present
- **Notes:** Auth supports either `APPLE_MAPS_API_KEY`/`MAPKIT_TOKEN` or JWT signing via `APPLE_MAPS_TEAM_ID`, `APPLE_MAPS_KEY_ID`, and private key (`APPLE_MAPS_PRIVATE_KEY` or `APPLE_MAPS_KEY_PATH`). WeatherKit signing keys can be reused as fallback.

### Apple Maps Snapshot
- **Endpoint:** `https://maps-api.apple.com/v1/snapshot`
- **Fields used:** binary map image payload for travel/weather map imagery integrations


---

## Sports

### NHL (Blackhawks)
- **Team schedule helper:** `https://api-web.nhle.com/v1/club-schedule-season/CHI/20252026` (configured team-season feed used by Blackhawks helper screens)
- **League scoreboard:** `https://api-web.nhle.com/v1/scoreboard/{date}` and `/scoreboard/now`
- **Legacy fallback endpoint still supported by some paths:** `https://statsapi.web.nhl.com/api/v1/schedule?date=YYYY-MM-DD&expand=schedule.linescore,schedule.teams`
- **Standings:**
  - `https://statsapi.web.nhl.com/api/v1/standings` (primary)
  - `https://api-web.nhle.com/v1/standings/now` (fallback)
- **Fields used:** game IDs, dates, game state, team records, scores, linescore info, venue, start times, and standings metrics (`divisionRank`, `leagueRecord`, `gamesBack`, `wildCardGamesBack`, `streak`, split records).

### MLB (Cubs / White Sox)
- **Schedule:** `https://statsapi.mlb.com/api/v1/schedule` with team IDs `112` (Cubs) and `145` (White Sox)
- **Standings:** `https://statsapi.mlb.com/api/v1/standings`
- **Fields used:** game IDs, dates, status, team records/scores, venue, probable pitchers, linescore, division standings (W/L/GB/WCGB/streak).

### NBA (Bulls)
- **Scoreboard:** `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard`
- **Standings (primary):** `https://cdn.nba.com/static/json/liveData/standings/league.json`
- **Standings (fallback):** `https://site.web.api.espn.com/apis/v2/sports/basketball/nba/standings`
- **Fields used:** game status, team IDs/triCodes, scores, venue/broadcast info, and standings stats (wins, losses, winPct, streaks, ranks, split records).

### NFL (Bears / league scoreboards)
- **Scoreboard:** `https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard`
- **Standings:** `https://raw.githubusercontent.com/nflverse/nfldata/master/data/standings.csv`
- **Fields used:** game status, team records, standings ranks, streaks, and split records.


### AHL (Chicago Wolves)
- **Schedule (ICS):** Stanza feed configured via `AHL_SCHEDULE_ICS_URL` (defaults to the Wolves calendar)
- **HockeyTech feeds:** `https://lscluster.hockeytech.com/feed/` (base) with league/team parameters
- **Fields used:** game dates, opponents, home/away flags, final scores, and recent scoring details (cached for faster redraws).

---

## Stocks

### Yahoo Finance (yfinance + chart API)
- **Library source:** `yfinance` (Yahoo Finance wrappers)
- **Direct diagnostic endpoint:** `https://query1.finance.yahoo.com/v8/finance/chart/SPY`
- **Fields used:** `regularMarketPrice`, `previousClose`, or historical close values (plus chart payload availability checks in diagnostics).

---

## Maps & imagery

- **Google Static Maps** is used for the weather radar basemap.
- **Team and league logos** are loaded from the local `images/` folder rather than external CDN endpoints.
