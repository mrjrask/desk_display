# External API Reference

This project pulls weather, travel, and sports data from a variety of third-party endpoints. The app normalizes all payloads into internal structures so screens can share logic regardless of the upstream source. This document lists the endpoints we call and the fields we rely on.

> **Tip:** Most API keys can be provided via `.env` (with `CONFIG_LOAD_DOTENV=1`) or the environment.

---


## Connectivity test script

Run the repository script below to probe all configured third-party API connections:

```bash
python scripts/test_api_connections.py
```

The script reports `OK`, `SKIP` (missing credentials), and `FAIL` (network/auth/payload issues) for each API family and returns a non-zero exit code only when at least one check fails.

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
- **Notes:** JWT auth signed via `WEATHERKIT_TEAM_ID`, `WEATHERKIT_KEY_ID`, `WEATHERKIT_SERVICE_ID`, and `WEATHERKIT_PRIVATE_KEY/WEATHERKIT_KEY_PATH`.

### OpenWeatherMap OneCall (fallback)
- **Endpoint:** `https://api.openweathermap.org/data/3.0/onecall`
- **Fields used:**
  - `current`: `temp`, `feels_like`, `wind_speed`, `wind_gust`, `wind_deg`, `humidity`, `pressure`, `uvi`, `sunrise`, `sunset`, `dt`, `clouds`, `weather[].description/icon`
  - `daily[]`: `temp.max`, `temp.min`, `sunrise`, `sunset`, `pop`, `weather[].description/icon`
  - `hourly[]`: `dt`, `temp`, `feels_like`, `pop`, `wind_speed`, `wind_gust`, `wind_deg`, `uvi`, `weather[].description/icon`
  - `alerts`: passed through for display
- **Notes:** Used when WeatherKit is unavailable; normalized into the same structure.

### RainViewer radar + Google Static Maps
- **RainViewer metadata:** `https://api.rainviewer.com/public/weather-maps.json`
- **RainViewer tiles:** `https://{host}/{path}/256/{zoom}/{x}/{y}/2/1_1.png`
- **Google Static Maps:** `https://maps.googleapis.com/maps/api/staticmap`
- **Fields used:**
  - RainViewer: `host`, `radar.past`, `radar.nowcast` (`path`, `time`)
  - Google Static Maps: map tiles for the radar background (with `center`, `zoom`, `size`, `maptype`, `key`)

---

## Travel

### Google Maps Directions API (v1 travel screens)
- **Endpoint:** `https://maps.googleapis.com/maps/api/directions/json`
- **Fields used:**
  - From each route’s first `leg`: `duration` and `duration_in_traffic` (`text`/`value`), `summary`, `steps[].html_instructions`
- **Notes:** Values are normalized into `_duration_text`, `_duration_sec`, `_summary`, and `_steps_text` for screen rendering.

### Apple Maps Web Services (v2 travel screens)
- **Directions:** `https://maps-api.apple.com/v1/directions`
- **Snapshot:** `https://maps-api.apple.com/v1/snapshot`
- **Fields used (directions):**
  - Route fields: `expectedTravelTime`, `staticTravelTime` / `typicalTravelTime`, `name` / `summary`, and route step instructions
- **Fields used (snapshots):**
  - Map image response bytes for the travel map background
- **Notes:** Auth uses either `APPLE_MAPS_API_KEY`/`MAPKIT_TOKEN` or a JWT signed with `APPLE_MAPS_TEAM_ID`, `APPLE_MAPS_KEY_ID`, and `APPLE_MAPS_PRIVATE_KEY/APPLE_MAPS_KEY_PATH`.

---

## Sports

### NHL (Blackhawks)
- **Schedule / scoreboard:**
  - `https://statsapi.web.nhl.com/api/v1/schedule?date=YYYY-MM-DD&expand=schedule.linescore,schedule.teams` (primary when DNS works)
  - `https://api-web.nhle.com/v1/scoreboard/{date}` and `/scoreboard/now` (fallback)
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

### Yahoo Finance (yfinance)
- **API source:** `yfinance` (Yahoo Finance)
- **Fields used:** `regularMarketPrice`, `previousClose`, or historical close values to compute price + change + all-time percentage.

---

## Maps & imagery

- **Google Static Maps** is used both for the weather radar basemap and the legacy travel map.
- **Team and league logos** are loaded from the local `images/` folder rather than external CDN endpoints.
