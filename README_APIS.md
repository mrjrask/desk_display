# External API Reference

This project pulls game, weather, and travel data from several third-party APIs. This document lists each source, the endpoints we call, and the fields we actually consume in the application.

## Weather

### Apple WeatherKit (primary)
* **Endpoint:** `https://weatherkit.apple.com/api/v1/weather/{language}/{lat}/{lon}` with `dataSets=currentWeather,forecastDaily,forecastHourly,weatherAlerts`.
* **Data we use:**
  * `currentWeather`: `temperature`, `temperatureApparent`, `windSpeed`, `windGust`, `windDirection`, `humidity`, `pressure`, `uvIndex`, `cloudCover`, `asOf`, plus mapped `conditionCode` → `description/icon`. Stored as `current.temp`, `current.feels_like`, `current.wind_speed`, `current.wind_gust`, `current.wind_deg`, `current.humidity`, `current.pressure`, `current.uvi`, `current.clouds`, `current.dt`, `current.sunrise`, `current.sunset`.
  * `forecastDaily.days`: `temperatureMax`, `temperatureMin`, `sunrise`, `sunset`, `precipitationChance`, `conditionCode`, `forecastStart` mapped into `daily[].temp.max/min`, `daily[].sunrise`, `daily[].sunset`, `daily[].pop`, `daily[].weather[0].description/icon`, `daily[].dt`.
  * `forecastHourly.hours`: `temperature`, `temperatureApparent`, `precipitationChance`, `windSpeed`, `windGust`, `windDirection`, `uvIndex`, `conditionCode`, `forecastStart` mapped into `hourly[].temp`, `hourly[].feels_like`, `hourly[].pop`, `hourly[].wind_speed`, `hourly[].wind_gust`, `hourly[].wind_deg`, `hourly[].uvi`, `hourly[].weather[0].description/icon`, `hourly[].dt`.
  * `weatherAlerts.alerts`: passed through as `alerts`.
* **Usage:** Primary source for weather screens; values are normalized to Fahrenheit and cached.

### OpenWeatherMap OneCall (fallback)
* **Endpoint:** `https://api.openweathermap.org/data/3.0/onecall` with `lat`, `lon`, `appid`, `units`, `lang`, `exclude=minutely`.
* **Data we use:**
  * `current`: `temp`, `feels_like`, `wind_speed`, `wind_gust`, `wind_deg`, `humidity`, `pressure`, `uvi`, `sunrise`, `sunset`, `dt`, `clouds`, plus weather `description/icon` mapping.
  * `daily[]`: `temp.max`, `temp.min`, `sunrise`, `sunset`, `pop`, and weather `description/icon` mapping.
  * `hourly[]`: `dt`, `temp`, `feels_like`, `pop`, `wind_speed`, `wind_gust`, `wind_deg`, `uvi`, and weather `description/icon` mapping.
  * `alerts`: passed through.
* **Usage:** Used when WeatherKit is unavailable; mapped into the same normalized structure.

### RainViewer radar + Google Static Maps
* **Endpoints:**
  * `https://api.rainviewer.com/public/weather-maps.json` → tile metadata (`host`, `radar.past`, `radar.nowcast` frames with `path` and `time`).
  * Tile fetches via the returned `host` and `path` for the configured latitude/longitude.
  * Google Static Maps: `https://maps.googleapis.com/maps/api/staticmap` with `center`, `zoom`, `size`, `maptype`, marker, and `key` parameters for the radar basemap.
* **Data we use:** Radar frame `path` and `time` values to build tile URLs, plus Static Maps imagery for the background.

## Travel time

### Google Maps Directions API
* **Endpoint:** `https://maps.googleapis.com/maps/api/directions/json` (configurable) with `origin`, `destination`, `alternatives=true`, `departure_time=now`, `traffic_model=best_guess`, `region=us`, optional `avoid=highways`, and `key`.
* **Data we use:** From each route’s first `leg`: `duration_in_traffic` or `duration` (`text`/`value`), `summary`, and step `html_instructions`. These are stored as `_duration_text`, `_duration_sec`, `_summary`, and `_steps_text`, then converted to travel time strings and minute counts.

## Sports

### NHL (Blackhawks focus)
* **Endpoints:**
  * Scoreboard/schedule: `https://statsapi.web.nhl.com/api/v1/schedule?date=YYYY-MM-DD&expand=schedule.linescore,schedule.teams` with fallback to `https://api-web.nhle.com/v1/scoreboard/{date}` or `/scoreboard/now`.
  * Club schedule: `https://api-web.nhle.com/v1/club-schedule-season/CHI/20252026` for upcoming game cards.
  * Standings: `https://statsapi.web.nhl.com/api/v1/standings` with fallback to `https://api-web.nhle.com/v1/standings/now`.
* **Data we use:** Game entries (`gamePk/id`, `gameDate`, `gameState`, `teams.home/away` scores/records, `linescore`, `venue`, `startTimeUTC/Central`), plus standings records (`divisionRank`, `leagueRecord.wins/losses/ot`, `gamesBack`, `wildCardGamesBack`, `streak`, split records). DNS checks decide between statsapi and api-web, and payloads are normalized into shared game dictionaries for the NHL screens.

### MLB (Cubs/White Sox)
* **Endpoint:** `https://statsapi.mlb.com/api/v1/schedule` with team IDs for Cubs (`112`) and White Sox (`145`) and expansions for linescores/teams.
* **Data we use:** Game `gamePk`, `gameDate`, `status.abstractGameState`, `teams.home/away` scores/records, `venue`, `probablePitchers`, `linescore` info, plus standings via `https://statsapi.mlb.com/api/v1/standings` (divisions, overall, wild card) for win/loss/GB/WCGB and streak data.

### NBA (Chicago Bulls)
* **Endpoints:**
  * Scoreboard: `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard` for upcoming, live, and recent games within a rolling window.
  * Standings: Prefer `https://cdn.nba.com/static/json/liveData/standings/league.json` with fallback to `https://site.web.api.espn.com/apis/v2/sports/basketball/nba/standings`.
* **Data we use:** Game `gameDate/officialDate`, `status.abstractGameState/detailedState`, team IDs/triCodes, `score`, `competition` data for venue and broadcasts. Standings fields include `wins`, `losses`, `winPct`, `streakText/streak`, conference/division rank and games-back, plus split records (`home`, `away`, `lastTen`).

### NFL (Chicago Bears)
* **Endpoint:** `https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard` for standings and schedule context.
* **Data we use:** `team` IDs, `records` with overall/home/away splits, `playoffSeed`, `streak`, and win/loss counts to power Bears standings displays.

### AHL (Chicago Wolves)
* **Endpoints:**
  * ICS schedule feed (default Wolves Stanza URL in `config.py`) for future games.
  * HockeyTech feeds configured via `AHL_TEAM_ID/AHL_TEAM_TRICODE` for recent results and cached scoring detail.
* **Data we use:** Game dates, opponents, home/away flags, final scores and status for last/next/next-home cards.

## Maps & imagery

* **Google Static Maps**: Used in both the weather radar background and the travel map screen for rendered map tiles (controlled by `GOOGLE_MAPS_API_KEY`).
* **Team/league logos**: Loaded from the repository’s `images/` folder rather than external APIs, but paired with the data above when rendering screens.
