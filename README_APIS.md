# External API Reference

Desk Display pulls live data from third-party weather, maps, travel, finance, and sports providers. Most provider payloads are normalized in `data_fetch.py`, `services/`, or individual screen modules before being rendered.

Most credentials can be supplied in `.env` when `CONFIG_LOAD_DOTENV=1` or through the process/service environment.

---

## Table of contents

- [Diagnostics](#diagnostics)
- [Weather and radar](#weather-and-radar)
- [Travel and map services](#travel-and-map-services)
- [Sports](#sports)
- [Finance](#finance)
- [AHL / Chicago Wolves](#ahl--chicago-wolves)
- [Wi-Fi probe endpoints](#wi-fi-probe-endpoints)
- [Credential quick reference](#credential-quick-reference)
- [Notes on payload handling](#notes-on-payload-handling)

---

## Diagnostics

Use the connectivity script to check provider reachability, credentials, response shape, and key app-level fetch helpers:

```bash
python scripts/test_api_connections.py
```

For automation:

```bash
python scripts/test_api_connections.py --json
```

Statuses:

| Status | Meaning |
| --- | --- |
| `OK` | The check passed. |
| `SKIP` | The check was intentionally skipped, usually because optional credentials are not configured. |
| `FAIL` | Connectivity, authentication, parsing, or payload-shape validation failed. |

Exit codes:

| Exit code | Meaning |
| --- | --- |
| `0` | All checks were `OK` or `SKIP`. |
| `1` | One or more checks returned `FAIL`. |

The diagnostic script covers WeatherKit/OpenWeatherMap, RainViewer, Google Directions/Static Maps, Apple Maps Directions/Snapshot, NHL/NBA/NFL/MLB scoreboards and standings, AHL ICS/HockeyTech, Yahoo Finance chart data, and app-level helpers for Bears, Bulls, Blackhawks, Cubs, Sox, and Wolves.

---

## Weather and radar

### Apple WeatherKit

| Item | Value |
| --- | --- |
| Role | Primary weather provider when configured. |
| Endpoint | `https://weatherkit.apple.com/api/v1/weather/{language}/{lat}/{lon}` |
| Config | `WEATHERKIT_TEAM_ID`, `WEATHERKIT_KEY_ID`, `WEATHERKIT_SERVICE_ID`, `WEATHERKIT_PRIVATE_KEY` or `WEATHERKIT_KEY_PATH`, optional `WEATHERKIT_LANGUAGE`, optional `WEATHERKIT_TIMEZONE`. |
| Location | `WEATHER_LATITUDE`, `WEATHER_LONGITUDE`. |

Data sets requested/used:

- `currentWeather`
- `forecastDaily`
- `forecastHourly`
- `weatherAlerts`

Fields used include:

- Current: `temperature`, `temperatureApparent`, `windSpeed`, `windGust`, `windDirection`, `humidity`, `pressure`, `uvIndex`, `cloudCover`, `asOf`, `conditionCode`, `sunrise`, `sunset`.
- Daily: `temperatureMax`, `temperatureMin`, `sunrise`, `sunset`, `precipitationChance`, `conditionCode`, `forecastStart`.
- Hourly: `temperature`, `temperatureApparent`, `precipitationChance`, `windSpeed`, `windGust`, `windDirection`, `uvIndex`, `conditionCode`, `forecastStart`.
- Alerts: alert payloads are passed through for display.

Important signing note: `WEATHERKIT_KEY_ID` is only the Apple key identifier. The app still needs the private key material via `WEATHERKIT_PRIVATE_KEY` or the downloaded `.p8` path in `WEATHERKIT_KEY_PATH`.

### OpenWeatherMap One Call

| Item | Value |
| --- | --- |
| Role | Fallback weather provider when WeatherKit is unavailable or unconfigured. |
| Endpoint | `https://api.openweathermap.org/data/3.0/onecall` |
| Config | `OWM_API_KEY`, optional `OWM_UNITS`, optional `OWM_LANGUAGE`. |
| Location | `WEATHER_LATITUDE`, `WEATHER_LONGITUDE`. |

Fields used include:

- `current.temp`, `current.feels_like`, `current.wind_speed`, `current.wind_gust`, `current.wind_deg`, `current.humidity`, `current.pressure`, `current.uvi`, `current.sunrise`, `current.sunset`, `current.dt`, `current.clouds`, `current.weather[].description`, `current.weather[].icon`.
- `daily[].temp.max`, `daily[].temp.min`, `daily[].sunrise`, `daily[].sunset`, `daily[].pop`, `daily[].weather[].description`, `daily[].weather[].icon`.
- `hourly[].dt`, `hourly[].temp`, `hourly[].feels_like`, `hourly[].pop`, `hourly[].wind_speed`, `hourly[].wind_gust`, `hourly[].wind_deg`, `hourly[].uvi`, `hourly[].weather[].description`, `hourly[].weather[].icon`.
- `alerts` payloads are passed through for display.

OpenWeatherMap data is normalized toward the same internal shape as WeatherKit data.

### RainViewer radar

| Item | Value |
| --- | --- |
| Role | Radar frame metadata and radar tiles. |
| Metadata endpoint, primary | `https://api.rainviewer.com/public/weather-maps.json` |
| Metadata endpoint, fallback | `https://api.rainviewer.com/public/maps.json` |
| Tile endpoint pattern | `https://{host}/{path}/256/{zoom}/{x}/{y}/2/1_1.png` |

Fields used:

- `host`
- `radar.past[]`
- `radar.nowcast[]`
- frame `path`
- frame `time`

### Additional weather map tiles

Weather radar/map rendering may also use:

| Provider | Endpoint pattern | Purpose |
| --- | --- | --- |
| Google Static Maps | `https://maps.googleapis.com/maps/api/staticmap` | Optional radar background/basemap when `GOOGLE_MAPS_API_KEY` is available. |
| Iowa State Mesonet | `https://mesonet.agron.iastate.edu/cache/tile.py/1.0.0/q2-hsr-900913/{zoom}/{x}/{y}.png` | Radar tile fallback/source used by weather map rendering. |
| OpenStreetMap | `https://tile.openstreetmap.org/{zoom}/{x}/{y}.png` | Basemap tile fallback. |
| CARTO | `https://basemaps.cartocdn.com/light_all/{zoom}/{x}/{y}.png` | Basemap tile fallback. |

---

## Travel and map services

### Google Directions

| Item | Value |
| --- | --- |
| Role | Travel route estimates. |
| Endpoint | `https://maps.googleapis.com/maps/api/directions/json` |
| Config | `GOOGLE_MAPS_API_KEY`, `TRAVEL_MODE`, `TRAVEL_TO_HOME_ORIGIN`, `TRAVEL_TO_HOME_DESTINATION`, `TRAVEL_TO_WORK_ORIGIN`, `TRAVEL_TO_WORK_DESTINATION`. |

Fields used:

- `routes[].summary`
- `routes[].legs[].duration`
- `routes[].legs[].duration_in_traffic`
- `routes[].legs[].steps[].html_instructions`

Route filtering is performed in-app after normalizing summary and step text.

### Google Static Maps

| Item | Value |
| --- | --- |
| Endpoint | `https://maps.googleapis.com/maps/api/staticmap` |
| Config | `GOOGLE_MAPS_API_KEY`, `WEATHER_LATITUDE`, `WEATHER_LONGITUDE`. |
| Used parameters | `center`, `zoom`, `size`, `maptype`, `key`. |

### Apple Maps Directions

| Item | Value |
| --- | --- |
| Role | Alternative travel route provider. |
| Endpoint | `https://maps-api.apple.com/v1/directions` by default; override with `APPLE_MAPS_DIRECTIONS_URL`. |
| Config | `APPLE_MAPS_API_KEY` or `MAPKIT_TOKEN`; JWT signing via `APPLE_MAPS_TEAM_ID`, `APPLE_MAPS_KEY_ID`, and `APPLE_MAPS_PRIVATE_KEY` or `APPLE_MAPS_KEY_PATH`. WeatherKit signing values can be reused as fallback where supported. |

Fields used:

- route time fields such as `expectedTravelTime`, `staticTravelTime`, and `typicalTravelTime`,
- route summary/name fields,
- step/polyline data when present.

### Apple Maps Snapshot

| Item | Value |
| --- | --- |
| Role | Binary map image payloads. |
| Endpoint | `https://maps-api.apple.com/v1/snapshot` by default; override with `APPLE_MAPS_SNAPSHOT_URL`. |
| Config | Same Apple Maps auth values as directions. |

---

## Sports

### NHL / Blackhawks / league

| Use | Endpoint |
| --- | --- |
| Blackhawks configured season schedule | `https://api-web.nhle.com/v1/club-schedule-season/CHI/20252026` |
| Team month schedule | `https://api-web.nhle.com/v1/club-schedule/{tric}/month/now` |
| Team season schedule | `https://api-web.nhle.com/v1/club-schedule-season/{tric}/now` |
| Game landing | `https://api-web.nhle.com/v1/gamecenter/{gid}/landing` |
| Game boxscore | `https://api-web.nhle.com/v1/gamecenter/{gid}/boxscore` |
| League scoreboard by date | `https://api-web.nhle.com/v1/scoreboard/{date}` |
| League scoreboard now | `https://api-web.nhle.com/v1/scoreboard/now` |
| Standings, current API | `https://api-web.nhle.com/v1/standings/now` |
| Legacy schedule fallback | `https://statsapi.web.nhl.com/api/v1/schedule` |
| Legacy live feed fallback | `https://statsapi.web.nhl.com/api/v1/game/{gamePk}/feed/live` |
| Legacy standings fallback | `https://statsapi.web.nhl.com/api/v1/standings` |
| ESPN standings fallback | `https://site.web.api.espn.com/apis/v2/sports/hockey/nhl/standings` |

Fields used include game IDs, dates, game state/status, team records, team scores, linescore info, venue, start times, conference/division ranks, games back, wild-card games back, streaks, and split records.

### NHL playoffs

| Use | Endpoint |
| --- | --- |
| Playoff series carousel by season | `https://api-web.nhle.com/v1/playoff-series/carousel/{season}` |
| Playoff bracket by season | `https://api-web.nhle.com/v1/playoff-bracket/{season}` |
| Current playoff series carousel | `https://api-web.nhle.com/v1/playoff-series/carousel/now` |
| Current playoff bracket | `https://api-web.nhle.com/v1/playoff-bracket/now` |
| Current schedule fallback | `https://api-web.nhle.com/v1/schedule/now` |
| Schedule by date fallback | `https://api-web.nhle.com/v1/schedule/{date}` |

### MLB / Cubs / White Sox / league

| Use | Endpoint |
| --- | --- |
| Schedule, team screens, scoreboard | `https://statsapi.mlb.com/api/v1/schedule` |
| Standings | `https://statsapi.mlb.com/api/v1/standings` |
| Probable-pitcher headshots | `https://img.mlbstatic.com/mlb-photos/image/upload/w_120,q_auto:best/v1/people/{pitcher_id}/headshot/67/current` |

Team IDs used by app helpers include `112` for the Chicago Cubs and `145` for the Chicago White Sox.

Fields used include game IDs, game dates, game status, team records, scores, venue, probable pitchers, linescore data, division standings, wins, losses, games back, wild-card games back, and streaks.

### NBA / Bulls / league

| Use | Endpoint |
| --- | --- |
| NBA.com live scoreboard | `https://cdn.nba.com/static/json/liveData/scoreboard` |
| NBA.com scoreboard fallback | `https://nba-prod-us-east-1-media.s3.amazonaws.com/json/liveData/scoreboard` |
| ESPN scoreboard fallback | `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard` |
| NBA.com standings | `https://cdn.nba.com/static/json/liveData/standings/league.json` |
| ESPN standings fallback | `https://site.web.api.espn.com/apis/v2/sports/basketball/nba/standings` |

Fields used include game status, clock/period, team IDs, team tricode values, scores, venue, broadcast info, wins, losses, win percentage, streaks, conference/division ranks, and split records.

### NBA playoffs

| Use | Endpoint |
| --- | --- |
| NBA.com playoff bracket | `https://cdn.nba.com/static/json/liveData/playoffbracket/playoffbracket_00.json` |
| NBA S3 playoff bracket fallback | `https://nba-prod-us-east-1-media.s3.amazonaws.com/json/liveData/playoffbracket/playoffbracket_00.json` |
| NBA.com bracket fallback | `https://cdn.nba.com/static/json/liveData/bracket/bracket_00.json` |
| NBA S3 bracket fallback | `https://nba-prod-us-east-1-media.s3.amazonaws.com/json/liveData/bracket/bracket_00.json` |

### NFL / Bears / league

| Use | Endpoint |
| --- | --- |
| ESPN NFL scoreboard | `https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard` |
| NFL standings CSV | `https://raw.githubusercontent.com/nflverse/nfldata/master/data/standings.csv` |

Fields used include game status, team records, scores, standings ranks, wins, losses, ties, streaks, and split records.

### NCAAM

| Use | Endpoint |
| --- | --- |
| ESPN men's college basketball scoreboard | `https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard` |

`NCAAM_SCOREBOARD_MODE` controls display filtering; the default is `top25`.

### FIFA World Cup

| Use | Endpoint |
| --- | --- |
| ESPN FIFA World Cup scoreboard | `https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard` |

Fields used include game status, team logos, start times, scores, and final/in-progress state.

---

## Finance

### Yahoo Finance / yfinance

| Use | Endpoint/source |
| --- | --- |
| Stock helpers | `yfinance` Python package. |
| Connectivity diagnostic | `https://query1.finance.yahoo.com/v8/finance/chart/SPY` |

Fields used include `regularMarketPrice`, `previousClose`, and historical close values when available. The diagnostics check chart payload availability with `range=1d` and `interval=1d`.

---

## AHL / Chicago Wolves

| Use | Endpoint/config |
| --- | --- |
| Schedule calendar | `AHL_SCHEDULE_ICS_URL`; defaults to the bundled Stanza Chicago Wolves webcal URL converted to HTTPS as needed. |
| HockeyTech feed base | `AHL_API_BASE_URL`; default family is `https://lscluster.hockeytech.com/feed/` / `https://lscluster.hockeytech.com/feed/index.php`. |
| HockeyTech params | `AHL_API_KEY`, `AHL_CLIENT_CODE`, `AHL_LEAGUE_ID`, `AHL_SITE_ID`, `AHL_SEASON_ID`, `AHL_TEAM_ID`, `AHL_TEAM_TRICODE`, `AHL_TEAM_NAME`. |

Fields used include game dates, opponent, home/away flags, final scores, recent scoring details, and schedule metadata. Responses are cached by helpers where useful for redraw performance.

---

## Wi-Fi probe endpoints

Wi-Fi utilities can probe configured HTTPS/TCP targets to decide whether recovery should run.

| Variable | Purpose |
| --- | --- |
| `WIFI_TCP_PROBE_HOST` | Single TCP host probe. |
| `WIFI_TCP_PROBE_HOSTS` | Multiple TCP host probes. |
| `WIFI_TCP_PROBE_PORT` | TCP probe port. |
| `WIFI_TCP_PROBE_URL` | Single URL probe. |
| `WIFI_TCP_PROBE_URLS` | Multiple URL probes. |
| `WIFI_HTTPS_PROBE_URL` | HTTPS probe URL. |

---

## Credential quick reference

| Feature | Required/important environment variables |
| --- | --- |
| WeatherKit | `WEATHERKIT_TEAM_ID`, `WEATHERKIT_KEY_ID`, `WEATHERKIT_SERVICE_ID`, `WEATHERKIT_PRIVATE_KEY` or `WEATHERKIT_KEY_PATH`. |
| OpenWeatherMap | `OWM_API_KEY`. |
| Weather/map location | `WEATHER_LATITUDE`, `WEATHER_LONGITUDE`. |
| Google Directions/Static Maps | `GOOGLE_MAPS_API_KEY`, travel origin/destination variables. |
| Apple Maps | `APPLE_MAPS_API_KEY` or `MAPKIT_TOKEN`, or JWT values: `APPLE_MAPS_TEAM_ID`, `APPLE_MAPS_KEY_ID`, `APPLE_MAPS_PRIVATE_KEY` or `APPLE_MAPS_KEY_PATH`. |
| Travel routes | `TRAVEL_MODE`, `TRAVEL_TO_HOME_ORIGIN`, `TRAVEL_TO_HOME_DESTINATION`, `TRAVEL_TO_WORK_ORIGIN`, `TRAVEL_TO_WORK_DESTINATION`. |
| AHL/Wolves | Optional `AHL_*` overrides; defaults are provided for the Chicago Wolves helper path. |

---

## Notes on payload handling

- External APIs may change without notice; run `python scripts/test_api_connections.py` after upgrades or when a screen goes blank.
- Missing optional credentials usually produce diagnostic `SKIP` results rather than failures.
- Weather providers are normalized to a shared internal structure so renderers can prefer WeatherKit while still using OpenWeatherMap fallback data.
- Team and league logos are loaded primarily from the local `images/` folder. Some sports providers may include remote logos, but the renderers generally prefer bundled assets for predictable offline rendering.
- The app uses local caching/normalization in several helpers to reduce redraw latency and isolate screens from provider-specific response shapes.
