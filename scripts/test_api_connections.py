#!/usr/bin/env python3
"""Test external API connectivity used by desk_display.

This script performs lightweight probes against every external API family listed
in README_APIS.md. It is safe to run locally and in CI:

- Authenticated APIs are skipped when credentials are not configured.
- Endpoints that are expected to reject empty params (e.g., HockeyTech base)
  are treated as reachable when they return any non-5xx HTTP status.

Exit code is non-zero only when one or more checks fail.
"""

from __future__ import annotations

import datetime as dt
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional

import requests

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("CONFIG_LOAD_DOTENV", "1")

from config import (
    AHL_SCHEDULE_ICS_URL,
    APPLE_MAPS_API_KEY,
    APPLE_MAPS_DIRECTIONS_URL,
    APPLE_MAPS_SNAPSHOT_URL,
    GOOGLE_MAPS_API_KEY,
    OWM_API_KEY,
    OWM_API_URL,
    TRAVEL_DESTINATION,
    TRAVEL_ORIGIN,
    LATITUDE,
    LONGITUDE,
)
from data_fetch import fetch_weather
from screens.draw_weather import _fetch_rainviewer_frames
from services.apple_maps import fetch_apple_maps_routes, fetch_apple_maps_snapshot
from utils import fetch_directions_routes


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str
    duration_ms: float


@dataclass
class Check:
    name: str
    func: Callable[[], tuple[str, str]]


def _run_check(check: Check) -> CheckResult:
    start = time.perf_counter()
    try:
        status, detail = check.func()
    except Exception as exc:  # pragma: no cover - diagnostic script
        status, detail = "fail", f"{type(exc).__name__}: {exc}"
    elapsed = round((time.perf_counter() - start) * 1000, 2)
    return CheckResult(check.name, status, detail, elapsed)


def _ok(detail: str) -> tuple[str, str]:
    return "ok", detail


def _skip(detail: str) -> tuple[str, str]:
    return "skip", detail


def _fail(detail: str) -> tuple[str, str]:
    return "fail", detail


def _http_json(url: str, *, params: Optional[dict] = None, timeout: int = 12) -> tuple[int, object]:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.status_code, response.json()


def check_weatherkit_or_owm() -> tuple[str, str]:
    payload = fetch_weather()
    if payload and isinstance(payload, dict) and payload.get("current"):
        source = payload.get("_source", "weather")
        return _ok(f"weather data returned (source={source})")
    return _fail("fetch_weather returned empty payload")


def check_openweathermap_direct() -> tuple[str, str]:
    if not OWM_API_KEY:
        return _skip("OWM_API_KEY not configured")
    params = {
        "lat": LATITUDE,
        "lon": LONGITUDE,
        "appid": OWM_API_KEY,
        "units": "imperial",
    }
    status, payload = _http_json(OWM_API_URL, params=params)
    if isinstance(payload, dict) and payload.get("current"):
        return _ok(f"HTTP {status}")
    return _fail(f"HTTP {status} but missing current payload")


def check_rainviewer_metadata() -> tuple[str, str]:
    frames = _fetch_rainviewer_frames(zoom=7, max_frames=2)
    if frames:
        return _ok(f"returned {len(frames)} frame(s)")
    return _fail("no radar frames returned")


def check_google_directions() -> tuple[str, str]:
    if not GOOGLE_MAPS_API_KEY:
        return _skip("GOOGLE_MAPS_API_KEY not configured")
    routes = fetch_directions_routes(
        TRAVEL_ORIGIN,
        TRAVEL_DESTINATION,
        GOOGLE_MAPS_API_KEY,
        url="https://maps.googleapis.com/maps/api/directions/json",
    )
    if routes:
        return _ok(f"returned {len(routes)} route(s)")
    return _fail("no routes returned")


def check_google_static_maps() -> tuple[str, str]:
    if not GOOGLE_MAPS_API_KEY:
        return _skip("GOOGLE_MAPS_API_KEY not configured")
    params = {
        "center": f"{LATITUDE},{LONGITUDE}",
        "zoom": "7",
        "size": "64x64",
        "maptype": "roadmap",
        "key": GOOGLE_MAPS_API_KEY,
    }
    response = requests.get("https://maps.googleapis.com/maps/api/staticmap", params=params, timeout=12)
    if response.ok and response.content:
        return _ok(f"HTTP {response.status_code}, bytes={len(response.content)}")
    return _fail(f"HTTP {response.status_code}")


def check_apple_directions() -> tuple[str, str]:
    routes = fetch_apple_maps_routes(
        TRAVEL_ORIGIN,
        TRAVEL_DESTINATION,
        APPLE_MAPS_API_KEY or "",
        url=APPLE_MAPS_DIRECTIONS_URL,
    )
    if routes:
        return _ok(f"returned {len(routes)} route(s)")
    if not APPLE_MAPS_API_KEY:
        return _skip("APPLE_MAPS_API_KEY / MAPKIT_TOKEN not configured (JWT fallback may also be unset)")
    return _fail("no routes returned")


def check_apple_snapshot() -> tuple[str, str]:
    content = fetch_apple_maps_snapshot(
        center=(LATITUDE, LONGITUDE),
        zoom=7,
        size=(64, 64),
        api_key=APPLE_MAPS_API_KEY or "",
        url=APPLE_MAPS_SNAPSHOT_URL,
    )
    if content:
        return _ok(f"bytes={len(content)}")
    if not APPLE_MAPS_API_KEY:
        return _skip("APPLE_MAPS_API_KEY / MAPKIT_TOKEN not configured (JWT fallback may also be unset)")
    return _fail("empty snapshot response")


def check_nhl_scoreboard() -> tuple[str, str]:
    day = dt.datetime.now().strftime("%Y-%m-%d")
    status, payload = _http_json(f"https://api-web.nhle.com/v1/scoreboard/{day}")
    if isinstance(payload, dict):
        return _ok(f"HTTP {status}")
    return _fail("unexpected payload shape")


def check_nhl_standings() -> tuple[str, str]:
    status, payload = _http_json("https://api-web.nhle.com/v1/standings/now")
    if isinstance(payload, dict) and (payload.get("standings") or payload.get("wildCardIndicator")):
        return _ok(f"HTTP {status}")
    return _fail("unexpected standings payload")


def check_mlb_schedule() -> tuple[str, str]:
    today = dt.datetime.now().strftime("%Y-%m-%d")
    status, payload = _http_json(
        "https://statsapi.mlb.com/api/v1/schedule",
        params={"sportId": 1, "date": today, "teamId": 112},
    )
    if isinstance(payload, dict) and "dates" in payload:
        return _ok(f"HTTP {status}")
    return _fail("unexpected MLB schedule payload")


def check_mlb_standings() -> tuple[str, str]:
    status, payload = _http_json(
        "https://statsapi.mlb.com/api/v1/standings",
        params={"leagueId": "103,104", "season": dt.datetime.now().year},
    )
    if isinstance(payload, dict) and "records" in payload:
        return _ok(f"HTTP {status}")
    return _fail("unexpected MLB standings payload")


def check_nba_scoreboard() -> tuple[str, str]:
    status, payload = _http_json("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard")
    if isinstance(payload, dict) and "events" in payload:
        return _ok(f"HTTP {status}")
    return _fail("unexpected NBA scoreboard payload")


def check_nba_standings() -> tuple[str, str]:
    status, payload = _http_json("https://cdn.nba.com/static/json/liveData/standings/league.json")
    if isinstance(payload, dict) and payload.get("standings"):
        return _ok(f"HTTP {status}")
    return _fail("unexpected NBA standings payload")


def check_nfl_scoreboard() -> tuple[str, str]:
    status, payload = _http_json("https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard")
    if isinstance(payload, dict) and "events" in payload:
        return _ok(f"HTTP {status}")
    return _fail("unexpected NFL scoreboard payload")


def check_nfl_standings_csv() -> tuple[str, str]:
    response = requests.get(
        "https://raw.githubusercontent.com/nflverse/nfldata/master/data/standings.csv",
        timeout=12,
    )
    response.raise_for_status()
    text = response.text
    if "season" in text and "team" in text:
        return _ok(f"HTTP {response.status_code}, rows~{max(0, text.count(chr(10)) - 1)}")
    return _fail("CSV did not contain expected headers")


def check_ahl_ics() -> tuple[str, str]:
    response = requests.get(AHL_SCHEDULE_ICS_URL, timeout=12)
    response.raise_for_status()
    text = response.text
    if "BEGIN:VCALENDAR" in text:
        return _ok(f"HTTP {response.status_code}")
    return _fail("ICS payload missing calendar header")


def check_hockeytech_feed_base() -> tuple[str, str]:
    response = requests.get("https://lscluster.hockeytech.com/feed/", timeout=12)
    if response.status_code < 500:
        return _ok(f"reachable (HTTP {response.status_code})")
    return _fail(f"unreachable (HTTP {response.status_code})")


def check_yahoo_chart_api() -> tuple[str, str]:
    status, payload = _http_json("https://query1.finance.yahoo.com/v8/finance/chart/SPY", params={"range": "1d", "interval": "1d"})
    chart = payload.get("chart") if isinstance(payload, dict) else None
    if isinstance(chart, dict) and chart.get("result"):
        return _ok(f"HTTP {status}")
    return _fail("unexpected Yahoo chart payload")


CHECKS = [
    Check("weather (weatherkit/owm via app helper)", check_weatherkit_or_owm),
    Check("openweathermap onecall", check_openweathermap_direct),
    Check("rainviewer metadata/tiles", check_rainviewer_metadata),
    Check("google directions", check_google_directions),
    Check("google static maps", check_google_static_maps),
    Check("apple maps directions", check_apple_directions),
    Check("apple maps snapshot", check_apple_snapshot),
    Check("nhl scoreboard", check_nhl_scoreboard),
    Check("nhl standings", check_nhl_standings),
    Check("mlb schedule", check_mlb_schedule),
    Check("mlb standings", check_mlb_standings),
    Check("nba scoreboard", check_nba_scoreboard),
    Check("nba standings", check_nba_standings),
    Check("nfl scoreboard", check_nfl_scoreboard),
    Check("nfl standings csv", check_nfl_standings_csv),
    Check("ahl schedule ics", check_ahl_ics),
    Check("ahl hockeytech feed base", check_hockeytech_feed_base),
    Check("yahoo finance chart api", check_yahoo_chart_api),
]


def main() -> int:
    results = [_run_check(check) for check in CHECKS]

    width = max(len(r.name) for r in results) + 2
    for res in results:
        print(f"{res.name:<{width}} {res.status.upper():<5} {res.duration_ms:>8.2f} ms  {res.detail}")

    failed = [r for r in results if r.status == "fail"]
    skipped = [r for r in results if r.status == "skip"]
    print("\nSummary:")
    print(f"  total={len(results)} ok={len(results)-len(failed)-len(skipped)} skip={len(skipped)} fail={len(failed)}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
