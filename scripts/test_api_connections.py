#!/usr/bin/env python3
"""Test external API connectivity used by desk_display.

This script performs lightweight probes against every external API family listed
in README_APIS.md. It is safe to run locally and in CI:

- Authenticated APIs are skipped when credentials are not configured.
- Endpoints that are expected to reject empty params (e.g., HockeyTech base)
  are treated as reachable when they return any non-5xx HTTP status.

Exit code behavior:
- 0: all checks are OK or SKIP
- 1: one or more checks FAIL
"""

from __future__ import annotations

import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    # Only re-exec when this script is run directly (`python3
    # scripts/test_api_connections.py`), not when it's imported under
    # another name (e.g. tests loading it via
    # importlib.util.spec_from_file_location) -- see _venv_bootstrap.py.
    try:
        from scripts._venv_bootstrap import reexec_with_project_venv
    except ImportError:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from _venv_bootstrap import reexec_with_project_venv
    reexec_with_project_venv()

import argparse
import datetime as dt
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests

sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("CONFIG_LOAD_DOTENV", "1")

from config import (
    AHL_SCHEDULE_ICS_URL,
    LATITUDE,
    LONGITUDE,
    OWM_API_KEY,
    OWM_API_URL,
)
from data_fetch import (
    fetch_bears_standings,
    fetch_blackhawks_last_game,
    fetch_blackhawks_live_game,
    fetch_blackhawks_next_game,
    fetch_blackhawks_next_home_game,
    fetch_blackhawks_standings,
    fetch_bulls_last_game,
    fetch_bulls_live_game,
    fetch_bulls_next_game,
    fetch_bulls_next_home_game,
    fetch_bulls_standings,
    fetch_cubs_games,
    fetch_cubs_standings,
    fetch_sox_games,
    fetch_sox_standings,
    fetch_weather,
    fetch_wolves_games,
)
from screens.draw_weather import _alert_message_text, _fetch_rainviewer_frames, _selected_alert
from screens.nhl_scoreboard import dns_diagnostics
from services.news_feeds import (
    fetch_topic_headlines,
    load_news_feed_config,
    load_news_feed_config_2,
)


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


def _has_data(payload: Any) -> bool:
    if payload is None:
        return False
    if isinstance(payload, (list, tuple, set, dict)):
        return len(payload) > 0
    return True


def _http_json(url: str, *, params: Optional[dict] = None, timeout: int = 12) -> tuple[int, object]:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.status_code, response.json()


def check_weatherkit_or_owm() -> tuple[str, str]:
    payload = fetch_weather()
    if payload and isinstance(payload, dict) and payload.get("current"):
        source = payload.get("_source") or payload.get("source") or "weather"
        return _ok(f"weather data returned (source={source})")
    return _fail("fetch_weather returned empty payload")


def check_weather_alerts() -> tuple[str, str]:
    payload = fetch_weather()
    if not isinstance(payload, dict) or not payload.get("current"):
        return _fail("fetch_weather returned empty payload")

    raw_alerts = payload.get("alerts")
    if isinstance(raw_alerts, dict):
        raw_count = len(raw_alerts.get("alerts") or []) if isinstance(raw_alerts.get("alerts"), list) else 1
    elif isinstance(raw_alerts, list):
        raw_count = len(raw_alerts)
    elif raw_alerts:
        raw_count = 1
    else:
        raw_count = 0

    severity, alert = _selected_alert(payload)
    source = payload.get("_source") or payload.get("source") or "weather"
    if alert is None:
        return _ok(f"no active weather alerts reported (source={source}, alerts={raw_count})")

    message = _alert_message_text(alert) or "active alert"
    compact_message = " ".join(message.split())
    if len(compact_message) > 160:
        compact_message = f"{compact_message[:157].rstrip()}..."
    return _ok(
        f"active {severity or 'weather'} alert reported "
        f"(source={source}, alerts={raw_count}): {compact_message}"
    )


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


def _check_app_fetch(name: str, func: Callable[[], Any], *, expect_data: bool) -> tuple[str, str]:
    payload = func()
    has_data = _has_data(payload)
    if has_data:
        return _ok(f"{name} returned data")
    if expect_data:
        return _fail(f"{name} returned no data")
    return _ok(f"{name} returned no data (allowed)")


def check_nhl_next_game_helper() -> tuple[str, str]:
    return _check_app_fetch("fetch_blackhawks_next_game", fetch_blackhawks_next_game, expect_data=False)


def check_nhl_next_home_game_helper() -> tuple[str, str]:
    return _check_app_fetch("fetch_blackhawks_next_home_game", fetch_blackhawks_next_home_game, expect_data=False)


def check_nhl_last_game_helper() -> tuple[str, str]:
    return _check_app_fetch("fetch_blackhawks_last_game", fetch_blackhawks_last_game, expect_data=False)


def check_nhl_live_game_helper() -> tuple[str, str]:
    return _check_app_fetch("fetch_blackhawks_live_game", fetch_blackhawks_live_game, expect_data=False)


def check_nhl_standings_helper() -> tuple[str, str]:
    return _check_app_fetch("fetch_blackhawks_standings", fetch_blackhawks_standings, expect_data=True)


def check_nba_next_game_helper() -> tuple[str, str]:
    return _check_app_fetch("fetch_bulls_next_game", fetch_bulls_next_game, expect_data=False)


def check_nba_next_home_game_helper() -> tuple[str, str]:
    return _check_app_fetch("fetch_bulls_next_home_game", fetch_bulls_next_home_game, expect_data=False)


def check_nba_last_game_helper() -> tuple[str, str]:
    return _check_app_fetch("fetch_bulls_last_game", fetch_bulls_last_game, expect_data=False)


def check_nba_live_game_helper() -> tuple[str, str]:
    return _check_app_fetch("fetch_bulls_live_game", fetch_bulls_live_game, expect_data=False)


def check_nba_standings_helper() -> tuple[str, str]:
    return _check_app_fetch("fetch_bulls_standings", fetch_bulls_standings, expect_data=True)


def check_nfl_standings_helper() -> tuple[str, str]:
    return _check_app_fetch("fetch_bears_standings", fetch_bears_standings, expect_data=True)


def check_mlb_cubs_games_helper() -> tuple[str, str]:
    return _check_app_fetch("fetch_cubs_games", fetch_cubs_games, expect_data=False)


def check_mlb_sox_games_helper() -> tuple[str, str]:
    return _check_app_fetch("fetch_sox_games", fetch_sox_games, expect_data=False)


def check_mlb_cubs_standings_helper() -> tuple[str, str]:
    return _check_app_fetch("fetch_cubs_standings", fetch_cubs_standings, expect_data=True)


def check_mlb_sox_standings_helper() -> tuple[str, str]:
    return _check_app_fetch("fetch_sox_standings", fetch_sox_standings, expect_data=True)


def check_ahl_wolves_games_helper() -> tuple[str, str]:
    return _check_app_fetch("fetch_wolves_games", fetch_wolves_games, expect_data=False)


def check_nhl_network_diagnostics() -> tuple[str, str]:
    report = dns_diagnostics()
    host_errors = [h for h in report.get("hosts", []) if h.get("status") == "error"]
    http_errors = [h for h in report.get("http_checks", []) if h.get("status") == "error"]
    if host_errors or http_errors:
        return _fail(f"dns/http errors: hosts={len(host_errors)} http={len(http_errors)}")
    return _ok("dns + endpoint diagnostics passed")


def _check_news_feeds_config(config_name: str, topics, headline_count: int) -> tuple[str, str]:
    if not topics:
        return _skip(f"no topics configured in {config_name}")

    failures: list[str] = []
    counts: list[str] = []
    for topic in topics:
        headlines = fetch_topic_headlines(topic, headline_count, timeout=10.0)
        if not headlines:
            failures.append(topic.id)
        else:
            counts.append(f"{topic.id}={len(headlines)}")

    if failures:
        return _fail(f"no headlines returned for: {', '.join(failures)} (ok: {', '.join(counts)})")
    return _ok(f"headlines returned for all topics ({', '.join(counts)})")


def check_news_feeds() -> tuple[str, str]:
    """Fetch every topic feed listed in news_feeds.json and report per-topic results.

    Feeds/topics are edited in news_feeds.json, not here — see README_APIS.md.
    """

    topics, headline_count, _refresh_minutes = load_news_feed_config()
    return _check_news_feeds_config("news_feeds.json", topics, headline_count)


def check_news_feeds_2() -> tuple[str, str]:
    """Fetch every topic feed listed in news_feeds_2.json and report per-topic results.

    Feeds/topics are edited in news_feeds_2.json, not here — see README_APIS.md.
    """

    topics, headline_count, _refresh_minutes = load_news_feed_config_2()
    return _check_news_feeds_config("news_feeds_2.json", topics, headline_count)


CHECKS = [
    Check("weather (weatherkit/owm via app helper)", check_weatherkit_or_owm),
    Check("weather alerts", check_weather_alerts),
    Check("openweathermap onecall", check_openweathermap_direct),
    Check("rainviewer metadata/tiles", check_rainviewer_metadata),
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
    Check("app helper nhl next game", check_nhl_next_game_helper),
    Check("app helper nhl next home game", check_nhl_next_home_game_helper),
    Check("app helper nhl last game", check_nhl_last_game_helper),
    Check("app helper nhl live game", check_nhl_live_game_helper),
    Check("app helper nhl standings", check_nhl_standings_helper),
    Check("app helper nba next game", check_nba_next_game_helper),
    Check("app helper nba next home game", check_nba_next_home_game_helper),
    Check("app helper nba last game", check_nba_last_game_helper),
    Check("app helper nba live game", check_nba_live_game_helper),
    Check("app helper nba standings", check_nba_standings_helper),
    Check("app helper nfl standings", check_nfl_standings_helper),
    Check("app helper mlb cubs games", check_mlb_cubs_games_helper),
    Check("app helper mlb sox games", check_mlb_sox_games_helper),
    Check("app helper mlb cubs standings", check_mlb_cubs_standings_helper),
    Check("app helper mlb sox standings", check_mlb_sox_standings_helper),
    Check("app helper ahl wolves games", check_ahl_wolves_games_helper),
    Check("nhl network diagnostics", check_nhl_network_diagnostics),
    Check("news headlines feeds", check_news_feeds),
    Check("news headlines 2 feeds", check_news_feeds_2),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_output", action="store_true", help="Emit JSON output.")
    args = parser.parse_args()

    results = [_run_check(check) for check in CHECKS]

    failed = [r for r in results if r.status == "fail"]
    skipped = [r for r in results if r.status == "skip"]
    ok_count = len(results) - len(failed) - len(skipped)

    if args.json_output:
        print(
            json.dumps(
                {
                    "summary": {
                        "total": len(results),
                        "ok": ok_count,
                        "skip": len(skipped),
                        "fail": len(failed),
                    },
                    "checks": [r.__dict__ for r in results],
                },
                indent=2,
            )
        )
    else:
        width = max(len(r.name) for r in results) + 2
        for res in results:
            print(f"{res.name:<{width}} {res.status.upper():<5} {res.duration_ms:>8.2f} ms  {res.detail}")
        print("\nSummary:")
        print(f"  total={len(results)} ok={ok_count} skip={len(skipped)} fail={len(failed)}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
