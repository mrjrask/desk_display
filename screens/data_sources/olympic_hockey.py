#!/usr/bin/env python3
"""Resilient Olympic hockey providers with normalized game output."""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional
from zoneinfo import ZoneInfo

from requests import HTTPError

from services.http_client import get_session

SESSION = get_session()
REQUEST_TIMEOUT = 12
CACHE_TTL_SECONDS = 30
RATE_LIMIT_SECONDS = 2
DEFAULT_TIMEZONE = os.getenv("OLYMPIC_HOCKEY_TIMEZONE", "America/Chicago")

LEAGUE_KEYS = {
    "men": "olympic_mhockey",
    "women": "olympic_whockey",
}
ESPN_URL_CANDIDATES = {
    "men": (
        "https://site.api.espn.com/apis/site/v2/sports/hockey/mens-olympics/scoreboard",
        "https://site.api.espn.com/apis/site/v2/sports/hockey/olympics/scoreboard",
    ),
    "women": (
        "https://site.api.espn.com/apis/site/v2/sports/hockey/womens-olympics/scoreboard",
        "https://site.api.espn.com/apis/site/v2/sports/hockey/olympics/scoreboard",
    ),
}
ESPN_RESULTS_PAGE_URL = "https://www.espn.com/olympics/winter/2026/results/_/discipline/29"
IIHF_URLS = {
    "men": "https://www.iihf.com/en/events/2026/olympics-m/schedule",
    "women": "https://www.iihf.com/en/events/2026/olympics-w/schedule",
}
THE_SPORTS_DB_URL = "https://www.thesportsdb.com/api/v1/json/3/eventsday.php"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

COUNTRY_NAME_TO_CODE3 = {
    "canada": "CAN", "united states": "USA", "usa": "USA", "sweden": "SWE", "finland": "FIN",
    "czechia": "CZE", "czech republic": "CZE", "switzerland": "SUI", "germany": "GER",
    "slovakia": "SVK", "latvia": "LAT", "denmark": "DEN", "norway": "NOR", "france": "FRA",
    "italy": "ITA", "japan": "JPN", "china": "CHN", "korea": "KOR", "south korea": "KOR",
    "austria": "AUT", "great britain": "GBR", "britain": "GBR",
}


@dataclass
class ProviderResult:
    provider_name: str
    games: list[dict[str, Any]]
    reason: str


@dataclass
class _CacheEntry:
    expires_at: float
    value: Any


_cache: dict[str, _CacheEntry] = {}
_last_fetch_times: dict[str, float] = {}
_last_good_by_league: dict[str, list[dict[str, Any]]] = {}
_lock = threading.Lock()


def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _code3(team_blob: dict[str, Any]) -> str:
    for key in ("abbreviation", "triCode", "shortDisplayName", "displayName", "name"):
        raw = team_blob.get(key)
        if not isinstance(raw, str) or not raw.strip():
            continue
        value = raw.strip()
        if len(value) == 3 and value.isalpha():
            return value.upper()
        mapped = COUNTRY_NAME_TO_CODE3.get(value.lower())
        if mapped:
            return mapped
    return ""


def _normal_status(state: str) -> str:
    s = (state or "").lower()
    if s in {"post", "final", "complete", "completed"}:
        return "final"
    if s in {"in", "live", "inprogress"}:
        return "live"
    return "pre"


def _status_clock(status_blob: dict[str, Any]) -> tuple[str, str]:
    period = ""
    clock = ""
    type_blob = status_blob.get("type") if isinstance(status_blob, dict) else {}
    if isinstance(type_blob, dict):
        period = str(type_blob.get("shortDetail") or type_blob.get("detail") or "").strip()
    if isinstance(status_blob, dict):
        clock = str(status_blob.get("displayClock") or status_blob.get("clock") or "").strip()
    return period, clock


def _cache_get(key: str) -> Any:
    with _lock:
        entry = _cache.get(key)
        if not entry:
            return None
        if entry.expires_at < time.time():
            _cache.pop(key, None)
            return None
        return entry.value


def _cache_set(key: str, value: Any, ttl: int = CACHE_TTL_SECONDS) -> None:
    with _lock:
        _cache[key] = _CacheEntry(expires_at=time.time() + ttl, value=value)


def _rate_limit(provider_name: str) -> None:
    with _lock:
        last = _last_fetch_times.get(provider_name, 0.0)
        now = time.time()
        wait = RATE_LIMIT_SECONDS - (now - last)
        if wait > 0:
            time.sleep(wait)
        _last_fetch_times[provider_name] = time.time()


def resolve_display_date(*, tz_name: str | None = None, now: Optional[dt.datetime] = None) -> dt.date:
    tz = ZoneInfo(tz_name or DEFAULT_TIMEZONE)
    local = now or dt.datetime.now(tz)
    if local.tzinfo is None:
        local = local.replace(tzinfo=tz)
    else:
        local = local.astimezone(tz)
    cutoff = local.replace(hour=9, minute=30, second=0, microsecond=0)
    if local < cutoff:
        return (local - dt.timedelta(days=1)).date()
    return local.date()


def _http_json(url: str, *, params: Optional[dict[str, Any]] = None, provider_name: str) -> dict[str, Any]:
    _rate_limit(provider_name)
    response = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("JSON response is not an object")
    return payload


def _http_text(url: str, *, params: Optional[dict[str, Any]] = None, provider_name: str) -> str:
    _rate_limit(provider_name)
    response = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def _extract_balanced_json_value(text: str, start_index: int) -> str:
    if start_index >= len(text):
        return ""
    opening = text[start_index]
    if opening not in "[{":
        return ""
    closing = "]" if opening == "[" else "}"
    depth = 0
    in_string = False
    escaped = False
    for index in range(start_index, len(text)):
        ch = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return text[start_index:index + 1]
    return ""


def _extract_embedded_events_from_html(html: str) -> list[dict[str, Any]]:
    best: list[dict[str, Any]] = []
    for match in re.finditer(r'"events"\s*:\s*\[', html):
        start = html.find("[", match.start())
        if start < 0:
            continue
        blob = _extract_balanced_json_value(html, start)
        if not blob:
            continue
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, list):
            continue
        if not any(isinstance(event, dict) and event.get("competitions") for event in parsed):
            continue
        if len(parsed) > len(best):
            best = parsed

    if best:
        return best

    for script_blob in re.findall(r"<script[^>]*>(.*?)</script>", html, flags=re.IGNORECASE | re.DOTALL):
        for marker in ("window.__DATA__", "__NEXT_DATA__", "initialState", "appState"):
            marker_index = script_blob.find(marker)
            if marker_index < 0:
                continue
            json_start = script_blob.find("{", marker_index)
            if json_start < 0:
                continue
            json_blob = _extract_balanced_json_value(script_blob, json_start)
            if not json_blob:
                continue
            try:
                payload = json.loads(json_blob)
            except json.JSONDecodeError:
                continue
            candidate = _find_best_event_list(payload)
            if len(candidate) > len(best):
                best = candidate

    return best


def _find_best_event_list(payload: Any) -> list[dict[str, Any]]:
    best: list[dict[str, Any]] = []
    stack: list[Any] = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, list):
                    if key.lower() in {"events", "eventlist", "competitions"}:
                        normalized = [item for item in value if isinstance(item, dict)]
                        if _looks_like_event_list(normalized) and len(normalized) > len(best):
                            best = normalized
                    stack.extend(value)
                elif isinstance(value, dict):
                    stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
    return best


def _looks_like_event_list(values: list[dict[str, Any]]) -> bool:
    if not values:
        return False
    return any(
        (item.get("competitions") and isinstance(item.get("competitions"), list))
        or ("id" in item and "date" in item)
        for item in values
    )


def _matches_division(event: dict[str, Any], division: str) -> bool:
    texts = [
        str(event.get("name") or ""),
        str(event.get("shortName") or ""),
        str(event.get("seasonType") or ""),
    ]
    competitions = event.get("competitions") or []
    if competitions and isinstance(competitions[0], dict):
        competition = competitions[0]
        texts.extend([
            str(competition.get("name") or ""),
            str(competition.get("shortName") or ""),
            str((competition.get("type") or {}).get("text") or ""),
        ])
    haystack = " ".join(texts).lower()
    is_women = any(token in haystack for token in ("women", "women's", "w ice hockey", "w hockey"))
    if division == "women":
        return is_women
    return not is_women


def _espn_results_page_provider(date: dt.date, division: str) -> ProviderResult:
    html = _http_text(
        ESPN_RESULTS_PAGE_URL,
        provider_name=f"espn_results_page_{division}",
    )
    events = _extract_embedded_events_from_html(html)
    payload = {"events": [event for event in events if _matches_division(event, division)]}
    games = normalize_espn_olympic_response(payload, league_key=LEAGUE_KEYS[division])
    return ProviderResult("espn_results_page", games, f"events={len(games)} for {date.isoformat()}")


def _espn_provider(date: dt.date, division: str) -> ProviderResult:
    date_key = date.strftime("%Y%m%d")
    provider_name = f"espn_{division}"
    errors: list[str] = []
    payload: Optional[dict[str, Any]] = None
    selected_url = ""
    for url in ESPN_URL_CANDIDATES[division]:
        try:
            payload = _http_json(
                url,
                params={"dates": date_key},
                provider_name=provider_name,
            )
            selected_url = url
            break
        except HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code != 400:
                errors.append(f"{url} ({status_code or 'unknown'}): {exc}")
                continue
            logging.warning(
                "Olympic hockey ESPN returned 400; retrying without date filter division=%s date=%s url=%s",
                division,
                date_key,
                url,
            )
            try:
                payload = _http_json(
                    url,
                    params=None,
                    provider_name=provider_name,
                )
                selected_url = url
                break
            except HTTPError as retry_exc:
                retry_code = retry_exc.response.status_code if retry_exc.response is not None else None
                errors.append(f"{url} ({retry_code or 'unknown'}): {retry_exc}")
                continue
            except Exception as retry_exc:
                errors.append(f"{url}: {retry_exc}")
                continue
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    if payload is None:
        raise RuntimeError(f"ESPN olympic hockey fetch failed ({'; '.join(errors)})")

    if selected_url.endswith("/sports/hockey/olympics/scoreboard"):
        payload = {
            "events": [
                event
                for event in (payload.get("events") or [])
                if isinstance(event, dict) and _matches_division(event, division)
            ]
        }

    games = normalize_espn_olympic_response(payload, league_key=LEAGUE_KEYS[division])
    return ProviderResult("espn", games, f"events={len(games)} for {date_key}")


def normalize_espn_olympic_response(json_payload: dict[str, Any], *, league_key: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    fetched_at = _now_utc().isoformat()
    for event in json_payload.get("events") or []:
        competitions = event.get("competitions") or []
        competition = competitions[0] if competitions else {}
        competitors = competition.get("competitors") or []
        away = next((c for c in competitors if (c.get("homeAway") or "").lower() == "away"), None)
        home = next((c for c in competitors if (c.get("homeAway") or "").lower() == "home"), None)
        if not away or not home:
            continue
        away_team = away.get("team") or {}
        home_team = home.get("team") or {}
        status = event.get("status") or {}
        status_type = status.get("type") or {}
        period, clock = _status_clock(status)
        output.append(
            {
                "leagueKey": league_key,
                "gameId": str(event.get("id") or ""),
                "startTimeUTC": event.get("date") or "",
                "status": _normal_status(str(status_type.get("state") or "")),
                "period": period,
                "clock": clock,
                "home": {
                    "code3": _code3(home_team),
                    "name": home_team.get("displayName") or home_team.get("shortDisplayName") or "",
                    "score": _safe_int(home.get("score")),
                },
                "away": {
                    "code3": _code3(away_team),
                    "name": away_team.get("displayName") or away_team.get("shortDisplayName") or "",
                    "score": _safe_int(away.get("score")),
                },
                "venue": ((competition.get("venue") or {}).get("fullName") or "") if isinstance(competition, dict) else "",
                "source": {"providerName": "espn", "fetchedAtUTC": fetched_at},
            }
        )
    return output


def _thesportsdb_provider(date: dt.date, division: str) -> ProviderResult:
    payload = _http_json(
        THE_SPORTS_DB_URL,
        params={"d": date.strftime("%Y-%m-%d"), "s": "Ice_Hockey"},
        provider_name=f"thesportsdb_{division}",
    )
    events = payload.get("events") or []
    games: list[dict[str, Any]] = []
    fetched_at = _now_utc().isoformat()
    want_women = division == "women"
    for event in events:
        league = str(event.get("strLeague") or "").lower()
        season = str(event.get("strSeason") or "")
        if "olympic" not in league and "winter olympics" not in season.lower():
            continue
        text = " ".join([
            str(event.get("strEvent") or ""),
            str(event.get("strLeague") or ""),
        ]).lower()
        has_women_token = any(token in text for token in (" women", "women's", "w "))
        if want_women != has_women_token and "women" in text:
            continue
        home_name = str(event.get("strHomeTeam") or "")
        away_name = str(event.get("strAwayTeam") or "")
        status_text = str(event.get("strStatus") or "")
        status = "final" if status_text.lower() in {"ft", "final", "match finished"} else "pre"
        if status_text.lower() in {"live", "in progress"}:
            status = "live"
        start_time = ""
        if event.get("dateEvent") and event.get("strTime"):
            start_time = f"{event['dateEvent']}T{event['strTime']}Z"
        games.append(
            {
                "leagueKey": LEAGUE_KEYS[division],
                "gameId": str(event.get("idEvent") or f"tsdb-{home_name}-{away_name}-{event.get('dateEvent') or ''}"),
                "startTimeUTC": start_time,
                "status": status,
                "period": status_text,
                "clock": "",
                "home": {"code3": COUNTRY_NAME_TO_CODE3.get(home_name.lower(), ""), "name": home_name, "score": _safe_int(event.get("intHomeScore"))},
                "away": {"code3": COUNTRY_NAME_TO_CODE3.get(away_name.lower(), ""), "name": away_name, "score": _safe_int(event.get("intAwayScore"))},
                "venue": str(event.get("strVenue") or ""),
                "source": {"providerName": "thesportsdb", "fetchedAtUTC": fetched_at},
            }
        )
    return ProviderResult("thesportsdb", games, f"events={len(games)}")


def _iihf_scrape_provider(date: dt.date, division: str) -> ProviderResult:
    _rate_limit(f"iihf_{division}")
    response = SESSION.get(
        IIHF_URLS[division],
        timeout=REQUEST_TIMEOUT,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.iihf.com/",
        },
    )
    response.raise_for_status()
    html = response.text
    # Lightweight extraction of embedded tricode and score snippets.
    matches = re.findall(r'([A-Z]{3})\s*(\d+)\s*[-:]\s*(\d+)\s*([A-Z]{3})', html)
    games: list[dict[str, Any]] = []
    fetched_at = _now_utc().isoformat()
    for index, (away_code, away_score, home_score, home_code) in enumerate(matches):
        games.append(
            {
                "leagueKey": LEAGUE_KEYS[division],
                "gameId": f"iihf-{division}-{date.isoformat()}-{index}",
                "startTimeUTC": f"{date.isoformat()}T00:00:00Z",
                "status": "final",
                "period": "Final",
                "clock": "",
                "home": {"code3": home_code, "name": home_code, "score": _safe_int(home_score)},
                "away": {"code3": away_code, "name": away_code, "score": _safe_int(away_score)},
                "venue": "",
                "source": {"providerName": "iihf_html", "fetchedAtUTC": fetched_at},
            }
        )
    return ProviderResult("iihf_html", games, f"matches={len(games)}")


def _wikipedia_provider(date: dt.date, division: str) -> ProviderResult:
    year = date.year
    title = f"Ice hockey at the {year} Winter Olympics"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
    }
    payload = _http_json(WIKIPEDIA_API_URL, params=params, provider_name=f"wikipedia_{division}")
    pages = ((payload.get("query") or {}).get("pages") or {})
    text = ""
    for page in pages.values():
        if isinstance(page, dict):
            text = str(page.get("extract") or "")
            break
    league_token = "Women" if division == "women" else "Men"
    if league_token.lower() not in text.lower():
        return ProviderResult("wikipedia", [], "no matching section")
    return ProviderResult("wikipedia", [], "structured parsing unavailable")


def _provider_chain(date: dt.date, division: str) -> Iterable[Callable[[dt.date, str], ProviderResult]]:
    return (
        _espn_provider,
        _espn_results_page_provider,
        _iihf_scrape_provider,
        _thesportsdb_provider,
        _wikipedia_provider,
    )


def fetch_olympic_hockey_games(
    *,
    division: str,
    date: Optional[dt.date] = None,
    tz_name: str | None = None,
) -> list[dict[str, Any]]:
    if division not in LEAGUE_KEYS:
        raise ValueError(f"Unknown division '{division}'")
    selected_date = date or resolve_display_date(tz_name=tz_name)
    cache_key = f"olympic_hockey:{division}:{selected_date.isoformat()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    failures: list[str] = []
    for provider in _provider_chain(selected_date, division):
        provider_name = provider.__name__
        try:
            result = provider(selected_date, division)
            if result.games:
                logging.info(
                    "Olympic hockey provider selected division=%s provider=%s reason=%s",
                    division,
                    result.provider_name,
                    result.reason,
                )
                _cache_set(cache_key, result.games)
                with _lock:
                    _last_good_by_league[LEAGUE_KEYS[division]] = result.games
                return result.games
            failures.append(f"{result.provider_name}: empty ({result.reason})")
            logging.warning(
                "Olympic hockey provider returned empty division=%s provider=%s reason=%s",
                division,
                result.provider_name,
                result.reason,
            )
        except Exception as exc:
            failures.append(f"{provider_name}: {exc}")
            logging.exception("Olympic hockey provider failed division=%s provider=%s", division, provider_name)

    with _lock:
        fallback = _last_good_by_league.get(LEAGUE_KEYS[division], [])
    logging.error(
        "All Olympic hockey providers failed division=%s date=%s failures=%s fallback_count=%s",
        division,
        selected_date.isoformat(),
        failures,
        len(fallback),
    )
    _cache_set(cache_key, fallback)
    return fallback


def fetch_olympic_scoreboard_men(date: Optional[dt.date] = None, *, tz_name: str | None = None) -> list[dict[str, Any]]:
    return fetch_olympic_hockey_games(division="men", date=date, tz_name=tz_name)


def fetch_olympic_scoreboard_women(date: Optional[dt.date] = None, *, tz_name: str | None = None) -> list[dict[str, Any]]:
    return fetch_olympic_hockey_games(division="women", date=date, tz_name=tz_name)
