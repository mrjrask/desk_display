"""Shared HTTP client utilities with browser-like headers and retries."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Optional
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

NHL_HEADERS: dict[str, str] = {
    "Origin": "https://www.nhl.com",
    "Referer": "https://www.nhl.com/",
}

_RETRY = Retry(
    total=4,
    connect=4,
    read=4,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=False,
    raise_on_status=False,
)

_USE_SYSTEM_PROXIES = (
    os.environ.get("HTTP_CLIENT_USE_SYSTEM_PROXIES", "").strip().lower()
    in {"1", "true", "yes", "on"}
)

# A 403 from a host almost always means "you are being rate-limited/blocked",
# not "this particular resource doesn't exist" -- every other request to that
# same host is likely to fail the same way for a while. Callers that scan many
# URLs on the same host in a tight loop (e.g. a day-by-day schedule lookahead)
# would otherwise keep making real, slow network round-trips for every
# remaining day even though the host is clearly not going to answer. Once a
# host returns 403, short-circuit further requests to it for a cooldown
# window instead of hammering it -- this fails fast (no network I/O) and
# gives the block a chance to clear.
FORBIDDEN_COOLDOWN_SECONDS = float(
    os.environ.get("HTTP_CLIENT_FORBIDDEN_COOLDOWN_SECONDS", "300")
)


class HostTemporarilyForbidden(requests.exceptions.RequestException):
    """Raised in place of a real request while a host is in its 403 cooldown."""


def _host_of(url: str) -> str:
    return (urlsplit(url).netloc or "").lower()


class _CircuitBreakerSession(requests.Session):
    """A session whose 403 cooldown state is private to this instance.

    Several unrelated screens (NFL, NBA, NCAAM, World Cup) all fetch from
    the same site.api.espn.com host. Day-by-day scan bugs in one of them
    (Bulls widget, NBA Playoffs) have repeatedly tripped a *shared*
    circuit breaker and collaterally blocked every other sport's requests
    to that host for the cooldown window -- including sports that never
    made an excessive request themselves. Keeping the forbidden-host
    tracking per session instance (see get_session()'s ``name`` param)
    means a 403 seen by one caller only cools that caller down, not
    everyone who happens to share the host.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._forbidden_hosts_lock = threading.Lock()
        self._forbidden_hosts_until: dict[str, float] = {}

    def request(self, method, url, *args, **kwargs):  # type: ignore[override]
        host = _host_of(url)
        now = time.monotonic()
        if host:
            with self._forbidden_hosts_lock:
                blocked_until = self._forbidden_hosts_until.get(host)
            if blocked_until is not None and now < blocked_until:
                raise HostTemporarilyForbidden(
                    f"{host} returned 403 recently; skipping request to {url} for "
                    f"another {blocked_until - now:.0f}s to avoid hammering it"
                )

        response = super().request(method, url, *args, **kwargs)

        if host:
            with self._forbidden_hosts_lock:
                if response.status_code == 403:
                    self._forbidden_hosts_until[host] = now + FORBIDDEN_COOLDOWN_SECONDS
                else:
                    self._forbidden_hosts_until.pop(host, None)

        return response


def _build_session() -> requests.Session:
    session = _CircuitBreakerSession()
    session.trust_env = _USE_SYSTEM_PROXIES
    session.headers.update(DEFAULT_HEADERS)
    adapter = HTTPAdapter(max_retries=_RETRY)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_sessions_lock = threading.Lock()
_sessions: dict[str, requests.Session] = {}


def get_session(name: str = "default") -> requests.Session:
    """Return a shared HTTP session for ``name``.

    Each distinct ``name`` gets its own session -- and therefore its own
    circuit-breaker cooldown state -- so a 403 one caller triggers never
    short-circuits a different caller's requests to that same host. Callers
    that don't care about isolation (most non-scoreboard fetches) can omit
    ``name`` and share the default session as before.
    """

    with _sessions_lock:
        session = _sessions.get(name)
        if session is None:
            session = _build_session()
            _sessions[name] = session
        return session


def http_get(
    url: str,
    *,
    params: Optional[dict[str, Any]] = None,
    timeout: float = 10.0,
    headers: Optional[dict[str, str]] = None,
    session: Optional[requests.Session] = None,
    **kwargs: Any,
) -> requests.Response:
    """Perform a GET request using the shared HTTP session."""

    sess = session or _SESSION
    return sess.get(url, params=params, headers=headers, timeout=timeout, **kwargs)


class DayScanCooldown:
    """Cooldown gate for day-by-day schedule scans.

    Several sport screens fall back to walking a range of individual days
    looking for games, one HTTP request per day, because the source API has
    no bulk "next game" endpoint for that lookup. Repeating that burst every
    refresh cycle -- e.g. during an off-season, when it's guaranteed to come
    up empty -- is enough to trip the host's rate limiter, which then blocks
    every other caller sharing that host's circuit breaker cooldown (see
    _CircuitBreakerSession above), including unrelated sports. Two separate
    day-by-day scans (an NBA Bulls widget and the NBA Playoffs screen) have
    already caused the NFL scoreboard to go blank this way.

    Call `blocked()` before a scan and skip it entirely while it returns
    True; call `mark_empty()` after a scan that found nothing to start the
    cooldown, and `reset()` as soon as a scan finds something so the next
    miss starts a fresh cooldown rather than inheriting a stale one.
    """

    def __init__(self, cooldown_seconds: float):
        self._cooldown_seconds = cooldown_seconds
        self._checked_at: Optional[float] = None
        self._lock = threading.Lock()

    def blocked(self) -> bool:
        with self._lock:
            checked_at = self._checked_at
        if checked_at is None:
            return False
        return (time.monotonic() - checked_at) < self._cooldown_seconds

    def mark_empty(self) -> None:
        with self._lock:
            self._checked_at = time.monotonic()

    def reset(self) -> None:
        with self._lock:
            self._checked_at = None


def request_json(
    url: str,
    *,
    params: Optional[dict[str, Any]] = None,
    timeout: float = 10.0,
    headers: Optional[dict[str, str]] = None,
    quiet: bool = False,
    session: Optional[requests.Session] = None,
    **kwargs: Any,
) -> Optional[Any]:
    """Perform a GET request that returns JSON, with optional quiet logging."""

    sess = session or _SESSION
    try:
        response = http_get(url, params=params, headers=headers, timeout=timeout, session=sess, **kwargs)
        response.raise_for_status()
        return response.json()
    except Exception as exc:  # pragma: no cover - defensive network layer
        if not quiet:
            logging.warning("Request failed: %s (%s)", url, exc)
        return None
