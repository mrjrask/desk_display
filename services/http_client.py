"""Shared HTTP client utilities with browser-like headers and retries."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

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


def _build_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = _USE_SYSTEM_PROXIES
    session.headers.update(DEFAULT_HEADERS)
    adapter = HTTPAdapter(max_retries=_RETRY)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_SESSION = _build_session()


def get_session() -> requests.Session:
    """Return the shared HTTP session."""

    return _SESSION


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
