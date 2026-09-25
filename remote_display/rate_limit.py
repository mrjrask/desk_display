"""Token-bucket rate limits for the render server API.

Buckets are keyed by the caller (a client ID once it is known, else the
remote address) and the kind of request. Limits are sized for a real client
(registration at startup and after a lapsed lease, a heartbeat about every
lease/3, a manifest check every sync interval, and bursts of artifact
downloads after a new manifest) with room to spare, so they only bite on
credential guessing or a runaway client.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Limit:
    burst: float
    per_second: float


LIMITS: dict[str, Limit] = {
    "register": Limit(burst=20, per_second=1 / 3),  # per remote address
    "auth_failure": Limit(burst=20, per_second=1 / 3),  # per remote address
    "heartbeat": Limit(burst=30, per_second=1),  # per client
    "manifest": Limit(burst=60, per_second=2),  # per client: config and manifest
    "artifact": Limit(burst=600, per_second=50),  # per client
}
MAX_BUCKETS = 4096


class RateLimiter:
    def __init__(self, limits: dict[str, Limit] | None = None, *,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.limits = dict(LIMITS if limits is None else limits)
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: dict[tuple[str, str], tuple[float, float]] = {}

    def _level(self, kind: str, key: str, now: float) -> tuple[Limit, float]:
        limit = self.limits[kind]
        tokens, updated = self._buckets.get((kind, key), (limit.burst, now))
        return limit, min(limit.burst, tokens + (now - updated) * limit.per_second)

    def retry_after(self, kind: str, key: str) -> float:
        """Seconds until one request of *kind* is allowed for *key* (0 when allowed now)."""

        with self._lock:
            limit, tokens = self._level(kind, key, self._clock())
        return 0.0 if tokens >= 1 else (1 - tokens) / limit.per_second

    def hit(self, kind: str, key: str) -> float:
        """Spend one token; return 0 when allowed, else the seconds to wait."""

        now = self._clock()
        with self._lock:
            limit, tokens = self._level(kind, key, now)
            if tokens < 1:
                self._buckets[(kind, key)] = (tokens, now)
                return (1 - tokens) / limit.per_second
            if len(self._buckets) >= MAX_BUCKETS and (kind, key) not in self._buckets:
                self._prune(now)
            self._buckets[(kind, key)] = (tokens - 1, now)
            return 0.0

    def _prune(self, now: float) -> None:
        """Forget full buckets; they behave exactly like missing ones."""

        for bucket_key, (tokens, updated) in list(self._buckets.items()):
            limit = self.limits[bucket_key[0]]
            if tokens + (now - updated) * limit.per_second >= limit.burst:
                del self._buckets[bucket_key]
        if len(self._buckets) >= MAX_BUCKETS:
            self._buckets.clear()


__all__ = ["LIMITS", "Limit", "RateLimiter"]
