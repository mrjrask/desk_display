"""Demand-aware, bounded rendering into the artifact store.

The coordinator turns the registry's merged demand (active dynamic clients,
static clients, administrator pre-render entries, and interaction
dependencies) into :class:`RenderKey` values and keeps the artifact store
current for exactly those keys:

* **Render once**: equal keys from many clients are one job, and a key that
  is already queued or in flight is never queued again.
* **Rerender only when relevant**: a lineage is rendered when it has no good
  output, when its screen's style, data or renderer revision changed (a new
  key), or when its output passed its refresh deadline.
* **Rate limits**: refreshes respect a per-screen minimum interval, and a
  failing lineage backs off exponentially.
* **Priority**: missing output first, then refreshes; within each, demand
  from connected clients before static clients before pre-render entries.
* **Bounded**: at most ``workers`` renders run at once, and a render that
  exceeds ``timeout_seconds`` is recorded as failed (its late result is
  discarded).
* **Demand follows leases**: queued work for demand that went away (an
  expired lease, a removed pre-render entry, a changed revision) is
  cancelled, and in-flight work for it is discarded when it finishes.
* **Last known good**: failures are recorded in the store, which keeps
  serving the previous good output as ``fallback``.
"""
from __future__ import annotations

import heapq
import itertools
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from remote_display.artifact_store import ArtifactStore, InvalidArtifactError, lineage_id
from remote_display.models import (
    ModelValidationError,
    RenderKey,
    ScreenRevisions,
    demand_render_keys,
)
from remote_display.registry import ClientRegistry

LOGGER = logging.getLogger("desk_display.render_coordinator")

PRIORITY_MISSING = 0
PRIORITY_REFRESH = 1
PRIORITY_NAMES = {PRIORITY_MISSING: "missing", PRIORITY_REFRESH: "refresh"}
SOURCE_RANK = {"dynamic": 0, "static": 1, "prerender": 2}
MAX_FAILURE_BACKOFF_SECONDS = 900
DURATION_SAMPLES = 20


@dataclass(frozen=True)
class RenderOutput:
    """What a renderer returns for one key: PNG bytes or a Pillow image."""

    data: bytes | None = None
    image: Any = None
    media_type: str = "image/png"
    refresh_seconds: float = 300
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Optional render package (remote_display/render_package.py) published
    # alongside the still image for screens that move.
    package: Mapping[str, Any] | None = None


Renderer = Callable[[RenderKey], RenderOutput]
RevisionSource = Callable[[Iterable[str]], Mapping[str, ScreenRevisions]]


class Executor(Protocol):
    def submit(self, fn: Callable[[], Any]) -> Future: ...


@dataclass
class _Job:
    key: RenderKey
    lineage: str
    priority: int
    source_rank: int
    clients: frozenset[str]
    enqueued_at: float
    started_at: float | None = None
    cancelled: bool = False
    timed_out: bool = False


@dataclass
class _LineageStats:
    screen_id: str
    render_profile: str
    client_scope: str | None
    last_started: float | None = None
    last_success: float | None = None
    last_failure: float | None = None
    last_error: str | None = None
    consecutive_failures: int = 0
    failed_digest: str | None = None
    renders: int = 0
    failures: int = 0
    durations: deque = field(default_factory=lambda: deque(maxlen=DURATION_SAMPLES))


def _iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class RenderCoordinator:
    """Keep the artifact store current for active demand with bounded work."""

    def __init__(
        self,
        registry: ClientRegistry,
        store: ArtifactStore,
        renderer: Renderer,
        revisions: RevisionSource,
        *,
        workers: int = 2,
        timeout_seconds: float = 30,
        min_interval_seconds: float = 30,
        screen_min_intervals: Mapping[str, float] | None = None,
        client_specific_screens: Iterable[str] = (),
        data_health: Callable[[], Mapping[str, Any]] | None = None,
        executor: Executor | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self.registry = registry
        self.store = store
        self.renderer = renderer
        self.revisions = revisions
        self.workers = int(workers)
        self.timeout_seconds = float(timeout_seconds)
        self.min_interval_seconds = float(min_interval_seconds)
        self.screen_min_intervals = dict(screen_min_intervals or {})
        self.client_specific_screens = tuple(client_specific_screens)
        self.data_health = data_health
        self._executor = executor or ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="render")
        self._clock = clock
        self._lock = threading.RLock()
        self._queue: list[tuple[int, int, float, int, str]] = []
        self._queued: dict[str, _Job] = {}
        self._in_flight: dict[str, _Job] = {}
        self._stats: dict[str, _LineageStats] = {}
        self._plan: dict[str, tuple[RenderKey, frozenset[str], int]] = {}
        self._invalid_demand: list[dict[str, Any]] = []
        self._counter = itertools.count()
        self._last_tick: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ── Planning ───────────────────────────────────────────────────────────

    def plan(self) -> dict[str, tuple[RenderKey, frozenset[str], int]]:
        """Return ``{key digest: (key, client IDs, best source rank)}`` for current demand."""

        entries = self.registry.demand_entries()
        screens = sorted({s for entry in entries for s in entry.demand.all_screens})
        revisions = dict(self.revisions(screens)) if screens else {}
        plan: dict[str, tuple[RenderKey, set[str], int]] = {}
        invalid: list[dict[str, Any]] = []
        for entry in entries:
            try:
                keys = demand_render_keys(
                    entry.capabilities, entry.demand, revisions,
                    client_specific_screens=self.client_specific_screens,
                )
            except ModelValidationError as exc:
                # One bad client must not stop everyone else's renders.
                invalid.append({"client_id": entry.client_id, "source": entry.source, "error": str(exc)})
                continue
            rank = SOURCE_RANK.get(entry.source, len(SOURCE_RANK))
            for key in keys:
                existing = plan.get(key.digest)
                if existing is None:
                    plan[key.digest] = (key, {entry.client_id}, rank)
                else:
                    existing[1].add(entry.client_id)
                    plan[key.digest] = (key, existing[1], min(existing[2], rank))
        with self._lock:
            self._invalid_demand = invalid
        return {digest: (key, frozenset(clients), rank) for digest, (key, clients, rank) in plan.items()}

    def tick(self) -> None:
        """Re-plan, cancel stale work, enqueue needed renders and fill free workers."""

        plan = self.plan()
        now = self._clock()
        with self._lock:
            self._plan = plan
            self._last_tick = now
            for digest in list(self._queued):
                if digest not in plan:
                    self._queued.pop(digest).cancelled = True
            for digest, job in self._in_flight.items():
                if digest not in plan:
                    job.cancelled = True
                elif not job.timed_out and job.started_at is not None and now - job.started_at > self.timeout_seconds:
                    job.timed_out = True
                    self._record_failure(job, "timeout", f"render exceeded {self.timeout_seconds:g}s", now)
            for digest, (key, clients, rank) in plan.items():
                if digest in self._queued or digest in self._in_flight:
                    continue
                priority = self._needed(key, now)
                if priority is None:
                    continue
                self._enqueue(key, clients, rank, priority, now)
        self._dispatch()

    def _needed(self, key: RenderKey, now: float) -> int | None:
        lineage = lineage_id(key.screen_id, key.render_profile, key.client_scope)
        stats = self._stats.get(lineage)
        # Back off a lineage whose latest attempt at this very key failed.
        if stats and stats.consecutive_failures and stats.failed_digest == key.digest and stats.last_failure:
            backoff = min(
                self._min_interval(key.screen_id) * 2 ** (stats.consecutive_failures - 1),
                MAX_FAILURE_BACKOFF_SECONDS,
            )
            if now - stats.last_failure < backoff:
                return None
        resolved = self.store.resolve(key.screen_id, key.render_profile, key.client_scope)
        if resolved.record is None:
            return PRIORITY_MISSING
        changed = resolved.record.render_key_digest != key.digest
        expired = now >= resolved.record.refresh_deadline
        if not changed and not expired:
            return None
        if stats and stats.last_started is not None and now - stats.last_started < self._min_interval(key.screen_id):
            return None
        return PRIORITY_REFRESH

    def _min_interval(self, screen: str) -> float:
        return float(self.screen_min_intervals.get(screen, self.min_interval_seconds))

    def _enqueue(self, key: RenderKey, clients: frozenset[str], rank: int, priority: int, now: float) -> None:
        lineage = lineage_id(key.screen_id, key.render_profile, key.client_scope)
        job = _Job(key, lineage, priority, rank, clients, now)
        self._queued[key.digest] = job
        heapq.heappush(self._queue, (priority, rank, now, next(self._counter), key.digest))
        self._stats.setdefault(lineage, _LineageStats(key.screen_id, key.render_profile, key.client_scope))
        if priority == PRIORITY_REFRESH:
            try:
                self.store.mark_pending(key)
            except OSError as exc:  # pragma: no cover - reported, not fatal
                LOGGER.warning("Could not mark %s pending: %s", key.screen_id, exc)

    # ── Execution ──────────────────────────────────────────────────────────

    def _dispatch(self) -> None:
        while True:
            with self._lock:
                if len(self._in_flight) >= self.workers or not self._queue:
                    return
                _, _, _, _, digest = heapq.heappop(self._queue)
                job = self._queued.pop(digest, None)
                if job is None or job.cancelled:
                    continue
                job.started_at = self._clock()
                self._in_flight[digest] = job
                self._stats[job.lineage].last_started = job.started_at
            future = self._executor.submit(lambda job=job: self._run(job))
            future.add_done_callback(lambda f, job=job: self._finished(job, f))

    def _run(self, job: _Job) -> None:
        output = self.renderer(job.key)
        if not isinstance(output, RenderOutput):
            self.store.record_failure(job.key, "invalid_output", "renderer returned an unsupported value")
            raise InvalidArtifactError("invalid_output", "renderer returned an unsupported value")
        with self._lock:
            discard = job.cancelled or job.timed_out
        if discard:
            return None
        kwargs = {"refresh_seconds": output.refresh_seconds, "metadata": output.metadata,
                  "package": output.package}
        if output.image is not None:
            return self.store.publish_image(job.key, output.image, **kwargs)
        return self.store.publish(job.key, output.data or b"", media_type=output.media_type, **kwargs)

    def _finished(self, job: _Job, future: Future) -> None:
        now = self._clock()
        error = future.exception()
        with self._lock:
            self._in_flight.pop(job.key.digest, None)
            stats = self._stats[job.lineage]
            if job.timed_out:
                pass  # already recorded as a failure when it timed out
            elif error is None:
                if not job.cancelled and future.result() is not None:
                    stats.renders += 1
                    stats.last_success = now
                    stats.consecutive_failures = 0
                    stats.failed_digest = None
                    stats.last_error = None
                    stats.durations.append(now - (job.started_at or now))
            elif isinstance(error, InvalidArtifactError):
                # The store already recorded the failure against the lineage.
                self._count_failure(stats, job, error.code, now)
            else:
                self._record_failure(job, "render_error", f"{type(error).__name__}: {error}", now)
        self._dispatch()

    def _record_failure(self, job: _Job, code: str, message: str, now: float) -> None:
        try:
            self.store.record_failure(job.key, code, message)
        except OSError as exc:  # pragma: no cover - reported, not fatal
            LOGGER.warning("Could not record render failure: %s", exc)
        self._count_failure(self._stats[job.lineage], job, code, now)

    @staticmethod
    def _count_failure(stats: _LineageStats, job: _Job, code: str, now: float) -> None:
        stats.failures += 1
        stats.consecutive_failures += 1
        stats.failed_digest = job.key.digest
        stats.last_failure = now
        stats.last_error = code

    # ── Background loop ────────────────────────────────────────────────────

    def start(self, interval_seconds: float = 5) -> threading.Thread:
        def loop() -> None:
            while not self._stop.is_set():
                try:
                    self.tick()
                except Exception:  # pragma: no cover - logged and retried
                    LOGGER.exception("Render coordinator tick failed")
                self._stop.wait(interval_seconds)

        self._thread = threading.Thread(target=loop, name="render-coordinator", daemon=True)
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        self._stop.set()
        if isinstance(self._executor, ThreadPoolExecutor):
            self._executor.shutdown(wait=False, cancel_futures=True)

    # ── Status ─────────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Operator view of demand, queue, render health, data and storage."""

        now = self._clock()
        with self._lock:
            plan = dict(self._plan)
            queued = sorted(self._queued.values(), key=lambda j: (j.priority, j.source_rank, j.enqueued_at))
            in_flight = list(self._in_flight.values())
            stats = dict(self._stats)
            invalid = list(self._invalid_demand)
        profiles: dict[str, int] = {}
        for key, _clients, _rank in plan.values():
            profiles[key.render_profile] = profiles.get(key.render_profile, 0) + 1
        lineages = []
        states: dict[str, int] = {}
        for key, clients, _rank in sorted(plan.values(), key=lambda p: (p[0].screen_id, p[0].render_profile)):
            resolved = self.store.resolve(key.screen_id, key.render_profile, key.client_scope)
            states[resolved.state] = states.get(resolved.state, 0) + 1
            s = stats.get(lineage_id(key.screen_id, key.render_profile, key.client_scope))
            durations = list(s.durations) if s else []
            lineages.append({
                "screen_id": key.screen_id,
                "render_profile": key.render_profile,
                "client_scope": key.client_scope,
                "clients": sorted(clients),
                "state": resolved.state,
                "generated_at": None if resolved.record is None else _iso(resolved.record.generated_at),
                "refresh_deadline": None if resolved.record is None else _iso(resolved.record.refresh_deadline),
                "last_success": _iso(s.last_success) if s else None,
                "last_failure": _iso(s.last_failure) if s else None,
                "last_error": s.last_error if s else None,
                "consecutive_failures": s.consecutive_failures if s else 0,
                "renders": s.renders if s else 0,
                "failures": s.failures if s else 0,
                "last_duration_ms": int(durations[-1] * 1000) if durations else None,
                "average_duration_ms": int(sum(durations) / len(durations) * 1000) if durations else None,
            })
        clients: dict[str, int] = {}
        for record in self.registry.records():
            state = record.lease_state(now)
            clients[state] = clients.get(state, 0) + 1
        return {
            "last_tick": _iso(self._last_tick),
            "workers": self.workers,
            "timeout_seconds": self.timeout_seconds,
            "clients": clients,
            "profiles": profiles,
            "render_keys": len(plan),
            "artifact_states": states,
            "queue": [
                {
                    "screen_id": j.key.screen_id,
                    "render_profile": j.key.render_profile,
                    "reason": PRIORITY_NAMES[j.priority],
                    "waiting_seconds": round(now - j.enqueued_at, 1),
                }
                for j in queued
            ],
            "in_flight": [
                {
                    "screen_id": j.key.screen_id,
                    "render_profile": j.key.render_profile,
                    "running_seconds": round(now - (j.started_at or now), 1),
                    "timed_out": j.timed_out,
                    "cancelled": j.cancelled,
                }
                for j in in_flight
            ],
            "lineages": lineages,
            "invalid_demand": invalid,
            "data_health": dict(self.data_health()) if self.data_health else None,
            "artifacts": self.store.stats(),
            "playlists": self._playlist_delivery(),
        }

    def _playlist_delivery(self) -> list[dict[str, Any]]:
        rows = []
        for record in self.registry.records():
            assignment = self.registry.assignments(record.client_id)
            saved = None if assignment is None else assignment.playlist_revision
            delivered = record.delivered_playlist_revision
            acknowledged = None
            if record.status is not None and record.status.accepted_revisions is not None:
                acknowledged = record.status.accepted_revisions.playlist_revision
            if saved is None:
                state = "unassigned"
            elif delivered != saved:
                state = "pending_delivery"
            elif acknowledged != saved:
                state = "pending_acknowledgment"
            else:
                state = "in_sync"
            rows.append({
                "client_id": record.client_id,
                "saved_revision": saved,
                "delivered_revision": delivered,
                "acknowledged_revision": acknowledged,
                "state": state,
            })
        return rows


__all__ = ["RenderCoordinator", "RenderOutput", "Renderer", "RevisionSource"]
