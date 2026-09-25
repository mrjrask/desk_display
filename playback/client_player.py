"""Stateful playback over locally cached server packages."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from schedule import ScreenScheduler


@dataclass(frozen=True)
class PlaybackItem:
    screen_id: str
    package: Any
    duration: float


@dataclass
class _LocalDefinition:
    id: str
    available: bool = True


class ClientPlayer:
    """Own playback position and controls; never refreshes or renders data."""

    def __init__(self, scheduler: ScreenScheduler, *, default_duration: float = 10.0) -> None:
        self.scheduler = scheduler
        self.default_duration = max(0.0, default_duration)
        self.manifest: dict[str, Any] = {}
        self.playlist: dict[str, dict[str, Any]] = {}
        self.packages: dict[str, Any] = {}
        self.history: list[str] = []
        self.current_id: str | None = None
        self.started_at: float | None = None
        self._focus: tuple[str, float | None] | None = None

    def load_cache(self, manifest: dict[str, Any], playlist: dict[str, Any], packages: dict[str, Any]) -> None:
        self.manifest = dict(manifest)
        self.playlist = {str(k): dict(v) if isinstance(v, dict) else {} for k, v in playlist.items()}
        self.packages = dict(packages)

    def focus(self, screen_id: str, *, seconds: float | None = None) -> None:
        self._focus = (screen_id, None if seconds is None else time.monotonic() + max(0.0, seconds))

    def clear_focus(self) -> None:
        self._focus = None

    def _focused_id(self) -> str | None:
        if self._focus is None:
            return None
        screen_id, deadline = self._focus
        if deadline is not None and time.monotonic() >= deadline:
            self._focus = None
            return None
        return screen_id

    def next(self) -> PlaybackItem | None:
        screen_id = self._focused_id()
        if screen_id is None:
            registry = {
                sid: _LocalDefinition(sid, sid in self.packages)
                for sid in self.scheduler.requested_ids
            }
            definition = self.scheduler.next_available(registry)
            if definition is None:
                return None
            screen_id = definition.id
        if self.current_id is not None:
            self.history.append(self.current_id)
        self.current_id = screen_id
        self.started_at = time.monotonic()
        return self._item(screen_id)

    def skip(self) -> PlaybackItem | None:
        return self.next()

    def previous(self) -> PlaybackItem | None:
        if not self.history:
            return None
        self.current_id = self.history.pop()
        self.started_at = time.monotonic()
        return self._item(self.current_id)

    def _item(self, screen_id: str) -> PlaybackItem:
        spec = self.playlist.get(screen_id, {})
        duration = float(
            spec.get(
                "duration",
                self.default_duration + self.scheduler.extra_seconds_for(screen_id),
            )
        )
        return PlaybackItem(screen_id, self.packages.get(screen_id), max(0.0, duration))

    def due(self, *, now: float | None = None) -> bool:
        if self.current_id is None or self.started_at is None:
            return True
        return (now if now is not None else time.monotonic()) - self.started_at >= self._item(self.current_id).duration
