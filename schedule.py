"""Frequency-based screen scheduling in numbered cycles.

Each positive-frequency base entry is emitted in cycle 1.  Thereafter,
frequency ``N`` is due on cycles ``1 + N``, ``1 + 2N``, and so on, while an
alternate's frequency counts only the due presentations of its base entry.
Frequency-zero entries have no independent slot, although their screen IDs may
still be referenced as alternates. Entry order is preserved within every cycle.
Constructing a new scheduler, including after a config reload, starts again at
cycle 1.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, Optional

from display_time import CENTRAL_TIME
from screens_catalog import SCREEN_IDS, canonical_screen_id

if TYPE_CHECKING:
    from screens.registry import ScreenDefinition


KNOWN_SCREENS: set[str] = set(SCREEN_IDS)
REPLACEMENT_ONLY_SCREENS: set[str] = {"cubs no game", "sox no game"}


@dataclass
class _AlternateSchedule:
    """Round-robin alternate IDs and their base-presentation interval."""

    screen_ids: tuple[str, ...]
    frequency: int
    cursor: int = 0

    def next_screen_id(self) -> str:
        """Return the next alternate ID and advance the round-robin cursor."""

        if not self.screen_ids:
            raise ValueError("Alternate schedule requires at least one screen id")

        screen_id = self.screen_ids[self.cursor]
        self.cursor = (self.cursor + 1) % len(self.screen_ids)
        return screen_id


@dataclass
class _ScheduleEntry:
    """Runtime state for one enabled, independently scheduled base screen."""

    screen_id: str
    frequency: int
    presentation_count: int = 0
    extra_seconds: int = 0
    hide_after: Optional[datetime] = None
    alternate: Optional[_AlternateSchedule] = None


@dataclass(frozen=True)
class ScheduledPreviewEntry:
    """One non-mutating preview result with its scheduling context.

    Rotation cycles are numbered starting at one; gaps in their numbers identify
    cycles where no configured entry was due, even though those empty cycles
    produce no result of their own.
    """

    screen_id: str
    cycle_number: int

    @property
    def pass_number(self) -> int:
        """Backward-compatible alias for callers that still call cycles passes."""

        return self.cycle_number

    @property
    def phase(self) -> Literal["normal"]:
        """Backward-compatible phase marker; startup is no longer separate."""

        return "normal"


class ScreenScheduler:
    """Yield screens in ordered, frequency-based cycles starting at cycle 1."""

    def __init__(self, entries: Sequence[_ScheduleEntry]):
        self._entries: list[_ScheduleEntry] = list(entries)
        self._cursor: int = 0
        self._pending_indices: Optional[list[int]] = None
        self._cycle_number: int = 1
        self._extra_seconds_by_id: dict[str, int] = {}
        requested: set[str] = set()
        for entry in self._entries:
            requested.add(entry.screen_id)
            self._extra_seconds_by_id[entry.screen_id] = max(
                self._extra_seconds_by_id.get(entry.screen_id, 0),
                int(entry.extra_seconds),
            )
            if entry.alternate is not None:
                requested.update(entry.alternate.screen_ids)
        self._requested = requested

    @property
    def node_count(self) -> int:
        return len(self._entries)

    @property
    def entry_ids(self) -> tuple[str, ...]:
        """Base screen IDs in play order, including zero-frequency entries."""

        return tuple(entry.screen_id for entry in self._entries)

    @property
    def enabled_ids(self) -> tuple[str, ...]:
        """Base screen IDs with a positive frequency, in play order."""

        return tuple(entry.screen_id for entry in self._entries if entry.frequency > 0)

    @property
    def requested_ids(self) -> set[str]:
        return set(self._requested)

    def extra_seconds_for(self, screen_id: str) -> int:
        return max(0, int(self._extra_seconds_by_id.get(screen_id, 0)))

    def alt_screen_ids_for(self, screen_id: str) -> tuple[str, ...]:
        """Return alternate screen ids configured on *screen_id*'s entry.

        Used to keep an alternate's own outputs (e.g. a screenshot) fresh on
        cycles where the base screen renders instead of the alternate.
        """

        ids: list[str] = []
        for entry in self._entries:
            if entry.screen_id == screen_id and entry.alternate is not None:
                for alt_id in entry.alternate.screen_ids:
                    if alt_id not in ids:
                        ids.append(alt_id)
        return tuple(ids)

    def preview_scheduled_ids(self, limit: int) -> list[str]:
        """Return upcoming scheduled screen IDs without mutating scheduler state."""

        return [entry.screen_id for entry in self.preview_scheduled_entries(limit)]

    def preview_scheduled_entries(self, limit: int) -> list[ScheduledPreviewEntry]:
        """Return upcoming IDs annotated with normal cycle information.

        This cycle-aware form is intended for diagnostics and scheduler tests.
        Production callers that only need IDs should use
        :meth:`preview_scheduled_ids`.
        """

        if limit <= 0 or not self._entries:
            return []

        cloned_entries: list[_ScheduleEntry] = []
        for entry in self._entries:
            cloned_alt: Optional[_AlternateSchedule] = None
            if entry.alternate is not None:
                cloned_alt = _AlternateSchedule(
                    screen_ids=entry.alternate.screen_ids,
                    frequency=entry.alternate.frequency,
                    cursor=entry.alternate.cursor,
                )

            cloned_entries.append(
                _ScheduleEntry(
                    screen_id=entry.screen_id,
                    frequency=entry.frequency,
                    presentation_count=entry.presentation_count,
                    extra_seconds=entry.extra_seconds,
                    hide_after=entry.hide_after,
                    alternate=cloned_alt,
                )
            )

        preview = ScreenScheduler(cloned_entries)
        preview._cursor = self._cursor
        preview._pending_indices = (
            None if self._pending_indices is None else self._pending_indices.copy()
        )
        preview._cycle_number = self._cycle_number

        scheduled_entries: list[ScheduledPreviewEntry] = []
        for _ in range(limit):
            next_id = preview._next_scheduled_id()
            if next_id is None:
                break
            scheduled_entries.append(
                ScheduledPreviewEntry(
                    screen_id=next_id,
                    cycle_number=preview._cycle_number,
                )
            )

        return scheduled_entries

    def _next_scheduled_id(self) -> Optional[str]:
        """Return the next ordered ID without availability checks.

        Cycle 1 returns every enabled base ID; later cycles may use alternates.
        """

        if not self._entries:
            return None

        if self._pending_indices is None:
            self._queue_current_cycle(datetime.now(UTC))
        if not self._pending_indices and not self._advance_until_pending(datetime.now(UTC)):
            return None

        assert self._pending_indices is not None
        entry_index = self._pending_indices.pop(0)
        self._cursor = (entry_index + 1) % len(self._entries)
        return self._scheduled_id_for(
            self._entries[entry_index],
            force_base=self._cycle_number == 1,
        )

    def _scheduled_id_for(
        self,
        entry: _ScheduleEntry,
        *,
        force_base: bool = False,
    ) -> str:
        """Resolve a due entry, always showing its base in cycle 1."""
        entry.presentation_count += 1
        if (
            not force_base
            and entry.alternate
            and entry.alternate.frequency > 0
            and entry.presentation_count % entry.alternate.frequency == 0
        ):
            return entry.alternate.next_screen_id()
        return entry.screen_id

    def _queue_current_cycle(self, now_utc: datetime) -> None:
        """Queue every due entry for the current configuration-ordered cycle.

        Cycle 1 queues each positive-frequency base exactly once in configuration
        order.

        Building the whole cycle before returning its first screen prevents later
        calls from interleaving entries from different frequency cycles.  The
        queue is also retained while registry data is refreshed, so rebuilding
        screen definitions cannot alter the configured playback order.
        """

        self._pending_indices = []

        for index, entry in enumerate(self._entries):
            if entry.frequency == 0:
                continue
            if entry.hide_after is not None and now_utc >= entry.hide_after:
                continue

            if (self._cycle_number - 1) % entry.frequency == 0:
                self._pending_indices.append(index)

        self._cursor = 0

    def _advance_until_pending(self, now_utc: datetime) -> bool:
        """Advance to the next nonempty ordered cycle, skipping empty ranges."""

        self._cycle_number += 1
        self._queue_current_cycle(now_utc)
        if self._pending_indices:
            return True

        active_frequencies = [
            entry.frequency
            for entry in self._entries
            if entry.frequency > 0 and (entry.hide_after is None or now_utc < entry.hide_after)
        ]
        if not active_frequencies:
            return False

        # The cycle just queued was empty. Jump to immediately before the
        # nearest future match rather than scanning every intervening cycle;
        # Then queue that selected cycle directly.
        next_due_cycle = min(
            self._cycle_number + (frequency - ((self._cycle_number - 1) % frequency))
            for frequency in active_frequencies
        )
        self._cycle_number = next_due_cycle
        self._queue_current_cycle(now_utc)
        return bool(self._pending_indices)

    def _next_available_from_entry(
        self,
        entry: _ScheduleEntry,
        registry: dict[str, ScreenDefinition],
        *,
        force_base: bool = False,
    ) -> Optional[ScreenDefinition]:
        """Resolve a queued entry, always showing its base in cycle 1."""

        entry.presentation_count += 1
        if (
            not force_base
            and entry.alternate
            and entry.alternate.frequency > 0
            and entry.presentation_count % entry.alternate.frequency == 0
        ):
            alternate = entry.alternate
            for _ in range(len(alternate.screen_ids)):
                alt_id = alternate.next_screen_id()
                alt_def = registry.get(alt_id)
                if alt_def and alt_def.available:
                    return alt_def

        definition = registry.get(entry.screen_id)
        if definition and definition.available:
            return definition
        return None

    # ── Position persistence (display clients) ─────────────────────────────

    def _signature(self) -> list[Any]:
        return [
            [e.screen_id, e.frequency, list(e.alternate.screen_ids) if e.alternate else None]
            for e in self._entries
        ]

    def export_state(self) -> dict[str, Any]:
        """Return the playback position as JSON-safe data."""

        return {
            "signature": self._signature(),
            "cycle": self._cycle_number,
            "cursor": self._cursor,
            "pending": None if self._pending_indices is None else list(self._pending_indices),
            "counts": [e.presentation_count for e in self._entries],
            "alternate_cursors": [e.alternate.cursor if e.alternate else 0 for e in self._entries],
        }

    def restore_state(self, state: Any) -> bool:
        """Resume from :meth:`export_state` output for the same schedule.

        Returns ``False`` and leaves the scheduler at cycle 1 when the state
        belongs to a different schedule or is malformed.
        """

        try:
            if not isinstance(state, dict) or state.get("signature") != self._signature():
                return False
            count = len(self._entries)
            cycle, cursor = int(state["cycle"]), int(state["cursor"])
            pending = state.get("pending")
            counts = [int(v) for v in state["counts"]]
            cursors = [int(v) for v in state["alternate_cursors"]]
            if cycle < 1 or not 0 <= cursor <= max(0, count - 1) or len(counts) != count or len(cursors) != count:
                return False
            if pending is not None:
                pending = [int(i) for i in pending]
                if any(not 0 <= i < count for i in pending):
                    return False
        except (KeyError, TypeError, ValueError):
            return False
        self._cycle_number, self._cursor, self._pending_indices = cycle, cursor, pending
        for entry, presentations, alt_cursor in zip(self._entries, counts, cursors, strict=True):
            entry.presentation_count = max(0, presentations)
            if entry.alternate:
                entry.alternate.cursor = max(0, alt_cursor)
        return True

    def seek_after(self, screen_id: str) -> bool:
        """Continue after *screen_id*'s slot in the first cycle, if it has one.

        Used when a new playlist still contains the screen being shown, so a
        configuration update does not restart playback from the top.
        """

        if not self._entries:
            return False
        if self._pending_indices is None:
            self._queue_current_cycle(datetime.now(UTC))
        pending = self._pending_indices or []
        for position, index in enumerate(pending):
            if self._entries[index].screen_id == screen_id:
                del pending[: position + 1]
                self._cursor = (index + 1) % len(self._entries)
                return True
        return False

    def next_available(self, registry: dict[str, ScreenDefinition]) -> Optional[ScreenDefinition]:
        """Return the next available definition from one ordered queued cycle.

        The method never mixes a later cycle into the currently queued cycle.
        Rebuilding the scheduler, as config reload does, restarts at cycle 1.
        """

        if not self._entries:
            return None

        now_utc = datetime.now(UTC)
        if self._pending_indices is None:
            self._queue_current_cycle(now_utc)
        if not self._pending_indices and not self._advance_until_pending(now_utc):
            return None

        # Drain exactly one queued cycle. Unavailable screens remain in their
        # configured slots rather than causing a second cycle to be mixed in.
        assert self._pending_indices is not None
        pending_count = len(self._pending_indices)
        for _ in range(pending_count):
            entry_index = self._pending_indices.pop(0)
            self._cursor = (entry_index + 1) % len(self._entries)
            entry = self._entries[entry_index]
            if entry.hide_after is not None and now_utc >= entry.hide_after:
                continue
            definition = self._next_available_from_entry(
                entry,
                registry,
                force_base=self._cycle_number == 1,
            )
            if definition is not None:
                return definition

        return None


def load_schedule_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Schedule configuration must be a JSON object")
    return data


def sanitize_schedule_config(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return a copy of *config* with unknown screens stripped.

    The display service should continue running even if stale screen IDs are
    present in persisted JSON from older releases.
    """

    if not isinstance(config, dict):
        return config, []

    screens = config.get("screens")
    if not isinstance(screens, dict):
        return dict(config), []

    sanitized = dict(config)
    cleaned_screens: dict[str, Any] = {}
    removed: list[str] = []

    for screen_id, raw in screens.items():
        canonical_id = canonical_screen_id(screen_id) if isinstance(screen_id, str) else screen_id
        if not isinstance(canonical_id, str) or canonical_id not in KNOWN_SCREENS:
            removed.append(str(screen_id))
            continue

        if not isinstance(raw, dict):
            existing = cleaned_screens.get(canonical_id)
            if isinstance(existing, dict):
                continue
            if isinstance(existing, int):
                try:
                    cleaned_screens[canonical_id] = max(existing, int(raw))
                except Exception:
                    cleaned_screens[canonical_id] = existing
            else:
                cleaned_screens[canonical_id] = raw
            continue

        cleaned_raw = dict(raw)
        alt_spec = cleaned_raw.get("alt")
        if isinstance(alt_spec, dict):
            alt_screen_value = alt_spec.get("screen")
            if isinstance(alt_screen_value, str):
                mapped_alt = canonical_screen_id(alt_screen_value)
                if mapped_alt not in KNOWN_SCREENS:
                    cleaned_raw.pop("alt", None)
                    removed.append(f"{screen_id}.alt:{alt_screen_value}")
                else:
                    cleaned_alt = dict(alt_spec)
                    cleaned_alt["screen"] = mapped_alt
                    cleaned_raw["alt"] = cleaned_alt
            elif isinstance(alt_screen_value, list):
                known_alt_screens: list[str] = []
                for alt in alt_screen_value:
                    if not isinstance(alt, str):
                        continue
                    mapped_alt = canonical_screen_id(alt)
                    if mapped_alt in KNOWN_SCREENS and mapped_alt not in known_alt_screens:
                        known_alt_screens.append(mapped_alt)
                if not known_alt_screens:
                    cleaned_raw.pop("alt", None)
                    removed.append(f"{screen_id}.alt")
                else:
                    cleaned_alt = dict(alt_spec)
                    cleaned_alt["screen"] = known_alt_screens
                    cleaned_raw["alt"] = cleaned_alt

        existing_raw = cleaned_screens.get(canonical_id)
        if isinstance(existing_raw, dict):
            existing_freq = existing_raw.get("frequency")
            new_freq = cleaned_raw.get("frequency")
            with contextlib.suppress(Exception):
                existing_raw["frequency"] = max(int(existing_freq), int(new_freq))
            existing_extra = existing_raw.get("extra_seconds")
            new_extra = cleaned_raw.get("extra_seconds")
            with contextlib.suppress(Exception):
                existing_raw["extra_seconds"] = max(int(existing_extra or 0), int(new_extra or 0))
            if "alt" not in existing_raw and "alt" in cleaned_raw:
                existing_raw["alt"] = cleaned_raw["alt"]
            cleaned_screens[canonical_id] = existing_raw
        else:
            cleaned_screens[canonical_id] = cleaned_raw

    sanitized["screens"] = cleaned_screens

    playlists = config.get("playlists")
    if isinstance(playlists, dict):
        cleaned_playlists = dict(playlists)
        for playlist_id, playlist in playlists.items():
            if not isinstance(playlist, dict) or not isinstance(playlist.get("steps"), list):
                continue

            cleaned_steps: list[Any] = []
            seen_step_screens: set[str] = set()
            for step in playlist["steps"]:
                if not isinstance(step, dict) or not isinstance(step.get("screen"), str):
                    cleaned_steps.append(step)
                    continue

                cleaned_step = dict(step)
                canonical_id = canonical_screen_id(step["screen"])
                if canonical_id in seen_step_screens:
                    continue
                cleaned_step["screen"] = canonical_id
                cleaned_steps.append(cleaned_step)
                seen_step_screens.add(canonical_id)

            cleaned_playlist = dict(playlist)
            cleaned_playlist["steps"] = cleaned_steps
            cleaned_playlists[playlist_id] = cleaned_playlist
        sanitized["playlists"] = cleaned_playlists

    return sanitized, removed


def build_scheduler(config: dict[str, Any]) -> ScreenScheduler:
    """Build a scheduler at cycle 1 in saved playlist/config-page order.

    Frequency values are interpreted directly: zero removes the independent
    base slot, while positive ``N`` means cycles ``1``, ``1 + N``, ``1 + 2N``,
    and so on. A zero-frequency screen ID remains
    valid as an alternate referenced by another enabled entry.
    """

    if not isinstance(config, dict):
        raise ValueError("Schedule configuration must be a JSON object")

    config, _ = sanitize_schedule_config(config)

    screens = config.get("screens")
    if not isinstance(screens, dict) or not screens:
        raise ValueError("Configuration must provide a non-empty 'screens' mapping")

    ordered_screens: list[tuple[str, Any]] = []
    seen_screen_ids: set[str] = set()

    playlists = config.get("playlists")
    sequence = config.get("sequence")
    if isinstance(playlists, dict):
        ordered_playlist_ids: list[str] = []
        if isinstance(sequence, list):
            for item in sequence:
                if not isinstance(item, dict):
                    continue
                playlist_id = item.get("playlist")
                if (
                    isinstance(playlist_id, str)
                    and playlist_id
                    and playlist_id in playlists
                    and playlist_id not in ordered_playlist_ids
                ):
                    ordered_playlist_ids.append(playlist_id)

        for playlist_id in playlists:
            if (
                isinstance(playlist_id, str)
                and playlist_id
                and playlist_id not in ordered_playlist_ids
            ):
                ordered_playlist_ids.append(playlist_id)

        playlist_assignments: dict[str, str] = {}
        for playlist_id in ordered_playlist_ids:
            playlist = playlists.get(playlist_id)
            if not isinstance(playlist, dict):
                continue
            steps = playlist.get("steps")
            if not isinstance(steps, list):
                continue
            for step in steps:
                if not isinstance(step, dict):
                    continue
                screen_id = step.get("screen")
                if (
                    isinstance(screen_id, str)
                    and screen_id in screens
                    and screen_id not in playlist_assignments
                ):
                    playlist_assignments[screen_id] = playlist_id

        # Match the Config page exactly: its first group is Ungrouped, followed
        # by playlists in sequence order. Within each group, drag-and-drop order
        # is the insertion order of the saved screens mapping.
        group_ids = ["", *ordered_playlist_ids]
        for group_id in group_ids:
            for screen_id, raw in screens.items():
                if playlist_assignments.get(screen_id, "") != group_id:
                    continue
                if screen_id in seen_screen_ids:
                    continue
                ordered_screens.append((screen_id, raw))
                seen_screen_ids.add(screen_id)

    for screen_id, raw in screens.items():
        if screen_id in seen_screen_ids:
            continue
        ordered_screens.append((screen_id, raw))

    entries: list[_ScheduleEntry] = []
    for raw_screen_id, raw in ordered_screens:
        canonical_id = canonical_screen_id(raw_screen_id)
        if not isinstance(canonical_id, str):
            raise ValueError("Screen identifiers must be strings")
        if canonical_id not in KNOWN_SCREENS:
            raise ValueError(f"Unknown screen id '{canonical_id}'")
        screen_id = canonical_id
        alternate: Optional[_AlternateSchedule] = None
        hide_after: Optional[datetime] = None

        if isinstance(raw, dict):
            if "frequency" not in raw:
                raise ValueError(f"Frequency for '{screen_id}' must be provided")
            try:
                frequency = int(raw["frequency"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Frequency for '{screen_id}' must be an integer") from exc
            try:
                extra_seconds = int(raw.get("extra_seconds", 0))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Additional seconds for '{screen_id}' must be an integer"
                ) from exc
            if extra_seconds < 0:
                raise ValueError(f"Additional seconds for '{screen_id}' cannot be negative")
            hide_after_enabled = bool(raw.get("hide_after_enabled", False))
            hide_after_raw = raw.get("hide_after_at")
            if hide_after_enabled:
                if not isinstance(hide_after_raw, str) or not hide_after_raw.strip():
                    raise ValueError(
                        f"Hide-after date/time for '{screen_id}' must be provided when enabled"
                    )
                try:
                    hide_after_value = datetime.fromisoformat(hide_after_raw.strip())
                except ValueError as exc:
                    raise ValueError(
                        f"Hide-after date/time for '{screen_id}' must be a valid ISO "
                        "date/time string"
                    ) from exc
                if hide_after_value.tzinfo is None:
                    hide_after_value = hide_after_value.replace(tzinfo=CENTRAL_TIME)
                hide_after = hide_after_value.astimezone(UTC)

            alt_spec = raw.get("alt")
            if alt_spec is not None:
                if not isinstance(alt_spec, dict):
                    raise ValueError(f"Alternate configuration for '{screen_id}' must be an object")
                alt_screen_value = alt_spec.get("screen")
                alt_frequency = alt_spec.get("frequency")

                if isinstance(alt_screen_value, str):
                    alt_screen_ids = [canonical_screen_id(alt_screen_value)]
                elif isinstance(alt_screen_value, (list, tuple)):
                    alt_screen_ids = []
                    for alt_item in alt_screen_value:
                        if not isinstance(alt_item, str):
                            raise ValueError(
                                f"Alternate screen ids for '{screen_id}' must be strings"
                            )
                        alt_screen_ids.append(canonical_screen_id(alt_item))
                else:
                    raise ValueError(
                        f"Alternate screen id for '{screen_id}' must be a string or list of strings"
                    )

                if not alt_screen_ids:
                    raise ValueError(f"Alternate screen list for '{screen_id}' cannot be empty")

                for alt_screen in alt_screen_ids:
                    if alt_screen not in KNOWN_SCREENS:
                        raise ValueError(
                            f"Unknown alternate screen id '{alt_screen}' for '{screen_id}'"
                        )

                try:
                    alt_frequency_int = int(alt_frequency)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Alternate frequency for '{screen_id}' must be an integer"
                    ) from exc
                if alt_frequency_int <= 0:
                    raise ValueError(
                        f"Alternate frequency for '{screen_id}' must be greater than zero"
                    )
                alternate = _AlternateSchedule(tuple(alt_screen_ids), alt_frequency_int)
        else:
            try:
                frequency = int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Frequency for '{screen_id}' must be an integer") from exc
            extra_seconds = 0

        if frequency < 0:
            raise ValueError(f"Frequency for '{screen_id}' cannot be negative")

        if screen_id in REPLACEMENT_ONLY_SCREENS:
            # These screens are rendered through their team's configured "next"
            # slot so existing local configs that contain both IDs do not get an
            # extra standalone no-game slot.
            continue

        if frequency == 0:
            # A frequency of zero disables the screen.  This allows playlists to
            # keep entries around for future use without removing them from the
            # configuration file while ensuring they never appear in the
            # rotation.
            continue

        entries.append(
            _ScheduleEntry(
                screen_id,
                frequency,
                extra_seconds=extra_seconds,
                hide_after=hide_after,
                alternate=alternate,
            )
        )

    if not entries:
        raise ValueError("Configuration must contain at least one enabled screen")

    return ScreenScheduler(entries)
