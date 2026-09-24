"""Frequency-based screen scheduling with a distinct startup hydration phase.

Each positive-frequency base entry is emitted once during startup hydration,
without advancing normal-pass or alternate-presentation counters.  Thereafter,
frequency ``N`` is due on normal passes ``N``, ``2N``, ``3N``, and so on, while
an alternate's frequency counts only the due presentations of its base entry.
Frequency-zero entries have no independent slot, although their screen IDs may
still be referenced as alternates.  Entry order is preserved in hydration and
normal passes, and constructing a new scheduler (including after a config
reload) starts a new hydration phase.
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

    ``pass_number`` is ``None`` for startup hydration.  Normal rotation passes
    are numbered starting at one; gaps in their numbers identify passes where
    no configured entry was due, even though those empty passes produce no
    result of their own.
    """

    screen_id: str
    phase: Literal["startup", "normal"]
    pass_number: Optional[int]


class ScreenScheduler:
    """Yield screens in ordered hydration and frequency-based normal passes.

    Instantiation begins a fresh startup hydration phase.  Hydration queues
    every positive-frequency base once in configuration order without counting
    a normal pass or an alternate presentation.  Normal frequency ``N`` then
    selects passes ``N``, ``2N``, ``3N``, etc.; alternate frequency is measured
    against those due presentations of the individual base entry.
    """

    def __init__(self, entries: Sequence[_ScheduleEntry]):
        self._entries: list[_ScheduleEntry] = list(entries)
        self._cursor: int = 0
        self._startup_indices: list[int] = [
            index for index, entry in enumerate(self._entries) if entry.frequency >= 1
        ]
        self._pending_indices: list[int] = []
        self._pass_number: int = 0
        self._startup_hydrated: bool = False
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
        """Return upcoming IDs annotated with startup/normal pass information.

        This pass-aware form is intended for diagnostics and scheduler tests.
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
        preview._startup_indices = self._startup_indices.copy()
        preview._pending_indices = self._pending_indices.copy()
        preview._pass_number = self._pass_number
        preview._startup_hydrated = self._startup_hydrated

        scheduled_entries: list[ScheduledPreviewEntry] = []
        for _ in range(limit):
            next_id = preview._next_scheduled_id()
            if next_id is None:
                break
            normal_pass_number = preview._pass_number or None
            scheduled_entries.append(
                ScheduledPreviewEntry(
                    screen_id=next_id,
                    phase="normal" if normal_pass_number is not None else "startup",
                    pass_number=normal_pass_number,
                )
            )

        return scheduled_entries

    def _next_scheduled_id(self) -> Optional[str]:
        """Return the next ordered ID without availability checks.

        Startup returns only base IDs and leaves presentation counters alone;
        normal passes may resolve a due base entry to its alternate.
        """

        if not self._entries:
            return None

        if not self._pending_indices and not self._hydrate_until_pending(datetime.now(UTC)):
            return None

        entry_index = self._pending_indices.pop(0)
        self._cursor = (entry_index + 1) % len(self._entries)
        return self._scheduled_id_for(
            self._entries[entry_index],
            advance_presentation=self._pass_number > 0,
        )

    def _scheduled_id_for(
        self,
        entry: _ScheduleEntry,
        *,
        advance_presentation: bool = True,
    ) -> str:
        """Resolve a due entry and count only normal passes as presentations.

        ``advance_presentation`` is false during startup hydration, ensuring it
        neither selects an alternate nor changes when one will next be due.
        """

        if not advance_presentation:
            return entry.screen_id

        entry.presentation_count += 1
        if (
            entry.alternate
            and entry.alternate.frequency > 0
            and entry.presentation_count % entry.alternate.frequency == 0
        ):
            return entry.alternate.next_screen_id()
        return entry.screen_id

    def _hydrate_next_pass(self, now_utc: datetime) -> None:
        """Queue every due entry for one complete, configuration-ordered pass.

        Startup hydration queues each positive-frequency base exactly once in
        configuration order and is a separate initial traversal, not ``pass 1``.
        Normal pass numbering begins with pass 1 after startup hydration has
        completed, and the scheduler-wide pass counter advances once here for
        each such pass.

        Building the whole pass before returning its first screen prevents later
        calls from interleaving entries from different frequency passes.  The
        queue is also retained while registry data is refreshed, so rebuilding
        screen definitions cannot alter the configured playback order.
        """

        self._pending_indices.clear()
        startup_hydration = not self._startup_hydrated
        if startup_hydration:
            self._startup_hydrated = True
        else:
            self._pass_number += 1

        for index, entry in enumerate(self._entries):
            if entry.frequency == 0:
                continue
            if entry.hide_after is not None and now_utc >= entry.hide_after:
                continue

            if startup_hydration or self._pass_number % entry.frequency == 0:
                self._pending_indices.append(index)

        self._cursor = 0

    def _hydrate_until_pending(self, now_utc: datetime) -> bool:
        """Queue the next nonempty ordered pass, skipping empty pass ranges."""

        self._hydrate_next_pass(now_utc)
        if self._pending_indices:
            return True

        active_frequencies = [
            entry.frequency
            for entry in self._entries
            if entry.frequency > 0 and (entry.hide_after is None or now_utc < entry.hide_after)
        ]
        if not active_frequencies:
            return False

        # The pass just hydrated was empty. Jump to immediately before the
        # nearest future multiple rather than scanning every intervening pass;
        # _hydrate_next_pass() remains the single place that increments the
        # scheduler-wide counter and queues the selected pass.
        next_due_pass = min(
            (self._pass_number // frequency + 1) * frequency for frequency in active_frequencies
        )
        self._pass_number = next_due_pass - 1
        self._hydrate_next_pass(now_utc)
        return bool(self._pending_indices)

    def _next_available_from_entry(
        self,
        entry: _ScheduleEntry,
        registry: dict[str, ScreenDefinition],
        *,
        advance_presentation: bool = True,
    ) -> Optional[ScreenDefinition]:
        """Resolve a queued entry, counting alternates only on normal passes."""

        if advance_presentation:
            entry.presentation_count += 1
        if (
            advance_presentation
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

    def next_available(self, registry: dict[str, ScreenDefinition]) -> Optional[ScreenDefinition]:
        """Return the next available definition from one ordered queued pass.

        The method never mixes a later pass into the currently queued pass.
        Rebuilding the scheduler, as config reload does, restarts hydration.
        """

        if not self._entries:
            return None

        now_utc = datetime.now(UTC)
        if not self._pending_indices and not self._hydrate_until_pending(now_utc):
            return None

        # Drain exactly one hydrated pass. Unavailable screens remain in their
        # configured slots rather than causing a second pass to be mixed in.
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
                advance_presentation=self._pass_number > 0,
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
    """Build a freshly hydrating scheduler in saved playlist/config-page order.

    Frequency values are interpreted directly: zero removes the independent
    base slot, while positive ``N`` means normal passes ``N``, ``2N``, and so
    on after one startup hydration display.  A zero-frequency screen ID remains
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
