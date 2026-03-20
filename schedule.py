"""Simple frequency-based screen scheduler."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple

from screens_catalog import SCREEN_IDS, canonical_screen_id

if TYPE_CHECKING:
    from screens.registry import ScreenDefinition


KNOWN_SCREENS: Set[str] = set(SCREEN_IDS)

# Backfill any new screens that may have been added out-of-band but not yet
# reflected in :mod:`screens_catalog`.  This keeps scheduler validation in sync
# with the screen registry so that deployments pulling the updated screen module
# before the catalog list do not fail to load the schedule configuration.
KNOWN_SCREENS.add("sensors")


@dataclass
class _AlternateSchedule:
    screen_ids: tuple[str, ...]
    frequency: int
    cursor: int = 0

    def next_screen_id(self) -> str:
        if not self.screen_ids:
            raise ValueError("Alternate schedule requires at least one screen id")

        screen_id = self.screen_ids[self.cursor]
        self.cursor = (self.cursor + 1) % len(self.screen_ids)
        return screen_id


@dataclass
class _ScheduleEntry:
    screen_id: str
    frequency: int
    cycle_count: int = 0
    alternate: Optional[_AlternateSchedule] = None


class ScreenScheduler:
    """Iterator that yields the next available screen based on frequencies."""

    def __init__(self, entries: Sequence[_ScheduleEntry]):
        self._entries: List[_ScheduleEntry] = list(entries)
        self._cursor: int = 0
        requested: Set[str] = set()
        for entry in self._entries:
            requested.add(entry.screen_id)
            if entry.alternate is not None:
                requested.update(entry.alternate.screen_ids)
        self._requested = requested

    @property
    def node_count(self) -> int:
        return len(self._entries)

    @property
    def requested_ids(self) -> Set[str]:
        return set(self._requested)

    def preview_scheduled_ids(self, limit: int) -> List[str]:
        """Return upcoming scheduled screen IDs without mutating scheduler state."""

        if limit <= 0 or not self._entries:
            return []

        cloned_entries: List[_ScheduleEntry] = []
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
                    cycle_count=entry.cycle_count,
                    alternate=cloned_alt,
                )
            )

        preview = ScreenScheduler(cloned_entries)
        preview._cursor = self._cursor

        scheduled_ids: List[str] = []
        for _ in range(limit):
            next_id = preview._next_scheduled_id()
            if next_id is None:
                break
            scheduled_ids.append(next_id)

        return scheduled_ids

    def _next_scheduled_id(self) -> Optional[str]:
        """Return the next scheduled ID without availability checks."""

        if not self._entries:
            return None

        for _ in range(len(self._entries)):
            entry = self._entries[self._cursor]
            self._cursor = (self._cursor + 1) % len(self._entries)

            entry.cycle_count += 1
            if (entry.cycle_count - 1) % entry.frequency != 0:
                continue

            if entry.alternate and entry.alternate.frequency > 0:
                if entry.cycle_count % entry.alternate.frequency == 0:
                    return entry.alternate.next_screen_id()

            return entry.screen_id

        return None

    def next_available(self, registry: Dict[str, "ScreenDefinition"]) -> Optional["ScreenDefinition"]:
        if not self._entries:
            return None

        for _ in range(len(self._entries)):
            entry = self._entries[self._cursor]
            self._cursor = (self._cursor + 1) % len(self._entries)

            # A frequency of ``n`` means the screen is shown once every
            # ``n`` scheduler passes for that entry.
            entry.cycle_count += 1
            if (entry.cycle_count - 1) % entry.frequency != 0:
                continue

            candidate_id = entry.screen_id
            if entry.alternate and entry.alternate.frequency > 0:
                if entry.cycle_count % entry.alternate.frequency == 0:
                    alternate = entry.alternate
                    for _ in range(len(alternate.screen_ids)):
                        alt_id = alternate.next_screen_id()
                        alt_def = registry.get(alt_id)
                        if alt_def and alt_def.available:
                            return alt_def

            definition = registry.get(candidate_id)
            if definition and definition.available:
                return definition

        return None


def load_schedule_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Schedule configuration must be a JSON object")
    return data


def sanitize_schedule_config(config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
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
    cleaned_screens: Dict[str, Any] = {}
    removed: List[str] = []

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
                known_alt_screens: List[str] = []
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
            try:
                existing_raw["frequency"] = max(int(existing_freq), int(new_freq))
            except Exception:
                pass
            if "alt" not in existing_raw and "alt" in cleaned_raw:
                existing_raw["alt"] = cleaned_raw["alt"]
            cleaned_screens[canonical_id] = existing_raw
        else:
            cleaned_screens[canonical_id] = cleaned_raw

    sanitized["screens"] = cleaned_screens
    return sanitized, removed


def build_scheduler(config: Dict[str, Any]) -> ScreenScheduler:
    if not isinstance(config, dict):
        raise ValueError("Schedule configuration must be a JSON object")

    config, _ = sanitize_schedule_config(config)

    screens = config.get("screens")
    if not isinstance(screens, dict) or not screens:
        raise ValueError("Configuration must provide a non-empty 'screens' mapping")

    ordered_screens: List[Tuple[str, Any]] = []
    seen_screen_ids: Set[str] = set()

    playlists = config.get("playlists")
    sequence = config.get("sequence")
    if isinstance(playlists, dict):
        ordered_playlist_ids: List[str] = []
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

        for playlist_id in playlists.keys():
            if isinstance(playlist_id, str) and playlist_id and playlist_id not in ordered_playlist_ids:
                ordered_playlist_ids.append(playlist_id)

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
                    and screen_id not in seen_screen_ids
                ):
                    ordered_screens.append((screen_id, screens[screen_id]))
                    seen_screen_ids.add(screen_id)

    for screen_id, raw in screens.items():
        if screen_id in seen_screen_ids:
            continue
        ordered_screens.append((screen_id, raw))

    entries: List[_ScheduleEntry] = []
    for screen_id, raw in ordered_screens:
        screen_id = canonical_screen_id(screen_id)
        if not isinstance(screen_id, str):
            raise ValueError("Screen identifiers must be strings")
        if screen_id not in KNOWN_SCREENS:
            raise ValueError(f"Unknown screen id '{screen_id}'")
        alternate: Optional[_AlternateSchedule] = None

        if isinstance(raw, dict):
            if "frequency" not in raw:
                raise ValueError(f"Frequency for '{screen_id}' must be provided")
            try:
                frequency = int(raw["frequency"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Frequency for '{screen_id}' must be an integer"
                ) from exc

            alt_spec = raw.get("alt")
            if alt_spec is not None:
                if not isinstance(alt_spec, dict):
                    raise ValueError(
                        f"Alternate configuration for '{screen_id}' must be an object"
                    )
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
                    raise ValueError(
                        f"Alternate screen list for '{screen_id}' cannot be empty"
                    )

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
                alternate = _AlternateSchedule(
                    tuple(alt_screen_ids), alt_frequency_int
                )
        else:
            try:
                frequency = int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Frequency for '{screen_id}' must be an integer") from exc

        if frequency < 0:
            raise ValueError(f"Frequency for '{screen_id}' cannot be negative")

        if frequency == 0:
            # A frequency of zero disables the screen.  This allows playlists to
            # keep entries around for future use without removing them from the
            # configuration file while ensuring they never appear in the
            # rotation.
            continue

        entries.append(_ScheduleEntry(screen_id, frequency, alternate=alternate))

    if not entries:
        raise ValueError("Configuration must contain at least one enabled screen")

    return ScreenScheduler(entries)
