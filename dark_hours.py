"""Dark-hours schedules: parsing ``DARK_HOURS`` and testing a moment against it.

Shared by the standalone display (``config``) and the thin display client,
so both read the same schedule the same way.  This module has no project
imports, so the client can use it without loading the rendering stack.
"""
from __future__ import annotations

import datetime
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

MINUTES_PER_DAY = 24 * 60

_DAY_NAME_TO_INDEX = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tues": 1,
    "tuesday": 1,
    "wed": 2,
    "weds": 2,
    "wednesday": 2,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}


def _parse_time_token(token: str) -> int:
    cleaned = token.strip()
    if not cleaned:
        raise ValueError("Empty time token")

    lowered = cleaned.lower()
    if lowered in {"midnight"}:
        return 0
    if lowered in {"noon"}:
        return 12 * 60
    if lowered in {"24:00", "24", "24h", "24hr", "24hrs"}:
        return MINUTES_PER_DAY

    for fmt in ("%H:%M", "%H", "%I:%M%p", "%I%p", "%I:%M %p", "%I %p"):
        try:
            parsed = datetime.datetime.strptime(cleaned.upper(), fmt)
        except ValueError:
            continue
        return parsed.hour * 60 + parsed.minute

    raise ValueError(f"Unrecognized time token '{token}'")


def _expand_day_spec(spec: str) -> list[int]:
    days: list[int] = []
    seen = set()
    for part in spec.split(","):
        piece = part.strip()
        if not piece:
            continue

        if "-" in piece:
            start_text, end_text = piece.split("-", 1)
            start_name = start_text.strip().lower()
            end_name = end_text.strip().lower()
            if start_name not in _DAY_NAME_TO_INDEX or end_name not in _DAY_NAME_TO_INDEX:
                raise ValueError(f"Unknown day name in range '{piece}'")
            start_idx = _DAY_NAME_TO_INDEX[start_name]
            end_idx = _DAY_NAME_TO_INDEX[end_name]
            idx = start_idx
            while True:
                if idx not in seen:
                    days.append(idx)
                    seen.add(idx)
                if idx == end_idx:
                    break
                idx = (idx + 1) % 7
        else:
            name = piece.lower()
            if name not in _DAY_NAME_TO_INDEX:
                raise ValueError(f"Unknown day name '{piece}'")
            idx = _DAY_NAME_TO_INDEX[name]
            if idx not in seen:
                days.append(idx)
                seen.add(idx)
    return days


@dataclass(frozen=True)
class DarkHoursSegment:
    weekday: int
    start_minute: int
    end_minute: int


def _parse_dark_hours_spec(raw_value: Optional[str]) -> tuple[DarkHoursSegment, ...]:
    if not raw_value:
        return ()

    entries = []
    for chunk in re.split(r"[;\n]+", raw_value):
        if not chunk:
            continue

        normalized = re.sub(r"\s*-\s*", "-", chunk.strip())
        if not normalized:
            continue

        parts = normalized.split(None, 1)
        if len(parts) != 2:
            logging.warning("Ignoring dark-hours entry '%s' (missing time range)", chunk)
            continue

        day_spec, time_spec = parts[0], parts[1].strip()
        if not day_spec:
            logging.warning("Ignoring dark-hours entry '%s' (missing day spec)", chunk)
            continue

        if not time_spec:
            logging.warning("Ignoring dark-hours entry '%s' (missing time spec)", chunk)
            continue

        try:
            days = _expand_day_spec(day_spec)
        except ValueError as exc:
            logging.warning("Ignoring dark-hours entry '%s': %s", chunk, exc)
            continue

        if not days:
            logging.warning("Ignoring dark-hours entry '%s' (no valid days)", chunk)
            continue

        normalized_time = time_spec.lower().replace(" ", "")
        if normalized_time in {"off", "allday", "all-day", "alldaylong"}:
            start_minutes = 0
            end_minutes = MINUTES_PER_DAY
        else:
            if "-" not in time_spec:
                logging.warning(
                    "Ignoring dark-hours entry '%s' (missing start/end times)", chunk
                )
                continue
            start_text, end_text = time_spec.split("-", 1)
            try:
                start_minutes = _parse_time_token(start_text)
                end_minutes = _parse_time_token(end_text)
            except ValueError as exc:
                logging.warning("Ignoring dark-hours entry '%s': %s", chunk, exc)
                continue

        for day in days:
            if start_minutes == end_minutes:
                entries.append(
                    DarkHoursSegment(day, 0, MINUTES_PER_DAY)
                )
                continue
            if start_minutes < end_minutes:
                entries.append(DarkHoursSegment(day, start_minutes, end_minutes))
            else:
                entries.append(DarkHoursSegment(day, start_minutes, MINUTES_PER_DAY))
                next_day = (day + 1) % 7
                entries.append(DarkHoursSegment(next_day, 0, end_minutes))

    return tuple(entries)


def segments_contain(
    segments: Iterable[DarkHoursSegment],
    moment: datetime.datetime,
    zone: datetime.tzinfo,
) -> bool:
    """Whether *moment*, read as wall-clock time in *zone*, falls in a segment."""

    current = moment.replace(tzinfo=zone) if moment.tzinfo is None else moment.astimezone(zone)
    weekday = current.weekday()
    minute_of_day = current.hour * 60 + current.minute
    return any(
        segment.weekday == weekday and segment.start_minute <= minute_of_day < segment.end_minute
        for segment in segments
    )
