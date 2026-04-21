import pytest
from datetime import datetime, timezone

from schedule import KNOWN_SCREENS, build_scheduler, sanitize_schedule_config
from screens.registry import ScreenDefinition


def make_registry(availability):
    return {
        sid: ScreenDefinition(id=sid, render=lambda sid=sid: sid, available=available)
        for sid, available in availability.items()
    }


def collect_sequence(scheduler, registry, length):
    results = []
    for _ in range(length):
        definition = scheduler.next_available(registry)
        results.append(definition.id if definition is not None else None)
    return results


def collect_played_ids(scheduler, registry, iterations):
    results = []
    for _ in range(iterations):
        definition = scheduler.next_available(registry)
        if definition is not None:
            results.append(definition.id)
    return results


def test_build_scheduler_from_config():
    config = {
        "screens": {
            "date": 1,
            "inside": 2,
            "weather1": 1,
        }
    }
    scheduler = build_scheduler(config)
    assert scheduler.node_count == 3
    assert scheduler.requested_ids == {"date", "inside", "weather1"}


def test_sensors_screen_is_known():
    assert "weather1" in KNOWN_SCREENS


def test_travel_screen_is_not_known():
    assert "travel" not in KNOWN_SCREENS


def test_travel_v2_screen_is_not_known():
    assert "travel v2" not in KNOWN_SCREENS

def test_travel_map_screen_is_not_known():
    assert "travel map" not in KNOWN_SCREENS

def test_travel_map_v2_screen_is_not_known():
    assert "travel map v2" not in KNOWN_SCREENS


def test_sanitize_schedule_config_canonicalizes_legacy_screen_ids():
    sanitized, removed = sanitize_schedule_config(
        {
            "screens": {
                "time": 1,
                "sensors": {"frequency": 2},
            }
        }
    )

    assert removed == []
    assert sanitized["screens"] == {"nixie": 1, "inside": {"frequency": 2}}


def test_build_scheduler_accepts_legacy_screen_ids():
    scheduler = build_scheduler({"screens": {"time": 1, "sensors": 1}})

    assert scheduler.requested_ids == {"nixie", "inside"}


def test_scheduler_with_alternate_screen():
    config = {
        "screens": {
            "date": {
                "frequency": 1,
                "alt": {"screen": "inside", "frequency": 2},
            }
        }
    }
    scheduler = build_scheduler(config)
    assert scheduler.requested_ids == {"date", "inside"}

    registry = make_registry({"date": True, "inside": True})
    sequence = collect_sequence(scheduler, registry, 6)
    assert sequence == [
        "date",
        "inside",
        "date",
        "inside",
        "date",
        "inside",
    ]


def test_scheduler_with_multiple_alternates():
    config = {
        "screens": {
            "date": {
                "frequency": 1,
                "alt": {"screen": ["inside", "weather1"], "frequency": 2},
            }
        }
    }

    scheduler = build_scheduler(config)
    assert scheduler.requested_ids == {"date", "inside", "weather1"}

    registry = make_registry({"date": True, "inside": True, "weather1": True})
    sequence = collect_sequence(scheduler, registry, 6)
    assert sequence == [
        "date",
        "inside",
        "date",
        "weather1",
        "date",
        "inside",
    ]


def test_build_scheduler_rejects_unknown_screen():
    config = {"screens": {"missing": 1}}
    with pytest.raises(ValueError):
        build_scheduler(config)


def test_scheduler_respects_frequency():
    config = {"screens": {"date": 1, "inside": 2}}
    scheduler = build_scheduler(config)
    registry = make_registry({"date": True, "inside": True})

    sequence = collect_sequence(scheduler, registry, 6)
    assert sequence == ["date", "inside", "date", "inside", "date", "date"]


def test_scheduler_frequency_interval_matches_configuration():
    config = {"screens": {"date": 1, "inside": 4}}
    scheduler = build_scheduler(config)
    registry = make_registry({"date": True, "inside": True})

    sequence = collect_sequence(scheduler, registry, 12)
    # ``inside`` appears once during the initial loop, then on every fourth pass.
    assert sequence == [
        "date",
        "inside",
        "date",
        "date",
        "date",
        "inside",
        "date",
        "date",
        "date",
        "date",
        "inside",
        "date",
    ]


def test_scheduler_skips_unavailable_screen():
    config = {"screens": {"inside": 1}}
    scheduler = build_scheduler(config)
    registry = make_registry({"inside": False})
    assert scheduler.next_available(registry) is None



def test_scheduler_respects_playlist_sequence_order():
    config = {
        "screens": {"inside": 1, "date": 1, "weather1": 1},
        "playlists": {
            "second": {"steps": [{"screen": "inside"}]},
            "first": {"steps": [{"screen": "date"}]},
        },
        "sequence": [{"playlist": "first"}, {"playlist": "second"}],
    }

    scheduler = build_scheduler(config)
    registry = make_registry({"date": True, "inside": True, "weather1": True})

    sequence = collect_sequence(scheduler, registry, 6)
    assert sequence == ["date", "inside", "weather1", "date", "inside", "weather1"]

def test_invalid_configuration_shapes():
    with pytest.raises(ValueError):
        build_scheduler({})
    with pytest.raises(ValueError):
        build_scheduler({"screens": []})
    with pytest.raises(ValueError):
        build_scheduler({"screens": {"date": -1}})
    with pytest.raises(ValueError):
        build_scheduler({"screens": {"date": "oops"}})
    with pytest.raises(ValueError):
        build_scheduler(
            {"screens": {"date": {"frequency": 1, "alt": {"screen": "inside"}}}}
        )
    with pytest.raises(ValueError):
        build_scheduler(
            {
                "screens": {
                    "date": {
                        "frequency": 1,
                        "alt": {"screen": "inside", "frequency": 0},
                    }
                }
            }
        )


def test_zero_frequency_entries_are_skipped():
    config = {"screens": {"date": 0, "nixie": 2}}
    scheduler = build_scheduler(config)
    registry = make_registry({"date": True, "nixie": True})

    played = collect_played_ids(scheduler, registry, 6)
    assert played
    assert set(played) == {"nixie"}


def test_all_zero_frequencies_raise_error():
    config = {"screens": {"date": 0, "nixie": 0}}

    with pytest.raises(ValueError):
        build_scheduler(config)


def test_preview_scheduled_ids_matches_next_sequence_without_mutation():
    config = {"screens": {"date": 1, "inside": 2}}
    scheduler = build_scheduler(config)
    registry = make_registry({"date": True, "inside": True})

    preview = scheduler.preview_scheduled_ids(6)
    actual = collect_sequence(scheduler, registry, 6)

    assert preview == actual


def test_preview_scheduled_ids_keeps_scheduler_state():
    config = {
        "screens": {
            "date": {
                "frequency": 1,
                "alt": {"screen": ["inside", "weather1"], "frequency": 2},
            }
        }
    }
    scheduler = build_scheduler(config)

    first_preview = scheduler.preview_scheduled_ids(4)
    second_preview = scheduler.preview_scheduled_ids(4)

    assert first_preview == second_preview


def test_scheduler_tracks_extra_seconds_per_screen():
    scheduler = build_scheduler(
        {
            "screens": {
                "date": {"frequency": 1, "extra_seconds": 4},
                "inside": 2,
            }
        }
    )

    assert scheduler.extra_seconds_for("date") == 4
    assert scheduler.extra_seconds_for("inside") == 0


def test_scheduler_rejects_negative_extra_seconds():
    with pytest.raises(ValueError):
        build_scheduler({"screens": {"date": {"frequency": 1, "extra_seconds": -1}}})


def test_scheduler_skips_screen_after_hide_after_datetime(monkeypatch):
    scheduler = build_scheduler(
        {
            "screens": {
                "date": {
                    "frequency": 1,
                    "hide_after_enabled": True,
                    "hide_after_at": "2020-01-01T00:00",
                },
                "inside": 1,
            }
        }
    )
    registry = make_registry({"date": True, "inside": True})

    class _FutureDateTime:
        @staticmethod
        def now(tz=None):
            return datetime(2026, 4, 6, 0, 0, tzinfo=timezone.utc)

    monkeypatch.setattr("schedule.datetime", _FutureDateTime)
    sequence = collect_sequence(scheduler, registry, 4)
    assert sequence == ["inside", "inside", "inside", "inside"]


def test_scheduler_rejects_hide_after_when_enabled_without_datetime():
    with pytest.raises(ValueError):
        build_scheduler(
            {
                "screens": {
                    "date": {
                        "frequency": 1,
                        "hide_after_enabled": True,
                    }
                }
            }
        )
