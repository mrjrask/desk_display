import time
from datetime import UTC, datetime

import pytest

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


def test_sanitize_schedule_config_migrates_removed_adsb_screen():
    sanitized, removed = sanitize_schedule_config(
        {
            "screens": {"adsb live airlines": 1},
            "playlists": {
                "other": {
                    "steps": [
                        {"screen": "date"},
                        {"screen": "adsb live airlines"},
                        {"screen": "adsb live"},
                    ]
                }
            },
        }
    )

    assert removed == []
    assert sanitized["screens"] == {"adsb live": 1}
    assert sanitized["playlists"]["other"]["steps"] == [
        {"screen": "date"},
        {"screen": "adsb live"},
    ]


def test_build_scheduler_keeps_migrated_adsb_screen_in_playlist_position():
    scheduler = build_scheduler(
        {
            "screens": {"date": 1, "adsb live airlines": 1, "inside": 1},
            "playlists": {
                "other": {
                    "steps": [
                        {"screen": "date"},
                        {"screen": "adsb live airlines"},
                        {"screen": "inside"},
                    ]
                }
            },
        }
    )

    assert scheduler.requested_ids == {"date", "adsb live", "inside"}
    assert scheduler.preview_scheduled_ids(3) == ["date", "adsb live", "inside"]


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
        "date",
        "inside",
        "date",
        "inside",
        "date",
    ]


def test_frequency_one_alternate_is_used_on_first_normal_presentation():
    config = {
        "screens": {
            "NFL Overview NFC": {
                "frequency": 4,
                "alt": {"screen": "NFL Standings NFC", "frequency": 1},
            }
        }
    }
    scheduler = build_scheduler(config)
    registry = make_registry(
        {"NFL Overview NFC": True, "NFL Standings NFC": True}
    )

    # Startup always hydrates the base screen without advancing normal state.
    assert scheduler.preview_scheduled_ids(1) == ["NFL Overview NFC"]
    assert scheduler.next_available(registry).id == "NFL Overview NFC"

    # Frequency four is still measured from the first normal pass, and the
    # alternate is selected when that entry receives its first presentation.
    assert collect_sequence(scheduler, registry, 4) == [
        None,
        None,
        None,
        "NFL Standings NFC",
    ]


def test_unavailable_frequency_one_alternate_falls_back_on_first_presentation():
    config = {
        "screens": {
            "NFL Overview NFC": {
                "frequency": 4,
                "alt": {"screen": "NFL Standings NFC", "frequency": 1},
            }
        }
    }
    scheduler = build_scheduler(config)
    registry = make_registry(
        {"NFL Overview NFC": True, "NFL Standings NFC": False}
    )

    assert scheduler.next_available(registry).id == "NFL Overview NFC"


def test_alternate_frequency_counts_scheduled_appearances_not_raw_passes():
    config = {
        "screens": {
            "NFL Overview NFC": {
                "frequency": 4,
                "alt": {"screen": "NFL Standings NFC", "frequency": 3},
            },
            "NFL Overview AFC": {
                "frequency": 4,
                "alt": {"screen": "NFL Standings AFC", "frequency": 3},
            },
        }
    }
    scheduler = build_scheduler(config)
    registry = make_registry(
        {
            "NFL Overview NFC": True,
            "NFL Overview AFC": True,
            "NFL Standings NFC": True,
            "NFL Standings AFC": True,
        }
    )

    # The second entry becomes due in the same scheduler pass as the first, but
    # is returned on the following call from the pending queue.  Observe that
    # call so both third presentations are included without re-advancing a new
    # pass after the cursor wraps.
    sequence = collect_played_ids(scheduler, registry, 17)

    assert sequence == [
        "NFL Overview NFC",
        "NFL Overview AFC",
        "NFL Overview NFC",
        "NFL Overview AFC",
        "NFL Overview NFC",
        "NFL Overview AFC",
        "NFL Standings NFC",
        "NFL Standings AFC",
    ]


def test_multiple_alternate_entries_do_not_starve_later_screens():
    config = {
        "screens": {
            "date": {
                "frequency": 1,
                "alt": {"screen": "nixie", "frequency": 1},
            },
            "inside": {
                "frequency": 1,
                "alt": {"screen": "weather2", "frequency": 1},
            },
            "weather1": 1,
        }
    }
    scheduler = build_scheduler(config)
    registry = make_registry(
        {
            "date": True,
            "nixie": True,
            "inside": True,
            "weather2": True,
            "weather1": True,
        }
    )

    assert scheduler.preview_scheduled_ids(6) == [
        "date",
        "inside",
        "weather1",
        "nixie",
        "weather2",
        "weather1",
    ]
    assert collect_sequence(scheduler, registry, 6) == [
        "date",
        "inside",
        "weather1",
        "nixie",
        "weather2",
        "weather1",
    ]


def test_queued_entry_is_skipped_after_hide_after_deadline(monkeypatch):
    scheduler = build_scheduler(
        {
            "screens": {
                "date": {
                    "frequency": 1,
                    "alt": {"screen": "nixie", "frequency": 2},
                },
                "inside": {
                    "frequency": 1,
                    "hide_after_enabled": True,
                    "hide_after_at": "2026-04-06T00:01+00:00",
                    "alt": {"screen": "weather2", "frequency": 2},
                },
                "weather1": 1,
            }
        }
    )
    registry = make_registry(
        {
            "date": True,
            "nixie": True,
            "inside": True,
            "weather2": True,
            "weather1": True,
        }
    )

    class _BeforeDeadline:
        @staticmethod
        def now(tz=None):
            return datetime(2026, 4, 6, 0, 0, tzinfo=UTC)

    class _AfterDeadline:
        @staticmethod
        def now(tz=None):
            return datetime(2026, 4, 6, 0, 2, tzinfo=UTC)

    monkeypatch.setattr("schedule.datetime", _BeforeDeadline)
    assert collect_sequence(scheduler, registry, 4) == [
        "date",
        "inside",
        "weather1",
        "date",
    ]

    monkeypatch.setattr("schedule.datetime", _AfterDeadline)
    assert scheduler.next_available(registry).id == "weather1"


def test_queued_entry_tries_all_alternates_before_base_fallback():
    scheduler = build_scheduler(
        {
            "screens": {
                "date": {
                    "frequency": 1,
                    "alt": {"screen": "nixie", "frequency": 2},
                },
                "inside": {
                    "frequency": 1,
                    "alt": {
                        "screen": ["weather2", "weather1"],
                        "frequency": 2,
                    },
                },
                "on this day": 1,
            }
        }
    )
    registry = make_registry(
        {
            "date": True,
            "nixie": True,
            "inside": True,
            "weather2": False,
            "weather1": True,
            "on this day": True,
        }
    )

    assert collect_sequence(scheduler, registry, 5) == [
        "date",
        "inside",
        "on this day",
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
        "date",
        "inside",
        "date",
        "weather1",
        "date",
    ]


def test_alt_screen_ids_for_returns_configured_alternates():
    config = {
        "screens": {
            "date": {
                "frequency": 1,
                "alt": {"screen": "nixie", "frequency": 3},
            },
            "nixie": 0,
        }
    }
    scheduler = build_scheduler(config)
    assert scheduler.alt_screen_ids_for("date") == ("nixie",)
    assert scheduler.alt_screen_ids_for("nixie") == ()
    assert scheduler.alt_screen_ids_for("missing") == ()


def test_alt_screen_ids_for_multiple_alternates():
    config = {
        "screens": {
            "date": {
                "frequency": 1,
                "alt": {"screen": ["inside", "weather1"], "frequency": 2},
            }
        }
    }
    scheduler = build_scheduler(config)
    assert scheduler.alt_screen_ids_for("date") == ("inside", "weather1")


def test_alt_screen_ids_for_without_alt_config():
    config = {"screens": {"date": 1, "inside": 2}}
    scheduler = build_scheduler(config)
    assert scheduler.alt_screen_ids_for("date") == ()


def test_build_scheduler_rejects_unknown_screen():
    config = {"screens": {"missing": 1}}
    with pytest.raises(ValueError):
        build_scheduler(config)


def test_scheduler_respects_frequency():
    config = {"screens": {"date": 1, "inside": 2}}
    scheduler = build_scheduler(config)
    registry = make_registry({"date": True, "inside": True})

    sequence = collect_sequence(scheduler, registry, 6)
    assert sequence == ["date", "inside", "date", "date", "inside", "date"]


def test_scheduler_frequency_interval_matches_configuration():
    config = {"screens": {"date": 1, "inside": 4}}
    scheduler = build_scheduler(config)
    registry = make_registry({"date": True, "inside": True})

    sequence = collect_sequence(scheduler, registry, 12)
    # Startup hydration is separate; normal rotation then begins with pass 1.
    # ``inside`` appears once during startup, then on every fourth normal pass.
    assert sequence == [
        "date",
        "inside",
        "date",
        "date",
        "date",
        "date",
        "inside",
        "date",
        "date",
        "date",
        "date",
        "inside",
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
    # The Config page renders Ungrouped first, then playlists in sequence order.
    assert sequence == ["weather1", "date", "inside", "weather1", "date", "inside"]


def test_scheduler_hydrates_each_frequency_pass_in_config_page_order():
    config = {
        "screens": {"inside": 2, "weather1": 1, "date": 1},
        "playlists": {
            "second": {"steps": [{"screen": "inside"}]},
            "first": {"steps": [{"screen": "date"}]},
        },
        "sequence": [{"playlist": "first"}, {"playlist": "second"}],
    }
    scheduler = build_scheduler(config)
    registry = make_registry({"date": True, "inside": True, "weather1": True})

    assert collect_sequence(scheduler, registry, 8) == [
        "weather1",
        "date",
        "inside",
        "weather1",
        "date",
        "weather1",
        "date",
        "inside",
    ]


def test_scheduler_jumps_across_empty_normal_passes():
    scheduler = build_scheduler({"screens": {"date": 1_000_000_000}})
    registry = make_registry({"date": True})

    assert scheduler.next_available(registry).id == "date"
    assert scheduler.next_available(registry).id == "date"
    assert scheduler._pass_number == 1_000_000_000


def test_scheduler_uses_first_playlist_assignment_for_duplicate_screen():
    config = {
        "screens": {"inside": 1, "date": 1, "weather1": 1},
        "playlists": {
            "first": {"steps": [{"screen": "date"}]},
            "second": {
                "steps": [{"screen": "date"}, {"screen": "inside"}],
            },
        },
        "sequence": [{"playlist": "first"}, {"playlist": "second"}],
    }
    scheduler = build_scheduler(config)
    registry = make_registry({"date": True, "inside": True, "weather1": True})

    assert collect_sequence(scheduler, registry, 3) == ["weather1", "date", "inside"]


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
            return datetime(2026, 4, 6, 0, 0, tzinfo=UTC)

    monkeypatch.setattr("schedule.datetime", _FutureDateTime)
    sequence = collect_sequence(scheduler, registry, 4)
    assert sequence == ["inside", "inside", "inside", "inside"]


def test_naive_hide_after_uses_central_time_regardless_of_host_timezone(monkeypatch):
    if not hasattr(time, "tzset"):
        pytest.skip("Changing the process timezone is not supported on this platform")

    with monkeypatch.context() as host_timezone:
        host_timezone.setenv("TZ", "Pacific/Honolulu")
        time.tzset()
        scheduler = build_scheduler(
            {
                "screens": {
                    "date": {
                        "frequency": 1,
                        "hide_after_enabled": True,
                        "hide_after_at": "2026-07-01T12:00",
                    }
                }
            }
        )
    time.tzset()

    assert scheduler._entries[0].hide_after == datetime(2026, 7, 1, 17, 0, tzinfo=UTC)


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


def test_mlb_no_game_entries_do_not_add_standalone_schedule_slots():
    scheduler = build_scheduler(
        {
            "screens": {
                "cubs no game": 1,
                "cubs next": 1,
                "sox no game": 1,
                "sox next": 1,
            }
        }
    )

    assert scheduler.node_count == 2
    assert scheduler.requested_ids == {"cubs next", "sox next"}

    registry = make_registry({"cubs next": True, "sox next": True})
    assert collect_sequence(scheduler, registry, 4) == [
        "cubs next",
        "sox next",
        "cubs next",
        "sox next",
    ]
