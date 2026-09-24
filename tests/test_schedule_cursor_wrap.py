from schedule import build_scheduler
from screens.registry import ScreenDefinition


def _make_registry(availability):
    return {
        screen_id: ScreenDefinition(
            id=screen_id,
            render=lambda screen_id=screen_id: screen_id,
            available=available,
        )
        for screen_id, available in availability.items()
    }


def test_final_entry_wrap_does_not_readvance_completed_pass():
    scheduler = build_scheduler(
        {
            "screens": {
                "date": {
                    "frequency": 1,
                    "alt": {"screen": "nixie", "frequency": 1},
                },
                "inside": {
                    "frequency": 1,
                    "alt": {"screen": "weather2", "frequency": 1},
                },
                "weather1": 2,
            }
        }
    )
    registry = _make_registry(
        {
            "date": True,
            "nixie": True,
            "inside": True,
            "weather2": True,
            "weather1": True,
        }
    )

    assert [scheduler.next_available(registry).id for _ in range(3)] == [
        "nixie",
        "weather2",
        "weather1",
    ]

    # Selecting the final entry wraps the cursor to zero. That marks the end of
    # the current pass, so synchronization must not advance or queue any entry
    # from the pass that just completed.
    assert scheduler._cursor == 0
    assert scheduler._pending_indices == []
    assert scheduler._pass_number == 0
