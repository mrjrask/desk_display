from types import SimpleNamespace

import pytest

from runtime import LegacyStandaloneRuntime


@pytest.mark.parametrize(
    ("screen_id", "expected"),
    [
        ("NFL Overview NFC", ("nfl", None)),
        ("NHL Standings East", ("nhl", False)),
        ("NHL Standings West v2", ("nhl", True)),
        ("MLB AL Standings", ("mlb", None)),
        ("NL Overview", ("mlb", None)),
        ("AL Overview", ("mlb", None)),
        ("NL Overview+WC", ("mlb", None)),
        ("AL Overview+WC", ("mlb", None)),
    ],
)
def test_standalone_runtime_hydrates_standings_before_snapshot(screen_id, expected):
    events = []

    class Data:
        def read_nfl_league_standings(self):
            events.append(("nfl", None))

        def read_nhl_league_standings(self, *, include_wildcard_order):
            events.append(("nhl", include_wildcard_order))

        def read_mlb_league_standings(self):
            events.append(("mlb", None))

        def snapshot(self):
            events.append(("snapshot", None))
            return "snapshot"

    artifact = SimpleNamespace()
    runtime = LegacyStandaloneRuntime(
        data=Data(),
        renderer=SimpleNamespace(render=lambda *_args: artifact),
        player=SimpleNamespace(next=lambda: SimpleNamespace(screen_id=screen_id)),
        presenter=SimpleNamespace(profile=object(), present=lambda _artifact: None),
        preferences=SimpleNamespace(),
    )

    assert runtime.step() is artifact
    assert events == [expected, ("snapshot", None)]


def test_standalone_runtime_does_not_hydrate_unrelated_screen():
    events = []
    data = SimpleNamespace(snapshot=lambda: events.append("snapshot") or "snapshot")
    runtime = LegacyStandaloneRuntime(
        data=data,
        renderer=SimpleNamespace(render=lambda *_args: SimpleNamespace()),
        player=SimpleNamespace(next=lambda: SimpleNamespace(screen_id="date_time")),
        presenter=SimpleNamespace(profile=object(), present=lambda _artifact: None),
        preferences=SimpleNamespace(),
    )

    runtime.step()

    assert events == ["snapshot"]
