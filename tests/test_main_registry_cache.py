"""Tests for main registry cache invalidation."""

import importlib
import sys


def _load_main():
    sys.modules.pop("main", None)
    return importlib.import_module("main")


def test_registry_cache_rebuilds_when_cache_revision_changes(monkeypatch):
    main = _load_main()

    calls = []

    def _fake_build_registry(_context):
        calls.append(True)
        return {}, {}

    monkeypatch.setattr(main, "build_screen_registry", _fake_build_registry)

    weather = {}
    context = main.ScreenContext(
        display=None,
        cache={"weather": weather},
        logos=None,
        image_dir=".",
        now=main.datetime.datetime.now(main.CENTRAL_TIME),
        now_utc=main.datetime.datetime.now(main.datetime.timezone.utc),
        offline=False,
        weather_fetched_at=None,
        skip_scoreboards=False,
    )

    main._registry_cache_key = None
    main._registry_cache_value = None
    main._cache_revision = 0

    main._build_registry_if_needed(context)
    main._build_registry_if_needed(context)

    assert len(calls) == 1

    main._cache_revision = 1
    main._build_registry_if_needed(context)

    assert len(calls) == 2
