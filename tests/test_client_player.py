"""Tests for cached-content playback state."""

import pytest

from playback.client_player import ClientPlayer, IncompatibleManifestError
from protocol import build_manifest
from protocol_versions import MANIFEST_SCHEMA_VERSION, RENDER_PACKAGE_SCHEMA_VERSION
from schedule import ScreenScheduler


def _player() -> ClientPlayer:
    return ClientPlayer(ScreenScheduler([]), default_duration=10.0)


def test_load_cache_installs_supported_manifest():
    player = _player()
    manifest = build_manifest(cache_complete=True)

    player.load_cache(manifest, {"clock": {"duration": 5}}, {"clock": b"pkg"})

    assert player.manifest == manifest
    assert player.playlist == {"clock": {"duration": 5}}
    assert player.packages == {"clock": b"pkg"}


@pytest.mark.parametrize(
    "overrides",
    [
        {"manifest_schema_version": MANIFEST_SCHEMA_VERSION + 1},
        {"render_package_schema_version": RENDER_PACKAGE_SCHEMA_VERSION + 1},
        {"manifest_schema_version": None},
        {"render_package_schema_version": "1"},
    ],
)
def test_load_cache_rejects_unsupported_schema_and_keeps_active_cache(overrides):
    player = _player()
    good = build_manifest()
    player.load_cache(good, {"clock": {}}, {"clock": b"old"})

    with pytest.raises(IncompatibleManifestError):
        player.load_cache({**good, **overrides}, {"news": {}}, {"news": b"new"})

    assert player.manifest == good
    assert player.playlist == {"clock": {}}
    assert player.packages == {"clock": b"old"}


def test_due_honors_shorter_focus_deadline(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("playback.client_player.time.monotonic", lambda: clock[0])
    player = _player()
    player.load_cache(build_manifest(), {"clock": {"duration": 10}}, {"clock": b"pkg"})

    player.focus("clock", seconds=2)
    assert player.next().screen_id == "clock"

    assert player.due(now=101.0) is False
    assert player.due(now=102.0) is True


def test_due_uses_item_duration_without_focus_deadline(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("playback.client_player.time.monotonic", lambda: clock[0])
    player = _player()
    player.load_cache(build_manifest(), {"clock": {"duration": 10}}, {"clock": b"pkg"})

    player.focus("clock")
    player.next()

    assert player.due(now=105.0) is False
    assert player.due(now=110.0) is True
