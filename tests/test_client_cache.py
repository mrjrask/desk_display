"""Tests for the client's revisioned playlist cache and playback state."""
from __future__ import annotations

import copy
import json
import os

import pytest

from remote_display import client_cache
from remote_display.client_cache import (
    ClientCache,
    PlaybackState,
    PlaylistRejected,
    playback_order,
    reconcile_playback,
    validate_playlist,
)
from remote_display.models import AcceptedRevisions
from remote_display.playlist_store import document_revision

DOC_A = {"screens": {"date": 1, "weather1": 1, "cubs last": 1}, "sequence": []}
DOC_B = {"screens": {"date": 1, "news headlines": 1}, "sequence": []}


def payload(document, playlist_id="default", **overrides):
    data = {
        "playlist_id": playlist_id,
        "playlist_revision": document_revision(document),
        "playlist_schema_version": 2,
        "document": copy.deepcopy(document),
    }
    data.update(overrides)
    return data


def everything(_screen):
    return True


@pytest.fixture
def cache(tmp_path):
    return ClientCache(tmp_path / "cache")


# ── Validation ──────────────────────────────────────────────────────────────


def test_valid_playlist_is_read_only(cache):
    playlist = cache.offer(payload(DOC_A), artifact_usable=everything)
    assert playlist.screens == ("cubs last", "date", "weather1")
    with pytest.raises(TypeError):
        playlist.document["screens"]["date"] = 5


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"playlist_schema_version": 99}, "unsupported_schema"),
        ({"playlist_revision": "r-00000000000000000000"}, "revision_mismatch"),
        ({"document": {"screens": {"no such screen": 1}}}, "invalid_playlist"),
        ({"document": {"screens": {"date": 0}}}, "invalid_playlist"),
        ({"playlist_id": "../etc"}, "invalid_playlist"),
    ],
)
def test_invalid_playlists_are_rejected(cache, change, code):
    body = payload(DOC_A)
    body.update(change)
    if "document" in change and "playlist_revision" not in change:
        body["playlist_revision"] = "r-00000000000000000000"
    with pytest.raises(PlaylistRejected) as excinfo:
        cache.offer(body, artifact_usable=everything)
    assert excinfo.value.code in {code, "revision_mismatch", "empty_playlist"}
    assert cache.load() is None


def test_empty_playlist_is_rejected():
    document = {"screens": {"date": 0}, "sequence": []}
    with pytest.raises(PlaylistRejected) as excinfo:
        validate_playlist(payload(document))
    assert excinfo.value.code in {"empty_playlist", "invalid_playlist"}


# ── Updates and activation ──────────────────────────────────────────────────


def test_update_keeps_previous_and_acknowledges(cache):
    assert cache.accepted_revisions() == AcceptedRevisions()
    first = cache.offer(payload(DOC_A), artifact_usable=everything, manifest_revision="m-1")
    second = cache.offer(payload(DOC_B), artifact_usable=everything, manifest_revision="m-2")
    assert cache.load() == second
    assert cache.previous() == first
    assert cache.accepted_revisions() == AcceptedRevisions(
        playlist_revision=second.playlist_revision, manifest_revision="m-2"
    )


def test_activation_waits_for_required_artifacts(cache):
    first = cache.offer(payload(DOC_A), artifact_usable=everything)
    with pytest.raises(PlaylistRejected) as excinfo:
        cache.offer(payload(DOC_B), artifact_usable=lambda screen: screen != "news headlines")
    assert excinfo.value.code == "artifacts_unavailable"
    assert "news headlines" in excinfo.value.message
    assert cache.load() == first
    assert cache.accepted_revisions().playlist_revision == first.playlist_revision


def test_rejected_update_keeps_current(cache):
    first = cache.offer(payload(DOC_A), artifact_usable=everything)
    with pytest.raises(PlaylistRejected):
        cache.offer(payload(DOC_B, playlist_revision="r-11111111111111111111"), artifact_usable=everything)
    assert cache.load() == first
    assert cache.previous() is None


def test_reoffering_the_same_revision_is_a_no_op(cache):
    first = cache.offer(payload(DOC_A), artifact_usable=everything)
    cache.offer(payload(DOC_A), artifact_usable=everything)
    assert cache.previous() is None
    assert cache.load() == first


# ── Offline boot, corruption and atomicity ──────────────────────────────────


def test_offline_boot_uses_the_cached_playlist(cache, tmp_path):
    first = cache.offer(payload(DOC_A), artifact_usable=everything)
    assert ClientCache(tmp_path / "cache").load() == first


def test_corrupt_current_falls_back_to_previous(cache, tmp_path):
    first = cache.offer(payload(DOC_A), artifact_usable=everything)
    cache.offer(payload(DOC_B), artifact_usable=everything)
    (tmp_path / "cache" / "playlist" / "current.json").write_text("{not json", encoding="utf-8")
    assert ClientCache(tmp_path / "cache").load() == first


def test_tampered_current_falls_back_to_previous(cache, tmp_path):
    first = cache.offer(payload(DOC_A), artifact_usable=everything)
    cache.offer(payload(DOC_B), artifact_usable=everything)
    current = tmp_path / "cache" / "playlist" / "current.json"
    data = json.loads(current.read_text(encoding="utf-8"))
    data["document"]["screens"]["weather1"] = 1  # content no longer matches its revision
    current.write_text(json.dumps(data), encoding="utf-8")
    assert cache.load() == first


def test_empty_cache_boots_with_nothing(cache):
    assert cache.load() is None


def test_failed_write_leaves_current_intact(cache, tmp_path, monkeypatch):
    first = cache.offer(payload(DOC_A), artifact_usable=everything)
    real_replace = os.replace

    def crash(src, dst):
        if str(dst).endswith("current.json"):
            raise OSError("power lost")
        return real_replace(src, dst)

    monkeypatch.setattr(client_cache.os, "replace", crash)
    with pytest.raises(OSError):
        cache.offer(payload(DOC_B), artifact_usable=everything)
    monkeypatch.undo()
    assert cache.load() == first
    assert list((tmp_path / "cache" / "staging").iterdir()) == []


# ── Emergency override ──────────────────────────────────────────────────────


def test_override_is_separate_and_visible(cache):
    server = cache.offer(payload(DOC_A), artifact_usable=everything)
    override = cache.set_override(DOC_B, reason="server down")
    assert override.source == "override"
    assert cache.load() == override
    assert cache.server_playlist() == server
    assert cache.accepted_revisions().playlist_revision == server.playlist_revision
    cache.clear_override()
    assert cache.load() == server


# ── Playback state ──────────────────────────────────────────────────────────


def test_playback_state_survives_restart(cache, tmp_path):
    state = PlaybackState(playlist_id="default", playlist_revision="r-1", sequence_index=2, step_index=1,
                          current_screen="weather1", hold={"screen": "weather1", "until": 5.0},
                          focus_return="date")
    for screen in ("date", "weather1"):
        state.remember(screen)
    cache.save_playback(state)
    assert ClientCache(tmp_path / "cache").load_playback() == state


def test_corrupt_playback_state_resets(cache, tmp_path):
    (tmp_path / "cache").mkdir(parents=True)
    (tmp_path / "cache" / "playback.json").write_text("[]", encoding="utf-8")
    assert cache.load_playback() == PlaybackState()


def test_history_is_bounded():
    state = PlaybackState()
    for index in range(client_cache.MAX_HISTORY + 10):
        state.remember(f"s{index}")
    assert len(state.history) == client_cache.MAX_HISTORY
    assert state.history[-1] == f"s{client_cache.MAX_HISTORY + 9}"


def test_reconcile_keeps_a_screen_that_is_still_valid(cache):
    old = cache.offer(payload(DOC_A), artifact_usable=everything)
    state = PlaybackState(playlist_id=old.playlist_id, playlist_revision=old.playlist_revision,
                          sequence_index=1, current_screen="date",
                          hold={"screen": "date"}, focus_return="weather1", history=["weather1", "date"])
    new = cache.offer(payload(DOC_B), artifact_usable=everything)
    result = reconcile_playback(state, new)
    assert result.current_screen == "date"
    assert result.hold == {"screen": "date"}
    assert result.history == ["date"]
    assert result.focus_return is None
    assert (result.playlist_id, result.playlist_revision) == (new.playlist_id, new.playlist_revision)
    assert result.sequence_index == 0


def test_reconcile_moves_off_a_removed_screen_deterministically(cache):
    old = cache.offer(payload(DOC_A), artifact_usable=everything)
    state = PlaybackState(playlist_id=old.playlist_id, playlist_revision=old.playlist_revision,
                          current_screen="cubs last", hold={"screen": "cubs last"}, history=["cubs last"])
    new = cache.offer(payload(DOC_B), artifact_usable=everything)
    result = reconcile_playback(state, new)
    assert result.current_screen == new.screens[0] == "date"
    assert result.hold is None and result.history == []
    assert reconcile_playback(state, new) == result


def test_reconcile_on_the_same_revision_keeps_position(cache):
    playlist = cache.offer(payload(DOC_A), artifact_usable=everything)
    state = PlaybackState(playlist_id=playlist.playlist_id, playlist_revision=playlist.playlist_revision,
                          sequence_index=3, step_index=2, current_screen="weather1", history=["date"])
    assert reconcile_playback(state, playlist) == state


def test_playback_order_follows_sequence_then_screens():
    document = {
        "screens": {"date": 1, "weather1": 0},
        "playlists": {"morning": {"steps": [{"screen": "news headlines"}, {"screen": "date"}]}},
        "sequence": [{"playlist": "morning"}],
    }
    assert playback_order(document) == ("news headlines", "date")


# ── Server config response ──────────────────────────────────────────────────


def test_server_config_carries_the_assigned_playlist(tmp_path):
    pytest.importorskip("flask")
    import display_server
    from display_profiles import PROFILE_PRESETS
    from remote_display.playlist_store import PlaylistStore

    store_file = tmp_path / "playlists.json"
    store = PlaylistStore(store_file)
    created = store.create("Office", DOC_A, actor="test")
    store.assign("office", created["id"], expected_playlist_id=None, actor="test")
    token = "server-token-" + "s" * 32
    config = display_server.DisplayServerConfig(
        auth_token=token, admin_token="admin-token-" + "a" * 32,
        artifact_dir=tmp_path / "artifacts", playlist_store_path=store_file,
    )
    api = display_server.create_app(config).test_client()
    preset = PROFILE_PRESETS["hyperpixel4"]
    caps = {
        "type": "client_capabilities", "version": 1, "protocol_version": 1, "client_software_version": "0.1",
        "client_id": "office", "display_profile": "hyperpixel4", "logical_width": preset.width,
        "logical_height": preset.height, "image_formats": ["PNG"], "color_modes": [preset.color_mode],
        "render_package_versions": [1],
    }
    response = api.post("/api/v1/register", json={"capabilities": caps}, headers={"Authorization": f"Bearer {token}"})
    credential = response.get_json()["client_credential"]
    body = api.get("/api/v1/clients/office/config", headers={"Authorization": f"Bearer {credential}"}).get_json()
    assert body["playlist"]["playlist_revision"] == created["revision"]
    assert token not in json.dumps(body) and credential not in json.dumps(body)

    cache = ClientCache(tmp_path / "client")
    playlist = cache.offer(body["playlist"], artifact_usable=everything)
    assert playlist.screens == ("cubs last", "date", "weather1")

    store.assign("office", None, expected_playlist_id=created["id"], actor="test")
    body = api.get("/api/v1/clients/office/config", headers={"Authorization": f"Bearer {credential}"}).get_json()
    assert body["playlist"] is None
