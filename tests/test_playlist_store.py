"""Tests for the server-managed playlist library and assignments."""
from __future__ import annotations

import json
import threading

import pytest

from remote_display import playlist_store as ps
from remote_display.playlist_store import (
    ConflictError,
    InUseError,
    NotFoundError,
    PlaylistStore,
    PlaylistValidationError,
)

DOC = {
    "screens": {
        "date": 1,
        "weather1": {"frequency": 2, "extra_seconds": 5, "alt": {"screen": "weather2", "frequency": 3}},
        "weather2": 0,
        "inside": 1,
    },
    "playlists": {
        "p-core": {"label": "core", "steps": [{"screen": "date"}, {"screen": "weather1"}]},
        "p-indoor": {"label": "indoor", "steps": [{"screen": "inside"}]},
    },
    "sequence": [{"playlist": "p-core"}, {"playlist": "p-indoor"}],
}


@pytest.fixture
def store(tmp_path):
    return PlaylistStore(tmp_path / "server" / "playlists.json")


def test_create_assigns_stable_id_and_content_revision(store):
    a = store.create("Kitchen", DOC, actor="jason")
    assert a["id"].startswith("pl-") and a["revision"].startswith("r-")
    b = store.create("Office", json.loads(json.dumps(DOC)), actor="jason")
    assert a["id"] != b["id"]
    assert a["revision"] == b["revision"]  # identical content, identical revision
    renamed = store.rename(a["id"], "Kitchen display", expected_revision=a["revision"], actor="jason")
    assert renamed["id"] == a["id"] and renamed["revision"] == a["revision"]


def test_document_semantics_are_preserved(store):
    saved = store.create("Kitchen", DOC, actor="jason")["document"]
    assert saved["screens"]["weather1"] == DOC["screens"]["weather1"]
    assert saved["sequence"] == DOC["sequence"]
    assert ps.document_screens(saved) == (("date", "inside", "weather1"), ("weather2",))


def test_legacy_ids_are_canonicalized(store):
    doc = {"screens": {"time": 1, "sensors": 1}}
    saved = store.create("Legacy", doc, actor="jason")["document"]
    assert set(saved["screens"]) == {"nixie", "inside"}


@pytest.mark.parametrize(
    "document, field",
    [
        ({"screens": {"bogus": 1}}, "document.screens['bogus']"),
        ({"screens": {"date": {"alt": {"screen": "bogus"}}}}, "document.screens['date'].alt.screen"),
        ({"screens": {"date": 1}, "playlists": {"p": {"steps": [{"screen": "nope"}]}}}, "document.playlists['p'].steps[0]"),
        ({"screens": {"date": 1}, "secret": "x"}, "document.secret"),
        ({"screens": []}, "document.screens"),
        ("not a doc", "document"),
        ({"screens": {"date": 1, "time": 1, "nixie": 1}}, "document.screens"),
    ],
)
def test_invalid_documents_are_rejected(store, document, field):
    with pytest.raises(PlaylistValidationError) as info:
        store.create("Bad", document, actor="jason")
    assert info.value.details["field"] == field
    assert store.snapshot()["playlists"] == {}


def test_scheduler_rejections_are_reported(store, monkeypatch):
    import schedule

    def boom(config):
        raise ValueError("sequence references missing playlist")

    monkeypatch.setattr(schedule, "build_scheduler", boom)
    with pytest.raises(PlaylistValidationError, match="missing playlist"):
        store.create("Bad", DOC, actor="jason")


def test_optimistic_concurrency(store):
    playlist = store.create("Kitchen", DOC, actor="jason")
    edited = json.loads(json.dumps(DOC))
    edited["screens"]["date"] = 2
    first = store.update(playlist["id"], edited, expected_revision=playlist["revision"], actor="alice")
    assert first["revision"] != playlist["revision"]
    stale = json.loads(json.dumps(DOC))
    stale["screens"]["inside"] = 3
    with pytest.raises(ConflictError) as info:
        store.update(playlist["id"], stale, expected_revision=playlist["revision"], actor="bob")
    assert info.value.details["current_revision"] == first["revision"]
    assert store.get(playlist["id"])["document"]["screens"]["date"] == 2
    with pytest.raises(ConflictError):
        store.rename(playlist["id"], "x", expected_revision=playlist["revision"], actor="bob")
    with pytest.raises(ConflictError):
        store.delete(playlist["id"], expected_revision=None, actor="bob")


def test_reorder(store):
    playlist = store.create("Kitchen", DOC, actor="jason")
    reordered = store.reorder(playlist["id"], [1, 0], expected_revision=playlist["revision"], actor="jason")
    assert reordered["document"]["sequence"] == [{"playlist": "p-indoor"}, {"playlist": "p-core"}]
    for order in ([0, 0], [0], [0, 1, 2], ["0", "1"], None):
        with pytest.raises(PlaylistValidationError):
            store.reorder(playlist["id"], order, expected_revision=reordered["revision"], actor="jason")


def test_assignment_is_single_per_client_and_shareable(store):
    a = store.create("A", DOC, actor="jason")
    b = store.create("B", {"screens": {"date": 1}}, actor="jason")
    store.assign("office", a["id"], expected_playlist_id=None, actor="jason")
    store.assign("den", a["id"], expected_playlist_id=None, actor="jason")
    assert store.clients_using(a["id"]) == ["den", "office"]
    store.assign("office", b["id"], expected_playlist_id=a["id"], actor="jason")
    assert store.clients_using(a["id"]) == ["den"] and store.clients_using(b["id"]) == ["office"]
    assert store.snapshot()["assignments"]["office"]["playlist_id"] == b["id"]
    with pytest.raises(ConflictError):
        store.assign("office", a["id"], expected_playlist_id=a["id"], actor="bob")
    with pytest.raises(NotFoundError):
        store.assign("office", "pl-missing", expected_playlist_id=b["id"], actor="jason")
    with pytest.raises(PlaylistValidationError):
        store.assign("../x", a["id"], expected_playlist_id=None, actor="jason")
    store.assign("office", None, expected_playlist_id=b["id"], actor="jason")
    assert store.assignment_for("office") is None
    lookup = store.assignment_for("den")
    assert lookup.playlist_id == a["id"] and lookup.playlist_revision == a["revision"]
    assert lookup.screens == ("date", "inside", "weather1") and lookup.alternates == ("weather2",)


def test_guarded_delete(store):
    playlist = store.create("A", DOC, actor="jason")
    store.assign("office", playlist["id"], expected_playlist_id=None, actor="jason")
    with pytest.raises(InUseError) as info:
        store.delete(playlist["id"], expected_revision=playlist["revision"], actor="jason")
    assert info.value.details["clients"] == ["office"]
    store.assign("office", None, expected_playlist_id=playlist["id"], actor="jason")
    store.delete(playlist["id"], expected_revision=playlist["revision"], actor="jason")
    with pytest.raises(NotFoundError):
        store.get(playlist["id"])


def test_clone_and_fork_for_client(store):
    shared = store.create("Shared", DOC, actor="jason")
    store.assign("office", shared["id"], expected_playlist_id=None, actor="jason")
    store.assign("den", shared["id"], expected_playlist_id=None, actor="jason")
    fork = store.fork_for_client("office", expected_playlist_id=shared["id"], actor="jason")
    assert fork["id"] != shared["id"] and fork["document"] == shared["document"]
    assert store.clients_using(shared["id"]) == ["den"]
    assert store.clients_using(fork["id"]) == ["office"]
    edited = json.loads(json.dumps(DOC))
    edited["screens"]["date"] = 5
    store.update(fork["id"], edited, expected_revision=fork["revision"], actor="jason")
    assert store.get(shared["id"])["document"]["screens"]["date"] == 1
    with pytest.raises(ConflictError):
        store.fork_for_client("den", expected_playlist_id=fork["id"], actor="jason")
    # The failed fork left no orphaned copy behind.
    assert len(store.snapshot()["playlists"]) == 2


def test_export_and_import_exclude_credentials(store, monkeypatch):
    monkeypatch.setenv("OWM_API_KEY", "owm-secret-value-123")
    doc = json.loads(json.dumps(DOC))
    doc["screens"]["date"] = {"frequency": 1, "note": "appid=owm-secret-value-123"}
    playlist = store.create("A", doc, actor="jason")
    exported = store.export(playlist["id"])
    text = json.dumps(exported)
    assert "owm-secret-value-123" not in text
    assert exported["format"] == "desk-display-playlist" and exported["schema_version"] == 1
    imported = store.import_playlist(exported, actor="jason")
    assert imported["id"] != playlist["id"] and imported["name"] == "A"
    hostile = {**exported, "document": {**exported["document"], "screens": {**exported["document"]["screens"], "owm_api_key": "x"}}}
    cleaned = store.import_playlist(hostile, actor="jason")
    assert "owm_api_key" not in cleaned["document"]["screens"]
    for bad in ({}, {**exported, "format": "other"}, {**exported, "schema_version": 2}, {**exported, "extra": 1}):
        with pytest.raises(PlaylistValidationError):
            store.import_playlist(bad, actor="jason")


def test_audit_entries(store):
    playlist = store.create("A", DOC, actor="alice")
    store.rename(playlist["id"], "B", expected_revision=playlist["revision"], actor="bob")
    store.assign("office", playlist["id"], expected_playlist_id=None, actor="carol")
    actions = [(e["actor"], e["action"]) for e in store.snapshot()["audit"]]
    assert actions == [("alice", "create"), ("bob", "rename"), ("carol", "assign")]


def test_atomic_writes_and_interrupted_commit(store, monkeypatch):
    playlist = store.create("A", DOC, actor="jason")
    before = store.path.read_text(encoding="utf-8")

    def fail_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(ps.os, "replace", fail_replace)
    with pytest.raises(OSError):
        store.rename(playlist["id"], "B", expected_revision=playlist["revision"], actor="jason")
    monkeypatch.undo()
    assert store.path.read_text(encoding="utf-8") == before
    assert not [p for p in store.path.parent.iterdir() if p.suffix == ".tmp"]
    assert store.get(playlist["id"])["name"] == "A"


def test_concurrent_writers_do_not_lose_updates(tmp_path):
    path = tmp_path / "playlists.json"
    stores = [PlaylistStore(path) for _ in range(4)]
    errors = []

    def create(i):
        try:
            stores[i % 4].create(f"P{i}", {"screens": {"date": 1}}, actor=f"user{i}")
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=create, args=(i,)) for i in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    data = PlaylistStore(path).snapshot()
    assert len(data["playlists"]) == 20 and data["store_revision"] == 20


def test_friendly_names(store):
    store.set_friendly_name("office", "  Office   desk ", actor="jason")
    assert store.snapshot()["clients"]["office"]["friendly_name"] == "Office desk"
    store.set_friendly_name("office", None, actor="jason")
    assert "office" not in store.snapshot()["clients"]


def test_store_is_not_env_and_path_is_configurable(tmp_path):
    path = ps.store_path({"DESK_DISPLAY_PLAYLIST_STORE_PATH": str(tmp_path / "x.json")})
    assert path == tmp_path / "x.json"
    assert ps.store_path({}).name == "playlists.json"
    assert ps.store_path({}).suffix == ".json"
