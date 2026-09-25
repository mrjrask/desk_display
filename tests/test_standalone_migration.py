"""Phase 17: migrating a standalone rotation onto the render server."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from remote_display import migration
from remote_display.playlist_store import PlaylistStore

ROOT = Path(__file__).resolve().parents[1]

LEGACY = {
    "screens": {
        "date": 1,
        "time": {"frequency": 2, "extra_seconds": 3},  # legacy ID for nixie
        "cubs next 2": 1,  # retired
        "not a screen": 1,
        "NHL Standings West": {"frequency": 0, "extra_seconds": 0,
                               "alt": {"screen": "NHL Standings West v2", "frequency": 2}},
        "weather1": {"frequency": 1, "alt": {"screen": "retired-or-unknown", "frequency": 2}},
        "inside": 1,
    },
    "playlists": {"p1": {"label": "morning", "steps": [{"screen": "date"}, {"screen": "sensors"},
                                                        {"screen": "sox last 2"}]}},
    "sequence": [{"playlist": "p1"}, {"playlist": "gone"}],
    "theme": "dark",
}


def write(path: Path, data) -> Path:
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def files(tmp_path):
    config = write(tmp_path / "screens_config.json", LEGACY)
    style = write(tmp_path / "screens_style.json", {"screens": {"time": {"bg": "#000"}, "date": {"bg": "#111"},
                                                                "old thing": {"bg": "#222"}}})
    layouts = write(tmp_path / "screens_layouts.json", {"screens": {"quad": {"pages": []}}})
    return {"config": config, "style": style, "layouts": layouts, "store": PlaylistStore(tmp_path / "pl.json"),
            "bundles": tmp_path / "bundles", "tmp": tmp_path}


def source(files, **kwargs):
    return migration.load_source(config_path=files["config"], style_path=files["style"],
                                 layouts_path=files["layouts"], **kwargs)


def kinds(plan):
    return {(c.kind, c.screen) for c in plan.changes}


# ── Reading and normalizing ────────────────────────────────────────────────


def test_legacy_ids_retired_and_unknown_screens_are_reported(files):
    plan = migration.plan(source(files), files["store"])
    screens = plan.document["screens"]
    assert "nixie" in screens and screens["nixie"] == {"frequency": 2, "extra_seconds": 3}
    assert "time" not in screens and "cubs next 2" not in screens and "not a screen" not in screens
    assert ("renamed", "nixie") in kinds(plan) and ("removed", "cubs next 2") in kinds(plan)
    assert "alt" not in screens["weather1"]  # its only alternate was unknown
    assert screens["NHL Standings West"]["alt"]["screen"] == "NHL Standings West v2"
    assert [s["screen"] for s in plan.document["playlists"]["p1"]["steps"]] == ["date", "inside"]
    assert plan.document["sequence"] == [{"playlist": "p1"}]
    messages = " ".join(c.message for c in plan.changes)
    assert "'theme' has no server equivalent" in messages and "missing playlist 'gone'" in messages
    assert ("remote_limit", "inside") in kinds(plan)
    assert not plan.blocked


def test_the_real_rotation_keeps_every_semantic(tmp_path):
    config = json.loads((ROOT / "screens_config.json").read_text())
    document, changes = migration.normalize(config)
    assert document["screens"] == config["screens"]
    assert list(document["screens"]) == list(config["screens"])  # order is play order
    assert document["playlists"] == config["playlists"] and document["sequence"] == config["sequence"]
    assert all(c.kind == "remote_limit" for c in changes)


@pytest.mark.parametrize("size", ["small", "large"])
def test_default_bundles_migrate_when_no_rotation_exists(files, size):
    files["config"].unlink()
    migrated = source(files, defaults=size)
    assert migrated.kind == f"bundled-{size}"
    plan = migration.plan(migrated, files["store"])
    assert plan.document["screens"] and not plan.blocked
    with pytest.raises(migration.MigrationError):
        source(files)  # no defaults named: refuse rather than guess


def test_local_override_wins(tmp_path, monkeypatch):
    base = write(tmp_path / "base.json", {"screens": {"date": 1}})
    local = write(tmp_path / "local.json", {"screens": {"nixie": 1}})
    monkeypatch.setenv("SCREENS_CONFIG_PATH", str(base))
    monkeypatch.setenv("SCREENS_CONFIG_LOCAL_PATH", str(local))
    assert list(migration.load_source().config["screens"]) == ["nixie"]
    local.unlink()
    assert list(migration.load_source().config["screens"]) == ["date"]


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", json.dumps({"playlists": {}})])
def test_malformed_content_is_refused_without_changes(files, content):
    write(files["config"], content)
    with pytest.raises(migration.MigrationError):
        migration.plan(source(files), files["store"])
    assert files["store"].snapshot()["playlists"] == {}


def test_style_and_layout_keys_are_canonicalized_but_nothing_is_lost(files):
    plan = migration.plan(source(files), files["store"])
    assert set(plan.style["screens"]) == {"nixie", "date", "old thing"}
    assert ("style", "nixie") in kinds(plan)


# ── Applying ───────────────────────────────────────────────────────────────


def test_apply_is_non_destructive_and_repeatable(files):
    before = {name: digest(files[name]) for name in ("config", "style", "layouts")}
    first = migration.apply(migration.plan(source(files), files["store"], clients=["office"]), files["store"],
                            bundle_dir=files["bundles"])
    assert first["created"] and first["assigned"] == {"office": None}
    second_plan = migration.plan(source(files), files["store"], clients=["office"])
    assert second_plan.existing_playlist_id == first["playlist_id"]
    second = migration.apply(second_plan, files["store"], bundle_dir=files["bundles"])
    assert not second["created"] and second["assigned"] == {}
    assert len(files["store"].snapshot()["playlists"]) == 1
    assert {name: digest(files[name]) for name in ("config", "style", "layouts")} == before
    bundle = json.loads(Path(first["bundle"]).read_text())
    assert bundle["format"] == migration.BUNDLE_FORMAT and bundle["originals"]["config"]["text"]


def test_assignments_are_never_changed_silently(files):
    other = files["store"].create("Other", {"screens": {"date": 1}}, actor="t")
    files["store"].assign("office", other["id"], expected_playlist_id=None, actor="t")
    plan = migration.plan(source(files), files["store"], clients=["office", "den"])
    assert plan.assignments == {"office": {"action": "conflict", "current": other["id"]},
                                "den": {"action": "assign", "current": None}}
    result = migration.apply(plan, files["store"], bundle_dir=files["bundles"])
    assert result["skipped"] == ["office"] and files["store"].assignment_for("office").playlist_id == other["id"]
    forced = migration.apply(migration.plan(source(files), files["store"], clients=["office"]), files["store"],
                             bundle_dir=files["bundles"], force_assign=True)
    assert forced["assigned"] == {"office": other["id"]}


def test_interrupted_run_leaves_originals_and_a_rerun_finishes(files, monkeypatch):
    store = files["store"]
    real_create = store.create
    monkeypatch.setattr(store, "create", lambda *a, **k: (_ for _ in ()).throw(OSError("power cut")))
    with pytest.raises(OSError):
        migration.apply(migration.plan(source(files), store, clients=["office"]), store,
                        bundle_dir=files["bundles"])
    assert store.snapshot()["playlists"] == {} and list(files["bundles"].iterdir())  # the record exists
    monkeypatch.setattr(store, "create", real_create)
    result = migration.apply(migration.plan(source(files), store, clients=["office"]), store,
                             bundle_dir=files["bundles"])
    assert result["created"] and len(store.snapshot()["playlists"]) == 1


def test_interrupted_style_install_keeps_the_old_file(files, monkeypatch):
    before = files["style"].read_text()
    real = migration.os.replace

    def flaky(src, dst):
        if str(dst) == str(files["style"]):
            raise OSError("disk full")
        return real(src, dst)

    monkeypatch.setattr(migration.os, "replace", flaky)
    with pytest.raises(OSError):
        migration.apply(migration.plan(source(files), files["store"]), files["store"],
                        bundle_dir=files["bundles"], install_style=True)
    assert files["style"].read_text() == before
    assert not [p for p in files["tmp"].iterdir() if p.name.endswith(".tmp")]


def test_rollback_restores_assignments_playlists_and_files(files):
    other = files["store"].create("Other", {"screens": {"date": 1}}, actor="t")
    files["store"].assign("office", other["id"], expected_playlist_id=None, actor="t")
    style_before = files["style"].read_text()
    result = migration.apply(migration.plan(source(files), files["store"], clients=["office", "den"]),
                             files["store"], bundle_dir=files["bundles"], force_assign=True, install_style=True)
    assert files["style"].read_text() != style_before
    undone = migration.rollback(Path(result["bundle"]), files["store"])
    assert sorted(undone["assignments"]) == ["den", "office"] and undone["deleted_playlist"]
    assert files["store"].assignment_for("office").playlist_id == other["id"]
    assert files["store"].assignment_for("den") is None
    assert set(files["store"].snapshot()["playlists"]) == {other["id"]}
    assert files["style"].read_text() == style_before


def test_rollback_leaves_later_changes_alone(files):
    result = migration.apply(migration.plan(source(files), files["store"], clients=["office"]), files["store"],
                             bundle_dir=files["bundles"])
    other = files["store"].create("Other", {"screens": {"date": 1}}, actor="t")
    files["store"].assign("office", other["id"], expected_playlist_id=result["playlist_id"], actor="t")
    files["store"].assign("den", result["playlist_id"], expected_playlist_id=None, actor="t")
    undone = migration.rollback(Path(result["bundle"]), files["store"])
    assert undone["assignments"] == [] and not undone["deleted_playlist"]  # den still uses it
    assert files["store"].assignment_for("office").playlist_id == other["id"]


def test_export_is_a_backup_without_secrets(files, monkeypatch):
    monkeypatch.setenv("DESK_DISPLAY_SERVER_ADMIN_TOKEN", "admin-token-" + "a" * 32)
    write(files["config"], {**LEGACY, "note": "admin-token-" + "a" * 32})
    path = migration.export_bundle(migration.plan(source(files), files["store"]), files["tmp"] / "backup.json")
    assert "admin-token-" + "a" * 32 not in path.read_text()
    assert files["store"].snapshot()["playlists"] == {}


def test_cli_previews_by_default(files, monkeypatch, capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location("migrate_cli", ROOT / "scripts" / "migrate_standalone_config.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setenv("SCREENS_CONFIG_PATH", str(files["config"]))
    monkeypatch.setenv("SCREENS_CONFIG_LOCAL_PATH", str(files["tmp"] / "none.json"))
    monkeypatch.setenv("SCREENS_STYLE_PATH", str(files["style"]))
    monkeypatch.setenv("SCREENS_LAYOUTS_PATH", str(files["layouts"]))
    monkeypatch.setenv("DESK_DISPLAY_CLIENT_REGISTRY_PATH", str(files["tmp"] / "clients.json"))
    store = str(files["tmp"] / "pl.json")
    assert cli.main(["--store", store, "--assign", "office"]) == 0
    out = capsys.readouterr().out
    assert "legacy ID 'time' is now 'nixie'" in out and "Preview only" in out
    assert PlaylistStore(store).snapshot()["playlists"] == {}
    assert cli.main(["--store", store, "--assign", "office", "--apply",
                     "--bundle-dir", str(files["bundles"])]) == 0
    assert "Applied." in capsys.readouterr().out
    assert PlaylistStore(store).assignment_for("office") is not None
