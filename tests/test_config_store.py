import json

import pytest

from config_store import ConfigStore


def make_config(value: int) -> dict:
    return {"screens": {"date": value, "inside": value + 1}}


def test_config_store_versioning_and_pruning(tmp_path):
    config_path = tmp_path / "screens_config.json"
    store = ConfigStore(str(config_path), retention=2)

    v1 = store.save(make_config(1), actor="tester1")
    v2 = store.save(make_config(2), actor="tester2")
    v3 = store.save(make_config(3), actor="tester3")

    versions = store.list_versions()
    assert versions[0]["id"] == v3
    assert len(versions) == 2
    assert store.latest_version_id() == v3

    archive_dir = config_path.parent / "config_versions"
    archived_files = sorted(archive_dir.glob("*.json"))
    assert len(archived_files) == 2

    with pytest.raises(KeyError):
        store.load_version(v1)


def test_config_store_rollback(tmp_path):
    config_path = tmp_path / "config.json"
    store = ConfigStore(str(config_path))

    first = make_config(10)
    store.save(first, actor="tester")
    second = make_config(20)
    version_id = store.save(second, actor="tester")

    rolled = store.rollback(version_id, actor="tester")
    assert rolled["screens"]["date"] == 20
    persisted = json.loads(config_path.read_text())
    assert persisted["screens"]["date"] == 20


def test_load_only_treats_missing_file_as_empty(tmp_path):
    config_path = tmp_path / "screens_config.json"
    store = ConfigStore(str(config_path))

    assert store.load() == {}

    config_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        store.load()

    config_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        store.load()


def test_load_propagates_io_errors(tmp_path, monkeypatch):
    config_path = tmp_path / "screens_config.json"
    config_path.write_text("{}", encoding="utf-8")
    store = ConfigStore(str(config_path))

    def fail_open(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(type(store.config_path), "open", fail_open)
    with pytest.raises(PermissionError, match="denied"):
        store.load()


def test_version_record_failure_restores_active_configuration(tmp_path, monkeypatch):
    config_path = tmp_path / "screens_config.json"
    store = ConfigStore(str(config_path))
    original = make_config(1)
    store.save(original, actor="tester")

    def fail_record(*args, **kwargs):
        raise sqlite3.OperationalError("history unavailable")

    import sqlite3

    monkeypatch.setattr(store, "_record_version", fail_record)
    with pytest.raises(sqlite3.OperationalError, match="history unavailable"):
        store.save(make_config(2), actor="tester")

    assert store.load() == original
