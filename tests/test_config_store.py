import json
import sqlite3
from pathlib import Path

import pytest

from config_store import ConfigStore


class InstrumentedConnection:
    def __init__(self, connection, events):
        object.__setattr__(self, "connection", connection)
        object.__setattr__(self, "events", events)

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def __setattr__(self, name, value):
        if name in {"connection", "events"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.connection, name, value)

    def __enter__(self):
        self.events.append("transaction entered")
        self.connection.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        result = self.connection.__exit__(exc_type, exc_value, traceback)
        self.events.append(
            "transaction committed" if exc_type is None else "transaction rolled back"
        )
        return result

    def commit(self):
        self.connection.commit()
        self.events.append("commit called")

    def close(self):
        self.connection.close()
        self.events.append("connection closed")


@pytest.fixture
def instrument_connections(monkeypatch):
    events = []
    real_connect = sqlite3.connect

    def connect(*args, **kwargs):
        return InstrumentedConnection(real_connect(*args, **kwargs), events)

    monkeypatch.setattr("config_store.sqlite3.connect", connect)
    return events, real_connect


def make_config(value: int) -> dict:
    return {"screens": {"date": value, "inside": value + 1}}


def test_config_store_versioning_and_pruning(tmp_path):
    config_path = tmp_path / "screens_config.json"
    store = ConfigStore(str(config_path), retention=2)

    v1 = store.save(make_config(1), actor="tester1")
    store.save(make_config(2), actor="tester2")
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

    monkeypatch.setattr(store, "_record_version", fail_record)
    with pytest.raises(sqlite3.OperationalError, match="history unavailable"):
        store.save(make_config(2), actor="tester")

    assert store.load() == original


def test_connections_close_after_success_and_writes_commit(tmp_path, instrument_connections):
    events, _ = instrument_connections
    store = ConfigStore(str(tmp_path / "screens_config.json"))

    version_id = store.save(make_config(7), actor="tester")
    assert store.latest_version_id() == version_id
    assert store.load_version(version_id) == make_config(7)
    assert store.list_versions() == [
        {
            "id": version_id,
            "created_at": store.list_versions()[0]["created_at"],
            "actor": "tester",
            "summary": "Added screens: date, inside",
        }
    ]

    assert events.count("connection closed") == events.count("transaction entered")
    assert "commit called" in events
    for index, event in enumerate(events):
        if event.startswith("transaction commit"):
            assert events[index + 1] == "connection closed"


def test_connection_rolls_back_and_closes_after_transaction_error(
    tmp_path, monkeypatch, instrument_connections
):
    events, real_connect = instrument_connections
    store = ConfigStore(str(tmp_path / "screens_config.json"))
    events.clear()
    real_open = Path.open

    def fail_archive_write(path, *args, **kwargs):
        if path.parent == store.archive_dir:
            raise OSError("archive unavailable")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_archive_write)
    with pytest.raises(OSError, match="archive unavailable"):
        store._record_version(make_config(8), actor="tester", summary="test", metadata={})

    assert events == ["transaction entered", "transaction rolled back", "connection closed"]
    with real_connect(store.db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM config_versions").fetchone()[0]
    assert count == 0
