import json

import schedule_migrations as sm


def test_migrate_legacy_sequence_to_schema_v2():
    """Regression test: migrate_config used to call build_scheduler on the
    migrated result, but the migration is structural only and never produces
    a "screens" mapping, so build_scheduler always raised ValueError and
    every legacy migration failed."""

    result = sm.migrate_config({"sequence": ["date", "nixie"]})

    assert result.migrated is True
    assert result.config["version"] == sm.TARGET_VERSION
    assert "screens" not in result.config
    assert result.config["playlists"]["main"]["steps"] == [
        {"screen": "date"},
        {"screen": "nixie"},
    ]
    assert result.config["sequence"] == [{"playlist": "main"}]
    assert result.config["metadata"]["migrated_from"] == sm.LEGACY_VERSION


def test_migrate_updates_stale_version_and_preserves_metadata():
    """Regression test: an old version number used to survive migration
    (setdefault never overwrites an existing key), and the caller's own
    metadata dict was mutated in place instead of copied."""

    source_metadata = {"note": "kept"}
    data = {
        "version": 1,
        "playlists": {},
        "sequence": [],
        "metadata": source_metadata,
    }

    result = sm.migrate_config(data)

    assert result.migrated is True
    assert result.config["version"] == sm.TARGET_VERSION
    assert result.config["metadata"] == {"note": "kept", "migrated_from": 1}
    # The caller's own metadata dict must not have been mutated in place.
    assert source_metadata == {"note": "kept"}


def test_migrate_already_current_version_is_a_noop():
    data = {
        "version": sm.TARGET_VERSION,
        "playlists": {"main": {"steps": []}},
        "sequence": [{"playlist": "main"}],
    }

    result = sm.migrate_config(data)

    assert result.migrated is False
    assert result.config == data


def test_write_json_preserves_key_order(tmp_path):
    """Regression test: write_json used sort_keys=True, which alphabetizes
    "screens" and silently reorders the display's rotation."""

    path = tmp_path / "out.json"
    data = {"screens": {"zebra": 1, "apple": 2, "date": 3}}

    sm.write_json(str(path), data)

    with open(path, encoding="utf-8") as fh:
        raw = fh.read()

    assert list(json.loads(raw)["screens"]) == ["zebra", "apple", "date"]
    # Confirm key order in the raw text too, not just after re-parsing.
    assert raw.index('"zebra"') < raw.index('"apple"') < raw.index('"date"')
