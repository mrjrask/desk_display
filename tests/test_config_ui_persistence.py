import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import config_ui


def _configure_paths(monkeypatch, tmp_path):
    config_path = tmp_path / "screens_config.json"
    layouts_path = tmp_path / "screens_layouts.json"
    monkeypatch.setattr(config_ui, "LOCAL_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(config_ui, "LAYOUTS_CONFIG_PATH", str(layouts_path))
    return config_path, layouts_path


def _write_generation(config_path, layouts_path, generation):
    config_path.write_text(json.dumps({"generation": generation}), encoding="utf-8")
    layouts_path.write_text(json.dumps({"generation": generation}), encoding="utf-8")


def _read_generation(path):
    return json.loads(path.read_text(encoding="utf-8"))["generation"]


def test_competing_bundle_saves_keep_files_in_the_same_generation(monkeypatch, tmp_path):
    config_path, layouts_path = _configure_paths(monkeypatch, tmp_path)
    _write_generation(config_path, layouts_path, "old")
    staged_paths = []
    original_stage = config_ui._stage_json_file

    def record_stage(target, payload):
        staged = original_stage(target, payload)
        staged_paths.append(staged)
        return staged

    monkeypatch.setattr(config_ui, "_stage_json_file", record_stage)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                config_ui._save_config_bundle,
                {"generation": generation},
                {"generation": generation},
            )
            for generation in ("request-a", "request-b")
        ]
        for future in futures:
            future.result()

    assert _read_generation(config_path) == _read_generation(layouts_path)
    assert len(staged_paths) == len(set(staged_paths)) == 4


def test_staging_failure_leaves_both_active_files_unchanged(monkeypatch, tmp_path):
    config_path, layouts_path = _configure_paths(monkeypatch, tmp_path)
    _write_generation(config_path, layouts_path, "old")
    original_stage = config_ui._stage_json_file

    def fail_layout_stage(target, payload):
        if target == layouts_path:
            raise OSError("simulated layout staging failure")
        return original_stage(target, payload)

    monkeypatch.setattr(config_ui, "_stage_json_file", fail_layout_stage)

    with pytest.raises(OSError, match="layout staging failure"):
        config_ui._save_config_bundle({"generation": "new"}, {"generation": "new"})

    assert _read_generation(config_path) == "old"
    assert _read_generation(layouts_path) == "old"
    assert not list(tmp_path.glob(".*.tmp"))


def test_second_commit_failure_rolls_back_first_file(monkeypatch, tmp_path):
    config_path, layouts_path = _configure_paths(monkeypatch, tmp_path)
    _write_generation(config_path, layouts_path, "old")
    original_replace = config_ui.os.replace

    def fail_layout_commit(source, target):
        if Path(target) == layouts_path and Path(source).suffix == ".tmp":
            raise OSError("simulated layout commit failure")
        return original_replace(source, target)

    monkeypatch.setattr(config_ui.os, "replace", fail_layout_commit)

    with pytest.raises(OSError, match="layout commit failure"):
        config_ui._save_config_bundle({"generation": "new"}, {"generation": "new"})

    assert _read_generation(config_path) == "old"
    assert _read_generation(layouts_path) == "old"
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.bak"))
