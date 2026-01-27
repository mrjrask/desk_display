import paths


def test_resolve_storage_paths_uses_project_root(tmp_path, monkeypatch):
    # Ensure the project root calculation can be redirected for the test
    monkeypatch.setattr(paths, "_project_root", lambda: tmp_path)

    storage_paths = paths.resolve_storage_paths(logger=None)

    assert storage_paths.screenshot_dir == tmp_path / "screenshots"
    assert storage_paths.current_screenshot_dir.name == "current"
    assert storage_paths.current_screenshot_dir.parent == storage_paths.screenshot_dir
    assert storage_paths.archive_base == tmp_path / "screenshot_archive"
    assert storage_paths.current_screenshot_dir.exists()
    assert storage_paths.archive_base.exists()
