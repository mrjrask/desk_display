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


def test_resolve_screens_config_paths_prefers_existing_local_override(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "_project_root", lambda: tmp_path)

    default_path = tmp_path / "screens_config.json"
    local_path = tmp_path / "screens_config.local.json"
    default_path.write_text('{"screens": {"date": 1}}', encoding="utf-8")
    local_path.write_text('{"screens": {"date": 2}}', encoding="utf-8")

    resolved = paths.resolve_screens_config_paths()

    assert resolved.default_path == default_path
    assert resolved.local_override_path == local_path
    assert resolved.active_path == local_path


def test_resolve_screens_config_paths_falls_back_when_local_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "_project_root", lambda: tmp_path)

    default_path = tmp_path / "screens_config.json"
    default_path.write_text('{"screens": {"date": 1}}', encoding="utf-8")

    resolved = paths.resolve_screens_config_paths()

    assert resolved.default_path == default_path
    assert resolved.local_override_path == tmp_path / "screens_config.local.json"
    assert resolved.active_path == default_path


def test_resolve_config_path_helpers_honor_environment_precedence(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "_project_root", lambda: tmp_path)
    monkeypatch.setenv("SCREENS_STYLE_PATH", "/tmp/custom_style.json")
    monkeypatch.setenv("SCREENS_LAYOUTS_PATH", "/tmp/custom_layouts.json")
    monkeypatch.setenv("SCREENS_CONFIG_PATH", "/tmp/custom_screens_config.json")
    monkeypatch.setenv("SCREENS_CONFIG_LOCAL_PATH", "/tmp/custom_screens_config.local.json")

    resolved = paths.resolve_screens_config_paths()

    assert resolved.default_path == paths.Path("/tmp/custom_screens_config.json")
    assert resolved.local_override_path == paths.Path("/tmp/custom_screens_config.local.json")
    assert resolved.active_path == paths.Path("/tmp/custom_screens_config.json")
    assert paths.resolve_style_config_path() == paths.Path("/tmp/custom_style.json")
    assert paths.resolve_layouts_config_path() == paths.Path("/tmp/custom_layouts.json")
