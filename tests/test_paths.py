import errno
import logging

import pytest

import paths

HISTORY_PATHS = (
    ("PRESSURE_HISTORY_PATH", "pressure_history.json"),
    ("WEATHER_METRIC_HISTORY_PATH", "weather_metric_history.json"),
)


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


@pytest.mark.parametrize(("env_var", "filename"), HISTORY_PATHS)
def test_resolve_cache_history_path_on_clean_install(tmp_path, monkeypatch, env_var, filename):
    monkeypatch.setattr(paths, "_project_root", lambda: tmp_path)
    monkeypatch.delenv(env_var, raising=False)

    resolved = paths.resolve_cache_file_path(env_var, filename)

    assert resolved == tmp_path / "cache" / filename
    assert not resolved.exists()
    assert not resolved.parent.exists()


@pytest.mark.parametrize(("env_var", "filename"), HISTORY_PATHS)
def test_resolve_cache_history_path_migrates_legacy_root_file(
    tmp_path, monkeypatch, env_var, filename
):
    monkeypatch.setattr(paths, "_project_root", lambda: tmp_path)
    monkeypatch.delenv(env_var, raising=False)
    legacy = tmp_path / filename
    legacy.write_text('{"legacy": true}', encoding="utf-8")

    resolved = paths.resolve_cache_file_path(env_var, filename)

    assert resolved.read_text(encoding="utf-8") == '{"legacy": true}'
    assert not legacy.exists()


@pytest.mark.parametrize(("env_var", "filename"), HISTORY_PATHS)
def test_resolve_cache_history_path_preserves_existing_canonical_file(
    tmp_path, monkeypatch, caplog, env_var, filename
):
    monkeypatch.setattr(paths, "_project_root", lambda: tmp_path)
    monkeypatch.delenv(env_var, raising=False)
    legacy = tmp_path / filename
    canonical = tmp_path / "cache" / filename
    canonical.parent.mkdir()
    legacy.write_text('{"source": "legacy"}', encoding="utf-8")
    canonical.write_text('{"source": "canonical"}', encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger=paths.__name__):
        resolved = paths.resolve_cache_file_path(env_var, filename)

    assert resolved == canonical
    assert canonical.read_text(encoding="utf-8") == '{"source": "canonical"}'
    assert legacy.read_text(encoding="utf-8") == '{"source": "legacy"}'
    assert "preserving both files" in caplog.text


def test_history_migration_does_not_clobber_canonical_file_created_concurrently(
    tmp_path, monkeypatch, caplog
):
    legacy = tmp_path / "pressure_history.json"
    canonical = tmp_path / "cache" / "pressure_history.json"
    canonical.parent.mkdir()
    legacy.write_text('{"source": "legacy"}', encoding="utf-8")
    real_link = paths.os.link

    def create_canonical_then_link(source, destination):
        canonical.write_text('{"source": "concurrent writer"}', encoding="utf-8")
        real_link(source, destination)

    monkeypatch.setattr(paths.os, "link", create_canonical_then_link)

    with caplog.at_level(logging.WARNING, logger=paths.__name__):
        paths._migrate_legacy_root_history(legacy, canonical)

    assert canonical.read_text(encoding="utf-8") == '{"source": "concurrent writer"}'
    assert legacy.read_text(encoding="utf-8") == '{"source": "legacy"}'
    assert "preserving both files" in caplog.text


@pytest.mark.parametrize("link_error", (errno.EXDEV, errno.EPERM))
def test_history_migration_copies_exclusively_when_hard_links_are_unavailable(
    tmp_path, monkeypatch, link_error
):
    legacy = tmp_path / "pressure_history.json"
    canonical = tmp_path / "cache" / "pressure_history.json"
    canonical.parent.mkdir()
    legacy.write_text('{"source": "legacy"}', encoding="utf-8")

    def reject_hard_link(source, destination):
        raise OSError(link_error, "hard links unavailable")

    monkeypatch.setattr(paths.os, "link", reject_hard_link)

    paths._migrate_legacy_root_history(legacy, canonical)

    assert canonical.read_text(encoding="utf-8") == '{"source": "legacy"}'
    assert not legacy.exists()


@pytest.mark.parametrize(("env_var", "filename"), HISTORY_PATHS)
def test_resolve_cache_history_path_honors_environment_override_without_migration(
    tmp_path, monkeypatch, env_var, filename
):
    monkeypatch.setattr(paths, "_project_root", lambda: tmp_path)
    monkeypatch.setenv(env_var, f"custom/{filename}")
    legacy = tmp_path / filename
    legacy.write_text('{"legacy": true}', encoding="utf-8")

    resolved = paths.resolve_cache_file_path(env_var, filename)

    assert resolved == tmp_path / "custom" / filename
    assert legacy.exists()
    assert not (tmp_path / "cache" / filename).exists()
