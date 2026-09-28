"""The config UI's service banner reports the unit that drives this panel."""
from __future__ import annotations

import os

import pytest

pytest.importorskip("flask")

import config_ui
import install_modes


@pytest.mark.parametrize(
    ("mode", "unit"),
    [
        ("standalone", "desk_display.service"),
        ("combined", "desk_display_client.service"),
        ("client", "desk_display_client.service"),
        ("server", "desk_display_server.service"),
    ],
)
def test_banner_unit_follows_the_installed_mode(tmp_path, monkeypatch, mode, unit):
    install_modes.write_marker(tmp_path, mode)
    monkeypatch.setattr(config_ui, "__file__", str(tmp_path / "config_ui.py"))
    assert config_ui._panel_service_unit() == unit


def test_unmarked_install_uses_the_client_unit_when_it_has_a_client_env(tmp_path, monkeypatch):
    monkeypatch.setattr(config_ui, "__file__", str(tmp_path / "config_ui.py"))
    assert config_ui._panel_service_unit() == "desk_display.service"
    (tmp_path / ".env.client").write_text("DESK_DISPLAY_ROLE=client\n")
    assert config_ui._panel_service_unit() == "desk_display_client.service"


def test_service_status_queries_the_panel_unit(monkeypatch):
    queried = []
    monkeypatch.setattr(config_ui, "_panel_service_unit", lambda: "desk_display_client.service")
    monkeypatch.setattr(config_ui, "_query_service_status", lambda unit: queried.append(unit) or {"unit": unit})
    monkeypatch.setattr(config_ui, "_SERVICE_STATUS_CACHE", {})
    assert config_ui._load_service_status()["unit"] == "desk_display_client.service"
    assert queried == ["desk_display_client.service"]


def test_combined_install_reads_the_panels_screenshot_dir(tmp_path, monkeypatch):
    install_modes.write_marker(tmp_path, "combined")
    (tmp_path / ".env.client").write_text("SCREENSHOT_DIR=/srv/shots\nSCREENSHOT_ARCHIVE_BASE=/srv/archive\n")
    monkeypatch.setenv("SCREENSHOT_DIR", "")  # registers a restore of the real value
    monkeypatch.delenv("SCREENSHOT_DIR")
    monkeypatch.setenv("SCREENSHOT_ARCHIVE_BASE", "/explicit/archive")
    config_ui._adopt_panel_screenshot_paths(tmp_path)
    assert os.environ["SCREENSHOT_DIR"] == "/srv/shots"
    assert os.environ["SCREENSHOT_ARCHIVE_BASE"] == "/explicit/archive"  # the UI's own setting wins


def test_other_modes_keep_their_own_screenshot_dir(tmp_path, monkeypatch):
    install_modes.write_marker(tmp_path, "server")
    (tmp_path / ".env.client").write_text("SCREENSHOT_DIR=/srv/shots\n")
    monkeypatch.setenv("SCREENSHOT_DIR", "")  # registers a restore of the real value
    monkeypatch.delenv("SCREENSHOT_DIR")
    config_ui._adopt_panel_screenshot_paths(tmp_path)
    assert "SCREENSHOT_DIR" not in os.environ


@pytest.mark.parametrize(
    ("env_role", "client_env", "unit", "serves"),
    [
        ("server", False, "desk_display_server.service", True),
        ("server", True, "desk_display_client.service", True),  # combined
        ("client", False, "desk_display_client.service", False),
    ],
)
def test_unmarked_install_follows_the_env_role(tmp_path, monkeypatch, env_role, client_env, unit, serves):
    monkeypatch.setattr(install_modes, "SYSTEMD_DIR", tmp_path / "systemd")
    monkeypatch.setattr(install_modes.detect_mode, "__defaults__", (tmp_path / "systemd",))
    monkeypatch.setattr(config_ui, "__file__", str(tmp_path / "config_ui.py"))
    (tmp_path / ".env").write_text(f"DESK_DISPLAY_ROLE={env_role}\n")
    if client_env:
        (tmp_path / ".env.client").write_text("DESK_DISPLAY_ROLE=client\n")
    assert config_ui._panel_service_unit() == unit
    assert config_ui._serves_displays() is serves


@pytest.fixture
def client_install(tmp_path, monkeypatch):
    install_modes.write_marker(tmp_path, "client")
    monkeypatch.setattr(config_ui, "__file__", str(tmp_path / "config_ui.py"))
    monkeypatch.setattr(config_ui, "_is_auth_enabled", lambda: False)
    monkeypatch.setattr(config_ui, "_load_service_status", lambda unit_name=None: {})
    config_ui.app.config["TESTING"] = True
    return config_ui.app.test_client()


def test_client_install_serves_only_the_screenshot_pages(client_install):
    assert client_install.get("/screenshots").status_code == 200
    assert client_install.get("/feed").status_code == 200
    assert client_install.get("/api/screenshots").status_code == 200
    assert client_install.get("/api/feed/screenshots").status_code == 200
    home = client_install.get("/")
    assert home.status_code == 302 and home.headers["Location"].endswith("/screenshots")
    for path in ("/playlists", "/clients", "/api/screens", "/api/screens/export"):
        assert client_install.get(path).status_code == 404, path
    assert client_install.post("/api/screens", json={}).status_code == 404


def test_client_screenshots_page_links_only_to_local_pages(client_install):
    page = client_install.get("/screenshots").get_data(as_text=True)
    assert 'href="/feed"' in page
    assert 'href="/playlists"' not in page and 'href="/clients"' not in page and 'href="/"' not in page


@pytest.mark.parametrize("mode", ["standalone", "server", "combined"])
def test_other_modes_keep_the_full_config_ui(tmp_path, monkeypatch, mode):
    install_modes.write_marker(tmp_path, mode)
    monkeypatch.setattr(config_ui, "__file__", str(tmp_path / "config_ui.py"))
    assert config_ui._screenshots_only() is False
