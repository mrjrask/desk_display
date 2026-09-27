"""The config UI's service banner reports the unit that drives this panel."""
from __future__ import annotations

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
