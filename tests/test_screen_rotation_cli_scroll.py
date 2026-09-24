"""Regression tests: the export/import CLI scripts used to silently drop
the global scroll settings and per-screen scroll overrides on a round trip,
resetting them to defaults (see scripts/export_screen_rotation_config.py and
scripts/import_screen_rotation_config.py)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT_PATH = REPO_ROOT / "scripts" / "export_screen_rotation_config.py"
IMPORT_SCRIPT_PATH = REPO_ROOT / "scripts" / "import_screen_rotation_config.py"


def _load_module(name: str, path: Path):
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def export_mod():
    return _load_module("export_screen_rotation_config_cli", EXPORT_SCRIPT_PATH)


@pytest.fixture
def import_mod():
    return _load_module("import_screen_rotation_config_cli", IMPORT_SCRIPT_PATH)


SOURCE_CONFIG = {
    "screens": {
        "date": {"frequency": 1, "scroll": {"speed": 2.0}},
        "nixie": 1,
    },
    "playlists": {},
    "sequence": [],
    "scroll": {"speed": 1.5, "smoothness": 1.2, "vertical_speed_adjustment": 0.3},
}


def test_export_preserves_global_and_per_screen_scroll(export_mod):
    from config_ui import _build_screen_entries

    entries = _build_screen_entries(SOURCE_CONFIG, {"screens": {}})
    exported = export_mod._build_config_payload(entries, SOURCE_CONFIG)

    assert exported["scroll"] == {
        "speed": 1.5,
        "smoothness": 1.2,
        "vertical_speed_adjustment": 0.3,
    }
    assert exported["screens"]["date"]["scroll"] == {"speed": 2.0}
    # A screen with no override doesn't gain a spurious "scroll" key.
    assert exported["screens"]["nixie"] == 1


def test_import_dict_config_preserves_global_and_per_screen_scroll(export_mod, import_mod):
    from config_ui import _build_screen_entries

    entries = _build_screen_entries(SOURCE_CONFIG, {"screens": {}})
    exported = export_mod._build_config_payload(entries, SOURCE_CONFIG)

    imported = import_mod._normalize_import_config_payload(dict(exported))

    assert imported["scroll"] == {
        "speed": 1.5,
        "smoothness": 1.2,
        "vertical_speed_adjustment": 0.3,
    }
    assert imported["screens"]["date"]["scroll"] == {"speed": 2.0}


def test_import_entries_list_config_preserves_global_and_per_screen_scroll(
    import_mod, monkeypatch
):
    """The config UI's own POST /api/screens/import also accepts a
    {"screens": [...]} entries-list payload (not just the dict-of-screens
    shape); the CLI import script's mirror of that path must preserve
    scroll too."""

    monkeypatch.setattr(import_mod, "_load_active_config", lambda: {"scroll": {"speed": 1.0}})

    payload = {
        "config": {
            "screens": [
                {"id": "date", "frequency": 1, "scroll_speed_enabled": True, "scroll_speed": 2.0},
                {"id": "nixie", "frequency": 1},
            ],
            "scroll": {"speed": 1.5, "smoothness": 1.2, "vertical_speed_adjustment": 0.3},
        }
    }

    config, _style_config, _layouts_config = import_mod._resolve_import(payload)

    assert config["scroll"] == {
        "speed": 1.5,
        "smoothness": 1.2,
        "vertical_speed_adjustment": 0.3,
    }
    assert config["screens"]["date"]["scroll"] == {"speed": 2.0}
