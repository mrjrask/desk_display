import json
from pathlib import Path

import pytest

import config_ui
from schedule import build_scheduler

ROOT = Path(__file__).resolve().parents[1]


def _preview_through_pass(config, last_pass=12):
    entries = build_scheduler(config).preview_scheduled_entries(2_000)
    return [
        entry
        for entry in entries
        if entry.phase == "startup"
        or (entry.pass_number is not None and entry.pass_number <= last_pass)
    ]


@pytest.mark.parametrize("profile", ["large", "small"])
def test_default_profile_api_import_export_round_trip_preserves_schedule(
    profile, tmp_path, monkeypatch
):
    """Defaults remain lossless through the same import/export path used by the UI."""

    default_path = ROOT / f"default_screens_{profile}.json"
    source_bundle = json.loads(default_path.read_text(encoding="utf-8"))
    source_config = source_bundle["config"]

    local_config_path = tmp_path / "screens_config.json"
    layouts_path = tmp_path / "screens_layouts.json"
    monkeypatch.setattr(config_ui, "LOCAL_CONFIG_PATH", str(local_config_path))
    monkeypatch.setattr(config_ui, "LAYOUTS_CONFIG_PATH", str(layouts_path))

    client = config_ui.app.test_client()
    defaults_response = client.get(f"/api/screens/defaults?profile={profile}")

    assert defaults_response.status_code == 200
    defaults = defaults_response.get_json()
    assert defaults["selected_default_profile"] == profile
    assert defaults["config"]["screens"] == source_config["screens"]
    assert defaults["config"]["playlists"] == source_config["playlists"]
    assert defaults["config"]["sequence"] == source_config["sequence"]
    assert [entry["id"] for entry in defaults["screens"][: len(source_config["screens"])]] == list(
        source_config["screens"]
    )

    expected_playlists, expected_assignments = config_ui._build_playlist_assignments(source_config)
    assert defaults["playlists"] == expected_playlists
    assert defaults["playlist_assignments"] == expected_assignments

    import_response = client.post("/api/screens/import", json={"config": defaults["config"]})
    assert import_response.status_code == 200
    export_response = client.get("/api/screens/export")
    assert export_response.status_code == 200
    round_tripped = json.loads(export_response.get_data(as_text=True))

    assert list(round_tripped["screens"]) == list(source_config["screens"])
    assert round_tripped["playlists"] == source_config["playlists"]
    assert round_tripped["sequence"] == source_config["sequence"]

    for screen_id, source_spec in source_config["screens"].items():
        exported_spec = round_tripped["screens"][screen_id]
        source_frequency = (
            source_spec["frequency"] if isinstance(source_spec, dict) else source_spec
        )
        exported_frequency = (
            exported_spec["frequency"] if isinstance(exported_spec, dict) else exported_spec
        )
        assert isinstance(exported_frequency, int)
        assert exported_frequency == source_frequency
        if isinstance(source_spec, dict) and "alt" in source_spec:
            assert isinstance(exported_spec["alt"]["frequency"], int)
            assert exported_spec["alt"]["frequency"] == source_spec["alt"]["frequency"]

    assert _preview_through_pass(round_tripped) == _preview_through_pass(source_config)


def test_vertical_scroll_adjustment_normalizes_fractional_values():
    assert config_ui._normalize_scroll_settings({"vertical_speed_adjustment": 0.25})[
        "vertical_speed_adjustment"
    ] == 0.25
    assert config_ui._normalize_scroll_settings({"vertical_speed_adjustment": -0.25})[
        "vertical_speed_adjustment"
    ] == -0.25
    assert config_ui._normalize_scroll_settings({})["vertical_speed_adjustment"] == 0.0


def test_build_playlist_assignments_preserves_order_and_labels():
    playlists, assignments = config_ui._build_playlist_assignments(
        {
            "playlists": {
                "second": {"label": "Second", "steps": [{"screen": "inside"}]},
                "first": {"label": "", "steps": [{"screen": "date"}]},
            },
            "sequence": [{"playlist": "first"}, {"playlist": "second"}],
        }
    )

    assert playlists == [{"id": "first", "name": "first"}, {"id": "second", "name": "Second"}]
    assert assignments == {"date": "first", "inside": "second"}


def test_screen_config_page_bootstraps_server_playlist_state(monkeypatch):
    monkeypatch.setattr(
        config_ui,
        "_load_active_config",
        lambda: {
            "screens": {"date": 1},
            "playlists": {"default": {"label": "Default", "steps": [{"screen": "date"}]}},
            "sequence": [{"playlist": "default"}],
        },
    )
    monkeypatch.setattr(config_ui, "_load_active_style_config", lambda: {"screens": {}})
    monkeypatch.setattr(
        config_ui,
        "_build_screen_entries",
        lambda config, style: [
            {"id": "date", "frequency": 1, "background": "", "alt_screen": "", "alt_frequency": ""}
        ],
    )

    client = config_ui.app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "const selectableScreenIds = " in html
    assert '"date"' in html
    assert 'const serverPlaylists = [{"id": "default", "name": "Default"}]' in html
    assert 'const serverPlaylistAssignments = {"date": "default"}' in html
    assert 'id="verticalSpeedAdjustment"' in html
    assert 'value="0.0"' in html
    expected_scroll_state = (
        'let scrollSettings = {"smoothness": 1.0, "speed": 1.0, '
        '"vertical_speed_adjustment": 0.0}'
    )
    assert expected_scroll_state in html
    assert "speed: clampNumber(scrollSettings.speed, 1, 0.25, 3)" in html

    frequency_help = (
        "0 disables the base screen. Enabled screens display once at startup. "
        "After startup: 1 = every pass, 2 = every second pass."
    )
    alternate_frequency_help = (
        "Counts the base screen's normal scheduled appearances; startup hydration is excluded."
    )
    # Each hint is rendered for initial rows and retained in the client-side row builder.
    assert html.count(frequency_help) == 2
    assert html.count(alternate_frequency_help) == 2


def test_screen_config_draft_includes_and_restores_scroll_settings(monkeypatch):
    monkeypatch.setattr(config_ui, "_load_active_config", lambda: {"screens": {"date": 1}})
    monkeypatch.setattr(config_ui, "_load_active_style_config", lambda: {"screens": {}})
    monkeypatch.setattr(
        config_ui,
        "_build_screen_entries",
        lambda config, style: [
            {
                "id": "date",
                "frequency": 1,
                "background": "#000000",
                "alt_screen": "",
                "alt_frequency": "",
            }
        ],
    )

    response = config_ui.app.test_client().get("/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "scroll: currentScrollSettings()," in html
    assert "applyScrollSettings(draft.scroll);" in html


def test_load_defaults_preserves_scroll_without_explicit_default_settings(monkeypatch):
    monkeypatch.setattr(
        config_ui,
        "_load_default_screens_bundle",
        lambda profile: (
            {"screens": {"date": 1}, "scroll": config_ui._normalize_scroll_settings(None)},
            {"screens": {}},
            {"screens": {}},
            False,
        ),
    )

    response = config_ui.app.test_client().get("/api/screens/defaults?profile=large")

    assert response.status_code == 200
    assert response.get_json()["has_explicit_scroll"] is False


def test_screen_config_page_renders_alt_screen_clear_control(monkeypatch):
    monkeypatch.setattr(config_ui, "_load_active_config", lambda: {"screens": {"date": 1}})
    monkeypatch.setattr(config_ui, "_load_active_style_config", lambda: {"screens": {}})
    monkeypatch.setattr(
        config_ui,
        "_build_screen_entries",
        lambda config, style: [
            {
                "id": "date",
                "frequency": 1,
                "background": "#000000",
                "alt_screen": "inside",
                "alt_frequency": "1",
            }
        ],
    )

    client = config_ui.app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'class="alt-screen-clear"' in html
    assert "Clear alternate screens" in html


def test_screen_config_page_renders_alt_screen_dropdown(monkeypatch):
    monkeypatch.setattr(config_ui, "_load_active_config", lambda: {"screens": {"date": 1, "inside": 1}})
    monkeypatch.setattr(config_ui, "_load_active_style_config", lambda: {"screens": {}})
    monkeypatch.setattr(
        config_ui,
        "_build_screen_entries",
        lambda config, style: [
            {
                "id": "date",
                "frequency": 1,
                "background": "#000000",
                "alt_screen": "inside",
                "alt_frequency": "1",
            }
        ],
    )

    client = config_ui.app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert '<select class="alt-screen-input">' in html
    assert '<option value="">No alternate</option>' in html
    assert '<option value="inside" selected>inside</option>' in html
    assert '<input type="text" list="screenIds"' not in html


def test_screen_config_page_labels_multiple_alternate_screens(monkeypatch):
    monkeypatch.setattr(config_ui, "_load_active_config", lambda: {"screens": {"date": 1}})
    monkeypatch.setattr(config_ui, "_load_active_style_config", lambda: {"screens": {}})
    monkeypatch.setattr(
        config_ui,
        "_build_screen_entries",
        lambda config, style: [
            {
                "id": "date",
                "frequency": 1,
                "background": "#000000",
                "alt_screen": "inside, weather1",
                "alt_frequency": "2",
            }
        ],
    )

    client = config_ui.app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert '<option value="inside, weather1" selected>Multiple</option>' in html
    assert 'new Option("Multiple", selectedValue, true, true)' in html


def test_screen_config_page_no_longer_renders_quad_mode_controls(monkeypatch):
    monkeypatch.setattr(config_ui, "_load_active_config", lambda: {"screens": {"date": 1}})
    monkeypatch.setattr(config_ui, "_load_active_style_config", lambda: {"screens": {}})
    monkeypatch.setattr(
        config_ui,
        "_build_screen_entries",
        lambda config, style: [
            {
                "id": "date",
                "frequency": 1,
                "background": "#000000",
                "alt_screen": "",
                "alt_frequency": "",
            }
        ],
    )

    client = config_ui.app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Enable quad mode" not in html
    assert 'id="quadEnabled"' not in html
    assert 'id="quadScrollSpeed"' not in html
    assert 'id="addQuadPageBtn"' not in html
    assert 'id="quadPagesContainer"' not in html


def test_build_screen_entries_includes_extra_seconds():
    entries = config_ui._build_screen_entries(
        {"screens": {"date": {"frequency": 1, "extra_seconds": 3}}},
        {"screens": {}},
    )

    date_entry = next(entry for entry in entries if entry["id"] == "date")
    assert date_entry["extra_seconds"] == 3


def test_build_config_persists_extra_seconds():
    config = config_ui._build_config(
        [
            {
                "id": "date",
                "frequency": 1,
                "extra_seconds": 5,
                "alt_screen": "",
                "alt_frequency": "",
            }
        ]
    )

    assert config["screens"]["date"] == {"frequency": 1, "extra_seconds": 5}


def test_build_config_persists_hide_after_fields():
    config = config_ui._build_config(
        [
            {
                "id": "date",
                "frequency": 1,
                "extra_seconds": 0,
                "alt_screen": "",
                "alt_frequency": "",
                "hide_after_enabled": True,
                "hide_after_at": "2026-04-06T12:00",
            }
        ]
    )

    assert config["screens"]["date"] == {
        "frequency": 1,
        "hide_after_enabled": True,
        "hide_after_at": "2026-04-06T12:00",
    }


def test_build_selectable_screen_ids_prioritizes_rendered_entries():
    screen_ids = config_ui._build_selectable_screen_ids(
        [
            {"id": "astronomical"},
            {"id": "date"},
            {"id": "astronomical"},
        ]
    )

    assert screen_ids[:2] == ["astronomical", "date"]


def test_screen_config_page_uses_rendered_entries_for_selectable_screen_ids(monkeypatch):
    monkeypatch.setattr(config_ui, "_load_active_config", lambda: {"screens": {"date": 1}})
    monkeypatch.setattr(config_ui, "_load_active_style_config", lambda: {"screens": {}})
    monkeypatch.setattr(
        config_ui,
        "_build_screen_entries",
        lambda config, style: [
            {"id": "astronomical", "frequency": 1, "background": "#000000", "alt_screen": "", "alt_frequency": ""}
        ],
    )

    client = config_ui.app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'const selectableScreenIds = ["astronomical"' in html


def test_screen_config_page_merges_restored_draft_with_server_catalog(monkeypatch):
    monkeypatch.setattr(config_ui, "_load_active_config", lambda: {"screens": {"date": 1}})
    monkeypatch.setattr(config_ui, "_load_active_style_config", lambda: {"screens": {}})
    monkeypatch.setattr(
        config_ui,
        "_build_screen_entries",
        lambda config, style: [
            {"id": "date", "frequency": 1, "background": "#000000", "alt_screen": "", "alt_frequency": ""},
            {"id": "astronomical", "frequency": 0, "background": "#000000", "alt_screen": "", "alt_frequency": ""},
        ],
    )

    client = config_ui.app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "function mergeScreensWithCatalog(baseScreens, preferredScreens)" in html
    assert "initialScreens = mergeScreensWithCatalog(initialScreens, draft.screens);" in html
