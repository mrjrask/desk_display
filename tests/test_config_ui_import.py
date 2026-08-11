import json

import config_ui


def test_import_screens_accepts_entries_payload(monkeypatch):
    saved = {}

    monkeypatch.setattr(config_ui, "_load_active_style_config", lambda: {"screens": {}})
    monkeypatch.setattr(config_ui, "_load_active_layouts_config", lambda: {"screens": {"quad": {"enabled": False, "scroll_speed": 1.0, "pages": [{"tiles": ["date", "weather1", "weather hourly", "inside"]}]}}})
    monkeypatch.setattr(config_ui, "build_scheduler", lambda config: None)
    monkeypatch.setattr(config_ui, "_save_config", lambda config: saved.setdefault("config", config))
    monkeypatch.setattr(config_ui, "_save_layouts_config", lambda layouts: saved.setdefault("layouts", layouts))
    monkeypatch.setattr(
        config_ui,
        "_build_screen_entries",
        lambda config, style: [{"id": "date", "frequency": 3}],
    )

    client = config_ui.app.test_client()
    response = client.post(
        "/api/screens/import",
        json={
            "config": {
                "screens": [
                    {
                        "id": "date",
                        "frequency": "3",
                        "alt_screen": "",
                        "alt_frequency": "",
                    }
                ],
                "playlists": {"default": {"label": "Default", "steps": [{"screen": "date"}]}},
                "sequence": [{"playlist": "default"}],
            }
        },
    )

    payload = response.get_json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert saved["config"]["screens"] == {"date": 3}
    assert saved["config"]["playlists"] == {
        "default": {"label": "Default", "steps": [{"screen": "date"}]}
    }
    assert saved["config"]["sequence"] == [{"playlist": "default"}]


def test_hidden_doubleheader_screens_are_not_configurable():
    entries = config_ui._build_screen_entries(
        {"screens": {"date": 1, "cubs next 2": 5, "sox last 2": 5}},
        {"screens": {}},
    )
    entry_ids = [entry["id"] for entry in entries]

    assert "date" in entry_ids
    assert "cubs next 2" not in entry_ids
    assert "sox last 2" not in entry_ids

    config = config_ui._build_config(
        [
            {"id": "date", "frequency": 1, "alt_screen": "", "alt_frequency": ""},
            {"id": "cubs next 2", "frequency": 5, "alt_screen": "", "alt_frequency": ""},
        ]
    )
    assert config["screens"] == {"date": 1}


def test_import_screens_accepts_export_payload_with_string_frequencies(monkeypatch):
    saved = {}

    monkeypatch.setattr(config_ui, "_load_active_style_config", lambda: {"screens": {}})
    monkeypatch.setattr(config_ui, "_load_active_layouts_config", lambda: {"screens": {"quad": {"enabled": False, "scroll_speed": 1.0, "pages": [{"tiles": ["date", "weather1", "weather hourly", "inside"]}]}}})
    monkeypatch.setattr(config_ui, "build_scheduler", lambda config: None)
    monkeypatch.setattr(config_ui, "_save_config", lambda config: saved.setdefault("config", config))
    monkeypatch.setattr(config_ui, "_save_layouts_config", lambda layouts: saved.setdefault("layouts", layouts))
    monkeypatch.setattr(
        config_ui,
        "_build_screen_entries",
        lambda config, style: [{"id": "date", "frequency": 3}],
    )

    client = config_ui.app.test_client()
    response = client.post(
        "/api/screens/import",
        json={
            "config": {
                "screens": {
                    "date": "3",
                    "NHL Standings West": {
                        "frequency": "3",
                        "alt": {"screen": "NHL Standings West v2", "frequency": "2"},
                    },
                },
                "playlists": {"default": {"label": "Default", "steps": [{"screen": "date"}]}},
                "sequence": [{"playlist": "default"}],
            },
            "layouts": {
                "screens": {
                    "quad": {
                        "enabled": True,
                        "scroll_speed": 1.5,
                        "pages": [
                            {"tiles": ["date", "date", "weather1", "inside"]},
                            {"tiles": ["nixie", "nixie", "nixie", "nixie"]},
                        ],
                    }
                }
            },
        },
    )

    payload = response.get_json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert saved["config"]["screens"]["date"] == 3
    assert saved["config"]["screens"]["NHL Standings West"]["frequency"] == 3
    assert saved["config"]["screens"]["NHL Standings West"]["alt"]["frequency"] == 2
    assert saved["layouts"]["screens"]["quad"]["enabled"] is True
    assert saved["layouts"]["screens"]["quad"]["scroll_speed"] == 1.5
    assert saved["layouts"]["screens"]["quad"]["pages"][0]["tiles"] == ["date", "date", "weather1", "inside"]


def test_import_screens_preserves_hide_after_fields(monkeypatch):
    saved = {}

    monkeypatch.setattr(config_ui, "_load_active_style_config", lambda: {"screens": {}})
    monkeypatch.setattr(config_ui, "_load_active_layouts_config", lambda: {"screens": {"quad": {"enabled": False, "scroll_speed": 1.0, "pages": [{"tiles": ["date", "weather1", "weather hourly", "inside"]}]}}})
    monkeypatch.setattr(config_ui, "build_scheduler", lambda config: None)
    monkeypatch.setattr(config_ui, "_save_config", lambda config: saved.setdefault("config", config))
    monkeypatch.setattr(config_ui, "_save_layouts_config", lambda layouts: saved.setdefault("layouts", layouts))
    monkeypatch.setattr(
        config_ui,
        "_build_screen_entries",
        lambda config, style: [{"id": "date", "frequency": 3}],
    )

    client = config_ui.app.test_client()
    response = client.post(
        "/api/screens/import",
        json={
            "config": {
                "screens": {
                    "date": {
                        "frequency": "3",
                        "hide_after_enabled": True,
                        "hide_after_at": "2026-05-15T08:30",
                    }
                }
            }
        },
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert saved["config"]["screens"]["date"]["frequency"] == 3
    assert saved["config"]["screens"]["date"]["hide_after_enabled"] is True
    assert saved["config"]["screens"]["date"]["hide_after_at"] == "2026-05-15T08:30"


def test_save_screens_persists_quad_pages(monkeypatch):
    saved = {}

    monkeypatch.setattr(config_ui, "_load_active_style_config", lambda: {"screens": {}})
    monkeypatch.setattr(config_ui, "build_scheduler", lambda config: None)
    monkeypatch.setattr(config_ui, "_save_config", lambda config: saved.setdefault("config", config))
    monkeypatch.setattr(config_ui, "_save_layouts_config", lambda layouts: saved.setdefault("layouts", layouts))
    monkeypatch.setattr(
        config_ui,
        "_build_screen_entries",
        lambda config, style: [
            {
                "id": "date",
                "frequency": 1,
                "alt_screen": "",
                "alt_frequency": "",
            }
        ],
    )

    client = config_ui.app.test_client()
    response = client.post(
        "/api/screens",
        json={
            "screens": [
                {
                    "id": "date",
                    "frequency": 1,
                    "alt_screen": "",
                    "alt_frequency": "",
                }
            ],
            "quad_enabled": True,
            "quad_scroll_speed": 2.0,
            "quad_pages": [
                {"tiles": ["date", "date", "weather1", "inside"]},
                {"tiles": ["nixie", "inside", "inside", "weather2"]},
            ],
        },
    )

    payload = response.get_json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["screens"] == [
        {
            "id": "date",
            "frequency": 1,
            "alt_screen": "",
            "alt_frequency": "",
        }
    ]
    assert payload["quad_enabled"] is True
    assert payload["quad_scroll_speed"] == 2.0
    assert payload["quad_pages"] == [
        {"tiles": ["date", "date", "weather1", "inside"]},
        {"tiles": ["nixie", "inside", "inside", "weather2"]},
    ]
    assert saved["layouts"] == {
        "screens": {
            "quad": {
                "enabled": True,
                "scroll_speed": 2.0,
                "pages": [
                    {"tiles": ["date", "date", "weather1", "inside"]},
                    {"tiles": ["nixie", "inside", "inside", "weather2"]},
                ],
            }
        }
    }


def test_save_screens_persists_playlists_and_sequence(monkeypatch):
    saved = {}

    monkeypatch.setattr(config_ui, "_load_active_style_config", lambda: {"screens": {}})
    monkeypatch.setattr(config_ui, "build_scheduler", lambda config: None)
    monkeypatch.setattr(config_ui, "_save_config", lambda config: saved.setdefault("config", config))
    monkeypatch.setattr(config_ui, "_save_layouts_config", lambda layouts: saved.setdefault("layouts", layouts))
    monkeypatch.setattr(
        config_ui,
        "_build_screen_entries",
        lambda config, style: [
            {
                "id": "date",
                "frequency": 1,
                "alt_screen": "",
                "alt_frequency": "",
            }
        ],
    )

    client = config_ui.app.test_client()
    response = client.post(
        "/api/screens",
        json={
            "screens": [
                {
                    "id": "date",
                    "frequency": 1,
                    "alt_screen": "",
                    "alt_frequency": "",
                }
            ],
            "playlists": {"default": {"label": "Default", "steps": [{"screen": "date"}]}},
            "sequence": [{"playlist": "default"}],
            "quad_enabled": False,
            "quad_pages": [],
        },
    )

    payload = response.get_json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert saved["config"]["playlists"] == {
        "default": {"label": "Default", "steps": [{"screen": "date"}]}
    }
    assert saved["config"]["sequence"] == [{"playlist": "default"}]


def test_save_screens_preserves_existing_layouts_when_quad_payload_missing(monkeypatch):
    saved = {}
    existing_layouts = {
        "screens": {
            "quad": {
                "enabled": True,
                "scroll_speed": 1.5,
                "pages": [
                    {"tiles": ["date", "inside", "weather1", "weather hourly"]},
                    {"tiles": ["nixie", "inside", "inside", "weather2"]},
                ],
            }
        }
    }

    monkeypatch.setattr(config_ui, "_load_active_style_config", lambda: {"screens": {}})
    monkeypatch.setattr(config_ui, "_load_active_layouts_config", lambda: existing_layouts)
    monkeypatch.setattr(config_ui, "build_scheduler", lambda config: None)
    monkeypatch.setattr(config_ui, "_save_config", lambda config: saved.setdefault("config", config))
    monkeypatch.setattr(config_ui, "_save_layouts_config", lambda layouts: saved.setdefault("layouts", layouts))
    monkeypatch.setattr(
        config_ui,
        "_build_screen_entries",
        lambda config, style: [
            {
                "id": "date",
                "frequency": 1,
                "alt_screen": "",
                "alt_frequency": "",
            }
        ],
    )

    client = config_ui.app.test_client()
    response = client.post(
        "/api/screens",
        json={
            "screens": [
                {
                    "id": "date",
                    "frequency": 1,
                    "alt_screen": "",
                    "alt_frequency": "",
                }
            ],
            "playlists": {"default": {"label": "Default", "steps": [{"screen": "date"}]}},
            "sequence": [{"playlist": "default"}],
        },
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert saved["layouts"] == existing_layouts


def test_save_screens_preserves_nhl_standings_v2_playlist_assignments(monkeypatch):
    saved = {}

    monkeypatch.setattr(config_ui, "_load_active_style_config", lambda: {"screens": {}})
    monkeypatch.setattr(config_ui, "build_scheduler", lambda config: None)
    monkeypatch.setattr(config_ui, "_save_config", lambda config: saved.setdefault("config", config))
    monkeypatch.setattr(config_ui, "_save_layouts_config", lambda layouts: saved.setdefault("layouts", layouts))
    monkeypatch.setattr(
        config_ui,
        "_build_screen_entries",
        lambda config, style: [
            {
                "id": "NHL Standings West v2",
                "frequency": 1,
                "alt_screen": "",
                "alt_frequency": "",
            },
            {
                "id": "NHL Standings East v2",
                "frequency": 1,
                "alt_screen": "",
                "alt_frequency": "",
            },
        ],
    )

    client = config_ui.app.test_client()
    response = client.post(
        "/api/screens",
        json={
            "screens": [
                {
                    "id": "NHL Standings West v2",
                    "frequency": 1,
                    "alt_screen": "",
                    "alt_frequency": "",
                },
                {
                    "id": "NHL Standings East v2",
                    "frequency": 1,
                    "alt_screen": "",
                    "alt_frequency": "",
                },
            ],
            "playlists": {
                "nhl": {
                    "label": "NHL",
                    "steps": [
                        {"screen": "NHL Standings West v2"},
                        {"screen": "NHL Standings East v2"},
                    ],
                }
            },
            "sequence": [{"playlist": "nhl"}],
            "quad_enabled": False,
            "quad_pages": [],
        },
    )

    assert response.status_code == 200
    assert saved["config"]["screens"]["NHL Standings West v2"] == 1
    assert saved["config"]["screens"]["NHL Standings East v2"] == 1
    assert saved["config"]["playlists"]["nhl"]["steps"] == [
        {"screen": "NHL Standings West v2"},
        {"screen": "NHL Standings East v2"},
    ]


def test_build_layouts_clamps_quad_scroll_speed():
    layouts = config_ui._build_layouts(
        {
            "quad_enabled": True,
            "quad_scroll_speed": 99,
            "quad_pages": [{"tiles": ["date", "nixie", "inside", "weather1"]}],
        }
    )
    assert layouts["screens"]["quad"]["scroll_speed"] == 3.0


def test_default_screens_endpoint_reads_repo_backed_file_each_request(tmp_path, monkeypatch):
    default_path = tmp_path / "default_screens.json"
    style_path = tmp_path / "screens_style.json"
    layouts_path = tmp_path / "screens_layouts.json"
    default_path.write_text(
        json.dumps(
            {
                "screens": {"date": 2},
                "playlists": {"daily": {"label": "Daily", "steps": [{"screen": "date"}]}},
                "sequence": [{"playlist": "daily"}],
            }
        ),
        encoding="utf-8",
    )
    style_path.write_text('{"screens": {}}', encoding="utf-8")
    layouts_path.write_text(
        '{"screens": {"quad": {"enabled": false, "scroll_speed": 1, "pages": []}}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(config_ui, "DEFAULT_SCREENS_PATH", str(default_path))
    monkeypatch.setattr(config_ui, "STYLE_CONFIG_PATH", str(style_path))
    monkeypatch.setattr(config_ui, "LAYOUTS_CONFIG_PATH", str(layouts_path))
    monkeypatch.setattr(
        config_ui,
        "_build_screen_entries",
        lambda config, style: [{"id": "date", "frequency": config["screens"]["date"]}],
    )

    client = config_ui.app.test_client()
    first_payload = client.get("/api/screens/defaults").get_json()

    default_path.write_text(
        json.dumps(
            {
                "screens": {"date": 7},
                "playlists": {"nightly": {"label": "Nightly", "steps": [{"screen": "date"}]}},
                "sequence": [{"playlist": "nightly"}],
            }
        ),
        encoding="utf-8",
    )
    second_payload = client.get("/api/screens/defaults").get_json()

    assert first_payload["config"]["screens"] == {"date": 2}
    assert first_payload["playlists"] == [{"id": "daily", "name": "Daily"}]
    assert first_payload["playlist_assignments"] == {"date": "daily"}
    assert second_payload["config"]["screens"] == {"date": 7}
    assert second_payload["playlists"] == [{"id": "nightly", "name": "Nightly"}]


def test_default_screens_endpoint_accepts_export_payload(tmp_path, monkeypatch):
    default_path = tmp_path / "default_screens.json"
    default_path.write_text(
        json.dumps(
            {
                "config": {
                    "screens": {"date": 4},
                    "playlists": {"default": {"label": "Default", "steps": [{"screen": "date"}]}},
                    "sequence": [{"playlist": "default"}],
                },
                "layouts": {
                    "screens": {
                        "quad": {
                            "enabled": True,
                            "scroll_speed": 1.5,
                            "pages": [{"tiles": ["date", "inside", "weather1", "weather hourly"]}],
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(config_ui, "DEFAULT_SCREENS_PATH", str(default_path))

    client = config_ui.app.test_client()
    response = client.get("/api/screens/defaults")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["config"]["screens"] == {"date": 4}
    assert payload["screens"][0]["id"] == "date"
    assert payload["playlists"] == [{"id": "default", "name": "Default"}]
    assert payload["quad_enabled"] is True
    assert payload["quad_scroll_speed"] == 1.5
