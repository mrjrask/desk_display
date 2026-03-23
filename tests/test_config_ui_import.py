import config_ui


def test_import_screens_accepts_entries_payload(monkeypatch):
    saved = {}

    monkeypatch.setattr(config_ui, "_load_active_style_config", lambda: {"screens": {}})
    monkeypatch.setattr(config_ui, "_load_active_layouts_config", lambda: {"screens": {"quad": {"enabled": False, "pages": [{"tiles": ["date", "weather1", "weather hourly", "inside"]}]}}})
    monkeypatch.setattr(config_ui, "build_scheduler", lambda config: None)
    monkeypatch.setattr(config_ui, "_save_config", lambda config: saved.setdefault("config", config))
    monkeypatch.setattr(config_ui, "_save_style_config", lambda style: saved.setdefault("style", style))
    monkeypatch.setattr(config_ui, "_save_layouts_config", lambda layouts: saved.setdefault("layouts", layouts))
    monkeypatch.setattr(
        config_ui,
        "_build_screen_entries",
        lambda config, style: [{"id": "date", "frequency": 3, "background": "#112233"}],
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
                        "background": "#112233",
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
    assert saved["style"] == {"screens": {"date": {"background": "#112233"}}
    }


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
    monkeypatch.setattr(config_ui, "_load_active_layouts_config", lambda: {"screens": {"quad": {"enabled": False, "pages": [{"tiles": ["date", "weather1", "weather hourly", "inside"]}]}}})
    monkeypatch.setattr(config_ui, "build_scheduler", lambda config: None)
    monkeypatch.setattr(config_ui, "_save_config", lambda config: saved.setdefault("config", config))
    monkeypatch.setattr(config_ui, "_save_style_config", lambda style: saved.setdefault("style", style))
    monkeypatch.setattr(config_ui, "_save_layouts_config", lambda layouts: saved.setdefault("layouts", layouts))
    monkeypatch.setattr(
        config_ui,
        "_build_screen_entries",
        lambda config, style: [{"id": "date", "frequency": 3, "background": "#112233"}],
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
            "style": {"screens": {"date": {"background": "#112233"}}},
            "layouts": {
                "screens": {
                    "quad": {
                        "enabled": True,
                        "pages": [
                            {"tiles": ["date", "date", "weather1", "inside"]},
                            {"tiles": ["time", "time", "time", "time"]},
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
    assert saved["style"] == {"screens": {"date": {"background": "#112233"}}}
    assert saved["layouts"]["screens"]["quad"]["enabled"] is True
    assert saved["layouts"]["screens"]["quad"]["pages"][0]["tiles"] == ["date", "date", "weather1", "inside"]


def test_save_screens_persists_quad_pages(monkeypatch):
    saved = {}

    monkeypatch.setattr(config_ui, "_load_active_style_config", lambda: {"screens": {}})
    monkeypatch.setattr(config_ui, "build_scheduler", lambda config: None)
    monkeypatch.setattr(config_ui, "_save_config", lambda config: saved.setdefault("config", config))
    monkeypatch.setattr(config_ui, "_save_style_config", lambda style: saved.setdefault("style", style))
    monkeypatch.setattr(config_ui, "_save_layouts_config", lambda layouts: saved.setdefault("layouts", layouts))
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
    response = client.post(
        "/api/screens",
        json={
            "screens": [
                {
                    "id": "date",
                    "frequency": 1,
                    "background": "#000000",
                    "alt_screen": "",
                    "alt_frequency": "",
                }
            ],
            "quad_enabled": True,
            "quad_pages": [
                {"tiles": ["date", "date", "weather1", "inside"]},
                {"tiles": ["time", "inside", "inside", "weather2"]},
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
            "background": "#000000",
            "alt_screen": "",
            "alt_frequency": "",
        }
    ]
    assert payload["quad_enabled"] is True
    assert payload["quad_pages"] == [
        {"tiles": ["date", "date", "weather1", "inside"]},
        {"tiles": ["time", "inside", "inside", "weather2"]},
    ]
    assert saved["layouts"] == {
        "screens": {
            "quad": {
                "enabled": True,
                "pages": [
                    {"tiles": ["date", "date", "weather1", "inside"]},
                    {"tiles": ["time", "inside", "inside", "weather2"]},
                ],
            }
        }
    }


def test_save_screens_persists_playlists_and_sequence(monkeypatch):
    saved = {}

    monkeypatch.setattr(config_ui, "_load_active_style_config", lambda: {"screens": {}})
    monkeypatch.setattr(config_ui, "build_scheduler", lambda config: None)
    monkeypatch.setattr(config_ui, "_save_config", lambda config: saved.setdefault("config", config))
    monkeypatch.setattr(config_ui, "_save_style_config", lambda style: saved.setdefault("style", style))
    monkeypatch.setattr(config_ui, "_save_layouts_config", lambda layouts: saved.setdefault("layouts", layouts))
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
    response = client.post(
        "/api/screens",
        json={
            "screens": [
                {
                    "id": "date",
                    "frequency": 1,
                    "background": "#000000",
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
