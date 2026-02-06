import config_ui


def test_import_screens_accepts_entries_payload(monkeypatch):
    saved = {}

    monkeypatch.setattr(config_ui, "_load_active_style_config", lambda: {"screens": {}})
    monkeypatch.setattr(config_ui, "build_scheduler", lambda config: None)
    monkeypatch.setattr(config_ui, "_save_config", lambda config: saved.setdefault("config", config))
    monkeypatch.setattr(config_ui, "_save_style_config", lambda style: saved.setdefault("style", style))
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
    assert saved["style"] == {"screens": {"date": {"background": "#112233"}}}
