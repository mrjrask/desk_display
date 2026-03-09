import config_ui


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
    assert 'const serverPlaylists = [{"id": "default", "name": "Default"}]' in html
    assert 'const serverPlaylistAssignments = {"date": "default"}' in html


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
