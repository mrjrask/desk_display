import importlib


def test_root_page_is_public(monkeypatch):
    monkeypatch.setenv("SCREEN_AUTH_ENABLED", "1")
    module = importlib.import_module("config_ui")
    config_ui = importlib.reload(module)

    client = config_ui.app.test_client()
    response = client.get("/")

    assert response.status_code == 200


def test_api_is_public(monkeypatch):
    monkeypatch.setenv("SCREEN_AUTH_ENABLED", "1")
    module = importlib.import_module("config_ui")
    config_ui = importlib.reload(module)

    client = config_ui.app.test_client()
    response = client.get("/api/screens")

    assert response.status_code == 200
