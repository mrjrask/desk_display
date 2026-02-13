import importlib


def test_auth_redirects_when_not_logged_in(monkeypatch):
    monkeypatch.setenv("SCREEN_AUTH_ENABLED", "1")
    module = importlib.import_module("config_ui")
    config_ui = importlib.reload(module)

    client = config_ui.app.test_client()
    response = client.get("/")

    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_auth_api_requires_login(monkeypatch):
    monkeypatch.setenv("SCREEN_AUTH_ENABLED", "1")
    module = importlib.import_module("config_ui")
    config_ui = importlib.reload(module)

    client = config_ui.app.test_client()
    response = client.get("/api/screens")

    assert response.status_code == 401
    assert response.get_json()["error"] == "authentication required"
