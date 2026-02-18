import importlib


def _reload_config_ui(monkeypatch):
    module = importlib.import_module("config_ui")
    return importlib.reload(module)


def test_root_page_redirects_to_login_when_password_set(monkeypatch):
    monkeypatch.setenv("SCREEN_UI_PASSWORD", "secret")
    monkeypatch.delenv("SCREEN_AUTH_ENABLED", raising=False)
    config_ui = _reload_config_ui(monkeypatch)

    client = config_ui.app.test_client()
    response = client.get("/")

    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_login_unlocks_protected_page(monkeypatch):
    monkeypatch.setenv("SCREEN_UI_PASSWORD", "secret")
    monkeypatch.delenv("SCREEN_AUTH_ENABLED", raising=False)
    config_ui = _reload_config_ui(monkeypatch)

    client = config_ui.app.test_client()
    login_response = client.post("/login", data={"password": "secret"}, follow_redirects=False)

    assert login_response.status_code == 302

    page_response = client.get("/")
    assert page_response.status_code == 200


def test_api_requires_auth_when_enabled(monkeypatch):
    monkeypatch.setenv("SCREEN_UI_PASSWORD", "secret")
    monkeypatch.delenv("SCREEN_AUTH_ENABLED", raising=False)
    config_ui = _reload_config_ui(monkeypatch)

    client = config_ui.app.test_client()
    response = client.get("/api/screens")

    assert response.status_code == 401


def test_auth_not_required_without_password(monkeypatch):
    monkeypatch.delenv("SCREEN_UI_PASSWORD", raising=False)
    monkeypatch.delenv("SCREEN_AUTH_ENABLED", raising=False)
    config_ui = _reload_config_ui(monkeypatch)

    client = config_ui.app.test_client()
    response = client.get("/")

    assert response.status_code == 200
