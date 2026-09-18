import importlib
from pathlib import Path

import pytest

import config_ui

MALICIOUS_ID = '<img src=x onerror="window.pwned=true">'
HISTORICAL_SESSION_SECRET = "desk-display-config-ui"


def _prepare_import_request(monkeypatch):
    saved = []
    monkeypatch.setattr(config_ui, "_load_active_layouts_config", lambda: {"screens": {}})
    monkeypatch.setattr(config_ui, "_save_config_bundle", lambda *args: saved.append(args))
    return config_ui.app.test_client(), saved


def test_password_remains_session_secret_fallback(monkeypatch):
    monkeypatch.delenv("SCREEN_SESSION_SECRET", raising=False)
    monkeypatch.setenv("SCREEN_UI_PASSWORD", "configured-password")

    reloaded = importlib.reload(config_ui)

    assert reloaded.app.secret_key == "configured-password"


def test_generated_session_secret_is_non_empty_random_and_stable(monkeypatch):
    monkeypatch.delenv("SCREEN_SESSION_SECRET", raising=False)
    monkeypatch.delenv("SCREEN_UI_PASSWORD", raising=False)

    reloaded = importlib.reload(config_ui)
    generated_secret = reloaded.app.secret_key
    reloaded.app.test_client().get("/")

    assert generated_secret
    assert generated_secret != HISTORICAL_SESSION_SECRET
    assert reloaded.app.secret_key == generated_secret


def test_build_config_rejects_unknown_primary_screen_id():
    with pytest.raises(ValueError, match="Unknown screen id"):
        config_ui._build_config(
            [
                {
                    "id": MALICIOUS_ID,
                    "frequency": 1,
                    "alt_screen": "",
                    "alt_frequency": "",
                }
            ]
        )


def test_build_config_rejects_unknown_alternate_screen_id():
    with pytest.raises(ValueError, match="Unknown screen id"):
        config_ui._build_config(
            [
                {
                    "id": "date",
                    "frequency": 1,
                    "alt_screen": MALICIOUS_ID,
                    "alt_frequency": 1,
                }
            ]
        )


@pytest.mark.parametrize(
    "screens",
    [
        {MALICIOUS_ID: 1},
        {"date": {"frequency": 1, "alt": {"screen": MALICIOUS_ID, "frequency": 1}}},
        {"date": {"frequency": 1, "alt": {"screen": ["nixie", MALICIOUS_ID], "frequency": 1}}},
    ],
)
def test_import_rejects_unknown_ids_without_persisting(monkeypatch, screens):
    client, saved = _prepare_import_request(monkeypatch)

    response = client.post("/api/screens/import", json={"config": {"screens": screens}})

    assert response.status_code == 400
    assert "Unknown screen id" in response.get_json()["error"]
    assert saved == []


def test_import_rejects_unknown_playlist_screen_without_persisting(monkeypatch):
    client, saved = _prepare_import_request(monkeypatch)

    response = client.post(
        "/api/screens/import",
        json={
            "config": {
                "screens": {"date": 1},
                "playlists": {
                    "default": {
                        "label": "Default",
                        "steps": [{"screen": MALICIOUS_ID}],
                    }
                },
            }
        },
    )

    assert response.status_code == 400
    assert "Unknown screen id" in response.get_json()["error"]
    assert saved == []


def test_dynamic_screen_rows_use_dom_apis_instead_of_html_interpolation():
    template = (
        Path(config_ui.__file__).resolve().parent / "templates" / "screen_config.html"
    ).read_text(encoding="utf-8")
    build_row = template.split("function buildRow(screen) {", 1)[1].split(
        "function updateZeroFrequencyVisibility()", 1
    )[0]

    assert "innerHTML" not in build_row
    assert "populateAltScreenOptions" in build_row
    assert 'makeElement("div", "screen-id", screen.id)' in build_row
