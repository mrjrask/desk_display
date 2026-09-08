import config_ui


def test_diagnostic_playback_api_starts_and_stops(monkeypatch):
    state = {"screen_id": None}
    monkeypatch.setattr(config_ui, "load_diagnostic_screen", lambda: state["screen_id"])

    def save(screen_id):
        state["screen_id"] = screen_id
        return screen_id

    monkeypatch.setattr(config_ui, "save_diagnostic_screen", save)
    client = config_ui.app.test_client()

    response = client.post("/api/diagnostic-playback", json={"screen_id": "date"})
    assert response.status_code == 200
    assert response.get_json()["screen_id"] == "date"

    response = client.post("/api/diagnostic-playback", json={"screen_id": None})
    assert response.status_code == 200
    assert response.get_json()["screen_id"] is None
