"""Live clock faces for clients that show images but cannot draw the time."""
from __future__ import annotations

import io

import pytest
from PIL import Image

pytest.importorskip("flask")

import display_server  # noqa: E402
from display_profiles import PROFILE_PRESETS  # noqa: E402
from remote_display.registry import Assignment  # noqa: E402
from remote_display.server_rendering import LiveClock  # noqa: E402

TOKEN = "server-token-" + "s" * 32


def caps(client_id="mirror", profile="display_hat_mini"):
    preset = PROFILE_PRESETS[profile]
    return {
        "type": "client_capabilities", "version": 1, "protocol_version": 1,
        "client_software_version": "0.1", "client_id": client_id, "display_profile": profile,
        "logical_width": preset.width, "logical_height": preset.height, "image_formats": ["PNG"],
        "color_modes": [preset.color_mode], "render_package_versions": [1], "supports_animation": False,
    }


def make_api(tmp_path, live_clock):
    config = display_server.DisplayServerConfig(enrollment="shared", auth_token=TOKEN,
                                                admin_token="admin-token-" + "a" * 32,
                                                artifact_dir=tmp_path / "artifacts")
    assignments = {"mirror": Assignment("default", "rev-1", ("date", "nixie"))}
    app = display_server.create_app(config, assignments=assignments.get, live_clock=live_clock)
    app.config["TESTING"] = True
    return app.test_client()


def register(api):
    response = api.post("/api/v1/register", json={"capabilities": caps()},
                        headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 201, response.get_json()
    return response.get_json()


def clock_get(api, credential, path):
    return api.get(f"/api/v1/clients/mirror/clock/{path}", headers={"Authorization": f"Bearer {credential}"})


def test_live_clock_draws_the_face_for_the_clients_profile(tmp_path):
    calls = []

    def live_clock(screen_id, profile_id, seed):
        calls.append((screen_id, profile_id, seed))
        return LiveClock().render(screen_id, profile_id, seed)

    api = make_api(tmp_path, live_clock)
    lease = register(api)
    assert lease["live_clock_faces"] == ["date", "nixie"]

    response = clock_get(api, lease["client_credential"], "date.png?colors=42")
    assert response.status_code == 200
    assert response.mimetype == "image/png"
    assert response.headers["Cache-Control"] == "no-store"
    image = Image.open(io.BytesIO(response.data))
    assert image.size == (320, 240)
    assert calls == [("date", "display_hat_mini", 42)]

    assert clock_get(api, lease["client_credential"], "nixie.png").status_code == 200
    assert calls[-1] == ("nixie", "display_hat_mini", None)


def test_the_colour_seed_keeps_the_date_face_colours():
    first = LiveClock().render("date", "display_hat_mini", 7)
    second = LiveClock().render("date", "display_hat_mini", 7)
    colours = lambda image: {c for _n, c in image.convert("RGB").getcolors(1 << 16)}  # noqa: E731
    assert colours(first) == colours(second)


def test_live_clock_refusals(tmp_path):
    def failing(*_args):
        raise RuntimeError("worker died")

    api = make_api(tmp_path, failing)
    credential = register(api)["client_credential"]
    assert clock_get(api, credential, "weather1.png").status_code == 404
    assert clock_get(api, credential, "date.png?colors=-1").status_code == 400
    assert clock_get(api, "not-a-lease", "date.png").status_code == 401
    response = clock_get(api, credential, "date.png")
    assert response.status_code == 503
    assert response.get_json()["error"] == "clock_unavailable"


def test_without_a_live_clock_the_server_neither_offers_nor_serves_one(tmp_path):
    api = make_api(tmp_path, None)
    lease = register(api)
    assert "live_clock_faces" not in lease
    assert clock_get(api, lease["client_credential"], "date.png").status_code == 404
