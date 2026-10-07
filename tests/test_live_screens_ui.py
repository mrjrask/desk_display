"""The Live page: every display's screenshots, fetched by the server and kept current."""
from __future__ import annotations

import base64
import urllib.error
from pathlib import Path

import pytest
from flask import Flask

import live_screens_ui
from remote_display.screenshot_uploads import ScreenshotInbox

ROOT = Path(__file__).resolve().parents[1]
NOW = 1_800_000_000.0
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class FakeUI:
    """Stands in for one display's config UI (the script's ConfigUI)."""

    def __init__(self, listing=None, files=None, error=None):
        self.listing = listing
        self.files = files or {}
        self.error = error
        self.calls: list[str] = []

    def get_json(self, path):
        self.calls.append(path)
        if self.error is not None:
            raise self.error
        return self.listing

    def get(self, path):
        self.calls.append(path)
        if path not in self.files:
            raise urllib.error.HTTPError(path, 404, "missing", None, None)
        return self.files[path], "image/png"


def row(client_id, address, state="online", dims="800x480", **extra):
    return {"client_id": client_id, "address": address, "state": state, "display_profile": "hyperpixel4",
            "dimensions": dims, "friendly_name": None, **extra}


def entry(screen, path, version=NOW - 30):
    return {"id": screen, "path": path, "version": version, "timestamp": "x", "elapsed": "y", "is_stale": False}


class Clock:
    def __init__(self):
        self.value = NOW

    def __call__(self):
        return self.value


@pytest.fixture
def setup(tmp_path):
    def make(rows, uis, local=None, inbox=None):
        clock = Clock()
        app = Flask(__name__, template_folder=str(ROOT / "templates"))
        app.jinja_env.globals.update(machine_hostname="square", screenshots_only=False)
        app.extensions["desk_display_uploaded_screenshots"] = inbox or ScreenshotInbox(tmp_path / "up")
        by_host = {}

        def make_ui(base_url):
            host = base_url.split("//", 1)[1].rsplit(":", 1)[0]
            return by_host.setdefault(host, uis[host])

        live_screens_ui.register(app, client_rows=lambda: rows, local_entries=lambda: local or [],
                                 env={}, clock=clock, make_ui=make_ui)
        return app.test_client(), clock

    return make


def test_lists_each_active_display_and_proxies_its_images(setup):
    hyper = FakeUI({"screens": [entry("date", "date/date_1.png"), entry("weather1", None)]},
                   {"/screenshots/file/date/date_1.png": PNG})
    client, _ = setup([row("hyper", "10.0.0.12", current_screen="date"),
                       row("gone", "10.0.0.13", state="expired")], {"10.0.0.12": hyper})

    payload = client.get("/api/live").get_json()

    assert payload["screen_order"] == ["date"]
    assert payload["skipped"] == [{"client_id": "gone", "label": "gone", "state": "expired"}]
    (display,) = payload["displays"]
    assert (display["width"], display["height"], display["current_screen"]) == (800, 480, "date")
    shot = display["screens"]["date"]
    assert shot["captured_at"] == NOW - 30 and shot["uploaded"] is False
    assert shot["url"].startswith("/api/live/hyper/image?")

    image = client.get(shot["url"])
    assert image.status_code == 200 and image.data == PNG and image.mimetype == "image/png"
    client.get(shot["url"])
    assert hyper.calls.count("/screenshots/file/date/date_1.png") == 1  # cached by capture time


def test_image_proxy_serves_only_listed_paths(setup):
    hyper = FakeUI({"screens": [entry("date", "date/date_1.png")]}, {"/screenshots/file/date/date_1.png": PNG})
    client, _ = setup([row("hyper", "10.0.0.12")], {"10.0.0.12": hyper})
    assert client.get("/api/live/hyper/image?path=date/date_1.png&v=1").status_code == 404  # nothing listed yet
    client.get("/api/live")
    assert client.get("/api/live/hyper/image?path=../secrets.png&v=1").status_code == 404
    assert client.get("/api/live/other/image?path=date/date_1.png&v=1").status_code == 404


def test_listing_is_reused_briefly_and_a_failure_longer(setup):
    hyper = FakeUI({"screens": [entry("date", "date/date_1.png")]})
    down = FakeUI(error=urllib.error.URLError("timed out"))
    client, clock = setup([row("hyper", "10.0.0.12"), row("down", "10.0.0.14")],
                          {"10.0.0.12": hyper, "10.0.0.14": down})
    payload = client.get("/api/live").get_json()
    assert payload["displays"][1]["error"] == "could not reach http://10.0.0.14:5002 (timed out)"
    client.get("/api/live")
    assert hyper.calls.count("/api/screenshots") == 1
    clock.value += live_screens_ui.LIST_TTL_SECONDS
    client.get("/api/live")
    assert hyper.calls.count("/api/screenshots") == 2
    assert down.calls.count("/api/screenshots") == 1
    clock.value += live_screens_ui.FAILURE_TTL_SECONDS
    client.get("/api/live")
    assert down.calls.count("/api/screenshots") == 2


def test_servers_own_panel_is_read_locally(setup):
    local = [entry("date", "current/date.png", version=NOW - 5), entry("clock", None)]
    client, _ = setup([row("square-panel", "127.0.0.1", dims="720x720")], {}, local=local)
    (display,) = client.get("/api/live").get_json()["displays"]
    assert display["error"] is None
    assert display["screen_order"] == ["date"]
    assert display["screens"]["date"]["url"] == f"/screenshots/file/current/date.png?v={NOW - 5}"


def test_unreachable_display_falls_back_to_its_uploads(setup, tmp_path):
    inbox = ScreenshotInbox(tmp_path / "up", clock=lambda: NOW)
    inbox.save("hyper", "weather1", PNG, captured_at=NOW - 600)
    inbox.save("hyper", "date", PNG, captured_at=NOW - 900)
    down = FakeUI(error=urllib.error.URLError("no route to host"))
    client, _ = setup([row("hyper", "192.168.1.202")], {"192.168.1.202": down}, inbox=inbox)
    (display,) = client.get("/api/live").get_json()["displays"]
    assert display["error"] is None
    assert display["note"].endswith("showing the screenshots it uploaded to the server")
    assert display["screen_order"] == ["weather1", "date"]
    shot = display["screens"]["weather1"]
    assert shot["uploaded"] is True and shot["captured_at"] == NOW - 600
    assert shot["url"].startswith("/api/clients/hyper/uploaded-screenshots/weather1-")


def test_uploads_fill_in_screens_the_display_did_not_return(setup, tmp_path):
    inbox = ScreenshotInbox(tmp_path / "up", clock=lambda: NOW)
    inbox.save("hyper", "date", PNG, captured_at=NOW - 900)
    inbox.save("hyper", "nhl", PNG, captured_at=NOW - 900)
    hyper = FakeUI({"screens": [entry("date", "date/date_1.png")]})
    client, _ = setup([row("hyper", "10.0.0.12")], {"10.0.0.12": hyper}, inbox=inbox)
    (display,) = client.get("/api/live").get_json()["displays"]
    assert display["screen_order"] == ["date", "nhl"]
    assert display["screens"]["date"]["uploaded"] is False
    assert display["screens"]["nhl"]["uploaded"] is True


def test_include_all_tries_offline_displays(setup):
    hyper = FakeUI({"screens": []})
    client, _ = setup([row("hyper", "10.0.0.12", state="expired")], {"10.0.0.12": hyper})
    assert client.get("/api/live").get_json()["displays"] == []
    payload = client.get("/api/live?all=1").get_json()
    assert [d["client_id"] for d in payload["displays"]] == ["hyper"] and payload["skipped"] == []


def test_page_renders_with_nav(setup):
    client, _ = setup([], {})
    page = client.get("/live")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert '<a href="/live" aria-current="page">Live</a>' in html
    assert "/api/live" in html


@pytest.mark.parametrize("name", ["clients", "playlists", "screenshots", "screen_config", "stats", "client_wizard"])
def test_other_pages_link_to_live(name):
    assert '<a href="/live">Live</a>' in (ROOT / "templates" / f"{name}.html").read_text()


def test_config_ui_serves_live_page_only_on_servers():
    import config_ui

    assert "live.live_page" in config_ui.app.view_functions
    assert "live.live_page" not in config_ui._SCREENSHOTS_ONLY_ENDPOINTS


def test_image_proxy_rejects_a_non_image_answer(setup):
    class LoginPage(FakeUI):
        def get(self, path):
            return b"<html>login</html>", "text/html; charset=utf-8"

    hyper = LoginPage({"screens": [entry("date", "date/date_1.png")]})
    client, _ = setup([row("hyper", "10.0.0.12")], {"10.0.0.12": hyper})
    url = client.get("/api/live").get_json()["displays"][0]["screens"]["date"]["url"]
    assert client.get(url).status_code == 502


def test_magicmirror_display_is_flagged_and_never_fetched(setup):
    mirror = row("mirror", "10.0.0.20", capabilities={"hardware": {"model": "MagicMirror", "driver": "MMM-desk_display"}})
    hyper = FakeUI({"screens": [entry("clock", "clock.png")]}, {"clock.png": PNG})
    client, _ = setup([mirror, row("hyper", "10.0.0.12")], {"10.0.0.12": hyper})
    shown = {d["client_id"]: d for d in client.get("/api/live").get_json()["displays"]}
    assert shown["mirror"]["magicmirror"] is True and shown["mirror"]["error"] is None
    assert shown["mirror"]["screens"] == {} and shown["hyper"]["magicmirror"] is False


def test_page_has_per_display_show_hide_controls(setup):
    client, _ = setup([], {})
    html = client.get("/live").get_data(as_text=True)
    assert 'id="show-hidden"' in html and "live.visibility" in html and "d.magicmirror" in html
