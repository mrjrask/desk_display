"""scripts/collect_client_screenshots.py gathers every display's screenshots into one page."""

from __future__ import annotations

import base64
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from scripts import collect_client_screenshots as ccs

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class FakeConfigUI:
    """A tiny stand-in for config_ui on one host: optional password, JSON routes, files."""

    def __init__(self, routes: dict[str, object], password: str | None = None):
        self.routes = routes
        self.password = password
        self.requests: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _authed(self) -> bool:
                return outer.password is None or "session=ok" in (self.headers.get("Cookie") or "")

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                form = urllib.parse.parse_qs(self.rfile.read(length).decode())
                if self.path == "/login" and form.get("password") == [outer.password]:
                    self.send_response(302)
                    self.send_header("Set-Cookie", "session=ok; Path=/")
                    self.send_header("Location", "/")
                else:
                    self.send_response(302)
                    self.send_header("Location", "/login")
                self.end_headers()

            def do_GET(self):
                outer.requests.append(self.path)
                if self.path in {"/", "/login"}:
                    self._send(200, b"<html></html>", "text/html")
                    return
                if not self._authed():
                    self._send(401, b'{"error": "Authentication required"}', "application/json")
                    return
                body = outer.routes.get(urllib.parse.unquote(self.path))
                if body is None:
                    self._send(404, b"", "text/plain")
                elif isinstance(body, bytes):
                    self._send(200, body, "image/png")
                else:
                    self._send(200, json.dumps(body).encode(), "application/json")

            def _send(self, status, body, content_type):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def hosts():
    started: list[FakeConfigUI] = []

    def start(routes, password=None):
        host = FakeConfigUI(routes, password)
        started.append(host)
        return host

    yield start
    for host in started:
        host.close()


def screenshots_payload(*entries):
    return {"screens": [{"id": sid, "path": path, "timestamp": "2026-09-29 12:00", "elapsed": "5s",
                         "is_stale": False} for sid, path in entries]}


def client(client_id, address, state="online", profile="hyperpixel4", dims="800x480", name=None):
    return {"client_id": client_id, "address": address, "state": state, "display_profile": profile,
            "dimensions": dims, "friendly_name": name}


def test_collects_each_screen_from_every_active_client(hosts, tmp_path):
    hyper = hosts({
        "/api/screenshots": screenshots_payload(("date", "date/date_1.png"), ("weather1", "weather1/w_1.png")),
        "/screenshots/file/date/date_1.png": PNG,
        "/screenshots/file/weather1/w_1.png": PNG,
    })
    mini = hosts({
        "/api/screenshots": screenshots_payload(("date", "current/date.png"), ("weather1", None)),
        "/screenshots/file/current/date.png": PNG,
    })
    server = hosts({"/api/clients": {"clients": [
        client("hyper", None),
        client("mini", None, profile="display_hat_mini", dims="320x240", name="Kitchen"),
        client("gone", "10.9.9.9", state="expired"),
    ]}})
    out = tmp_path / "shots.html"
    rc = ccs.main(["--server", server.url, "--output", str(out),
                   "--client-host", f"hyper={hyper.url}", "--client-host", f"mini={mini.url}"])
    assert rc == 0
    page = out.read_text()
    # Screens are headings, in playback order, each with every client's image under it.
    assert page.index("<h2>date</h2>") < page.index("<h2>weather1</h2>")
    date_section = page[page.index("<h2>date</h2>"):page.index("<h2>weather1</h2>")]
    assert date_section.count("<figure>") == 2
    assert "Kitchen (mini)" in date_section and "display_hat_mini · 320x240" in date_section
    assert "data:image/png;base64," + base64.b64encode(PNG).decode() in date_section
    # A screen one display has no screenshot for says so, and inactive clients are listed as skipped.
    assert "No screenshot from: Kitchen (mini)" in page
    assert "Skipped gone: not active (expired)" in page


def test_logs_in_when_the_config_ui_has_a_password(hosts, tmp_path):
    panel = hosts({"/api/screenshots": screenshots_payload(("date", "d.png")),
                   "/screenshots/file/d.png": PNG}, password="pw")
    server = hosts({"/api/clients": {"clients": [client("panel", None)]}}, password="pw")
    out = tmp_path / "shots.html"
    assert ccs.main(["--server", server.url, "--output", str(out), "--password", "pw",
                     "--client-host", f"panel={panel.url}"]) == 0
    assert "<h2>date</h2>" in out.read_text()


def test_unreachable_and_wrong_password_clients_are_noted_not_fatal(hosts, tmp_path):
    good = hosts({"/api/screenshots": screenshots_payload(("date", "d.png")), "/screenshots/file/d.png": PNG})
    locked = hosts({"/api/screenshots": screenshots_payload()}, password="other")
    server = hosts({"/api/clients": {"clients": [client("good", None), client("locked", None),
                                                 client("down", None)]}})
    out = tmp_path / "shots.html"
    rc = ccs.main(["--server", server.url, "--output", str(out), "--password", "pw", "--timeout", "2",
                   "--client-host", f"good={good.url}", "--client-host", f"locked={locked.url}",
                   "--client-host", "down=127.0.0.1:9"])
    assert rc == 0
    page = out.read_text()
    assert "Skipped locked:" in page and "rejected the password" in page
    assert "Skipped down: could not reach http://127.0.0.1:9" in page


def test_client_address_resolution():
    server = "http://square.local:5002"
    assert ccs.client_base_url(client("a", "10.0.0.7"), server, {}) == "http://10.0.0.7:5002"
    # A combined server's own panel heartbeats from loopback: use the server's host.
    assert ccs.client_base_url(client("square-panel", "127.0.0.1"), server, {}) == "http://square.local:5002"
    assert ccs.client_base_url(client("x", "::ffff:127.0.0.1"), server, {}) == "http://square.local:5002"
    assert ccs.client_base_url(client("b", "fe80::1"), server, {}, port=6000) == "http://[fe80::1]:6000"
    # Older servers record no address: fall back to the hostname the client ID is named after.
    assert ccs.client_base_url(client("hyper-panel", None), server, {}) == "http://hyper.local:5002"
    assert ccs.client_base_url(client("a", "10.0.0.7"), server, {"a": "pi4"}) == "http://pi4:5002"


def test_normalize_base_url():
    assert ccs.normalize_base_url("square.local") == "http://square.local:5002"
    assert ccs.normalize_base_url("https://square.local:8443/") == "https://square.local:8443"
    assert ccs.normalize_base_url("10.0.0.5:5003") == "http://10.0.0.5:5003"
