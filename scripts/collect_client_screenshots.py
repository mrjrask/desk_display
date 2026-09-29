#!/usr/bin/env python3
"""Collect every display client's latest screenshots into one HTML page.

The page groups screenshots by screen: each screen's title is a heading, and
under it is the latest screenshot from every active display client that has
one, labelled with the client's name, display profile and size. Use it to
compare how one screen looks on each display and spot inconsistencies.

How it finds the screenshots:

1. It asks the server's config UI (``/api/clients``, the data behind the
   Clients page) for the display clients. Only clients that are online (or
   briefly late with a heartbeat, "stale") are collected; the rest are listed
   at the top of the page as skipped.
2. Each client's own config UI (port 5002 by default; client-only installs
   run it in screenshots-only mode) is asked for ``/api/screenshots`` and
   each image is downloaded from ``/screenshots/file/<path>``. The client's
   address is the one it last reached the server from. A combined server's
   own panel reports 127.0.0.1, so it is fetched from the server's address.
   Servers from before client addresses were recorded give none; then the
   script tries ``<client id>.local`` (``-panel`` suffix dropped), and
   ``--client-host ID=HOST`` sets an address by hand.
3. Every image is embedded in the page, so the single ``.html`` file can be
   opened, moved or shared on its own.

If the config UI asks for a password (SCREEN_UI_PASSWORD), give it with
``--password`` or the SCREEN_UI_PASSWORD environment variable, or type it at
the prompt. The same password is tried on every display.

Uses only the Python standard library, so no venv or packages are needed.

Usage:
    python3 scripts/collect_client_screenshots.py                 # asks for the server
    python3 scripts/collect_client_screenshots.py --server square.local
    python3 scripts/collect_client_screenshots.py --server http://10.0.0.5:5002 \\
        --output ~/Desktop/screens.html
    python3 scripts/collect_client_screenshots.py --server square.local \\
        --client-host hyper=10.0.0.12 --include-inactive
"""

from __future__ import annotations

import argparse
import base64
import datetime
import getpass
import html
import http.cookiejar
import ipaddress
import json
import os
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

DEFAULT_PORT = 5002
DEFAULT_TIMEOUT = 10.0
ACTIVE_STATES = frozenset({"online", "stale"})
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif"}


class AuthRequired(Exception):
    """The config UI wants a password and none (or the wrong one) was given."""


@dataclass
class Screenshot:
    screen_id: str
    image: bytes
    content_type: str
    timestamp: Optional[str] = None
    elapsed: Optional[str] = None
    is_stale: bool = False


@dataclass
class ClientResult:
    client_id: str
    name: str
    display_profile: Optional[str]
    dimensions: Optional[str]
    base_url: Optional[str] = None
    state: Optional[str] = None
    screen_order: list[str] = field(default_factory=list)
    screenshots: dict[str, Screenshot] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def label(self) -> str:
        return self.name if self.name == self.client_id else f"{self.name} ({self.client_id})"


# ── HTTP ──────────────────────────────────────────────────────────────────────


class ConfigUI:
    """A cookie-keeping session with one config UI, logging in when asked."""

    def __init__(self, base_url: str, password: Callable[[], Optional[str]], username: str = "",
                 timeout: float = DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self._password = password
        self._username = username
        self._timeout = timeout
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )
        self._logged_in = False

    def _open(self, path: str, data: Optional[bytes] = None) -> tuple[bytes, str, str]:
        request = urllib.request.Request(self.base_url + path, data=data)
        with self._opener.open(request, timeout=self._timeout) as response:
            return response.read(), response.headers.get("Content-Type", ""), response.geturl()

    def _login(self) -> None:
        password = self._password()
        if not password:
            raise AuthRequired(f"{self.base_url} asks for a password (use --password)")
        form = urllib.parse.urlencode({"username": self._username, "password": password}).encode()
        _, _, final_url = self._open("/login", form)
        if urllib.parse.urlsplit(final_url).path.rstrip("/") == "/login":
            raise AuthRequired(f"{self.base_url} rejected the password")
        self._logged_in = True

    def get(self, path: str) -> tuple[bytes, str]:
        try:
            body, content_type, final_url = self._open(path)
        except urllib.error.HTTPError as exc:
            if exc.code != 401 or self._logged_in:
                raise
            self._login()
            body, content_type, final_url = self._open(path)
        else:
            # Pages (not /api/) redirect to the login form instead of answering 401.
            if urllib.parse.urlsplit(final_url).path.rstrip("/") == "/login" and not self._logged_in:
                self._login()
                body, content_type, final_url = self._open(path)
        return body, content_type

    def get_json(self, path: str) -> Any:
        body, _ = self.get(path)
        return json.loads(body.decode("utf-8"))


def normalize_base_url(value: str, default_port: int = DEFAULT_PORT) -> str:
    """``square.local`` -> ``http://square.local:5002``; full URLs are kept."""

    value = value.strip().rstrip("/")
    if "://" not in value:
        value = f"http://{value}"
    parts = urllib.parse.urlsplit(value)
    if not parts.hostname:
        raise ValueError(f"not a host or URL: {value!r}")
    netloc = parts.netloc if parts.port else f"{_host_for_url(parts.hostname)}:{default_port}"
    return urllib.parse.urlunsplit((parts.scheme, netloc, parts.path.rstrip("/"), "", ""))


def _host_for_url(host: str) -> str:
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def _is_loopback(host: str) -> bool:
    if host.lower() in {"localhost", "localhost.localdomain"}:
        return True
    try:
        address = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return address.is_loopback or bool(mapped and mapped.is_loopback)


def client_base_url(client: dict[str, Any], server_url: str, overrides: dict[str, str],
                    port: int = DEFAULT_PORT) -> str:
    """Where this client's own config UI answers."""

    client_id = str(client.get("client_id") or "")
    if client_id in overrides:
        return normalize_base_url(overrides[client_id], port)
    address = str(client.get("address") or "").strip()
    if address and _is_loopback(address):
        # The server's own (combined-mode) panel: same machine as the server.
        server_host = urllib.parse.urlsplit(server_url).hostname or "localhost"
        return f"{urllib.parse.urlsplit(server_url).scheme}://{_host_for_url(server_host)}:{port}"
    if address:
        return f"http://{_host_for_url(address)}:{port}"
    host = client_id[: -len("-panel")] if client_id.endswith("-panel") else client_id
    return f"http://{host}.local:{port}"


# ── Collection ────────────────────────────────────────────────────────────────


def collect_client(client: dict[str, Any], base_url: str, make_ui: Callable[[str], ConfigUI]) -> ClientResult:
    client_id = str(client.get("client_id") or "")
    result = ClientResult(
        client_id=client_id,
        name=str(client.get("friendly_name") or client_id),
        display_profile=client.get("display_profile"),
        dimensions=client.get("dimensions"),
        base_url=base_url,
        state=client.get("state"),
    )
    ui = make_ui(base_url)
    try:
        payload = ui.get_json("/api/screenshots")
    except AuthRequired as exc:
        result.error = str(exc)
        return result
    except (urllib.error.URLError, OSError, ValueError) as exc:
        result.error = f"could not reach {base_url} ({_reason(exc)})"
        return result

    for entry in payload.get("screens") or []:
        screen_id = entry.get("id")
        path = entry.get("path")
        if not screen_id:
            continue
        result.screen_order.append(screen_id)
        if not path:
            continue
        suffix = Path(path).suffix.lower()
        try:
            image, content_type = ui.get("/screenshots/file/" + urllib.parse.quote(path))
        except (urllib.error.URLError, OSError, AuthRequired) as exc:
            print(f"  {result.label}: skipped {screen_id} ({_reason(exc)})", file=sys.stderr)
            continue
        if not content_type.startswith("image/"):
            content_type = IMAGE_TYPES.get(suffix, "image/png")
        result.screenshots[screen_id] = Screenshot(
            screen_id=screen_id,
            image=image,
            content_type=content_type.split(";", 1)[0],
            timestamp=entry.get("timestamp"),
            elapsed=entry.get("elapsed"),
            is_stale=bool(entry.get("is_stale")),
        )
    return result


def _reason(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}"
    if isinstance(exc, urllib.error.URLError):
        return str(exc.reason)
    return str(exc) or exc.__class__.__name__


def screen_order(results: list[ClientResult]) -> list[str]:
    """Screens in playback order, merged across clients (first seen wins)."""

    seen: dict[str, None] = {}
    for result in results:
        for screen_id in result.screen_order:
            if screen_id in result.screenshots:
                seen.setdefault(screen_id, None)
    return list(seen)


# ── HTML ──────────────────────────────────────────────────────────────────────


def _anchor(screen_id: str) -> str:
    return "screen-" + "".join(ch if ch.isalnum() else "-" for ch in screen_id.lower())


def render_html(results: list[ClientResult], skipped: list[ClientResult], server_url: str,
                generated_at: datetime.datetime) -> str:
    collected = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    screens = screen_order(collected)
    esc = html.escape
    parts: list[str] = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Display client screenshots</title>",
        "<style>",
        ":root{color-scheme:light dark;--bg:#f6f6f4;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#ddd;--warn:#b3261e}",
        "@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#9a9a93;--card:#222220;--line:#3a3a37;--warn:#f2b8b5}}",
        "body{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,system-ui,sans-serif}",
        "h1{margin:0 0 4px;font-size:24px}h2{margin:40px 0 12px;font-size:19px;border-bottom:1px solid var(--line);padding-bottom:6px}",
        ".meta,.note,figcaption small{color:var(--muted)}.note{margin:4px 0}.warn{color:var(--warn)}",
        "nav{margin:16px 0;columns:16em;font-size:14px}nav a{display:block;color:inherit}",
        ".row{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start}",
        "figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;max-width:100%}",
        "figure img{display:block;max-width:100%;max-height:320px;width:auto;height:auto;cursor:zoom-in;background:#000}",
        "figure img.full{max-height:none;cursor:zoom-out}",
        "figcaption{margin-top:8px;font-size:14px}figcaption b{display:block}",
        "</style></head><body>",
        "<h1>Display client screenshots</h1>",
        '<p class="meta">Click a screenshot to see it at full size.</p>',
        f'<p class="meta">Collected {esc(generated_at.strftime("%Y-%m-%d %H:%M"))} from {esc(server_url)}: '
        f"{len(screens)} screens across {len(collected)} display{'s' if len(collected) != 1 else ''}.</p>",
    ]
    for result in failed:
        parts.append(f'<p class="note warn">Skipped {esc(result.label)}: {esc(result.error or "")}</p>')
    for result in skipped:
        parts.append(f'<p class="note">Skipped {esc(result.label)}: not active ({esc(result.state or "unknown")}).</p>')
    if screens:
        parts.append("<nav>")
        parts.extend(f'<a href="#{_anchor(s)}">{esc(s)}</a>' for s in screens)
        parts.append("</nav>")
    for screen_id in screens:
        parts.append(f'<section id="{_anchor(screen_id)}"><h2>{esc(screen_id)}</h2>')
        missing = [r.label for r in collected if screen_id not in r.screenshots]
        if missing:
            parts.append(f'<p class="note">No screenshot from: {esc(", ".join(missing))}</p>')
        parts.append('<div class="row">')
        for result in collected:
            shot = result.screenshots.get(screen_id)
            if shot is None:
                continue
            data = base64.b64encode(shot.image).decode("ascii")
            detail = " · ".join(x for x in (result.display_profile, result.dimensions) if x)
            when = shot.timestamp or ""
            if shot.elapsed:
                when = f"{when} ({shot.elapsed})" if when else shot.elapsed  # "0d 0h 1m 5s ago"
            stale = ' <span class="warn">stale</span>' if shot.is_stale else ""
            parts.append(
                f'<figure><img src="data:{esc(shot.content_type)};base64,{data}" '
                f'alt="{esc(screen_id)} on {esc(result.label)}" loading="lazy" '
                'onclick="this.classList.toggle(\'full\')">'
                f"<figcaption><b>{esc(result.label)}</b>"
                f"<small>{esc(detail)}</small><br><small>{esc(when)}{stale}</small></figcaption></figure>"
            )
        parts.append("</div></section>")
    if not screens:
        parts.append('<p class="note">No screenshots were collected.</p>')
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


# ── CLI ───────────────────────────────────────────────────────────────────────


def _default_output(now: datetime.datetime) -> Path:
    desktop = Path.home() / "Desktop"
    folder = desktop if desktop.is_dir() else Path.cwd()
    return folder / f"desk_display_screenshots_{now.strftime('%Y%m%d-%H%M')}.html"


def _parse_overrides(values: list[str]) -> dict[str, str]:
    overrides = {}
    for value in values:
        client_id, sep, host = value.partition("=")
        if not sep or not client_id.strip() or not host.strip():
            raise SystemExit(f"--client-host expects ID=HOST, got {value!r}")
        overrides[client_id.strip()] = host.strip()
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--server", help="the server's config UI: host, host:port or URL "
                        f"(port {DEFAULT_PORT} by default); asked for when omitted")
    parser.add_argument("--output", type=Path, help="HTML file to write "
                        "(default: ~/Desktop/desk_display_screenshots_<date-time>.html)")
    parser.add_argument("--client-port", type=int, default=DEFAULT_PORT,
                        help=f"config UI port on each display (default {DEFAULT_PORT})")
    parser.add_argument("--client-host", action="append", default=[], metavar="ID=HOST",
                        help="address of one client's config UI, overriding the recorded one (repeatable)")
    parser.add_argument("--include-inactive", action="store_true",
                        help="also try clients that are not online (expired, never connected)")
    parser.add_argument("--username", default=os.environ.get("SCREEN_UI_USERNAME", ""),
                        help="config UI username, if one is set")
    parser.add_argument("--password", default=os.environ.get("SCREEN_UI_PASSWORD"),
                        help="config UI password (default: $SCREEN_UI_PASSWORD, else asked when needed)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="seconds per request")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    server = args.server
    if not server:
        if not sys.stdin.isatty():
            print("error: give the server with --server", file=sys.stderr)
            return 2
        server = input(f"Server config UI (host or URL) [localhost:{DEFAULT_PORT}]: ").strip() or "localhost"
    try:
        server_url = normalize_base_url(server)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    overrides = _parse_overrides(args.client_host)

    password_cache: dict[str, Optional[str]] = {}
    password_lock = threading.Lock()

    def password() -> Optional[str]:
        # Clients are collected in parallel; ask for the password only once.
        with password_lock:
            return _password_once()

    def _password_once() -> Optional[str]:
        if "value" not in password_cache:
            value = args.password
            if not value and sys.stdin.isatty():
                value = getpass.getpass("Config UI password: ")
            password_cache["value"] = value or None
        return password_cache["value"]

    def make_ui(base_url: str) -> ConfigUI:
        return ConfigUI(base_url, password, args.username, args.timeout)

    try:
        payload = make_ui(server_url).get_json("/api/clients")
    except AuthRequired as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except urllib.error.HTTPError as exc:
        hint = " (is this a server install? client-only displays have no client list)" if exc.code == 404 else ""
        print(f"error: {server_url}/api/clients answered HTTP {exc.code}{hint}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"error: could not reach {server_url} ({_reason(exc)})", file=sys.stderr)
        return 1

    clients = [c for c in payload.get("clients") or [] if c.get("client_id")]
    active = [c for c in clients if args.include_inactive or c.get("state") in ACTIVE_STATES]
    skipped = [
        ClientResult(client_id=c["client_id"], name=str(c.get("friendly_name") or c["client_id"]),
                     display_profile=c.get("display_profile"), dimensions=c.get("dimensions"),
                     state=c.get("state"))
        for c in clients if c not in active
    ]
    if not active:
        print("No active display clients found.", file=sys.stderr)

    targets = [(c, client_base_url(c, server_url, overrides, args.client_port)) for c in active]
    for client, base_url in targets:
        print(f"Collecting {client['client_id']} from {base_url} ...", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=max(1, min(8, len(targets)))) as pool:
        results = list(pool.map(lambda t: collect_client(t[0], t[1], make_ui), targets))

    for result in results:
        if result.error:
            print(f"  {result.label}: {result.error}", file=sys.stderr)
        else:
            print(f"  {result.label}: {len(result.screenshots)} screenshots", file=sys.stderr)

    now = datetime.datetime.now()
    output = (args.output.expanduser() if args.output else _default_output(now))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(results, skipped, server_url, now), encoding="utf-8")
    print(f"Wrote {output}")
    return 0 if any(r.error is None for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
