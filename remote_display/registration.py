"""Guided registration of a new display client from the config UI.

The Clients page's "Add a display" wizard uses this module to:

* check that the render server can actually be reached by a display on the
  LAN before a credential is issued (bind address, advertised URL, enrollment
  mode, and a live TCP probe of this machine's LAN address);
* describe each display profile in words a person recognises, with the
  installer profile that sets up its panel;
* build the one paste-able shell command that writes the client's
  ``.env.client`` on the new Pi and runs the client installer.

Nothing here reads or returns a server credential: the only secret in a
result is the new client's own enrollment credential, which the caller
passes in and which is shown once.
"""
from __future__ import annotations

import contextlib
import ipaddress
import shlex
import socket
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from display_profiles import PROFILE_PRESETS

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROBE_TIMEOUT_SECONDS = 1.5


@dataclass(frozen=True)
class ProfileInfo:
    """How the wizard presents a display profile."""

    label: str
    installer: str | None  # Installers/install.sh profile; None when the installer should ask
    installer_env: tuple[tuple[str, str], ...] = ()
    detail: str = ""  # a second line for the choice, e.g. the screens a multi-screen HAT has
    in_wizard: bool = True  # False for a profile no device is set up as on its own
    after_install: str = ""  # what to do once the installer finishes, shown by the join script


PROFILE_INFO: dict[str, ProfileInfo] = {
    "display_hat_mini": ProfileInfo("Pimoroni Display HAT Mini", "display_hat_mini"),
    "adafruit_minipitft_114": ProfileInfo('Adafruit Mini PiTFT 1.14"', "adafruit_minipitft"),
    "hyperpixel4": ProfileInfo("Pimoroni HyperPixel 4.0 (rectangular)", "hyperpixel",
                               (("HYPERPIXEL_PANEL", "hyperpixel4"),)),
    "hyperpixel4_square": ProfileInfo("Pimoroni HyperPixel 4.0 Square", "hyperpixel",
                                      (("HYPERPIXEL_PANEL", "hyperpixel4sq"),)),
    # One HAT, three screens: the LCD plays the playlist and the installer adds
    # desk_display_waveshare_oled.service, which drives both OLEDs from the
    # client's heartbeat (scripts/waveshare_oled_status.py).
    "waveshare_lcd_320x240": ProfileInfo(
        "Waveshare OLED/LCD HAT (A)", "waveshare_oled_lcd_hat_a",
        detail="320x240 LCD plus its two 128x64 status OLEDs",
        after_install="Reboot now (sudo reboot) so the HAT's LCD and OLEDs turn on."),
    # Only ever the HAT's side OLEDs, which the HAT choice above sets up.
    "waveshare_oled_128x64": ProfileInfo("Waveshare 128x64 OLED", None, in_wizard=False),
    "hdmi_1080p": ProfileInfo("HDMI monitor, 1080p", "kernel"),
    "fallback_default": ProfileInfo("Generic 320x240 panel", None),
    "fallback_hd": ProfileInfo("Generic 720p panel", None),
}


def profile_choices() -> list[dict[str, Any]]:
    """Every display profile a device can be set up as, most common first, with a readable label."""

    order = list(PROFILE_INFO)
    choices = []
    for profile_id in sorted(PROFILE_PRESETS, key=lambda p: (order.index(p) if p in order else len(order), p)):
        preset = PROFILE_PRESETS[profile_id]
        info = PROFILE_INFO.get(profile_id) or ProfileInfo(profile_id.replace("_", " "), None)
        if not info.in_wizard:
            continue
        choices.append({
            "id": profile_id,
            "label": info.label,
            "detail": info.detail,
            "size": f"{preset.width}x{preset.height}",
            "installer": info.installer,
        })
    return choices


# ── Server URL ─────────────────────────────────────────────────────────────


class ServerUrlError(ValueError):
    pass


def normalize_server_url(value: Any) -> str:
    """A client-facing server base URL: http(s), a host, no credentials, path, query or fragment."""

    if not isinstance(value, str) or not value.strip():
        raise ServerUrlError("enter the address displays use to reach this server, e.g. http://square.local:8765")
    text = value.strip()
    if "://" not in text:
        text = "http://" + text
    parts = urlsplit(text)
    if parts.scheme not in {"http", "https"}:
        raise ServerUrlError("the server address must start with http:// or https://")
    if not parts.hostname:
        raise ServerUrlError("the server address needs a host name")
    if parts.username or parts.password:
        raise ServerUrlError("the server address must not contain a user name or password")
    if parts.query or parts.fragment or parts.path not in {"", "/"}:
        raise ServerUrlError("the server address is just scheme, host and port, with no path")
    if any(ch.isspace() for ch in text):
        raise ServerUrlError("the server address must not contain spaces")
    try:
        port = parts.port
    except ValueError as exc:
        raise ServerUrlError("the server address has an invalid port") from exc
    netloc = parts.hostname if ":" not in parts.hostname else f"[{parts.hostname}]"
    if port is not None:
        netloc += f":{port}"
    return urlunsplit((parts.scheme, netloc, "", "", ""))


def is_loopback(host: str) -> bool:
    host = (host or "").strip().strip("[]").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_wildcard(host: str) -> bool:
    return host.strip().strip("[]") in {"0.0.0.0", "::", ""}


def lan_address() -> str | None:
    """This machine's address on the LAN (the one its default route uses), or None."""

    with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        # UDP connect sends nothing; it only selects the outgoing interface.
        sock.connect(("192.0.2.1", 9))
        address = sock.getsockname()[0]
        if address and not address.startswith("127."):
            return address
    return None


def suggested_server_url(settings: Mapping[str, Any], browser_host: str | None) -> str:
    """The URL a new display should use: the configured public URL, else this host on the server port."""

    configured = settings.get("DESK_DISPLAY_SERVER_PUBLIC_URL")
    if configured:
        with contextlib.suppress(ServerUrlError):
            return normalize_server_url(configured)
    scheme = "https" if settings.get("DESK_DISPLAY_SERVER_TLS_CERT") else "http"
    port = int(settings.get("DESK_DISPLAY_SERVER_PORT") or 8765)
    host = (browser_host or "").strip()
    # The browser reached the config UI by this name, so a display on the same
    # LAN almost certainly can too; loopback names are useless to another Pi.
    if not host or is_loopback(host):
        name = socket.gethostname().split(".")[0]
        host = f"{name}.local" if name and name != "localhost" else (lan_address() or "localhost")
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"{scheme}://{host}:{port}"


def probe(host: str, port: int, timeout: float = PROBE_TIMEOUT_SECONDS) -> bool:
    """True when a TCP connection to *host*:*port* succeeds."""

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ── Readiness ──────────────────────────────────────────────────────────────


def _check(checks: list[dict[str, Any]], code: str, status: str, title: str, detail: str,
           fix: list[str] | None = None) -> None:
    checks.append({"code": code, "status": status, "title": title, "detail": detail, "fix": fix or []})


def readiness(
    settings: Mapping[str, Any],
    *,
    browser_host: str | None = None,
    prober: Callable[[str, int], bool] | None = None,
    lan: Callable[[], str | None] | None = None,
) -> dict[str, Any]:
    """Whether a display on the LAN can register with this server, and how to fix it if not.

    *settings* are the server's parsed settings. Each check has a ``status`` of
    ``ok``, ``warning`` or ``error`` and, when not ``ok``, the ``.env`` lines
    that fix it. ``ready`` is false while any check is an error.
    """

    prober = prober or probe
    lan = lan or lan_address
    checks: list[dict[str, Any]] = []
    host = str(settings.get("DESK_DISPLAY_SERVER_HOST") or "127.0.0.1")
    port = int(settings.get("DESK_DISPLAY_SERVER_PORT") or 8765)
    url = suggested_server_url(settings, browser_host)
    url_host = urlsplit(url).hostname or ""
    restart = "then restart the server: bash scripts/restart_services.sh"

    if (settings.get("DESK_DISPLAY_SERVER_ENROLLMENT") or "provisioned") == "shared":
        _check(checks, "enrollment", "error", "The server does not accept per-display credentials",
               "DESK_DISPLAY_SERVER_ENROLLMENT=shared makes every display use one shared token, so a "
               "credential from this wizard would be refused.",
               ["DESK_DISPLAY_SERVER_ENROLLMENT=provisioned", restart])

    if is_loopback(host):
        _check(checks, "bind", "error", "Only this Pi can connect",
               f"The render server listens on {host}:{port}, which other devices cannot reach.",
               ["DESK_DISPLAY_SERVER_HOST=0.0.0.0", restart])
    else:
        _check(checks, "bind", "ok", "The server listens on the network", f"Bound to {host}:{port}.")

    if settings.get("DESK_DISPLAY_SERVER_PUBLIC_URL"):
        _check(checks, "public_url", "ok", "Displays are given a fixed address", url)
    else:
        _check(checks, "public_url", "warning", "No server address is configured",
               f"The wizard will use {url}, the name your browser used for this page. Save it so "
               "rotated credentials get it too.",
               [f"DESK_DISPLAY_SERVER_PUBLIC_URL={url}", restart])
    if is_loopback(url_host):
        _check(checks, "public_url_loopback", "error", "The server address points at the display itself",
               f"{url} means \"this device\" on the display, so it would never find the server.",
               [f"DESK_DISPLAY_SERVER_PUBLIC_URL=http://<this-pi>.local:{port}", restart])

    reachable: bool | None = None
    address = None
    if not is_loopback(host):
        address = lan() if _is_wildcard(host) else host
        if address:
            reachable = prober(address, port)
            if reachable:
                _check(checks, "reachable", "ok", "The server answers on the LAN", f"Connected to {address}:{port}.")
            else:
                _check(checks, "reachable", "error", "The server is not answering on the LAN",
                       f"Nothing accepted a connection on {address}:{port}. Check that "
                       "desk_display_server.service is running and that no firewall blocks the port.",
                       ["sudo systemctl status desk_display_server.service"])

    insecure = urlsplit(url).scheme == "http" and not is_loopback(url_host)
    if insecure:
        _check(checks, "transport", "warning", "Traffic is not encrypted",
               "Displays talk to the server over plain HTTP. That is fine on a trusted home network; the "
               "wizard adds DESK_DISPLAY_ALLOW_INSECURE_TRANSPORT=1 to the display's settings so it "
               "agrees to connect. Use HTTPS (DESK_DISPLAY_SERVER_TLS_CERT/KEY) otherwise.")

    return {
        "ready": not any(c["status"] == "error" for c in checks),
        "server_url": url,
        "insecure_transport": insecure,
        "bind_host": host,
        "port": port,
        "lan_address": address,
        "reachable": reachable,
        "checks": checks,
    }


# ── Install command ────────────────────────────────────────────────────────


def repository_url(project_dir: Path = _PROJECT_ROOT) -> str | None:
    """The git remote this server was installed from, without any embedded credentials."""

    try:
        result = subprocess.run(
            ["git", "-C", str(project_dir), "config", "--get", "remote.origin.url"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    url = result.stdout.strip()
    if not url or any(ch.isspace() for ch in url):
        return None
    parts = urlsplit(url)
    if parts.scheme in {"http", "https"}:
        if not parts.hostname:
            return None
        netloc = parts.hostname + (f":{parts.port}" if parts.port else "")
        return urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    if parts.scheme == "" and "@" in url.split(":", 1)[0] and ":" in url:
        return url  # scp-style git@host:owner/repo.git
    return None


def credentials_filename(client_id: str) -> str:
    return f"{client_id}.env.client"


def _install_parts(client_env_text: str, client_id: str, display_profile: str) -> tuple[list[str], str, str]:
    """(lines writing the credentials file, the installer command, the file's path)."""

    info = PROFILE_INFO.get(display_profile)
    filename = "~/" + credentials_filename(client_id)
    marker = "DESK_DISPLAY_CLIENT_EOF"
    write = [f"(umask 077 && cat > {filename} <<'{marker}'",
             *client_env_text.rstrip("\n").splitlines(), marker, ")"]
    installer = "bash Installers/install.sh --mode client --credentials " + filename
    if info and info.installer:
        installer += " " + info.installer
    prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in (info.installer_env if info else ()))
    return write, (prefix + " " if prefix else "") + installer, filename


# A checkout from before the server/client work (v0.1) has an installer with
# no --mode, which would install the old standalone display instead.
_UPDATE_OLD_CHECKOUT = [
    "if ! grep -q -- '--mode' ~/desk_display/Installers/install.sh; then",
    '  echo "==> Updating the old Desk Display checkout in ~/desk_display"',
    "  git -C ~/desk_display pull --ff-only || {",
    '    echo "Could not update ~/desk_display. If git reported a permission problem, run:" >&2',
    '    echo "  sudo chown -R $(id -un):$(id -gn) ~/desk_display" >&2',
    '    echo "If it reported local changes, run: git -C ~/desk_display stash" >&2',
    '    echo "Then make a new setup command on the Clients page and run it." >&2',
    "    exit 1",
    "  }",
    "fi",
]


def install_command(client_env_text: str, client_id: str, display_profile: str,
                    repo_url: str | None = None) -> str:
    """One paste-able command that sets up a new client Pi.

    It clones the project when ``~/desk_display`` is missing, writes the
    credentials file with mode 600, runs the client installer, and deletes the
    credentials file once the installer has copied it into ``.env.client``.
    """

    write, installer, filename = _install_parts(client_env_text, client_id, display_profile)
    lines = []
    if repo_url:
        lines.append(f"[ -d ~/desk_display ] || git clone {shlex.quote(repo_url)} ~/desk_display")
    lines += write
    lines.append(f"cd ~/desk_display && {installer} && rm -f {filename}")
    return "\n".join(lines) + "\n"


def join_command(server_url: str, code: str) -> str:
    """The short command a new display runs to fetch and run its setup script.

    ``bash -c "$(curl ...)"`` rather than ``curl | bash`` keeps the terminal
    on the installer's standard input, so its prompts (panel type, reboot)
    still work. The code goes in the POST body, which access logs never show.
    """

    url = normalize_server_url(server_url) + "/api/v1/join"
    return f'bash -c "$(curl -fsS -d code={shlex.quote(code)} {shlex.quote(url)})"'


def join_script(client_env_text: str, client_id: str, display_profile: str,
                repo_url: str | None = None) -> str:
    """The setup script ``/api/v1/join`` returns: clone (or update a v0.1
    checkout), write settings, install, clean up."""

    write, installer, filename = _install_parts(client_env_text, client_id, display_profile)
    lines = [
        "#!/usr/bin/env bash",
        f"# Desk Display client setup for {client_id}. The credential below works only on this display.",
        "set -euo pipefail",
        f'echo "==> Setting up Desk Display client {client_id}"',
    ]
    if repo_url:
        lines += [
            "if [ ! -d ~/desk_display ]; then",
            '  command -v git >/dev/null || { sudo apt-get update && sudo apt-get install -y git; }',
            f"  git clone {shlex.quote(repo_url)} ~/desk_display",
            "fi",
        ]
    else:
        lines += [
            "if [ ! -d ~/desk_display ]; then",
            '  echo "Clone desk_display into ~/desk_display first, then run this command again." >&2',
            "  exit 1",
            "fi",
        ]
    lines += _UPDATE_OLD_CHECKOUT
    lines += [
        "if [ -f ~/desk_display/.env.client ]; then",
        '  backup=~/desk_display/.env.client.before-join-$(date +%Y%m%d%H%M%S)',
        '  mv ~/desk_display/.env.client "$backup"',
        '  echo "==> Moved the old client settings to $backup"',
        "fi",
        *write,
        "cd ~/desk_display",
        installer,
        f"rm -f {filename}",
    ]
    info = PROFILE_INFO.get(display_profile)
    if info and info.after_install:
        lines.append(f"echo {shlex.quote(f'==> Done. {info.after_install} {client_id} then connects to the server on its own.')}")
    else:
        lines.append(f'echo "==> Done. {client_id} connects to the server on its own; '
                     'if the installer asked for a reboot, reboot now."')
    return "\n".join(lines) + "\n"


__all__ = [
    "PROFILE_INFO",
    "ProfileInfo",
    "ServerUrlError",
    "credentials_filename",
    "install_command",
    "is_loopback",
    "join_command",
    "join_script",
    "lan_address",
    "normalize_server_url",
    "probe",
    "profile_choices",
    "readiness",
    "repository_url",
    "suggested_server_url",
]
