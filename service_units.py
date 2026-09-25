"""systemd service definitions for each installation mode.

``standalone``
    The legacy single process: ``main.py`` renders and draws to the panel.
``server``
    ``display_server.py`` renders for remote clients; nothing draws locally.
``client``
    ``display_client.py`` plays what a server publishes.
``combined``
    A server with its own panel. The panel is an ordinary client on the
    loopback address, with its own ID, profile, playlist, cache, rotation
    and touch settings, and the same protocol and playback code as any
    remote client.

The client unit never ``Requires``, ``BindsTo`` or is ``PartOf`` the server,
so restarting the server leaves the panel playing from its cache, and the
client starts without waiting for the server. The standalone unit
``Conflicts`` with the client unit because both drive the same panel.

``python3 -m service_units --mode combined --output DIR`` writes the unit
files; the installers (Phase 19) copy them into /etc/systemd/system.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Mode(str, Enum):
    STANDALONE = "standalone"
    SERVER = "server"
    CLIENT = "client"
    COMBINED = "combined"


STANDALONE_SERVICE = "desk_display.service"
SERVER_SERVICE = "desk_display_server.service"
CLIENT_SERVICE = "desk_display_client.service"
CONFIG_UI_SERVICE = "config_ui_desk_display.service"

# Unit dependencies that would stop or restart the client along with the server.
COUPLING_KEYS = ("Requires", "BindsTo", "PartOf", "Requisite")


@dataclass(frozen=True)
class Unit:
    name: str
    description: str
    script: str
    env_file: str
    role: str | None = None
    unit: tuple[tuple[str, str], ...] = ()
    service: tuple[tuple[str, str], ...] = field(default=())

    def render(self, *, project_dir: str, python: str, user: str) -> str:
        lines = ["[Unit]", f"Description={self.description}", "Wants=network-online.target",
                 "After=network-online.target"]
        lines += [f"{key}={value}" for key, value in self.unit]
        lines += ["", "[Service]", f"WorkingDirectory={project_dir}",
                  f"EnvironmentFile=-{project_dir}/{self.env_file}"]
        if self.role is not None:
            # The process's role when its env file has none. systemd lets the env file
            # override this, and a conflicting role there fails the startup check.
            lines.append(f"Environment=DESK_DISPLAY_ROLE={self.role}")
        lines += [f"ExecStart={python} {project_dir}/{self.script}"]
        lines += [f"{key}={value}" for key, value in self.service]
        lines += ["TimeoutStopSec=10", "KillSignal=SIGTERM", "Restart=always", "RestartSec=5", f"User={user}",
                  "", "[Install]", "WantedBy=multi-user.target", ""]
        return "\n".join(lines)


_STANDALONE = Unit(STANDALONE_SERVICE, "Desk Display Service - main", "main.py", ".env",
                   unit=(("Conflicts", CLIENT_SERVICE),))
_SERVER = Unit(SERVER_SERVICE, "Desk Display - render server", "display_server.py", ".env", role="server",
               unit=(("Conflicts", STANDALONE_SERVICE),))
_CLIENT = Unit(CLIENT_SERVICE, "Desk Display - display client", "display_client.py", ".env.client",
               role="client", unit=(("Conflicts", STANDALONE_SERVICE),))
_CONFIG_UI = Unit(CONFIG_UI_SERVICE, "Desk Display Service - config UI", "config_ui.py", ".env")

UNITS: Mapping[Mode, tuple[Unit, ...]] = {
    Mode.STANDALONE: (_STANDALONE, _CONFIG_UI),
    Mode.SERVER: (_SERVER, _CONFIG_UI),
    Mode.CLIENT: (_CLIENT,),
    Mode.COMBINED: (_SERVER, _CLIENT, _CONFIG_UI),
}
ALL_SERVICES = (STANDALONE_SERVICE, SERVER_SERVICE, CLIENT_SERVICE, CONFIG_UI_SERVICE)


def services_for(mode: Mode | str) -> tuple[str, ...]:
    return tuple(unit.name for unit in UNITS[Mode(mode)])


def disabled_for(mode: Mode | str) -> tuple[str, ...]:
    """Project services an installer should stop and disable for *mode*."""

    wanted = set(services_for(mode))
    return tuple(name for name in ALL_SERVICES if name not in wanted)


def render_units(mode: Mode | str, *, project_dir: str, python: str, user: str) -> dict[str, str]:
    return {unit.name: unit.render(project_dir=project_dir, python=python, user=user)
            for unit in UNITS[Mode(mode)]}


def parse_unit(text: str) -> dict[str, dict[str, list[str]]]:
    """``{section: {key: [values]}}`` for a rendered unit file."""

    sections: dict[str, dict[str, list[str]]] = {}
    current: dict[str, list[str]] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = sections.setdefault(line[1:-1], {})
        elif current is not None and "=" in line:
            key, value = line.split("=", 1)
            current.setdefault(key, []).append(value)
    return sections


def _cli(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m service_units", description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=[m.value for m in Mode], required=True)
    parser.add_argument("--output", required=True, help="directory to write the unit files into")
    parser.add_argument("--project-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--python", help="interpreter; defaults to the project's venv")
    parser.add_argument("--user", default="pi")
    args = parser.parse_args(list(argv) if argv is not None else None)
    python = args.python or f"{args.project_dir}/venv/bin/python"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, text in render_units(args.mode, project_dir=args.project_dir, python=python,
                                   user=args.user).items():
        (output / name).write_text(text, encoding="utf-8")
        print(output / name)
    for name in disabled_for(args.mode):
        print(f"disable: {name}")
    return 0


__all__ = [
    "ALL_SERVICES",
    "CLIENT_SERVICE",
    "CONFIG_UI_SERVICE",
    "COUPLING_KEYS",
    "Mode",
    "SERVER_SERVICE",
    "STANDALONE_SERVICE",
    "Unit",
    "disabled_for",
    "parse_unit",
    "render_units",
    "services_for",
]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
