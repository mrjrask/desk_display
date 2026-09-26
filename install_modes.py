"""What each installation mode installs, keeps on upgrade, and removes on uninstall.

The shell installers (``Installers/install.sh --mode``), ``scripts/upgrade.sh``,
``Installers/uninstall.sh`` and ``scripts/cleanup.sh`` ask this module rather
than hard-coding service names, dependency files and data paths, so every
workflow agrees on one documented answer per mode:

``standalone``
    ``main.py`` draws to the panel (the legacy install, unchanged).
``server``
    Renders for remote clients. No panel driver or GPIO stack is installed.
``client``
    Plays what a server publishes. It installs no upstream provider libraries.
``combined``
    A server whose own panel is a loopback client.

It uses only the standard library so it runs before the virtualenv exists and
after the uninstaller has removed it.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import service_units as su
from service_units import Mode

PROJECT_DIR = Path(__file__).resolve().parent
MODE_MARKER = ".runtime/install_mode"
SYSTEMD_DIR = Path("/etc/systemd/system")

# The env file holding the panel's settings (hardware, identity) in each mode.
PANEL_ENV = {Mode.STANDALONE: ".env", Mode.CLIENT: ".env.client", Mode.COMBINED: ".env.client"}
# Env files each mode reads.
ENV_FILES = {
    Mode.STANDALONE: (".env",),
    Mode.SERVER: (".env",),
    Mode.CLIENT: (".env.client",),
    Mode.COMBINED: (".env", ".env.client"),
}

# Hardware layer per display output; "" is headless (no panel driver).
_HARDWARE = {
    "displayhatmini": "displayhatmini",
    "minipitft": "minipitft",
    "framebuffer": "framebuffer",
    "kernel": "kernel",
    "window": "window",
    "headless": "",
}


def normalize_output(output: str | None) -> str:
    value = (output or "").strip().lower()
    if value in ("", "auto"):
        return "displayhatmini"
    if value not in _HARDWARE:
        raise ValueError(f"unknown display output {output!r}; expected one of {', '.join(sorted(_HARDWARE))}")
    return value


def requirements_file(mode: Mode | str, output: str | None = None) -> str:
    """The requirements file (relative to the project) that *mode* installs."""

    mode = Mode(mode)
    if mode is Mode.SERVER:
        return "requirements/server.txt"
    hardware = _HARDWARE[normalize_output(output)]
    if mode is Mode.CLIENT:
        return f"requirements/client-{hardware}.txt" if hardware else "requirements/client.txt"
    # Standalone and combined run the full application plus the panel driver.
    return f"requirements/{hardware}.txt" if hardware else "requirements/base.txt"


def expand_requirements(path: str | os.PathLike[str]) -> list[str]:
    """Every requirement line *path* installs, following ``-r`` includes."""

    seen: set[Path] = set()
    found: list[str] = []

    def walk(current: Path) -> None:
        current = current.resolve()
        if current in seen:
            return
        seen.add(current)
        for raw in current.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith(("-r ", "--requirement ")):
                walk(current.parent / line.split(None, 1)[1])
            elif line not in found:
                found.append(line)

    walk(Path(path))
    return found


def requirement_names(path: str | os.PathLike[str]) -> set[str]:
    names = set()
    for line in expand_requirements(path):
        if line.startswith("-"):
            continue
        name = line
        for marker in ("[", ";", "=", ">", "<", "!", "~"):
            name = name.split(marker, 1)[0]
        names.add(name.strip().lower())
    return names


# ─── Services ──────────────────────────────────────────────────────────────


def start_order(mode: Mode | str) -> tuple[str, ...]:
    """Services in the order the installer enables and starts them.

    In a combined install the server starts first, but the client never waits
    for it: its unit has no ordering or requirement on the server, so after a
    reboot the panel shows its cache while the server is still warming up.
    """

    return su.services_for(mode)


def write_units(mode: Mode | str, directory: Path, *, project_dir: str, python: str, user: str,
                output: str | None = None, environment: Mapping[str, str] | None = None) -> list[Path]:
    mode = Mode(mode)
    hooks = None
    if mode is not Mode.SERVER:
        hooks = su.panel_hooks(normalize_output(output), project_dir=project_dir, user=user,
                               environment=environment)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in su.render_units(mode, project_dir=project_dir, python=python, user=user,
                                      hooks=hooks).items():
        path = directory / name
        path.write_text(text, encoding="utf-8")
        os.chmod(path, 0o644)
        written.append(path)
    return written


# ─── Data each mode owns ───────────────────────────────────────────────────


@dataclass(frozen=True)
class Data:
    path: str  # relative to the project, or ~/ for the service user's home
    what: str
    modes: frozenset[Mode]
    # Copied to the uninstall backup folder before the project is removed.
    # Everything else under the project (venv, caches, artifacts) is removed.
    backup: bool = True
    # Holds credentials: backed up with mode 0600 and never printed.
    secret: bool = False


_SERVERS = frozenset({Mode.SERVER, Mode.COMBINED})
_CLIENTS = frozenset({Mode.CLIENT, Mode.COMBINED})
_ENV_HOSTS = frozenset({Mode.STANDALONE, Mode.SERVER, Mode.COMBINED})
_EVERY = frozenset(Mode)

DATA: tuple[Data, ...] = (
    Data(".env", "configuration and provider credentials", _ENV_HOSTS, secret=True),
    Data(".env.client", "client identity, credential and panel settings", _CLIENTS, secret=True),
    Data("~/keys", "WeatherKit and other key files", _ENV_HOSTS, secret=True),
    Data("screens_config.local.json", "local screen rotation override", frozenset({Mode.STANDALONE, *_SERVERS})),
    Data(".runtime/server/playlists.json", "server playlists and client assignments", _SERVERS),
    Data(".runtime/server/provisioned_clients.json", "per-client credential hashes", _SERVERS, secret=True),
    Data(".runtime/server/clients.json", "known clients and their capabilities", _SERVERS),
    Data(".runtime/server/migrations", "standalone migration bundles", _SERVERS),
    Data(".runtime/server/backups", "upgrade snapshots", _SERVERS),
    Data("cache/artifacts", "rendered artifacts (the server re-renders them)", _SERVERS, backup=False),
    Data("cache/client", "the client's offline cache", _CLIENTS, backup=False),
    Data("cache", "feed caches and weather history", frozenset({Mode.STANDALONE, *_SERVERS}), backup=False),
)


def data_for(mode: Mode | str) -> tuple[Data, ...]:
    mode = Mode(mode)
    return tuple(item for item in DATA if mode in item.modes)


def preserved_on_upgrade(mode: Mode | str) -> tuple[str, ...]:
    """Everything an upgrade keeps exactly as it was: all of the mode's data."""

    return tuple(item.path for item in data_for(mode))


def uninstall_backups(mode: Mode | str) -> tuple[Data, ...]:
    return tuple(item for item in data_for(mode) if item.backup)


def resolve(item_path: str, project_dir: Path, home: Path) -> Path:
    if item_path.startswith("~/"):
        return home / item_path[2:]
    return project_dir / item_path


# ─── Mode detection and upgrade snapshots ──────────────────────────────────


@dataclass(frozen=True)
class Installed:
    """What the installer recorded, so an upgrade rewrites identical units."""

    mode: Mode
    output: str | None = None
    user: str | None = None
    environment: tuple[tuple[str, str], ...] = ()


def read_marker(project_dir: Path) -> Installed | None:
    try:
        text = (project_dir / MODE_MARKER).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError
        return Installed(Mode(data["mode"]), data.get("output") or None, data.get("user") or None,
                         tuple((str(k), str(v)) for k, v in (data.get("environment") or {}).items()))
    except (ValueError, KeyError, TypeError):
        try:
            return Installed(Mode(text))
        except ValueError:
            return None


def write_marker(project_dir: Path, mode: Mode | str, *, output: str | None = None, user: str | None = None,
                 environment: Mapping[str, str] | None = None) -> Path:
    path = project_dir / MODE_MARKER
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    payload = {"mode": Mode(mode).value, "output": output or None, "user": user or None,
               "environment": dict(environment or {})}
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def detect_mode(project_dir: Path, systemd_dir: Path = SYSTEMD_DIR) -> Mode:
    """The installed mode: the installer's marker, else the installed units."""

    marked = read_marker(project_dir)
    if marked is not None:
        return marked.mode
    installed = {name for name in su.ALL_SERVICES if (systemd_dir / name).exists()}
    server, client = su.SERVER_SERVICE in installed, su.CLIENT_SERVICE in installed
    if server and client:
        return Mode.COMBINED
    if server:
        return Mode.SERVER
    if client:
        return Mode.CLIENT
    return Mode.STANDALONE


def snapshot(mode: Mode | str, project_dir: Path, *, home: Path | None = None,
             now: float | None = None) -> Path | None:
    """Copy a server's state into ``.runtime/server/backups/upgrade-<time>/``.

    Returns ``None`` for modes without server state. The snapshot is a backup
    an operator can restore by hand; the upgrade itself changes none of it.
    """

    mode = Mode(mode)
    if mode not in _SERVERS:
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time() if now is None else now))
    dest = project_dir / ".runtime" / "server" / "backups" / f"upgrade-{stamp}"
    counter = 1
    while dest.exists():
        dest = dest.with_name(f"upgrade-{stamp}-{counter}")
        counter += 1
    dest.mkdir(parents=True, mode=0o700)
    os.chmod(dest, 0o700)
    for item in uninstall_backups(mode):
        if item.path.startswith("~/") or item.path.endswith(("backups", "migrations")):
            continue
        source = resolve(item.path, project_dir, home or Path.home())
        if not source.exists():
            continue
        target = dest / item.path.replace("/", "__")
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
            os.chmod(target, 0o600)
    return dest


def backup_for_uninstall(mode: Mode | str, project_dir: Path, backup_dir: Path, *,
                         home: Path | None = None) -> list[tuple[str, Path]]:
    """Copy the mode's documented data to *backup_dir*; return what was copied."""

    copied = []
    backup_dir.mkdir(parents=True, exist_ok=True)
    for item in uninstall_backups(mode):
        source = resolve(item.path, project_dir, home or Path.home())
        if not source.exists():
            continue
        name = item.path.replace("~/", "").replace("/", "__")
        name = "dot" + name if name.startswith(".") else name
        target = backup_dir / name
        if target.exists():
            target = target.with_name(f"{name}_{time.strftime('%Y%m%d%H%M%S')}")
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        if item.secret:
            for path in [target, *(target.rglob("*") if target.is_dir() else ())]:
                os.chmod(path, 0o700 if path.is_dir() else 0o600)
        copied.append((item.path, target))
    return copied


# ─── Env files ─────────────────────────────────────────────────────────────

# Installer profile -> display profile, when the panel env file names none.
INSTALL_PROFILE_DISPLAY = {
    "display_hat_mini": "display_hat_mini",
    "adafruit_minipitft": "adafruit_minipitft_114",
    "waveshare_oled_lcd_hat_a": "waveshare_lcd_320x240",
}
_HYPERPIXEL_PANELS = {"hyperpixel4": "hyperpixel4", "hyperpixel4sq": "hyperpixel4_square",
                      "hyperpixel4_square": "hyperpixel4_square"}


def panel_profile(env: Mapping[str, str], install_profile: str | None = None) -> str | None:
    profile = (env.get("DESK_DISPLAY_PROFILE") or "").strip()
    if profile:
        return profile
    panel = (env.get("HYPERPIXEL_PANEL") or "").strip().lower()
    if panel in _HYPERPIXEL_PANELS:
        return _HYPERPIXEL_PANELS[panel]
    return INSTALL_PROFILE_DISPLAY.get(install_profile or "")


def prepare_env(mode: Mode | str, project_dir: Path, *, install_profile: str | None = None,
                credentials: Path | None = None, client_id: str | None = None) -> list[str]:
    """Make the mode's env files ready for its services; return what was done.

    An existing ``.env.client`` is never touched: it holds the client's stable
    identity and credential. A new one starts from the standalone ``.env``'s
    panel settings, converted to the client role (every server and provider
    setting removed). In a combined install the panel's credential is issued
    by the local server. A server's ``.env`` is converted in place with a
    backup when it is still a standalone configuration.
    """

    import deployment_config as dc
    import env_conversion

    mode = Mode(mode)
    notes: list[str] = []
    env_path = project_dir / ".env"
    if mode in _CLIENTS:
        target = project_dir / ".env.client"
        if target.exists():
            notes.append(f"kept {target.name} (client identity and credential unchanged)")
        else:
            standalone = dc.parse_env_file(env_path) if env_path.exists() else {}
            supplied: dict[str, str] = {}
            if credentials is not None:
                provisioned = dc.parse_env_file(credentials)
                supplied.update({k: v for k, v in provisioned.items()
                                 if k in ("DESK_DISPLAY_SERVER_URL", "DESK_DISPLAY_CLIENT_ID",
                                          "DESK_DISPLAY_CLIENT_TOKEN", "DESK_DISPLAY_PROFILE") and v})
            profile = supplied.get("DESK_DISPLAY_PROFILE") or panel_profile(standalone, install_profile)
            if mode is Mode.COMBINED and "DESK_DISPLAY_CLIENT_TOKEN" not in supplied:
                if not profile:
                    raise ValueError("set DESK_DISPLAY_PROFILE to this panel's display profile and rerun")
                supplied.update(_provision_local_panel(project_dir, standalone, profile, client_id))
                notes.append(f"provisioned local panel client {supplied['DESK_DISPLAY_CLIENT_ID']!r}")
            if profile:
                supplied["DESK_DISPLAY_PROFILE"] = profile
            scratch = project_dir / ".runtime" / "install" / "env.client.tmp"
            scratch.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(scratch, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(env_path.read_text(encoding="utf-8") if env_path.exists() else "")
            try:
                conversion = env_conversion.convert(scratch, dc.Role.CLIENT, supplied)
            except env_conversion.ConversionError as exc:
                raise ValueError(str(exc)) from None
            finally:
                scratch.unlink()
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(conversion.text)
            notes.append(f"wrote {target.name} from the panel settings in .env")
            if not conversion.report.ok:
                notes.append(f"{target.name} still needs: "
                             + ", ".join(sorted({str(i.name) for i in conversion.report.errors}))
                             + " (see scripts/convert_env.py --credentials)")
    if mode in _SERVERS and env_path.exists():
        conversion = env_conversion.convert(env_path, dc.Role.SERVER)
        if conversion.changed:
            if not conversion.report.ok:
                raise ValueError(".env does not pass the server checks:\n" + conversion.render_report())
            saved = env_conversion.write(conversion)
            notes.append(f"converted .env to the server role (original at {saved.name})")
        else:
            notes.append("kept .env (already a server configuration)")
    return notes


def _provision_local_panel(project_dir: Path, env: Mapping[str, str], profile: str,
                           client_id: str | None) -> dict[str, str]:
    from remote_display import provisioning

    port = (env.get("DESK_DISPLAY_SERVER_PORT") or "8765").strip()
    client_id = client_id or f"{os.uname().nodename.split('.')[0].lower()}-panel"
    if (env.get("DESK_DISPLAY_SERVER_ENROLLMENT") or "").strip().lower() == "shared":
        token = (env.get("DESK_DISPLAY_SERVER_AUTH_TOKEN") or "").strip()
        if not token:
            raise ValueError("shared enrollment needs DESK_DISPLAY_SERVER_AUTH_TOKEN in .env")
        return {
            "DESK_DISPLAY_SERVER_URL": f"http://127.0.0.1:{port}",
            "DESK_DISPLAY_CLIENT_ID": client_id,
            "DESK_DISPLAY_CLIENT_TOKEN": token,
        }
    store = provisioning.ProvisioningStore(provisioning.provisioning_path(env))
    try:
        try:
            issued = store.provision(client_id, profile, actor="installer")
        except provisioning.AlreadyProvisionedError:
            # Its .env.client is gone, so the old credential is unusable anyway.
            issued = store.rotate(client_id, actor="installer")
    except provisioning.ProvisioningError as exc:
        raise ValueError(f"could not provision the local panel: {exc}") from None
    return {
        "DESK_DISPLAY_SERVER_URL": f"http://127.0.0.1:{port}",
        "DESK_DISPLAY_CLIENT_ID": issued.client_id,
        "DESK_DISPLAY_CLIENT_TOKEN": issued.credential,
    }


# ─── Command line ──────────────────────────────────────────────────────────


def panel_output(mode: Mode | str, project_dir: Path) -> str | None:
    """The panel's DESK_DISPLAY_OUTPUT from its env file, if set."""

    mode = Mode(mode)
    if mode is Mode.SERVER:
        return None
    marked = read_marker(project_dir)
    if marked is not None and marked.output:
        return marked.output
    import deployment_config as dc

    path = project_dir / PANEL_ENV[mode]
    if not path.exists() and mode is not Mode.STANDALONE:
        path = project_dir / ".env"
    return (dc.parse_env_file(path).get("DESK_DISPLAY_OUTPUT") or None) if path.exists() else None


def _plan(mode: Mode, output: str | None) -> dict:
    return {
        "mode": mode.value,
        "requirements": requirements_file(mode, output),
        "env_files": list(ENV_FILES[mode]),
        "panel_env": PANEL_ENV.get(mode),
        "hardware": mode is not Mode.SERVER,
        "services": list(start_order(mode)),
        "disable": list(su.disabled_for(mode)),
        "preserved_on_upgrade": list(preserved_on_upgrade(mode)),
        "uninstall_backups": [item.path for item in uninstall_backups(mode)],
    }


def _cli(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m install_modes", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {}
    for name in ("plan", "requirements", "services", "disable", "units", "mark", "snapshot", "backup",
                 "prepare-env", "detect"):
        cmd = commands[name] = sub.add_parser(name)
        cmd.add_argument("--mode", choices=[m.value for m in Mode], help="default: the installed mode")
        cmd.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
        cmd.add_argument("--systemd-dir", type=Path, default=SYSTEMD_DIR)
        cmd.add_argument("--output", dest="display_output",
                         help="display output driver (default: recorded at install, else the panel env file)")
    for name in ("units", "mark"):
        commands[name].add_argument("--user", help="service user (default: recorded at install)")
        commands[name].add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                                    help="Environment= line for the panel unit")
    commands["units"].add_argument("--dir", type=Path, required=True, help="directory to write the unit files into")
    commands["units"].add_argument("--python", help="interpreter; default: the project's venv")
    commands["backup"].add_argument("--to", type=Path, required=True)
    commands["backup"].add_argument("--home", type=Path)
    commands["prepare-env"].add_argument("--install-profile", help="the Installers/install.sh panel profile")
    commands["prepare-env"].add_argument("--credentials", type=Path, help="a provisioned .env.client")
    commands["prepare-env"].add_argument("--client-id", help="combined: the local panel's client ID")
    args = parser.parse_args(list(argv) if argv is not None else None)

    project: Path = args.project_dir
    mode = Mode(args.mode) if args.mode else detect_mode(project, args.systemd_dir)
    marked = read_marker(project)
    recorded = marked if marked is not None and marked.mode is mode else Installed(mode)
    try:
        output = None
        if args.command in ("plan", "requirements", "units", "mark"):
            output = args.display_output or panel_output(mode, project)
        if args.command == "detect":
            print(mode.value)
        elif args.command == "plan":
            print(json.dumps(_plan(mode, output), indent=2))
        elif args.command == "requirements":
            print(requirements_file(mode, output))
        elif args.command == "services":
            print("\n".join(start_order(mode)))
        elif args.command == "disable":
            print("\n".join(su.disabled_for(mode)))
        elif args.command in ("units", "mark"):
            environment = dict(item.split("=", 1) for item in args.env if "=" in item) or dict(recorded.environment)
            user = args.user or recorded.user or os.environ.get("SUDO_USER") or os.environ.get("USER") or "pi"
            if args.command == "mark":
                print(write_marker(project, mode, output=output, user=user, environment=environment))
            else:
                for path in write_units(mode, args.dir, project_dir=str(project),
                                        python=args.python or f"{project}/venv/bin/python", user=user,
                                        output=output, environment=environment):
                    print(path)
        elif args.command == "snapshot":
            print(snapshot(mode, project) or "")
        elif args.command == "backup":
            for source, target in backup_for_uninstall(mode, project, args.to, home=args.home):
                print(f"{source} -> {target}")
        elif args.command == "prepare-env":
            for note in prepare_env(mode, project, install_profile=args.install_profile,
                                    credentials=args.credentials, client_id=args.client_id):
                print(note)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
