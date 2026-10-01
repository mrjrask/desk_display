"""Maintenance commands the config UI sends to a display client.

The Display Clients page can ask a client to update its checkout, upgrade,
restart its service, reset its screenshots or clear its caches. Nothing reaches the client directly: the config UI queues the
command in a small store the render server shares, the server hands it to the
client in the response to the client's next authenticated heartbeat, and the
client reports the outcome in a later heartbeat. A client only ever runs one
of :data:`ACTIONS`, never a command line sent over the network:

``update``
    ``git pull --ff-only`` in the client's own checkout, as the service user.
    ``scripts/upgrade.sh`` is deliberately not run: it restarts every Desk
    Display service with sudo, which would kill the client running it
    mid-upgrade. Use ``upgrade`` for dependency or unit changes.
``upgrade``
    ``scripts/upgrade.sh`` in a transient systemd unit (``sudo -n systemd-run``
    as the service user), so it outlives the client restart it ends with. Its
    output goes to a log beside the results file; whichever client process
    finds the exit code there (usually the restarted one) reports it. Needs
    the passwordless sudo ``upgrade.sh`` already relies on.
``reset_screenshots`` / ``clear_caches``
    ``scripts/reset_screenshots.sh`` or ``scripts/clear-caches.sh``, run by
    the client and reported with their output. Their ``sudo`` steps work only
    where sudo needs no password (a service has no terminal to ask on).
``restart``
    The client stops playback and exits; ``desk_display_client.service`` runs
    under ``Restart=always``, so systemd starts it again a few seconds later.
    No sudo is involved. The result is reported by the restarted process, so
    a success on the page means the client really came back.

Both sides negotiate the feature like telemetry: the server lists
``client_command_versions`` in lease responses, and a client that speaks one
sends ``commands: {"version": 1, "results": [...]}`` with each heartbeat. The
server delivers commands only in reply to such heartbeats, so an older client
never receives one; a command nobody picks up expires after
:data:`COMMAND_TIMEOUT_SECONDS`.

The server-side store is one JSON file (``DESK_DISPLAY_CLIENT_COMMANDS_PATH``,
default ``.runtime/server/client_commands.json``), written atomically under a
process lock and an advisory file lock like the provisioning store::

    {"schema_version": 1,
     "clients": {"<id>": [{"id", "action", "state", "requested_at",
                           "requested_by", "delivered_at", "finished_at",
                           "exit_code", "output"}]}}

``state`` is ``pending`` (queued), ``delivered`` (sent to the client),
``succeeded``, ``failed`` or ``expired``.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import secrets
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from remote_display.models import ModelValidationError, identifier

try:  # pragma: no cover - fcntl is unavailable on Windows
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

LOGGER = logging.getLogger("desk_display.client_commands")

WIRE_VERSION = 1
SCHEMA_VERSION = 1
ACTIONS = {
    "update": "Update code (git pull)",
    "upgrade": "Upgrade",
    "restart": "Restart client",
    "reset_screenshots": "Reset Screenshots",
    "clear_caches": "Clear caches",
}
# Actions that run one of the project's scripts and report its output.
SCRIPTS = {"reset_screenshots": "scripts/reset_screenshots.sh", "clear_caches": "scripts/clear-caches.sh"}
UPGRADE_SCRIPT = "scripts/upgrade.sh"
RUN_LOGGED_SCRIPT = "scripts/helpers/run_logged.sh"
STATES = ("pending", "delivered", "succeeded", "failed", "expired")
UNFINISHED = frozenset({"pending", "delivered"})
# A command the client has not picked up, or not answered, by then is expired.
COMMAND_TIMEOUT_SECONDS = 15 * 60
MAX_PER_CLIENT = 10
MAX_OUTPUT_CHARS = 4000
MAX_RESULTS = 20
GIT_TIMEOUT_SECONDS = 300
SCRIPT_TIMEOUT_SECONDS = 600
# An upgrade with no exit code by then is reported failed with its log so far.
UPGRADE_TIMEOUT_SECONDS = 2 * 60 * 60
_ID_RE = re.compile(r"^[0-9a-f]{16}$")
_URL_USERINFO_RE = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)[^/\s@]+@")
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMMANDS_PATH = _PROJECT_ROOT / ".runtime" / "server" / "client_commands.json"


def commands_path(env: Mapping[str, str] | None = None) -> Path:
    source = os.environ if env is None else env
    raw = (source.get("DESK_DISPLAY_CLIENT_COMMANDS_PATH") or "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_COMMANDS_PATH


class CommandError(Exception):
    status = 400
    code = "invalid_request"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.details = details

    def as_response(self) -> dict[str, Any]:
        return {"error": self.code, "message": str(self), **self.details}


class CommandPendingError(CommandError):
    status = 409
    code = "command_pending"


def clean_output(text: str) -> str:
    """Output safe to store and show: no URL credentials, bounded, tail kept."""

    text = _URL_USERINFO_RE.sub(r"\g<scheme>", text or "")
    if len(text) > MAX_OUTPUT_CHARS:
        text = "…" + text[-(MAX_OUTPUT_CHARS - 1):]
    return text


# ─── Server side ────────────────────────────────────────────────────────────


class CommandStore:
    """Queued commands per client, shared by the config UI and the server."""

    def __init__(self, path: str | os.PathLike[str], *, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path).expanduser()
        self._clock = clock
        self._lock = threading.RLock()
        self._cache: tuple[tuple[int, int] | None, dict[str, Any]] | None = None

    def _stat(self) -> tuple[int, int] | None:
        try:
            info = self.path.stat()
        except FileNotFoundError:
            return None
        return info.st_mtime_ns, info.st_size

    def _load(self) -> dict[str, Any]:
        stamp = self._stat()
        if self._cache is not None and self._cache[0] == stamp:
            return self._cache[1]
        data: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "clients": {}}
        if stamp is not None:
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                LOGGER.warning("Ignoring unreadable client command store %s: %s", self.path, exc)
            else:
                if isinstance(loaded, dict) and isinstance(loaded.get("clients"), dict):
                    data = loaded
        self._cache = (stamp, data)
        return data

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[dict[str, Any]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        with self._lock, open(lock_path, "a+") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                self._cache = None
                data = json.loads(json.dumps(self._load()))
                before = json.dumps(data, sort_keys=True)
                self._expire(data)
                yield data
                if json.dumps(data, sort_keys=True) != before:
                    self._write(data)
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _write(self, data: dict[str, Any]) -> None:
        fd, tmp = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise
        self._cache = None

    def _expire(self, data: Mapping[str, Any]) -> None:
        now = self._clock()
        for commands in data["clients"].values():
            for command in commands:
                self._expire_one(command, now)

    @staticmethod
    def _expire_one(command: dict[str, Any], now: float) -> None:
        if command.get("state") not in UNFINISHED:
            return
        since = command.get("delivered_at") or command.get("requested_at") or 0
        timeout = COMMAND_TIMEOUT_SECONDS
        if command.get("action") == "upgrade" and command["state"] == "delivered":
            timeout += UPGRADE_TIMEOUT_SECONDS  # the display reports it when upgrade.sh ends
        if now - since < timeout:
            return
        picked_up = command["state"] == "delivered"
        command["state"] = "expired"
        command["finished_at"] = now
        command["output"] = (
            "The display took this command but never reported back." if picked_up else
            "The display did not pick this up. It may be offline, or its software predates remote "
            "commands (run bash scripts/upgrade.sh on it once)."
        )

    def queue(self, client_id: str, action: str, *, actor: str = "admin") -> dict[str, Any]:
        client_id = identifier(client_id, "client_id")
        if action not in ACTIONS:
            raise CommandError(f"unknown action {action!r}", field="action")
        with self._transaction() as data:
            commands = data["clients"].setdefault(client_id, [])
            if any(c["action"] == action and c["state"] in UNFINISHED for c in commands):
                raise CommandPendingError(f"{ACTIONS[action]} is already waiting on {client_id}",
                                          client_id=client_id, action=action)
            command = {"id": secrets.token_hex(8), "action": action, "state": "pending",
                       "requested_at": self._clock(), "requested_by": actor, "delivered_at": None,
                       "finished_at": None, "exit_code": None, "output": None}
            commands.append(command)
            del commands[:-MAX_PER_CLIENT]
        LOGGER.info("Queued %s for client %s (by %s)", action, client_id, actor)
        return dict(command)

    def for_client(self, client_id: str) -> list[dict[str, Any]]:
        """This client's commands, newest first, with timeouts applied."""

        now = self._clock()
        with self._lock:
            commands = [dict(c) for c in self._load()["clients"].get(client_id) or []]
        for command in commands:
            self._expire_one(command, now)
        return commands[::-1]

    def take_pending(self, client_id: str) -> list[dict[str, str]]:
        """Mark this client's queued commands delivered and return them."""

        with self._lock:
            waiting = any(c.get("state") == "pending" for c in self._load()["clients"].get(client_id) or [])
        if not waiting:
            return []  # the common heartbeat: nothing written
        taken = []
        with self._transaction() as data:
            for command in data["clients"].get(client_id) or []:
                if command["state"] == "pending":
                    command["state"] = "delivered"
                    command["delivered_at"] = self._clock()
                    taken.append({"id": command["id"], "action": command["action"]})
        return taken

    def record_results(self, client_id: str, results: Iterable[Mapping[str, Any]]) -> None:
        results = list(results)
        if not results:
            return
        with self._transaction() as data:
            commands = {c["id"]: c for c in data["clients"].get(client_id) or []}
            for result in results:
                command = commands.get(result["id"])
                if command is None or command["state"] not in UNFINISHED | {"expired"}:
                    continue  # unknown, or already answered
                command["state"] = result["status"]
                command["exit_code"] = result.get("exit_code")
                command["output"] = result.get("output")
                command["finished_at"] = self._clock()
                LOGGER.info("Client %s reported %s: %s", client_id, command["action"], result["status"])


def parse_heartbeat_commands(value: Any, path: str = "commands") -> list[dict[str, Any]]:
    """Validate a heartbeat's ``commands`` document; return its results."""

    if not isinstance(value, dict):
        raise ModelValidationError(path, "must be an object")
    unknown = sorted(set(value) - {"version", "results"})
    if unknown:
        raise ModelValidationError(f"{path}.{unknown[0]}", "unknown field")
    if value.get("version") != WIRE_VERSION:
        raise ModelValidationError(f"{path}.version", f"must be {WIRE_VERSION}")
    raw = value.get("results") or []
    if not isinstance(raw, list) or len(raw) > MAX_RESULTS:
        raise ModelValidationError(f"{path}.results", f"must be a list of at most {MAX_RESULTS}")
    results = []
    for index, item in enumerate(raw):
        where = f"{path}.results[{index}]"
        if not isinstance(item, dict) or set(item) - {"id", "status", "exit_code", "output"}:
            raise ModelValidationError(where, "must hold only id, status, exit_code and output")
        if not isinstance(item.get("id"), str) or not _ID_RE.match(item["id"]):
            raise ModelValidationError(f"{where}.id", "must be a command ID")
        if item.get("status") not in {"succeeded", "failed"}:
            raise ModelValidationError(f"{where}.status", "must be succeeded or failed")
        code = item.get("exit_code")
        if code is not None and (not isinstance(code, int) or isinstance(code, bool)):
            raise ModelValidationError(f"{where}.exit_code", "must be an integer or null")
        output = item.get("output")
        if output is not None and not isinstance(output, str):
            raise ModelValidationError(f"{where}.output", "must be a string or null")
        results.append({"id": item["id"], "status": item["status"], "exit_code": code,
                        "output": None if output is None else clean_output(output)})
    return results


# ─── Client side ────────────────────────────────────────────────────────────


class CommandRunner:
    """Run the allow-listed commands a server sends; keep results until sent.

    Results live in a small file beside the client cache so a restart's
    result is sent by the process that comes back, and a command ID already
    handled is never run twice.
    """

    def __init__(
        self,
        outbox_path: str | os.PathLike[str],
        *,
        project_dir: str | os.PathLike[str] = _PROJECT_ROOT,
        restart: Callable[[], None] | None = None,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        background: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.outbox_path = Path(outbox_path)
        self.jobs_dir = self.outbox_path.parent / "command_jobs"
        self._clock = clock
        self.project_dir = Path(project_dir)
        self.restart = restart
        self._run = run
        self._background = background
        self._lock = threading.RLock()
        self._busy = threading.Lock()

    def _read(self) -> dict[str, Any]:
        with contextlib.suppress(OSError, ValueError):
            data = json.loads(self.outbox_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {"results": list(data.get("results") or []), "seen": list(data.get("seen") or []),
                        "jobs": list(data.get("jobs") or [])}
        return {"results": [], "seen": [], "jobs": []}

    def _save(self, data: Mapping[str, Any]) -> None:
        self.outbox_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.outbox_path.with_name(self.outbox_path.name + ".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, self.outbox_path)

    def results(self) -> list[dict[str, Any]]:
        """Results waiting for the next heartbeat."""

        pid = os.getpid()
        with self._lock:
            self._collect_jobs()
            ready = [r for r in self._read()["results"] if r.get("hold_pid") != pid]
        return [{k: v for k, v in r.items() if k != "hold_pid"} for r in ready[:MAX_RESULTS]]

    def acknowledge(self, ids: Iterable[str]) -> None:
        """Forget results the server has accepted."""

        done = set(ids)
        if not done:
            return
        with self._lock:
            data = self._read()
            data["results"] = [r for r in data["results"] if r.get("id") not in done]
            self._save(data)

    def _finish(self, command_id: str, status: str, exit_code: int | None, output: str, *,
                hold: bool = False) -> None:
        result = {"id": command_id, "status": status, "exit_code": exit_code, "output": clean_output(output)}
        if hold:  # sent only by a later process
            result["hold_pid"] = os.getpid()
        with self._lock:
            data = self._read()
            data["results"].append(result)
            self._save(data)

    def handle(self, commands: Any) -> None:
        """Start the commands of a heartbeat response not already handled."""

        if not isinstance(commands, list):
            return
        todo = []
        with self._lock:
            data = self._read()
            seen = set(data["seen"])
            for command in commands:
                if not isinstance(command, dict):
                    continue
                command_id, action = command.get("id"), command.get("action")
                if not isinstance(command_id, str) or not _ID_RE.match(command_id) or command_id in seen:
                    continue
                seen.add(command_id)
                data["seen"] = (data["seen"] + [command_id])[-50:]
                if action not in ACTIONS:
                    data["results"].append({"id": command_id, "status": "failed", "exit_code": None,
                                            "output": f"This client does not know the action {action!r}."})
                    continue
                todo.append((command_id, action))
            self._save(data)
        if not todo:
            return
        if self._background:
            threading.Thread(target=self._execute, args=(todo,), name="client-commands", daemon=True).start()
        else:
            self._execute(todo)

    def _execute(self, todo: list[tuple[str, str]]) -> None:
        with self._busy:
            for command_id, action in todo:
                LOGGER.info("Running remote command %s (%s)", action, command_id)
                try:
                    if action == "update":
                        self._update(command_id)
                    elif action == "upgrade":
                        self._upgrade(command_id)
                    elif action in SCRIPTS:
                        self._script(command_id, SCRIPTS[action])
                    else:
                        self._restart(command_id)
                except Exception as exc:  # noqa: BLE001 - report every failure to the page
                    LOGGER.exception("Remote command %s failed", action)
                    self._finish(command_id, "failed", None, f"{type(exc).__name__}: {exc}")

    def _git(self, *args: str, timeout: float = 30) -> subprocess.CompletedProcess:
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}
        return self._run(["git", "-C", str(self.project_dir), *args], capture_output=True, text=True,
                         timeout=timeout, check=False, env=env)

    def _head(self) -> str | None:
        result = self._git("rev-parse", "--short", "HEAD")
        return (result.stdout.strip() or None) if result.returncode == 0 else None

    def _update(self, command_id: str) -> None:
        before = self._head()
        try:
            result = self._git("pull", "--ff-only", timeout=GIT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self._finish(command_id, "failed", None, f"git pull did not finish in {GIT_TIMEOUT_SECONDS} s.")
            return
        output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part and part.strip())
        if result.returncode != 0:
            self._finish(command_id, "failed", result.returncode, output or "git pull failed.")
            return
        after = self._head()
        if before and after and before != after:
            summary = f"Updated {before} → {after}. Restart the client to run the new code."
        else:
            summary = f"Already up to date ({after or before or 'unknown commit'})."
        self._finish(command_id, "succeeded", 0, f"{summary}\n{output}".strip())

    def _script(self, command_id: str, script: str) -> None:
        path = self.project_dir / script
        if not path.is_file():
            self._finish(command_id, "failed", None, f"{script} is missing on this display. Update it first.")
            return
        try:
            result = self._run(["bash", str(path)], cwd=str(self.project_dir), stdin=subprocess.DEVNULL,
                               capture_output=True, text=True, timeout=SCRIPT_TIMEOUT_SECONDS, check=False,
                               env={**os.environ, "LC_ALL": "C.UTF-8"})
        except subprocess.TimeoutExpired:
            self._finish(command_id, "failed", None, f"{script} did not finish in {SCRIPT_TIMEOUT_SECONDS} s.")
            return
        output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part and part.strip())
        status = "succeeded" if result.returncode == 0 else "failed"
        self._finish(command_id, status, result.returncode, output or f"{script} exited with {result.returncode}.")

    def _upgrade(self, command_id: str) -> None:
        """Start upgrade.sh outside this service; :meth:`_collect_jobs` reports it."""

        for script in (UPGRADE_SCRIPT, RUN_LOGGED_SCRIPT):
            if not (self.project_dir / script).is_file():
                self._finish(command_id, "failed", None, f"{script} is missing on this display. Update it first.")
                return
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        log, status = self.jobs_dir / f"upgrade-{command_id}.log", self.jobs_dir / f"upgrade-{command_id}.status"
        for stale in (log, status):
            with contextlib.suppress(FileNotFoundError):
                stale.unlink()
        # A transient system unit is outside this service's cgroup, so the
        # restart upgrade.sh ends with does not kill it. It runs as this user,
        # like a manual upgrade, and upgrade.sh uses sudo where it needs to.
        argv = [
            "sudo", "-n", "systemd-run", f"--unit=desk-display-upgrade-{command_id}", "--collect", "--quiet",
            f"--uid={os.getuid()}", f"--gid={os.getgid()}", f"--working-directory={self.project_dir}",
            f"--setenv=HOME={Path.home()}", f"--setenv=PATH={os.environ.get('PATH') or os.defpath}",
            "--", "bash", str(self.project_dir / RUN_LOGGED_SCRIPT), str(log), str(status),
            "bash", str(self.project_dir / UPGRADE_SCRIPT),
        ]
        try:
            result = self._run(argv, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60,
                               check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            result = subprocess.CompletedProcess(argv, None, "", f"{type(exc).__name__}: {exc}")
        if result.returncode != 0:
            output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part and part.strip())
            self._finish(command_id, "failed", result.returncode,
                         "Could not start the upgrade. It needs systemd-run and passwordless sudo on the "
                         "display; otherwise run bash scripts/upgrade.sh there.\n" + output)
            return
        LOGGER.info("Upgrade %s started; log in %s", command_id, log)
        with self._lock:
            data = self._read()
            data["jobs"].append({"id": command_id, "log": str(log), "status": str(status),
                                 "started_at": self._clock()})
            self._save(data)

    def _collect_jobs(self) -> None:
        """Turn finished (or overdue) upgrades into results. Caller holds the lock."""

        data = self._read()
        if not data["jobs"]:
            return
        keep = []
        for job in data["jobs"]:
            log, status = Path(job["log"]), Path(job["status"])
            code: int | None = None
            try:
                code = int(status.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                if self._clock() - float(job.get("started_at") or 0) < UPGRADE_TIMEOUT_SECONDS:
                    keep.append(job)
                    continue
            try:
                output = log.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                output = ""
            if code is None:
                output = f"No result after {UPGRADE_TIMEOUT_SECONDS // 3600} h. Log so far:\n{output}"
            data["results"].append({"id": job["id"], "status": "succeeded" if code == 0 else "failed",
                                    "exit_code": code, "output": clean_output(output or "upgrade.sh printed nothing.")})
            for path in (log, status):
                with contextlib.suppress(OSError):
                    path.unlink()
        data["jobs"] = keep
        self._save(data)

    def _restart(self, command_id: str) -> None:
        if self.restart is None:
            self._finish(command_id, "failed", None, "This client cannot restart itself here.")
            return
        # Sent by the process systemd starts next, so success means it came back.
        self._finish(command_id, "succeeded", 0, "The client restarted.", hold=True)
        self.restart()


__all__ = [
    "ACTIONS",
    "COMMAND_TIMEOUT_SECONDS",
    "CommandError",
    "CommandPendingError",
    "CommandRunner",
    "CommandStore",
    "WIRE_VERSION",
    "clean_output",
    "commands_path",
    "parse_heartbeat_commands",
]
