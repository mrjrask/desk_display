"""Per-client credentials for the render server.

Each provisioned client gets a stable ID, an initial display profile and its
own enrollment credential. The credential is shown exactly once, when it is
created or rotated; the store keeps only its SHA-256. It is what the client
puts in ``DESK_DISPLAY_CLIENT_TOKEN`` and presents to ``/api/v1/register``,
which then issues the short-lived lease credential used for everything else.

Every credential has a ``credential_id``. A lease remembers the one it was
issued under, so rotating, revoking or disabling a client ends its current
lease at once, without touching any other client.

The store is one JSON file, written atomically with mode 0600 under a
process lock and an advisory file lock, so the render server and the
configuration UI can both use it::

    {"schema_version": 1,
     "clients": {"<id>": {"client_id", "display_profile", "credential_hash",
                          "credential_id", "state", "created_at", "updated_at",
                          "history": [{"at", "action", "actor"}]}}}

``state`` is ``active``, ``disabled`` (credential kept, refused until
enabled) or ``revoked`` (credential destroyed; rotate to issue a new one).
Disabling or revoking never removes the client's playlist assignment or
history.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import hmac
import json
import os
import secrets
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from display_profiles import PROFILE_PRESETS
from remote_display.models import ModelValidationError, identifier, profile_id

try:  # pragma: no cover - fcntl is unavailable on Windows
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

SCHEMA_VERSION = 1
MAX_HISTORY = 50
STATES = ("active", "disabled", "revoked")
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROVISIONING_PATH = _PROJECT_ROOT / ".runtime" / "server" / "provisioned_clients.json"


def provisioning_path(env: Mapping[str, str] | None = None) -> Path:
    source = os.environ if env is None else env
    raw = (source.get("DESK_DISPLAY_SERVER_CLIENTS_PATH") or "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_PROVISIONING_PATH


class ProvisioningError(Exception):
    status = 400
    code = "invalid_request"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.details = details

    def as_response(self) -> dict[str, Any]:
        return {"error": self.code, "message": str(self), **self.details}


class AlreadyProvisionedError(ProvisioningError):
    status = 409
    code = "client_exists"


class UnknownProvisionedClientError(ProvisioningError):
    status = 404
    code = "unknown_client"


@dataclass(frozen=True)
class Issued:
    """A newly created credential; ``credential`` is never available again."""

    client_id: str
    display_profile: str
    credential: str
    credential_id: str


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _public(record: Mapping[str, Any]) -> dict[str, Any]:
    """A record without its credential hash."""

    return {k: (list(v) if k == "history" else v) for k, v in record.items() if k != "credential_hash"}


class ProvisioningStore:
    def __init__(self, path: str | os.PathLike[str], *, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path).expanduser()
        self._clock = clock
        self._lock = threading.RLock()
        self._cache: tuple[tuple[int, int] | None, dict[str, Any]] | None = None

    # ── Storage ────────────────────────────────────────────────────────────

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
        if stamp is None:
            data: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "clients": {}}
        else:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("clients"), dict):
                raise ProvisioningError(f"{self.path} is not a provisioning store")
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
                yield data
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

    def _record(self, data: Mapping[str, Any], client_id: str) -> dict[str, Any]:
        record = data["clients"].get(client_id)
        if record is None:
            raise UnknownProvisionedClientError("no provisioned client with this ID", client_id=client_id)
        return record

    def _event(self, record: dict[str, Any], action: str, actor: str) -> None:
        now = self._clock()
        record["updated_at"] = now
        record["history"] = (list(record.get("history") or []) + [{"at": now, "action": action,
                                                                    "actor": actor}])[-MAX_HISTORY:]

    @staticmethod
    def _issue(record: dict[str, Any]) -> tuple[str, str]:
        secret = "ddc_" + secrets.token_urlsafe(32)
        credential_id = secrets.token_hex(8)
        record["credential_hash"] = _hash(secret)
        record["credential_id"] = credential_id
        return secret, credential_id

    # ── Operations ─────────────────────────────────────────────────────────

    def provision(self, client_id: str, display_profile: str, *, actor: str = "admin") -> Issued:
        client_id = identifier(client_id, "client_id")
        display_profile = profile_id(display_profile, "display_profile")
        if display_profile not in PROFILE_PRESETS:
            raise ModelValidationError("display_profile", "unknown display profile")
        with self._transaction() as data:
            if client_id in data["clients"]:
                raise AlreadyProvisionedError("this client ID is already provisioned; rotate its credential "
                                              "instead", client_id=client_id)
            now = self._clock()
            record: dict[str, Any] = {"client_id": client_id, "display_profile": display_profile,
                                      "state": "active", "created_at": now, "history": []}
            secret, credential_id = self._issue(record)
            self._event(record, "provisioned", actor)
            data["clients"][client_id] = record
        return Issued(client_id, display_profile, secret, credential_id)

    def rotate(self, client_id: str, *, actor: str = "admin") -> Issued:
        """Issue a new credential; the old one and its lease stop working. Re-activates a revoked client."""

        client_id = identifier(client_id, "client_id")
        with self._transaction() as data:
            record = self._record(data, client_id)
            if record["state"] == "disabled":
                raise ProvisioningError("enable this client before rotating its credential", client_id=client_id)
            secret, credential_id = self._issue(record)
            record["state"] = "active"
            self._event(record, "rotated", actor)
        return Issued(client_id, record["display_profile"], secret, credential_id)

    def revoke(self, client_id: str, *, actor: str = "admin") -> dict[str, Any]:
        return self._set_state(client_id, "revoked", actor, destroy=True)

    def set_disabled(self, client_id: str, disabled: bool, *, actor: str = "admin") -> dict[str, Any]:
        client_id = identifier(client_id, "client_id")
        with self._transaction() as data:
            record = self._record(data, client_id)
            if record["state"] == "revoked":
                raise ProvisioningError("a revoked client is re-activated by rotating its credential",
                                        client_id=client_id)
            record["state"] = "disabled" if disabled else "active"
            self._event(record, "disabled" if disabled else "enabled", actor)
            return _public(record)

    def _set_state(self, client_id: str, state: str, actor: str, *, destroy: bool) -> dict[str, Any]:
        client_id = identifier(client_id, "client_id")
        with self._transaction() as data:
            record = self._record(data, client_id)
            record["state"] = state
            if destroy:
                record["credential_hash"] = None
                record["credential_id"] = None
            self._event(record, state, actor)
            return _public(record)

    # ── Queries ────────────────────────────────────────────────────────────

    def get(self, client_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._load()["clients"].get(client_id)
            return None if record is None else _public(record)

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [_public(r) for _, r in sorted(self._load()["clients"].items())]

    def verify(self, client_id: str, secret: str | None) -> str | None:
        """The ``credential_id`` when *secret* is *client_id*'s active credential, else ``None``."""

        if not secret or not isinstance(secret, str):
            return None
        with self._lock:
            record = self._load()["clients"].get(client_id)
        if record is None or record.get("state") != "active" or not record.get("credential_hash"):
            return None
        if not hmac.compare_digest(_hash(secret), record["credential_hash"]):
            return None
        return record["credential_id"]

    def current_credential_id(self, client_id: str) -> str | None:
        """The credential a lease must have been issued under, or ``None`` if the client may not connect."""

        with self._lock:
            record = self._load()["clients"].get(client_id)
        if record is None or record.get("state") != "active":
            return None
        return record.get("credential_id")


def client_env(issued: Issued, server_url: str | None) -> tuple[str, list[str]]:
    """The ``.env.client`` for a newly issued credential, and any transport warnings.

    It carries only the client's own settings: never the server token, the
    admin token or any provider credential.
    """

    url = server_url or "https://render-server.example:8765"
    lines = [
        "# Desk Display client, provisioned by the render server.",
        "# Keep this file private (chmod 600): the token below is shown only once.",
        "DESK_DISPLAY_ROLE=client",
        f"DESK_DISPLAY_SERVER_URL={url}",
        f"DESK_DISPLAY_CLIENT_ID={issued.client_id}",
        f"DESK_DISPLAY_CLIENT_TOKEN={issued.credential}",
        f"DESK_DISPLAY_PROFILE={issued.display_profile}",
        "",
    ]
    return "\n".join(lines), transport_warnings(url)


def transport_warnings(url: str) -> list[str]:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    loopback = host == "localhost" or host.startswith("127.") or host == "::1"
    if parts.scheme == "http" and not loopback:
        return [f"{url} is plain HTTP: the client's bearer credential would cross the network unencrypted. "
                "Use HTTPS (DESK_DISPLAY_SERVER_TLS_CERT/KEY or a reverse proxy)."]
    return []


def _cli(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m remote_display.provisioning",
                                     description="Provision and manage display client credentials.")
    parser.add_argument("--store", help="provisioning store (default: DESK_DISPLAY_SERVER_CLIENTS_PATH)")
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("provision", help="create a client and print its .env.client once")
    add.add_argument("client_id")
    add.add_argument("--profile", required=True, choices=sorted(PROFILE_PRESETS))
    add.add_argument("--server-url", default=os.environ.get("DESK_DISPLAY_SERVER_PUBLIC_URL"))
    rotate = sub.add_parser("rotate", help="issue a new credential and print its .env.client once")
    rotate.add_argument("client_id")
    rotate.add_argument("--server-url", default=os.environ.get("DESK_DISPLAY_SERVER_PUBLIC_URL"))
    for name in ("revoke", "disable", "enable"):
        sub.add_parser(name).add_argument("client_id")
    sub.add_parser("list", help="list provisioned clients (never their credentials)")
    args = parser.parse_args(list(argv) if argv is not None else None)
    store = ProvisioningStore(args.store or provisioning_path())
    try:
        if args.command in {"provision", "rotate"}:
            issued = (store.provision(args.client_id, args.profile, actor="cli") if args.command == "provision"
                      else store.rotate(args.client_id, actor="cli"))
            text, warnings = client_env(issued, args.server_url)
            for warning in warnings:
                print(f"warning: {warning}", file=sys.stderr)
            sys.stdout.write(text)
        elif args.command == "revoke":
            store.revoke(args.client_id, actor="cli")
        elif args.command in {"disable", "enable"}:
            store.set_disabled(args.client_id, args.command == "disable", actor="cli")
        else:
            for record in store.records():
                print(f"{record['client_id']}\t{record['state']}\t{record['display_profile']}")
    except (ProvisioningError, ModelValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


__all__ = [
    "AlreadyProvisionedError",
    "Issued",
    "ProvisioningError",
    "ProvisioningStore",
    "UnknownProvisionedClientError",
    "client_env",
    "provisioning_path",
    "transport_warnings",
]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
