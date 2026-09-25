"""Client leases and demand discovery for the render server.

The registry tracks three kinds of demand:

* **dynamic** clients that registered and keep renewing a lease with
  heartbeats.  A lease that misses its deadline expires, its demand is
  dropped, and its client ID becomes free to register again;
* **static** clients configured on the server (``DESK_DISPLAY_STATIC_CLIENTS``),
  whose demand is known before, and regardless of, any connection;
* explicit **pre-render** demand added by an administrator, such as a
  profile the server should keep warm for clients that are about to arrive.

:meth:`ClientRegistry.demand_entries` merges all three; :meth:`render_plan`
turns them into de-duplicated render keys with :func:`models.plan_renders`.

Lease lifecycle for one client ID::

    (none) --register--> active --heartbeat--> active
                           |  \\--deadline passes--> expired --register--> active
                           \\--admin disable--> disabled --admin enable--> (none)

Duplicate registrations are deterministic: while a lease is active, only a
request carrying that lease's credential may re-register the ID (a renewal);
any other registration is refused until the lease expires.  The credential
is returned once, at registration, and only its SHA-256 is stored.
"""
from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import secrets
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from display_profiles import PROFILE_PRESETS
from protocol_versions import RENDER_PACKAGE_SCHEMA_VERSION
from remote_display.models import (
    ClientCapabilities,
    ClientDemand,
    ClientStatus,
    ModelValidationError,
    PackageCapabilities,
    RenderKey,
    ScreenRevisions,
    identifier,
    plan_renders,
    profile_id,
    screen_id,
)

STATIC_CLIENT_SOFTWARE_VERSION = "static"
MAX_PRERENDER_ENTRIES = 64


class RegistryError(Exception):
    """A request the registry refuses.  ``status`` is the HTTP status to use."""

    status = 400
    code = "invalid_request"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.details = details

    def as_response(self) -> dict[str, Any]:
        return {"error": self.code, "message": str(self), **self.details}


class DuplicateClientError(RegistryError):
    status = 409
    code = "client_id_in_use"


class ClientDisabledError(RegistryError):
    status = 403
    code = "client_disabled"


class StaticProfileMismatchError(RegistryError):
    status = 409
    code = "static_profile_mismatch"


class UnknownClientError(RegistryError):
    status = 404
    code = "unknown_client"


class UnknownLeaseError(RegistryError):
    status = 401
    code = "invalid_client_credential"


@dataclass(frozen=True)
class Assignment:
    """The playlist a client is assigned, as the server's playlist store reports it."""

    playlist_id: str
    playlist_revision: str
    screens: tuple[str, ...] = ()
    alternates: tuple[str, ...] = ()


AssignmentLookup = Callable[[str], "Assignment | None"]


def _no_assignment(_client_id: str) -> Assignment | None:
    return None


@dataclass
class ClientRecord:
    client_id: str
    capabilities: ClientCapabilities
    static: bool = False
    credential_hash: str | None = None
    registered_at: float | None = None
    last_seen: float | None = None
    lease_expires_at: float | None = None
    demand: ClientDemand | None = None
    status: ClientStatus | None = None
    disabled: bool = False
    delivered_playlist_revision: str | None = None
    # The provisioned credential this lease was issued under (None in shared mode).
    enrollment_id: str | None = None

    def lease_state(self, now: float) -> str:
        if self.disabled:
            return "disabled"
        if self.lease_expires_at is None:
            return "static" if self.static else "expired"
        return "active" if now < self.lease_expires_at else "expired"


@dataclass(frozen=True)
class DemandEntry:
    """One source of render demand after merging."""

    source: str  # "dynamic", "static" or "prerender"
    client_id: str
    capabilities: ClientCapabilities
    demand: ClientDemand


@dataclass(frozen=True)
class Registration:
    record: ClientRecord
    credential: str
    renewed: bool


@dataclass
class ClientRegistry:
    """Thread-safe store of client leases, static clients and pre-render demand."""

    lease_seconds: int = 300
    sync_interval_seconds: int = 30
    static_clients: Mapping[str, str] = field(default_factory=dict)
    assignments: AssignmentLookup = _no_assignment
    clock: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        if self.lease_seconds < 30:
            raise ValueError("lease_seconds must be at least 30")
        self._lock = threading.RLock()
        self._clients: dict[str, ClientRecord] = {}
        self._prerender: dict[str, DemandEntry] = {}
        for client_id, profile in self.static_clients.items():
            client_id = identifier(client_id, "static_clients")
            self._clients[client_id] = ClientRecord(
                client_id=client_id,
                capabilities=synthetic_capabilities(client_id, profile),
                static=True,
            )

    # ── Lease lifecycle ────────────────────────────────────────────────────

    @property
    def heartbeat_interval_seconds(self) -> int:
        """Recommended heartbeat: three chances to renew within one lease."""

        return max(5, self.lease_seconds // 3)

    def register(
        self,
        capabilities: ClientCapabilities,
        demand: ClientDemand | None = None,
        *,
        credential: str | None = None,
        enrollment_id: str | None = None,
    ) -> Registration:
        if demand is not None:
            demand.matches(capabilities)
        capabilities.require_supported()
        now = self.clock()
        with self._lock:
            record = self._clients.get(capabilities.client_id)
            renewed = False
            if record is not None:
                if record.disabled:
                    raise ClientDisabledError("this client ID has been disabled by an administrator")
                if record.static and record.capabilities.display_profile != capabilities.display_profile:
                    raise StaticProfileMismatchError(
                        "this client ID is statically configured for a different display profile",
                        configured_profile=record.capabilities.display_profile,
                    )
                if record.lease_state(now) == "active":
                    if credential is None or not self._credential_matches(record, credential):
                        raise DuplicateClientError(
                            "another client holds an active lease for this ID",
                            retry_after_seconds=max(1, int(record.lease_expires_at - now + 0.999)),
                        )
                    renewed = True
            new_credential = secrets.token_urlsafe(32)
            record = ClientRecord(
                client_id=capabilities.client_id,
                capabilities=capabilities,
                static=record.static if record else False,
                credential_hash=_hash(new_credential),
                registered_at=now,
                last_seen=now,
                lease_expires_at=now + self.lease_seconds,
                demand=demand if demand is not None else (record.demand if renewed and record else None),
                status=record.status if record else None,
                delivered_playlist_revision=record.delivered_playlist_revision if record else None,
                enrollment_id=enrollment_id,
            )
            self._clients[record.client_id] = record
            return Registration(record=replace(record), credential=new_credential, renewed=renewed)

    def authenticate(self, client_id: str, credential: str) -> ClientRecord:
        """Return the client's active record, or raise :class:`UnknownLeaseError`."""

        now = self.clock()
        with self._lock:
            record = self._clients.get(client_id)
            if record is None or record.disabled or not self._credential_matches(record, credential):
                raise UnknownLeaseError("unknown client or credential")
            if record.lease_state(now) != "active":
                raise UnknownLeaseError("lease expired; register again", lease_state="expired")
            return replace(record)

    def heartbeat(
        self,
        client_id: str,
        credential: str,
        status: ClientStatus,
        demand: ClientDemand | None = None,
    ) -> ClientRecord:
        record = self.authenticate(client_id, credential)
        if status.client_id != client_id:
            raise ModelValidationError("client_id", "status names a different client")
        if demand is not None:
            demand.matches(record.capabilities)
        now = self.clock()
        with self._lock:
            current = self._clients[client_id]
            current.status = status
            current.last_seen = now
            current.lease_expires_at = now + self.lease_seconds
            if demand is not None:
                current.demand = demand
            return replace(current)

    def expire(self) -> list[str]:
        """Drop the demand of every lapsed dynamic lease; return their IDs."""

        now = self.clock()
        expired = []
        with self._lock:
            for client_id, record in list(self._clients.items()):
                if record.lease_expires_at is None or record.lease_state(now) != "expired":
                    continue
                expired.append(client_id)
                record.credential_hash = None
                record.demand = None
                if record.static:
                    # Static clients keep their configured demand; only the lease ends.
                    record.lease_expires_at = None
        return expired

    def end_lease(self, client_id: str) -> None:
        """End *client_id*'s lease now (its credential was rotated or revoked)."""

        with self._lock:
            record = self._clients.get(client_id)
            if record is not None and record.lease_expires_at is not None:
                record.credential_hash = None
                record.lease_expires_at = None
                record.enrollment_id = None
                if not record.static:
                    record.demand = None

    def set_disabled(self, client_id: str, disabled: bool) -> ClientRecord:
        client_id = identifier(client_id)
        with self._lock:
            record = self._clients.get(client_id)
            if record is None:
                raise UnknownClientError("unknown client")
            record.disabled = disabled
            if disabled:
                record.credential_hash = None
                record.lease_expires_at = None
                record.demand = None
            return replace(record)

    def mark_delivered(self, client_id: str, playlist_revision: str | None) -> None:
        """Record the playlist revision the server last sent to *client_id*."""

        with self._lock:
            record = self._clients.get(client_id)
            if record is not None:
                record.delivered_playlist_revision = playlist_revision

    def snapshot(self) -> dict[str, Any]:
        """Credential-free view of every client, for the configuration UI."""

        now = self.clock()
        clients = {}
        for record in self.records():
            clients[record.client_id] = {
                "client_id": record.client_id,
                "static": record.static,
                "disabled": record.disabled,
                "lease_state": record.lease_state(now),
                "registered_at": _iso(record.registered_at),
                "last_seen": _iso(record.last_seen),
                "lease_expires_at": _iso(record.lease_expires_at),
                "capabilities": record.capabilities.to_wire(),
                "status": None if record.status is None else record.status.to_wire(),
                "delivered_playlist_revision": record.delivered_playlist_revision,
            }
        return {
            "schema_version": 1,
            "generated_at": _iso(now),
            "lease_seconds": self.lease_seconds,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "clients": clients,
        }

    def write_snapshot(self, path: str | os.PathLike[str]) -> None:
        """Atomically publish :meth:`snapshot` to *path*."""

        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.snapshot(), handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, target)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise

    def get(self, client_id: str) -> ClientRecord | None:
        with self._lock:
            record = self._clients.get(client_id)
            return None if record is None else replace(record)

    def records(self) -> list[ClientRecord]:
        with self._lock:
            return [replace(record) for _, record in sorted(self._clients.items())]

    # ── Demand ─────────────────────────────────────────────────────────────

    def add_prerender(self, name: str, profile: str, screens: Iterable[str]) -> DemandEntry:
        name = identifier(name, "name")
        capabilities = synthetic_capabilities(f"prerender.{name}"[:64], profile)
        demand = _demand_for(capabilities, list(screens), revision="prerender")
        entry = DemandEntry("prerender", capabilities.client_id, capabilities, demand)
        with self._lock:
            if name not in self._prerender and len(self._prerender) >= MAX_PRERENDER_ENTRIES:
                raise RegistryError(f"at most {MAX_PRERENDER_ENTRIES} pre-render entries are allowed")
            self._prerender[name] = entry
        return entry

    def remove_prerender(self, name: str) -> bool:
        with self._lock:
            return self._prerender.pop(identifier(name, "name"), None) is not None

    def demand_entries(self) -> list[DemandEntry]:
        """Merge active dynamic, static and pre-render demand.

        A client's own reported demand wins; otherwise the screens of its
        assigned playlist are used.  Expired, disabled and unassigned clients
        without reported demand contribute nothing.
        """

        self.expire()
        now = self.clock()
        entries: list[DemandEntry] = []
        with self._lock:
            records = [replace(r) for _, r in sorted(self._clients.items())]
            prerender = [entry for _, entry in sorted(self._prerender.items())]
        for record in records:
            state = record.lease_state(now)
            if state in {"disabled", "expired"}:
                continue
            demand = record.demand
            if demand is None:
                assignment = self.assignments(record.client_id)
                if assignment is None or not assignment.screens:
                    continue
                demand = _demand_for(
                    record.capabilities,
                    assignment.screens,
                    revision=assignment.playlist_revision,
                    alternates=assignment.alternates,
                )
            source = "dynamic" if state == "active" else "static"
            entries.append(DemandEntry(source, record.client_id, record.capabilities, demand))
        return entries + prerender

    def render_plan(
        self,
        revisions: Mapping[str, ScreenRevisions],
        *,
        client_specific_screens: Iterable[str] = (),
    ) -> dict[RenderKey, frozenset[str]]:
        entries = self.demand_entries()
        return plan_renders(
            [(entry.capabilities, entry.demand) for entry in entries],
            revisions,
            client_specific_screens=client_specific_screens,
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _credential_matches(record: ClientRecord, credential: str) -> bool:
        if not record.credential_hash or not isinstance(credential, str) or not credential:
            return False
        return hmac.compare_digest(_hash(credential), record.credential_hash)


def _iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_snapshot(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a registry snapshot; a missing or unreadable file means no clients."""

    try:
        data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"schema_version": 1, "clients": {}}
    if not isinstance(data, dict) or not isinstance(data.get("clients"), dict):
        return {"schema_version": 1, "clients": {}}
    return data


def _hash(credential: str) -> str:
    return hashlib.sha256(credential.encode("utf-8")).hexdigest()


def synthetic_capabilities(client_id: str, profile: str) -> ClientCapabilities:
    """Capabilities for a client the server knows only by ID and profile."""

    preset = PROFILE_PRESETS[profile_id(profile)]
    return ClientCapabilities(
        protocol_version=1,
        client_software_version=STATIC_CLIENT_SOFTWARE_VERSION,
        client_id=client_id,
        display_profile=preset.profile_id,
        logical_width=preset.width,
        logical_height=preset.height,
        image_formats=preset.image_formats,
        color_modes=(preset.color_mode,),
        render_package_versions=(RENDER_PACKAGE_SCHEMA_VERSION,),
    )


def _demand_for(
    capabilities: ClientCapabilities,
    screens: Iterable[str],
    *,
    revision: str,
    alternates: Iterable[str] = (),
) -> ClientDemand:
    screens = [screen_id(s, "screens") for s in screens]
    return ClientDemand(
        client_id=capabilities.client_id,
        playlist_revision=revision,
        required_screens=tuple(screens),
        alternate_screens=tuple(screen_id(s, "alternates") for s in alternates),
        package_capabilities=PackageCapabilities(
            render_package_versions=capabilities.render_package_versions,
            image_formats=tuple(
                f for f in capabilities.image_formats if f in capabilities.render_profile.image_formats
            ),
        ),
        sync_interval_seconds=30,
    )
