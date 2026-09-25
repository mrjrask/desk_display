"""Protocol models exchanged between the render server and display clients.

Four documents describe a remote display:

:class:`ClientCapabilities`
    What a client *is*: versions, stable ID, display profile, canonical
    logical size, formats, colour modes, render-package versions and input
    hardware.  Sent at registration.
:class:`ClientDemand`
    What a client *needs rendered*: the screens its playlist revision uses,
    alternates, touch-expansion targets, package capabilities and its sync
    interval.
:class:`ClientStatus`
    What a client *is doing*: current screen and playlist, accepted
    revisions, sync/cache ages, playback state, diagnostic physical rotation
    and summarized recent errors.
:class:`RenderKey`
    The canonical identity of one rendered artifact.  It contains only the
    inputs that change pixels, so clients with equivalent demand share
    artifacts, and playlist order, playback position, physical rotation and
    client ID never fragment the render cache.

Every model is an immutable, hashable dataclass that validates itself on
construction, so invalid capabilities are rejected before any work is
scheduled.  Each has an explicit versioned wire representation::

    {"type": "client_capabilities", "version": 1, ...fields}

``from_wire`` accepts only a known ``type``/``version`` pair and the exact
field set for that version, and bounds every string, number and list.

This module depends only on the standard library and the dependency-light
:mod:`display_profiles`, :mod:`screens_catalog`, :mod:`protocol_versions` and
:mod:`deployment_config` modules.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import MISSING, dataclass, fields
from typing import Any, ClassVar, TypeVar

from deployment_config import redact_text
from display_profiles import PROFILE_PRESETS, RenderProfile
from protocol_versions import NETWORK_PROTOCOL_VERSION, RENDER_PACKAGE_SCHEMA_VERSION
from screens_catalog import LEGACY_RETIRED_SCREEN_IDS, SCREEN_IDS, canonical_screen_id

# ─── Bounds ─────────────────────────────────────────────────────────────────

MAX_WIRE_BYTES = 64 * 1024
MAX_IDENTIFIER_LENGTH = 64
MAX_REVISION_LENGTH = 128
MAX_TEXT_LENGTH = 200
MAX_SCREENS = 256
MAX_SMALL_LIST = 16
MAX_RECENT_ERRORS = 10
MAX_DIMENSION = 8192
MAX_AGE_SECONDS = 10 * 365 * 24 * 3600
MIN_SYNC_INTERVAL_SECONDS = 5
MAX_SYNC_INTERVAL_SECONDS = 24 * 3600
MAX_VERSION = 1_000_000

IMAGE_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})
COLOR_MODES = frozenset({"RGB", "RGBA", "L", "1"})
PHYSICAL_ROTATIONS = frozenset({0, 90, 180, 270})
PLAYBACK_STATES = frozenset(
    {"starting", "playing", "focus", "paused", "dark", "offline", "error"}
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")
_VERSION_TEXT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

_ACTIVE_SCREEN_IDS = frozenset(SCREEN_IDS)


class ModelValidationError(ValueError):
    """A model or wire document is malformed, out of bounds, or unsupported."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}" if path else message)


class UnsupportedCapabilitiesError(ModelValidationError):
    """Well-formed capabilities that this server cannot serve."""


# ─── Field validators ───────────────────────────────────────────────────────


def _fail(path: str, message: str) -> None:
    raise ModelValidationError(path, message)


def _int(value: Any, path: str, *, minimum: int = 0, maximum: int = MAX_VERSION) -> int:
    if type(value) is not int:
        _fail(path, "must be an integer")
    if not minimum <= value <= maximum:
        _fail(path, f"must be between {minimum} and {maximum}")
    return value


def _number(value: Any, path: str, *, minimum: float = 0, maximum: float = MAX_AGE_SECONDS) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        _fail(path, "must be a finite number")
    if not minimum <= value <= maximum:
        _fail(path, f"must be between {minimum:g} and {maximum:g}")
    return float(value)


def _bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        _fail(path, "must be true or false")
    return value


def _str(value: Any, path: str, *, max_length: int = MAX_TEXT_LENGTH) -> str:
    if not isinstance(value, str):
        _fail(path, "must be a string")
    if len(value) > max_length:
        _fail(path, f"must be at most {max_length} characters")
    return value


def _pattern(value: Any, path: str, pattern: re.Pattern[str], what: str) -> str:
    text = _str(value, path, max_length=MAX_REVISION_LENGTH)
    if not pattern.match(text):
        _fail(path, f"is not a valid {what}")
    return text


def identifier(value: Any, path: str = "client_id") -> str:
    """Validate a client or playlist ID (1-64 of ``A-Z a-z 0-9 . _ -``)."""

    return _pattern(value, path, _IDENTIFIER_RE, "identifier")


def revision(value: Any, path: str = "revision") -> str:
    """Validate an opaque revision token (1-128 of ``A-Z a-z 0-9 . _ : + -``)."""

    return _pattern(value, path, _REVISION_RE, "revision")


def _optional(value: Any, path: str, validator) -> Any:
    return None if value is None else validator(value, path)


def _text(value: Any, path: str) -> str:
    """Bounded, single-line, secret-redacted free text."""

    text = _str(value, path)
    return redact_text(_CONTROL_RE.sub(" ", text)).strip()


def screen_id(value: Any, path: str = "screen_id") -> str:
    """Return the canonical ID for an active screen in :mod:`screens_catalog`."""

    text = _str(value, path, max_length=MAX_IDENTIFIER_LENGTH)
    canonical = canonical_screen_id(text)
    if canonical in LEGACY_RETIRED_SCREEN_IDS or canonical not in _ACTIVE_SCREEN_IDS:
        _fail(path, f"unknown screen ID {text!r}")
    return canonical


def profile_id(value: Any, path: str = "display_profile") -> str:
    text = _str(value, path, max_length=MAX_IDENTIFIER_LENGTH)
    if text not in PROFILE_PRESETS:
        _fail(path, f"unknown display profile {text!r}")
    return text


def _list(value: Any, path: str, *, max_items: int) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        _fail(path, "must be a list")
    if len(value) > max_items:
        _fail(path, f"must have at most {max_items} items")
    return list(value)


def _screen_set(value: Any, path: str) -> tuple[str, ...]:
    """Canonical, de-duplicated, sorted screen IDs.

    Demand describes *which* screens need artifacts; order lives in the
    playlist document, so it never distinguishes two demands.
    """

    items = _list(value, path, max_items=MAX_SCREENS)
    return tuple(sorted({screen_id(item, f"{path}[{i}]") for i, item in enumerate(items)}))


def _choice_set(value: Any, path: str, allowed: frozenset[str], *, upper: bool = True) -> tuple[str, ...]:
    items = _list(value, path, max_items=MAX_SMALL_LIST)
    result = set()
    for index, item in enumerate(items):
        text = _str(item, f"{path}[{index}]", max_length=16)
        text = text.upper() if upper else text
        if text not in allowed:
            _fail(f"{path}[{index}]", f"expected one of {', '.join(sorted(allowed))}")
        result.add(text)
    return tuple(sorted(result))


def _version_set(value: Any, path: str) -> tuple[int, ...]:
    items = _list(value, path, max_items=MAX_SMALL_LIST)
    versions = {_int(item, f"{path}[{i}]", minimum=1) for i, item in enumerate(items)}
    if not versions:
        _fail(path, "must list at least one version")
    return tuple(sorted(versions))


# ─── Wire base ──────────────────────────────────────────────────────────────

M = TypeVar("M", bound="WireModel")


class WireModel:
    """Base for immutable models with an explicit ``type``/``version`` envelope."""

    WIRE_TYPE: ClassVar[str]
    WIRE_VERSION: ClassVar[int] = 1

    def _normalize(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def __post_init__(self) -> None:
        self._normalize()

    def _set(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

    def _fields_to_wire(self) -> dict[str, Any]:
        return {f.name: _to_wire_value(getattr(self, f.name)) for f in fields(self)}  # type: ignore[arg-type]

    def to_wire(self) -> dict[str, Any]:
        return {"type": self.WIRE_TYPE, "version": self.WIRE_VERSION, **self._fields_to_wire()}

    def to_json(self) -> str:
        return json.dumps(self.to_wire(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def _fields_from_wire(cls, payload: Mapping[str, Any], path: str) -> dict[str, Any]:
        return dict(payload)

    @classmethod
    def from_wire(cls: type[M], payload: Any, *, path: str = "") -> M:
        if not isinstance(payload, Mapping):
            _fail(path, "must be a JSON object")
        prefix = f"{path}." if path else ""
        if payload.get("type") != cls.WIRE_TYPE:
            _fail(f"{prefix}type", f"expected {cls.WIRE_TYPE!r}")
        version = payload.get("version")
        if type(version) is not int or version != cls.WIRE_VERSION:
            _fail(f"{prefix}version", f"unsupported {cls.WIRE_TYPE} version {version!r}")
        body = {k: v for k, v in payload.items() if k not in ("type", "version")}
        expected = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        unknown = sorted(str(key) for key in set(body) - expected)
        if unknown:
            _fail(f"{prefix}{unknown[0]}", "unknown field")
        missing = sorted(expected - set(body) - cls._optional_fields())
        if missing:
            _fail(f"{prefix}{missing[0]}", "is required")
        try:
            return cls(**cls._fields_from_wire(body, path))
        except ModelValidationError as exc:
            if not path or exc.path == path or exc.path.startswith((f"{path}.", f"{path}[")):
                raise
            raise type(exc)(f"{prefix}{exc.path}" if exc.path else path, exc.message) from None

    @classmethod
    def from_json(cls: type[M], text: str | bytes) -> M:
        if len(text) > MAX_WIRE_BYTES:
            _fail("", f"document exceeds {MAX_WIRE_BYTES} bytes")
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ModelValidationError("", f"invalid JSON: {exc}") from None
        return cls.from_wire(payload)

    @classmethod
    def _optional_fields(cls) -> frozenset[str]:
        return frozenset(
            f.name for f in fields(cls)  # type: ignore[arg-type]
            if f.default is not MISSING or f.default_factory is not MISSING
        )


def _to_wire_value(value: Any) -> Any:
    if isinstance(value, WireModel):
        return value._fields_to_wire()
    if isinstance(value, tuple):
        return [_to_wire_value(item) for item in value]
    return value


# ─── Capabilities ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HardwareDescription(WireModel):
    """Optional free-form description of the client's hardware, for diagnostics."""

    WIRE_TYPE: ClassVar[str] = "hardware_description"

    model: str | None = None
    panel: str | None = None
    driver: str | None = None

    def _normalize(self) -> None:
        for name in ("model", "panel", "driver"):
            value = getattr(self, name)
            self._set(name, None if value is None else _text(value, f"hardware.{name}") or None)


@dataclass(frozen=True)
class ClientCapabilities(WireModel):
    """What a client can display.  Sent with every registration."""

    WIRE_TYPE: ClassVar[str] = "client_capabilities"

    protocol_version: int
    client_software_version: str
    client_id: str
    display_profile: str
    logical_width: int
    logical_height: int
    image_formats: tuple[str, ...]
    color_modes: tuple[str, ...]
    render_package_versions: tuple[int, ...]
    supports_animation: bool = False
    has_touch: bool = False
    buttons: tuple[str, ...] = ()
    hardware: HardwareDescription | None = None

    def _normalize(self) -> None:
        self._set("protocol_version", _int(self.protocol_version, "protocol_version", minimum=1))
        self._set(
            "client_software_version",
            _pattern(self.client_software_version, "client_software_version", _VERSION_TEXT_RE, "version"),
        )
        self._set("client_id", identifier(self.client_id, "client_id"))
        profile = PROFILE_PRESETS[profile_id(self.display_profile, "display_profile")]
        width = _int(self.logical_width, "logical_width", minimum=1, maximum=MAX_DIMENSION)
        height = _int(self.logical_height, "logical_height", minimum=1, maximum=MAX_DIMENSION)
        if (width, height) != (profile.width, profile.height):
            _fail(
                "logical_width",
                f"{width}x{height} is not the canonical {profile.width}x{profile.height} "
                f"size of {profile.profile_id}; report logical dimensions before physical rotation",
            )
        formats = _choice_set(self.image_formats, "image_formats", IMAGE_FORMATS)
        if not set(formats) & set(profile.image_formats):
            _fail("image_formats", f"none of {', '.join(profile.image_formats)} is supported")
        self._set("image_formats", formats)
        modes = _choice_set(self.color_modes, "color_modes", COLOR_MODES)
        if profile.color_mode not in modes:
            _fail("color_modes", f"profile {profile.profile_id} renders {profile.color_mode!r}")
        self._set("color_modes", modes)
        self._set("render_package_versions", _version_set(self.render_package_versions, "render_package_versions"))
        self._set("supports_animation", _bool(self.supports_animation, "supports_animation"))
        self._set("has_touch", _bool(self.has_touch, "has_touch"))
        buttons = _list(self.buttons, "buttons", max_items=MAX_SMALL_LIST)
        self._set(
            "buttons",
            tuple(sorted({identifier(b, f"buttons[{i}]") for i, b in enumerate(buttons)})),
        )
        if self.hardware is not None and not isinstance(self.hardware, HardwareDescription):
            _fail("hardware", "must be a hardware description")

    @classmethod
    def _fields_from_wire(cls, payload: Mapping[str, Any], path: str) -> dict[str, Any]:
        body = dict(payload)
        hardware = body.get("hardware")
        if hardware is not None:
            if not isinstance(hardware, Mapping):
                _fail("hardware", "must be an object or null")
            unknown = sorted(set(hardware) - {"model", "panel", "driver"})
            if unknown:
                _fail(f"hardware.{unknown[0]}", "unknown field")
            body["hardware"] = HardwareDescription(**hardware)
        return body

    @property
    def render_profile(self) -> RenderProfile:
        return PROFILE_PRESETS[self.display_profile]

    @property
    def color_mode(self) -> str:
        """The colour mode artifacts are rendered in for this client."""

        return self.render_profile.color_mode

    def require_supported(
        self,
        *,
        protocol_versions: Iterable[int] = (NETWORK_PROTOCOL_VERSION,),
        render_package_versions: Iterable[int] = (RENDER_PACKAGE_SCHEMA_VERSION,),
    ) -> None:
        """Raise :class:`UnsupportedCapabilitiesError` unless this server can serve the client."""

        if self.protocol_version not in set(protocol_versions):
            raise UnsupportedCapabilitiesError(
                "protocol_version", f"protocol {self.protocol_version} is not accepted"
            )
        if not set(self.render_package_versions) & set(render_package_versions):
            raise UnsupportedCapabilitiesError(
                "render_package_versions", "no render package version in common with the server"
            )


# ─── Demand ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PackageCapabilities(WireModel):
    """Which render packages a client can download for this demand."""

    WIRE_TYPE: ClassVar[str] = "package_capabilities"

    render_package_versions: tuple[int, ...]
    image_formats: tuple[str, ...]
    supports_animation: bool = False
    max_package_bytes: int = 16 * 1024 * 1024

    def _normalize(self) -> None:
        self._set("render_package_versions", _version_set(self.render_package_versions, "package_capabilities.render_package_versions"))
        formats = _choice_set(self.image_formats, "package_capabilities.image_formats", IMAGE_FORMATS)
        if not formats:
            _fail("package_capabilities.image_formats", "must list at least one format")
        self._set("image_formats", formats)
        self._set("supports_animation", _bool(self.supports_animation, "package_capabilities.supports_animation"))
        self._set(
            "max_package_bytes",
            _int(self.max_package_bytes, "package_capabilities.max_package_bytes", minimum=1024, maximum=1 << 30),
        )


@dataclass(frozen=True)
class ClientDemand(WireModel):
    """The screens a client needs artifacts for at one playlist revision.

    Screen lists are sets: they are canonicalized, de-duplicated and sorted,
    so the order a playlist plays them in never changes the demand.
    """

    WIRE_TYPE: ClassVar[str] = "client_demand"

    client_id: str
    playlist_revision: str
    required_screens: tuple[str, ...]
    package_capabilities: PackageCapabilities
    sync_interval_seconds: int
    alternate_screens: tuple[str, ...] = ()
    touch_targets: tuple[str, ...] = ()

    def _normalize(self) -> None:
        self._set("client_id", identifier(self.client_id, "client_id"))
        self._set("playlist_revision", revision(self.playlist_revision, "playlist_revision"))
        required = _screen_set(self.required_screens, "required_screens")
        alternates = tuple(s for s in _screen_set(self.alternate_screens, "alternate_screens") if s not in required)
        self._set("required_screens", required)
        self._set("alternate_screens", alternates)
        self._set("touch_targets", _screen_set(self.touch_targets, "touch_targets"))
        if not isinstance(self.package_capabilities, PackageCapabilities):
            _fail("package_capabilities", "must be package capabilities")
        self._set(
            "sync_interval_seconds",
            _int(self.sync_interval_seconds, "sync_interval_seconds",
                 minimum=MIN_SYNC_INTERVAL_SECONDS, maximum=MAX_SYNC_INTERVAL_SECONDS),
        )

    @classmethod
    def _fields_from_wire(cls, payload: Mapping[str, Any], path: str) -> dict[str, Any]:
        body = dict(payload)
        package = body.get("package_capabilities")
        if not isinstance(package, Mapping):
            _fail("package_capabilities", "must be an object")
        body["package_capabilities"] = PackageCapabilities.from_wire(
            {"type": PackageCapabilities.WIRE_TYPE, "version": PackageCapabilities.WIRE_VERSION, **package},
            path="package_capabilities",
        )
        return body

    @property
    def all_screens(self) -> tuple[str, ...]:
        """Every screen that needs an artifact: required, alternate and touch targets."""

        return tuple(sorted(set(self.required_screens) | set(self.alternate_screens) | set(self.touch_targets)))

    def matches(self, capabilities: ClientCapabilities) -> None:
        """Raise unless this demand belongs to, and is servable for, *capabilities*."""

        if self.client_id != capabilities.client_id:
            _fail("client_id", "demand and capabilities name different clients")
        if not set(self.package_capabilities.render_package_versions) <= set(capabilities.render_package_versions):
            raise UnsupportedCapabilitiesError(
                "package_capabilities.render_package_versions",
                "requests a package version the client did not advertise",
            )
        if not set(self.package_capabilities.image_formats) & set(capabilities.render_profile.image_formats):
            raise UnsupportedCapabilitiesError(
                "package_capabilities.image_formats",
                f"none are rendered for {capabilities.display_profile}",
            )


# ─── Status ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ErrorSummary(WireModel):
    """One class of recent client error, counted rather than listed."""

    WIRE_TYPE: ClassVar[str] = "error_summary"

    code: str
    message: str
    count: int = 1
    last_seen_age_seconds: float = 0.0

    def _normalize(self) -> None:
        self._set("code", identifier(self.code, "code"))
        self._set("message", _text(self.message, "message"))
        self._set("count", _int(self.count, "count", minimum=1, maximum=1_000_000_000))
        self._set("last_seen_age_seconds", _number(self.last_seen_age_seconds, "last_seen_age_seconds"))


@dataclass(frozen=True)
class AcceptedRevisions(WireModel):
    """Revisions the client has fully downloaded and switched to."""

    WIRE_TYPE: ClassVar[str] = "accepted_revisions"

    manifest_revision: str | None = None
    playlist_revision: str | None = None
    config_revision: str | None = None

    def _normalize(self) -> None:
        for name in ("manifest_revision", "playlist_revision", "config_revision"):
            self._set(name, _optional(getattr(self, name), f"accepted_revisions.{name}", revision))


@dataclass(frozen=True)
class ClientStatus(WireModel):
    """A client's heartbeat report.

    ``physical_rotation`` is diagnostic only: rotation is applied by the
    client's presenter and is never part of a :class:`RenderKey`.
    """

    WIRE_TYPE: ClassVar[str] = "client_status"

    client_id: str
    playback_state: str
    accepted_revisions: AcceptedRevisions
    current_screen: str | None = None
    current_playlist: str | None = None
    last_sync_age_seconds: float | None = None
    cache_age_seconds: float | None = None
    physical_rotation: int = 0
    recent_errors: tuple[ErrorSummary, ...] = ()

    def _normalize(self) -> None:
        self._set("client_id", identifier(self.client_id, "client_id"))
        state = _str(self.playback_state, "playback_state", max_length=16)
        if state not in PLAYBACK_STATES:
            _fail("playback_state", f"expected one of {', '.join(sorted(PLAYBACK_STATES))}")
        if not isinstance(self.accepted_revisions, AcceptedRevisions):
            _fail("accepted_revisions", "must be accepted revisions")
        self._set("current_screen", _optional(self.current_screen, "current_screen", screen_id))
        self._set("current_playlist", _optional(self.current_playlist, "current_playlist", identifier))
        for name in ("last_sync_age_seconds", "cache_age_seconds"):
            self._set(name, _optional(getattr(self, name), name, _number))
        rotation = _int(self.physical_rotation, "physical_rotation", maximum=270)
        if rotation not in PHYSICAL_ROTATIONS:
            _fail("physical_rotation", "expected 0, 90, 180 or 270")
        errors = _list(self.recent_errors, "recent_errors", max_items=MAX_RECENT_ERRORS)
        for index, error in enumerate(errors):
            if not isinstance(error, ErrorSummary):
                _fail(f"recent_errors[{index}]", "must be an error summary")
        self._set("recent_errors", tuple(errors))

    @classmethod
    def _fields_from_wire(cls, payload: Mapping[str, Any], path: str) -> dict[str, Any]:
        body = dict(payload)
        accepted = body.get("accepted_revisions")
        if not isinstance(accepted, Mapping):
            _fail("accepted_revisions", "must be an object")
        body["accepted_revisions"] = AcceptedRevisions.from_wire(
            {"type": AcceptedRevisions.WIRE_TYPE, "version": AcceptedRevisions.WIRE_VERSION, **accepted},
            path="accepted_revisions",
        )
        errors = _list(body.get("recent_errors", []), "recent_errors", max_items=MAX_RECENT_ERRORS)
        parsed = []
        for index, error in enumerate(errors):
            if not isinstance(error, Mapping):
                _fail(f"recent_errors[{index}]", "must be an object")
            parsed.append(ErrorSummary.from_wire(
                {"type": ErrorSummary.WIRE_TYPE, "version": ErrorSummary.WIRE_VERSION, **error},
                path=f"recent_errors[{index}]",
            ))
        body["recent_errors"] = tuple(parsed)
        return body


# ─── Render identity ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScreenRevisions(WireModel):
    """The server-side revisions that change a screen's pixels."""

    WIRE_TYPE: ClassVar[str] = "screen_revisions"

    style_revision: str
    data_revision: str
    renderer_revision: str

    def _normalize(self) -> None:
        for name in ("style_revision", "data_revision", "renderer_revision"):
            self._set(name, revision(getattr(self, name), name))


@dataclass(frozen=True)
class RenderKey(WireModel):
    """Canonical identity of one rendered artifact.

    Two requests with equal keys produce identical output, so the server
    renders once and every client shares the artifact.  The key deliberately
    has no field for playlist order, playback position, physical rotation or
    (unless the output is intentionally client-specific) client ID.
    """

    WIRE_TYPE: ClassVar[str] = "render_key"

    screen_id: str
    render_profile: str
    width: int
    height: int
    color_mode: str
    style_revision: str
    data_revision: str
    renderer_revision: str
    client_scope: str | None = None

    def _normalize(self) -> None:
        self._set("screen_id", screen_id(self.screen_id, "screen_id"))
        profile = PROFILE_PRESETS[profile_id(self.render_profile, "render_profile")]
        width = _int(self.width, "width", minimum=1, maximum=MAX_DIMENSION)
        height = _int(self.height, "height", minimum=1, maximum=MAX_DIMENSION)
        if (width, height) != (profile.width, profile.height):
            _fail("width", f"{width}x{height} is not the canonical size of {profile.profile_id}")
        mode = _str(self.color_mode, "color_mode", max_length=8)
        if mode != profile.color_mode:
            _fail("color_mode", f"profile {profile.profile_id} renders {profile.color_mode!r}")
        for name in ("style_revision", "data_revision", "renderer_revision"):
            self._set(name, revision(getattr(self, name), name))
        self._set("client_scope", _optional(self.client_scope, "client_scope", identifier))

    @classmethod
    def for_screen(
        cls,
        screen: str,
        profile: RenderProfile | str,
        revisions: ScreenRevisions,
        *,
        client_id: str | None = None,
    ) -> RenderKey:
        """Build the key for *screen* on *profile*.

        Pass ``client_id`` only for output that is intentionally specific to
        one client; everything else is shared across clients.
        """

        resolved = PROFILE_PRESETS[profile_id(profile if isinstance(profile, str) else profile.profile_id)]
        return cls(
            screen_id=screen,
            render_profile=resolved.profile_id,
            width=resolved.width,
            height=resolved.height,
            color_mode=resolved.color_mode,
            style_revision=revisions.style_revision,
            data_revision=revisions.data_revision,
            renderer_revision=revisions.renderer_revision,
            client_scope=client_id,
        )

    @property
    def digest(self) -> str:
        """Stable content address for the artifact (hex SHA-256 of the wire form)."""

        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def demand_render_keys(
    capabilities: ClientCapabilities,
    demand: ClientDemand,
    revisions: Mapping[str, ScreenRevisions],
    *,
    client_specific_screens: Iterable[str] = (),
) -> frozenset[RenderKey]:
    """Return the render keys needed to satisfy *demand*.

    Validation happens first, so an incompatible client fails here, before
    any render work is scheduled.  ``revisions`` maps each canonical screen ID
    to its current revisions; a screen without an entry cannot be scheduled.
    """

    capabilities.require_supported()
    demand.matches(capabilities)
    scoped = {screen_id(s, "client_specific_screens") for s in client_specific_screens}
    keys = set()
    for screen in demand.all_screens:
        screen_revisions = revisions.get(screen)
        if screen_revisions is None:
            _fail("required_screens", f"no revisions are known for screen {screen!r}")
        keys.add(RenderKey.for_screen(
            screen,
            capabilities.render_profile,
            screen_revisions,
            client_id=capabilities.client_id if screen in scoped else None,
        ))
    return frozenset(keys)


def plan_renders(
    demands: Iterable[tuple[ClientCapabilities, ClientDemand]],
    revisions: Mapping[str, ScreenRevisions],
    *,
    client_specific_screens: Iterable[str] = (),
) -> dict[RenderKey, frozenset[str]]:
    """De-duplicate render work across clients.

    Returns each distinct :class:`RenderKey` mapped to the client IDs that
    need it.  Invalid capabilities or demand raise before anything is
    returned, so no partial plan is ever scheduled.
    """

    scoped = tuple(client_specific_screens)
    plan: dict[RenderKey, set[str]] = {}
    for capabilities, demand in demands:
        for key in demand_render_keys(capabilities, demand, revisions, client_specific_screens=scoped):
            plan.setdefault(key, set()).add(capabilities.client_id)
    return {key: frozenset(clients) for key, clients in plan.items()}


WIRE_MODELS: dict[str, type[WireModel]] = {
    model.WIRE_TYPE: model
    for model in (ClientCapabilities, ClientDemand, ClientStatus, RenderKey, ScreenRevisions)
}


def parse_wire(payload: Any) -> WireModel:
    """Parse any top-level wire document by its ``type``."""

    if not isinstance(payload, Mapping):
        _fail("", "must be a JSON object")
    model = WIRE_MODELS.get(payload.get("type"))  # type: ignore[arg-type]
    if model is None:
        _fail("type", f"unknown document type {payload.get('type')!r}")
    return model.from_wire(payload)


__all__ = [
    "AcceptedRevisions",
    "ClientCapabilities",
    "ClientDemand",
    "ClientStatus",
    "ErrorSummary",
    "HardwareDescription",
    "ModelValidationError",
    "PackageCapabilities",
    "RenderKey",
    "ScreenRevisions",
    "UnsupportedCapabilitiesError",
    "WIRE_MODELS",
    "demand_render_keys",
    "identifier",
    "parse_wire",
    "plan_renders",
    "revision",
    "screen_id",
]
