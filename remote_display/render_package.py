"""Versioned render packages: everything a client needs to play motion itself.

A still PNG artifact is enough for a static screen. A screen that moves
also gets a render package, published through the artifact store next to
its still image, so the server renders once and the client animates without
fetching a frame per step. One JSON document per package::

    {
      "type": "render_package",
      "render_package_schema_version": 1,
      "screen_id": "...", "render_profile": "...",
      "width": 800, "height": 480, "color_mode": "RGB",
      "render_key_digest": "<sha256 of the still image's render key>",
      "classification": "<rendering.screen_classes class>",
      "kind": "scroll" | "ticker" | "animation" | "composite" | "clock",
      "assets": {"a0": {"media_type": "image/png", "width", "height",
                        "color_mode", "sha256", "length", "data": <base64>}},
      "<kind>": { ...kind body, referring to assets by id... }
    }

Kind bodies:

``scroll``     ``canvas`` (full-height image), ``viewport`` [w, h], ``step_px``,
               ``frame_seconds``, ``pause_start_seconds``,
               ``pause_end_seconds``, ``direction`` ("down" or "up").
``ticker``     ``base`` image, ``duration_seconds`` and ``lanes``: each lane
               has logical ``bounds`` [l, t, r, b], a looping ``strip``
               image, ``speed_px_per_second``, ``offset_px`` and
               ``background``. This extends the news ticker sidecar with
               pixels, so the client needs no fonts.
``animation``  Either ``frames`` (at most :data:`MAX_ANIMATION_FRAMES`, each
               ``{"asset", "duration_ms"}``) with ``loops``, or ``slide``: a
               ``sprite`` crossing the screen at ``speed_px_per_second`` on
               row ``y``, ending centred.
``composite``  ``base`` image, ``frame_seconds``, ``duration_seconds`` and
               ``tiles``: logical hit ``bounds``, up to
               :data:`MAX_TILE_FRAMES` ``frames`` and, for interactive
               quads, ``focus_screen`` (whose own artifact the manifest
               lists as an interaction dependency).
``clock``      ``background`` image and ``layout`` (see
               :mod:`rendering.clock_faces`); the client draws the time.

Assets are embedded (base64 PNG) so a package is one immutable object:
atomic to publish, and garbage collected with the still image it belongs
to. Bounds are logical coordinates; physical rotation stays on the client.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
from collections.abc import Mapping
from typing import Any

from PIL import Image, UnidentifiedImageError

from protocol_versions import RENDER_PACKAGE_SCHEMA_VERSION

RENDER_PACKAGE_MEDIA_TYPE = "application/vnd.desk-display.render-package+json"
PACKAGE_TYPE = "render_package"
KINDS = ("scroll", "ticker", "animation", "composite", "clock")
MAX_PACKAGE_BYTES = 16 * 1024 * 1024
MAX_ASSETS = 160
MAX_ASSET_PIXELS = 24_000_000
MAX_ANIMATION_FRAMES = 24
MAX_TILE_FRAMES = 10
MAX_TICKER_LANES = 12
ASSET_MODES = frozenset({"1", "L", "RGB", "RGBA"})


class PackageError(ValueError):
    """A render package is malformed; ``code`` is machine-readable."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _fail(code: str, message: str) -> None:
    raise PackageError(code, message)


class PackageBuilder:
    """Collect de-duplicated image assets and assemble one package."""

    def __init__(self) -> None:
        self.assets: dict[str, dict[str, Any]] = {}
        self._by_sha: dict[str, str] = {}

    def add(self, image: Image.Image) -> str:
        if image.mode not in ASSET_MODES:
            image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=False)
        data = buffer.getvalue()
        sha = hashlib.sha256(data).hexdigest()
        existing = self._by_sha.get(sha)
        if existing is not None:
            return existing
        asset_id = f"a{len(self.assets)}"
        self.assets[asset_id] = {
            "media_type": "image/png",
            "width": image.width,
            "height": image.height,
            "color_mode": image.mode,
            "sha256": sha,
            "length": len(data),
            "data": base64.b64encode(data).decode("ascii"),
        }
        self._by_sha[sha] = asset_id
        return asset_id

    def build(
        self,
        *,
        screen_id: str,
        render_profile: str,
        width: int,
        height: int,
        color_mode: str,
        render_key_digest: str,
        classification: str,
        kind: str,
        body: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "type": PACKAGE_TYPE,
            "render_package_schema_version": RENDER_PACKAGE_SCHEMA_VERSION,
            "screen_id": screen_id,
            "render_profile": render_profile,
            "width": int(width),
            "height": int(height),
            "color_mode": color_mode,
            "render_key_digest": render_key_digest,
            "classification": classification,
            "kind": kind,
            "assets": dict(self.assets),
            kind: dict(body),
        }


def package_bytes(package: Mapping[str, Any]) -> bytes:
    return json.dumps(package, sort_keys=True, separators=(",", ":")).encode("utf-8")


def load_package(data: bytes | str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(data, Mapping):
        return dict(data)
    if isinstance(data, bytes | bytearray) and len(data) > MAX_PACKAGE_BYTES:
        _fail("too_large", f"package exceeds {MAX_PACKAGE_BYTES} bytes")
    try:
        document = json.loads(data)
    except (UnicodeDecodeError, ValueError) as exc:
        raise PackageError("invalid_json", f"package is not valid JSON: {exc}") from None
    if not isinstance(document, dict):
        _fail("invalid_schema", "package must be a JSON object")
    return document


def asset_bytes(package: Mapping[str, Any], asset_id: str) -> bytes:
    asset = (package.get("assets") or {}).get(asset_id)
    if not isinstance(asset, Mapping):
        _fail("missing_asset", f"package has no asset {asset_id!r}")
    try:
        return base64.b64decode(str(asset.get("data", "")), validate=True)
    except (binascii.Error, ValueError):
        raise PackageError("invalid_asset", f"asset {asset_id} is not base64") from None


def asset_image(package: Mapping[str, Any], asset_id: str) -> Image.Image:
    image = Image.open(io.BytesIO(asset_bytes(package, asset_id)))
    image.load()
    return image


# ── Validation ─────────────────────────────────────────────────────────────


def _number(value: Any, path: str, *, minimum: float = 0.0, maximum: float = 1e6) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not minimum <= value <= maximum:
        _fail("invalid_field", f"{path} must be a number from {minimum} to {maximum}")
    return float(value)


def _int(value: Any, path: str, *, minimum: int = 0, maximum: int = 1 << 20) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail("invalid_field", f"{path} must be an integer from {minimum} to {maximum}")
    return value


def _bounds(value: Any, path: str, width: int, height: int) -> None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        _fail("invalid_bounds", f"{path} must be [left, top, right, bottom]")
    if any(type(v) is not int for v in value):
        _fail("invalid_bounds", f"{path} must be integers")
    left, top, right, bottom = value
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        _fail("invalid_bounds", f"{path} must lie inside the {width}x{height} logical frame")


def _color(value: Any, path: str) -> None:
    if not isinstance(value, list | tuple) or len(value) not in (3, 4):
        _fail("invalid_field", f"{path} must be an RGB colour")
    for channel in value:
        _int(channel, path, maximum=255)


def _body(package: Mapping[str, Any], kind: str) -> Mapping[str, Any]:
    body = package.get(kind)
    if not isinstance(body, Mapping):
        _fail("invalid_schema", f"{kind} package needs a {kind!r} object")
    extra = set(KINDS) & set(package) - {kind}
    if extra:
        _fail("invalid_schema", f"{kind} package also carries {sorted(extra)}")
    return body


def validate_package(
    data: bytes | str | Mapping[str, Any],
    *,
    key: Any = None,
    verify_assets: bool = True,
) -> dict[str, Any]:
    """Return the package document or raise :class:`PackageError`.

    With *key* (a :class:`remote_display.models.RenderKey`), the package must
    belong to that screen and profile. With *verify_assets*, every asset is
    decoded and checked against its declared checksum, size and mode.
    """

    from display_profiles import PROFILE_PRESETS

    package = load_package(data)
    if package.get("type") != PACKAGE_TYPE:
        _fail("invalid_schema", "not a render package")
    version = package.get("render_package_schema_version")
    if version != RENDER_PACKAGE_SCHEMA_VERSION:
        _fail("unsupported_version", f"render package schema {version!r} is not supported")
    profile = PROFILE_PRESETS.get(str(package.get("render_profile")))
    if profile is None:
        _fail("invalid_field", "unknown render_profile")
    width, height = package.get("width"), package.get("height")
    if (width, height, package.get("color_mode")) != (profile.width, profile.height, profile.color_mode):
        _fail("invalid_field", "size and colour mode must match the render profile")
    if not isinstance(package.get("screen_id"), str) or not package["screen_id"]:
        _fail("invalid_field", "screen_id is required")
    if key is not None:
        if (package["screen_id"], package["render_profile"]) != (key.screen_id, key.render_profile):
            _fail("wrong_key", "package belongs to a different screen or profile")
        if package.get("render_key_digest") != key.digest:
            _fail("wrong_key", "package was rendered for a different render key")
    kind = package.get("kind")
    if kind not in KINDS:
        _fail("invalid_schema", f"unknown package kind {kind!r}")
    from rendering.screen_classes import CLASSES

    if package.get("classification") not in CLASSES:
        _fail("invalid_schema", "unknown classification")

    assets = package.get("assets")
    if not isinstance(assets, Mapping) or len(assets) > MAX_ASSETS:
        _fail("invalid_schema", f"assets must be an object with at most {MAX_ASSETS} entries")
    for asset_id, asset in assets.items():
        if not isinstance(asset, Mapping) or asset.get("media_type") != "image/png":
            _fail("invalid_asset", f"asset {asset_id} must be a PNG")
        w = _int(asset.get("width"), f"assets.{asset_id}.width", minimum=1, maximum=65535)
        h = _int(asset.get("height"), f"assets.{asset_id}.height", minimum=1, maximum=65535)
        if w * h > MAX_ASSET_PIXELS:
            _fail("invalid_asset", f"asset {asset_id} exceeds {MAX_ASSET_PIXELS} pixels")
        if asset.get("color_mode") not in ASSET_MODES:
            _fail("invalid_asset", f"asset {asset_id} has an unsupported colour mode")
        if verify_assets:
            raw = asset_bytes(package, asset_id)
            if len(raw) != asset.get("length") or hashlib.sha256(raw).hexdigest() != asset.get("sha256"):
                _fail("invalid_asset", f"asset {asset_id} does not match its checksum")
            try:
                with Image.open(io.BytesIO(raw)) as probe:
                    probe.verify()
                with Image.open(io.BytesIO(raw)) as image:
                    if image.format != "PNG" or image.size != (w, h) or image.mode != asset["color_mode"]:
                        _fail("invalid_asset", f"asset {asset_id} does not match its declared size or mode")
            except (UnidentifiedImageError, OSError, SyntaxError) as exc:
                raise PackageError("invalid_asset", f"asset {asset_id} is not a valid PNG: {exc}") from None

    def asset(ref: Any, path: str) -> Mapping[str, Any]:
        if not isinstance(ref, str) or ref not in assets:
            _fail("missing_asset", f"{path} refers to an unknown asset")
        return assets[ref]

    def full_frame(ref: Any, path: str) -> None:
        entry = asset(ref, path)
        if (entry["width"], entry["height"]) != (width, height):
            _fail("invalid_asset", f"{path} must be a full {width}x{height} frame")

    body = _body(package, kind)
    if kind == "scroll":
        canvas = asset(body.get("canvas"), "scroll.canvas")
        if list(body.get("viewport") or []) != [width, height]:
            _fail("invalid_field", "scroll.viewport must be the logical size")
        if canvas["width"] != width or canvas["height"] <= height:
            _fail("invalid_asset", "scroll.canvas must be as wide as the screen and taller than it")
        _int(body.get("step_px"), "scroll.step_px", minimum=1, maximum=height)
        _number(body.get("frame_seconds"), "scroll.frame_seconds", minimum=0.001, maximum=5)
        _number(body.get("pause_start_seconds"), "scroll.pause_start_seconds", maximum=600)
        _number(body.get("pause_end_seconds"), "scroll.pause_end_seconds", maximum=600)
        if body.get("direction") not in ("down", "up"):
            _fail("invalid_field", "scroll.direction must be down or up")
    elif kind == "ticker":
        full_frame(body.get("base"), "ticker.base")
        _number(body.get("duration_seconds"), "ticker.duration_seconds", maximum=3600)
        lanes = body.get("lanes")
        if not isinstance(lanes, list) or not 1 <= len(lanes) <= MAX_TICKER_LANES:
            _fail("invalid_field", f"ticker.lanes must list 1 to {MAX_TICKER_LANES} lanes")
        for index, lane in enumerate(lanes):
            path = f"ticker.lanes[{index}]"
            if not isinstance(lane, Mapping):
                _fail("invalid_field", f"{path} must be an object")
            _bounds(lane.get("bounds"), f"{path}.bounds", width, height)
            strip = asset(lane.get("strip"), f"{path}.strip")
            if strip["height"] != lane["bounds"][3] - lane["bounds"][1]:
                _fail("invalid_asset", f"{path}.strip must be as tall as its lane")
            _number(lane.get("speed_px_per_second"), f"{path}.speed_px_per_second", maximum=10000)
            _number(lane.get("offset_px"), f"{path}.offset_px", maximum=float(strip["width"]))
            _color(lane.get("background"), f"{path}.background")
    elif kind == "animation":
        frames, slide = body.get("frames"), body.get("slide")
        if (frames is None) == (slide is None):
            _fail("invalid_schema", "animation needs exactly one of frames or slide")
        if frames is not None:
            if not isinstance(frames, list) or not 2 <= len(frames) <= MAX_ANIMATION_FRAMES:
                _fail("invalid_field", f"animation.frames must list 2 to {MAX_ANIMATION_FRAMES} frames")
            for index, frame in enumerate(frames):
                if not isinstance(frame, Mapping):
                    _fail("invalid_field", f"animation.frames[{index}] must be an object")
                full_frame(frame.get("asset"), f"animation.frames[{index}].asset")
                _int(frame.get("duration_ms"), f"animation.frames[{index}].duration_ms", maximum=60_000)
            _int(body.get("loops"), "animation.loops", minimum=1, maximum=100)
        else:
            if not isinstance(slide, Mapping):
                _fail("invalid_field", "animation.slide must be an object")
            asset(slide.get("sprite"), "animation.slide.sprite")
            _int(slide.get("y"), "animation.slide.y", minimum=-height, maximum=height)
            _number(slide.get("speed_px_per_second"), "animation.slide.speed_px_per_second",
                    minimum=1, maximum=100_000)
            _color(slide.get("background"), "animation.slide.background")
    elif kind == "composite":
        full_frame(body.get("base"), "composite.base")
        _number(body.get("frame_seconds"), "composite.frame_seconds", minimum=0.001, maximum=5)
        _number(body.get("duration_seconds"), "composite.duration_seconds", maximum=3600)
        tiles = body.get("tiles")
        if not isinstance(tiles, list) or not 1 <= len(tiles) <= 4:
            _fail("invalid_field", "composite.tiles must list 1 to 4 tiles")
        for index, tile in enumerate(tiles):
            path = f"composite.tiles[{index}]"
            if not isinstance(tile, Mapping):
                _fail("invalid_field", f"{path} must be an object")
            _bounds(tile.get("bounds"), f"{path}.bounds", width, height)
            frames = tile.get("frames")
            if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_TILE_FRAMES:
                _fail("invalid_field", f"{path}.frames must list 1 to {MAX_TILE_FRAMES} frames")
            left, top, right, bottom = tile["bounds"]
            for ref in frames:
                entry = asset(ref, f"{path}.frames")
                if (entry["width"], entry["height"]) != (right - left, bottom - top):
                    _fail("invalid_asset", f"{path} frames must match the tile bounds")
            focus = tile.get("focus_screen")
            if focus is not None and (not isinstance(focus, str) or not focus):
                _fail("invalid_field", f"{path}.focus_screen must be a screen id or null")
    elif kind == "clock":
        from rendering.clock_faces import CLOCK_FACES, LAYOUT_KEYS

        full_frame(body.get("background"), "clock.background")
        layout = body.get("layout")
        if not isinstance(layout, Mapping) or set(layout) != LAYOUT_KEYS:
            _fail("invalid_field", f"clock.layout must have exactly {sorted(LAYOUT_KEYS)}")
        if layout["face"] not in CLOCK_FACES.values() or layout["time_format"] not in ("12", "24"):
            _fail("invalid_field", "clock.layout has an unknown face or time format")
        if not isinstance(layout["time_zone"], str) or not isinstance(layout["show_ip"], bool):
            _fail("invalid_field", "clock.layout time_zone and show_ip are malformed")
        _color(layout["background_color"], "clock.layout.background_color")
    return package


__all__ = [
    "KINDS",
    "MAX_ANIMATION_FRAMES",
    "MAX_PACKAGE_BYTES",
    "MAX_TILE_FRAMES",
    "PackageBuilder",
    "PackageError",
    "RENDER_PACKAGE_MEDIA_TYPE",
    "asset_bytes",
    "asset_image",
    "load_package",
    "package_bytes",
    "validate_package",
]
