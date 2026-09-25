"""Versioned client manifests built from the artifact store.

A manifest tells one client exactly what to download and play for its
assigned playlist.  Every artifact entry carries an immutable URL (content
addressed by SHA-256), its length, media type and dimensions, when it was
generated and when it should be refreshed, and whether it is stale or a
fallback after a failed render.  ``manifest_revision`` is a hash of the
manifest's content (excluding ``generated_at``), so it changes exactly when
something the client should act on changes.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from display_profiles import PROFILE_PRESETS
from protocol import build_manifest
from remote_display.artifact_store import ArtifactStore

MANIFEST_TYPE = "client_manifest"


def _iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def manifest_revision(manifest: Mapping[str, Any]) -> str:
    basis = {k: v for k, v in manifest.items() if k not in {"generated_at", "manifest_revision"}}
    digest = hashlib.sha256(json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"m-{digest[:20]}"


def _remote_class(screen: str) -> str | None:
    from rendering.screen_classes import CLASSIFICATIONS

    entry = CLASSIFICATIONS.get(screen)
    return None if entry is None else entry.kind


def _package_entry(package: Any, artifact_url: Callable[[str], str]) -> dict[str, Any] | None:
    """Where to download the render package that belongs to an artifact."""

    if not isinstance(package, Mapping) or not package.get("sha256"):
        return None
    from remote_display.artifact_store import object_name

    return {
        "url": artifact_url(object_name(str(package["sha256"]), str(package["media_type"]))),
        **{k: package.get(k) for k in ("sha256", "length", "media_type", "kind", "classification",
                                       "render_package_schema_version")},
    }


def build_client_manifest(
    store: ArtifactStore,
    *,
    client_id: str,
    display_profile: str,
    requested_screens: Iterable[str],
    interactive_screens: Iterable[str] = (),
    client_specific_screens: Iterable[str] = (),
    assignment: Mapping[str, Any],
    configuration: Mapping[str, Any],
    artifact_url: Callable[[str], str],
    now: float,
) -> dict[str, Any]:
    """Build the manifest for one client.

    ``assignment`` holds ``assignment_state`` and ``assigned_playlist`` as the
    registration response reports them; ``configuration`` holds server-side
    configuration revisions (for example the playlist store revision).
    """

    profile = PROFILE_PRESETS[display_profile]
    requested = sorted(set(requested_screens))
    interactive = sorted(set(interactive_screens) - set(requested))
    scoped = set(client_specific_screens)
    artifacts: list[dict[str, Any]] = []
    missing: list[str] = []
    deadlines: list[float] = []
    states: set[str] = set()
    for role, screens in (("requested", requested), ("interactive_dependency", interactive)):
        for screen in screens:
            resolved = store.resolve(screen, profile.profile_id, client_id if screen in scoped else None)
            states.add(resolved.state)
            if resolved.record is None:
                missing.append(screen)
                if resolved.failure:
                    artifacts.append({
                        "screen_id": screen,
                        "role": role,
                        "state": resolved.state,
                        "stale": True,
                        "failure": dict(resolved.failure),
                    })
                continue
            record = resolved.record
            deadlines.append(record.refresh_deadline)
            metadata = dict(record.metadata)
            artifacts.append({
                "screen_id": screen,
                "role": role,
                "artifact_type": record.artifact_type,
                "url": artifact_url(record.name),
                "sha256": record.sha256,
                "length": record.length,
                "media_type": record.media_type,
                "width": record.width,
                "height": record.height,
                "color_mode": record.color_mode,
                "render_key_digest": record.render_key_digest,
                "generated_at": _iso(record.generated_at),
                "refresh_deadline": _iso(record.refresh_deadline),
                "state": resolved.state,
                "stale": resolved.stale,
                "failure": None if resolved.failure is None else dict(resolved.failure),
                "animation": metadata.get("animation"),
                "required_capabilities": sorted(metadata.get("required_capabilities") or []),
                "remote_class": _remote_class(screen),
                "package": _package_entry(metadata.get("package"), artifact_url),
            })
    requested_missing = [s for s in missing if s in requested]
    if requested_missing:
        overall = "incomplete"
    elif states - {"fresh"}:
        overall = "stale"
    else:
        overall = "fresh"
    manifest = build_manifest(
        type=MANIFEST_TYPE,
        client_id=client_id,
        display_profile=profile.profile_id,
        logical_width=profile.width,
        logical_height=profile.height,
        color_mode=profile.color_mode,
        **assignment,
        configuration=dict(configuration),
        requested_screens=requested,
        interactive_dependency_screens=interactive,
        artifacts=artifacts,
        missing_screens=missing,
        cache_complete=not requested_missing,
        state=overall,
        refresh_deadline=_iso(min(deadlines)) if deadlines else None,
    )
    manifest["manifest_revision"] = manifest_revision(manifest)
    manifest["generated_at"] = _iso(now)
    return manifest


def referenced_hashes(manifest: Mapping[str, Any]) -> set[str]:
    hashes = {a["sha256"] for a in manifest.get("artifacts", []) if a.get("sha256")}
    hashes.update(a["package"]["sha256"] for a in manifest.get("artifacts", [])
                  if isinstance(a.get("package"), Mapping) and a["package"].get("sha256"))
    return hashes


__all__ = ["MANIFEST_TYPE", "build_client_manifest", "manifest_revision", "referenced_hashes"]
