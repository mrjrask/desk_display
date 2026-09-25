"""Tests for the immutable, atomically published artifact store."""
from __future__ import annotations

import hashlib
import io
import json
import os
import threading

import pytest
from PIL import Image

from display_profiles import PROFILE_PRESETS
from remote_display import artifact_store as store_module
from remote_display.artifact_store import ArtifactStore, InvalidArtifactError
from remote_display.models import RenderKey, ScreenRevisions

GRACE = 3600


class Clock:
    now = 1_800_000_000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(tmp_path, clock):
    return ArtifactStore(tmp_path / "artifacts", grace_seconds=GRACE, clock=clock)


def key(screen="date", profile="hyperpixel4", data="d1", client_scope=None):
    return RenderKey.for_screen(screen, profile, ScreenRevisions("s1", data, "r1"), client_id=client_scope)


def png(profile="hyperpixel4", color=0, *, size=None, mode=None, fmt="PNG"):
    preset = PROFILE_PRESETS[profile]
    image = Image.new(mode or preset.color_mode, size or (preset.width, preset.height), color)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def objects(store):
    return sorted(p.name for p in (store.root / "objects").glob("*/*"))


# ── Publication and immutability ────────────────────────────────────────────


def test_publish_is_content_addressed_and_resolves_fresh(store):
    data = png()
    record = store.publish(key(), data, refresh_seconds=60)
    assert record.sha256 == hashlib.sha256(data).hexdigest() and record.length == len(data)
    assert (record.width, record.height, record.color_mode) == (800, 480, "RGB")
    assert record.name == f"{record.sha256}.png"
    path, media_type = store.open_object(record.name)
    assert path.read_bytes() == data and media_type == "image/png"
    resolved = store.resolve("date", "hyperpixel4")
    assert resolved.state == "fresh" and resolved.record == record and not resolved.stale


def test_published_objects_never_change(store):
    first = store.publish(key(data="d1"), png(color=1))
    path, _ = store.open_object(first.name)
    before = path.read_bytes()
    store.publish(key(data="d2"), png(color=2))
    store.publish(key(data="d3"), png(color=1))  # same bytes as the first: reuses the object
    assert path.read_bytes() == before
    assert store.resolve("date", "hyperpixel4").record.sha256 == first.sha256


def test_lineages_are_separate_per_profile_and_client_scope(store):
    shared = store.publish(key(), png(color=1))
    scoped = store.publish(key(client_scope="office"), png(color=2))
    oled = store.publish(key(profile="waveshare_oled_128x64"), png("waveshare_oled_128x64"))
    assert store.resolve("date", "hyperpixel4").record == shared
    assert store.resolve("date", "hyperpixel4", "office").record == scoped
    assert store.resolve("date", "waveshare_oled_128x64").record == oled
    assert store.resolve("weather1", "hyperpixel4").state == "missing"


def test_render_package_media_type(store):
    from remote_display.render_package import PackageBuilder, package_bytes

    preset = PROFILE_PRESETS["hyperpixel4"]
    builder = PackageBuilder()
    background = builder.add(Image.new("RGB", (preset.width, preset.height)))
    layout = {"face": "date", "time_zone": "UTC", "time_format": "12", "show_ip": False,
              "background_color": [0, 0, 0]}
    package = package_bytes(builder.build(
        screen_id="date", render_profile="hyperpixel4", width=preset.width, height=preset.height,
        color_mode=preset.color_mode, render_key_digest=key().digest, classification="client_timed",
        kind="clock", body={"background": background, "layout": layout}))
    record = store.publish(key(), package, media_type="application/vnd.desk-display.render-package+json")
    assert record.name.endswith(".json") and record.artifact_type == "render_package"
    with pytest.raises(InvalidArtifactError) as info:
        store.publish(key(data="d2"), b'{"frames": []}', media_type="application/vnd.desk-display.render-package+json")
    assert info.value.code == "invalid_schema"


# ── Invalid output and render failures ──────────────────────────────────────


@pytest.mark.parametrize(
    "data, kwargs, code",
    [
        (b"", {}, "empty_output"),
        (b"\x89PNG not really", {}, "invalid_media"),
        (png(fmt="JPEG"), {}, "invalid_media"),
        (png(size=(100, 100)), {}, "wrong_dimensions"),
        (png(mode="L"), {}, "wrong_color_mode"),
        (png(), {"expected_sha256": "0" * 64}, "checksum_mismatch"),
        (png(), {"metadata": {"error_image": True}}, "error_image"),
        (png(), {"media_type": "image/gif"}, "unsupported_media_type"),
        (png(), {"metadata": {"bad": object()}}, "invalid_metadata"),
    ],
)
def test_invalid_output_never_replaces_good_output(store, data, kwargs, code):
    good = store.publish(key(data="d1"), png(color=5))
    before = objects(store)
    with pytest.raises(InvalidArtifactError) as info:
        store.publish(key(data="d2"), data, **kwargs)
    assert info.value.code == code
    assert objects(store) == before
    resolved = store.resolve("date", "hyperpixel4")
    assert resolved.record == good
    assert resolved.state == "fallback" and resolved.stale
    assert resolved.failure["code"] == code


def test_render_failure_serves_last_known_good_until_next_success(store):
    good = store.publish(key(data="d1"), png(color=5))
    store.record_failure(key(data="d2"), "timeout", "render took too long")
    store.record_failure(key(data="d2"), "timeout", "render took too long")
    resolved = store.resolve("date", "hyperpixel4")
    assert resolved.record == good and resolved.state == "fallback"
    assert resolved.failure["consecutive"] == 2
    fixed = store.publish(key(data="d2"), png(color=6))
    assert store.resolve("date", "hyperpixel4") == store_module.ResolvedArtifact("fresh", fixed, None)


def test_failure_before_any_good_output_is_missing(store):
    store.record_failure(key(), "timeout", "render took too long")
    resolved = store.resolve("date", "hyperpixel4")
    assert resolved.state == "missing" and resolved.record is None and resolved.failure["code"] == "timeout"


def test_staleness_from_deadline_and_pending_render(store, clock):
    store.publish(key(data="d1"), png(), refresh_seconds=60)
    clock.advance(61)
    assert store.resolve("date", "hyperpixel4").state == "stale"
    store.publish(key(data="d1"), png(), refresh_seconds=60)
    assert store.resolve("date", "hyperpixel4").state == "fresh"
    store.mark_pending(key(data="d1"))  # same key as current: nothing newer pending
    assert store.resolve("date", "hyperpixel4").state == "fresh"
    store.mark_pending(key(data="d2"))
    assert store.resolve("date", "hyperpixel4").state == "stale"


# ── Interrupted publication ─────────────────────────────────────────────────


def test_interrupted_object_rename_leaves_no_partial_object(store, monkeypatch):
    good = store.publish(key(data="d1"), png(color=1))
    before = objects(store)

    def crash(src, dst):
        raise OSError("power lost")

    monkeypatch.setattr(store_module.os, "replace", crash)
    with pytest.raises(OSError):
        store.publish(key(data="d2"), png(color=2))
    monkeypatch.undo()
    assert objects(store) == before
    assert not list((store.root / "staging").iterdir())
    assert store.resolve("date", "hyperpixel4").record == good


def test_interrupted_lineage_update_keeps_previous_record(store, monkeypatch, clock):
    good = store.publish(key(data="d1"), png(color=1))
    real_replace = os.replace
    calls = []

    def crash_on_lineage(src, dst):
        calls.append(dst)
        if str(dst).endswith(".json"):
            raise OSError("power lost")
        return real_replace(src, dst)

    monkeypatch.setattr(store_module.os, "replace", crash_on_lineage)
    with pytest.raises(OSError):
        store.publish(key(data="d2"), png(color=2))
    monkeypatch.undo()
    assert store.resolve("date", "hyperpixel4").record == good
    orphan = [name for name in objects(store) if not name.startswith(good.sha256)]
    assert len(orphan) == 1
    # The orphan was never referenced; it goes once the grace period passes.
    clock.advance(GRACE + 1)
    os.utime(store.open_object(orphan[0])[0], (clock.now - GRACE - 1,) * 2)
    assert store.collect_garbage() == orphan
    assert store.open_object(good.name) is not None


def test_stale_staging_files_are_cleaned(store, clock):
    store.publish(key(), png())
    leftover = store.root / "staging" / "abandoned.part"
    leftover.write_bytes(b"half")
    os.utime(leftover, (clock.now - 7200,) * 2)
    fresh = store.root / "staging" / "in-progress.part"
    fresh.write_bytes(b"half")
    os.utime(fresh, (clock.now,) * 2)
    store.collect_garbage()
    assert not leftover.exists() and fresh.exists()


# ── Previous revisions and concurrent readers ───────────────────────────────


def test_previous_revisions_are_retained_and_retrievable(store):
    records = [store.publish(key(data=f"d{i}"), png(color=i)) for i in range(5)]
    previous = store.previous("date", "hyperpixel4")
    assert [p.sha256 for p in previous] == [r.sha256 for r in reversed(records[1:4])]
    for record in records[1:]:
        path, _ = store.open_object(record.name)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record.sha256


def test_concurrent_readers_only_see_complete_objects(store):
    errors = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            resolved = store.resolve("date", "hyperpixel4")
            if resolved.record is None:
                continue
            found = store.open_object(resolved.record.name)
            if found is None:
                continue  # collected between resolve and open; a client re-fetches the manifest
            data = found[0].read_bytes()
            if hashlib.sha256(data).hexdigest() != resolved.record.sha256:
                errors.append(resolved.record.name)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for thread in threads:
        thread.start()
    try:
        for i in range(30):
            store.publish(key(data=f"d{i}"), png(color=i))
    finally:
        stop.set()
        for thread in threads:
            thread.join()
    assert not errors


def test_open_reader_survives_garbage_collection(store, clock):
    record = store.publish(key(data="d1"), png(color=1))
    store.previous_revisions = 0
    store.publish(key(data="d2"), png(color=2))
    path, _ = store.open_object(record.name)
    with open(path, "rb") as handle:
        clock.advance(GRACE + 1)
        assert store.collect_garbage() == [record.name]
        assert hashlib.sha256(handle.read()).hexdigest() == record.sha256


# ── References and garbage collection ───────────────────────────────────────


def test_referenced_objects_survive_grace_period(store, clock):
    store.previous_revisions = 0
    held = store.publish(key(data="d1"), png(color=1))
    store.reference("office", [held.sha256])
    store.publish(key(data="d2"), png(color=2))  # evicted from the lineage, still referenced
    clock.advance(GRACE * 10)
    assert store.collect_garbage() == []
    assert store.referenced_by("office") == {held.sha256}


def test_previous_manifest_stays_referenced(store, clock):
    store.previous_revisions = 0
    a = store.publish(key(data="d1"), png(color=1))
    store.reference("office", [a.sha256])
    b = store.publish(key(data="d2"), png(color=2))
    store.reference("office", [b.sha256])
    clock.advance(GRACE + 1)
    assert store.collect_garbage() == []  # a is still the office's previous manifest
    c = store.publish(key(data="d3"), png(color=3))
    store.reference("office", [c.sha256])
    assert store.collect_garbage() == []  # a was released just now; its grace starts here
    clock.advance(GRACE + 1)
    assert store.collect_garbage() == [a.name]
    assert store.referenced_by("office") == {b.sha256, c.sha256}


def test_shared_objects_need_every_holder_to_release(store, clock):
    store.previous_revisions = 0
    shared = store.publish(key(data="d1"), png(color=1))
    store.reference("office", [shared.sha256])
    store.reference("kitchen", [shared.sha256])
    store.publish(key(data="d2"), png(color=2))
    store.release("office")
    clock.advance(GRACE + 1)
    assert store.collect_garbage() == []
    assert store.prune_holders(keep=set()) == ["kitchen"]
    assert store.collect_garbage() == []
    clock.advance(GRACE + 1)
    assert store.collect_garbage() == [shared.name]


def test_lineage_output_is_never_collected(store, clock):
    current = store.publish(key(), png())
    clock.advance(GRACE * 100)
    assert store.collect_garbage() == []
    assert store.resolve("date", "hyperpixel4").record == current


def test_unreadable_reference_file_is_treated_as_empty(store):
    record = store.publish(key(), png())
    (store.root / "references.json").write_text("{not json", encoding="utf-8")
    store.reference("office", [record.sha256])
    assert store.referenced_by("office") == {record.sha256}


def test_open_object_rejects_bad_names(store):
    record = store.publish(key(), png())
    for name in ("../references.json", record.sha256, record.sha256 + ".txt", "x.png", "", record.name.upper()):
        assert store.open_object(name) is None, name
