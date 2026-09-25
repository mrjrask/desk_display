"""Tests for remote display protocol models and render-key de-duplication."""
from __future__ import annotations

import json
from dataclasses import fields, replace

import pytest

from display_profiles import PROFILE_PRESETS
from remote_display import models as m
from remote_display.models import (
    AcceptedRevisions,
    ClientCapabilities,
    ClientDemand,
    ClientStatus,
    ErrorSummary,
    HardwareDescription,
    ModelValidationError,
    PackageCapabilities,
    RenderKey,
    ScreenRevisions,
    UnsupportedCapabilitiesError,
)

REVS = ScreenRevisions(style_revision="style-7", data_revision="data-42", renderer_revision="r1")
REVISIONS = {
    screen: REVS
    for screen in ("date", "weather1", "news headlines", "cubs next", "inside", "NL Overview")
}


def caps(client_id="office", profile="hyperpixel4", **overrides):
    preset = PROFILE_PRESETS[profile]
    values = dict(
        protocol_version=1,
        client_software_version="0.1",
        client_id=client_id,
        display_profile=profile,
        logical_width=preset.width,
        logical_height=preset.height,
        image_formats=("PNG",),
        color_modes=(preset.color_mode,),
        render_package_versions=(1,),
    )
    values.update(overrides)
    return ClientCapabilities(**values)


def demand(client_id="office", screens=("date", "weather1"), **overrides):
    values = dict(
        client_id=client_id,
        playlist_revision="pl-3",
        required_screens=screens,
        package_capabilities=PackageCapabilities(render_package_versions=(1,), image_formats=("PNG",)),
        sync_interval_seconds=30,
    )
    values.update(overrides)
    return ClientDemand(**values)


def status(**overrides):
    values = dict(
        client_id="office",
        playback_state="playing",
        accepted_revisions=AcceptedRevisions(manifest_revision="m-1", playlist_revision="pl-3"),
        current_screen="weather1",
        current_playlist="default",
        last_sync_age_seconds=12.5,
        cache_age_seconds=300,
        physical_rotation=180,
        recent_errors=(ErrorSummary(code="fetch_failed", message="timeout", count=3,
                                    last_seen_age_seconds=40),),
    )
    values.update(overrides)
    return ClientStatus(**values)


# ── Round trips and wire representation ─────────────────────────────────────


@pytest.mark.parametrize(
    "model",
    [
        caps(),
        caps(has_touch=True, supports_animation=True, buttons=("A", "B"),
             hardware=HardwareDescription(model="Raspberry Pi 4", panel="HyperPixel 4", driver="kms")),
        caps(profile="waveshare_oled_128x64"),
        demand(alternate_screens=("inside",), touch_targets=("news headlines",)),
        status(),
        status(current_screen=None, current_playlist=None, last_sync_age_seconds=None,
               cache_age_seconds=None, recent_errors=(), accepted_revisions=AcceptedRevisions()),
        RenderKey.for_screen("date", "hyperpixel4", REVS),
        RenderKey.for_screen("inside", "hyperpixel4", REVS, client_id="office"),
        REVS,
    ],
)
def test_round_trip(model):
    wire = model.to_wire()
    assert wire["type"] == model.WIRE_TYPE
    assert wire["version"] == 1
    assert json.loads(json.dumps(wire)) == wire  # JSON-safe
    assert type(model).from_wire(wire) == model
    assert type(model).from_json(model.to_json()) == model
    assert m.parse_wire(wire) == model


def test_wire_fields_are_explicit():
    wire = caps().to_wire()
    assert set(wire) == {"type", "version"} | {f.name for f in fields(ClientCapabilities)}
    key_wire = RenderKey.for_screen("date", "hyperpixel4", REVS).to_wire()
    assert key_wire == {
        "type": "render_key", "version": 1, "screen_id": "date", "render_profile": "hyperpixel4",
        "width": 800, "height": 480, "color_mode": "RGB", "style_revision": "style-7",
        "data_revision": "data-42", "renderer_revision": "r1", "client_scope": None,
    }


def test_optional_fields_may_be_omitted_on_the_wire():
    wire = caps().to_wire()
    for name in ("supports_animation", "has_touch", "buttons", "hardware"):
        wire.pop(name)
    assert ClientCapabilities.from_wire(wire) == caps()


# ── Malformed input ─────────────────────────────────────────────────────────


def _mutate(wire, **changes):
    result = dict(wire)
    for key, value in changes.items():
        if value is _DROP:
            result.pop(key)
        else:
            result[key] = value
    return result


_DROP = object()


@pytest.mark.parametrize(
    "changes, path",
    [
        ({"type": "client_demand"}, "type"),
        ({"version": 2}, "version"),
        ({"version": "1"}, "version"),
        ({"client_id": _DROP}, "client_id"),
        ({"surprise": 1}, "surprise"),
        ({"client_id": "../etc/passwd"}, "client_id"),
        ({"client_id": "x" * 65}, "client_id"),
        ({"client_id": 7}, "client_id"),
        ({"protocol_version": True}, "protocol_version"),
        ({"protocol_version": 1.0}, "protocol_version"),
        ({"display_profile": "crt"}, "display_profile"),
        ({"logical_width": 480, "logical_height": 800}, "logical_width"),
        ({"logical_width": -1}, "logical_width"),
        ({"image_formats": ["GIF"]}, "image_formats[0]"),
        ({"image_formats": "PNG"}, "image_formats"),
        ({"image_formats": ["PNG"] * 17}, "image_formats"),
        ({"color_modes": ["1"]}, "color_modes"),
        ({"render_package_versions": []}, "render_package_versions"),
        ({"render_package_versions": [0]}, "render_package_versions[0]"),
        ({"supports_animation": 1}, "supports_animation"),
        ({"buttons": ["A B"]}, "buttons[0]"),
        ({"hardware": {"model": "x" * 201}}, "hardware.model"),
        ({"hardware": {"cpu": "arm"}}, "hardware.cpu"),
        ({"hardware": "pi"}, "hardware"),
        ({"client_software_version": "1.0; rm -rf"}, "client_software_version"),
    ],
)
def test_malformed_capabilities_are_rejected(changes, path):
    with pytest.raises(ModelValidationError) as info:
        ClientCapabilities.from_wire(_mutate(caps().to_wire(), **changes))
    assert info.value.path == path


@pytest.mark.parametrize(
    "changes, path",
    [
        ({"required_screens": ["not a screen"]}, "required_screens[0]"),
        ({"required_screens": ["cubs next 2"]}, "required_screens[0]"),  # retired
        ({"required_screens": ["date"] * 257}, "required_screens"),
        ({"playlist_revision": "has space"}, "playlist_revision"),
        ({"sync_interval_seconds": 1}, "sync_interval_seconds"),
        ({"sync_interval_seconds": 10**9}, "sync_interval_seconds"),
        ({"package_capabilities": None}, "package_capabilities"),
        ({"package_capabilities": {"render_package_versions": [1], "image_formats": []}},
         "package_capabilities.image_formats"),
        ({"package_capabilities": {"render_package_versions": [1], "image_formats": ["PNG"], "x": 1}},
         "package_capabilities.x"),
        ({"touch_targets": [None]}, "touch_targets[0]"),
    ],
)
def test_malformed_demand_is_rejected(changes, path):
    with pytest.raises(ModelValidationError) as info:
        ClientDemand.from_wire(_mutate(demand().to_wire(), **changes))
    assert info.value.path == path


@pytest.mark.parametrize(
    "changes, path",
    [
        ({"playback_state": "dancing"}, "playback_state"),
        ({"physical_rotation": 45}, "physical_rotation"),
        ({"current_screen": "nope"}, "current_screen"),
        ({"last_sync_age_seconds": float("nan")}, "last_sync_age_seconds"),
        ({"cache_age_seconds": -1}, "cache_age_seconds"),
        ({"accepted_revisions": {"manifest_revision": "a b"}}, "accepted_revisions.manifest_revision"),
        ({"recent_errors": [{"code": "x", "message": "m"}] * 11}, "recent_errors"),
        ({"recent_errors": [{"code": "x", "message": "m", "count": 0}]}, "recent_errors[0].count"),
        ({"recent_errors": ["boom"]}, "recent_errors[0]"),
    ],
)
def test_malformed_status_is_rejected(changes, path):
    with pytest.raises(ModelValidationError) as info:
        ClientStatus.from_wire(_mutate(status().to_wire(), **changes))
    assert info.value.path == path


def test_malformed_documents():
    with pytest.raises(ModelValidationError, match="invalid JSON"):
        ClientStatus.from_json("{not json")
    with pytest.raises(ModelValidationError, match="exceeds"):
        ClientStatus.from_json(" " * (m.MAX_WIRE_BYTES + 1))
    with pytest.raises(ModelValidationError, match="JSON object"):
        ClientStatus.from_wire(["client_status"])
    with pytest.raises(ModelValidationError, match="unknown document type"):
        m.parse_wire({"type": "manifest", "version": 1})


def test_render_key_rejects_client_specific_physical_inputs():
    with pytest.raises(ModelValidationError, match="canonical size"):
        RenderKey(screen_id="date", render_profile="hyperpixel4", width=480, height=800,
                  color_mode="RGB", style_revision="a", data_revision="b", renderer_revision="c")
    with pytest.raises(ModelValidationError, match="renders"):
        RenderKey(screen_id="date", render_profile="waveshare_oled_128x64", width=128, height=64,
                  color_mode="RGB", style_revision="a", data_revision="b", renderer_revision="c")


def test_status_messages_are_single_line_and_redacted(monkeypatch):
    monkeypatch.setenv("DESK_DISPLAY_CLIENT_TOKEN", "client-token-abcdef123")
    summary = ErrorSummary(code="auth", message="bad\ntoken client-token-abcdef123")
    assert summary.message == "bad token [redacted]"


# ── Canonicalization, equality and hashing ──────────────────────────────────


def test_legacy_screen_ids_are_canonicalized():
    assert demand(screens=("time", "sensors")).required_screens == ("inside", "nixie")
    assert RenderKey.for_screen("time", "hyperpixel4", REVS).screen_id == "nixie"


def test_demand_ignores_playlist_order_and_duplicates():
    a = demand(screens=("date", "weather1", "cubs next"))
    b = demand(screens=("cubs next", "weather1", "date", "date"))
    assert a == b and hash(a) == hash(b)
    assert a.to_json() == b.to_json()


def test_alternates_exclude_required_screens():
    d = demand(screens=("date",), alternate_screens=("date", "inside"))
    assert d.alternate_screens == ("inside",)
    assert d.all_screens == ("date", "inside")


def test_capabilities_equality_and_hash_ignore_list_order():
    a = caps(image_formats=("WEBP", "png"), buttons=("B", "A"), render_package_versions=(2, 1))
    b = caps(image_formats=("PNG", "WEBP"), buttons=("A", "B", "A"), render_package_versions=(1, 2))
    assert a == b and hash(a) == hash(b)
    assert a != caps(client_id="kitchen")


def test_models_are_immutable():
    key = RenderKey.for_screen("date", "hyperpixel4", REVS)
    with pytest.raises(AttributeError):
        key.screen_id = "weather1"  # type: ignore[misc]
    # replace() re-validates.
    with pytest.raises(ModelValidationError):
        replace(key, screen_id="bogus")


def test_render_key_hash_and_digest_are_stable():
    a = RenderKey.for_screen("date", "hyperpixel4", REVS)
    b = RenderKey.for_screen("date", PROFILE_PRESETS["hyperpixel4"], REVS)
    assert a == b and hash(a) == hash(b) and a.digest == b.digest
    assert len({a, b}) == 1
    assert a.digest != RenderKey.for_screen("date", "hyperpixel4", replace(REVS, data_revision="data-43")).digest
    assert a.digest != RenderKey.for_screen("date", "hdmi_1080p", REVS).digest


def test_render_key_has_no_client_placement_fields():
    names = {f.name for f in fields(RenderKey)}
    forbidden = {"client_id", "playlist_revision", "playlist_position", "position",
                 "sequence", "physical_rotation", "rotation"}
    assert not names & forbidden


# ── Render de-duplication ───────────────────────────────────────────────────


def test_equivalent_demand_shares_artifacts():
    office = (caps("office"), demand("office", screens=("date", "weather1")))
    den = (caps("den", buttons=("A",)), demand("den", screens=("weather1", "date"),
                                               playlist_revision="other-rev", sync_interval_seconds=90))
    plan = m.plan_renders([office, den], REVISIONS)
    assert len(plan) == 2
    assert all(clients == {"office", "den"} for clients in plan.values())


def test_rotation_and_playlist_position_do_not_fragment_the_cache():
    upright = m.demand_render_keys(caps("a"), demand("a"), REVISIONS)
    for rotation in (0, 90, 180, 270):
        # Physical rotation is only reported in status; it never reaches a key.
        status(client_id="b", physical_rotation=rotation)
        rotated = m.demand_render_keys(caps("b"), demand("b", screens=("weather1", "date")), REVISIONS)
        assert rotated == upright


def test_different_profiles_render_separately():
    plan = m.plan_renders(
        [(caps("a"), demand("a")), (caps("b", profile="display_hat_mini"), demand("b"))], REVISIONS
    )
    assert len(plan) == 4
    assert {key.render_profile for key in plan} == {"hyperpixel4", "display_hat_mini"}


def test_client_specific_output_is_scoped_intentionally():
    plan = m.plan_renders(
        [(caps("a"), demand("a", screens=("date", "inside"))),
         (caps("b"), demand("b", screens=("date", "inside")))],
        REVISIONS,
        client_specific_screens=("inside",),
    )
    shared = [k for k in plan if k.screen_id == "date"]
    scoped = [k for k in plan if k.screen_id == "inside"]
    assert len(shared) == 1 and plan[shared[0]] == {"a", "b"}
    assert {k.client_scope for k in scoped} == {"a", "b"}


def test_touch_targets_and_alternates_are_rendered():
    keys = m.demand_render_keys(
        caps(), demand(screens=("date",), alternate_screens=("inside",), touch_targets=("news headlines",)),
        REVISIONS,
    )
    assert {k.screen_id for k in keys} == {"date", "inside", "news headlines"}


# ── Invalid capabilities fail before scheduling ─────────────────────────────


def test_unsupported_protocol_fails_before_scheduling():
    with pytest.raises(UnsupportedCapabilitiesError, match="protocol"):
        m.plan_renders([(caps(), demand()), (caps("b", protocol_version=99), demand("b"))], REVISIONS)


def test_unsupported_package_version_fails_before_scheduling():
    with pytest.raises(UnsupportedCapabilitiesError, match="render package"):
        m.demand_render_keys(caps(render_package_versions=(7,)), demand(), REVISIONS)


def test_demand_must_match_capabilities():
    with pytest.raises(ModelValidationError, match="different clients"):
        m.demand_render_keys(caps("a"), demand("b"), REVISIONS)
    greedy = demand(package_capabilities=PackageCapabilities(render_package_versions=(1, 2), image_formats=("PNG",)))
    with pytest.raises(UnsupportedCapabilitiesError, match="did not advertise"):
        m.demand_render_keys(caps(), greedy, REVISIONS)
    oled = caps(profile="waveshare_oled_128x64")
    jpeg_only = demand(package_capabilities=PackageCapabilities(render_package_versions=(1,), image_formats=("JPEG",)))
    with pytest.raises(UnsupportedCapabilitiesError, match="image_formats"):
        m.demand_render_keys(oled, jpeg_only, REVISIONS)


def test_unknown_screen_revisions_fail_before_scheduling():
    with pytest.raises(ModelValidationError, match="no revisions"):
        m.plan_renders([(caps(), demand(screens=("sox next",)))], REVISIONS)


def test_every_profile_accepts_its_own_capabilities():
    for profile_id in PROFILE_PRESETS:
        c = caps(profile=profile_id)
        assert c.render_profile.profile_id == profile_id
        key = RenderKey.for_screen("date", profile_id, REVS)
        assert (key.width, key.height, key.color_mode) == (
            c.logical_width, c.logical_height, c.color_mode
        )
