"""Phase 12: physical rotation stays on the client, at final presentation."""
from __future__ import annotations

import pytest
from PIL import Image

from display.hardware_presenter import HardwarePresenter
from display.rotation import (
    ROTATIONS,
    hit_test,
    parse_rotation,
    physical_size,
    resolve_rotation,
    rotate_frame,
    to_logical,
    to_physical,
)
from display_profiles import PROFILE_PRESETS

PROFILES = ("hyperpixel4", "hyperpixel4_square", "display_hat_mini")


def marked_frame(width, height):
    """A frame with a distinct color near each logical corner."""

    image = Image.new("RGB", (width, height), "black")
    image.putpixel((0, 0), (255, 0, 0))
    image.putpixel((width - 1, 0), (0, 255, 0))
    image.putpixel((0, height - 1), (0, 0, 255))
    image.putpixel((width - 1, height - 1), (255, 255, 0))
    return image


@pytest.mark.parametrize("profile_id", PROFILES)
@pytest.mark.parametrize("rotation", ROTATIONS)
def test_output_dimensions_and_pixel_orientation(profile_id, rotation):
    profile = PROFILE_PRESETS[profile_id]
    w, h = profile.width, profile.height
    frame = marked_frame(w, h)
    panel = rotate_frame(frame, rotation)
    assert panel.size == physical_size(w, h, rotation)
    for point in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (w // 3, h // 5)):
        assert panel.getpixel(to_physical(*point, w, h, rotation)) == frame.getpixel(point)


@pytest.mark.parametrize("profile_id", PROFILES)
@pytest.mark.parametrize("rotation", ROTATIONS)
def test_touch_maps_back_to_logical_coordinates(profile_id, rotation):
    profile = PROFILE_PRESETS[profile_id]
    w, h = profile.width, profile.height
    for point in ((0, 0), (w - 1, h - 1), (w // 4, h // 3), (w - 7, 3)):
        assert to_logical(*to_physical(*point, w, h, rotation), w, h, rotation) == point
    pw, ph = physical_size(w, h, rotation)
    assert to_logical(pw + 50, -10, w, h, rotation)[0] in range(w)  # clamped


@pytest.mark.parametrize("profile_id", PROFILES)
@pytest.mark.parametrize("rotation", ROTATIONS)
def test_quad_hit_bounds_are_logical(profile_id, rotation):
    profile = PROFILE_PRESETS[profile_id]
    w, h = profile.width, profile.height
    quads = {
        "top_left": (0, 0, w // 2, h // 2),
        "top_right": (w // 2, 0, w, h // 2),
        "bottom_left": (0, h // 2, w // 2, h),
        "bottom_right": (w // 2, h // 2, w, h),
    }
    for name, (left, top, right, bottom) in quads.items():
        centre = ((left + right) // 2, (top + bottom) // 2)
        tap = to_physical(*centre, w, h, rotation)
        assert hit_test(quads, *to_logical(*tap, w, h, rotation)) == name


def test_parse_rotation_accepts_documented_forms():
    assert [parse_rotation(v) for v in ("0", "90", "180", "270", "1", "2", "3", None, "")] == [
        0, 90, 180, 270, 90, 180, 270, 0, 0]
    for bad in ("45", "left", "360"):
        with pytest.raises(ValueError):
            parse_rotation(bad)


def test_double_rotation_guard():
    strict = resolve_rotation(90, kernel_overlay=90, strict=True)
    assert strict.applied == 0 and "kernel overlay 90" in strict.describe()
    assert resolve_rotation(90, kernel_overlay=90, strict=False).applied == 90
    assert resolve_rotation(180, kernel_overlay=None, strict=True).applied == 180


class FakeDisplay:
    def __init__(self, rotation):
        self.rotation = rotation
        self.frames = []

    def image(self, frame):
        self.frames.append(frame)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_presenter_hands_logical_frames_to_the_driver(rotation):
    profile = PROFILE_PRESETS["hyperpixel4"]
    display = FakeDisplay(rotation)
    presenter = HardwarePresenter(display, profile=profile)
    presenter.present(Image.new(profile.color_mode, (profile.width, profile.height)))
    # utils.Display rotates at output; the presenter never rotates first.
    assert display.frames[-1].size == (profile.width, profile.height)
    assert presenter.rotation == rotation
    tap = to_physical(10, 20, profile.width, profile.height, rotation)
    assert presenter.touch_to_logical(*tap) == (10, 20)


def test_differently_mounted_clients_share_artifacts(tmp_path):
    pytest.importorskip("flask")
    import display_server
    from remote_display.models import RenderKey, ScreenRevisions
    from remote_display.registry import Assignment

    token = "server-token-" + "s" * 32
    config = display_server.DisplayServerConfig(auth_token=token, artifact_dir=tmp_path / "a")
    assignments = {cid: Assignment("default", "rev-1", ("date",)) for cid in ("upright", "sideways")}
    app = display_server.create_app(config, assignments=assignments.get)
    profile = PROFILE_PRESETS["hyperpixel4"]
    key = RenderKey.for_screen("date", profile.profile_id, ScreenRevisions("s", "d", "r"))
    app.extensions["desk_display_artifacts"].publish_image(
        key, Image.new(profile.color_mode, (profile.width, profile.height), 9))
    api = app.test_client()
    shas = {}
    for client_id, rotation in (("upright", 0), ("sideways", 90)):
        import display_client
        from display.rotation import RotationDecision

        caps = display_client.capabilities_for(client_id, profile,
                                               rotation=RotationDecision(rotation, None, rotation, "configured"))
        assert (caps.logical_width, caps.logical_height) == (profile.width, profile.height)
        demand = {"type": "client_demand", "version": 1, "client_id": client_id, "playlist_revision": "rev-1",
                  "required_screens": ["date"], "sync_interval_seconds": 30,
                  "package_capabilities": {"render_package_versions": [1], "image_formats": ["PNG"]}}
        body = api.post("/api/v1/register", json={"capabilities": caps.to_wire(), "demand": demand},
                        headers={"Authorization": f"Bearer {token}"}).get_json()
        credential = body["client_credential"]
        status = {"type": "client_status", "version": 1, "client_id": client_id, "playback_state": "playing",
                  "accepted_revisions": {}, "physical_rotation": rotation}
        assert api.post(f"/api/v1/clients/{client_id}/heartbeat", json={"status": status},
                        headers={"Authorization": f"Bearer {credential}"}).status_code == 200
        manifest = api.get(f"/api/v1/clients/{client_id}/manifest",
                           headers={"Authorization": f"Bearer {credential}"}).get_json()
        shas[client_id] = [a["sha256"] for a in manifest["artifacts"]]
    assert shas["upright"] == shas["sideways"] and shas["upright"]


def test_client_touch_gesture_is_mount_independent(tmp_path):
    import display_client

    profile = PROFILE_PRESETS["hyperpixel4"]

    class Presenter:
        def present(self, image):
            return image

    for rotation in ROTATIONS:
        client = display_client.build_client({
            "DESK_DISPLAY_PROFILE": profile.profile_id, "DESK_DISPLAY_CLIENT_ID": "office",
            "DESK_DISPLAY_SERVER_URL": "https://render.lan", "DESK_DISPLAY_CLIENT_CACHE_DIR": str(tmp_path),
            "DISPLAY_ROTATION": rotation,
        }, presenter=Presenter(), transport=lambda *a, **k: None)
        assert client.physical_rotation == rotation
        assert client.sync.capabilities.hardware.driver.startswith(f"rotation {rotation}")
        client.on_touch(*to_physical(5, profile.height // 2, profile.width, profile.height, rotation))
        assert client.controls.back and not client.controls.skip
        client.controls.back = False
        client.on_touch(*to_physical(profile.width - 5, 5, profile.width, profile.height, rotation))
        assert client.controls.skip
