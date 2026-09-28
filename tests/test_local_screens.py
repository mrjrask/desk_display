"""Screens a display client draws from its own hardware (the inside sensor)."""
from __future__ import annotations

from PIL import Image

from display_profiles import PROFILE_PRESETS
from playback.local_screens import SensorScreen, default_local_screens, is_local, local_entries
from remote_display.models import ClientCapabilities, ClientDemand, PackageCapabilities, ScreenRevisions, demand_render_keys
from rendering import screen_classes


def test_inside_is_drawn_by_clients_not_the_server():
    entry = screen_classes.classify("inside")
    assert entry.kind == screen_classes.CLIENT_SENSOR
    assert entry.remote_supported and not entry.server_rendered
    assert not screen_classes.server_renders("inside")
    assert screen_classes.server_renders("weather1")
    assert set(default_local_screens()) == {"inside"}


def test_render_plan_skips_client_drawn_screens():
    profile = PROFILE_PRESETS["hyperpixel4"]
    caps = ClientCapabilities(
        protocol_version=1, client_software_version="test", client_id="office",
        display_profile=profile.profile_id, logical_width=profile.width, logical_height=profile.height,
        image_formats=("PNG",), color_modes=(profile.color_mode,), render_package_versions=(1,),
    )
    demand = ClientDemand(client_id="office", playlist_revision="r1", required_screens=("inside", "weather1"),
                          package_capabilities=PackageCapabilities(render_package_versions=(1,),
                                                                   image_formats=("PNG",)),
                          sync_interval_seconds=30)
    revisions = {s: ScreenRevisions("s1", "d1", "r1") for s in ("inside", "weather1")}
    keys = demand_render_keys(caps, demand, revisions)
    assert {key.screen_id for key in keys} == {"weather1"}


def test_sensor_screen_is_skipped_until_its_probe_finds_a_sensor():
    sensor = SensorScreen(probe=lambda: True, render=lambda: Image.new("RGB", (320, 240), "red"))
    assert not sensor.available and sensor.render(320, 240, "RGB") is None
    entries = local_entries(["weather1", "inside"], {"inside": sensor})
    assert list(entries) == ["inside"] and is_local(entries["inside"])
    assert sensor.wait(5) and sensor.available
    assert sensor.render(320, 240, "RGB").getpixel((0, 0)) == (255, 0, 0)


def test_sensor_screen_fits_the_client_panel():
    sensor = SensorScreen(probe=lambda: True, render=lambda: Image.new("RGB", (320, 240), "white"))
    sensor.start()
    assert sensor.wait(5)
    image = sensor.render(128, 64, "1")
    assert image.size == (128, 64) and image.mode == "1"


def test_sensor_faults_never_reach_playback():
    def broken_probe():
        raise OSError("no i2c")

    def broken_render():
        raise OSError("read failed")

    missing = SensorScreen(probe=broken_probe)
    missing.start()
    assert missing.wait(5) and not missing.available
    flaky = SensorScreen(probe=lambda: True, render=broken_render)
    flaky.start()
    assert flaky.wait(5) and flaky.render(320, 240, "RGB") is None
    assert not is_local({"screen_id": "weather1", "sha256": "x"})
