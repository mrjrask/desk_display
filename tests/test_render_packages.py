"""Phase 10a: every screen is classified, and motion ships as a render package."""
from __future__ import annotations

import copy
import datetime as dt
import json

import pytest
from PIL import Image

import screens_catalog
from display_profiles import PROFILE_PRESETS
from remote_display.artifact_store import ArtifactStore
from remote_display.manifest import build_client_manifest, referenced_hashes
from remote_display.models import RenderKey, ScreenRevisions
from remote_display.render_package import (
    MAX_ANIMATION_FRAMES,
    RENDER_PACKAGE_MEDIA_TYPE,
    PackageBuilder,
    PackageError,
    asset_image,
    package_bytes,
    validate_package,
)
from rendering import screen_classes
from rendering.clock_faces import clock_background, clock_layout, render_clock
from rendering.packaging import build_package, clock_package
from rendering.screen_renderer import _CaptureDisplay
from utils import animate_scroll, scroll_vertical_content

PROFILE = PROFILE_PRESETS["hyperpixel4"]
W, H = PROFILE.width, PROFILE.height


def key(screen="date", profile="hyperpixel4", data="d1"):
    return RenderKey.for_screen(screen, profile, ScreenRevisions("s1", data, "r1"))


class Artifact:
    def __init__(self, capture=None, frames=(), image=None):
        self.capture = capture
        self.recorded_frames = tuple(frames)
        self.image = image or Image.new(PROFILE.color_mode, (W, H))


def gradient(width, height):
    image = Image.new("RGB", (width, height))
    image.putdata([((y * 7) % 256, (y // 3) % 256, (x * 5) % 256) for y in range(height) for x in range(width)])
    return image


# ── Classification ─────────────────────────────────────────────────────────


def test_every_catalogued_screen_has_an_explicit_class():
    assert set(screen_classes.CLASSIFICATIONS) == set(screens_catalog.SCREEN_IDS)
    for entry in screen_classes.CLASSIFICATIONS.values():
        assert entry.kind in screen_classes.CLASSES
        assert entry.package_kind == screen_classes.PACKAGE_KINDS[entry.kind]


def test_every_registered_screen_is_classified():
    from rendering.screen_renderer import _CaptureDisplay
    from screens.registry import ScreenContext, build_screen_registry
    from services import feeds

    for profile_id in ("hyperpixel4", "display_hat_mini", "hdmi_1080p"):
        profile = PROFILE_PRESETS[profile_id]
        now = dt.datetime(2026, 9, 25, 12, tzinfo=dt.timezone.utc)
        context = ScreenContext(
            display=_CaptureDisplay(profile), cache=copy.deepcopy(feeds.default_cache()), logos={},
            image_dir="images", now=now, now_utc=now, offline=False, weather_fetched_at=None,
            skip_scoreboards=False, render_profile=profile, allow_upstream_requests=False,
        )
        registry, _ = build_screen_registry(context)
        assert set(registry) <= set(screen_classes.CLASSIFICATIONS), profile_id


def test_expected_classes_for_representative_screens():
    kind = {s: c.kind for s, c in screen_classes.CLASSIFICATIONS.items()}
    assert kind["date"] == kind["nixie"] == screen_classes.CLIENT_TIMED
    assert kind["quad"] == kind["weather quad"] == screen_classes.INTERACTIVE_FOCUS
    assert kind["cubs schedule quad"] == screen_classes.COMPOSITE
    assert kind["news headlines"] == screen_classes.TICKER_OVERLAY
    assert kind["MLB Scoreboard"] == screen_classes.SCROLLING_CANVAS
    assert kind["cubs logo"] == kind["weather radar"] == screen_classes.FINITE_ANIMATION
    assert kind["cubs live"] == screen_classes.PERIODIC
    assert kind["inside"] == screen_classes.UNSUPPORTED
    assert kind["weather1"] == screen_classes.STATIC


def test_focus_targets_for_interactive_quads(monkeypatch):
    import importlib

    # Other tests may reload screens.registry; patch the module sys.modules holds.
    registry = importlib.import_module("screens.registry")

    assert screen_classes.focus_targets("weather quad") == ("weather1", "air quality", "weather hourly",
                                                            "weather daily")
    monkeypatch.setattr(registry, "_quad_layout_from_layouts",
                        lambda: (True, 1.0, [["date", "quad", "bogus"], ["weather1"]]))
    assert screen_classes.focus_targets("quad") == ("date", "weather1")
    assert screen_classes.focus_targets("cubs schedule quad") == ()
    assert screen_classes.interaction_targets(["weather1", "cubs schedule quad"]) == set()


# ── Capture ────────────────────────────────────────────────────────────────


def test_scroll_capture_rebuilds_the_whole_canvas():
    display = _CaptureDisplay(PROFILE)
    tall = gradient(W, H * 2 + 37)
    shown = []

    def render_at(offset):
        shown.append(offset)
        display.image(tall.crop((0, offset, W, offset + H)))

    scroll_vertical_content(display=display, content_height=tall.height, viewport_width=W, viewport_height=H,
                            render_at_offset=render_at, base_step=3, pause_start=2.0, pause_end=1.5,
                            min_frame_time=0.02)
    capture = display.capture
    assert capture["kind"] == "scroll" and capture["canvas"].tobytes() == tall.tobytes()
    assert capture["pause_start_seconds"] == 2.0 and capture["direction"] == "down"
    assert shown[-1] == 0 and len(shown) <= 5  # pages, not per-pixel frames
    package = build_package(key("MLB Scoreboard"), PROFILE, Artifact(capture))
    assert package["kind"] == "scroll" and package["scroll"]["viewport"] == [W, H]
    assert asset_image(package, package["scroll"]["canvas"]).size == (W, tall.height)


def test_content_that_fits_needs_no_package():
    display = _CaptureDisplay(PROFILE)
    scroll_vertical_content(display=display, content_height=H, viewport_width=W, viewport_height=H,
                            render_at_offset=lambda o: display.image(gradient(W, H)), base_step=3,
                            pause_start=0, pause_end=0)
    assert display.capture is None
    assert build_package(key("MLB Scoreboard"), PROFILE, Artifact(display.capture)) is None


def test_real_displays_still_scroll_frame_by_frame():
    class Display:
        width, height = W, H
        frames = 0

        def image(self, image):
            self.frames += 1

        def wait_for_skip(self, seconds):
            return False

    display = Display()
    scroll_vertical_content(display=display, content_height=H + 60, viewport_width=W, viewport_height=H,
                            render_at_offset=lambda o: display.image(None), base_step=2, pause_start=0,
                            pause_end=0, min_frame_time=0.0)
    assert display.frames > 10


def test_logo_slide_is_a_sprite_not_frames():
    display = _CaptureDisplay(PROFILE)
    logo = Image.new("RGBA", (120, 90), (200, 30, 30, 255))
    animate_scroll(display, logo, speed=2.2)
    slide = display.capture
    assert slide["kind"] == "slide" and slide["speed_px_per_second"] > 0
    assert display.current_image.getpixel((W // 2, H // 2)) == (200, 30, 30)  # still is centred
    package = build_package(key("cubs logo"), PROFILE, Artifact(slide))
    assert set(package["animation"]) == {"slide"} and len(package["assets"]) == 1


def test_recorded_animation_is_bounded_and_keeps_the_final_frame():
    display = _CaptureDisplay(PROFILE, record_frames=True)
    for step in range(200):
        display.image(Image.new("RGB", (W, H), (step, 0, 0)))
        display.wait_for_skip(0.02)
    frames = display.recorded_frames()
    assert 2 <= len(frames) <= MAX_ANIMATION_FRAMES
    assert frames[-1][0].getpixel((0, 0)) == (199, 0, 0)
    assert sum(seconds for _image, seconds in frames) == pytest.approx(4.0)
    package = build_package(key("NHL Standings Overview West"), PROFILE, Artifact(frames=frames))
    assert package["animation"]["loops"] == 1 and len(package["animation"]["frames"]) == len(frames)


def test_unshown_frames_are_not_recorded():
    display = _CaptureDisplay(PROFILE, record_frames=True)
    display.image(Image.new("RGB", (W, H), "red"))  # cleared at once, never shown
    display.image(Image.new("RGB", (W, H), "blue"))
    assert [image.getpixel((0, 0)) for image, _ in display.recorded_frames()] == [(0, 0, 255)]


def test_radar_frames_are_captured_once(monkeypatch):
    import screens.draw_weather as weather

    frames = [weather.RadarFrame(Image.new("RGBA", (W, H), (0, 0, i * 40, 128)), None) for i in range(4)]
    monkeypatch.setattr(weather, "_fetch_radar_frames", lambda **kwargs: list(frames))
    monkeypatch.setattr(weather, "_fetch_base_map", lambda **kwargs: None)
    monkeypatch.setattr(weather.time, "sleep", lambda s: pytest.fail("server capture must not sleep"))
    display = _CaptureDisplay(PROFILE)
    weather.draw_weather_radar(display, {}, transition=True)
    assert display.capture["kind"] == "frames" and len(display.capture["frames"]) == 4
    package = build_package(key("weather radar"), PROFILE, Artifact(display.capture))
    assert package["animation"]["loops"] == weather.RADAR_ANIMATION_LOOPS


def test_quad_tiles_carry_bounds_frames_and_focus():
    from screens.draw_quad import _TileSpec, draw_quad_screen
    from screens.registry import _invoke_for_profile

    display = _CaptureDisplay(PROFILE)
    tiles = [_TileSpec(label, lambda i=i: [Image.new("RGB", (W, H), (i * 50, f, 0)) for f in range(12)])
             for i, label in enumerate(["weather1", "air quality", "weather hourly", "weather daily"])]
    _invoke_for_profile(draw_quad_screen, PROFILE, display, tiles, transition=True)
    capture = display.capture
    assert [t["bounds"] for t in capture["tiles"]] == [(0, 0, W // 2, H // 2), (W // 2, 0, W, H // 2),
                                                       (0, H // 2, W // 2, H), (W // 2, H // 2, W, H)]
    package = build_package(key("weather quad"), PROFILE, Artifact(capture, image=display.current_image))
    body = package["composite"]
    assert [t["focus_screen"] for t in body["tiles"]] == ["weather1", "air quality", "weather hourly",
                                                          "weather daily"]
    assert all(len(t["frames"]) == 10 for t in body["tiles"])
    schedule = build_package(key("cubs schedule quad"), PROFILE, Artifact(capture, image=display.current_image))
    assert all(t["focus_screen"] is None for t in schedule["composite"]["tiles"])


def test_ticker_lanes_ship_as_looping_strips():
    import screens.draw_news_headlines as dnh
    from services.news_feeds import NewsTopic

    row_height, row_tops = dnh._compute_row_layout(2)
    rows = [dnh._TickerRow(
        topic=NewsTopic(id=f"t{n}", label=f"T{n}", name="t", url=""), theme=dnh._FALLBACK_THEME,
        entries=[dnh._TickerEntry(None, f"Headline {n}-{i}", 140, None, 0) for i in range(5)],
        speed=2.0, offset=33.0,
    ) for n in range(2)]
    display = _CaptureDisplay(PROFILE)
    dnh._run_ticker(display, rows, {"rows": []})
    lanes = display.capture["lanes"]
    assert [lane["strip"].size for lane in lanes] == [(700, row_height)] * 2
    assert lanes[0]["offset_px"] == 33.0
    assert lanes[0]["speed_px_per_second"] == pytest.approx(2.0 / dnh._FRAME_INTERVAL_SECONDS)
    package = build_package(key("news headlines"), PROFILE, Artifact(display.capture))
    assert len(package["ticker"]["lanes"]) == 2


# ── Clocks ─────────────────────────────────────────────────────────────────


def test_client_drawn_clock_matches_the_standalone_screen():
    from screens import draw_date_time
    from screens.registry import _invoke_for_profile

    now = dt.datetime(2026, 9, 25, 22, 7, tzinfo=dt.timezone.utc)
    layout = {**clock_layout("date", PROFILE), "time_zone": "America/Chicago", "show_ip": False}
    colors = ((250, 90, 20), (30, 200, 240))
    client = render_clock(layout, PROFILE, now, colors=colors)
    standalone = _invoke_for_profile(draw_date_time._compose_frame, PROFILE, "date_time", *colors, False, "date",
                                     now=now, show_ip=False)
    assert client.tobytes() == standalone.tobytes()
    later = render_clock(layout, PROFILE, now + dt.timedelta(minutes=1), colors=colors)
    assert later.tobytes() != client.tobytes()  # the client, not the server, advances the time


def test_nixie_follows_the_layout_time_format():
    now = dt.datetime(2026, 9, 25, 22, 7, 9, tzinfo=dt.timezone.utc)
    layout = clock_layout("nixie", PROFILE)
    twelve = render_clock({**layout, "time_format": "12"}, PROFILE, now)
    assert twelve.tobytes() != render_clock({**layout, "time_format": "24"}, PROFILE, now).tobytes()


def test_clock_shows_only_the_clients_own_address():
    now = dt.datetime(2026, 9, 25, 22, 7, tzinfo=dt.timezone.utc)
    layout = {**clock_layout("nixie", PROFILE), "show_ip": True}
    without = render_clock(layout, PROFILE, now)
    with_ip = render_clock(layout, PROFILE, now, ip_text="IP: 10.0.0.7")
    assert without.tobytes() != with_ip.tobytes()


def test_clock_package_round_trip():
    layout = clock_layout("date", PROFILE)
    package = clock_package(key("date"), PROFILE, layout, clock_background(layout, PROFILE))
    loaded = validate_package(package_bytes(package), key=key("date"))
    assert loaded["clock"]["layout"] == layout and loaded["classification"] == "client_timed"


# ── Validation ─────────────────────────────────────────────────────────────


def _valid_scroll():
    builder = PackageBuilder()
    canvas = builder.add(gradient(W, H * 2))
    return builder.build(screen_id="MLB Scoreboard", render_profile="hyperpixel4", width=W, height=H,
                         color_mode="RGB", render_key_digest=key("MLB Scoreboard").digest,
                         classification="scrolling_canvas", kind="scroll",
                         body={"canvas": canvas, "viewport": [W, H], "step_px": 3, "frame_seconds": 0.02,
                               "pause_start_seconds": 1, "pause_end_seconds": 1, "direction": "down"})


@pytest.mark.parametrize("mutate, code", [
    (lambda p: p.update(render_package_schema_version=2), "unsupported_version"),
    (lambda p: p.update(kind="hologram"), "invalid_schema"),
    (lambda p: p["scroll"].update(canvas="nope"), "missing_asset"),
    (lambda p: p["scroll"].update(viewport=[1, 1]), "invalid_field"),
    (lambda p: p["assets"]["a0"].update(sha256="0" * 64), "invalid_asset"),
    (lambda p: p["assets"]["a0"].update(data="!!"), "invalid_asset"),
    (lambda p: p.update(width=1), "invalid_field"),
    (lambda p: p.update(clock={}), "invalid_schema"),
])
def test_invalid_packages_are_rejected(mutate, code):
    package = _valid_scroll()
    validate_package(package, key=key("MLB Scoreboard"))
    mutate(package)
    with pytest.raises(PackageError) as info:
        validate_package(json.loads(json.dumps(package)), key=key("MLB Scoreboard"))
    assert info.value.code == code


def test_package_must_belong_to_its_key():
    with pytest.raises(PackageError) as info:
        validate_package(_valid_scroll(), key=key("MLB Scoreboard", data="d2"))
    assert info.value.code == "wrong_key"


def test_animation_frame_limit_and_bounds():
    builder = PackageBuilder()
    frames = [{"asset": builder.add(Image.new("RGB", (W, H), (i, 0, 0))), "duration_ms": 50}
              for i in range(MAX_ANIMATION_FRAMES + 1)]
    package = builder.build(screen_id="weather radar", render_profile="hyperpixel4", width=W, height=H,
                            color_mode="RGB", render_key_digest="x", classification="finite_animation",
                            kind="animation", body={"frames": frames, "loops": 1})
    with pytest.raises(PackageError):
        validate_package(package)
    package["animation"]["frames"] = frames[:3]
    validate_package(package)
    tile = {"bounds": [0, 0, W + 1, H], "frames": [frames[0]["asset"]], "focus_screen": None}
    composite = dict(package, kind="composite", classification="composite",
                     composite={"base": frames[0]["asset"], "tiles": [tile], "frame_seconds": 0.1,
                                "duration_seconds": 5})
    del composite["animation"]
    with pytest.raises(PackageError) as info:
        validate_package(composite)
    assert info.value.code == "invalid_bounds"


# ── Publication and manifests ──────────────────────────────────────────────


def _publish_with_package(store, data="d1"):
    k = key("MLB Scoreboard", data=data)
    package = dict(_valid_scroll(), render_key_digest=k.digest)
    return store.publish_image(k, Image.new("RGB", (W, H), 9), package=package)


def test_package_is_published_with_its_still_image(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    record = _publish_with_package(store)
    ref = record.metadata["package"]
    assert ref["kind"] == "scroll" and ref["media_type"] == RENDER_PACKAGE_MEDIA_TYPE
    path, media_type = store.open_object(f"{ref['sha256']}.json")
    assert media_type == RENDER_PACKAGE_MEDIA_TYPE and validate_package(path.read_bytes())["kind"] == "scroll"

    manifest = build_client_manifest(
        store, client_id="office", display_profile="hyperpixel4", requested_screens=["MLB Scoreboard"],
        assignment={}, configuration={}, artifact_url=lambda n: f"/a/{n}", now=0)
    entry = manifest["artifacts"][0]
    assert entry["media_type"] == "image/png" and entry["remote_class"] == "scrolling_canvas"
    assert entry["package"]["url"] == f"/a/{ref['sha256']}.json"
    assert referenced_hashes(manifest) == {record.sha256, ref["sha256"]}


def test_invalid_package_keeps_the_last_good_output(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    good = _publish_with_package(store)
    bad = dict(_valid_scroll(), render_key_digest=key("MLB Scoreboard", data="d2").digest, kind="nope")
    with pytest.raises(Exception) as info:
        store.publish_image(key("MLB Scoreboard", data="d2"), Image.new("RGB", (W, H), 1), package=bad)
    assert getattr(info.value, "code", None) == "invalid_package"
    resolved = store.resolve("MLB Scoreboard", "hyperpixel4")
    assert resolved.state == "fallback" and resolved.record == good


def test_package_objects_are_kept_while_referenced(tmp_path):
    class Clock:
        now = 1000.0

        def __call__(self):
            return self.now

    clock = Clock()
    store = ArtifactStore(tmp_path / "store", grace_seconds=10, previous_revisions=0, clock=clock)
    first = _publish_with_package(store)
    package_object = f"{first.metadata['package']['sha256']}.json"
    clock.now += 100
    store.collect_garbage()
    assert store.open_object(package_object) is not None  # current output
    store.publish_image(key("MLB Scoreboard", data="d2"), Image.new("RGB", (W, H), 5))
    clock.now += 100
    store.collect_garbage()
    assert store.open_object(package_object) is None  # released with its still image


# ── Interaction demand ─────────────────────────────────────────────────────


def test_touch_clients_demand_quad_focus_targets(tmp_path):
    pytest.importorskip("flask")
    import display_client
    import display_server
    from display.rotation import RotationDecision

    token = "server-token-" + "s" * 32
    app = display_server.create_app(display_server.DisplayServerConfig(enrollment="shared", auth_token=token,
                                                                        artifact_dir=tmp_path / "a"))
    api = app.test_client()
    registry = app.extensions["desk_display_registry"]
    for client_id, touch in (("kiosk", True), ("desk", False)):
        caps = display_client.capabilities_for(client_id, PROFILE, has_touch=touch,
                                               rotation=RotationDecision(0, None, 0, "configured"))
        demand = {"type": "client_demand", "version": 1, "client_id": client_id, "playlist_revision": "rev-1",
                  "required_screens": ["weather quad", "date"], "sync_interval_seconds": 30,
                  "package_capabilities": {"render_package_versions": [1], "image_formats": ["PNG"]}}
        response = api.post("/api/v1/register", json={"capabilities": caps.to_wire(), "demand": demand},
                            headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 201
    demands = {e.client_id: e.demand for e in registry.demand_entries()}
    assert set(demands["kiosk"].touch_targets) == {"weather1", "air quality", "weather hourly", "weather daily"}
    assert demands["desk"].touch_targets == ()
