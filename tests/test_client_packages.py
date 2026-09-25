"""Phase 10b: client playback of render packages, touch focus and fallbacks.

Playback, timing and focus are exercised against a real display server app
(through the thin client tests' transport) so a package travels the actual
wire protocol before the client plays it.
"""
from __future__ import annotations

import datetime as dt
import random

import pytest
from PIL import Image

from display_profiles import PROFILE_PRESETS
from playback.package_player import PackagePlayback
from remote_display import fallbacks
from remote_display.render_package import PackageBuilder

pytest.importorskip("flask")

import display_client  # noqa: E402
import display_server  # noqa: E402
from remote_display.client_sync import Backoff  # noqa: E402
from remote_display.models import RenderKey, ScreenRevisions  # noqa: E402
from remote_display.playlist_store import PlaylistStore  # noqa: E402
from tests.test_display_client import TOKEN, Clock, FlaskTransport, Presenter  # noqa: E402

PROFILE = PROFILE_PRESETS["hyperpixel4"]
W, H = PROFILE.width, PROFILE.height
TILES = ("weather1", "air quality", "weather hourly", "weather daily")


def key(screen, data="d1"):
    return RenderKey.for_screen(screen, PROFILE.profile_id, ScreenRevisions("s1", data, "r1"))


def solid(color, size=(W, H)):
    return Image.new("RGB", size, color)


def build(screen, classification, kind, body_fn, *, data="d1"):
    builder = PackageBuilder()
    body = body_fn(builder)
    return builder.build(screen_id=screen, render_profile=PROFILE.profile_id, width=W, height=H,
                         color_mode="RGB", render_key_digest=key(screen, data).digest,
                         classification=classification, kind=kind, body=body)


def scroll_package(screen="MLB Scoreboard"):
    def body(b):
        canvas = Image.new("RGB", (W, H * 2))
        canvas.paste(solid((255, 0, 0), (W, H)), (0, H))
        return {"canvas": b.add(canvas), "viewport": [W, H], "step_px": H // 4, "frame_seconds": 0.5,
                "pause_start_seconds": 1, "pause_end_seconds": 2, "direction": "down"}
    return build(screen, "scrolling_canvas", "scroll", body)


def frames_package():
    def body(b):
        return {"frames": [{"asset": b.add(solid((i * 60, 0, 0))), "duration_ms": 250} for i in range(3)],
                "loops": 2}
    return build("weather radar", "finite_animation", "animation", body)


def slide_package():
    def body(b):
        return {"slide": {"sprite": b.add(solid((0, 0, 255), (40, 20))), "y": 10,
                          "speed_px_per_second": 200, "background": [0, 0, 0]}}
    return build("bears logo", "finite_animation", "animation", body)


def ticker_package():
    def body(b):
        strip = Image.new("RGB", (100, 20))
        strip.paste(solid((0, 255, 0), (10, 20)), (0, 0))
        return {"base": b.add(solid((0, 0, 0))), "duration_seconds": 30,
                "lanes": [{"bounds": [0, 0, W, 20], "strip": b.add(strip), "speed_px_per_second": 40,
                           "offset_px": 0, "background": [0, 0, 0]}]}
    return build("news headlines", "ticker_overlay", "ticker", body)


def quad_package(screen="weather quad", data="d1"):
    half_w, half_h = W // 2, H // 2
    bounds = [[0, 0, half_w, half_h], [half_w, 0, W, half_h], [0, half_h, half_w, H], [half_w, half_h, W, H]]

    def body(b):
        tiles = [{"bounds": box, "frames": [b.add(solid((40 * i, 40, 40), (half_w, half_h))),
                                            b.add(solid((40 * i, 80, 80), (half_w, half_h)))],
                  "focus_screen": tile} for i, (box, tile) in enumerate(zip(bounds, TILES))]
        return {"base": b.add(solid((0, 0, 0))), "tiles": tiles, "frame_seconds": 1, "duration_seconds": 12}
    return build(screen, "interactive_focus", "composite", body, data=data)


def play(package, hold=5.0, **kwargs):
    return PackagePlayback(package, PROFILE, hold_seconds=hold, rng=random.Random(0), **kwargs)


# ── Timing per kind ────────────────────────────────────────────────────────


def test_scroll_plays_once_then_holds_the_bottom():
    playback = play(scroll_package())
    assert playback.motion_seconds == pytest.approx(1 + 4 * 0.5 + 2)
    assert playback.duration == pytest.approx(playback.motion_seconds + 5)
    assert playback.frame_at(0).getpixel((0, 0)) == (0, 0, 0)
    assert playback.key_at(0.9) == playback.key_at(0)  # starting pause
    assert playback.key_at(1.5) != playback.key_at(0)
    assert playback.frame_at(10).getpixel((0, 0)) == (255, 0, 0)  # bottom of the canvas


def test_recorded_frames_loop_then_hold_the_last():
    playback = play(frames_package())
    assert playback.motion_seconds == pytest.approx(0.75 * 2)
    assert [playback.frame_at(t).getpixel((0, 0))[0] for t in (0, 0.3, 0.6, 0.8)] == [0, 60, 120, 0]
    assert playback.frame_at(5).getpixel((0, 0))[0] == 120


def test_logo_slides_across_and_settles_centred():
    playback = play(slide_package())
    assert playback.motion_seconds == pytest.approx((W + 40) / 200)
    settled = playback.frame_at(playback.motion_seconds + 1)
    assert settled.getpixel((W // 2, 15)) == (0, 0, 255)
    assert settled.getpixel((0, 15)) == (0, 0, 0)


def test_ticker_scrolls_its_lane_without_touching_the_rest():
    playback = play(ticker_package(), hold=10)
    assert playback.duration == 30  # its own window beats a shorter hold
    first, later = playback.frame_at(0), playback.frame_at(0.5)
    assert first.getpixel((0, 5)) == (0, 255, 0)
    assert later.getpixel((0, 5)) == (0, 0, 0) and later.getpixel((80, 5)) == (0, 255, 0)
    assert later.getpixel((0, H - 1)) == first.getpixel((0, H - 1))


def test_composite_cycles_tile_frames_and_reports_focus_targets():
    playback = play(quad_package(), hold=20)
    assert playback.duration == 20
    assert playback.frame_at(0).getpixel((1, 1)) != playback.frame_at(1.2).getpixel((1, 1))
    targets = playback.focus_targets()
    assert set(targets) == set(TILES) and targets["weather1"] == (0, 0, W // 2, H // 2)


def test_clock_is_drawn_from_the_clients_own_time():
    from rendering.clock_faces import clock_background, clock_layout
    from rendering.packaging import clock_package

    layout = clock_layout("nixie", PROFILE)
    package = clock_package(key("nixie"), PROFILE, layout, clock_background(layout, PROFILE))
    now = [dt.datetime(2026, 9, 25, 12, 0, 0, tzinfo=dt.timezone.utc)]
    playback = play(package, clock=lambda: now[0])
    first = playback.frame_at(0)
    assert playback.key_at(0.5) == playback.key_at(0)
    now[0] += dt.timedelta(seconds=1)
    assert playback.frame_at(0.5).tobytes() != first.tobytes()  # seconds tick with no server


# ── Fallbacks ──────────────────────────────────────────────────────────────


def test_fallbacks_are_deterministic_and_explain_themselves():
    still = fallbacks.playback_mode("MLB Scoreboard", supports_animation=False, has_touch=False, color_mode="RGB")
    assert still.mode == fallbacks.STILL and [n[0] for n in still.notes] == ["no_animation"]
    quad = fallbacks.playback_mode("weather quad", supports_animation=True, has_touch=False, color_mode="RGB")
    assert quad.animated and not quad.expands and quad.notes[0][0] == "no_touch"
    touch = fallbacks.playback_mode("weather quad", supports_animation=True, has_touch=True, color_mode="RGB")
    assert touch.expands and touch.notes == ()
    mono = fallbacks.playback_mode("weather radar", supports_animation=True, has_touch=False, color_mode="1")
    assert [n[0] for n in mono.notes] == ["monochrome"]
    assert fallbacks.playback_mode("inside", supports_animation=True, has_touch=True,
                                   color_mode="RGB").mode == fallbacks.UNAVAILABLE
    assert fallbacks.playback_mode("nope", supports_animation=True, has_touch=True,
                                   color_mode="RGB").mode == fallbacks.UNAVAILABLE


def test_assignment_warnings_come_from_the_same_fallbacks():
    from remote_playlists_ui import capability_warnings

    caps = {"capabilities": {"display_profile": "waveshare_oled_128x64", "supports_animation": False,
                             "has_touch": False}}
    document = {"screens": {"MLB Scoreboard": 1, "weather quad": 1, "weather radar": 1}, "sequence": []}
    codes = {w["code"] for w in capability_warnings(document, caps)}
    assert {"no_animation", "no_touch", "monochrome", "small_display"} <= codes


# ── The client: playback, touch focus and offline interaction ──────────────


@pytest.fixture
def env(tmp_path):
    clock = Clock()
    store_path = tmp_path / "playlists.json"
    store = PlaylistStore(store_path)
    doc = {"screens": {"weather quad": 1, "MLB Scoreboard": 1}, "sequence": []}
    playlist = store.create("Office", doc, actor="test")
    store.assign("office", playlist["id"], expected_playlist_id=None, actor="test")
    config = display_server.DisplayServerConfig(enrollment="shared",
        auth_token=TOKEN, admin_token="admin-token-" + "a" * 32, lease_seconds=300,
        artifact_dir=tmp_path / "server-artifacts", playlist_store_path=store_path,
    )
    app = display_server.create_app(config, clock=clock)
    artifacts = app.extensions["desk_display_artifacts"]

    def publish(screen, color=0, package=None):
        return artifacts.publish_image(key(screen), solid((color, color, color)), package=package)

    def make_client(**settings):
        values = {
            "DESK_DISPLAY_PROFILE": PROFILE.profile_id,
            "DESK_DISPLAY_CLIENT_ID": "office",
            "DESK_DISPLAY_SERVER_URL": "https://render.lan:8765",
            "DESK_DISPLAY_CLIENT_TOKEN": TOKEN,
            "DESK_DISPLAY_CLIENT_CACHE_DIR": str(tmp_path / "client"),
            "DESK_DISPLAY_CLIENT_CACHE_MAX_MB": 16,
        }
        values.update(settings)
        client = display_client.build_client(values, presenter=Presenter(), transport=transport)
        client.sync.backoff = Backoff(rng=random.Random(1))
        return client

    transport = FlaskTransport(app)

    class Env:
        pass

    result = Env()
    result.__dict__.update(publish=publish, make_client=make_client, transport=transport)
    return result


def publish_all(env, *, tiles=TILES):
    env.publish("weather quad", 10, quad_package())
    env.publish("MLB Scoreboard", 20, scroll_package())
    for index, tile in enumerate(tiles):
        env.publish(tile, 100 + index)


def synced(client, passes=3):
    for _ in range(passes):
        client.sync.sync_once()
    return client


def test_touch_setting_defaults_from_the_profile(env):
    assert env.make_client().has_touch  # hyperpixel4 has a touch layer
    assert not env.make_client(DESK_DISPLAY_CLIENT_TOUCH="off").has_touch
    assert not env.make_client(DESK_DISPLAY_CLIENT_ANIMATION="0").supports_animation


def test_client_plays_packages_with_their_own_timing(env):
    publish_all(env)
    client = synced(env.make_client())
    screens = {}
    for _ in range(2):
        screen, seconds = client.step()
        screens[screen] = (seconds, client.animation)
    assert screens["MLB Scoreboard"][1].kind == "scroll"
    assert screens["MLB Scoreboard"][0] == pytest.approx(screens["MLB Scoreboard"][1].duration)
    assert screens["weather quad"][1].kind == "composite"


def test_stills_only_client_skips_package_downloads(env):
    publish_all(env)
    client = synced(env.make_client(DESK_DISPLAY_CLIENT_ANIMATION="0"))
    assert not any(path.endswith(".json") for _method, path in env.transport.calls if "/artifacts/" in path)
    screen, _seconds = client.step()
    assert screen is not None and client.animation is None


def test_activation_waits_for_every_focus_target(env):
    publish_all(env, tiles=TILES[:3])
    blocked = env.publish(TILES[3], 103).sha256
    real = env.transport.__call__

    def failing(method, path, **kwargs):
        if blocked in path:
            from remote_display.client_sync import TransportError

            raise TransportError("connection reset")
        return real(method, path, **kwargs)

    client = env.make_client()
    client.sync.transport = failing
    for _ in range(3):
        try:
            client.sync.sync_once()
        except Exception:  # noqa: BLE001 - the failed download is the point
            pass
    assert client.step()[0] is None  # a quad whose tile is not cached is not accepted yet
    client.sync.transport = real
    synced(client)
    assert client.step()[0] is not None


def test_tiles_the_server_has_not_rendered_do_not_block_the_quad(env):
    publish_all(env, tiles=TILES[:3])
    client = synced(env.make_client())
    _show(client, "weather quad")
    client.on_touch(W - 5, H - 5)  # the unrendered tile falls back to skip
    assert client.controls.focus is None and client.controls.skip


def _show(client, screen):
    for _ in range(4):
        if client.step()[0] == screen:
            return
    raise AssertionError(f"{screen} never played")


def test_tapping_a_tile_opens_it_and_returns_to_the_quad_offline(env):
    publish_all(env)
    client = synced(env.make_client())
    env.transport.down = True  # everything below is local
    _show(client, "weather quad")
    scheduler_before = client._player.scheduler.export_state()

    client.on_touch(W - 5, H - 5)  # bottom-right tile
    assert client.controls.focus == "weather daily"
    assert client.step()[0] == "weather daily"
    assert client.playback.focus_return == "weather quad"
    assert client.presenter.frames[-1].getpixel((0, 0)) == (103, 103, 103)

    client.on_touch(5, 5)  # any tap on a focused tile goes back
    assert client.controls.unfocus
    assert client.step()[0] == "weather quad"
    assert client._player.scheduler.export_state() == scheduler_before  # the rotation never moved
    assert client.step()[0] == "MLB Scoreboard"


def test_focused_tile_returns_on_its_own_when_its_time_is_up(env):
    publish_all(env)
    client = synced(env.make_client())
    _show(client, "weather quad")
    client.on_touch(5, 5)
    assert client.step()[0] == "weather1"
    assert client.step()[0] == "weather quad"


def test_without_touch_taps_only_skip_and_go_back(env):
    publish_all(env)
    client = synced(env.make_client(DESK_DISPLAY_CLIENT_TOUCH="off"))
    _show(client, "weather quad")
    client.on_touch(W - 5, H - 5)
    assert client.controls.focus is None and client.controls.skip


def test_non_interactive_screens_keep_the_back_and_skip_halves(env):
    publish_all(env)
    client = synced(env.make_client())
    _show(client, "MLB Scoreboard")
    client.on_touch(5, 5)
    assert client.controls.back and client.controls.focus is None


def test_wait_animates_and_polls_taps(env):
    publish_all(env)
    client = synced(env.make_client())
    _show(client, "weather quad")
    taps = [[], [(5.0, 5.0)]]
    client.presenter.poll_taps = lambda: taps.pop(0) if taps else []
    client.wait(5)  # returns as soon as the tap arrives
    assert client.controls.focus == "weather1"


def test_hardware_presenter_reads_sdl_taps():
    from display.hardware_presenter import HardwarePresenter

    class Event:
        def __init__(self, type, **fields):
            self.type = type
            self.__dict__.update(fields)

    class Pygame:
        FINGERDOWN, MOUSEBUTTONDOWN = 1, 2

        class event:
            @staticmethod
            def get(types):
                return [Event(1, x=0.5, y=0.25), Event(2, button=1, pos=(200, 100)), Event(2, button=3, pos=(0, 0))]

    class Display:
        _pygame = Pygame
        render_width, render_height = 800, 480
        screen_width, screen_height = 400, 240

    taps = HardwarePresenter(Display(), profile=PROFILE).poll_taps()
    assert taps == [(400.0, 120.0), (400.0, 200.0)]
