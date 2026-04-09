import datetime as dt

from PIL import Image

from screens import draw_nixie
from utils import ScreenImage


class _FakeDisplay:
    def __init__(self):
        self._frame_id = 0
        self.images = []

    def image(self, img):
        self.images.append(img)
        self._frame_id += 1

    def show(self):
        return None

    def frame_id(self):
        return self._frame_id


def test_nixie_frame_changes_by_second():
    now = dt.datetime(2025, 1, 1, 10, 11, 12)
    later = dt.datetime(2025, 1, 1, 10, 11, 13)

    frame_a = draw_nixie.nixie_frame(now)
    frame_b = draw_nixie.nixie_frame(later)

    assert frame_a.tobytes() != frame_b.tobytes()


def test_draw_nixie_starts_live_updates(monkeypatch):
    display = _FakeDisplay()
    seen = {"started": False, "frame_id": None}

    monkeypatch.setattr(draw_nixie, "clear_display", lambda d: None)
    monkeypatch.setattr(draw_nixie, "_play_flicker", lambda d, i: None)
    monkeypatch.setattr(draw_nixie, "_start_update_checks", lambda *args, **kwargs: None)

    def _capture(display_obj, *, expected_frame_id=None, frame_state=None):
        seen["started"] = True
        seen["frame_id"] = expected_frame_id

    monkeypatch.setattr(draw_nixie, "_start_live_updates", _capture)

    result = draw_nixie.draw_nixie(display, transition=False)

    assert isinstance(result, ScreenImage)
    assert result.displayed is True
    assert seen["started"] is True
    assert seen["frame_id"] == display.frame_id()
    assert isinstance(result.image, Image.Image)


def test_live_updates_keep_running_after_self_render(monkeypatch):
    display = _FakeDisplay()
    display._frame_id = 7

    monkeypatch.setattr(draw_nixie, "SCREEN_DELAY", 1)
    times = iter([0.0, 0.1, 0.2, 1.1])
    monkeypatch.setattr(draw_nixie.time, "monotonic", lambda: next(times, 1.1))
    monkeypatch.setattr(draw_nixie.time, "sleep", lambda _: None)

    now_values = iter([
        dt.datetime(2025, 1, 1, 10, 11, 12),
        dt.datetime(2025, 1, 1, 10, 11, 13),
    ])

    class _FakeDateTime:
        @staticmethod
        def now():
            return next(now_values)

    monkeypatch.setattr(draw_nixie.dt, "datetime", _FakeDateTime)
    monkeypatch.setattr(draw_nixie, "_compose_frame", lambda now=None, **kwargs: Image.new("RGB", (4, 4), "black"))

    class _ImmediateThread:
        def __init__(self, *, target, daemon):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(draw_nixie.threading, "Thread", _ImmediateThread)

    draw_nixie._start_live_updates(display, expected_frame_id=display.frame_id())

    # Worker should continue after writing its own frame and render both seconds.
    assert len(display.images) == 2


def test_live_updates_refresh_github_state_each_second(monkeypatch):
    display = _FakeDisplay()

    monkeypatch.setattr(draw_nixie, "SCREEN_DELAY", 1)
    times = iter([0.0, 0.1, 0.2, 1.1])
    monkeypatch.setattr(draw_nixie.time, "monotonic", lambda: next(times, 1.1))
    monkeypatch.setattr(draw_nixie.time, "sleep", lambda _: None)

    now_values = iter([
        dt.datetime(2025, 1, 1, 10, 11, 12),
        dt.datetime(2025, 1, 1, 10, 11, 13),
    ])

    class _FakeDateTime:
        @staticmethod
        def now():
            return next(now_values)

    monkeypatch.setattr(draw_nixie.dt, "datetime", _FakeDateTime)

    statuses = iter([False, True])

    class _Status:
        def __init__(self, github):
            self.github = github

    monkeypatch.setattr(draw_nixie, "get_update_status", lambda: _Status(next(statuses)))
    seen = []

    def _compose(now=None, *, gh_on=False):
        seen.append(gh_on)
        return Image.new("RGB", (4, 4), "black")

    monkeypatch.setattr(draw_nixie, "_compose_frame", _compose)

    class _ImmediateThread:
        def __init__(self, *, target, daemon):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(draw_nixie.threading, "Thread", _ImmediateThread)

    draw_nixie._start_live_updates(display, expected_frame_id=display.frame_id())

    assert seen == [False, True]


def test_update_checks_allow_self_advanced_frame_ids(monkeypatch):
    display = _FakeDisplay()
    display._frame_id = 8

    monkeypatch.setattr(draw_nixie, "check_github_updates", lambda: True)
    monkeypatch.setattr(draw_nixie, "check_apt_updates", lambda: None)
    monkeypatch.setattr(
        draw_nixie,
        "_compose_frame",
        lambda now=None, **kwargs: Image.new("RGB", (4, 4), "white"),
    )

    class _ImmediateThread:
        def __init__(self, *, target, daemon):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(draw_nixie.threading, "Thread", _ImmediateThread)

    frame_state = {"lock": draw_nixie.threading.Lock(), "value": 8}
    draw_nixie._start_update_checks(
        display,
        expected_frame_id=7,
        frame_state=frame_state,
    )

    assert len(display.images) == 1
