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

    def _capture(display_obj, *, expected_frame_id=None):
        seen["started"] = True
        seen["frame_id"] = expected_frame_id

    monkeypatch.setattr(draw_nixie, "_start_live_updates", _capture)

    result = draw_nixie.draw_nixie(display, transition=False)

    assert isinstance(result, ScreenImage)
    assert result.displayed is True
    assert seen["started"] is True
    assert seen["frame_id"] == display.frame_id()
    assert isinstance(result.image, Image.Image)


def test_live_update_worker_keeps_running_after_own_renders(monkeypatch):
    display = _FakeDisplay()

    monkeypatch.setattr(draw_nixie, "SCREEN_DELAY", 1)
    monkeypatch.setattr(draw_nixie, "_compose_frame", lambda now=None: Image.new("RGB", (4, 4)))
    monkeypatch.setattr(draw_nixie.time, "sleep", lambda *_: None)

    monotonic_values = iter([0.0, 0.1, 0.2, 0.3, 1.1])
    monkeypatch.setattr(draw_nixie.time, "monotonic", lambda: next(monotonic_values))

    now_values = iter(
        [
            dt.datetime(2025, 1, 1, 0, 0, 0),
            dt.datetime(2025, 1, 1, 0, 0, 1),
            dt.datetime(2025, 1, 1, 0, 0, 1),
        ]
    )

    class _FakeDateTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return next(now_values)

    monkeypatch.setattr(draw_nixie.dt, "datetime", _FakeDateTime)

    class _InlineThread:
        def __init__(self, target, daemon):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(draw_nixie.threading, "Thread", _InlineThread)

    draw_nixie._start_live_updates(display, expected_frame_id=display.frame_id())

    assert len(display.images) == 2
