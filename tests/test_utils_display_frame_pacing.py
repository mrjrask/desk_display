from PIL import Image

import utils


class _FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def _display_with_writer():
    display = utils.Display()
    writes = []

    def write_frame(frame):
        writes.append((utils.time.monotonic(), frame.copy()))

    display._frame_writer = write_frame
    return display, writes


def test_display_image_writes_are_capped_by_display_frame_interval(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(utils, "DISPLAY_FRAME_INTERVAL", 0.25)
    monkeypatch.setattr(utils.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(utils.time, "sleep", clock.sleep)

    display, writes = _display_with_writer()
    frame = Image.new("RGB", (display.width, display.height), "white")

    display.image(frame)
    display.image(frame)
    display.image(frame)

    assert [write[0] for write in writes] == [0.0, 0.25, 0.5]
    assert clock.sleeps == [0.25, 0.25]


def test_display_clear_bypasses_frame_pacing(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(utils, "DISPLAY_FRAME_INTERVAL", 1.0)
    monkeypatch.setattr(utils.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(utils.time, "sleep", clock.sleep)

    display, writes = _display_with_writer()
    frame = Image.new("RGB", (display.width, display.height), "white")

    display.image(frame)
    display.clear()

    assert [write[0] for write in writes] == [0.0, 0.0]
    assert clock.sleeps == []
