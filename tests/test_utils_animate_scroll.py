from PIL import Image

import utils


class _DisplayStub:
    width = 64
    height = 32

    def __init__(self):
        self.frames = []

    def image(self, frame):
        self.frames.append(frame.copy())


class _SkipDisplayStub(_DisplayStub):
    def __init__(self):
        super().__init__()
        self.skip = False

    def skip_requested(self):
        return self.skip

    def wait_for_skip(self, _timeout, *, poll_interval=0.05):
        _ = poll_interval
        self.skip = True
        return True


def test_animate_scroll_finishes_on_centered_logo(monkeypatch):
    display = _DisplayStub()
    logo = Image.new("RGB", (10, 10), (255, 0, 0))

    monkeypatch.setattr(utils.random, "choice", lambda _: "ltr")
    monkeypatch.setattr(utils.time, "sleep", lambda _: None)

    utils.animate_scroll(display, logo, speed=20)

    assert display.frames, "animate_scroll should render at least one frame"
    final = display.frames[-1]

    # The final frame should still contain logo pixels (not a fully black frame).
    assert final.getbbox() is not None


def test_animate_scroll_stops_immediately_when_skip_requested(monkeypatch):
    display = _SkipDisplayStub()
    logo = Image.new("RGB", (10, 10), (255, 0, 0))

    monkeypatch.setattr(utils.random, "choice", lambda _: "ltr")

    utils.animate_scroll(display, logo, speed=1)

    # A skip request should interrupt scrolling before a centered final frame is rendered.
    assert len(display.frames) == 1
