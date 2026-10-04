"""utils.Display.image: one private copy per frame, with the bottom strip cleared."""

import pytest

import utils


@pytest.mark.parametrize("kernel, border, rows", [(False, False, 5), (True, False, 25), (True, True, 5)])
def test_image_clears_the_bottom_strip_without_touching_the_callers_frame(monkeypatch, kernel, border, rows):
    display = utils.Display()
    display._uses_kernel_output = kernel
    display._indicator_border_enabled = border
    frame = utils.Image.new("RGB", (display.width, display.height), (200, 100, 50))
    try:
        display.image(frame)
        shown = display.capture()
        assert shown.getpixel((0, display.height - rows)) == (0, 0, 0)
        assert shown.getpixel((0, display.height - rows - 1)) == (200, 100, 50)
        assert frame.getpixel((0, display.height - 1)) == (200, 100, 50)  # caller's frame untouched
        assert display._buffer is not frame
    finally:
        display.close()


def test_image_copies_each_frame_once(monkeypatch):
    display = utils.Display()
    frame = utils.Image.new("RGB", (display.width, display.height), "white")
    copies = []
    real_copy = utils.Image.Image.copy

    def counting_copy(self):
        copies.append(self)
        return real_copy(self)

    monkeypatch.setattr(utils.Image.Image, "copy", counting_copy)
    try:
        display.image(frame)
        assert len(copies) == 1
        # A frame that needs converting is already a new image: no copy at all.
        display.image(frame.convert("RGBA"))
        assert len(copies) == 1
    finally:
        monkeypatch.undo()
        display.close()
