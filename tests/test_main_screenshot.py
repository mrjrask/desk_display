from PIL import Image

import main


def test_select_screenshot_image_prefers_full_scroll_capture():
    display_frame = Image.new("RGB", (320, 240), "black")
    full_scroll = Image.new("RGB", (320, 900), "blue")

    selected = main._select_screenshot_image(display_frame, full_scroll)

    assert selected is full_scroll
    assert selected.size == (320, 900)


def test_select_screenshot_image_falls_back_to_display_frame():
    display_frame = Image.new("RGB", (320, 240), "black")

    selected = main._select_screenshot_image(display_frame, None)

    assert selected is display_frame
    assert selected.size == (320, 240)
