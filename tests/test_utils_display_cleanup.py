import threading

import utils


class _Closable:
    def __init__(self):
        self.close_count = 0

    def close(self):
        self.close_count += 1


def _display_for_cleanup():
    display = utils.Display.__new__(utils.Display)
    display._closed = False
    display._button_callback = lambda _name: None
    display._display_io_watchdog_stop = threading.Event()
    display._display_io_watchdog_thread = None
    display._gpio_buttons = {"A": _Closable()}
    display._display = _Closable()
    display._display_driver = "minipitft"
    display._retired_display_hat_mini = None
    display._minipitft_backlight = _Closable()
    display._kernel_display = _Closable()
    display._framebuffer = _Closable()
    display._display_io_lock = threading.RLock()
    display._frame_writer = lambda _image: None
    display._output_strategy = "minipitft"
    return display


def test_display_close_releases_backends_and_is_idempotent(monkeypatch):
    display = _display_for_cleanup()
    driver = display._display
    backlight = display._minipitft_backlight
    button = display._gpio_buttons["A"]
    kernel = display._kernel_display
    framebuffer = display._framebuffer
    monkeypatch.setattr(utils, "_ACTIVE_DISPLAY", display)

    display.close()
    display.close()

    assert display._closed is True
    assert display._button_callback is None
    assert button.close_count == 1
    assert driver.close_count == 1
    assert backlight.close_count == 1
    assert kernel.close_count == 1
    assert framebuffer.close_count == 1
    assert utils.get_active_display() is None


def test_kernel_display_close_quits_sdl_once():
    class _PygameDisplay:
        def __init__(self):
            self.quit_count = 0

        def quit(self):
            self.quit_count += 1

    class _Pygame:
        def __init__(self):
            self.display = _PygameDisplay()

    kernel = utils._KernelDisplay.__new__(utils._KernelDisplay)
    kernel._pygame = _Pygame()
    kernel._closed = False

    kernel.close()
    kernel.close()

    assert kernel._pygame.display.quit_count == 1
