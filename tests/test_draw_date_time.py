from screens import draw_date_time
from screens.draw_date_time import _color_cycle_profile


def test_color_cycle_profile_kernel_is_fast_and_continuous():
    initial_delay, interval, steps = _color_cycle_profile(
        kernel_driven=True,
        hyperpixel_layout=False,
        hyperpixel_square=False,
    )

    assert initial_delay == 0.18
    assert interval == 0.2
    assert steps is None


def test_color_cycle_profile_non_kernel_is_short_and_limited():
    initial_delay, interval, steps = _color_cycle_profile(
        kernel_driven=False,
        hyperpixel_layout=False,
        hyperpixel_square=False,
    )

    assert initial_delay == 0.6
    assert interval == 0.45
    assert steps == 6


def test_color_cycle_profile_hyperpixel_cycles_continuously():
    _, _, steps = _color_cycle_profile(
        kernel_driven=False,
        hyperpixel_layout=True,
        hyperpixel_square=False,
    )

    assert steps is None


def test_kernel_color_cycle_stops_when_frame_id_changes(monkeypatch):
    calls = {"compose": 0, "images": 0}

    class DriftingDisplay:
        def __init__(self):
            self._frame = 0

        def frame_id(self):
            self._frame += 1
            return self._frame

        def image(self, _img):
            calls["images"] += 1

    monkeypatch.setattr(
        draw_date_time,
        "_color_cycle_profile",
        lambda **_kwargs: (0.0, 0.0, 2),
    )
    monkeypatch.setattr(draw_date_time, "is_hyperpixel_next_layout", lambda: False)
    monkeypatch.setattr(draw_date_time, "is_hyperpixel_4_square_layout", lambda: False)
    monkeypatch.setattr(draw_date_time, "is_kernel_driven_display", lambda: True)
    monkeypatch.setattr(draw_date_time.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(draw_date_time, "bright_color", lambda: (255, 0, 0))

    def _fake_compose(*_args, **_kwargs):
        calls["compose"] += 1
        return object()

    monkeypatch.setattr(draw_date_time, "_compose_frame", _fake_compose)

    frame_state = {"value": 1, "lock": draw_date_time.threading.Lock()}
    draw_date_time._cycle_colors_after_load(
        DriftingDisplay(),
        "date_time",
        lambda: False,
        "date",
        frame_state,
    )

    assert calls["compose"] == 0
    assert calls["images"] == 0
