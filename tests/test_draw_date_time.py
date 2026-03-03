from screens import draw_date_time
from screens.draw_date_time import _color_cycle_profile


def test_color_cycle_profile_display_hat_mini_kernel_is_fast_and_continuous():
    initial_delay, interval, steps = _color_cycle_profile(
        kernel_driven=True,
        display_profile_id="display_hat_mini",
        hyperpixel_layout=False,
        hyperpixel_square=False,
    )

    assert initial_delay == 0.0
    assert interval == 0.08
    assert steps is None


def test_color_cycle_profile_non_kernel_is_rapid_and_limited():
    initial_delay, interval, steps = _color_cycle_profile(
        kernel_driven=False,
        display_profile_id="display_hat_mini",
        hyperpixel_layout=False,
        hyperpixel_square=False,
    )

    assert initial_delay == 0.0
    assert interval == 0.08
    assert steps == 60


def test_color_cycle_profile_hyperpixel_cycles_continuously():
    initial_delay, interval, steps = _color_cycle_profile(
        kernel_driven=False,
        display_profile_id="hyperpixel4",
        hyperpixel_layout=True,
        hyperpixel_square=False,
    )

    assert initial_delay == 0.0
    assert interval == 0.08
    assert steps is None


def test_color_cycle_profile_kernel_non_display_hat_is_not_rapid():
    initial_delay, interval, steps = _color_cycle_profile(
        kernel_driven=True,
        display_profile_id="hdmi_1080p",
        hyperpixel_layout=False,
        hyperpixel_square=False,
    )

    assert initial_delay == 0.0
    assert interval == 0.08
    assert steps is None


def test_color_cycle_reconciles_startup_frame_race(monkeypatch):
    calls = {"compose": 0, "images": 0}

    class OneFrameDisplay:
        def __init__(self):
            self._frame = 2

        def frame_id(self):
            return self._frame

        def image(self, _img):
            calls["images"] += 1
            self._frame += 1

    monkeypatch.setattr(
        draw_date_time,
        "_color_cycle_profile",
        lambda **_kwargs: (0.0, 0.0, 1),
    )
    monkeypatch.setattr(draw_date_time, "is_hyperpixel_next_layout", lambda: False)
    monkeypatch.setattr(draw_date_time, "is_hyperpixel_4_square_layout", lambda: False)
    monkeypatch.setattr(draw_date_time, "is_kernel_driven_display", lambda: False)
    monkeypatch.setattr(draw_date_time.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(draw_date_time, "bright_color", lambda: (255, 0, 0))

    def _fake_compose(*_args, **_kwargs):
        calls["compose"] += 1
        return object()

    monkeypatch.setattr(draw_date_time, "_compose_frame", _fake_compose)

    frame_state = {"value": 1, "lock": draw_date_time.threading.Lock()}
    draw_date_time._cycle_colors_after_load(
        OneFrameDisplay(),
        "date_time",
        lambda: False,
        "date",
        frame_state,
    )

    assert calls["compose"] == 1
    assert calls["images"] == 1


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


def test_kernel_color_cycle_ignores_transient_frame_id_race(monkeypatch):
    calls = {"compose": 0, "images": 0}

    class LaggingFrameDisplay:
        def __init__(self):
            self._frame = 1

        def frame_id(self):
            return self._frame

        def image(self, _img):
            calls["images"] += 1
            self._frame += 1

    monkeypatch.setattr(
        draw_date_time,
        "_color_cycle_profile",
        lambda **_kwargs: (0.0, 0.0, 1),
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

    # Simulate frame_state already being advanced by a sibling writer while
    # display.frame_id() still momentarily reports the previous frame.
    frame_state = {"value": 2, "lock": draw_date_time.threading.Lock()}
    draw_date_time._cycle_colors_after_load(
        LaggingFrameDisplay(),
        "date_time",
        lambda: False,
        "date",
        frame_state,
    )

    assert calls["compose"] == 1
    assert calls["images"] == 1


def test_color_cycle_ignores_display_ahead_of_shared_frame_state(monkeypatch):
    calls = {"compose": 0, "images": 0}

    class AheadOfStateDisplay:
        def __init__(self):
            self._frame = 1
            self._first_loop_read = True

        def frame_id(self):
            if self._first_loop_read:
                self._first_loop_read = False
                # Simulate a sibling writer updating the display first.
                return 2
            return self._frame

        def image(self, _img):
            calls["images"] += 1
            self._frame += 1

    monkeypatch.setattr(
        draw_date_time,
        "_color_cycle_profile",
        lambda **_kwargs: (0.0, 0.0, 1),
    )
    monkeypatch.setattr(draw_date_time, "is_hyperpixel_next_layout", lambda: False)
    monkeypatch.setattr(draw_date_time, "is_hyperpixel_4_square_layout", lambda: False)
    monkeypatch.setattr(draw_date_time, "is_kernel_driven_display", lambda: False)
    monkeypatch.setattr(draw_date_time.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(draw_date_time, "bright_color", lambda: (255, 0, 0))

    def _fake_compose(*_args, **_kwargs):
        calls["compose"] += 1
        return object()

    monkeypatch.setattr(draw_date_time, "_compose_frame", _fake_compose)

    # Simulate the shared frame state catching up after display.frame_id().
    frame_state = {"value": 1, "lock": draw_date_time.threading.Lock()}
    with frame_state["lock"]:
        frame_state["value"] = 2

    draw_date_time._cycle_colors_after_load(
        AheadOfStateDisplay(),
        "date_time",
        lambda: False,
        "date",
        frame_state,
    )

    assert calls["compose"] == 1
    assert calls["images"] == 1


def test_color_cycle_ignores_single_transient_takeover_observation(monkeypatch):
    calls = {"compose": 0, "images": 0}

    class SingleTransientLeadDisplay:
        def __init__(self):
            self._frame = 1
            self._reads = 0

        def frame_id(self):
            self._reads += 1
            # First loop read appears ahead of expected frame.
            if self._reads == 1:
                return 2
            return self._frame

        def image(self, _img):
            calls["images"] += 1
            self._frame += 1

    monkeypatch.setattr(
        draw_date_time,
        "_color_cycle_profile",
        lambda **_kwargs: (0.0, 0.0, 1),
    )
    monkeypatch.setattr(draw_date_time, "is_hyperpixel_next_layout", lambda: False)
    monkeypatch.setattr(draw_date_time, "is_hyperpixel_4_square_layout", lambda: False)
    monkeypatch.setattr(draw_date_time, "is_kernel_driven_display", lambda: False)
    monkeypatch.setattr(draw_date_time.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(draw_date_time, "bright_color", lambda: (255, 0, 0))

    def _fake_compose(*_args, **_kwargs):
        calls["compose"] += 1
        return object()

    monkeypatch.setattr(draw_date_time, "_compose_frame", _fake_compose)

    frame_state = {"value": 1, "lock": draw_date_time.threading.Lock()}
    draw_date_time._cycle_colors_after_load(
        SingleTransientLeadDisplay(),
        "date_time",
        lambda: False,
        "date",
        frame_state,
    )

    assert calls["compose"] == 1
    assert calls["images"] == 1


def test_color_cycle_calls_show_when_display_requires_flush(monkeypatch):
    calls = {"images": 0, "shows": 0}

    class FlushDisplay:
        def __init__(self):
            self._frame = 1

        def frame_id(self):
            return self._frame

        def image(self, _img):
            calls["images"] += 1
            self._frame += 1

        def show(self):
            calls["shows"] += 1

    monkeypatch.setattr(
        draw_date_time,
        "_color_cycle_profile",
        lambda **_kwargs: (0.0, 0.0, 1),
    )
    monkeypatch.setattr(draw_date_time, "is_hyperpixel_next_layout", lambda: False)
    monkeypatch.setattr(draw_date_time, "is_hyperpixel_4_square_layout", lambda: False)
    monkeypatch.setattr(draw_date_time, "is_kernel_driven_display", lambda: False)
    monkeypatch.setattr(draw_date_time.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(draw_date_time, "bright_color", lambda: (255, 0, 0))
    monkeypatch.setattr(draw_date_time, "_compose_frame", lambda *_args, **_kwargs: object())

    frame_state = {"value": 1, "lock": draw_date_time.threading.Lock()}
    draw_date_time._cycle_colors_after_load(
        FlushDisplay(),
        "date_time",
        lambda: False,
        "date",
        frame_state,
    )

    assert calls["images"] == 1
    assert calls["shows"] == 1
