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
