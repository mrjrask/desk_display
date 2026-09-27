"""The per-profile render workers of rendering.profile_process."""
from __future__ import annotations

import sys
import time

import pytest

from display_profiles import PROFILE_PRESETS
from remote_display.models import RenderKey, ScreenRevisions
from rendering.profile_process import ProfileProcessError, ProfileProcessPool
from services.data_coordinator import DataCoordinator

PROFILE = PROFILE_PRESETS["display_hat_mini"]


def key(screen_id):
    return RenderKey.for_screen(screen_id, PROFILE.profile_id, ScreenRevisions("s", "d", "r"))


@pytest.fixture
def pool():
    workers = ProfileProcessPool()
    yield workers
    workers.close()


def test_a_render_error_in_a_worker_reaches_the_caller(pool):
    # No indoor sensor here, so the screen is unavailable and the render raises.
    with pytest.raises(KeyError):
        pool.render_screen(key("inside"), PROFILE, DataCoordinator().snapshot())


def test_input_that_cannot_be_sent_fails_that_render_only(pool):
    data = DataCoordinator()
    data.publish("weather", {"callback": lambda: None})
    with pytest.raises(ProfileProcessError, match="cannot be sent"):
        pool.render_screen(key("date"), PROFILE, data.snapshot())
    # The worker still has no snapshot, so the next render sends one.
    data.publish("weather", {})
    image = pool.render_screen(key("AL Overview"), PROFILE, data.snapshot())["image"]
    assert image.size == (PROFILE.width, PROFILE.height)


def test_a_worker_that_hangs_is_killed_and_replaced(tmp_path):
    hang = tmp_path / "python"
    hang.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
    hang.chmod(0o755)
    workers = ProfileProcessPool(python=str(hang), timeout_seconds=1)
    try:
        started = time.monotonic()
        with pytest.raises(ProfileProcessError, match="did not start"):
            workers.render_screen(key("date"), PROFILE, DataCoordinator().snapshot())
        assert time.monotonic() - started < 30
    finally:
        workers.close()
