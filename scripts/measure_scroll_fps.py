#!/usr/bin/env python3
"""Measure how smoothly this panel plays a scrolling screen.

Plays a synthetic scroll package (the same kind the server sends for
scoreboards and standings) through the client's real presenter and display
driver, and reports the frame rate and how evenly the picture moved. A smooth
scroll moves the same number of pixels every frame; "px per frame" spread
over several values is the judder you see.

The panel's SPI and GPIO lines belong to the running display service, so stop
it first::

    sudo systemctl stop desk_display_client.service
    python3 scripts/measure_scroll_fps.py            # current playback
    python3 scripts/measure_scroll_fps.py --legacy   # playback before the fix
    bash scripts/restart_services.sh

``--legacy`` reproduces the earlier client loop (wall-clock offsets, the frame
pushed twice, a full frame's sleep after each push) for a before/after
comparison on the same device.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    # Only re-exec when run directly, not when imported by the tests.
    try:
        from scripts._venv_bootstrap import reexec_with_project_venv
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _venv_bootstrap import reexec_with_project_venv
    reexec_with_project_venv()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from display_profiles import RenderProfile, resolve_display_profile_by_id  # noqa: E402
from playback.motion_clock import MotionClock  # noqa: E402
from playback.package_player import PackagePlayback  # noqa: E402
from remote_display.render_package import PackageBuilder  # noqa: E402


@dataclass
class ScrollStats:
    frames: int
    seconds: float
    intervals_ms: list[float]
    steps: Counter

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0

    def report(self) -> str:
        intervals = sorted(self.intervals_ms) or [0.0]
        p95 = intervals[min(len(intervals) - 1, int(len(intervals) * 0.95))]
        steps = ", ".join(f"{px}px x{count}" for px, count in sorted(self.steps.items()))
        return (f"{self.frames} frames in {self.seconds:.2f}s = {self.fps:.1f} fps\n"
                f"frame interval: mean {statistics.fmean(intervals):.1f} ms, p95 {p95:.1f} ms, "
                f"max {intervals[-1]:.1f} ms\n"
                f"px per frame: {steps or 'none'}")


def scroll_package(profile: RenderProfile, *, pages: int = 4) -> dict[str, Any]:
    """A striped canvas *pages* screens tall, paced like the profile's scoreboards."""

    width, height = profile.width, profile.height
    canvas = Image.new("RGB", (width, height * pages), "black")
    draw = ImageDraw.Draw(canvas)
    for y in range(0, canvas.height, 20):
        draw.rectangle((0, y, width - 1, y + 9), fill=(40 + (y * 7) % 200, 90, 160))
        draw.text((8, y), f"row {y // 20}", fill="white")
    builder = PackageBuilder()
    body = {"canvas": builder.add(canvas), "viewport": [width, height],
            "step_px": profile.scoreboard_scroll_step, "frame_seconds": profile.scoreboard_scroll_delay,
            "pause_start_seconds": 0.0, "pause_end_seconds": 0.0, "direction": "down"}
    return builder.build(screen_id="scroll test", render_profile=profile.profile_id, width=width,
                         height=height, color_mode="RGB", render_key_digest="0" * 64,
                         classification="scrolling_canvas", kind="scroll", body=body)


def measure(playback: PackagePlayback, present: Callable[[Image.Image], Any], *, legacy: bool = False,
            show: Callable[[], Any] | None = None, clock: Callable[[], float] = time.monotonic,
            sleep: Callable[[float], Any] = time.sleep) -> ScrollStats:
    """Play *playback*'s motion as the client does and time each pushed frame."""

    frame_seconds = playback.frame_seconds
    started = clock()
    motion = MotionClock(0.0, started, max_lag=playback.motion_seconds)
    last_key = playback.key_at(0.0)
    present(playback.frame_at(0.0))
    last_offset = last_key[1] if isinstance(last_key, tuple) else 0
    pushed_at: list[float] = []
    steps: Counter = Counter()
    while True:
        now = clock()
        t = (now - started) if legacy else motion.tick(now, frame_seconds)
        if t >= playback.motion_seconds and playback.key_at(t) == last_key:
            break
        key = playback.key_at(t)
        drawn = key != last_key
        if not legacy:
            motion.drew(drawn)
        interval = min(0.05, frame_seconds)
        if drawn:
            present(playback.frame_at(t))
            if legacy and show is not None:
                show()
            pushed_at.append(clock())
            offset = key[1]
            steps[abs(offset - last_offset)] += 1
            last_key, last_offset = key, offset
            if not legacy:
                interval = min(0.05, max(0.0, now + frame_seconds - clock()))
        sleep(interval)
    intervals = [(b - a) * 1000 for a, b in zip(pushed_at, pushed_at[1:], strict=False)]
    seconds = pushed_at[-1] - started if pushed_at else 0.0
    return ScrollStats(len(pushed_at), seconds, intervals, steps)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--legacy", action="store_true", help="use the playback loop from before the fix")
    parser.add_argument("--pages", type=int, default=4, help="canvas height in screens (default 4)")
    parser.add_argument("--profile", help="display profile id (default: DESK_DISPLAY_PROFILE from .env.client)")
    args = parser.parse_args(argv)

    import deployment_config
    import display_client

    display_client.prepare_environment()
    settings = deployment_config.load_settings(deployment_config.Role.CLIENT)
    profile_id = args.profile or settings.get("DESK_DISPLAY_PROFILE")
    profile = resolve_display_profile_by_id(profile_id) if profile_id else None
    if profile is None:
        print(f"Unknown display profile {profile_id!r}; pass --profile", file=sys.stderr)
        return 2
    from rendering.profile_process import configure_native

    configure_native(profile)
    from display.hardware_presenter import HardwarePresenter

    presenter = HardwarePresenter(profile=profile)
    display = presenter.display
    try:
        playback = PackagePlayback(scroll_package(profile, pages=args.pages), profile, hold_seconds=0)
        show = getattr(display, "show", None)
        if args.legacy:
            # The old presenter always called show() after image(), pushing twice.
            def present(image: Image.Image) -> None:
                display.image(image.resize((profile.width, profile.height)).convert(profile.color_mode))
        else:
            present = presenter.present
        stats = measure(playback, present, legacy=args.legacy, show=show)
    finally:
        presenter.close()
    label = "legacy playback" if args.legacy else "current playback"
    print(f"{profile.profile_id} {profile.width}x{profile.height}, {label}; "
          f"package asks {1 / playback.frame_seconds:.0f} fps at {profile.scoreboard_scroll_step}px per frame")
    print(stats.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
