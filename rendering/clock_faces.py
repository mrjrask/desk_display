"""Client-timed clock faces (the ``date`` and ``nixie`` screens).

A clock rendered on the server is wrong a minute later, and wrong for as
long as the client is offline. So in remote mode the server sends a clock
*package*: the face's background and a small layout, and the client draws
the current time itself with :func:`render_clock`, using the same composers
the standalone screens use. The client supplies its own IP address for the
optional overlay; the server's address and update indicator never appear.
"""
from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from PIL import Image

from display_profiles import RenderProfile

CLOCK_FACES = {"date": "date", "nixie": "nixie"}
LAYOUT_KEYS = frozenset({"face", "time_zone", "time_format", "show_ip", "background_color"})


def _invoke(func, profile: RenderProfile, *args: Any, **kwargs: Any) -> Any:
    from screens.registry import _invoke_for_profile

    return _invoke_for_profile(func, profile, *args, **kwargs)


def clock_layout(screen_id: str, profile: RenderProfile) -> dict[str, Any]:
    """The layout the server ships for *screen_id*, from its own settings."""

    import os

    import config

    face = CLOCK_FACES[screen_id]
    fmt = os.environ.get("NIXIE_TIME_FORMAT", "12").strip() if face == "nixie" else "12"
    color = config.get_screen_background_color(screen_id, (0, 0, 0))
    return {
        "face": face,
        "time_zone": _zone_name(config.CENTRAL_TIME),
        "time_format": fmt if fmt in ("12", "24") else "12",
        "show_ip": bool(config.IP_WITH_TIME),
        "background_color": [int(c) for c in tuple(color)[:3]],
    }


def _zone_name(tz: Any) -> str:
    zone = getattr(tz, "_zone", tz)
    return str(getattr(zone, "key", None) or "America/Chicago")


def clock_background(layout: Mapping[str, Any], profile: RenderProfile) -> Image.Image:
    color = tuple(int(c) for c in layout.get("background_color") or (0, 0, 0))
    return Image.new("RGB", (profile.width, profile.height), color).convert(profile.color_mode)


def _local(now: dt.datetime, zone: str) -> dt.datetime:
    try:
        tz = ZoneInfo(zone)
    except (ZoneInfoNotFoundError, ValueError):
        return now
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    return now.astimezone(tz)


def render_clock(
    layout: Mapping[str, Any],
    profile: RenderProfile,
    now: dt.datetime,
    *,
    ip_text: str | None = None,
    colors: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
) -> Image.Image:
    """Draw the face for *now*; *ip_text* is the client's own address label."""

    now = _local(now, str(layout.get("time_zone") or "UTC"))
    show_ip = bool(layout.get("show_ip")) and ip_text is not None
    if layout.get("face") == "nixie":
        from screens.draw_nixie import _compose_frame

        image = _invoke(_compose_frame, profile, now, gh_on=False, time_format=layout.get("time_format"),
                        show_ip=show_ip, ip_text=ip_text)
    else:
        from screens.draw_date_time import _compose_frame
        from utils import bright_color

        top, bottom = colors or (bright_color(), bright_color())
        image = _invoke(_compose_frame, profile, "date_time", top, bottom, False, "date",
                        now=now, show_ip=show_ip, ip_text=ip_text)
    if image.size != (profile.width, profile.height):
        image = image.resize((profile.width, profile.height))
    return image.convert(profile.color_mode)


__all__ = ["CLOCK_FACES", "LAYOUT_KEYS", "clock_background", "clock_layout", "render_clock"]
