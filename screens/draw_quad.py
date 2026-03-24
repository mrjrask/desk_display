"""Quad-view mode that renders four screens simultaneously in a 2x2 grid."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Optional

from PIL import Image, ImageDraw, ImageFont

from config import HEIGHT, SCREEN_DELAY, WIDTH
from utils import ScreenImage

RenderResult = Optional[Image.Image | ScreenImage | list[Image.Image]]
RenderFunc = Callable[[], RenderResult]


@dataclass
class _TileSpec:
    label: str
    render: RenderFunc


def _extract_images(rendered: RenderResult) -> list[Image.Image]:
    if isinstance(rendered, list):
        return [frame for frame in rendered if isinstance(frame, Image.Image)]
    if isinstance(rendered, ScreenImage):
        return [rendered.image]
    if isinstance(rendered, Image.Image):
        return [rendered]
    return []


def _error_tile(size: tuple[int, int], label: str) -> Image.Image:
    img = Image.new("RGB", size, (20, 20, 20))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=(65, 65, 65), width=1)
    draw.text((8, 8), label, fill=(220, 220, 220), font=font)
    draw.text((8, 22), "unavailable", fill=(170, 170, 170), font=font)
    return img


def draw_quad_screen(
    display,
    tiles: list[_TileSpec],
    *,
    transition: bool = False,
    scroll_speed: float = 1.0,
) -> ScreenImage:
    """Render a four-tile dashboard using the provided tile renderers."""

    cols = 2
    rows = 2
    tile_w = WIDTH // cols
    tile_h = HEIGHT // rows
    has_wait_for_skip = callable(getattr(display, "wait_for_skip", None))

    tile_sequences: list[list[Image.Image]] = []
    tile_regions: list[tuple[int, int, int, int]] = []

    for index in range(cols * rows):
        row = index // cols
        col = index % cols
        x0 = col * tile_w
        y0 = row * tile_h
        x1 = WIDTH if col == cols - 1 else (x0 + tile_w)
        y1 = HEIGHT if row == rows - 1 else (y0 + tile_h)
        region_size = (x1 - x0, y1 - y0)

        tile = tiles[index] if index < len(tiles) else None
        if tile is None:
            tile_imgs = [_error_tile(region_size, "empty")]
        else:
            try:
                sources = _extract_images(tile.render())
            except Exception:
                sources = []
            if sources:
                tile_imgs = [source.resize(region_size, Image.Resampling.LANCZOS) for source in sources]
            else:
                tile_imgs = [_error_tile(region_size, tile.label)]

        tile_sequences.append(tile_imgs)
        tile_regions.append((x0, y0, x1, y1))

    def _render_composite(frame_number: int) -> Image.Image:
        frame = Image.new("RGB", (WIDTH, HEIGHT), "black")
        draw = ImageDraw.Draw(frame)

        for index, seq in enumerate(tile_sequences):
            x0, y0, x1, y1 = tile_regions[index]
            tile_img = seq[frame_number % len(seq)]
            frame.paste(tile_img, (x0, y0))
            if x0 > 0:
                draw.line((x0, y0, x0, y1), fill=(70, 70, 70), width=1)
            if y0 > 0:
                draw.line((x0, y0, x1, y0), fill=(70, 70, 70), width=1)

        return frame

    max_frames = max((len(seq) for seq in tile_sequences), default=1)
    # Pace quad playback so animations stay smooth while still respecting
    # the overall screen window.
    # - Minimum frame time keeps CPU usage reasonable.
    # - Maximum frame time avoids very "choppy" updates when tiles only
    #   provide a handful of sampled frames.
    min_frame_time = 0.016  # ~60 FPS ceiling
    max_frame_time = 0.100  # ~10 FPS floor for visible smoothness
    speed_factor = max(0.25, min(3.0, float(scroll_speed)))
    target_frame_time = min(
        max_frame_time,
        max(min_frame_time, float(SCREEN_DELAY) / float(max_frames)) / speed_factor,
    )
    animated = transition and any(len(seq) > 1 for seq in tile_sequences)
    displayed_frame = _render_composite(0)
    display.image(displayed_frame)
    if transition:
        display.show()

    if animated:
        end_time = time.monotonic() + float(SCREEN_DELAY)
        frame_number = 1
        while time.monotonic() < end_time:
            frame_start = time.monotonic()
            displayed_frame = _render_composite(frame_number)
            display.image(displayed_frame)
            if transition:
                display.show()
            frame_number += 1

            elapsed = time.monotonic() - frame_start
            sleep_for = max(0.0, target_frame_time - elapsed)
            if sleep_for <= 0:
                continue
            if has_wait_for_skip and display.wait_for_skip(sleep_for):
                break
            if not has_wait_for_skip:
                time.sleep(sleep_for)

    return ScreenImage(displayed_frame, displayed=bool(transition), consumed_delay=animated)
