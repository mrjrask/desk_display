#!/usr/bin/env python3
"""Downscale and optimize images under the repo images directory.

The tool is intentionally source-controlled instead of committing rewritten binary
assets from an agent run. Run it locally when you want to resize oversized image
assets, preserving transparency for formats that support alpha.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

from PIL import Image

DEFAULT_IMAGE_ROOT = Path("images")
DEFAULT_MAX_DIMENSION = 256
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class ImageAdjustment:
    path: Path
    original_size: tuple[int, int]
    adjusted_size: tuple[int, int]
    original_bytes: int
    adjusted_bytes: int
    changed: bool


def iter_images(paths: Iterable[Path]) -> Iterable[Path]:
    """Yield supported image files from explicit files or recursive directories."""
    for path in paths:
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
        elif path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                    yield candidate


def _target_size(size: tuple[int, int], max_dimension: int) -> tuple[int, int]:
    width, height = size
    if max(width, height) <= max_dimension:
        return size
    ratio = max_dimension / float(max(width, height))
    return max(1, round(width * ratio)), max(1, round(height * ratio))


def _save_kwargs(path: Path) -> dict[str, object]:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return {"format": "PNG", "optimize": True}
    if suffix in {".jpg", ".jpeg"}:
        return {"format": "JPEG", "quality": 85, "optimize": True, "progressive": True}
    if suffix == ".webp":
        return {"format": "WEBP", "quality": 85, "method": 6}
    return {}


def adjust_image(path: Path, *, max_dimension: int, dry_run: bool = False) -> ImageAdjustment:
    """Resize a single image if needed and optimize it for storage."""
    original_bytes = path.stat().st_size
    with Image.open(path) as image:
        original_size = image.size
        adjusted_size = _target_size(original_size, max_dimension)
        adjusted = image
        if adjusted_size != original_size:
            adjusted = image.resize(adjusted_size, Image.Resampling.LANCZOS)

        save_kwargs = _save_kwargs(path)
        if path.suffix.lower() in {".jpg", ".jpeg"} and adjusted.mode in {"RGBA", "LA", "P"}:
            adjusted = adjusted.convert("RGB")

        if dry_run:
            buffer = BytesIO()
            adjusted.save(buffer, **save_kwargs)
            adjusted_bytes = buffer.tell()
        else:
            adjusted.save(path, **save_kwargs)
            adjusted_bytes = path.stat().st_size

    changed = adjusted_size != original_size or adjusted_bytes != original_bytes
    return ImageAdjustment(
        path=path,
        original_size=original_size,
        adjusted_size=adjusted_size,
        original_bytes=original_bytes,
        adjusted_bytes=adjusted_bytes,
        changed=changed,
    )


def adjust_images(
    paths: Iterable[Path],
    *,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    dry_run: bool = False,
) -> list[ImageAdjustment]:
    """Resize and optimize all supported images under the supplied paths."""
    return [
        adjust_image(path, max_dimension=max_dimension, dry_run=dry_run)
        for path in iter_images(paths)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[DEFAULT_IMAGE_ROOT])
    parser.add_argument("--max-dimension", type=int, default=DEFAULT_MAX_DIMENSION)
    parser.add_argument("--dry-run", action="store_true", help="Report changes without rewriting files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = adjust_images(args.paths, max_dimension=args.max_dimension, dry_run=args.dry_run)
    changed = [result for result in results if result.changed]
    for result in changed:
        print(
            f"{result.path}: {result.original_size[0]}x{result.original_size[1]} "
            f"({result.original_bytes} bytes) -> "
            f"{result.adjusted_size[0]}x{result.adjusted_size[1]} "
            f"({result.adjusted_bytes} bytes)"
        )
    print(f"Scanned {len(results)} image(s); {'would change' if args.dry_run else 'changed'} {len(changed)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
