#!/usr/bin/env python3
"""Convert image assets that violate the repository size policy.

The script scans the project's ``images`` folder by default, finds only images
whose dimensions or file size exceed the limits used by ``check_image_assets.py``,
and writes resized/optimized copies to a new folder in the current user's home
directory. Source images are never modified.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

from PIL import Image

from check_image_assets import IMAGE_ROOT, PROJECT_ROOT, SUPPORTED_EXTENSIONS, policy_for

DEFAULT_OUTPUT_ROOT = Path.home() / f"{PROJECT_ROOT.name}_corrected_images"


@dataclass(frozen=True)
class ConvertedImage:
    """Summary of a converted image asset."""

    source_path: Path
    output_path: Path
    original_size: tuple[int, int]
    converted_size: tuple[int, int]
    original_bytes: int
    converted_bytes: int


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def ensure_output_root(output_root: Path, *, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve a safe output folder outside the project tree."""
    resolved_output = _resolve(output_root)
    resolved_project = _resolve(project_root)
    if resolved_output == resolved_project or _is_relative_to(resolved_output, resolved_project):
        raise ValueError(f"Output folder must be outside the project folder: {resolved_output}")
    return resolved_output


def iter_images(paths: Iterable[Path]) -> Iterable[Path]:
    """Yield supported image files from explicit files or recursive folders."""
    for path in paths:
        resolved_path = _resolve(path)
        if resolved_path.is_file() and resolved_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield resolved_path
        elif resolved_path.is_dir():
            for candidate in sorted(resolved_path.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                    yield candidate


def image_exceeds_policy(path: Path) -> bool:
    """Return whether an image exceeds its configured dimensions or byte limit."""
    policy = policy_for(path)
    if policy is None:
        return False

    with Image.open(path) as image:
        width, height = image.size
    return max(width, height) > policy.max_dimension or path.stat().st_size > policy.max_bytes


def output_path_for(path: Path, *, input_root: Path, output_root: Path) -> Path:
    """Map a source image path to the matching output path."""
    source = _resolve(path)
    root = _resolve(input_root)
    try:
        relative_path = source.relative_to(root)
    except ValueError:
        relative_path = Path(source.name)
    return output_root / relative_path


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


def convert_image(path: Path, *, input_root: Path, output_root: Path, dry_run: bool = False) -> ConvertedImage:
    """Resize and optimize a single image into the output folder."""
    source_path = _resolve(path)
    policy = policy_for(source_path)
    if policy is None:
        raise ValueError(f"No image asset policy applies to {source_path}")

    output_path = output_path_for(source_path, input_root=input_root, output_root=output_root)
    original_bytes = source_path.stat().st_size
    with Image.open(source_path) as image:
        original_size = image.size
        converted_size = _target_size(original_size, policy.max_dimension)
        converted = image.resize(converted_size, Image.Resampling.LANCZOS) if converted_size != original_size else image
        if source_path.suffix.lower() in {".jpg", ".jpeg"} and converted.mode in {"RGBA", "LA", "P"}:
            converted = converted.convert("RGB")

        buffer = BytesIO()
        converted.save(buffer, **_save_kwargs(source_path))
        converted_bytes = buffer.tell()
        if not dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(buffer.getvalue())

    return ConvertedImage(
        source_path=source_path,
        output_path=output_path,
        original_size=original_size,
        converted_size=converted_size,
        original_bytes=original_bytes,
        converted_bytes=converted_bytes,
    )


def convert_incorrectly_sized_images(
    paths: Iterable[Path] | None = None,
    *,
    input_root: Path = IMAGE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    dry_run: bool = False,
) -> list[ConvertedImage]:
    """Convert only images that violate the repository image asset policy."""
    resolved_input_root = _resolve(input_root)
    resolved_output_root = ensure_output_root(output_root)
    scan_paths = list(paths) if paths is not None else [resolved_input_root]
    oversized_images = [path for path in iter_images(scan_paths) if image_exceeds_policy(path)]
    return [
        convert_image(path, input_root=resolved_input_root, output_root=resolved_output_root, dry_run=dry_run)
        for path in oversized_images
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional image files or folders to scan; defaults to the project's images folder.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Folder for converted images; defaults to {DEFAULT_OUTPUT_ROOT}.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report conversions without writing files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = convert_incorrectly_sized_images(args.paths or None, output_root=args.output_root, dry_run=args.dry_run)
    verb = "Would write" if args.dry_run else "Wrote"
    for result in results:
        print(
            f"{verb} {result.output_path}: {result.original_size[0]}x{result.original_size[1]} "
            f"({result.original_bytes} bytes) -> {result.converted_size[0]}x{result.converted_size[1]} "
            f"({result.converted_bytes} bytes) from {result.source_path}"
        )
    print(f"Converted {len(results)} incorrectly sized image(s) into {ensure_output_root(args.output_root)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
