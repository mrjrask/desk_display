#!/usr/bin/env python3
"""Downscale and optimize project images into a home-directory output folder.

The tool is intentionally source-controlled instead of committing rewritten binary
assets from an agent run. Run it locally when you want to resize oversized image
assets. Converted files are written outside the project tree while preserving the
same relative organization as the project's ``images`` directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    try:
        from scripts._venv_bootstrap import reexec_with_project_venv
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _venv_bootstrap import reexec_with_project_venv
    reexec_with_project_venv()

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "images"
DEFAULT_OUTPUT_ROOT = Path.home() / f"{PROJECT_ROOT.name}_converted_images"
DEFAULT_MAX_DIMENSION = 128
LEAGUE_LOGO_MAX_DIMENSION = 160
LEAGUE_LOGO_NAMES = {
    "afc",
    "ahl",
    "al",
    "mlb",
    "nba",
    "nfc",
    "nfl",
    "nhl",
    "nl",
    "oly",
    "sb",
    "scp",
    "wc",
}
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class ImageAdjustment:
    path: Path
    output_path: Path
    original_size: tuple[int, int]
    adjusted_size: tuple[int, int]
    original_bytes: int
    adjusted_bytes: int
    changed: bool


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def ensure_output_root(output_root: Path, *, project_root: Path = PROJECT_ROOT) -> Path:
    """Return a safe output directory, refusing locations inside the project."""
    resolved_output = _resolve(output_root)
    resolved_project = _resolve(project_root)
    if resolved_output == resolved_project or _is_relative_to(resolved_output, resolved_project):
        raise ValueError(f"Output folder must be outside the project folder: {resolved_output}")
    return resolved_output


def iter_images(paths: Iterable[Path]) -> Iterable[Path]:
    """Yield supported image files from explicit files or recursive directories."""
    for path in paths:
        path = _resolve(path)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
        elif path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                    yield candidate


def max_dimension_for(path: Path, default: int = DEFAULT_MAX_DIMENSION) -> int:
    """Return the canonical maximum edge length for an image asset."""
    if path.stem.lower() in LEAGUE_LOGO_NAMES:
        return LEAGUE_LOGO_MAX_DIMENSION
    return default


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


def output_path_for(path: Path, *, input_root: Path, output_root: Path) -> Path:
    """Map a project image path to the matching output path."""
    source = _resolve(path)
    root = _resolve(input_root)
    try:
        relative_path = source.relative_to(root)
    except ValueError:
        relative_path = Path(source.name)
    return output_root / relative_path


def adjust_image(
    path: Path,
    *,
    output_path: Path,
    max_dimension: int,
    dry_run: bool = False,
) -> ImageAdjustment:
    """Resize a single image if needed and write the optimized copy to output_path."""
    source_path = _resolve(path)
    original_bytes = source_path.stat().st_size
    with Image.open(source_path) as image:
        original_size = image.size
        adjusted_size = _target_size(original_size, max_dimension)
        adjusted = image
        if adjusted_size != original_size:
            adjusted = image.resize(adjusted_size, Image.Resampling.LANCZOS)

        save_kwargs = _save_kwargs(source_path)
        if source_path.suffix.lower() in {".jpg", ".jpeg"} and adjusted.mode in {"RGBA", "LA", "P"}:
            adjusted = adjusted.convert("RGB")

        buffer = BytesIO()
        adjusted.save(buffer, **save_kwargs)
        adjusted_bytes = buffer.tell()
        if not dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(buffer.getvalue())

    changed = adjusted_size != original_size or adjusted_bytes != original_bytes
    return ImageAdjustment(
        path=source_path,
        output_path=output_path,
        original_size=original_size,
        adjusted_size=adjusted_size,
        original_bytes=original_bytes,
        adjusted_bytes=adjusted_bytes,
        changed=changed,
    )


def adjust_images(
    paths: Iterable[Path] | None = None,
    *,
    input_root: Path = DEFAULT_IMAGE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    dry_run: bool = False,
) -> list[ImageAdjustment]:
    """Resize and optimize all supported images into output_root."""
    resolved_input_root = _resolve(input_root)
    resolved_output_root = ensure_output_root(output_root)
    scan_paths = list(paths) if paths is not None else [resolved_input_root]
    return [
        adjust_image(
            path,
            output_path=output_path_for(path, input_root=resolved_input_root, output_root=resolved_output_root),
            max_dimension=max_dimension_for(path, max_dimension),
            dry_run=dry_run,
        )
        for path in iter_images(scan_paths)
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
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=DEFAULT_MAX_DIMENSION,
        help="Default max edge length for team logos and flags; league logos use 160 px.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing converted files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = adjust_images(
        args.paths or None,
        output_root=args.output_root,
        max_dimension=args.max_dimension,
        dry_run=args.dry_run,
    )
    changed = [result for result in results if result.changed]
    for result in results:
        marker = "would write" if args.dry_run else "wrote"
        change_marker = "converted" if result.changed else "optimized copy"
        print(
            f"{result.path} -> {result.output_path}: {result.original_size[0]}x{result.original_size[1]} "
            f"({result.original_bytes} bytes) -> "
            f"{result.adjusted_size[0]}x{result.adjusted_size[1]} "
            f"({result.adjusted_bytes} bytes); {marker} {change_marker}"
        )
    print(f"Scanned {len(results)} image(s); {'would change' if args.dry_run else 'changed'} {len(changed)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
