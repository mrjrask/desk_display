from pathlib import Path

from PIL import Image

from scripts.adjust_image_assets import adjust_images, ensure_output_root


def test_adjust_images_downscales_and_preserves_png_alpha(tmp_path: Path):
    image_path = tmp_path / "logo.png"
    Image.new("RGBA", (512, 256), (255, 0, 0, 128)).save(image_path)

    output_root = tmp_path / "converted"
    [result] = adjust_images([tmp_path], input_root=tmp_path, output_root=output_root, max_dimension=128)

    assert result.path == image_path.resolve()
    assert result.output_path == output_root / "logo.png"
    assert result.original_size == (512, 256)
    assert result.adjusted_size == (128, 64)
    assert result.changed is True
    with Image.open(image_path) as original:
        assert original.size == (512, 256)
    with Image.open(result.output_path) as adjusted:
        assert adjusted.size == (128, 64)
        assert adjusted.mode == "RGBA"
        assert adjusted.getpixel((0, 0))[3] == 128


def test_adjust_images_dry_run_does_not_rewrite(tmp_path: Path):
    image_path = tmp_path / "logo.png"
    Image.new("RGBA", (512, 256), (0, 0, 255, 255)).save(image_path)

    output_root = tmp_path / "converted"
    [result] = adjust_images([image_path], input_root=tmp_path, output_root=output_root, max_dimension=128, dry_run=True)

    assert result.adjusted_size == (128, 64)
    assert result.changed is True
    assert not result.output_path.exists()
    with Image.open(image_path) as adjusted:
        assert adjusted.size == (512, 256)


def test_adjust_images_preserves_project_relative_folders(tmp_path: Path):
    input_root = tmp_path / "images"
    image_path = input_root / "teams" / "logo.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), (255, 255, 255)).save(image_path)
    output_root = tmp_path / "home" / "converted"

    [result] = adjust_images(output_root=output_root, input_root=input_root, max_dimension=128)

    assert result.output_path == output_root / "teams" / "logo.png"
    assert result.output_path.exists()


def test_output_root_must_not_be_inside_project(tmp_path: Path):
    project_root = tmp_path / "project"
    project_root.mkdir()

    try:
        ensure_output_root(project_root / "converted", project_root=project_root)
    except ValueError as exc:
        assert "outside the project folder" in str(exc)
    else:
        raise AssertionError("expected project-local output root to be rejected")
