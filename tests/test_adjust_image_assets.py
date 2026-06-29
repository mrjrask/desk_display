from pathlib import Path

from PIL import Image

from tools.adjust_image_assets import adjust_images


def test_adjust_images_downscales_and_preserves_png_alpha(tmp_path: Path):
    image_path = tmp_path / "logo.png"
    Image.new("RGBA", (512, 256), (255, 0, 0, 128)).save(image_path)

    [result] = adjust_images([tmp_path], max_dimension=128)

    assert result.path == image_path
    assert result.original_size == (512, 256)
    assert result.adjusted_size == (128, 64)
    assert result.changed is True
    with Image.open(image_path) as adjusted:
        assert adjusted.size == (128, 64)
        assert adjusted.mode == "RGBA"
        assert adjusted.getpixel((0, 0))[3] == 128


def test_adjust_images_dry_run_does_not_rewrite(tmp_path: Path):
    image_path = tmp_path / "logo.png"
    Image.new("RGBA", (512, 256), (0, 0, 255, 255)).save(image_path)

    [result] = adjust_images([image_path], max_dimension=128, dry_run=True)

    assert result.adjusted_size == (128, 64)
    assert result.changed is True
    with Image.open(image_path) as adjusted:
        assert adjusted.size == (512, 256)
