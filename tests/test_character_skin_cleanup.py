import numpy as np

from scripts.clean_character_skin_texture import (
    normalize_region,
    rasterize_uv_triangles,
)


def test_uv_rasterizer_flips_blender_v_axis():
    triangles = np.array(
        [[[0.0, 1.0], [0.5, 1.0], [0.0, 0.5]]],
        dtype=np.float32,
    )

    mask = rasterize_uv_triangles(triangles, width=32, height=32)

    assert mask[0, 0] == 255
    assert mask[-1, -1] == 0


def test_skin_normalization_reduces_color_variation():
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    image[:, :12] = [42, 58, 105]  # BGR warm dark skin
    image[:, 12:] = [72, 88, 155]
    region = np.full((24, 24), 255, dtype=np.uint8)

    cleaned, skin_mask, _ = normalize_region(image, region, strength=1.0)

    assert np.all(skin_mask == 255)
    assert cleaned.std(axis=(0, 1)).mean() < image.std(axis=(0, 1)).mean()
