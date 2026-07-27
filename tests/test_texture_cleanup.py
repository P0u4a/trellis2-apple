import numpy as np

from o_voxel.texture_cleanup import clean_base_color


def test_despeckle_replaces_only_strong_interior_outlier():
    image = np.full((9, 9, 3), 30, dtype=np.uint8)
    image[4, 4] = [240, 240, 240]
    mask = np.ones((9, 9), dtype=bool)

    cleaned, replaced = clean_base_color(image, mask, despeckle=1.0)

    assert replaced == 1
    np.testing.assert_array_equal(cleaned[4, 4], [30, 30, 30])


def test_despeckle_preserves_uv_boundary():
    image = np.full((9, 9, 3), 30, dtype=np.uint8)
    image[0, 0] = [240, 240, 240]
    mask = np.zeros((9, 9), dtype=bool)
    mask[:2, :2] = True

    cleaned, replaced = clean_base_color(image, mask, despeckle=1.0)

    assert replaced == 0
    np.testing.assert_array_equal(cleaned[0, 0], [240, 240, 240])
