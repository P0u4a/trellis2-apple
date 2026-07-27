import torch

from trellis2.pipelines.trellis2_image_to_3d import fuse_multiview_tokens


def test_single_view_is_unchanged():
    cond = torch.randn(1, 21, 8)
    assert fuse_multiview_tokens(cond, aux_grid=2) is cond


def test_auxiliary_views_are_spatially_pooled():
    cond = torch.arange(3 * 21 * 2, dtype=torch.float32).reshape(3, 21, 2)
    fused = fuse_multiview_tokens(cond, aux_grid=2)

    assert fused.shape == (1, 31, 2)
    torch.testing.assert_close(fused[:, :21], cond[0:1])

    expected_global = cond[1, :5].mean(dim=0)
    torch.testing.assert_close(fused[0, 21], expected_global)

    patches = cond[1, 5:].reshape(4, 4, 2).permute(2, 0, 1).unsqueeze(0)
    expected_patches = torch.nn.functional.adaptive_avg_pool2d(
        patches, (2, 2)
    ).squeeze(0).permute(1, 2, 0).reshape(4, 2)
    torch.testing.assert_close(fused[0, 22:26], expected_patches)


def test_rejects_non_square_patch_sequence():
    cond = torch.randn(2, 22, 8)
    try:
        fuse_multiview_tokens(cond, aux_grid=2)
    except ValueError as exc:
        assert "square DINO patch grid" in str(exc)
    else:
        raise AssertionError("Expected non-square patch grid to be rejected")
