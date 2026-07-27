"""Compare the first FlexiDualGrid decoder ConvNeXt block in PyTorch and MLX."""

from __future__ import annotations

import os

os.environ.setdefault("SPARSE_CONV_BACKEND", "pytorch")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open

from mlx_backend import remap_vae_decoder_weights
from mlx_backend.sparse_tensor import MlxSparseTensor
from mlx_backend.vae_blocks import MlxSparseConvNeXtBlock3d
from trellis2.models.sc_vaes.sparse_unet_vae import SparseConvNeXtBlock3d
from trellis2.modules.sparse import SparseTensor


WEIGHTS_PATH = "weights/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors"
PREFIX = "blocks.0.0."


def main() -> None:
    pt_block = SparseConvNeXtBlock3d(1024).eval()
    with safe_open(WEIGHTS_PATH, framework="pt", device="cpu") as handle:
        state = {
            key.removeprefix(PREFIX): handle.get_tensor(key)
            for key in handle.keys()
            if key.startswith(PREFIX)
        }
    pt_block.load_state_dict(state)

    mlx_block = MlxSparseConvNeXtBlock3d(1024)
    weights = mx.load(WEIGHTS_PATH)
    selected = {
        key: value for key, value in weights.items()
        if key.startswith(PREFIX)
    }
    selected = remap_vae_decoder_weights(selected)
    selected = {
        key.removeprefix(PREFIX): value
        for key, value in selected.items()
    }
    mlx_block.load_weights(list(selected.items()))
    mlx_block.set_dtype(mx.float32)

    torch.manual_seed(31)
    coords = torch.tensor(
        [[0, 1, 1, 1], [0, 1, 1, 2], [0, 1, 2, 1],
         [0, 2, 1, 1], [0, 2, 2, 2], [0, 3, 2, 1]],
        dtype=torch.int32,
    )
    feats = torch.randn(len(coords), 1024)
    pt_x = SparseTensor(
        feats=feats,
        coords=coords,
        shape=torch.Size([1, 1024]),
        spatial_shape=[4, 4, 4],
    )
    with torch.no_grad():
        pt_output = pt_block(pt_x).feats.numpy()

    mlx_x = MlxSparseTensor(
        feats=mx.array(feats.numpy()),
        coords=mx.array(coords.numpy()),
        shape=(1, 1024),
    )
    mlx_x.register_spatial_cache("shape", (4, 4, 4))
    mlx_output = mlx_block(mlx_x).feats
    mx.eval(mlx_output)
    mlx_output = np.asarray(mlx_output)

    delta = mlx_output - pt_output
    ref = pt_output.reshape(-1).astype(np.float64)
    got = mlx_output.reshape(-1).astype(np.float64)
    cosine = np.dot(ref, got) / (np.linalg.norm(ref) * np.linalg.norm(got))
    print(
        f"max={np.max(np.abs(delta)):.9g} "
        f"mean={np.mean(np.abs(delta)):.9g} "
        f"rmse={np.sqrt(np.mean(delta**2)):.9g} "
        f"cos={cosine:.12f}"
    )


if __name__ == "__main__":
    main()
