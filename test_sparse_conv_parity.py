"""Compare MLX sparse convolution against the Metal flex_gemm backend."""

from __future__ import annotations

import os

os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("FLEX_GEMM_QUIET", "1")

import mlx.core as mx
import numpy as np
import torch

from mlx_backend.sparse_conv import MlxSparseConv3d
from mlx_backend.sparse_tensor import MlxSparseTensor
from trellis2.modules.sparse import SparseConv3d, SparseTensor


def main() -> None:
    torch.manual_seed(23)
    coords = torch.tensor(
        [
            [0, 1, 1, 1],
            [0, 1, 1, 2],
            [0, 1, 2, 1],
            [0, 2, 1, 1],
            [0, 2, 2, 2],
            [0, 3, 2, 1],
        ],
        dtype=torch.int32,
    )
    feats = torch.randn(len(coords), 3, dtype=torch.float32)
    weight = torch.arange(2 * 3 * 3 * 3 * 3, dtype=torch.float32).reshape(2, 3, 3, 3, 3)
    weight = (weight - weight.mean()) / 100
    bias = torch.tensor([0.125, -0.25], dtype=torch.float32)

    pt_conv = SparseConv3d(3, 2, 3)
    pt_conv.weight.data.copy_(weight)
    pt_conv.bias.data.copy_(bias)
    pt_conv = pt_conv.to("mps")
    pt_x = SparseTensor(
        feats=feats.to("mps"),
        coords=coords.to("mps"),
        shape=torch.Size([1, 3]),
        spatial_shape=[4, 4, 4],
    )
    pt_output = pt_conv(pt_x).feats.detach().cpu().numpy()

    mlx_conv = MlxSparseConv3d(3, 2, 3)
    mlx_conv.load_weights(
        [
            ("weight", mx.array(weight.numpy())),
            ("bias", mx.array(bias.numpy())),
        ]
    )
    mlx_x = MlxSparseTensor(
        feats=mx.array(feats.numpy()),
        coords=mx.array(coords.numpy()),
        shape=(1, 3),
    )
    mlx_x.register_spatial_cache("shape", (4, 4, 4))
    mlx_output = mlx_conv(mlx_x).feats
    mx.eval(mlx_output)
    mlx_output = np.asarray(mlx_output)

    delta = mlx_output - pt_output
    print("PyTorch/flex_gemm:")
    print(pt_output)
    print("MLX:")
    print(mlx_output)
    print(
        f"max={np.max(np.abs(delta)):.9g} "
        f"mean={np.mean(np.abs(delta)):.9g} "
        f"rmse={np.sqrt(np.mean(delta**2)):.9g}"
    )


if __name__ == "__main__":
    main()
