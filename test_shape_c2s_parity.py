"""Compare the first shape-decoder channel-to-spatial block with PyTorch."""

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
from mlx_backend.vae_blocks import MlxSparseResBlockC2S3d
from trellis2.models.sc_vaes.sparse_unet_vae import SparseResBlockC2S3d
from trellis2.modules.sparse import SparseTensor


WEIGHTS_PATH = "weights/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors"
PREFIX = "blocks.0.4."


def _report(reference: np.ndarray, candidate: np.ndarray) -> None:
    ref = reference.astype(np.float64).reshape(-1)
    got = candidate.astype(np.float64).reshape(-1)
    delta = got - ref
    cosine = np.dot(ref, got) / (np.linalg.norm(ref) * np.linalg.norm(got))
    print(
        f"max={np.max(np.abs(delta)):.9g} "
        f"mean={np.mean(np.abs(delta)):.9g} "
        f"rmse={np.sqrt(np.mean(delta**2)):.9g} "
        f"cos={cosine:.12f}"
    )


def main() -> None:
    pt_block = SparseResBlockC2S3d(1024, 512, pred_subdiv=True).eval()
    with safe_open(WEIGHTS_PATH, framework="pt", device="cpu") as handle:
        state = {
            key.removeprefix(PREFIX): handle.get_tensor(key)
            for key in handle.keys()
            if key.startswith(PREFIX)
        }
    pt_block.load_state_dict(state)

    mlx_block = MlxSparseResBlockC2S3d(1024, 512, pred_subdiv=True)
    selected = {
        key: value
        for key, value in mx.load(WEIGHTS_PATH).items()
        if key.startswith(PREFIX)
    }
    selected = remap_vae_decoder_weights(selected)
    selected = {
        key.removeprefix(PREFIX): value
        for key, value in selected.items()
    }
    mlx_block.load_weights(list(selected.items()))
    mlx_block.set_dtype(mx.float32)

    torch.manual_seed(71)
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
    feats = torch.randn(len(coords), 1024)
    pt_x = SparseTensor(
        feats=feats,
        coords=coords,
        shape=torch.Size([1, 1024]),
        spatial_shape=[4, 4, 4],
    )
    with torch.no_grad():
        pt_output, pt_subdiv = pt_block(pt_x)

    mlx_x = MlxSparseTensor(
        feats=mx.array(feats.numpy()),
        coords=mx.array(coords.numpy()),
        shape=(1, 1024),
    )
    mlx_x.register_spatial_cache("shape", (4, 4, 4))
    mlx_output, mlx_subdiv = mlx_block(mlx_x)
    mx.eval(
        mlx_output.feats,
        mlx_output.coords,
        mlx_subdiv.feats,
    )

    mlx_coords = np.asarray(mlx_output.coords)
    pt_coords = pt_output.coords.numpy()
    print(
        f"coords_equal={np.array_equal(pt_coords, mlx_coords)} "
        f"pt_tokens={len(pt_coords)} mlx_tokens={len(mlx_coords)}"
    )
    _report(pt_subdiv.feats.numpy(), np.asarray(mlx_subdiv.feats))
    _report(pt_output.feats.numpy(), np.asarray(mlx_output.feats))


if __name__ == "__main__":
    main()
