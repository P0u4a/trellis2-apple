"""Compare one sparse SLat flow block between PyTorch and MLX."""

from __future__ import annotations

import os

os.environ.setdefault("SPARSE_CONV_BACKEND", "pytorch")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")

import json

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open

from mlx_backend import remap_flow_model_weights
from mlx_backend.rope import build_rope_freqs, compute_rope_phases
from mlx_backend.sparse_tensor import MlxSparseTensor
from mlx_backend.transformer_block import MlxModulatedSparseTransformerCrossBlock
from trellis2.modules.sparse import SparseTensor
from trellis2.modules.sparse.transformer import ModulatedSparseTransformerCrossBlock
from trellis2.modules.utils import convert_module_to


CONFIG_PATH = "weights/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.json"
WEIGHTS_PATH = "weights/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors"


def as_numpy(value: mx.array) -> np.ndarray:
    return np.asarray(value.astype(mx.float32))


def report(name: str, reference: np.ndarray, candidate: np.ndarray) -> None:
    ref = reference.astype(np.float64).reshape(-1)
    got = candidate.astype(np.float64).reshape(-1)
    delta = got - ref
    cosine = np.dot(ref, got) / (np.linalg.norm(ref) * np.linalg.norm(got))
    print(
        f"{name:>12}: max={np.max(np.abs(delta)):.6g} "
        f"mean={np.mean(np.abs(delta)):.6g} "
        f"rmse={np.sqrt(np.mean(delta**2)):.6g} "
        f"cos={cosine:.9f}"
    )


def main() -> None:
    with open(CONFIG_PATH) as handle:
        args = json.load(handle)["args"]
    channels = args["model_channels"]
    heads = args["num_heads"]
    kwargs = {
        "channels": channels,
        "ctx_channels": args["cond_channels"],
        "num_heads": heads,
        "mlp_ratio": args["mlp_ratio"],
        "use_rope": True,
        "share_mod": True,
        "qk_rms_norm": True,
        "qk_rms_norm_cross": True,
    }

    pt_block = ModulatedSparseTransformerCrossBlock(**kwargs).eval()
    pt_block.apply(lambda module: convert_module_to(module, torch.bfloat16))
    with safe_open(WEIGHTS_PATH, framework="pt", device="cpu") as handle:
        state = {
            key.removeprefix("blocks.0."): handle.get_tensor(key)
            for key in handle.keys()
            if key.startswith("blocks.0.")
        }
    pt_block.load_state_dict(state)

    mlx_block = MlxModulatedSparseTransformerCrossBlock(**kwargs)
    weights = mx.load(WEIGHTS_PATH)
    weights = {
        key: value for key, value in weights.items()
        if key.startswith("blocks.0.")
    }
    weights = remap_flow_model_weights(weights)
    weights = {
        key.removeprefix("blocks.0."): value
        for key, value in weights.items()
    }
    mlx_block.load_weights(list(weights.items()))

    torch.manual_seed(29)
    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0],
         [0, 1, 1, 1], [0, 2, 3, 4], [0, 7, 5, 3], [0, 15, 15, 15]],
        dtype=torch.int32,
    )
    feats = torch.randn(len(coords), channels).to(torch.bfloat16)
    mod = torch.randn(1, 6 * channels).to(torch.bfloat16)
    context = torch.randn(1, 11, args["cond_channels"]).to(torch.bfloat16)
    pt_x = SparseTensor(
        feats=feats,
        coords=coords,
        shape=torch.Size([1, channels]),
        spatial_shape=[64, 64, 64],
    )

    with torch.no_grad():
        pt_output = pt_block(pt_x, mod, context).feats

    mlx_coords = mx.array(coords.numpy())
    freqs = build_rope_freqs(channels // heads, dim=3)
    rope = compute_rope_phases(mlx_coords[:, 1:], freqs, channels // heads)
    mlx_x = MlxSparseTensor(
        feats=mx.array(feats.float().numpy()).astype(mx.bfloat16),
        coords=mlx_coords,
        shape=(1, channels),
    )
    mlx_output = mlx_block(
        mlx_x.feats,
        mx.array(mod.float().numpy()).astype(mx.bfloat16),
        mx.array(context.float().numpy()).astype(mx.bfloat16),
        rope_cache=rope,
    )
    mx.eval(mlx_output)
    report("slat_bf16", pt_output.float().numpy(), as_numpy(mlx_output))

    pt_block.float()
    mlx_block.set_dtype(mx.float32)
    pt_x_f32 = pt_x.replace(feats.float())
    with torch.no_grad():
        pt_output = pt_block(pt_x_f32, mod.float(), context.float()).feats
    mlx_output = mlx_block(
        mx.array(feats.float().numpy()),
        mx.array(mod.float().numpy()),
        mx.array(context.float().numpy()),
        rope_cache=rope,
    )
    mx.eval(mlx_output)
    report("slat_f32", pt_output.numpy(), as_numpy(mlx_output))


if __name__ == "__main__":
    main()
