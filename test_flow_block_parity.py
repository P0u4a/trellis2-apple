"""Compare one TRELLIS.2 flow transformer block between PyTorch and MLX."""

from __future__ import annotations

import os

# Set before importing TRELLIS attention modules.
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")

import json

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open

from mlx_backend import remap_flow_model_weights
from mlx_backend.rope import build_rope_freqs, compute_rope_phases
from mlx_backend.transformer_block import MlxModulatedTransformerCrossBlock
from trellis2.modules.attention import RotaryPositionEmbedder
from trellis2.modules.transformer.modulated import ModulatedTransformerCrossBlock
from trellis2.modules.utils import convert_module_to


CONFIG_PATH = "weights/TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16.json"
WEIGHTS_PATH = "weights/TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors"


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


def mlx_numpy(value: mx.array) -> np.ndarray:
    return np.asarray(value.astype(mx.float32))


def main() -> None:
    with open(CONFIG_PATH) as handle:
        args = json.load(handle)["args"]

    channels = args["model_channels"]
    heads = args["num_heads"]
    head_dim = channels // heads
    kwargs = {
        "channels": channels,
        "ctx_channels": args["cond_channels"],
        "num_heads": heads,
        "mlp_ratio": args["mlp_ratio"],
        "use_rope": True,
        "share_mod": args["share_mod"],
        "qk_rms_norm": args["qk_rms_norm"],
        "qk_rms_norm_cross": args["qk_rms_norm_cross"],
    }

    print("Loading PyTorch block 0...")
    pt_block = ModulatedTransformerCrossBlock(**kwargs).eval()
    pt_block.apply(lambda module: convert_module_to(module, torch.bfloat16))
    pt_state = {}
    mlx_prefixed = {}
    with safe_open(WEIGHTS_PATH, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith("blocks.0."):
                pt_state[key.removeprefix("blocks.0.")] = handle.get_tensor(key)
    pt_block.load_state_dict(pt_state)

    print("Loading MLX block 0...")
    mlx_block = MlxModulatedTransformerCrossBlock(**kwargs)
    with safe_open(WEIGHTS_PATH, framework="np") as handle:
        for key in handle.keys():
            if key.startswith("blocks.0."):
                # NumPy cannot represent bfloat16; read these through MLX below.
                mlx_prefixed[key] = None
    all_mlx_weights = mx.load(WEIGHTS_PATH)
    selected = {key: all_mlx_weights[key] for key in mlx_prefixed}
    selected = remap_flow_model_weights(selected)
    selected = {
        key.removeprefix("blocks.0."): value
        for key, value in selected.items()
    }
    mlx_block.load_weights(list(selected.items()))

    torch.manual_seed(19)
    batch, tokens, context_tokens = 1, 8, 11
    x = torch.randn(batch, tokens, channels).to(torch.bfloat16)
    mod = torch.randn(batch, 6 * channels).to(torch.bfloat16)
    context = torch.randn(batch, context_tokens, args["cond_channels"]).to(torch.bfloat16)
    coords = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0],
         [1, 1, 1], [2, 3, 4], [7, 5, 3], [15, 15, 15]],
        dtype=torch.int32,
    )

    pt_rope = RotaryPositionEmbedder(head_dim, 3)(coords)
    mlx_freqs = build_rope_freqs(head_dim, dim=3)
    mlx_cos, mlx_sin = compute_rope_phases(mx.array(coords.numpy()), mlx_freqs, head_dim)
    mx.eval(mlx_cos, mlx_sin)
    report("rope_cos", pt_rope.real.numpy(), np.asarray(mlx_cos))
    report("rope_sin", pt_rope.imag.numpy(), np.asarray(mlx_sin))

    with torch.no_grad():
        pt_output = pt_block(x, mod, context, pt_rope)

    mlx_x = mx.array(x.float().numpy()).astype(mx.bfloat16)
    mlx_mod = mx.array(mod.float().numpy()).astype(mx.bfloat16)
    mlx_context = mx.array(context.float().numpy()).astype(mx.bfloat16)
    mlx_output = mlx_block(
        mlx_x,
        mlx_mod,
        mlx_context,
        rope_cache=(mlx_cos, mlx_sin),
    )
    mx.eval(mlx_output)
    report("block_0", pt_output.float().numpy(), mlx_numpy(mlx_output))

    # Break the block down to identify the first divergent operation.
    with torch.no_grad():
        pt_mods = (pt_block.modulation + mod).type(mod.dtype)
        mlx_mods = (mlx_block.modulation + mlx_mod).astype(mlx_mod.dtype)
        mx.eval(mlx_mods)
        report("mods", pt_mods.float().numpy(), mlx_numpy(mlx_mods))

        pt_parts = pt_mods.chunk(6, dim=-1)
        mlx_parts = mx.split(mlx_mods, 6, axis=-1)

        pt_h = pt_block.norm1(x)
        mlx_h = mlx_block.norm1(mlx_x)
        mx.eval(mlx_h)
        report("norm1", pt_h.float().numpy(), mlx_numpy(mlx_h))

        pt_h = pt_h * (1 + pt_parts[1].unsqueeze(1)) + pt_parts[0].unsqueeze(1)
        mlx_h = mlx_h * (1 + mlx_parts[1][:, None, :]) + mlx_parts[0][:, None, :]
        mx.eval(mlx_h)
        report("mod_norm1", pt_h.float().numpy(), mlx_numpy(mlx_h))

        pt_h = pt_block.self_attn(pt_h, phases=pt_rope)
        mlx_h = mlx_block.self_attn(mlx_h, rope_cache=(mlx_cos, mlx_sin))
        mx.eval(mlx_h)
        report("self_attn", pt_h.float().numpy(), mlx_numpy(mlx_h))

        pt_x = x + pt_h * pt_parts[2].unsqueeze(1)
        mlx_x2 = mlx_x + mlx_h * mlx_parts[2][:, None, :]
        mx.eval(mlx_x2)
        report("self_resid", pt_x.float().numpy(), mlx_numpy(mlx_x2))

        pt_h = pt_block.norm2(pt_x)
        mlx_h = mlx_block.norm2(mlx_x2)
        mx.eval(mlx_h)
        report("norm2", pt_h.float().numpy(), mlx_numpy(mlx_h))

        pt_h = pt_block.cross_attn(pt_h, context)
        mlx_h = mlx_block.cross_attn(mlx_h, context=mlx_context)
        mx.eval(mlx_h)
        report("cross_attn", pt_h.float().numpy(), mlx_numpy(mlx_h))

        pt_x = pt_x + pt_h
        mlx_x2 = mlx_x2 + mlx_h
        mx.eval(mlx_x2)
        report("cross_resid", pt_x.float().numpy(), mlx_numpy(mlx_x2))

        pt_h = pt_block.norm3(pt_x)
        mlx_h = mlx_block.norm3(mlx_x2)
        mx.eval(mlx_h)
        report("norm3", pt_h.float().numpy(), mlx_numpy(mlx_h))

        pt_h = pt_h * (1 + pt_parts[4].unsqueeze(1)) + pt_parts[3].unsqueeze(1)
        mlx_h = mlx_h * (1 + mlx_parts[4][:, None, :]) + mlx_parts[3][:, None, :]
        mx.eval(mlx_h)
        report("mod_norm3", pt_h.float().numpy(), mlx_numpy(mlx_h))

        pt_fc1 = pt_block.mlp.mlp[0]
        mlx_fc1 = mlx_block.mlp.mlp.layers[0]
        pt_fc2 = pt_block.mlp.mlp[2]
        mlx_fc2 = mlx_block.mlp.mlp.layers[2]
        report("fc1_weight", pt_fc1.weight.float().numpy(), mlx_numpy(mlx_fc1.weight))
        report("fc2_weight", pt_fc2.weight.float().numpy(), mlx_numpy(mlx_fc2.weight))

        pt_h = pt_fc1(pt_h)
        mlx_h = mlx_fc1(mlx_h)
        mx.eval(mlx_h)
        report("fc1", pt_h.float().numpy(), mlx_numpy(mlx_h))

        pt_h = torch.nn.functional.gelu(pt_h, approximate="tanh")
        mlx_h = mlx_block.mlp.mlp.layers[1](mlx_h)
        mx.eval(mlx_h)
        report("gelu", pt_h.float().numpy(), mlx_numpy(mlx_h))

        pt_h = pt_fc2(pt_h)
        mlx_h = mlx_fc2(mlx_h)
        mx.eval(mlx_h)
        report("mlp", pt_h.float().numpy(), mlx_numpy(mlx_h))

        pt_x = pt_x + pt_h * pt_parts[5].unsqueeze(1)
        mlx_x2 = mlx_x2 + mlx_h * mlx_parts[5][:, None, :]
        mx.eval(mlx_x2)
        report("manual_out", pt_x.float().numpy(), mlx_numpy(mlx_x2))

    print("Rechecking the same block in float32...")
    pt_block.float()
    mlx_block.set_dtype(mx.float32)
    with torch.no_grad():
        pt_output_f32 = pt_block(x.float(), mod.float(), context.float(), pt_rope)
    mlx_output_f32 = mlx_block(
        mx.array(x.float().numpy()),
        mx.array(mod.float().numpy()),
        mx.array(context.float().numpy()),
        rope_cache=(mlx_cos, mlx_sin),
    )
    mx.eval(mlx_output_f32)
    report("block_f32", pt_output_f32.numpy(), mlx_numpy(mlx_output_f32))


if __name__ == "__main__":
    main()
