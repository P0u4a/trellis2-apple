"""Compare the MLX sparse-structure decoder with the PyTorch reference."""

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from mlx_backend.structure_decoder import load_structure_decoder
from trellis2.models.sparse_structure_vae import SparseStructureDecoder


REPO = "microsoft/TRELLIS-image-large"
STEM = "ckpts/ss_dec_conv3d_16l8_fp16"


def report(name: str, reference: np.ndarray, candidate: np.ndarray) -> None:
    ref = reference.astype(np.float64).reshape(-1)
    got = candidate.astype(np.float64).reshape(-1)
    delta = got - ref
    cosine = np.dot(ref, got) / (np.linalg.norm(ref) * np.linalg.norm(got))
    print(
        f"{name:>14}: shape={reference.shape} "
        f"max={np.max(np.abs(delta)):.6g} "
        f"mean={np.mean(np.abs(delta)):.6g} "
        f"rmse={np.sqrt(np.mean(delta**2)):.6g} "
        f"cos={cosine:.9f}"
    )


def main() -> None:
    config_path = hf_hub_download(REPO, f"{STEM}.json", local_files_only=True)
    weights_path = hf_hub_download(REPO, f"{STEM}.safetensors", local_files_only=True)
    model_path = config_path.removesuffix(".json")
    with open(config_path) as handle:
        args = json.load(handle)["args"]

    torch.manual_seed(7)
    input_tensor = torch.randn(1, args["latent_channels"], 16, 16, 16)

    print("Loading PyTorch reference...")
    pt_model = SparseStructureDecoder(**args)
    pt_model.load_state_dict(load_file(weights_path))
    pt_model.convert_to_fp32()
    pt_model.eval()
    with torch.no_grad():
        pt_output = pt_model(input_tensor)

    print("Loading MLX port...")
    mlx_model = load_structure_decoder(model_path)
    mlx_model.set_dtype(mx.float32)
    mlx_model.use_fp16 = False
    mlx_output = mlx_model(mx.array(input_tensor.numpy()))
    mx.eval(mlx_output)

    report("decoder", pt_output.numpy(), np.asarray(mlx_output))
    pt_mask = pt_output.numpy() > 0
    mlx_mask = np.asarray(mlx_output) > 0
    intersection = np.logical_and(pt_mask, mlx_mask).sum()
    union = np.logical_or(pt_mask, mlx_mask).sum()
    iou = 1.0 if union == 0 else intersection / union
    print(
        f"occupancy: pt={pt_mask.sum()} mlx={mlx_mask.sum()} "
        f"IoU={iou:.9f}"
    )


if __name__ == "__main__":
    main()
