"""Compare the MLX DINOv3 port against the Hugging Face PyTorch reference."""

from __future__ import annotations

import argparse

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import DINOv3ViTModel

from mlx_backend.dinov3 import load_dinov3_from_hf


MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = reference.astype(np.float64).reshape(-1)
    got = candidate.astype(np.float64).reshape(-1)
    delta = got - ref
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "cosine": float(np.dot(ref, got) / (np.linalg.norm(ref) * np.linalg.norm(got))),
    }


def report(name: str, reference: np.ndarray, candidate: np.ndarray) -> None:
    values = metrics(reference, candidate)
    print(
        f"{name:>12}: max={values['max_abs']:.6g} "
        f"mean={values['mean_abs']:.6g} rmse={values['rmse']:.6g} "
        f"cos={values['cosine']:.9f}"
    )


def make_image(size: int) -> Image.Image:
    y, x = np.mgrid[:size, :size]
    image = np.stack(
        [
            (3 * x + 5 * y) % 256,
            (11 * x + 7 * y + 31) % 256,
            (13 * x + 17 * y + 79) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    if args.size % 16:
        raise SystemExit("--size must be divisible by 16")

    image = make_image(args.size)
    array = np.asarray(image).astype(np.float32) / 255.0
    normalized = (array - MEAN) / STD
    torch_input = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)

    print("Loading PyTorch reference...")
    pt_model = DINOv3ViTModel.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()

    print("Loading MLX port...")
    mlx_model = load_dinov3_from_hf(MODEL_NAME, image_size=args.size)
    mlx_input = mlx_model._preprocess([image])

    with torch.no_grad():
        pt_hidden = pt_model.embeddings(torch_input, bool_masked_pos=None)
        pt_cos, pt_sin = pt_model.rope_embeddings(torch_input)

    mlx_hidden = mlx_model.embeddings(mlx_input)
    mlx_cos, mlx_sin = mlx_model._build_rope_2d(args.size // 16, args.size // 16)
    mx.eval(mlx_hidden, mlx_cos, mlx_sin)

    report("embedding", pt_hidden.numpy(), np.asarray(mlx_hidden))
    report("rope_cos", pt_cos.numpy(), np.asarray(mlx_cos)[0])
    report("rope_sin", pt_sin.numpy(), np.asarray(mlx_sin)[0])

    with torch.no_grad():
        for index, (pt_layer, mlx_layer) in enumerate(zip(pt_model.layer, mlx_model.layers)):
            pt_hidden = pt_layer(
                pt_hidden,
                position_embeddings=(pt_cos, pt_sin),
            )
            mlx_hidden = mlx_layer(
                mlx_hidden,
                mlx_cos,
                mlx_sin,
                num_prefix_tokens=5,
            )
            mx.eval(mlx_hidden)
            report(f"layer_{index:02d}", pt_hidden.numpy(), np.asarray(mlx_hidden))

    with torch.no_grad():
        pt_output = F.layer_norm(pt_hidden, pt_hidden.shape[-1:])
    mlx_output = mx.fast.layer_norm(mlx_hidden, weight=None, bias=None, eps=1e-5)
    mx.eval(mlx_output)
    report("output", pt_output.numpy(), np.asarray(mlx_output))


if __name__ == "__main__":
    main()
