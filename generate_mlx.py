"""Generate a textured GLB with the low-memory TRELLIS.2 MLX backend."""

from __future__ import annotations

import argparse
import os
import sys
import time

# Must be set before importing torch/TRELLIS modules.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")


def _memory_gib(value: int) -> float:
    return value / (1024 ** 3)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a textured GLB using TRELLIS.2, DINOv3 and MLX."
    )
    parser.add_argument("image", help="Input image (transparent PNG works best)")
    parser.add_argument("--output", default="outputs/trellis2.glb")
    parser.add_argument("--weights", default="weights/TRELLIS.2-4B")
    parser.add_argument(
        "--pipeline-type",
        choices=("512", "1024", "1024_cascade", "1536_cascade"),
        default="512",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--decimation-target", type=int, default=200_000)
    parser.add_argument("--texture-size", type=int, choices=(512, 1024, 2048), default=1024)
    parser.add_argument(
        "--preprocessed",
        action="store_true",
        help="Skip crop/background removal; use only for an already prepared square image.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Run generation and report memory without texture baking (for probes).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        parser.error(f"input image does not exist: {args.image}")
    if not os.path.isfile(os.path.join(args.weights, "pipeline.json")):
        parser.error(
            f"weights are incomplete at {args.weights}; run scripts/download_weights.py"
        )

    import mlx.core as mx
    from PIL import Image
    from mlx_backend.pipeline import create_mlx_pipeline, to_glb

    mx.reset_peak_memory()
    mx.set_cache_limit(2 * 1024 ** 3)

    started = time.perf_counter()
    pipeline = create_mlx_pipeline(args.weights, pipeline_type=args.pipeline_type)
    loaded = time.perf_counter()
    print(
        f"[MLX] Models loaded in {loaded - started:.1f}s; "
        f"active={_memory_gib(mx.get_active_memory()):.2f} GiB"
    )

    image = Image.open(args.image)
    sampler = {"steps": args.steps}
    meshes = pipeline.run(
        image,
        seed=args.seed,
        pipeline_type=args.pipeline_type,
        preprocess_image=not args.preprocessed,
        sparse_structure_sampler_params=sampler,
        shape_slat_sampler_params=sampler,
        tex_slat_sampler_params=sampler,
    )
    generated = time.perf_counter()

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    if not args.skip_export:
        to_glb(
            meshes[0],
            output,
            decimation_target=args.decimation_target,
            texture_size=args.texture_size,
        )
    finished = time.perf_counter()

    print(f"Generation: {generated - loaded:.1f}s")
    print(f"Texture/export: {finished - generated:.1f}s")
    print(f"Peak MLX memory: {_memory_gib(mx.get_peak_memory()):.2f} GiB")
    if not args.skip_export:
        print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
