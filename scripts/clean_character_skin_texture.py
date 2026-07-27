"""Normalize face/hand texels using UV regions exported from Blender."""

from __future__ import annotations

import argparse

import cv2
import numpy as np


def rasterize_uv_triangles(
    triangles: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for triangle in triangles:
        points = np.empty((3, 2), dtype=np.int32)
        points[:, 0] = np.clip(
            np.rint(triangle[:, 0] * (width - 1)),
            0,
            width - 1,
        )
        points[:, 1] = np.clip(
            np.rint((1.0 - triangle[:, 1]) * (height - 1)),
            0,
            height - 1,
        )
        cv2.fillConvexPoly(mask, points, 255)
    return mask


def normalize_region(
    image: np.ndarray,
    region_mask: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    rgb = image[..., ::-1].astype(np.float32)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    warm = (
        (region_mask > 0)
        & (red > 42)
        & (green > 24)
        & (blue > 18)
        & (red > green * 1.045)
        & (red > blue * 1.075)
    )

    # Fill tiny gaps inside skin UV islands while keeping the spatial Blender
    # region as a hard boundary, so hair and clothing remain untouched.
    skin_mask = cv2.morphologyEx(
        warm.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )
    skin_mask = cv2.dilate(
        skin_mask,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    skin_mask[region_mask == 0] = 0
    selected = skin_mask > 0
    if not np.any(selected):
        return image, skin_mask, (0.0, 0.0, 0.0)

    luminance = (
        rgb[..., 0] * 0.2126
        + rgb[..., 1] * 0.7152
        + rgb[..., 2] * 0.0722
    )
    median_rgb = np.median(rgb[selected], axis=0)
    median_luminance = float(np.median(luminance[selected]))
    tone_luminance = float(
        median_rgb[0] * 0.2126
        + median_rgb[1] * 0.7152
        + median_rgb[2] * 0.0722
    )
    compressed_luminance = (
        median_luminance
        + (luminance - median_luminance) * 0.38
    )
    normalized = (
        median_rgb[None, None, :]
        * (compressed_luminance / max(tone_luminance, 1e-5))[..., None]
    )
    normalized = np.clip(normalized, 0, 255)

    bilateral = cv2.bilateralFilter(
        image,
        d=7,
        sigmaColor=28,
        sigmaSpace=5,
    )[..., ::-1].astype(np.float32)
    cleaned_rgb = bilateral * 0.28 + normalized * 0.72
    feather = cv2.GaussianBlur(
        skin_mask.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=0.65,
    )
    feather *= strength
    result_rgb = (
        rgb * (1.0 - feather[..., None])
        + cleaned_rgb * feather[..., None]
    )
    result = np.clip(result_rgb[..., ::-1], 0, 255).astype(np.uint8)
    return result, skin_mask, tuple(float(value) for value in median_rgb)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("regions")
    parser.add_argument("output")
    parser.add_argument("--mask-output")
    parser.add_argument("--strength", type=float, default=0.82)
    args = parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    regions = np.load(args.regions)
    height, width = image.shape[:2]
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    stats = []
    cleaned = image
    for name in ("face", "hands"):
        region_mask = rasterize_uv_triangles(
            regions[name],
            width,
            height,
        )
        cleaned, skin_mask, median_rgb = normalize_region(
            cleaned,
            region_mask,
            float(np.clip(args.strength, 0.0, 1.0)),
        )
        combined_mask = np.maximum(combined_mask, skin_mask)
        stats.append(
            (
                name,
                int((region_mask > 0).sum()),
                int((skin_mask > 0).sum()),
                median_rgb,
            )
        )

    if not cv2.imwrite(args.output, cleaned):
        raise RuntimeError(f"Failed to write {args.output}")
    if args.mask_output and not cv2.imwrite(args.mask_output, combined_mask):
        raise RuntimeError(f"Failed to write {args.mask_output}")
    for name, region_pixels, skin_pixels, median_rgb in stats:
        print(
            f"{name}: region={region_pixels:,}, skin={skin_pixels:,}, "
            f"median RGB=({median_rgb[0]:.1f}, "
            f"{median_rgb[1]:.1f}, {median_rgb[2]:.1f})"
        )
    print(args.output)


if __name__ == "__main__":
    main()
