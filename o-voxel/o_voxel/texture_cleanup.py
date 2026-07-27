"""Conservative cleanup for baked TRELLIS PBR textures."""

import cv2
import numpy as np


def clean_base_color(
    image: np.ndarray,
    valid_mask: np.ndarray,
    despeckle: float = 0.0,
    sharpen: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Remove strong local color outliers, then apply a mild unsharp mask.

    Replacement is restricted to well-covered UV interiors and pixels that
    differ sharply from a local median. This avoids globally blurring facial
    features, seams, fabric edges, or small UV islands.
    """
    despeckle = float(np.clip(despeckle, 0.0, 1.0))
    sharpen = float(np.clip(sharpen, 0.0, 1.0))
    if despeckle == 0.0 and sharpen == 0.0:
        return image, 0

    cleaned = image.copy()
    replaced_count = 0
    if despeckle > 0.0:
        kernel = 5 if despeckle >= 0.5 else 3
        median = cv2.medianBlur(image, kernel)
        delta = np.max(
            np.abs(image.astype(np.int16) - median.astype(np.int16)), axis=-1
        )
        coverage = cv2.boxFilter(
            valid_mask.astype(np.uint8),
            ddepth=cv2.CV_16U,
            ksize=(kernel, kernel),
            normalize=False,
        )
        min_coverage = int(kernel * kernel * 0.72)
        threshold = int(round(82 - 42 * despeckle))
        replace = (
            valid_mask
            & (coverage >= min_coverage)
            & (delta >= threshold)
        )
        cleaned[replace] = median[replace]
        replaced_count = int(replace.sum())

    if sharpen > 0.0:
        blur = cv2.GaussianBlur(cleaned, (0, 0), sigmaX=0.8)
        sharpened = cv2.addWeighted(
            cleaned, 1.0 + 0.65 * sharpen, blur, -0.65 * sharpen, 0
        )
        cleaned[valid_mask] = sharpened[valid_mask]

    return cleaned, replaced_count
