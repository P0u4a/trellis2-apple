"""Blend clean body PBR voxels with sharper character-detail PBR voxels."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from trellis2.representations import MeshWithVoxel


def _clip(value):
    return np.clip(value, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Use the base field over the body while preserving detail-field "
            "attributes around the head, hands, shirt and tie."
        )
    )
    parser.add_argument("base")
    parser.add_argument("detail")
    parser.add_argument("output")
    args = parser.parse_args()

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    detail = torch.load(args.detail, map_location="cpu", weights_only=False)
    if not torch.equal(base.coords, detail.coords):
        raise ValueError("Texture fields must share identical voxel coordinates")
    if base.layout != detail.layout:
        raise ValueError("Texture fields use different PBR layouts")

    coords = base.coords.numpy().astype(np.float32, copy=False)
    xyz = coords * float(base.voxel_size) + base.origin.numpy()
    x, vertical = xyz[:, 0], xyz[:, 1]

    # TRELLIS raw coordinates become Blender Z=-Y during GLB export.
    # Feather every region to avoid a visible field boundary.
    head = _clip((-0.28 - vertical) / 0.10)
    hand_x = _clip((np.abs(x) - 0.25) / 0.08)
    hand_height = 1.0 - _clip((np.abs(vertical + 0.02) - 0.10) / 0.08)
    hands = hand_x * hand_height

    # Preserve the narrow tie/placket center, but let the cleaner base field
    # cover most of the shirt and coat. A broad torso box would encompass too
    # much of the sparse field because clothing layers are densely occupied.
    shirt_x = _clip((0.075 - np.abs(x)) / 0.035)
    shirt_top = _clip((vertical + 0.27) / 0.06)
    shirt_bottom = _clip((0.055 - vertical) / 0.05)
    shirt = shirt_x * shirt_top * shirt_bottom

    detail_weight = np.maximum.reduce([head, hands, shirt]).astype(np.float32)
    base_attrs = base.attrs.numpy()
    detail_attrs = detail.attrs.numpy()
    blended = (
        base_attrs * (1.0 - detail_weight[:, None])
        + detail_attrs * detail_weight[:, None]
    ).astype(base_attrs.dtype, copy=False)

    output_mesh = MeshWithVoxel(
        vertices=base.vertices,
        faces=base.faces,
        origin=base.origin.tolist(),
        voxel_size=base.voxel_size,
        coords=base.coords,
        attrs=torch.from_numpy(blended),
        voxel_shape=base.voxel_shape,
        layout=base.layout,
    )
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    torch.save(output_mesh, output)
    print(
        f"Detail weights: >0={int((detail_weight > 0).sum()):,}, "
        f">0.5={int((detail_weight > 0.5).sum()):,}, "
        f"full={int((detail_weight >= 1).sum()):,}"
    )
    print(output)


if __name__ == "__main__":
    main()
