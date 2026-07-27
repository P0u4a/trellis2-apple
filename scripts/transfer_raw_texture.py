"""Transfer voxel PBR attributes between raw TRELLIS mesh checkpoints."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from scipy.spatial import cKDTree

from trellis2.representations import MeshWithVoxel


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Keep geometry from one raw checkpoint and transfer nearest "
            "voxel PBR attributes from another checkpoint."
        )
    )
    parser.add_argument("geometry")
    parser.add_argument("texture")
    parser.add_argument("output")
    parser.add_argument("--chunk-size", type=int, default=250_000)
    args = parser.parse_args()

    geometry = torch.load(args.geometry, map_location="cpu", weights_only=False)
    texture = torch.load(args.texture, map_location="cpu", weights_only=False)
    if geometry.layout != texture.layout:
        raise ValueError("Geometry and texture checkpoints use different PBR layouts")
    if abs(float(geometry.voxel_size) - float(texture.voxel_size)) > 1e-9:
        raise ValueError("Geometry and texture checkpoints use different voxel sizes")

    source_coords = texture.coords.cpu().numpy().astype(np.float32, copy=False)
    target_coords = geometry.coords.cpu().numpy().astype(np.float32, copy=False)
    source_attrs = texture.attrs.cpu().numpy()
    tree = cKDTree(source_coords, compact_nodes=True, balanced_tree=True)

    transferred = np.empty((len(target_coords), source_attrs.shape[1]), dtype=source_attrs.dtype)
    distances = np.empty(len(target_coords), dtype=np.float32)
    for start in range(0, len(target_coords), args.chunk_size):
        stop = min(start + args.chunk_size, len(target_coords))
        distance, index = tree.query(target_coords[start:stop], k=1, workers=-1)
        transferred[start:stop] = source_attrs[index]
        distances[start:stop] = distance
        print(f"Transferred {stop:,}/{len(target_coords):,} voxels")

    output_mesh = MeshWithVoxel(
        vertices=geometry.vertices,
        faces=geometry.faces,
        origin=geometry.origin.tolist(),
        voxel_size=geometry.voxel_size,
        coords=geometry.coords,
        attrs=torch.from_numpy(transferred),
        voxel_shape=geometry.voxel_shape,
        layout=geometry.layout,
    )
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    torch.save(output_mesh, output)
    quantiles = np.quantile(distances, [0.5, 0.9, 0.99, 1.0])
    print(
        "Nearest source distance (voxels): "
        f"p50={quantiles[0]:.2f}, p90={quantiles[1]:.2f}, "
        f"p99={quantiles[2]:.2f}, max={quantiles[3]:.2f}"
    )
    print(output)


if __name__ == "__main__":
    main()
