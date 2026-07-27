import os

os.environ.setdefault("TRELLIS_DISABLE_METAL", "1")

import torch

from o_voxel.postprocess import _guarded_project_back


class FakeBVH:
    def __init__(self, distances, face_ids, barycentrics):
        self.distances = torch.tensor(distances, dtype=torch.float32)
        self.face_ids = torch.tensor(face_ids, dtype=torch.long)
        self.barycentrics = torch.tensor(barycentrics, dtype=torch.float32)

    def unsigned_distance(self, query, return_uvw=False):
        assert return_uvw
        assert len(query) == len(self.distances)
        return self.distances, self.face_ids, self.barycentrics


def _plane():
    vertices = torch.tensor(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)
    return vertices, faces


def test_clean_parallel_surface_projects():
    src_vertices, src_faces = _plane()
    dc_vertices = src_vertices + torch.tensor([0.0, 0.0, 0.1])
    dc_faces = src_faces.clone()
    bvh = FakeBVH(
        [0.1] * 4,
        [0, 0, 0, 1],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]],
    )

    projected, moved, reverted = _guarded_project_back(
        dc_vertices,
        dc_faces,
        src_vertices,
        src_faces,
        bvh,
        strength=1.0,
        voxel_size=0.1,
    )

    torch.testing.assert_close(projected, src_vertices)
    assert moved == 4
    assert reverted == 0


def test_distance_guard_rejects_far_vertex():
    src_vertices, src_faces = _plane()
    dc_vertices = torch.tensor(
        [[-1.0, -1.0, 0.1], [1.0, -1.0, 0.1], [1.0, 1.0, 2.0]],
        dtype=torch.float32,
    )
    dc_faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    bvh = FakeBVH(
        [0.1, 0.1, 2.0],
        [0, 0, 0],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )

    projected, moved, reverted = _guarded_project_back(
        dc_vertices,
        dc_faces,
        src_vertices,
        src_faces,
        bvh,
        strength=1.0,
        voxel_size=0.5,
        max_dist_voxels=1.0,
        min_normal_agreement=0.0,
    )

    torch.testing.assert_close(projected[2], dc_vertices[2])
    assert moved == 2
    assert reverted == 0


def test_normal_guard_rejects_perpendicular_source():
    src_vertices = torch.tensor(
        [[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 1.0, 1.0]],
        dtype=torch.float32,
    )
    src_faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    dc_vertices = torch.tensor(
        [[-0.5, -0.5, 0.1], [0.5, -0.5, 0.1], [0.0, 0.5, 0.1]],
        dtype=torch.float32,
    )
    dc_faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    bvh = FakeBVH(
        [0.1] * 3,
        [0] * 3,
        [[0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]],
    )

    projected, moved, reverted = _guarded_project_back(
        dc_vertices,
        dc_faces,
        src_vertices,
        src_faces,
        bvh,
        strength=1.0,
        voxel_size=0.1,
    )

    torch.testing.assert_close(projected, dc_vertices)
    assert moved == 0
    assert reverted == 0


def test_face_flip_is_reverted():
    src_vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.1, 0.0]],
        dtype=torch.float32,
    )
    src_faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    dc_vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    dc_faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    bvh = FakeBVH(
        [0.0, 0.0, 0.9],
        [0, 0, 0],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )

    projected, moved, reverted = _guarded_project_back(
        dc_vertices,
        dc_faces,
        src_vertices,
        src_faces,
        bvh,
        strength=2.0,
        voxel_size=1.0,
        max_dist_voxels=2.0,
    )

    torch.testing.assert_close(projected, dc_vertices)
    assert moved == 0
    assert reverted == 3
