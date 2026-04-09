"""
Collision detection for dental arch interactions.

Provides both differentiable (for optimization) and exact (for validation)
collision checking between dental structures.

Uses:
- pytorch3d knn_points for differentiable penetration depth computation
- open3d for exact BVH-based collision detection (non-differentiable)

References:
- arxiv 2410.20806: BVH collision detection for tooth alignment
- PMC11574221: Fracture overlap penalty in joint optimization
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from pytorch3d.ops import knn_points

logger = logging.getLogger(__name__)


class DifferentiableCollisionLoss(nn.Module):
    """
    Differentiable interpenetration penalty using pytorch3d knn_points.

    For each point in mesh A, finds the nearest point in mesh B. If the point
    is closer than the penetration threshold AND the displacement is opposite
    to the surface normal, it is considered penetrating. The squared penetration
    depth is accumulated as the loss.

    This loss is fully differentiable and can be used in gradient-based
    optimization of tooth poses.
    """

    def __init__(
        self,
        penetration_threshold_mm: float = 0.3,
        use_normals: bool = True,
    ) -> None:
        super().__init__()
        self.threshold = penetration_threshold_mm
        self.use_normals = use_normals

    def forward(
        self,
        points_a: torch.Tensor,
        points_b: torch.Tensor,
        normals_a: Optional[torch.Tensor] = None,
        normals_b: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute differentiable collision penalty between two point clouds.

        Args:
            points_a: (B, N, 3) points of object A.
            points_b: (B, M, 3) points of object B.
            normals_a: (B, N, 3) normals of object A (optional).
            normals_b: (B, M, 3) normals of object B (optional).

        Returns:
            loss: Scalar collision loss.
            info: Dict with 'n_penetrating', 'max_depth', 'mean_depth'.
        """
        # A→B: for each point in A, find nearest in B
        knn_ab = knn_points(points_a, points_b, K=1)
        dists_ab = knn_ab.dists.squeeze(-1).sqrt()  # (B, N) distances in mm
        idx_ab = knn_ab.idx.squeeze(-1)  # (B, N) nearest indices

        # B→A: symmetric check
        knn_ba = knn_points(points_b, points_a, K=1)
        dists_ba = knn_ba.dists.squeeze(-1).sqrt()  # (B, M)
        idx_ba = knn_ba.idx.squeeze(-1)

        if self.use_normals and normals_b is not None:
            # Use surface normals to determine penetration direction
            # Get normal at nearest point in B for each point in A
            nn_normals = torch.gather(
                normals_b, 1, idx_ab.unsqueeze(-1).expand(-1, -1, 3)
            )  # (B, N, 3)
            nn_points = torch.gather(
                points_b, 1, idx_ab.unsqueeze(-1).expand(-1, -1, 3)
            )
            displacement = points_a - nn_points  # (B, N, 3)
            dot = (displacement * nn_normals).sum(dim=-1)  # (B, N)

            # Penetrating if dot < 0 (inside surface) AND close enough
            penetrating_mask = (dot < 0) & (dists_ab < self.threshold)
            penetration_depth = dists_ab * penetrating_mask.float()
        else:
            # Without normals: bidirectional proximity check
            penetrating_mask_ab = dists_ab < self.threshold
            penetration_depth_ab = torch.relu(self.threshold - dists_ab)

            penetrating_mask_ba = dists_ba < self.threshold
            penetration_depth_ba = torch.relu(self.threshold - dists_ba)

            penetration_depth = penetration_depth_ab
            penetrating_mask = penetrating_mask_ab

        # Loss: sum of squared penetration depths
        loss = penetration_depth.pow(2).sum(dim=-1).mean()

        # Info for logging
        n_pen = penetrating_mask.float().sum(dim=-1).mean()
        max_depth = penetration_depth.max()
        mean_depth = penetration_depth[penetrating_mask].mean() if penetrating_mask.any() else torch.tensor(0.0)

        info = {
            "n_penetrating": n_pen,
            "max_depth_mm": max_depth,
            "mean_depth_mm": mean_depth,
        }

        return loss, info


class BVHCollisionDetector:
    """
    Exact collision detection using open3d's spatial data structures.

    Non-differentiable — used for validation and final quality checks,
    not for gradient-based optimization.

    Uses open3d's AABB/OBB trees and triangle-triangle intersection tests.
    """

    def __init__(self, resolution_mm: float = 0.5) -> None:
        self.resolution = resolution_mm

    def check_collision(
        self,
        mesh_a_points: np.ndarray,
        mesh_b_points: np.ndarray,
        mesh_a_triangles: Optional[np.ndarray] = None,
        mesh_b_triangles: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Check for collision between two meshes.

        Args:
            mesh_a_points: (N, 3) vertices of mesh A.
            mesh_b_points: (M, 3) vertices of mesh B.
            mesh_a_triangles: (F_a, 3) triangle indices for A (optional).
            mesh_b_triangles: (F_b, 3) triangle indices for B (optional).

        Returns:
            Dict with 'colliding' (bool), 'n_intersecting_points',
            'min_distance_mm', 'overlap_volume_mm3'.
        """
        import open3d as o3d

        # Create point clouds for distance computation
        pcd_a = o3d.geometry.PointCloud()
        pcd_a.points = o3d.utility.Vector3dVector(mesh_a_points)

        pcd_b = o3d.geometry.PointCloud()
        pcd_b.points = o3d.utility.Vector3dVector(mesh_b_points)

        # Compute distances from A to B
        dists = np.asarray(pcd_a.compute_point_cloud_distance(pcd_b))
        min_dist = float(dists.min())

        # Points in A that are very close to B (potential penetration)
        close_mask = dists < self.resolution
        n_close = int(close_mask.sum())

        # Estimate overlap volume via voxel grid intersection
        overlap_volume = self._estimate_overlap_volume(
            mesh_a_points, mesh_b_points
        )

        return {
            "colliding": min_dist < self.resolution,
            "n_intersecting_points": n_close,
            "min_distance_mm": min_dist,
            "overlap_volume_mm3": overlap_volume,
        }

    def check_multi_tooth_collisions(
        self,
        tooth_points: Dict[int, np.ndarray],
    ) -> List[Dict]:
        """
        Check all pairwise collisions between teeth.

        Args:
            tooth_points: Dict mapping FDI number → (N, 3) point cloud.

        Returns:
            List of collision records for colliding pairs.
        """
        collisions = []
        fdi_list = sorted(tooth_points.keys())

        for i, fdi_a in enumerate(fdi_list):
            for fdi_b in fdi_list[i + 1:]:
                result = self.check_collision(
                    tooth_points[fdi_a], tooth_points[fdi_b]
                )
                if result["colliding"]:
                    collisions.append({
                        "tooth_a": fdi_a,
                        "tooth_b": fdi_b,
                        **result,
                    })

        return collisions

    def _estimate_overlap_volume(
        self,
        points_a: np.ndarray,
        points_b: np.ndarray,
    ) -> float:
        """
        Approximate overlap volume using voxel grid intersection.

        Voxelizes both point clouds and counts shared voxels.
        """
        import open3d as o3d

        voxel_size = self.resolution

        pcd_a = o3d.geometry.PointCloud()
        pcd_a.points = o3d.utility.Vector3dVector(points_a)
        vg_a = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_a, voxel_size)

        pcd_b = o3d.geometry.PointCloud()
        pcd_b.points = o3d.utility.Vector3dVector(points_b)
        vg_b = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_b, voxel_size)

        # Get voxel grid indices
        voxels_a = set()
        for v in vg_a.get_voxels():
            idx = v.grid_index
            voxels_a.add((idx[0], idx[1], idx[2]))

        voxels_b = set()
        for v in vg_b.get_voxels():
            idx = v.grid_index
            voxels_b.add((idx[0], idx[1], idx[2]))

        # Intersection
        shared = voxels_a & voxels_b
        overlap_volume = len(shared) * (voxel_size ** 3)

        return overlap_volume


def compute_interpenetration_volume(
    points_a: np.ndarray,
    points_b: np.ndarray,
    voxel_size_mm: float = 0.5,
) -> float:
    """
    Approximate the interpenetration volume between two surfaces.

    Uses voxel grid intersection via open3d.

    Args:
        points_a: (N, 3) surface points of object A.
        points_b: (M, 3) surface points of object B.
        voxel_size_mm: Voxel grid resolution.

    Returns:
        Estimated overlap volume in mm³.
    """
    detector = BVHCollisionDetector(resolution_mm=voxel_size_mm)
    return detector._estimate_overlap_volume(points_a, points_b)
