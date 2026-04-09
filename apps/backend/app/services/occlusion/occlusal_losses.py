"""
Differentiable loss functions for dental occlusion analysis and optimization.

All losses are designed for gradient-based optimization of tooth poses.
Uses pytorch3d for Chamfer distance and knn_points; all other losses are
custom dental-domain logic.

References:
- PMC11574221: Composite objective for simultaneous dental occlusion + fracture fitting
- arxiv 2410.20806: Occlusal projection overlap + distance uniformity losses
- arxiv 2312.15139 (TADPM): Dental arch curve Fréchet distance
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points


# ─── Chamfer-based inter-arch loss ───────────────────────────────────────────


class ChamferOcclusionLoss(nn.Module):
    """
    Chamfer distance between upper and lower arch surfaces.

    Measures how well the arches fit together by computing bidirectional
    nearest-neighbor distances. Lower = better occlusal fit.

    Wraps pytorch3d.loss.chamfer_distance.
    """

    def __init__(self, single_directional: bool = False) -> None:
        super().__init__()
        self.single_directional = single_directional

    def forward(
        self,
        upper_points: torch.Tensor,
        lower_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            upper_points: (B, N, 3) upper arch point cloud.
            lower_points: (B, M, 3) lower arch point cloud.

        Returns:
            Scalar Chamfer loss.
        """
        loss, _ = chamfer_distance(
            upper_points,
            lower_points,
            single_directional=self.single_directional,
        )
        return loss


# ─── Occlusal projection overlap loss ───────────────────────────────────────


class OcclusalProjectionOverlapLoss(nn.Module):
    """
    Per arxiv 2410.20806 (Swin-transformer tooth alignment):

    Projects upper and lower teeth onto the occlusal plane, computes the
    overlap area (soft IoU), and penalizes deviation from target overlap.

    The occlusal plane is the XY plane (z=0) by default. For each tooth pair,
    the projection overlap should match the anatomically correct contact area.

    Differentiable via soft rasterization of projected point clouds into
    2D density maps.
    """

    def __init__(
        self,
        grid_resolution: int = 64,
        sigma: float = 0.5,
        target_overlap_ratio: float = 0.3,
        occlusal_plane_normal: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.grid_resolution = grid_resolution
        self.sigma = sigma
        self.target_overlap_ratio = target_overlap_ratio
        # Default: occlusal plane is XY (normal along Z)
        self.register_buffer(
            "plane_normal",
            occlusal_plane_normal
            if occlusal_plane_normal is not None
            else torch.tensor([0.0, 0.0, 1.0]),
        )

    def forward(
        self,
        upper_points: torch.Tensor,
        lower_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            upper_points: (B, N, 3) upper teeth points.
            lower_points: (B, M, 3) lower teeth points.

        Returns:
            Scalar overlap deviation loss.
        """
        # Project onto occlusal plane (drop the component along plane normal)
        upper_2d = self._project_to_plane(upper_points)  # (B, N, 2)
        lower_2d = self._project_to_plane(lower_points)  # (B, M, 2)

        # Soft-rasterize into density maps
        upper_density = self._soft_rasterize(upper_2d)  # (B, G, G)
        lower_density = self._soft_rasterize(lower_2d)  # (B, G, G)

        # Soft IoU between density maps
        intersection = (upper_density * lower_density).sum(dim=(-1, -2))
        union = (upper_density + lower_density - upper_density * lower_density).sum(
            dim=(-1, -2)
        )
        iou = intersection / (union + 1e-8)

        # Loss: penalize deviation from target overlap ratio
        loss = (iou - self.target_overlap_ratio).pow(2).mean()
        return loss

    def _project_to_plane(self, points: torch.Tensor) -> torch.Tensor:
        """Project 3D points onto the occlusal plane, returning 2D coords."""
        normal = self.plane_normal.to(points.device)
        # Remove component along normal
        proj = points - (points @ normal).unsqueeze(-1) * normal.unsqueeze(0).unsqueeze(0)
        # Take first two orthogonal dimensions as 2D coords
        # Build orthonormal basis for the plane
        if abs(normal[0]) < 0.9:
            u = torch.cross(normal, torch.tensor([1.0, 0.0, 0.0], device=points.device))
        else:
            u = torch.cross(normal, torch.tensor([0.0, 1.0, 0.0], device=points.device))
        u = u / (u.norm() + 1e-8)
        v = torch.cross(normal, u)
        v = v / (v.norm() + 1e-8)
        coords_2d = torch.stack([proj @ u, proj @ v], dim=-1)  # (B, N, 2)
        return coords_2d

    def _soft_rasterize(self, points_2d: torch.Tensor) -> torch.Tensor:
        """
        Soft-rasterize 2D points into a density grid using Gaussian splatting.
        Fully differentiable.
        """
        B, N, _ = points_2d.shape
        G = self.grid_resolution
        device = points_2d.device

        # Normalize points to [0, 1] range
        mins = points_2d.min(dim=1, keepdim=True).values  # (B, 1, 2)
        maxs = points_2d.max(dim=1, keepdim=True).values
        span = (maxs - mins).clamp(min=1e-6)
        normalized = (points_2d - mins) / span  # (B, N, 2) in [0, 1]

        # Create grid centers
        grid = torch.linspace(0, 1, G, device=device)
        gx, gy = torch.meshgrid(grid, grid, indexing="ij")  # (G, G) each
        grid_centers = torch.stack([gx, gy], dim=-1).view(1, 1, G * G, 2)  # (1, 1, G*G, 2)

        # Compute Gaussian weights: each point contributes to each grid cell
        pts = normalized.unsqueeze(2)  # (B, N, 1, 2)
        dists_sq = ((pts - grid_centers) ** 2).sum(dim=-1)  # (B, N, G*G)
        weights = torch.exp(-dists_sq / (2 * self.sigma**2 / G**2))  # (B, N, G*G)

        # Sum contributions from all points
        density = weights.sum(dim=1)  # (B, G*G)
        density = density.view(B, G, G)

        # Normalize to [0, 1]
        density = density / (density.max(dim=-1, keepdim=True).values.max(
            dim=-2, keepdim=True
        ).values + 1e-8)
        return density


# ─── Occlusal distance uniformity loss ──────────────────────────────────────


class OcclusalDistanceUniformityLoss(nn.Module):
    """
    Per arxiv 2410.20806: Penalizes non-uniform inter-arch distances.

    Measures the variance of nearest-neighbor distances between upper and
    lower teeth across the arch. Uniform contact distribution is desirable
    for stable occlusion.

    Uses pytorch3d.ops.knn_points for nearest-neighbor computation.
    """

    def forward(
        self,
        upper_points: torch.Tensor,
        lower_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            upper_points: (B, N, 3) upper arch surface points.
            lower_points: (B, M, 3) lower arch surface points.

        Returns:
            Scalar uniformity loss (variance of inter-arch distances).
        """
        # Find nearest lower point for each upper point
        knn_result = knn_points(upper_points, lower_points, K=1)
        nn_dists = knn_result.dists.squeeze(-1)  # (B, N)

        # Variance of distances across the arch
        loss = nn_dists.var(dim=-1).mean()
        return loss


# ─── Collision / penetration loss ────────────────────────────────────────────


class CollisionLoss(nn.Module):
    """
    Differentiable penetration penalty using knn_points from pytorch3d.

    Detects points that penetrate into the opposing arch by checking if
    nearest-neighbor distance is below a threshold, indicating interpenetration.
    """

    def __init__(self, penetration_threshold_mm: float = 0.5) -> None:
        super().__init__()
        self.threshold = penetration_threshold_mm

    def forward(
        self,
        points_a: torch.Tensor,
        points_b: torch.Tensor,
        normals_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            points_a: (B, N, 3) surface points of object A.
            points_b: (B, M, 3) surface points of object B.
            normals_b: (B, M, 3) surface normals of object B (optional, for
                       sign-based penetration detection).

        Returns:
            Scalar collision penalty loss.
        """
        knn_result = knn_points(points_a, points_b, K=1)
        nn_dists = knn_result.dists.squeeze(-1).sqrt()  # (B, N) in mm
        nn_idx = knn_result.idx.squeeze(-1)  # (B, N)

        if normals_b is not None:
            # Sign-based: check if point A is on the inside of surface B
            # by comparing displacement direction to surface normal
            nn_points_b = torch.gather(
                points_b, 1, nn_idx.unsqueeze(-1).expand(-1, -1, 3)
            )
            nn_normals_b = torch.gather(
                normals_b, 1, nn_idx.unsqueeze(-1).expand(-1, -1, 3)
            )
            displacement = points_a - nn_points_b  # (B, N, 3)
            # Negative dot product with normal = penetration
            dot = (displacement * nn_normals_b).sum(dim=-1)  # (B, N)
            penetrating = (dot < 0).float()
            penetration_depth = nn_dists * penetrating
        else:
            # Without normals: penalize any points closer than threshold
            penetration_depth = torch.relu(self.threshold - nn_dists)

        loss = penetration_depth.pow(2).mean()
        return loss


# ─── Midline deviation loss ─────────────────────────────────────────────────


class MidlineDeviationLoss(nn.Module):
    """
    L2 distance between upper and lower arch midlines.

    The midline is computed as the average position of the central incisors
    (FDI 11, 21 for upper; 31, 41 for lower). Deviations indicate
    asymmetric mandible positioning.
    """

    def forward(
        self,
        upper_midline: torch.Tensor,
        lower_midline: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            upper_midline: (B, 3) upper midline point (between 11 and 21).
            lower_midline: (B, 3) lower midline point (between 31 and 41).

        Returns:
            Scalar midline deviation loss.
        """
        # Only penalize lateral (X) and AP (Y) deviation, not vertical
        deviation = upper_midline[:, :2] - lower_midline[:, :2]  # (B, 2)
        loss = deviation.pow(2).sum(dim=-1).mean()
        return loss


# ─── Molar relation loss ────────────────────────────────────────────────────


class MolarRelationLoss(nn.Module):
    """
    Cusp-fossa landmark distance for Angle Class I molar relationship.

    Penalizes deviation from ideal cusp-fossa alignment where the
    mesiobuccal cusp of the upper first molar (FDI 16/26) occludes in
    the buccal groove of the lower first molar (FDI 46/36).

    Per PMC11574221 objective function.
    """

    def __init__(self, target_ap_offset_mm: float = 0.0) -> None:
        """
        Args:
            target_ap_offset_mm: Target anteroposterior offset for Class I.
                0.0 = ideal Class I, positive = Class III direction.
        """
        super().__init__()
        self.target_offset = target_ap_offset_mm

    def forward(
        self,
        upper_molar_landmarks: torch.Tensor,
        lower_molar_landmarks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            upper_molar_landmarks: (B, K, 3) landmarks on upper first molars.
            lower_molar_landmarks: (B, K, 3) landmarks on lower first molars.
                K = number of cusp/fossa landmarks (typically 5).

        Returns:
            Scalar molar relation loss.
        """
        # Chamfer-like distance between molar landmarks
        dists_sq = torch.cdist(upper_molar_landmarks, lower_molar_landmarks).pow(2)
        min_upper_to_lower = dists_sq.min(dim=-1).values.mean(dim=-1)  # (B,)
        min_lower_to_upper = dists_sq.min(dim=-2).values.mean(dim=-1)  # (B,)
        landmark_loss = (min_upper_to_lower + min_lower_to_upper) / 2

        # AP offset loss for Class I relationship
        upper_centroid = upper_molar_landmarks.mean(dim=1)  # (B, 3)
        lower_centroid = lower_molar_landmarks.mean(dim=1)
        ap_offset = lower_centroid[:, 1] - upper_centroid[:, 1]  # Y = AP axis
        offset_loss = (ap_offset - self.target_offset).pow(2)

        loss = (landmark_loss + offset_loss).mean()
        return loss


# ─── Dental arch curve Fréchet distance loss ────────────────────────────────


class DentalArchCurveLoss(nn.Module):
    """
    Fréchet distance between predicted and target dental arch curves.

    Per arxiv 2312.15139 (TADPM): measures how well the predicted tooth
    positions trace the correct parabolic dental arch form. The arch curve
    is the centroid trajectory through all teeth in FDI order.
    """

    def forward(
        self,
        predicted_centroids: torch.Tensor,
        target_centroids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differentiable approximation of Fréchet distance between arch curves.

        Args:
            predicted_centroids: (B, T, 3) predicted tooth centroids in FDI order.
            target_centroids: (B, T, 3) target tooth centroids.

        Returns:
            Scalar arch curve loss.
        """
        B, T, _ = predicted_centroids.shape

        # Dynamic programming Fréchet distance approximation
        # Use squared distances for differentiability
        pairwise_dist = torch.cdist(predicted_centroids, target_centroids)  # (B, T, T)

        # Soft-DTW-style computation (differentiable Fréchet approximation)
        # Initialize DP table
        dp = torch.full((B, T, T), float("inf"), device=predicted_centroids.device)
        dp[:, 0, 0] = pairwise_dist[:, 0, 0]

        for i in range(T):
            for j in range(T):
                if i == 0 and j == 0:
                    continue
                candidates = []
                if i > 0:
                    candidates.append(dp[:, i - 1, j])
                if j > 0:
                    candidates.append(dp[:, i, j - 1])
                if i > 0 and j > 0:
                    candidates.append(dp[:, i - 1, j - 1])
                if candidates:
                    dp[:, i, j] = torch.stack(candidates, dim=-1).min(dim=-1).values
                    dp[:, i, j] = torch.max(dp[:, i, j], pairwise_dist[:, i, j])

        loss = dp[:, -1, -1].mean()
        return loss


# ─── Composite dental loss ──────────────────────────────────────────────────


class CompositeDentalLoss(nn.Module):
    """
    Weighted combination of all dental occlusion losses.

    Per PMC11574221 objective function:
    f(θ) = w1*chamfer + w2*overlap + w3*uniformity + w4*collision
         + w5*midline + w6*molar + w7*arch_curve

    Weights are configurable per patient condition. Default weights follow
    the priority: occlusion > collision > symmetry > aesthetics.
    """

    def __init__(
        self,
        w_chamfer: float = 1.0,
        w_overlap: float = 2.0,
        w_uniformity: float = 1.5,
        w_collision: float = 3.0,
        w_midline: float = 1.0,
        w_molar: float = 2.0,
        w_arch_curve: float = 0.5,
    ) -> None:
        super().__init__()
        self.weights = {
            "chamfer": w_chamfer,
            "overlap": w_overlap,
            "uniformity": w_uniformity,
            "collision": w_collision,
            "midline": w_midline,
            "molar": w_molar,
            "arch_curve": w_arch_curve,
        }

        self.chamfer_loss = ChamferOcclusionLoss()
        self.overlap_loss = OcclusalProjectionOverlapLoss()
        self.uniformity_loss = OcclusalDistanceUniformityLoss()
        self.collision_loss = CollisionLoss()
        self.midline_loss = MidlineDeviationLoss()
        self.molar_loss = MolarRelationLoss()
        self.arch_curve_loss = DentalArchCurveLoss()

    def forward(
        self,
        upper_points: torch.Tensor,
        lower_points: torch.Tensor,
        upper_midline: Optional[torch.Tensor] = None,
        lower_midline: Optional[torch.Tensor] = None,
        upper_molar_landmarks: Optional[torch.Tensor] = None,
        lower_molar_landmarks: Optional[torch.Tensor] = None,
        predicted_centroids: Optional[torch.Tensor] = None,
        target_centroids: Optional[torch.Tensor] = None,
        normals_lower: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute weighted composite loss.

        Args:
            upper_points: (B, N, 3) upper arch surface.
            lower_points: (B, M, 3) lower arch surface.
            upper_midline: (B, 3) upper midline point (optional).
            lower_midline: (B, 3) lower midline point (optional).
            upper_molar_landmarks: (B, K, 3) upper molar landmarks (optional).
            lower_molar_landmarks: (B, K, 3) lower molar landmarks (optional).
            predicted_centroids: (B, T, 3) predicted arch curve (optional).
            target_centroids: (B, T, 3) target arch curve (optional).
            normals_lower: (B, M, 3) lower surface normals for collision (optional).

        Returns:
            total_loss: Scalar weighted sum.
            loss_dict: Dict of individual named losses for logging.
        """
        loss_dict: Dict[str, torch.Tensor] = {}
        device = upper_points.device
        total = torch.tensor(0.0, device=device)

        # Chamfer
        l_chamfer = self.chamfer_loss(upper_points, lower_points)
        loss_dict["chamfer"] = l_chamfer
        total = total + self.weights["chamfer"] * l_chamfer

        # Overlap
        l_overlap = self.overlap_loss(upper_points, lower_points)
        loss_dict["overlap"] = l_overlap
        total = total + self.weights["overlap"] * l_overlap

        # Uniformity
        l_uniform = self.uniformity_loss(upper_points, lower_points)
        loss_dict["uniformity"] = l_uniform
        total = total + self.weights["uniformity"] * l_uniform

        # Collision
        l_collision = self.collision_loss(upper_points, lower_points, normals_lower)
        loss_dict["collision"] = l_collision
        total = total + self.weights["collision"] * l_collision

        # Midline (if landmarks provided)
        if upper_midline is not None and lower_midline is not None:
            l_midline = self.midline_loss(upper_midline, lower_midline)
            loss_dict["midline"] = l_midline
            total = total + self.weights["midline"] * l_midline

        # Molar relation (if landmarks provided)
        if upper_molar_landmarks is not None and lower_molar_landmarks is not None:
            l_molar = self.molar_loss(upper_molar_landmarks, lower_molar_landmarks)
            loss_dict["molar"] = l_molar
            total = total + self.weights["molar"] * l_molar

        # Arch curve (if centroids provided)
        if predicted_centroids is not None and target_centroids is not None:
            l_curve = self.arch_curve_loss(predicted_centroids, target_centroids)
            loss_dict["arch_curve"] = l_curve
            total = total + self.weights["arch_curve"] * l_curve

        loss_dict["total"] = total
        return total, loss_dict
