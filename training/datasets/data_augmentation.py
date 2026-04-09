"""
3D-specific data augmentations for dental and fracture training.

Augmentations are designed for point cloud and mesh data in the
dental/CMF domain:
- Random SE(3) perturbation of fragments (simulate fracture configs)
- Random point cloud jitter, dropout, subsampling
- Random tooth removal (simulate missing teeth)
- Elastic deformation of mesh surfaces
- Noise injection on dental surfaces
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class AugmentationConfig:
    """Configuration for 3D augmentations."""

    # SE(3) perturbation
    enable_se3_perturbation: bool = True
    rotation_range_deg: float = 5.0
    translation_range_mm: float = 2.0

    # Point cloud jitter
    enable_jitter: bool = True
    jitter_std_mm: float = 0.1

    # Point dropout
    enable_dropout: bool = True
    dropout_ratio: float = 0.05

    # Subsampling
    enable_subsampling: bool = False
    subsample_ratio: float = 0.8

    # Tooth removal (simulate missing teeth)
    enable_tooth_removal: bool = True
    tooth_removal_prob: float = 0.1

    # Elastic deformation
    enable_elastic: bool = False
    elastic_sigma_mm: float = 1.0
    elastic_alpha: float = 0.5

    # Surface noise
    enable_surface_noise: bool = True
    surface_noise_std_mm: float = 0.05


def augment_point_cloud(
    points: torch.Tensor,
    config: AugmentationConfig,
) -> torch.Tensor:
    """
    Apply configured augmentations to a point cloud.

    Args:
        points: (N, 3) point cloud tensor.
        config: augmentation configuration.

    Returns:
        (N', 3) augmented point cloud (N' may differ due to dropout/subsampling).
    """
    if config.enable_se3_perturbation:
        points = _random_se3_perturbation(
            points,
            rotation_range_deg=config.rotation_range_deg,
            translation_range_mm=config.translation_range_mm,
        )

    if config.enable_jitter:
        points = _random_jitter(points, std=config.jitter_std_mm)

    if config.enable_dropout and points.shape[0] > 100:
        points = _random_dropout(points, ratio=config.dropout_ratio)

    if config.enable_subsampling and points.shape[0] > 100:
        points = _random_subsample(points, ratio=config.subsample_ratio)

    if config.enable_elastic:
        points = _elastic_deformation(
            points, sigma=config.elastic_sigma_mm, alpha=config.elastic_alpha,
        )

    if config.enable_surface_noise:
        points = _surface_noise(points, std=config.surface_noise_std_mm)

    return points


def augment_tooth_clouds(
    tooth_clouds: Dict[int, torch.Tensor],
    config: AugmentationConfig,
) -> Dict[int, torch.Tensor]:
    """
    Apply augmentations to a dictionary of per-tooth point clouds.

    Optionally removes random teeth to simulate missing dentition.

    Args:
        tooth_clouds: FDI number → (N, 3) point cloud.
        config: augmentation configuration.

    Returns:
        Augmented tooth clouds (some may be removed).
    """
    result: Dict[int, torch.Tensor] = {}

    for fdi, pts in tooth_clouds.items():
        # Random tooth removal
        if config.enable_tooth_removal and torch.rand(1).item() < config.tooth_removal_prob:
            continue

        result[fdi] = augment_point_cloud(pts, config)

    return result


def augment_transform(
    transform_4x4: np.ndarray,
    rotation_noise_deg: float = 1.0,
    translation_noise_mm: float = 0.5,
) -> np.ndarray:
    """
    Add small noise to a ground truth SE(3) transform for regularization.

    Args:
        transform_4x4: (4, 4) ground truth transform.
        rotation_noise_deg: std of rotation noise in degrees.
        translation_noise_mm: std of translation noise in mm.

    Returns:
        (4, 4) perturbed transform.
    """
    T = transform_4x4.copy()

    # Rotation noise via axis-angle
    axis = np.random.randn(3)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    angle = np.random.randn() * np.radians(rotation_noise_deg)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R_noise = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    T[:3, :3] = R_noise @ T[:3, :3]

    # Translation noise
    T[:3, 3] += np.random.randn(3) * translation_noise_mm

    return T


# ─── Internal augmentation functions ────────────────────────────────────────


def _random_se3_perturbation(
    points: torch.Tensor,
    rotation_range_deg: float = 5.0,
    translation_range_mm: float = 2.0,
) -> torch.Tensor:
    """Apply a random rigid body transform to the point cloud."""
    device = points.device

    # Random axis-angle rotation
    axis = torch.randn(3, device=device)
    axis = axis / (axis.norm() + 1e-8)
    angle = torch.randn(1, device=device) * (rotation_range_deg * np.pi / 180.0)

    # Rodrigues formula
    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ], device=device, dtype=points.dtype)

    R = (
        torch.eye(3, device=device, dtype=points.dtype)
        + torch.sin(angle) * K
        + (1 - torch.cos(angle)) * (K @ K)
    )

    # Random translation
    t = torch.randn(3, device=device, dtype=points.dtype) * translation_range_mm

    return points @ R.T + t


def _random_jitter(
    points: torch.Tensor,
    std: float = 0.1,
) -> torch.Tensor:
    """Add Gaussian noise to each point."""
    noise = torch.randn_like(points) * std
    return points + noise


def _random_dropout(
    points: torch.Tensor,
    ratio: float = 0.05,
) -> torch.Tensor:
    """Randomly drop a fraction of points."""
    n = points.shape[0]
    keep = max(int(n * (1.0 - ratio)), 1)
    idx = torch.randperm(n, device=points.device)[:keep]
    return points[idx]


def _random_subsample(
    points: torch.Tensor,
    ratio: float = 0.8,
) -> torch.Tensor:
    """Subsample to a fraction of the original points."""
    n = points.shape[0]
    target = max(int(n * ratio), 1)
    idx = torch.randperm(n, device=points.device)[:target]
    return points[idx]


def _elastic_deformation(
    points: torch.Tensor,
    sigma: float = 1.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Apply elastic deformation to a point cloud.

    Generates a smooth displacement field via random offsets
    convolved with a Gaussian kernel, then applies it.
    """
    device = points.device
    n = points.shape[0]

    # Random displacement field
    displacement = torch.randn(n, 3, device=device, dtype=points.dtype)

    # Smooth the displacement field using k-NN averaging
    # Simple approximation: use distance-weighted averaging with nearby points
    k = min(16, n)
    dists = torch.cdist(points, points)  # (N, N)
    _, nn_idx = dists.topk(k, dim=-1, largest=False)  # (N, k)

    # Average displacement over neighbors (smoothing)
    gathered = displacement[nn_idx]  # (N, k, 3)
    weights = torch.exp(-dists.gather(1, nn_idx) / (2 * sigma**2))  # (N, k)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
    smoothed = (gathered * weights.unsqueeze(-1)).sum(dim=1)  # (N, 3)

    return points + alpha * smoothed


def _surface_noise(
    points: torch.Tensor,
    std: float = 0.05,
) -> torch.Tensor:
    """
    Add surface-oriented noise (along estimated normals).

    Approximates surface normals using local PCA, then perturbs
    each point along its normal direction.
    """
    device = points.device
    n = points.shape[0]

    if n < 10:
        return points + torch.randn_like(points) * std

    # Estimate normals via local PCA (k nearest neighbors)
    k = min(16, n)
    dists = torch.cdist(points, points)
    _, nn_idx = dists.topk(k, dim=-1, largest=False)

    normals = torch.zeros(n, 3, device=device, dtype=points.dtype)
    for i in range(n):
        neighbors = points[nn_idx[i]]  # (k, 3)
        centered = neighbors - neighbors.mean(dim=0, keepdim=True)
        try:
            _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
            normals[i] = Vh[-1]  # Smallest singular vector ≈ normal
        except Exception:
            normals[i] = torch.tensor([0.0, 0.0, 1.0], device=device)

    # Perturb along normal
    noise_mag = torch.randn(n, 1, device=device, dtype=points.dtype) * std
    return points + noise_mag * normals
