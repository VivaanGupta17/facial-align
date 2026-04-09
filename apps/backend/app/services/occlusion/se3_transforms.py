"""
SE(3) rigid body transform utilities for dental tooth pose prediction.

Thin wrappers around pytorch3d.transforms for the dental domain.
All rotations use the continuous 6D representation (Zhou et al., 2019)
for gradient-friendly optimization.

References:
- Zhou et al., "On the Continuity of Rotation Representations" (CVPR 2019)
- pytorch3d.transforms: se3_exp_map, rotation_6d_to_matrix, etc.
- arxiv 2312.15139 (TADPM): SE(3) transform prediction for teeth
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from pytorch3d.transforms import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    se3_exp_map,
    so3_exp_map,
)


# ─── Core transform application ─────────────────────────────────────────────


def apply_se3_transform(
    points: torch.Tensor,
    transform_params: torch.Tensor,
) -> torch.Tensor:
    """
    Apply an SE(3) transform to a point cloud.

    Uses pytorch3d's se3_exp_map to convert a 6-vector (3 rotation + 3 translation
    in the Lie algebra) into a 4x4 transformation matrix, then applies it.

    Args:
        points: (B, N, 3) point cloud.
        transform_params: (B, 6) Lie algebra parameters [omega_x, omega_y, omega_z,
                          v_x, v_y, v_z] where omega is rotation (axis-angle)
                          and v is translation.

    Returns:
        (B, N, 3) transformed points.
    """
    # se3_exp_map expects (B, 6) → returns (B, 4, 4)
    T = se3_exp_map(transform_params)  # (B, 4, 4)
    R = T[:, :3, :3]  # (B, 3, 3)
    t = T[:, :3, 3:]  # (B, 3, 1)

    # Apply: p' = R @ p + t
    transformed = torch.bmm(points, R.transpose(1, 2)) + t.transpose(1, 2)
    return transformed


def apply_rotation_translation(
    points: torch.Tensor,
    rotation_6d: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotation (6D representation) + translation to points.

    Uses the continuous 6D rotation representation which avoids gimbal lock
    and discontinuities in the rotation space.

    Args:
        points: (B, N, 3) point cloud.
        rotation_6d: (B, 6) continuous rotation representation.
        translation: (B, 3) translation vector in mm.

    Returns:
        (B, N, 3) transformed points.
    """
    R = rotation_6d_to_matrix(rotation_6d)  # (B, 3, 3)
    transformed = torch.bmm(points, R.transpose(1, 2)) + translation.unsqueeze(1)
    return transformed


def compose_transforms(
    T1: torch.Tensor,
    T2: torch.Tensor,
) -> torch.Tensor:
    """
    Compose two SE(3) transforms: T_composed = T2 @ T1.

    Args:
        T1: (B, 4, 4) first transform (applied first).
        T2: (B, 4, 4) second transform (applied second).

    Returns:
        (B, 4, 4) composed transform.
    """
    return torch.bmm(T2, T1)


def invert_transform(T: torch.Tensor) -> torch.Tensor:
    """
    Invert an SE(3) transform.

    For a rigid transform [R|t], the inverse is [R^T | -R^T @ t].

    Args:
        T: (B, 4, 4) transform matrix.

    Returns:
        (B, 4, 4) inverted transform.
    """
    R = T[:, :3, :3]  # (B, 3, 3)
    t = T[:, :3, 3:]  # (B, 3, 1)

    R_inv = R.transpose(1, 2)
    t_inv = -torch.bmm(R_inv, t)

    T_inv = torch.zeros_like(T)
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3:] = t_inv
    T_inv[:, 3, 3] = 1.0
    return T_inv


# ─── 6D rotation utilities ──────────────────────────────────────────────────


def rotation_matrix_to_6d(R: torch.Tensor) -> torch.Tensor:
    """
    Convert 3x3 rotation matrix to continuous 6D representation.
    Wraps pytorch3d.transforms.matrix_to_rotation_6d.

    Args:
        R: (B, 3, 3) rotation matrices.

    Returns:
        (B, 6) continuous rotation representation.
    """
    return matrix_to_rotation_6d(R)


def rotation_6d_to_mat(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D continuous representation back to 3x3 rotation matrix.
    Wraps pytorch3d.transforms.rotation_6d_to_matrix.

    Args:
        rot_6d: (B, 6) continuous rotation.

    Returns:
        (B, 3, 3) rotation matrices (orthonormal).
    """
    return rotation_6d_to_matrix(rot_6d)


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle (Rodrigues) representation to rotation matrix.
    Uses pytorch3d's so3_exp_map.

    Args:
        axis_angle: (B, 3) axis-angle vector (direction = axis, magnitude = angle).

    Returns:
        (B, 3, 3) rotation matrix.
    """
    return so3_exp_map(axis_angle)


# ─── Per-tooth transform application ────────────────────────────────────────


def per_tooth_transform(
    tooth_points: List[torch.Tensor],
    per_tooth_params: torch.Tensor,
    param_type: str = "6d_translation",
) -> List[torch.Tensor]:
    """
    Apply N separate SE(3) transforms to N tooth point clouds.

    Each tooth gets its own 6-DoF rigid body transform, parameterized
    as (6D rotation + 3D translation) = 9 parameters per tooth.

    Args:
        tooth_points: List of (P_i, 3) point clouds, one per tooth.
        per_tooth_params: (N, 9) transform parameters per tooth.
            Columns [0:6] = 6D continuous rotation.
            Columns [6:9] = translation (mm).
        param_type: "6d_translation" (default) or "se3_lie" (6-param Lie algebra).

    Returns:
        List of (P_i, 3) transformed point clouds.
    """
    N = len(tooth_points)
    assert per_tooth_params.shape[0] == N, (
        f"Expected {N} transform params, got {per_tooth_params.shape[0]}"
    )

    transformed = []
    for i in range(N):
        pts = tooth_points[i]  # (P_i, 3)
        if pts.dim() == 2:
            pts = pts.unsqueeze(0)  # (1, P_i, 3)

        if param_type == "6d_translation":
            rot_6d = per_tooth_params[i, :6].unsqueeze(0)  # (1, 6)
            trans = per_tooth_params[i, 6:9].unsqueeze(0)  # (1, 3)
            result = apply_rotation_translation(pts, rot_6d, trans)
        elif param_type == "se3_lie":
            lie_params = per_tooth_params[i, :6].unsqueeze(0)  # (1, 6)
            result = apply_se3_transform(pts, lie_params)
        else:
            raise ValueError(f"Unknown param_type: {param_type}")

        transformed.append(result.squeeze(0))

    return transformed


# ─── Transform head module ───────────────────────────────────────────────────


class SE3TransformHead(nn.Module):
    """
    Neural network head that predicts per-tooth SE(3) transforms.

    Takes per-tooth feature embeddings and outputs 6D rotation + 3D translation
    for each tooth. Used as the final layer of the occlusion transformer.

    Output: 9 parameters per tooth (6D rotation + 3D translation).
    The 6D rotation is converted to a 3x3 matrix via Gram-Schmidt
    orthogonalization (pytorch3d rotation_6d_to_matrix).
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.rotation_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6),  # 6D continuous rotation
        )
        self.translation_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # Translation in mm
        )

        # Initialize to identity: 6D rep of identity rotation is [1,0,0,0,1,0]
        nn.init.zeros_(self.rotation_head[-1].weight)
        nn.init.zeros_(self.rotation_head[-1].bias)
        self.rotation_head[-1].bias.data.copy_(
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        )

        nn.init.zeros_(self.translation_head[-1].weight)
        nn.init.zeros_(self.translation_head[-1].bias)

    def forward(
        self, tooth_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tooth_features: (B, N_teeth, D) per-tooth embeddings.

        Returns:
            rotation_6d: (B, N_teeth, 6) predicted rotations.
            translation: (B, N_teeth, 3) predicted translations in mm.
        """
        rot = self.rotation_head(tooth_features)  # (B, N_teeth, 6)
        trans = self.translation_head(tooth_features)  # (B, N_teeth, 3)
        return rot, trans

    def to_transform_matrices(
        self,
        rotation_6d: torch.Tensor,
        translation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert predicted parameters to 4x4 transform matrices.

        Args:
            rotation_6d: (B, N, 6) rotation parameters.
            translation: (B, N, 3) translation parameters.

        Returns:
            (B, N, 4, 4) homogeneous transform matrices.
        """
        B, N, _ = rotation_6d.shape
        R = rotation_6d_to_matrix(rotation_6d.reshape(-1, 6))  # (B*N, 3, 3)
        R = R.reshape(B, N, 3, 3)

        T = torch.zeros(B, N, 4, 4, device=rotation_6d.device)
        T[:, :, :3, :3] = R
        T[:, :, :3, 3] = translation
        T[:, :, 3, 3] = 1.0
        return T
