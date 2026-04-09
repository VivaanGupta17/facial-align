"""
Apply predicted SE(3) transforms to bone and tooth meshes.

Given original STL meshes and per-fragment/tooth transforms predicted by the
supervised model, produces repositioned geometry ready for clinical review
and STL export.

Transform convention
--------------------
p_reduced = R @ p_current + t

where R is a 3x3 rotation matrix (SO(3)) and t is a translation vector in
patient-space millimetres.  Transforms are represented as 4x4 homogeneous
SE(3) matrices throughout this module.

Interpolation
-------------
Transform interpolation uses SLERP (via scipy) for the rotation component
and linear interpolation for translation, ensuring smooth intermediate poses
for animation and incremental adjustment.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as ScipyRotation
from scipy.spatial.transform import Slerp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum allowable translation magnitude (mm) before flagging
MAX_TRANSLATION_MM: float = 80.0

# Maximum allowable rotation angle (degrees) before flagging
MAX_ROTATION_DEG: float = 90.0

# Tolerance for orthonormality check of rotation matrices
ORTHONORMALITY_ATOL: float = 1e-6

# Tolerance for determinant check (det(R) must equal 1)
DETERMINANT_ATOL: float = 1e-6


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TransformValidationResult:
    """Result of validating an SE(3) transform."""

    is_valid: bool
    rotation_valid: bool
    translation_valid: bool
    det_R: float
    orthonormality_error: float
    rotation_angle_deg: float
    translation_magnitude_mm: float
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class RepositionResult:
    """Result of applying a transform to a mesh."""

    mesh: trimesh.Trimesh
    fragment_id: str
    transform_4x4: np.ndarray
    rotation_angle_deg: float
    translation_magnitude_mm: float
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# TransformApplicator
# ---------------------------------------------------------------------------

class TransformApplicator:
    """
    Applies predicted SE(3) transforms to bone/tooth meshes.

    Given original STL meshes and per-fragment/tooth transforms predicted
    by the supervised model, produces repositioned geometry.

    Thread-safe: no mutable instance state beyond configuration.
    """

    def __init__(
        self,
        max_translation_mm: float = MAX_TRANSLATION_MM,
        max_rotation_deg: float = MAX_ROTATION_DEG,
        orthonormality_atol: float = ORTHONORMALITY_ATOL,
        determinant_atol: float = DETERMINANT_ATOL,
    ) -> None:
        """
        Initialise the transform applicator.

        Args:
            max_translation_mm: Flag transforms with translation above this.
            max_rotation_deg: Flag transforms with rotation above this.
            orthonormality_atol: Tolerance for R^T R = I check.
            determinant_atol: Tolerance for det(R) = 1 check.
        """
        self._max_translation_mm = max_translation_mm
        self._max_rotation_deg = max_rotation_deg
        self._ortho_atol = orthonormality_atol
        self._det_atol = determinant_atol

    # ------------------------------------------------------------------
    # Public: Apply transforms
    # ------------------------------------------------------------------

    def apply_fragment_transforms(
        self,
        fragments: Dict[str, trimesh.Trimesh],
        transforms: Dict[str, np.ndarray],
    ) -> Dict[str, RepositionResult]:
        """
        Apply per-fragment SE(3) transforms to a set of bone meshes.

        Args:
            fragments: Mapping of fragment_id -> original trimesh mesh.
            transforms: Mapping of fragment_id -> 4x4 homogeneous transform.

        Returns:
            Mapping of fragment_id -> RepositionResult with transformed mesh.

        Raises:
            ValueError: If a transform is invalid or fragment_id mismatch.
        """
        results: Dict[str, RepositionResult] = {}
        for frag_id, mesh in fragments.items():
            if frag_id not in transforms:
                logger.warning("No transform for fragment '%s'; keeping original position", frag_id)
                results[frag_id] = RepositionResult(
                    mesh=mesh.copy(),
                    fragment_id=frag_id,
                    transform_4x4=np.eye(4),
                    rotation_angle_deg=0.0,
                    translation_magnitude_mm=0.0,
                    elapsed_seconds=0.0,
                )
                continue

            T = np.asarray(transforms[frag_id], dtype=np.float64)
            validation = self.validate_transform(T)
            if not validation.is_valid:
                raise ValueError(
                    f"Invalid transform for fragment '{frag_id}': "
                    + "; ".join(validation.errors)
                )
            for w in validation.warnings:
                logger.warning("Fragment '%s' transform warning: %s", frag_id, w)

            t0 = time.monotonic()
            transformed_mesh = self._apply_transform_to_mesh(mesh, T)
            elapsed = time.monotonic() - t0

            results[frag_id] = RepositionResult(
                mesh=transformed_mesh,
                fragment_id=frag_id,
                transform_4x4=T,
                rotation_angle_deg=validation.rotation_angle_deg,
                translation_magnitude_mm=validation.translation_magnitude_mm,
                elapsed_seconds=elapsed,
            )
            logger.info(
                "Fragment '%s' repositioned: rot=%.2f° trans=%.2fmm (%.3fs)",
                frag_id,
                validation.rotation_angle_deg,
                validation.translation_magnitude_mm,
                elapsed,
            )

        return results

    def apply_tooth_transforms(
        self,
        teeth: Dict[str, trimesh.Trimesh],
        transforms: Dict[str, np.ndarray],
    ) -> Dict[str, RepositionResult]:
        """
        Apply per-tooth SE(3) transforms to a set of dental meshes.

        Identical logic to fragment transforms but with separate logging
        context for traceability.

        Args:
            teeth: Mapping of tooth_id (e.g. FDI number string) -> trimesh mesh.
            transforms: Mapping of tooth_id -> 4x4 homogeneous transform.

        Returns:
            Mapping of tooth_id -> RepositionResult with transformed mesh.
        """
        results: Dict[str, RepositionResult] = {}
        for tooth_id, mesh in teeth.items():
            if tooth_id not in transforms:
                logger.warning("No transform for tooth '%s'; keeping original", tooth_id)
                results[tooth_id] = RepositionResult(
                    mesh=mesh.copy(),
                    fragment_id=tooth_id,
                    transform_4x4=np.eye(4),
                    rotation_angle_deg=0.0,
                    translation_magnitude_mm=0.0,
                    elapsed_seconds=0.0,
                )
                continue

            T = np.asarray(transforms[tooth_id], dtype=np.float64)
            validation = self.validate_transform(T)
            if not validation.is_valid:
                raise ValueError(
                    f"Invalid transform for tooth '{tooth_id}': "
                    + "; ".join(validation.errors)
                )

            t0 = time.monotonic()
            transformed_mesh = self._apply_transform_to_mesh(mesh, T)
            elapsed = time.monotonic() - t0

            results[tooth_id] = RepositionResult(
                mesh=transformed_mesh,
                fragment_id=tooth_id,
                transform_4x4=T,
                rotation_angle_deg=validation.rotation_angle_deg,
                translation_magnitude_mm=validation.translation_magnitude_mm,
                elapsed_seconds=elapsed,
            )
            logger.info(
                "Tooth '%s' repositioned: rot=%.2f° trans=%.2fmm (%.3fs)",
                tooth_id,
                validation.rotation_angle_deg,
                validation.translation_magnitude_mm,
                elapsed,
            )

        return results

    # ------------------------------------------------------------------
    # Public: Compose and invert
    # ------------------------------------------------------------------

    def compose_transforms(
        self,
        T1: np.ndarray,
        T2: np.ndarray,
    ) -> np.ndarray:
        """
        Compose two SE(3) transforms: T_composed = T2 @ T1.

        The result applies T1 first, then T2.

        Args:
            T1: 4x4 homogeneous transform (applied first).
            T2: 4x4 homogeneous transform (applied second).

        Returns:
            4x4 composed homogeneous transform.
        """
        T1 = np.asarray(T1, dtype=np.float64)
        T2 = np.asarray(T2, dtype=np.float64)
        composed = T2 @ T1
        # Re-orthonormalise the rotation block to prevent drift
        composed[:3, :3] = self._closest_rotation(composed[:3, :3])
        return composed

    def invert_transform(self, T: np.ndarray) -> np.ndarray:
        """
        Invert an SE(3) transform.

        For [R|t], the inverse is [R^T | -R^T @ t].

        Args:
            T: 4x4 homogeneous transform.

        Returns:
            4x4 inverted transform.
        """
        T = np.asarray(T, dtype=np.float64)
        R = T[:3, :3]
        t = T[:3, 3]
        R_inv = R.T
        t_inv = -R_inv @ t
        T_inv = np.eye(4)
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = t_inv
        return T_inv

    # ------------------------------------------------------------------
    # Public: Validation
    # ------------------------------------------------------------------

    def validate_transform(self, T: np.ndarray) -> TransformValidationResult:
        """
        Validate an SE(3) transform for clinical use.

        Checks:
        - Matrix shape is 4x4
        - Bottom row is [0, 0, 0, 1]
        - R^T R = I (orthonormality)
        - det(R) = +1 (proper rotation, no reflection)
        - Translation magnitude within clinical threshold
        - Rotation angle within clinical threshold

        Args:
            T: 4x4 homogeneous transform matrix.

        Returns:
            TransformValidationResult with pass/fail and diagnostics.
        """
        T = np.asarray(T, dtype=np.float64)
        warnings: List[str] = []
        errors: List[str] = []
        rotation_valid = True
        translation_valid = True

        # Shape check
        if T.shape != (4, 4):
            return TransformValidationResult(
                is_valid=False,
                rotation_valid=False,
                translation_valid=False,
                det_R=0.0,
                orthonormality_error=float("inf"),
                rotation_angle_deg=0.0,
                translation_magnitude_mm=0.0,
                errors=[f"Expected (4,4) matrix, got {T.shape}"],
            )

        # Bottom row
        if not np.allclose(T[3, :], [0, 0, 0, 1], atol=1e-8):
            errors.append(f"Bottom row must be [0,0,0,1], got {T[3,:].tolist()}")

        R = T[:3, :3]
        t = T[:3, 3]

        # Orthonormality: R^T R should be I
        RtR = R.T @ R
        ortho_error = float(np.max(np.abs(RtR - np.eye(3))))
        if ortho_error > self._ortho_atol:
            errors.append(f"Rotation not orthonormal: max|R^TR - I| = {ortho_error:.2e}")
            rotation_valid = False

        # Determinant: must be +1 (not -1, which would be a reflection)
        det_R = float(np.linalg.det(R))
        if abs(det_R - 1.0) > self._det_atol:
            errors.append(f"det(R) = {det_R:.6f}, expected 1.0")
            rotation_valid = False

        # Rotation angle
        trace_val = float(np.trace(R))
        cos_angle = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
        rotation_angle_deg = float(np.degrees(np.arccos(cos_angle)))

        # Translation magnitude
        translation_mm = float(np.linalg.norm(t))

        # Clinical range warnings
        if rotation_angle_deg > self._max_rotation_deg:
            warnings.append(
                f"Rotation {rotation_angle_deg:.1f}° exceeds {self._max_rotation_deg}° threshold"
            )
        if translation_mm > self._max_translation_mm:
            warnings.append(
                f"Translation {translation_mm:.1f}mm exceeds {self._max_translation_mm}mm threshold"
            )
            translation_valid = False

        is_valid = len(errors) == 0
        return TransformValidationResult(
            is_valid=is_valid,
            rotation_valid=rotation_valid,
            translation_valid=translation_valid,
            det_R=det_R,
            orthonormality_error=ortho_error,
            rotation_angle_deg=rotation_angle_deg,
            translation_magnitude_mm=translation_mm,
            warnings=warnings,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Public: Interpolation
    # ------------------------------------------------------------------

    def interpolate_transforms(
        self,
        T_start: np.ndarray,
        T_end: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """
        Interpolate between two SE(3) transforms.

        Uses SLERP for the rotation component and linear interpolation
        for translation.

        Args:
            T_start: 4x4 starting transform.
            T_end: 4x4 ending transform.
            alpha: Interpolation parameter in [0, 1].
                   0 = T_start, 1 = T_end.

        Returns:
            4x4 interpolated transform.
        """
        alpha = float(np.clip(alpha, 0.0, 1.0))
        T_start = np.asarray(T_start, dtype=np.float64)
        T_end = np.asarray(T_end, dtype=np.float64)

        R_start = T_start[:3, :3]
        R_end = T_end[:3, :3]
        t_start = T_start[:3, 3]
        t_end = T_end[:3, 3]

        # SLERP for rotation via scipy
        rot_start = ScipyRotation.from_matrix(R_start)
        rot_end = ScipyRotation.from_matrix(R_end)
        slerp = Slerp([0.0, 1.0], ScipyRotation.concatenate([rot_start, rot_end]))
        R_interp = slerp([alpha])[0].as_matrix()

        # Linear interpolation for translation
        t_interp = (1.0 - alpha) * t_start + alpha * t_end

        T_interp = np.eye(4)
        T_interp[:3, :3] = R_interp
        T_interp[:3, 3] = t_interp
        return T_interp

    def interpolate_trajectory(
        self,
        T_start: np.ndarray,
        T_end: np.ndarray,
        n_steps: int,
    ) -> List[np.ndarray]:
        """
        Generate a sequence of interpolated transforms between start and end.

        Useful for animation or incremental repositioning visualisation.

        Args:
            T_start: 4x4 starting transform.
            T_end: 4x4 ending transform.
            n_steps: Number of intermediate steps (including endpoints).

        Returns:
            List of n_steps 4x4 transforms from T_start to T_end.
        """
        if n_steps < 2:
            return [T_start.copy(), T_end.copy()]
        alphas = np.linspace(0.0, 1.0, n_steps)
        return [self.interpolate_transforms(T_start, T_end, float(a)) for a in alphas]

    # ------------------------------------------------------------------
    # Public: Utility conversions
    # ------------------------------------------------------------------

    @staticmethod
    def rotation_translation_to_4x4(
        R: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        """
        Build a 4x4 homogeneous SE(3) matrix from R (3x3) and t (3,).

        Args:
            R: 3x3 rotation matrix.
            t: 3-element translation vector (mm).

        Returns:
            4x4 homogeneous transform matrix.
        """
        T = np.eye(4)
        T[:3, :3] = np.asarray(R, dtype=np.float64)
        T[:3, 3] = np.asarray(t, dtype=np.float64).ravel()[:3]
        return T

    @staticmethod
    def decompose_4x4(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract rotation matrix and translation vector from a 4x4 transform.

        Args:
            T: 4x4 homogeneous transform.

        Returns:
            (R, t) where R is (3,3) and t is (3,).
        """
        T = np.asarray(T, dtype=np.float64)
        return T[:3, :3].copy(), T[:3, 3].copy()

    @staticmethod
    def axis_angle_to_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
        """
        Convert axis-angle representation to 3x3 rotation matrix (Rodrigues).

        Args:
            axis: 3-element unit axis vector.
            angle_rad: Rotation angle in radians.

        Returns:
            3x3 rotation matrix.
        """
        axis = np.asarray(axis, dtype=np.float64)
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            return np.eye(3)
        axis = axis / norm
        rotvec = axis * angle_rad
        return ScipyRotation.from_rotvec(rotvec).as_matrix()

    @staticmethod
    def matrix_to_axis_angle(R: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Convert 3x3 rotation matrix to axis-angle representation.

        Args:
            R: 3x3 rotation matrix.

        Returns:
            (axis, angle_rad) — unit axis and angle in radians.
        """
        rotvec = ScipyRotation.from_matrix(np.asarray(R, dtype=np.float64)).as_rotvec()
        angle = float(np.linalg.norm(rotvec))
        if angle < 1e-12:
            return np.array([0.0, 0.0, 1.0]), 0.0
        return rotvec / angle, angle

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_transform_to_mesh(
        mesh: trimesh.Trimesh,
        T: np.ndarray,
    ) -> trimesh.Trimesh:
        """
        Apply a 4x4 SE(3) transform to a trimesh mesh.

        Creates a copy so the original is not modified.

        Args:
            mesh: Source trimesh mesh.
            T: 4x4 homogeneous transform.

        Returns:
            New trimesh.Trimesh with transformed vertices and normals.
        """
        transformed = mesh.copy()
        transformed.apply_transform(T)
        return transformed

    @staticmethod
    def _closest_rotation(M: np.ndarray) -> np.ndarray:
        """
        Project a 3x3 matrix onto SO(3) via SVD (closest rotation matrix).

        Used to correct numerical drift after composing many transforms.

        Args:
            M: 3x3 matrix (approximately a rotation).

        Returns:
            3x3 proper rotation matrix.
        """
        U, _, Vt = np.linalg.svd(M)
        # Ensure proper rotation (det = +1)
        d = np.linalg.det(U @ Vt)
        S = np.diag([1.0, 1.0, np.sign(d)])
        return U @ S @ Vt
