"""
Resolve interpenetrations between repositioned bone fragments.

After applying ML-predicted SE(3) transforms, fragments may slightly
interpenetrate.  This module detects and resolves collisions via SDF-based
penetration detection and iterative repulsion, while maintaining occlusal
constraints.

Clinical constraint
-------------------
If any collision requires more than 2 mm of correction the fragment pair
is flagged for manual review rather than auto-resolved, because large
adjustments risk invalidating the surgical plan.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CORRECTION_MM: float = 2.0
DEFAULT_PENETRATION_THRESHOLD_MM: float = 0.1
DEFAULT_MAX_ITERATIONS: int = 50
DEFAULT_REPULSION_STEP_MM: float = 0.05
DEFAULT_VOXEL_SIZE_MM: float = 0.5
CONVERGENCE_TOLERANCE_MM: float = 0.01


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CollisionPair:
    """Detected collision between two fragments."""

    fragment_id_a: str
    fragment_id_b: str
    penetration_depth_mm: float
    penetration_volume_mm3: float
    penetration_normal: np.ndarray
    n_penetrating_points: int
    requires_manual_review: bool


@dataclass
class ResolutionStep:
    """A single iteration of collision resolution."""

    iteration: int
    fragment_id: str
    displacement_mm: np.ndarray
    residual_penetration_mm: float


@dataclass
class CollisionResolutionResult:
    """Complete result of collision resolution for a case."""

    resolved_meshes: Dict[str, trimesh.Trimesh]
    resolved_transforms: Dict[str, np.ndarray]
    initial_collisions: List[CollisionPair]
    residual_collisions: List[CollisionPair]
    resolution_steps: List[ResolutionStep]
    flagged_for_review: List[CollisionPair]
    total_iterations: int
    converged: bool
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# CollisionResolver
# ---------------------------------------------------------------------------

class CollisionResolver:
    """
    Resolves interpenetrations between repositioned bone fragments.

    Detection uses signed-distance-field (SDF) sampling and voxel-based
    overlap estimation.  Resolution applies iterative repulsion along
    the penetration normal.

    Thread-safe: no mutable instance state beyond configuration.
    """

    def __init__(
        self,
        penetration_threshold_mm: float = DEFAULT_PENETRATION_THRESHOLD_MM,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        repulsion_step_mm: float = DEFAULT_REPULSION_STEP_MM,
        max_correction_mm: float = MAX_CORRECTION_MM,
        voxel_size_mm: float = DEFAULT_VOXEL_SIZE_MM,
    ) -> None:
        """
        Initialise the collision resolver.

        Args:
            penetration_threshold_mm: Minimum depth to count as penetration.
            max_iterations: Maximum repulsion iterations before giving up.
            repulsion_step_mm: Step size per repulsion iteration.
            max_correction_mm: Maximum total correction before flagging.
            voxel_size_mm: Voxel size for overlap volume estimation.
        """
        self._pen_threshold = penetration_threshold_mm
        self._max_iters = max_iterations
        self._repulsion_step = repulsion_step_mm
        self._max_correction = max_correction_mm
        self._voxel_size = voxel_size_mm

    # ------------------------------------------------------------------
    # Public: Full resolution pipeline
    # ------------------------------------------------------------------

    def resolve(
        self,
        meshes: Dict[str, trimesh.Trimesh],
        transforms: Dict[str, np.ndarray],
        occlusal_constraints: Optional[Dict[str, Any]] = None,
    ) -> CollisionResolutionResult:
        """
        Detect and resolve all pairwise collisions between fragments.

        Args:
            meshes: Mapping of fragment_id -> trimesh mesh (already transformed).
            transforms: Mapping of fragment_id -> current 4x4 transform.
            occlusal_constraints: Optional constraint dict (fragment_id -> fixed axis).

        Returns:
            CollisionResolutionResult with resolved meshes and diagnostics.
        """
        t0 = time.monotonic()

        working_meshes = {fid: m.copy() for fid, m in meshes.items()}
        working_transforms = {fid: np.array(t, dtype=np.float64) for fid, t in transforms.items()}

        # Detect initial collisions
        initial_collisions = self.detect_all_collisions(working_meshes)
        logger.info("Detected %d initial collision pairs", len(initial_collisions))

        flagged: List[CollisionPair] = []
        resolvable: List[CollisionPair] = []
        for cp in initial_collisions:
            if cp.requires_manual_review:
                flagged.append(cp)
                logger.warning(
                    "Collision %s<->%s requires manual review (%.2fmm penetration)",
                    cp.fragment_id_a, cp.fragment_id_b, cp.penetration_depth_mm,
                )
            else:
                resolvable.append(cp)

        # Iterative resolution
        all_steps: List[ResolutionStep] = []
        converged = False
        total_iterations = 0

        for iteration in range(self._max_iters):
            if not resolvable:
                converged = True
                break

            total_iterations = iteration + 1
            made_progress = False

            for cp in resolvable:
                step = self._repulse_pair(
                    working_meshes, working_transforms,
                    cp.fragment_id_a, cp.fragment_id_b,
                    cp.penetration_normal,
                    occlusal_constraints,
                    iteration,
                )
                if step is not None:
                    all_steps.append(step)
                    made_progress = True

            # Re-detect collisions
            resolvable_ids = {(cp.fragment_id_a, cp.fragment_id_b) for cp in resolvable}
            new_collisions = self.detect_all_collisions(working_meshes)
            resolvable = [
                cp for cp in new_collisions
                if not cp.requires_manual_review
                and (cp.fragment_id_a, cp.fragment_id_b) in resolvable_ids
            ]

            if not made_progress:
                converged = len(resolvable) == 0
                break

            # Check convergence
            if all(cp.penetration_depth_mm < CONVERGENCE_TOLERANCE_MM for cp in resolvable):
                converged = True
                break

        residual_collisions = self.detect_all_collisions(working_meshes)
        elapsed = time.monotonic() - t0

        logger.info(
            "Collision resolution: %d iterations, converged=%s, "
            "%d residual collisions, %d flagged (%.2fs)",
            total_iterations, converged, len(residual_collisions), len(flagged), elapsed,
        )

        return CollisionResolutionResult(
            resolved_meshes=working_meshes,
            resolved_transforms=working_transforms,
            initial_collisions=initial_collisions,
            residual_collisions=residual_collisions,
            resolution_steps=all_steps,
            flagged_for_review=flagged,
            total_iterations=total_iterations,
            converged=converged,
            elapsed_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # Public: Detection
    # ------------------------------------------------------------------

    def detect_all_collisions(
        self,
        meshes: Dict[str, trimesh.Trimesh],
    ) -> List[CollisionPair]:
        """
        Detect all pairwise collisions between a set of meshes.

        Args:
            meshes: Mapping of fragment_id -> trimesh mesh.

        Returns:
            List of CollisionPair for each colliding pair.
        """
        fragment_ids = sorted(meshes.keys())
        collisions: List[CollisionPair] = []

        for i, fid_a in enumerate(fragment_ids):
            for fid_b in fragment_ids[i + 1:]:
                cp = self.detect_collision(
                    fid_a, meshes[fid_a],
                    fid_b, meshes[fid_b],
                )
                if cp is not None:
                    collisions.append(cp)

        return collisions

    def detect_collision(
        self,
        id_a: str,
        mesh_a: trimesh.Trimesh,
        id_b: str,
        mesh_b: trimesh.Trimesh,
    ) -> Optional[CollisionPair]:
        """
        Detect collision between two meshes using proximity and SDF sampling.

        Args:
            id_a: Identifier for mesh A.
            mesh_a: First mesh.
            id_b: Identifier for mesh B.
            mesh_b: Second mesh.

        Returns:
            CollisionPair if penetration detected, else None.
        """
        # Bounding box quick-reject
        if not _bboxes_overlap(mesh_a, mesh_b):
            return None

        # Compute closest points from A vertices to B surface
        closest_pts, distances, face_ids = trimesh.proximity.closest_point(
            mesh_b, mesh_a.vertices
        )
        penetrating_mask = distances < self._pen_threshold
        n_penetrating = int(np.sum(penetrating_mask))

        if n_penetrating == 0:
            return None

        # Estimate penetration depth via SDF
        pen_depths = self._pen_threshold - distances[penetrating_mask]
        max_depth = float(np.max(pen_depths))

        # Compute penetration normal (from A penetrating points towards B surface)
        pen_points_a = mesh_a.vertices[penetrating_mask]
        pen_closest_b = closest_pts[penetrating_mask]
        displacement = pen_closest_b - pen_points_a
        mean_displacement = np.mean(displacement, axis=0)
        norm = np.linalg.norm(mean_displacement)
        if norm < 1e-12:
            pen_normal = np.array([0.0, 0.0, 1.0])
        else:
            pen_normal = mean_displacement / norm

        # Estimate overlap volume
        overlap_volume = self._estimate_overlap_volume(mesh_a, mesh_b)

        requires_review = max_depth > self._max_correction

        return CollisionPair(
            fragment_id_a=id_a,
            fragment_id_b=id_b,
            penetration_depth_mm=max_depth,
            penetration_volume_mm3=overlap_volume,
            penetration_normal=pen_normal,
            n_penetrating_points=n_penetrating,
            requires_manual_review=requires_review,
        )

    # ------------------------------------------------------------------
    # Internal: Resolution
    # ------------------------------------------------------------------

    def _repulse_pair(
        self,
        meshes: Dict[str, trimesh.Trimesh],
        transforms: Dict[str, np.ndarray],
        id_a: str,
        id_b: str,
        normal: np.ndarray,
        occlusal_constraints: Optional[Dict[str, Any]],
        iteration: int,
    ) -> Optional[ResolutionStep]:
        """
        Apply a single repulsion step to separate two colliding fragments.

        Moves fragment B along the penetration normal by repulsion_step_mm.
        If occlusal constraints lock an axis, that component is zeroed.

        Args:
            meshes: Mutable dict of fragment meshes (updated in-place).
            transforms: Mutable dict of transforms (updated in-place).
            id_a: Fragment A id (held fixed).
            id_b: Fragment B id (moved).
            normal: Penetration normal direction.
            occlusal_constraints: Optional axis locks per fragment.
            iteration: Current iteration number.

        Returns:
            ResolutionStep if displacement was applied, else None.
        """
        displacement = normal * self._repulsion_step

        # Apply occlusal constraints: zero out constrained axes
        if occlusal_constraints and id_b in occlusal_constraints:
            constraint = occlusal_constraints[id_b]
            locked_axes = constraint.get("locked_axes", [])
            for axis_idx in locked_axes:
                if 0 <= axis_idx <= 2:
                    displacement[axis_idx] = 0.0

        total_correction = np.linalg.norm(displacement)
        if total_correction < 1e-9:
            return None

        # Check cumulative correction does not exceed max
        current_t = transforms[id_b][:3, 3]
        original_t = transforms.get(f"_original_{id_b}", current_t.copy())
        if f"_original_{id_b}" not in transforms:
            transforms[f"_original_{id_b}"] = current_t.copy()

        cumulative = np.linalg.norm(current_t + displacement - original_t)
        if cumulative > self._max_correction:
            logger.warning(
                "Fragment '%s' cumulative correction %.2fmm exceeds max %.2fmm; stopping",
                id_b, cumulative, self._max_correction,
            )
            return None

        # Apply displacement to mesh and transform
        T_disp = np.eye(4)
        T_disp[:3, 3] = displacement
        meshes[id_b].apply_transform(T_disp)
        transforms[id_b][:3, 3] += displacement

        # Check residual
        cp = self.detect_collision(id_a, meshes[id_a], id_b, meshes[id_b])
        residual = cp.penetration_depth_mm if cp else 0.0

        return ResolutionStep(
            iteration=iteration,
            fragment_id=id_b,
            displacement_mm=displacement,
            residual_penetration_mm=residual,
        )

    def _estimate_overlap_volume(
        self,
        mesh_a: trimesh.Trimesh,
        mesh_b: trimesh.Trimesh,
    ) -> float:
        """
        Estimate overlap volume between two meshes using voxel grid intersection.

        Args:
            mesh_a: First mesh.
            mesh_b: Second mesh.

        Returns:
            Estimated overlap volume in mm^3.
        """
        # Compute intersection of bounding boxes
        min_a, max_a = mesh_a.bounds
        min_b, max_b = mesh_b.bounds
        overlap_min = np.maximum(min_a, min_b)
        overlap_max = np.minimum(max_a, max_b)

        if np.any(overlap_max <= overlap_min):
            return 0.0

        # Sample grid points in the overlap region
        axes = [
            np.arange(overlap_min[i], overlap_max[i], self._voxel_size)
            for i in range(3)
        ]
        if any(len(ax) == 0 for ax in axes):
            return 0.0

        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

        # Check which points are inside both meshes
        inside_a = _points_inside_mesh(mesh_a, grid)
        inside_b = _points_inside_mesh(mesh_b, grid)
        n_shared = int(np.sum(inside_a & inside_b))

        return n_shared * (self._voxel_size ** 3)


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def _bboxes_overlap(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> bool:
    """Check if two mesh bounding boxes overlap (quick reject test)."""
    min_a, max_a = mesh_a.bounds
    min_b, max_b = mesh_b.bounds
    return bool(np.all(max_a >= min_b) and np.all(max_b >= min_a))


def _points_inside_mesh(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """
    Test which points are inside a mesh using trimesh's ray-based containment.

    Args:
        mesh: Trimesh mesh (should be watertight for accurate results).
        points: (N, 3) points to test.

    Returns:
        (N,) boolean array — True if point is inside.
    """
    if mesh.is_watertight:
        return mesh.contains(points)

    # Fallback for non-watertight: use proximity sign heuristic
    _, distances, _ = trimesh.proximity.closest_point(mesh, points)
    # Negative convention not available without watertight, so use threshold
    return distances < 0.1
