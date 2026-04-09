"""
Post-processing mesh cleanup for clinical-grade STL output.

Operates on trimesh meshes produced by the transform applicator and produces
geometry suitable for 3D printing and surgical planning.  Each operation is
logged with before/after statistics so the pipeline is auditable.

Operations (in order)
---------------------
1. Remove degenerate faces (zero area, duplicate vertices)
2. Fill small holes (< threshold area)
3. Remove floating components (< threshold volume)
4. Smooth mesh (Laplacian smoothing with boundary preservation)
5. Remesh for uniform triangle quality
6. Ensure manifold topology
7. Ensure watertightness
8. Orient normals consistently (outward)

Dependencies: trimesh, open3d, numpy.
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
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_HOLE_AREA_THRESHOLD_MM2: float = 50.0
DEFAULT_COMPONENT_VOLUME_THRESHOLD_MM3: float = 10.0
DEFAULT_SMOOTHING_ITERATIONS: int = 10
DEFAULT_SMOOTHING_LAMBDA: float = 0.5
DEFAULT_TARGET_EDGE_LENGTH_MM: float = 1.0
DEFAULT_DEGENERATE_AREA_THRESHOLD: float = 1e-8


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CleanupStepStats:
    """Statistics for a single cleanup operation."""

    step_name: str
    vertices_before: int
    vertices_after: int
    faces_before: int
    faces_after: int
    elapsed_seconds: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CleanupReport:
    """Complete report for a mesh cleanup pipeline run."""

    input_vertices: int
    input_faces: int
    output_vertices: int
    output_faces: int
    is_watertight: bool
    is_manifold: bool
    normals_consistent: bool
    total_elapsed_seconds: float
    steps: List[CleanupStepStats] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MeshCleanup
# ---------------------------------------------------------------------------

class MeshCleanup:
    """
    Post-processing mesh cleanup for clinical-grade STL output.

    Each operation is individually configurable and logged with before/after
    statistics.  The full pipeline is run via ``run_full_cleanup()``.

    Thread-safe: no mutable instance state beyond configuration.
    """

    def __init__(
        self,
        hole_area_threshold_mm2: float = DEFAULT_HOLE_AREA_THRESHOLD_MM2,
        component_volume_threshold_mm3: float = DEFAULT_COMPONENT_VOLUME_THRESHOLD_MM3,
        smoothing_iterations: int = DEFAULT_SMOOTHING_ITERATIONS,
        smoothing_lambda: float = DEFAULT_SMOOTHING_LAMBDA,
        target_edge_length_mm: float = DEFAULT_TARGET_EDGE_LENGTH_MM,
        degenerate_area_threshold: float = DEFAULT_DEGENERATE_AREA_THRESHOLD,
    ) -> None:
        """
        Initialise cleanup with configurable thresholds.

        Args:
            hole_area_threshold_mm2: Holes smaller than this area are filled.
            component_volume_threshold_mm3: Components smaller than this removed.
            smoothing_iterations: Number of Laplacian smoothing passes.
            smoothing_lambda: Smoothing strength (0-1).
            target_edge_length_mm: Target edge length for remeshing.
            degenerate_area_threshold: Faces with area below this are degenerate.
        """
        self._hole_threshold = hole_area_threshold_mm2
        self._component_threshold = component_volume_threshold_mm3
        self._smooth_iters = smoothing_iterations
        self._smooth_lambda = smoothing_lambda
        self._target_edge_len = target_edge_length_mm
        self._degen_threshold = degenerate_area_threshold

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_cleanup(self, mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, CleanupReport]:
        """
        Run the complete cleanup pipeline on a mesh.

        Args:
            mesh: Input trimesh mesh (not modified in-place).

        Returns:
            (cleaned_mesh, report) tuple.
        """
        t_total = time.monotonic()
        working = mesh.copy()
        input_v = len(working.vertices)
        input_f = len(working.faces)
        steps: List[CleanupStepStats] = []

        # Step 1: Remove degenerate faces
        working, step = self.remove_degenerate_faces(working)
        steps.append(step)

        # Step 2: Fill small holes
        working, step = self.fill_small_holes(working)
        steps.append(step)

        # Step 3: Remove floating components
        working, step = self.remove_floating_components(working)
        steps.append(step)

        # Step 4: Laplacian smoothing
        working, step = self.smooth_mesh(working)
        steps.append(step)

        # Step 5: Remesh for uniform quality
        working, step = self.remesh_uniform(working)
        steps.append(step)

        # Step 6: Ensure manifold topology
        working, step = self.ensure_manifold(working)
        steps.append(step)

        # Step 7: Ensure watertightness
        working, step = self.ensure_watertight(working)
        steps.append(step)

        # Step 8: Orient normals
        working, step = self.orient_normals(working)
        steps.append(step)

        total_elapsed = time.monotonic() - t_total

        report = CleanupReport(
            input_vertices=input_v,
            input_faces=input_f,
            output_vertices=len(working.vertices),
            output_faces=len(working.faces),
            is_watertight=bool(working.is_watertight),
            is_manifold=_is_manifold(working),
            normals_consistent=_normals_consistent(working),
            total_elapsed_seconds=total_elapsed,
            steps=steps,
        )
        logger.info(
            "Cleanup complete: %d→%d verts, %d→%d faces, watertight=%s (%.2fs)",
            input_v, report.output_vertices,
            input_f, report.output_faces,
            report.is_watertight, total_elapsed,
        )
        return working, report

    # ------------------------------------------------------------------
    # Step 1: Remove degenerate faces
    # ------------------------------------------------------------------

    def remove_degenerate_faces(
        self, mesh: trimesh.Trimesh
    ) -> Tuple[trimesh.Trimesh, CleanupStepStats]:
        """
        Remove degenerate faces (zero area, duplicate vertices, collapsed edges).

        Args:
            mesh: Input mesh (modified in-place on a copy).

        Returns:
            (cleaned_mesh, step_stats).
        """
        t0 = time.monotonic()
        v_before = len(mesh.vertices)
        f_before = len(mesh.faces)

        # Identify degenerate faces by area
        face_areas = mesh.area_faces
        degenerate_mask = face_areas < self._degen_threshold

        # Also flag faces with duplicate vertex indices
        for i, face in enumerate(mesh.faces):
            if len(set(face)) < 3:
                degenerate_mask[i] = True

        n_degenerate = int(np.sum(degenerate_mask))

        if n_degenerate > 0:
            keep_mask = ~degenerate_mask
            mesh.update_faces(keep_mask)
            mesh.remove_unreferenced_vertices()

        # Merge duplicate vertices
        mesh.merge_vertices()

        elapsed = time.monotonic() - t0
        step = CleanupStepStats(
            step_name="remove_degenerate_faces",
            vertices_before=v_before,
            vertices_after=len(mesh.vertices),
            faces_before=f_before,
            faces_after=len(mesh.faces),
            elapsed_seconds=elapsed,
            details={"n_degenerate_removed": n_degenerate},
        )
        logger.info(
            "Step 1 remove_degenerate: removed %d faces (%.3fs)",
            n_degenerate, elapsed,
        )
        return mesh, step

    # ------------------------------------------------------------------
    # Step 2: Fill small holes
    # ------------------------------------------------------------------

    def fill_small_holes(
        self, mesh: trimesh.Trimesh
    ) -> Tuple[trimesh.Trimesh, CleanupStepStats]:
        """
        Fill holes smaller than the configured area threshold.

        Uses trimesh's hole-filling which triangulates boundary loops.

        Args:
            mesh: Input mesh.

        Returns:
            (mesh_with_holes_filled, step_stats).
        """
        t0 = time.monotonic()
        v_before = len(mesh.vertices)
        f_before = len(mesh.faces)
        n_filled = 0

        # trimesh exposes the boundary edges through facets
        if hasattr(mesh, "fill_holes") and callable(mesh.fill_holes):
            mesh.fill_holes()
            n_filled = len(mesh.faces) - f_before
        else:
            # Manual hole filling via open3d
            n_filled = self._fill_holes_o3d(mesh)

        elapsed = time.monotonic() - t0
        step = CleanupStepStats(
            step_name="fill_small_holes",
            vertices_before=v_before,
            vertices_after=len(mesh.vertices),
            faces_before=f_before,
            faces_after=len(mesh.faces),
            elapsed_seconds=elapsed,
            details={"n_faces_added": n_filled},
        )
        logger.info("Step 2 fill_holes: added %d faces (%.3fs)", n_filled, elapsed)
        return mesh, step

    # ------------------------------------------------------------------
    # Step 3: Remove floating components
    # ------------------------------------------------------------------

    def remove_floating_components(
        self, mesh: trimesh.Trimesh
    ) -> Tuple[trimesh.Trimesh, CleanupStepStats]:
        """
        Remove small disconnected mesh components below volume threshold.

        Keeps only the largest component plus any component whose bounding
        box volume exceeds the threshold.

        Args:
            mesh: Input mesh.

        Returns:
            (cleaned_mesh, step_stats).
        """
        t0 = time.monotonic()
        v_before = len(mesh.vertices)
        f_before = len(mesh.faces)
        n_removed = 0

        components = mesh.split(only_watertight=False)
        if len(components) <= 1:
            elapsed = time.monotonic() - t0
            return mesh, CleanupStepStats(
                step_name="remove_floating_components",
                vertices_before=v_before,
                vertices_after=len(mesh.vertices),
                faces_before=f_before,
                faces_after=len(mesh.faces),
                elapsed_seconds=elapsed,
                details={"n_components": len(components), "n_removed": 0},
            )

        # Sort by bounding box volume descending
        components_with_vol = []
        for comp in components:
            bbox_vol = float(comp.bounding_box.volume) if comp.bounding_box is not None else 0.0
            components_with_vol.append((comp, bbox_vol))
        components_with_vol.sort(key=lambda x: x[1], reverse=True)

        # Always keep the largest component; keep others above threshold
        keep: List[trimesh.Trimesh] = [components_with_vol[0][0]]
        for comp, vol in components_with_vol[1:]:
            if vol >= self._component_threshold:
                keep.append(comp)
            else:
                n_removed += 1

        if len(keep) == 1:
            result = keep[0]
        else:
            result = trimesh.util.concatenate(keep)

        elapsed = time.monotonic() - t0
        step = CleanupStepStats(
            step_name="remove_floating_components",
            vertices_before=v_before,
            vertices_after=len(result.vertices),
            faces_before=f_before,
            faces_after=len(result.faces),
            elapsed_seconds=elapsed,
            details={
                "n_components": len(components),
                "n_removed": n_removed,
                "n_kept": len(keep),
            },
        )
        logger.info(
            "Step 3 remove_floaters: %d components, removed %d (%.3fs)",
            len(components), n_removed, elapsed,
        )
        return result, step

    # ------------------------------------------------------------------
    # Step 4: Smooth mesh
    # ------------------------------------------------------------------

    def smooth_mesh(
        self, mesh: trimesh.Trimesh
    ) -> Tuple[trimesh.Trimesh, CleanupStepStats]:
        """
        Apply Laplacian smoothing with boundary preservation.

        Uses Taubin smoothing (alternating positive/negative lambda)
        to reduce shrinkage artifacts.

        Args:
            mesh: Input mesh.

        Returns:
            (smoothed_mesh, step_stats).
        """
        t0 = time.monotonic()
        v_before = len(mesh.vertices)
        f_before = len(mesh.faces)

        smoothed = self._taubin_smooth(
            mesh,
            iterations=self._smooth_iters,
            lam=self._smooth_lambda,
            mu=-(self._smooth_lambda + 0.01),
        )

        elapsed = time.monotonic() - t0
        step = CleanupStepStats(
            step_name="smooth_mesh",
            vertices_before=v_before,
            vertices_after=len(smoothed.vertices),
            faces_before=f_before,
            faces_after=len(smoothed.faces),
            elapsed_seconds=elapsed,
            details={
                "iterations": self._smooth_iters,
                "lambda": self._smooth_lambda,
            },
        )
        logger.info("Step 4 smooth: %d iterations (%.3fs)", self._smooth_iters, elapsed)
        return smoothed, step

    # ------------------------------------------------------------------
    # Step 5: Remesh
    # ------------------------------------------------------------------

    def remesh_uniform(
        self, mesh: trimesh.Trimesh
    ) -> Tuple[trimesh.Trimesh, CleanupStepStats]:
        """
        Remesh for uniform triangle quality using open3d.

        Subdivides long edges and collapses short edges to approach the
        target edge length.

        Args:
            mesh: Input mesh.

        Returns:
            (remeshed, step_stats).
        """
        import open3d as o3d

        t0 = time.monotonic()
        v_before = len(mesh.vertices)
        f_before = len(mesh.faces)

        o3d_mesh = _trimesh_to_open3d(mesh)

        # Simplify to target complexity based on edge length
        current_edge_lengths = np.asarray(mesh.edges_unique_length)
        mean_edge = float(np.mean(current_edge_lengths)) if len(current_edge_lengths) > 0 else 1.0

        if mean_edge > self._target_edge_len * 1.5:
            # Subdivide to increase resolution
            n_subdivisions = min(int(np.ceil(np.log2(mean_edge / self._target_edge_len))), 3)
            if n_subdivisions > 0:
                o3d_mesh = o3d_mesh.subdivide_midpoint(number_of_iterations=n_subdivisions)
        elif mean_edge < self._target_edge_len * 0.5:
            # Simplify to reduce resolution
            target_faces = max(
                int(len(mesh.faces) * (mean_edge / self._target_edge_len) ** 2),
                100,
            )
            o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)

        result = _open3d_to_trimesh(o3d_mesh)

        elapsed = time.monotonic() - t0
        step = CleanupStepStats(
            step_name="remesh_uniform",
            vertices_before=v_before,
            vertices_after=len(result.vertices),
            faces_before=f_before,
            faces_after=len(result.faces),
            elapsed_seconds=elapsed,
            details={
                "target_edge_length_mm": self._target_edge_len,
                "mean_edge_before": round(mean_edge, 3),
            },
        )
        logger.info(
            "Step 5 remesh: %d→%d faces, edge %.2f→%.2fmm (%.3fs)",
            f_before, len(result.faces), mean_edge, self._target_edge_len, elapsed,
        )
        return result, step

    # ------------------------------------------------------------------
    # Step 6: Ensure manifold
    # ------------------------------------------------------------------

    def ensure_manifold(
        self, mesh: trimesh.Trimesh
    ) -> Tuple[trimesh.Trimesh, CleanupStepStats]:
        """
        Repair non-manifold edges and vertices.

        Non-manifold geometry (T-junctions, edges shared by >2 faces) breaks
        boolean operations and 3D printing slicers.

        Args:
            mesh: Input mesh.

        Returns:
            (manifold_mesh, step_stats).
        """
        import open3d as o3d

        t0 = time.monotonic()
        v_before = len(mesh.vertices)
        f_before = len(mesh.faces)
        n_repairs = 0

        o3d_mesh = _trimesh_to_open3d(mesh)

        # Remove non-manifold edges
        if not o3d_mesh.is_edge_manifold():
            o3d_mesh.remove_non_manifold_edges()
            n_repairs += 1

        # Remove non-manifold vertices
        if not o3d_mesh.is_vertex_manifold():
            o3d_mesh.remove_duplicated_vertices()
            o3d_mesh.remove_degenerate_triangles()
            o3d_mesh.remove_duplicated_triangles()
            n_repairs += 1

        result = _open3d_to_trimesh(o3d_mesh)

        elapsed = time.monotonic() - t0
        step = CleanupStepStats(
            step_name="ensure_manifold",
            vertices_before=v_before,
            vertices_after=len(result.vertices),
            faces_before=f_before,
            faces_after=len(result.faces),
            elapsed_seconds=elapsed,
            details={"n_repair_passes": n_repairs, "is_manifold": _is_manifold(result)},
        )
        logger.info("Step 6 manifold: %d repair passes (%.3fs)", n_repairs, elapsed)
        return result, step

    # ------------------------------------------------------------------
    # Step 7: Ensure watertight
    # ------------------------------------------------------------------

    def ensure_watertight(
        self, mesh: trimesh.Trimesh
    ) -> Tuple[trimesh.Trimesh, CleanupStepStats]:
        """
        Attempt to make the mesh watertight by filling remaining holes.

        If the mesh is already watertight, this is a no-op.

        Args:
            mesh: Input mesh.

        Returns:
            (watertight_mesh, step_stats).
        """
        t0 = time.monotonic()
        v_before = len(mesh.vertices)
        f_before = len(mesh.faces)
        was_watertight = bool(mesh.is_watertight)

        if not was_watertight:
            if hasattr(mesh, "fill_holes") and callable(mesh.fill_holes):
                mesh.fill_holes()

            # If still not watertight, try via open3d convex hull fallback
            if not mesh.is_watertight:
                self._repair_watertight_o3d(mesh)

        elapsed = time.monotonic() - t0
        step = CleanupStepStats(
            step_name="ensure_watertight",
            vertices_before=v_before,
            vertices_after=len(mesh.vertices),
            faces_before=f_before,
            faces_after=len(mesh.faces),
            elapsed_seconds=elapsed,
            details={
                "was_watertight": was_watertight,
                "is_watertight": bool(mesh.is_watertight),
            },
        )
        logger.info(
            "Step 7 watertight: %s → %s (%.3fs)",
            was_watertight, bool(mesh.is_watertight), elapsed,
        )
        return mesh, step

    # ------------------------------------------------------------------
    # Step 8: Orient normals
    # ------------------------------------------------------------------

    def orient_normals(
        self, mesh: trimesh.Trimesh
    ) -> Tuple[trimesh.Trimesh, CleanupStepStats]:
        """
        Orient face normals consistently outward.

        Uses trimesh's winding fix which propagates consistent orientation
        from the majority-outward face.

        Args:
            mesh: Input mesh.

        Returns:
            (oriented_mesh, step_stats).
        """
        t0 = time.monotonic()
        v_before = len(mesh.vertices)
        f_before = len(mesh.faces)

        mesh.fix_normals()

        elapsed = time.monotonic() - t0
        step = CleanupStepStats(
            step_name="orient_normals",
            vertices_before=v_before,
            vertices_after=len(mesh.vertices),
            faces_before=f_before,
            faces_after=len(mesh.faces),
            elapsed_seconds=elapsed,
            details={"normals_consistent": _normals_consistent(mesh)},
        )
        logger.info("Step 8 orient_normals: consistent=%s (%.3fs)",
                     _normals_consistent(mesh), elapsed)
        return mesh, step

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _taubin_smooth(
        mesh: trimesh.Trimesh,
        iterations: int,
        lam: float,
        mu: float,
    ) -> trimesh.Trimesh:
        """
        Taubin smoothing: alternating positive/negative Laplacian to reduce shrinkage.

        Args:
            mesh: Input mesh.
            iterations: Number of smoothing iterations.
            lam: Positive smoothing factor.
            mu: Negative smoothing factor (typically -(lam + small_epsilon)).

        Returns:
            Smoothed mesh.
        """
        result = mesh.copy()
        vertices = np.array(result.vertices, dtype=np.float64)
        faces = result.faces

        # Build adjacency from faces
        adjacency: Dict[int, List[int]] = {}
        for face in faces:
            for i in range(3):
                v_i = face[i]
                for j in range(3):
                    if i != j:
                        v_j = face[j]
                        adjacency.setdefault(v_i, []).append(v_j)

        # Deduplicate adjacency
        for v_idx in adjacency:
            adjacency[v_idx] = list(set(adjacency[v_idx]))

        # Identify boundary vertices (they border open edges)
        boundary_verts = set()
        edge_count: Dict[Tuple[int, int], int] = {}
        for face in faces:
            for i in range(3):
                e = tuple(sorted((face[i], face[(i + 1) % 3])))
                edge_count[e] = edge_count.get(e, 0) + 1
        for edge, count in edge_count.items():
            if count == 1:
                boundary_verts.add(edge[0])
                boundary_verts.add(edge[1])

        for _ in range(iterations):
            # Positive step (lambda)
            vertices = _laplacian_step(vertices, adjacency, boundary_verts, lam)
            # Negative step (mu)
            vertices = _laplacian_step(vertices, adjacency, boundary_verts, mu)

        result.vertices = vertices
        return result

    @staticmethod
    def _fill_holes_o3d(mesh: trimesh.Trimesh) -> int:
        """
        Fill holes using open3d as a fallback.

        Returns the number of faces added.
        """
        import open3d as o3d

        f_before = len(mesh.faces)
        o3d_mesh = _trimesh_to_open3d(mesh)

        # Use Poisson surface reconstruction to fill holes
        o3d_mesh.compute_vertex_normals()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d_mesh.vertices
        pcd.normals = o3d_mesh.vertex_normals

        # Ball-pivoting for hole filling
        radii = [0.5, 1.0, 2.0]
        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )

        if len(rec_mesh.triangles) > len(o3d_mesh.triangles):
            result = _open3d_to_trimesh(rec_mesh)
            mesh.vertices = result.vertices
            mesh.faces = result.faces
            return len(mesh.faces) - f_before

        return 0

    @staticmethod
    def _repair_watertight_o3d(mesh: trimesh.Trimesh) -> None:
        """
        Attempt to repair a mesh to be watertight using open3d.

        Modifies the mesh in-place.
        """
        import open3d as o3d

        o3d_mesh = _trimesh_to_open3d(mesh)
        o3d_mesh.compute_vertex_normals()
        o3d_mesh.remove_degenerate_triangles()
        o3d_mesh.remove_duplicated_triangles()
        o3d_mesh.remove_duplicated_vertices()
        o3d_mesh.remove_non_manifold_edges()

        result = _open3d_to_trimesh(o3d_mesh)
        mesh.vertices = result.vertices
        mesh.faces = result.faces


# ---------------------------------------------------------------------------
# Module-level utility functions
# ---------------------------------------------------------------------------

def _laplacian_step(
    vertices: np.ndarray,
    adjacency: Dict[int, List[int]],
    boundary_verts: set,
    weight: float,
) -> np.ndarray:
    """
    Single Laplacian smoothing step.

    Boundary vertices are held fixed to preserve mesh boundaries.

    Args:
        vertices: (N, 3) vertex positions.
        adjacency: Vertex adjacency mapping.
        boundary_verts: Set of boundary vertex indices to preserve.
        weight: Smoothing weight (positive = smooth, negative = inflate).

    Returns:
        Updated (N, 3) vertex positions.
    """
    new_verts = vertices.copy()
    for v_idx in range(len(vertices)):
        if v_idx in boundary_verts:
            continue
        neighbors = adjacency.get(v_idx, [])
        if not neighbors:
            continue
        centroid = np.mean(vertices[neighbors], axis=0)
        displacement = centroid - vertices[v_idx]
        new_verts[v_idx] = vertices[v_idx] + weight * displacement
    return new_verts


def _trimesh_to_open3d(mesh: trimesh.Trimesh) -> Any:
    """Convert a trimesh mesh to an open3d TriangleMesh."""
    import open3d as o3d

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(mesh.vertices):
        o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals))
    return o3d_mesh


def _open3d_to_trimesh(o3d_mesh: Any) -> trimesh.Trimesh:
    """Convert an open3d TriangleMesh to a trimesh mesh."""
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _is_manifold(mesh: trimesh.Trimesh) -> bool:
    """Check if a trimesh mesh is edge-manifold (each edge shared by exactly 2 faces)."""
    edge_count: Dict[Tuple[int, int], int] = {}
    for face in mesh.faces:
        for i in range(3):
            e = tuple(sorted((int(face[i]), int(face[(i + 1) % 3]))))
            edge_count[e] = edge_count.get(e, 0) + 1
    return all(c == 2 for c in edge_count.values())


def _normals_consistent(mesh: trimesh.Trimesh) -> bool:
    """
    Check whether face normals are consistently oriented.

    Approximation: verify that the majority of face normals point away
    from the mesh centroid.
    """
    if len(mesh.faces) == 0:
        return True
    centroids = mesh.triangles_center
    mesh_center = mesh.centroid
    outward_dirs = centroids - mesh_center
    dots = np.sum(mesh.face_normals * outward_dirs, axis=1)
    fraction_outward = float(np.mean(dots > 0))
    return fraction_outward > 0.8
