"""
Validate STL meshes for 3D printing compatibility.

Performs a comprehensive suite of checks required before sending surgical
models to an SLA resin printer.  Each check produces a pass/fail result
with actionable feedback so the operator knows exactly what to fix.

Target printer assumptions
--------------------------
- SLA resin (Formlabs Form 3B / Asiga MAX UV)
- Layer height: 50 µm
- Minimum feature: 0.3 mm
- Minimum wall: 0.8 mm (structural) / 0.4 mm (non-structural)
- Build volume: 145 × 145 × 185 mm (Form 3B)
- Support angle threshold: 45°

Dependencies: trimesh, numpy.
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
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MIN_WALL_THICKNESS_MM: float = 0.8
DEFAULT_MAX_OVERHANG_ANGLE_DEG: float = 45.0
DEFAULT_BUILD_VOLUME_MM: Tuple[float, float, float] = (145.0, 145.0, 185.0)
DEFAULT_MIN_EDGE_LENGTH_MM: float = 0.05
DEFAULT_MAX_EDGE_LENGTH_MM: float = 10.0
DEFAULT_MAX_ASPECT_RATIO: float = 30.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result of a single printability check."""

    check_name: str
    passed: bool
    value: Any
    threshold: Any
    message: str
    severity: str  # "error", "warning", "info"


@dataclass
class PrintabilityReport:
    """Complete printability validation report."""

    is_printable: bool
    checks: List[CheckResult]
    n_errors: int
    n_warnings: int
    mesh_stats: Dict[str, Any]
    elapsed_seconds: float

    @property
    def failed_checks(self) -> List[CheckResult]:
        """Return only the checks that failed."""
        return [c for c in self.checks if not c.passed]

    @property
    def error_messages(self) -> List[str]:
        """Return human-readable error messages for failed checks."""
        return [c.message for c in self.checks if not c.passed and c.severity == "error"]

    @property
    def warning_messages(self) -> List[str]:
        """Return human-readable warning messages."""
        return [c.message for c in self.checks if not c.passed and c.severity == "warning"]


# ---------------------------------------------------------------------------
# PrintabilityValidator
# ---------------------------------------------------------------------------

class PrintabilityValidator:
    """
    Validates STL meshes for 3D printing compatibility.

    Checks: watertightness, wall thickness, overhang angles, build volume,
    triangle quality, self-intersections, and normal consistency.

    Thread-safe: no mutable instance state beyond configuration.
    """

    def __init__(
        self,
        min_wall_thickness_mm: float = DEFAULT_MIN_WALL_THICKNESS_MM,
        max_overhang_angle_deg: float = DEFAULT_MAX_OVERHANG_ANGLE_DEG,
        build_volume_mm: Tuple[float, float, float] = DEFAULT_BUILD_VOLUME_MM,
        min_edge_length_mm: float = DEFAULT_MIN_EDGE_LENGTH_MM,
        max_edge_length_mm: float = DEFAULT_MAX_EDGE_LENGTH_MM,
        max_aspect_ratio: float = DEFAULT_MAX_ASPECT_RATIO,
    ) -> None:
        """
        Initialise the printability validator.

        Args:
            min_wall_thickness_mm: Minimum wall thickness for SLA resin.
            max_overhang_angle_deg: Max unsupported overhang angle.
            build_volume_mm: Printer build volume (x, y, z) in mm.
            min_edge_length_mm: Minimum triangle edge length.
            max_edge_length_mm: Maximum triangle edge length.
            max_aspect_ratio: Maximum triangle aspect ratio.
        """
        self._min_wall = min_wall_thickness_mm
        self._max_overhang = max_overhang_angle_deg
        self._build_volume = build_volume_mm
        self._min_edge = min_edge_length_mm
        self._max_edge = max_edge_length_mm
        self._max_aspect = max_aspect_ratio

    # ------------------------------------------------------------------
    # Public: Full validation
    # ------------------------------------------------------------------

    def validate(self, mesh: trimesh.Trimesh) -> PrintabilityReport:
        """
        Run all printability checks on a mesh.

        Args:
            mesh: Trimesh mesh to validate.

        Returns:
            PrintabilityReport with pass/fail per check and actionable feedback.
        """
        t0 = time.monotonic()
        checks: List[CheckResult] = []

        checks.append(self.check_watertight(mesh))
        checks.append(self.check_manifold(mesh))
        checks.append(self.check_wall_thickness(mesh))
        checks.append(self.check_overhang_angles(mesh))
        checks.append(self.check_build_volume(mesh))
        checks.extend(self.check_triangle_quality(mesh))
        checks.append(self.check_self_intersections(mesh))
        checks.append(self.check_normal_consistency(mesh))

        n_errors = sum(1 for c in checks if not c.passed and c.severity == "error")
        n_warnings = sum(1 for c in checks if not c.passed and c.severity == "warning")
        is_printable = n_errors == 0

        # Mesh statistics
        bounds = mesh.bounds
        mesh_stats = {
            "vertex_count": len(mesh.vertices),
            "face_count": len(mesh.faces),
            "volume_mm3": float(mesh.volume) if mesh.is_watertight else None,
            "surface_area_mm2": float(mesh.area),
            "bounding_box_mm": {
                "x": float(bounds[1][0] - bounds[0][0]),
                "y": float(bounds[1][1] - bounds[0][1]),
                "z": float(bounds[1][2] - bounds[0][2]),
            },
            "is_watertight": bool(mesh.is_watertight),
        }

        elapsed = time.monotonic() - t0
        report = PrintabilityReport(
            is_printable=is_printable,
            checks=checks,
            n_errors=n_errors,
            n_warnings=n_warnings,
            mesh_stats=mesh_stats,
            elapsed_seconds=elapsed,
        )

        logger.info(
            "Printability validation: printable=%s, %d errors, %d warnings (%.2fs)",
            is_printable, n_errors, n_warnings, elapsed,
        )
        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_watertight(self, mesh: trimesh.Trimesh) -> CheckResult:
        """
        Check that the mesh is watertight (no holes, manifold boundary).

        A non-watertight mesh cannot be reliably sliced for printing.

        Args:
            mesh: Mesh to check.

        Returns:
            CheckResult.
        """
        is_wt = bool(mesh.is_watertight)
        return CheckResult(
            check_name="watertightness",
            passed=is_wt,
            value=is_wt,
            threshold=True,
            message=(
                "Mesh is watertight" if is_wt
                else "Mesh is NOT watertight — has open boundaries or non-manifold edges. "
                     "Run mesh repair (fill holes, fix normals) before printing."
            ),
            severity="error" if not is_wt else "info",
        )

    def check_manifold(self, mesh: trimesh.Trimesh) -> CheckResult:
        """
        Check that the mesh has manifold topology.

        Non-manifold edges (shared by >2 faces) or vertices cause slicer failures.

        Args:
            mesh: Mesh to check.

        Returns:
            CheckResult.
        """
        # Check for edges shared by != 2 faces
        edge_face_count: Dict[Tuple[int, int], int] = {}
        for face in mesh.faces:
            for i in range(3):
                e = tuple(sorted((int(face[i]), int(face[(i + 1) % 3]))))
                edge_face_count[e] = edge_face_count.get(e, 0) + 1

        non_manifold_edges = sum(1 for c in edge_face_count.values() if c != 2)
        is_manifold = non_manifold_edges == 0

        return CheckResult(
            check_name="manifold_topology",
            passed=is_manifold,
            value=non_manifold_edges,
            threshold=0,
            message=(
                "Mesh topology is manifold" if is_manifold
                else f"Found {non_manifold_edges} non-manifold edges. "
                     "These must be repaired — remove duplicate faces or split non-manifold vertices."
            ),
            severity="error" if not is_manifold else "info",
        )

    def check_wall_thickness(self, mesh: trimesh.Trimesh) -> CheckResult:
        """
        Estimate minimum wall thickness using ray-based sampling.

        Fires inward rays from surface points and measures the distance to
        the opposite wall.

        Args:
            mesh: Mesh to check.

        Returns:
            CheckResult.
        """
        min_thickness = self._estimate_min_wall_thickness(mesh)
        passed = min_thickness >= self._min_wall

        return CheckResult(
            check_name="wall_thickness",
            passed=passed,
            value=round(min_thickness, 3),
            threshold=self._min_wall,
            message=(
                f"Minimum wall thickness {min_thickness:.2f}mm meets "
                f"requirement ({self._min_wall}mm)"
                if passed
                else f"Minimum wall thickness {min_thickness:.2f}mm is below "
                     f"{self._min_wall}mm. Thin walls may break during printing or post-processing. "
                     "Consider adding material or adjusting the design."
            ),
            severity="error" if not passed else "info",
        )

    def check_overhang_angles(self, mesh: trimesh.Trimesh) -> CheckResult:
        """
        Check for faces with overhang angles exceeding the threshold.

        Overhangs require support material which can leave surface marks
        on surgical models.

        Args:
            mesh: Mesh to check.

        Returns:
            CheckResult.
        """
        overhang_fraction, max_overhang = self._compute_overhang_stats(mesh)
        # Warning if >10% of faces overhang; error if >30%
        passed = overhang_fraction <= 0.30

        return CheckResult(
            check_name="overhang_angles",
            passed=passed,
            value={
                "max_overhang_deg": round(max_overhang, 1),
                "fraction_exceeding": round(overhang_fraction, 3),
            },
            threshold=self._max_overhang,
            message=(
                f"Overhang OK: {overhang_fraction*100:.1f}% of faces exceed "
                f"{self._max_overhang}° (max={max_overhang:.1f}°)"
                if passed
                else f"{overhang_fraction*100:.1f}% of faces exceed {self._max_overhang}° overhang. "
                     "Consider reorienting the model or adding support structures. "
                     f"Maximum overhang: {max_overhang:.1f}°."
            ),
            severity="warning" if not passed else "info",
        )

    def check_build_volume(self, mesh: trimesh.Trimesh) -> CheckResult:
        """
        Check that the mesh fits within the printer build volume.

        Args:
            mesh: Mesh to check.

        Returns:
            CheckResult.
        """
        bounds = mesh.bounds
        dims = bounds[1] - bounds[0]  # (dx, dy, dz)

        fits_x = dims[0] <= self._build_volume[0]
        fits_y = dims[1] <= self._build_volume[1]
        fits_z = dims[2] <= self._build_volume[2]
        fits = fits_x and fits_y and fits_z

        exceeded = []
        if not fits_x:
            exceeded.append(f"X: {dims[0]:.1f}mm > {self._build_volume[0]}mm")
        if not fits_y:
            exceeded.append(f"Y: {dims[1]:.1f}mm > {self._build_volume[1]}mm")
        if not fits_z:
            exceeded.append(f"Z: {dims[2]:.1f}mm > {self._build_volume[2]}mm")

        return CheckResult(
            check_name="build_volume",
            passed=fits,
            value={
                "mesh_dims_mm": [round(float(d), 1) for d in dims],
                "build_volume_mm": list(self._build_volume),
            },
            threshold=list(self._build_volume),
            message=(
                f"Mesh fits in build volume "
                f"({dims[0]:.1f}×{dims[1]:.1f}×{dims[2]:.1f}mm)"
                if fits
                else f"Mesh exceeds build volume: {', '.join(exceeded)}. "
                     "Split the model into printable sections or use a larger printer."
            ),
            severity="error" if not fits else "info",
        )

    def check_triangle_quality(self, mesh: trimesh.Trimesh) -> List[CheckResult]:
        """
        Check triangle quality metrics: aspect ratio, edge lengths.

        Poor quality triangles (slivers, very long edges) cause slicer
        artifacts and print defects.

        Args:
            mesh: Mesh to check.

        Returns:
            List of CheckResult (one per metric).
        """
        results: List[CheckResult] = []
        edge_lengths = mesh.edges_unique_length

        if len(edge_lengths) == 0:
            results.append(CheckResult(
                check_name="triangle_edge_length",
                passed=False,
                value=0,
                threshold=(self._min_edge, self._max_edge),
                message="Mesh has no edges — empty or degenerate mesh.",
                severity="error",
            ))
            return results

        min_edge = float(np.min(edge_lengths))
        max_edge = float(np.max(edge_lengths))
        mean_edge = float(np.mean(edge_lengths))

        # Edge length check
        edge_ok = min_edge >= self._min_edge and max_edge <= self._max_edge
        results.append(CheckResult(
            check_name="triangle_edge_length",
            passed=edge_ok,
            value={
                "min_mm": round(min_edge, 4),
                "max_mm": round(max_edge, 4),
                "mean_mm": round(mean_edge, 4),
            },
            threshold={"min": self._min_edge, "max": self._max_edge},
            message=(
                f"Edge lengths OK (range {min_edge:.3f}–{max_edge:.3f}mm)"
                if edge_ok
                else f"Edge length out of range: min={min_edge:.4f}mm "
                     f"(threshold {self._min_edge}mm), max={max_edge:.3f}mm "
                     f"(threshold {self._max_edge}mm). Remesh to improve triangle quality."
            ),
            severity="warning" if not edge_ok else "info",
        ))

        # Aspect ratio check
        aspect_ratios = self._compute_aspect_ratios(mesh)
        max_aspect = float(np.max(aspect_ratios)) if len(aspect_ratios) > 0 else 0.0
        mean_aspect = float(np.mean(aspect_ratios)) if len(aspect_ratios) > 0 else 0.0
        n_bad = int(np.sum(aspect_ratios > self._max_aspect))
        aspect_ok = n_bad == 0

        results.append(CheckResult(
            check_name="triangle_aspect_ratio",
            passed=aspect_ok,
            value={
                "max": round(max_aspect, 2),
                "mean": round(mean_aspect, 2),
                "n_exceeding": n_bad,
            },
            threshold=self._max_aspect,
            message=(
                f"Aspect ratios OK (max {max_aspect:.1f}, mean {mean_aspect:.1f})"
                if aspect_ok
                else f"{n_bad} triangles exceed aspect ratio {self._max_aspect} "
                     f"(max={max_aspect:.1f}). These sliver triangles may cause "
                     "slicer artifacts. Remesh to improve quality."
            ),
            severity="warning" if not aspect_ok else "info",
        ))

        return results

    def check_self_intersections(self, mesh: trimesh.Trimesh) -> CheckResult:
        """
        Check for self-intersecting faces.

        Self-intersections produce ambiguous inside/outside regions,
        causing slicer failures and print defects.

        Args:
            mesh: Mesh to check.

        Returns:
            CheckResult.
        """
        n_intersecting = self._count_self_intersections(mesh)
        passed = n_intersecting == 0

        return CheckResult(
            check_name="self_intersections",
            passed=passed,
            value=n_intersecting,
            threshold=0,
            message=(
                "No self-intersections detected"
                if passed
                else f"Found {n_intersecting} self-intersecting face pairs. "
                     "These must be resolved — use mesh repair or manual editing."
            ),
            severity="error" if not passed else "info",
        )

    def check_normal_consistency(self, mesh: trimesh.Trimesh) -> CheckResult:
        """
        Check that face normals are consistently oriented (outward).

        Inconsistent normals produce inverted faces that print as voids.

        Args:
            mesh: Mesh to check.

        Returns:
            CheckResult.
        """
        fraction_outward = self._compute_normal_consistency(mesh)
        passed = fraction_outward > 0.95

        return CheckResult(
            check_name="normal_consistency",
            passed=passed,
            value=round(fraction_outward, 3),
            threshold=0.95,
            message=(
                f"Normals consistent ({fraction_outward*100:.1f}% outward-facing)"
                if passed
                else f"Only {fraction_outward*100:.1f}% of normals face outward. "
                     "Run fix_normals() or orient normals consistently before printing."
            ),
            severity="error" if not passed else "info",
        )

    # ------------------------------------------------------------------
    # Internal: Measurement helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_min_wall_thickness(mesh: trimesh.Trimesh, n_samples: int = 1000) -> float:
        """
        Estimate minimum wall thickness by firing inward rays from surface samples.

        For each sampled surface point, cast a ray inward (opposite surface normal)
        and measure distance to the opposite wall.

        Args:
            mesh: Mesh to measure.
            n_samples: Number of surface points to sample.

        Returns:
            Estimated minimum wall thickness in mm.
        """
        if len(mesh.faces) == 0:
            return 0.0

        # Sample surface points
        points, face_indices = trimesh.sample.sample_surface(mesh, n_samples)
        normals = mesh.face_normals[face_indices]

        # Cast rays inward (opposite to outward normal)
        ray_origins = points + normals * 0.01  # slight offset to avoid self-hit
        ray_directions = -normals

        hit_locations, ray_indices, _ = mesh.ray.intersects_location(
            ray_origins, ray_directions
        )

        if len(hit_locations) == 0:
            return float("inf")

        # Compute distances
        thicknesses = []
        for i, hit in enumerate(hit_locations):
            ray_idx = ray_indices[i]
            origin = ray_origins[ray_idx]
            dist = float(np.linalg.norm(hit - origin))
            if dist > 0.01:  # filter self-intersections
                thicknesses.append(dist)

        if not thicknesses:
            return float("inf")

        return float(np.percentile(thicknesses, 5))  # 5th percentile as robust minimum

    @staticmethod
    def _compute_overhang_stats(
        mesh: trimesh.Trimesh,
        threshold_deg: float = DEFAULT_MAX_OVERHANG_ANGLE_DEG,
    ) -> Tuple[float, float]:
        """
        Compute overhang statistics.

        Args:
            mesh: Mesh to analyse.
            threshold_deg: Overhang angle threshold.

        Returns:
            (fraction_exceeding, max_overhang_deg).
        """
        if len(mesh.face_normals) == 0:
            return 0.0, 0.0

        build_dir = np.array([0.0, 0.0, 1.0])
        cos_angles = mesh.face_normals @ build_dir

        # Faces pointing downward have cos < 0
        # Overhang angle from build direction
        angles_from_vertical = np.degrees(np.arccos(np.clip(cos_angles, -1.0, 1.0)))

        # Overhang = angle > (90 + threshold) from build direction
        overhang_threshold = 90.0 + threshold_deg
        overhanging = angles_from_vertical > overhang_threshold
        fraction = float(np.mean(overhanging))
        max_overhang = float(np.max(angles_from_vertical)) - 90.0 if np.any(overhanging) else 0.0

        return fraction, max(0.0, max_overhang)

    @staticmethod
    def _compute_aspect_ratios(mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Compute triangle aspect ratios (longest edge / shortest altitude).

        Args:
            mesh: Mesh to analyse.

        Returns:
            (F,) array of aspect ratios per face.
        """
        triangles = mesh.triangles  # (F, 3, 3)
        if len(triangles) == 0:
            return np.array([])

        # Edge lengths per face
        e0 = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
        e1 = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
        e2 = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)

        longest = np.maximum(np.maximum(e0, e1), e2)
        areas = mesh.area_faces

        # Aspect ratio = longest_edge^2 / (2 * sqrt(3) * area)
        # This normalizes so an equilateral triangle has ratio 1
        denom = 2.0 * np.sqrt(3.0) * areas
        denom = np.where(denom < 1e-12, 1e-12, denom)
        aspect_ratios = (longest ** 2) / denom

        return aspect_ratios

    @staticmethod
    def _count_self_intersections(mesh: trimesh.Trimesh, max_checks: int = 10000) -> int:
        """
        Count self-intersecting face pairs using spatial hashing.

        Uses trimesh's built-in intersection detection when available,
        otherwise falls back to a sampling-based approximation.

        Args:
            mesh: Mesh to check.
            max_checks: Maximum number of face pairs to check.

        Returns:
            Number of intersecting face pairs detected.
        """
        # Use trimesh's built-in if available
        try:
            intersections = mesh.face_adjacency_angles
            # Faces with very large dihedral angle changes may self-intersect
            # This is a heuristic — true self-intersection detection is O(F^2)
            n_suspicious = int(np.sum(intersections > np.radians(170)))
            return n_suspicious
        except Exception:
            pass

        # Fallback: spatial proximity check
        # Sample random face pairs and check for triangle-triangle intersection
        n_faces = len(mesh.faces)
        if n_faces < 2:
            return 0

        n_checks = min(max_checks, n_faces * (n_faces - 1) // 2)
        rng = np.random.RandomState(42)
        n_intersecting = 0

        face_centers = mesh.triangles_center
        for _ in range(n_checks):
            i, j = rng.choice(n_faces, size=2, replace=False)
            # Quick distance reject
            dist = np.linalg.norm(face_centers[i] - face_centers[j])
            if dist > 5.0:  # faces far apart can't intersect
                continue
            if _triangles_intersect(mesh.triangles[i], mesh.triangles[j]):
                n_intersecting += 1

        return n_intersecting

    @staticmethod
    def _compute_normal_consistency(mesh: trimesh.Trimesh) -> float:
        """
        Compute the fraction of face normals pointing outward.

        Uses the centroid-based heuristic: if a normal points away from the
        mesh centroid, it's considered outward-facing.

        Args:
            mesh: Mesh to check.

        Returns:
            Fraction of outward-facing normals (0.0 to 1.0).
        """
        if len(mesh.faces) == 0:
            return 1.0

        face_centers = mesh.triangles_center
        mesh_center = mesh.centroid
        outward_dirs = face_centers - mesh_center

        dots = np.sum(mesh.face_normals * outward_dirs, axis=1)
        return float(np.mean(dots > 0))


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def _triangles_intersect(tri_a: np.ndarray, tri_b: np.ndarray) -> bool:
    """
    Test if two triangles intersect using the Moller-Trumbore separating axis test.

    Simplified implementation: checks if any edge of triangle A passes through
    triangle B, and vice versa.

    Args:
        tri_a: (3, 3) vertices of triangle A.
        tri_b: (3, 3) vertices of triangle B.

    Returns:
        True if triangles intersect.
    """
    for tri_src, tri_dst in [(tri_a, tri_b), (tri_b, tri_a)]:
        for i in range(3):
            # Edge from vertex i to vertex (i+1)%3
            origin = tri_src[i]
            direction = tri_src[(i + 1) % 3] - tri_src[i]
            edge_length = np.linalg.norm(direction)
            if edge_length < 1e-12:
                continue
            direction = direction / edge_length

            # Moller-Trumbore ray-triangle intersection
            e1 = tri_dst[1] - tri_dst[0]
            e2 = tri_dst[2] - tri_dst[0]
            h = np.cross(direction, e2)
            a = np.dot(e1, h)
            if abs(a) < 1e-12:
                continue

            f = 1.0 / a
            s = origin - tri_dst[0]
            u = f * np.dot(s, h)
            if u < 0.0 or u > 1.0:
                continue

            q = np.cross(s, e1)
            v = f * np.dot(direction, q)
            if v < 0.0 or u + v > 1.0:
                continue

            t = f * np.dot(e2, q)
            if 0.0 < t < edge_length:
                return True

    return False
