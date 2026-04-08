"""
Mesh extraction and processing service.
Converts volumetric segmentation masks to surface meshes using marching cubes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.exceptions import (
    EmptyMaskError,
    MeshExtractionError,
    MeshQualityError,
    MeshSimplificationError,
)
from app.core.logging import TimedOperation, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Material colour map for anatomical structures
# Used when exporting multi-structure GLB files with proper material assignment
# ---------------------------------------------------------------------------

ANATOMY_MATERIAL_COLORS: Dict[str, Tuple[int, int, int, int]] = {
    # Bone structures — white/ivory (RGBA 0–255)
    "skull":              (245, 240, 225, 220),
    "mandible":           (240, 230, 210, 230),
    "maxilla":            (240, 232, 215, 230),
    "condyle_left":       (255, 210, 80,  255),
    "condyle_right":      (255, 210, 80,  255),
    "zygomatic_arch_left":  (210, 190, 170, 230),
    "zygomatic_arch_right": (210, 190, 170, 230),
    # Dental — bright white
    "upper_dental_arch":  (255, 255, 252, 255),
    "lower_dental_arch":  (255, 255, 252, 255),
    # Soft tissue — skin tone (semi-transparent)
    "skin":               (210, 160, 130, 140),
    "masseter_left":      (230, 110, 110, 150),
    "masseter_right":     (230, 110, 110, 150),
    "temporalis_left":    (240, 130, 120, 140),
    "temporalis_right":   (240, 130, 120, 140),
    # Air spaces — translucent blue
    "maxillary_sinus_left":  (100, 180, 255, 80),
    "maxillary_sinus_right": (100, 180, 255, 80),
    "frontal_sinus":         (100, 180, 255, 80),
    # Nerve canal
    "inferior_alveolar_nerve_left":  (255, 230, 50, 220),
    "inferior_alveolar_nerve_right": (255, 230, 50, 220),
}


@dataclass
class MeshMetrics:
    """Quality metrics for a triangular surface mesh."""
    vertex_count: int
    face_count: int
    is_watertight: bool
    euler_number: int
    volume_mm3: float
    surface_area_mm2: float
    centroid_mm: List[float]
    bounding_box: Dict[str, float]
    aspect_ratio_min: float  # Quality metric: min face aspect ratio
    aspect_ratio_mean: float


@dataclass
class MeshQualityResult:
    """
    Detailed mesh quality assessment for surgical planning validation.

    A mesh used for surgical planning must satisfy strict topological
    requirements to ensure accurate volume measurements, collision detection,
    and boolean operations during osteotomy simulation.

    Watertightness:
        A watertight mesh has no holes or open boundaries. Required for:
        - Accurate volume computation
        - Boolean cutting operations
        - 3D printing without repair

    Manifold geometry:
        A manifold mesh has exactly 2 faces per edge (no T-junctions,
        no non-manifold vertices). Required for:
        - Topologically correct surface operations
        - Reliable normal computation

    Self-intersections:
        Faces that intersect each other invalidate volume measurements
        and cause failures in surgical simulation.
    """
    is_watertight: bool
    is_manifold: bool
    has_self_intersections: bool
    open_boundary_edges: int        # Edges with only 1 adjacent face
    non_manifold_edges: int         # Edges with > 2 adjacent faces
    non_manifold_vertices: int
    degenerate_face_count: int      # Zero-area triangles
    duplicate_face_count: int
    unreferenced_vertex_count: int
    euler_characteristic: int       # Should be 2 for genus-0 (sphere-like) meshes
    genus: int                      # Topological genus (0 = sphere-like)
    quality_score: float            # 0.0–1.0 composite quality score
    issues: List[str]               # Human-readable list of problems found
    suitable_for_planning: bool     # Final verdict


@dataclass
class MultiResolutionMeshSet:
    """
    A mesh exported at three resolution levels for different use cases.

    High resolution: full-detail mesh for surgical planning and measurement.
                     Preserves all anatomical features.
    Medium resolution: reduced for web viewer interaction (< 100K faces).
                       Imperceptible quality loss at rendering distance.
    Low resolution: thumbnail and preview rendering (< 5K faces).
                    Used for case list thumbnails and mobile viewers.
    """
    structure_name: str
    high_res_path: Optional[Path]    # Full resolution — for planning
    medium_res_path: Optional[Path]  # Web viewer
    low_res_path: Optional[Path]     # Thumbnails
    high_res_faces: int = 0
    medium_res_faces: int = 0
    low_res_faces: int = 0
    quality_result: Optional[MeshQualityResult] = None


class MeshService:
    """
    Surface mesh extraction and processing service.

    Converts 3D segmentation masks (integer label volumes) to triangular
    surface meshes suitable for surgical visualization, 3D printing,
    and geometric analysis.

    Pipeline:
    1. extract_mesh_from_mask: Marching cubes → raw mesh
    2. smooth_mesh: Laplacian smoothing to reduce staircase artifacts
    3. simplify_mesh: Quadric decimation to reduce polygon count
    4. export_glb: Export to binary GLTF for web viewer
    """

    # Quality thresholds
    MIN_FACE_COUNT = 100
    MAX_FACE_COUNT = 5_000_000
    WATERTIGHT_REQUIRED = True

    def __init__(self) -> None:
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Verify required libraries are available."""
        missing = []
        for lib in ["trimesh", "skimage"]:
            try:
                __import__(lib.replace("-", "_"))
            except ImportError:
                missing.append(lib)
        if missing:
            logger.warning("mesh_service_missing_libs", missing=missing)

    def extract_mesh_from_mask(
        self,
        mask: np.ndarray,
        spacing: Tuple[float, float, float],
        label: int = 1,
        step_size: int = 1,
        padding: int = 2,
    ) -> "trimesh.Trimesh":
        """
        Extract a surface mesh from a binary/label mask using Marching Cubes.

        Args:
            mask: 3D integer label volume (Z, Y, X)
            spacing: Voxel spacing in mm (x, y, z)
            label: Label value to extract surface for
            step_size: Marching cubes step size (1=full resolution, 2=faster)
            padding: Voxels of padding to add around the structure

        Returns:
            trimesh.Trimesh surface mesh in millimeter coordinates

        Raises:
            EmptyMaskError: If no voxels match the label
            MeshExtractionError: If marching cubes fails
        """
        try:
            import trimesh
            from skimage.measure import marching_cubes
        except ImportError as e:
            raise MeshExtractionError(f"Required library not installed: {e}")

        with TimedOperation(logger, "mesh_extraction", label=label):
            # Extract binary mask for this label
            binary_mask = (mask == label).astype(np.uint8)
            voxel_count = int(np.sum(binary_mask))

            if voxel_count == 0:
                raise EmptyMaskError(
                    f"Label {label} has no voxels in mask",
                    context={"label": label, "mask_shape": list(mask.shape)},
                )

            # Add padding to avoid mesh touching volume boundary
            if padding > 0:
                binary_mask = np.pad(binary_mask, padding, mode="constant", constant_values=0)

            # Run marching cubes (skimage implementation, level=0.5 for binary)
            try:
                vertices, faces, normals, _ = marching_cubes(
                    binary_mask,
                    level=0.5,
                    step_size=step_size,
                    allow_degenerate=False,
                )
            except (ValueError, RuntimeError) as exc:
                raise MeshExtractionError(
                    f"Marching cubes failed for label {label}: {exc}",
                    context={"label": label, "voxel_count": voxel_count},
                    cause=exc,
                )

            # Remove padding offset and scale to millimeters
            if padding > 0:
                vertices -= padding

            # Scale from voxel to mm (vertices are in (col, row, slice) = (x, y, z) order)
            # skimage returns vertices in (row, col, slice) ~ (y, x, z) order
            # We need to convert to (x, y, z) in mm
            vertices_mm = np.zeros_like(vertices)
            vertices_mm[:, 0] = vertices[:, 1] * spacing[0]  # x = col * x_spacing
            vertices_mm[:, 1] = vertices[:, 0] * spacing[1]  # y = row * y_spacing
            vertices_mm[:, 2] = vertices[:, 2] * spacing[2]  # z = slice * z_spacing

            mesh = trimesh.Trimesh(
                vertices=vertices_mm,
                faces=faces,
                vertex_normals=normals,
                process=True,  # Basic cleanup on construction
            )

            logger.info(
                "mesh_extracted",
                label=label,
                vertices=len(mesh.vertices),
                faces=len(mesh.faces),
                is_watertight=mesh.is_watertight,
                volume_cc=round(abs(mesh.volume) / 1000, 3) if mesh.is_watertight else None,
            )

            return mesh

    def smooth_mesh(
        self,
        mesh: "trimesh.Trimesh",
        iterations: int = 5,
        lamb: float = 0.5,
        mu: float = -0.53,
        method: str = "taubin",
    ) -> "trimesh.Trimesh":
        """
        Smooth a mesh to reduce marching cubes staircase artifacts.

        Taubin smoothing (λ/μ smoothing) preserves volume better than
        simple Laplacian smoothing.

        Args:
            mesh: Input mesh
            iterations: Number of smoothing passes
            lamb: Laplacian shrink factor (0 < λ < 1)
            mu: Anti-shrink factor (μ < -λ)
            method: "taubin" (volume-preserving) or "laplacian"

        Returns:
            Smoothed mesh (new object, original unchanged)
        """
        import trimesh
        from trimesh.smoothing import filter_taubin, filter_laplacian

        smoothed = mesh.copy()

        try:
            if method == "taubin":
                filter_taubin(smoothed, lamb=lamb, nu=mu, iterations=iterations)
            else:
                filter_laplacian(smoothed, iterations=iterations)

            logger.debug(
                "mesh_smoothed",
                method=method,
                iterations=iterations,
                vertices_before=len(mesh.vertices),
                vertices_after=len(smoothed.vertices),
            )
        except Exception as exc:
            logger.warning("mesh_smoothing_failed", error=str(exc))
            return mesh

        return smoothed

    def simplify_mesh(
        self,
        mesh: "trimesh.Trimesh",
        target_faces: Optional[int] = None,
        target_ratio: float = 0.2,
        preserve_topology: bool = True,
    ) -> "trimesh.Trimesh":
        """
        Reduce mesh polygon count using quadric error decimation.

        Args:
            mesh: Input mesh
            target_faces: Absolute target face count (overrides target_ratio)
            target_ratio: Fraction of faces to keep (0.2 = 80% reduction)
            preserve_topology: Prevent topology changes (holes, non-manifold)

        Returns:
            Simplified mesh

        Raises:
            MeshSimplificationError: If simplification fails critically
        """
        import trimesh

        if target_faces is None:
            target_faces = max(self.MIN_FACE_COUNT, int(len(mesh.faces) * target_ratio))

        if len(mesh.faces) <= target_faces:
            logger.debug("mesh_simplification_skipped", reason="already_at_target")
            return mesh

        try:
            simplified = mesh.simplify_quadric_decimation(target_faces)
            logger.info(
                "mesh_simplified",
                faces_before=len(mesh.faces),
                faces_after=len(simplified.faces),
                ratio=round(len(simplified.faces) / len(mesh.faces), 3),
            )
            return simplified
        except Exception as exc:
            logger.warning(
                "mesh_simplification_failed",
                error=str(exc),
                note="Returning original mesh",
            )
            return mesh

    def export_glb(
        self,
        mesh: "trimesh.Trimesh",
        output_path: Path,
        embed_texture: bool = False,
    ) -> Path:
        """
        Export mesh to binary GLTF (.glb) for the web-based 3D viewer.

        GLB is the preferred format for Three.js / Babylon.js rendering.

        Args:
            mesh: Mesh to export
            output_path: Destination file path (must end with .glb)
            embed_texture: Whether to embed material texture

        Returns:
            Path to the exported file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            mesh.export(str(output_path), file_type="glb")
            file_size = output_path.stat().st_size
            logger.info(
                "mesh_exported_glb",
                path=str(output_path),
                size_mb=round(file_size / 1e6, 2),
                vertices=len(mesh.vertices),
                faces=len(mesh.faces),
            )
            return output_path
        except Exception as exc:
            raise MeshExtractionError(
                f"GLB export failed: {exc}",
                context={"path": str(output_path)},
                cause=exc,
            )

    def export_stl(
        self,
        mesh: "trimesh.Trimesh",
        output_path: Path,
        ascii_format: bool = False,
    ) -> Path:
        """
        Export mesh to STL for CAD/CAM and 3D printing workflows.

        Args:
            mesh: Mesh to export
            output_path: Destination file path (must end with .stl)
            ascii_format: Use ASCII STL (False = binary, much smaller)

        Returns:
            Path to exported file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            mesh.export(str(output_path), file_type="stl_ascii" if ascii_format else "stl")
            logger.info("mesh_exported_stl", path=str(output_path))
            return output_path
        except Exception as exc:
            raise MeshExtractionError(f"STL export failed: {exc}", cause=exc)

    def export_ply(self, mesh: "trimesh.Trimesh", output_path: Path) -> Path:
        """Export mesh to PLY format (for Open3D/MeshLab processing)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(output_path), file_type="ply")
        return output_path

    def compute_mesh_metrics(self, mesh: "trimesh.Trimesh") -> MeshMetrics:
        """
        Compute quality and geometric metrics for a mesh.

        Args:
            mesh: Input mesh

        Returns:
            MeshMetrics dataclass
        """
        import trimesh

        centroid = mesh.centroid.tolist() if mesh.vertices.shape[0] > 0 else [0.0, 0.0, 0.0]

        bounds = mesh.bounds  # (2, 3) array: [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        bounding_box = {
            "min_x": float(bounds[0][0]), "max_x": float(bounds[1][0]),
            "min_y": float(bounds[0][1]), "max_y": float(bounds[1][1]),
            "min_z": float(bounds[0][2]), "max_z": float(bounds[1][2]),
        }

        # Volume and surface area (only meaningful for watertight meshes)
        volume_mm3 = abs(float(mesh.volume)) if mesh.is_watertight else 0.0
        surface_area_mm2 = float(mesh.area)

        # Face quality (aspect ratio distribution)
        if len(mesh.faces) > 0:
            try:
                # Compute edge lengths for each triangle
                v0 = mesh.vertices[mesh.faces[:, 0]]
                v1 = mesh.vertices[mesh.faces[:, 1]]
                v2 = mesh.vertices[mesh.faces[:, 2]]
                e1 = np.linalg.norm(v1 - v0, axis=1)
                e2 = np.linalg.norm(v2 - v1, axis=1)
                e3 = np.linalg.norm(v0 - v2, axis=1)
                # Aspect ratio = max_edge / min_edge
                max_edge = np.maximum(np.maximum(e1, e2), e3)
                min_edge = np.minimum(np.minimum(e1, e2), e3) + 1e-10
                aspect_ratios = max_edge / min_edge
                aspect_min = float(np.percentile(aspect_ratios, 10))
                aspect_mean = float(np.mean(aspect_ratios))
            except Exception:
                aspect_min = 1.0
                aspect_mean = 1.0
        else:
            aspect_min = 0.0
            aspect_mean = 0.0

        return MeshMetrics(
            vertex_count=len(mesh.vertices),
            face_count=len(mesh.faces),
            is_watertight=mesh.is_watertight,
            euler_number=int(mesh.euler_number),
            volume_mm3=round(volume_mm3, 2),
            surface_area_mm2=round(surface_area_mm2, 2),
            centroid_mm=[round(c, 3) for c in centroid],
            bounding_box=bounding_box,
            aspect_ratio_min=round(aspect_min, 3),
            aspect_ratio_mean=round(aspect_mean, 3),
        )

    def extract_and_process_all_structures(
        self,
        masks: np.ndarray,
        labels: Dict[str, int],
        spacing: Tuple[float, float, float],
        output_dir: Path,
        smooth_iterations: int = 5,
        target_face_ratio: float = 0.25,
    ) -> Dict[str, Dict[str, Path]]:
        """
        Full pipeline: extract, smooth, simplify, and export all structures.

        Args:
            masks: Segmentation mask volume
            labels: {structure_name: label_value}
            spacing: Voxel spacing in mm
            output_dir: Directory to write mesh files
            smooth_iterations: Laplacian smoothing iterations
            target_face_ratio: Target face ratio for simplification

        Returns:
            {structure_name: {"glb": path, "stl": path, "ply": path}}
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        mesh_paths: Dict[str, Dict[str, Path]] = {}

        for structure_name, label_val in labels.items():
            logger.info("processing_structure", structure=structure_name, label=label_val)
            try:
                # Extract
                raw_mesh = self.extract_mesh_from_mask(masks, spacing, label=label_val)

                # Smooth
                smooth_mesh = self.smooth_mesh(raw_mesh, iterations=smooth_iterations)

                # Simplify
                simple_mesh = self.simplify_mesh(smooth_mesh, target_ratio=target_face_ratio)

                # Export in all formats
                glb_path = self.export_glb(simple_mesh, output_dir / f"{structure_name}.glb")
                stl_path = self.export_stl(simple_mesh, output_dir / f"{structure_name}.stl")
                ply_path = self.export_ply(raw_mesh, output_dir / f"{structure_name}.ply")

                mesh_paths[structure_name] = {
                    "glb": glb_path,
                    "stl": stl_path,
                    "ply": ply_path,
                }

            except EmptyMaskError:
                logger.warning("structure_empty", structure=structure_name)
            except Exception as exc:
                logger.error(
                    "structure_processing_failed",
                    structure=structure_name,
                    error=str(exc),
                )

        return mesh_paths

    # ==================================================================
    # Mesh Quality Checks
    # ==================================================================

    def check_mesh_quality(
        self,
        mesh: "trimesh.Trimesh",
        check_self_intersections: bool = True,
    ) -> MeshQualityResult:
        """
        Comprehensive mesh quality check for surgical planning validation.

        Checks watertightness, manifold geometry, self-intersections,
        and degenerate elements. All issues are collected and a composite
        quality score is computed.

        Args:
            mesh: Input mesh to evaluate
            check_self_intersections: Whether to run self-intersection test.
                                      Can be slow for large meshes (> 500K faces).
                                      Disable for quick checks.

        Returns:
            MeshQualityResult with all metrics and suitability verdict
        """
        import trimesh

        issues: List[str] = []

        # --- Watertightness -------------------------------------------
        is_watertight = bool(mesh.is_watertight)
        if not is_watertight:
            issues.append("Mesh is NOT watertight (has open boundary edges)")

        # --- Manifold check -------------------------------------------
        is_manifold = bool(mesh.is_volume)  # trimesh: is_volume = watertight AND winding

        # Count open boundary edges: edges shared by only 1 face
        unique_edges, edge_counts = np.unique(
            np.sort(mesh.edges_unique, axis=1),
            axis=0,
            return_counts=True,
        ) if len(mesh.edges_unique) > 0 else (np.array([]), np.array([]))

        # trimesh provides edge adjacency directly
        try:
            edges_raw = mesh.edges.reshape(-1, 2)
            edge_sorted = np.sort(edges_raw, axis=1)
            _, edge_face_counts = np.unique(edge_sorted, axis=0, return_counts=True)
            open_boundary_edges = int(np.sum(edge_face_counts == 1))
            non_manifold_edges = int(np.sum(edge_face_counts > 2))
        except Exception:
            open_boundary_edges = 0 if is_watertight else -1
            non_manifold_edges = 0

        if open_boundary_edges > 0:
            issues.append(
                f"{open_boundary_edges} open boundary edge(s) detected — mesh has holes"
            )
        if non_manifold_edges > 0:
            issues.append(
                f"{non_manifold_edges} non-manifold edge(s) — mesh has T-junctions"
            )

        # --- Non-manifold vertices ------------------------------------
        try:
            # A vertex is non-manifold if it is shared by disjoint triangle fans
            non_manifold_verts = len(trimesh.graph.nondegenerate_faces(mesh))
            # trimesh doesn't expose this directly; use vertex degree heuristic
            # Vertices connected to > 30 faces are suspect for complex topology
            vertex_face_counts = np.bincount(
                mesh.faces.ravel(), minlength=len(mesh.vertices)
            )
            non_manifold_vertices = int(np.sum(vertex_face_counts > 40))
        except Exception:
            non_manifold_vertices = 0

        if non_manifold_vertices > 0:
            issues.append(f"{non_manifold_vertices} potentially non-manifold vertex/vertices")

        # --- Degenerate faces (zero-area triangles) -------------------
        try:
            face_areas = mesh.area_faces
            degenerate_face_count = int(np.sum(face_areas < 1e-10))
        except Exception:
            degenerate_face_count = 0

        if degenerate_face_count > 0:
            issues.append(
                f"{degenerate_face_count} degenerate (zero-area) face(s) — may cause numerical errors"
            )

        # --- Duplicate faces ------------------------------------------
        try:
            face_sorted = np.sort(mesh.faces, axis=1)
            _, face_counts = np.unique(face_sorted, axis=0, return_counts=True)
            duplicate_face_count = int(np.sum(face_counts > 1))
        except Exception:
            duplicate_face_count = 0

        if duplicate_face_count > 0:
            issues.append(f"{duplicate_face_count} duplicate face(s) detected")

        # --- Unreferenced vertices ------------------------------------
        try:
            referenced = np.unique(mesh.faces.ravel())
            unreferenced_vertex_count = len(mesh.vertices) - len(referenced)
        except Exception:
            unreferenced_vertex_count = 0

        if unreferenced_vertex_count > 0:
            issues.append(f"{unreferenced_vertex_count} unreferenced vertex/vertices")

        # --- Euler characteristic and genus ----------------------------
        euler = int(mesh.euler_number)
        # For a closed orientable surface: genus = (2 - euler) / 2
        genus = max(0, (2 - euler) // 2) if is_watertight else -1
        if genus > 0:
            issues.append(
                f"Mesh has genus {genus} (not simply connected — has handles/tunnels)"
            )

        # --- Self-intersections (optional, slow) ----------------------
        has_self_intersections = False
        if check_self_intersections and len(mesh.faces) < 200_000:
            try:
                has_self_intersections = bool(mesh.is_self_intersecting)
                if has_self_intersections:
                    issues.append(
                        "Self-intersecting faces detected — volume computation unreliable"
                    )
            except Exception as exc:
                logger.debug(f"Self-intersection check failed: {exc}")
        elif len(mesh.faces) >= 200_000:
            logger.debug(
                f"Skipping self-intersection check for large mesh ({len(mesh.faces)} faces)"
            )

        # --- Quality score -------------------------------------------
        # Penalise each issue category
        score = 1.0
        if not is_watertight:          score -= 0.30
        if non_manifold_edges > 0:     score -= 0.20
        if has_self_intersections:     score -= 0.25
        if degenerate_face_count > 5:  score -= 0.10
        if genus > 0:                  score -= 0.10 * min(genus, 3)
        score = max(0.0, score)

        suitable = (
            is_watertight and
            non_manifold_edges == 0 and
            not has_self_intersections and
            degenerate_face_count < len(mesh.faces) * 0.001  # < 0.1% degenerate
        )

        result = MeshQualityResult(
            is_watertight=is_watertight,
            is_manifold=is_manifold,
            has_self_intersections=has_self_intersections,
            open_boundary_edges=open_boundary_edges,
            non_manifold_edges=non_manifold_edges,
            non_manifold_vertices=non_manifold_vertices,
            degenerate_face_count=degenerate_face_count,
            duplicate_face_count=duplicate_face_count,
            unreferenced_vertex_count=unreferenced_vertex_count,
            euler_characteristic=euler,
            genus=genus,
            quality_score=round(score, 3),
            issues=issues,
            suitable_for_planning=suitable,
        )

        logger.info(
            "mesh_quality_check",
            watertight=is_watertight,
            manifold=is_manifold,
            self_intersects=has_self_intersections,
            degenerate_faces=degenerate_face_count,
            score=score,
            suitable=suitable,
        )

        return result

    # ==================================================================
    # Mesh Repair Pipeline
    # ==================================================================

    def repair_mesh(
        self,
        mesh: "trimesh.Trimesh",
        fill_holes: bool = True,
        remove_degenerate: bool = True,
        fix_normals: bool = True,
        max_hole_edges: int = 500,
    ) -> Tuple["trimesh.Trimesh", List[str]]:
        """
        Automated mesh repair pipeline for surgical planning readiness.

        Repair operations applied in sequence:
          1. Remove duplicate faces
          2. Remove unreferenced vertices
          3. Remove degenerate (zero-area) faces
          4. Fix face winding order (consistent normals)
          5. Fill small holes (up to max_hole_edges boundary edges)

        Note: Repair may not succeed for severely damaged meshes. Always
        run check_mesh_quality() after repair to verify the result.

        Args:
            mesh: Input mesh (modified in-place in a copy)
            fill_holes: Attempt to fill open boundary holes
            remove_degenerate: Remove zero-area and degenerate faces
            fix_normals: Reorient face normals consistently outward
            max_hole_edges: Maximum hole size (in boundary edges) to fill.
                           Larger holes require manual landmark annotation.

        Returns:
            Tuple of (repaired_mesh, list_of_actions_taken)
        """
        import trimesh

        repaired = mesh.copy()
        actions: List[str] = []

        original_face_count = len(repaired.faces)
        original_vertex_count = len(repaired.vertices)

        # --- Step 1: Remove duplicate faces ----------------------------
        try:
            face_sorted = np.sort(repaired.faces, axis=1)
            _, unique_idx = np.unique(face_sorted, axis=0, return_index=True)
            if len(unique_idx) < len(repaired.faces):
                removed = len(repaired.faces) - len(unique_idx)
                repaired.update_faces(unique_idx)
                actions.append(f"Removed {removed} duplicate face(s)")
        except Exception as exc:
            logger.warning(f"Duplicate face removal failed: {exc}")

        # --- Step 2: Remove degenerate faces ---------------------------
        if remove_degenerate:
            try:
                face_areas = repaired.area_faces
                valid_faces = face_areas > 1e-12
                if not np.all(valid_faces):
                    removed = int(np.sum(~valid_faces))
                    repaired.update_faces(np.where(valid_faces)[0])
                    actions.append(f"Removed {removed} degenerate (zero-area) face(s)")
            except Exception as exc:
                logger.warning(f"Degenerate face removal failed: {exc}")

        # --- Step 3: Remove unreferenced vertices ----------------------
        try:
            repaired.remove_unreferenced_vertices()
            removed_verts = original_vertex_count - len(repaired.vertices)
            if removed_verts > 0:
                actions.append(f"Removed {removed_verts} unreferenced vertex/vertices")
        except Exception as exc:
            logger.warning(f"Unreferenced vertex removal failed: {exc}")

        # --- Step 4: Fix face normals (consistent outward winding) -----
        if fix_normals:
            try:
                trimesh.repair.fix_winding(repaired)
                trimesh.repair.fix_normals(repaired)
                actions.append("Fixed face winding order and normals")
            except Exception as exc:
                logger.warning(f"Normal fixing failed: {exc}")

        # --- Step 5: Fill holes ----------------------------------------
        if fill_holes and not repaired.is_watertight:
            try:
                trimesh.repair.fill_holes(repaired)
                if repaired.is_watertight:
                    actions.append("Filled holes — mesh is now watertight")
                else:
                    # Check remaining boundary edges
                    edges_raw = repaired.edges.reshape(-1, 2)
                    edge_sorted = np.sort(edges_raw, axis=1)
                    _, ecounts = np.unique(edge_sorted, axis=0, return_counts=True)
                    remaining_boundary = int(np.sum(ecounts == 1))
                    actions.append(
                        f"Partial hole fill — {remaining_boundary} boundary edge(s) remain"
                    )
            except Exception as exc:
                logger.warning(f"Hole filling failed: {exc}")
                actions.append(f"Hole filling failed: {exc}")

        # --- Step 6: Remove duplicate vertices (merge close vertices) --
        try:
            before = len(repaired.vertices)
            repaired.merge_vertices(merge_tex=False, merge_norm=False)
            merged = before - len(repaired.vertices)
            if merged > 0:
                actions.append(f"Merged {merged} duplicate vertex/vertices")
        except Exception as exc:
            logger.debug(f"Vertex merging failed: {exc}")

        logger.info(
            "mesh_repair_complete",
            actions=len(actions),
            faces_before=original_face_count,
            faces_after=len(repaired.faces),
            is_watertight_after=repaired.is_watertight,
        )

        return repaired, actions

    # ==================================================================
    # Multi-Resolution Export
    # ==================================================================

    def export_multi_resolution(
        self,
        mesh: "trimesh.Trimesh",
        output_dir: Path,
        structure_name: str,
        high_res_faces: Optional[int] = None,
        medium_res_faces: int = 50_000,
        low_res_faces: int = 3_000,
        material_color: Optional[Tuple[int, int, int, int]] = None,
    ) -> MultiResolutionMeshSet:
        """
        Export a mesh at three resolution levels for different use cases.

        Resolution targets:
          High:   Full resolution (or capped at high_res_faces if provided).
                  Used for precise surgical planning and measurements.
          Medium: Reduced to ~50K faces for smooth web viewer interaction.
                  Imperceptible quality loss at normal rendering distance.
          Low:    Reduced to ~3K faces for thumbnails and mobile previews.

        Each level is exported as .glb with proper material assignment.

        Args:
            mesh: Input mesh (full resolution)
            output_dir: Directory to write output files
            structure_name: Used for filenames and material lookup
            high_res_faces: Cap for high-res (None = keep all)
            medium_res_faces: Target face count for medium res
            low_res_faces: Target face count for low res
            material_color: RGBA tuple; if None, looked up from ANATOMY_MATERIAL_COLORS

        Returns:
            MultiResolutionMeshSet with paths and face counts
        """
        import trimesh

        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine material colour
        if material_color is None:
            color = ANATOMY_MATERIAL_COLORS.get(
                structure_name.lower(),
                (200, 200, 200, 220),  # Default: light grey
            )
        else:
            color = material_color

        def _apply_color_and_export(src_mesh: "trimesh.Trimesh", path: Path) -> int:
            """Apply material colour and export to GLB."""
            colored = src_mesh.copy()
            colored.visual = trimesh.visual.ColorVisuals(
                mesh=colored,
                face_colors=np.tile(list(color), (len(colored.faces), 1)),
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            data = colored.export(file_type="glb")
            path.write_bytes(data)
            logger.info(
                "mesh_exported",
                path=str(path),
                faces=len(colored.faces),
                size_kb=round(len(data) / 1024, 1),
            )
            return len(colored.faces)

        result = MultiResolutionMeshSet(structure_name=structure_name)

        # --- High resolution -------------------------------------------
        if high_res_faces is not None and len(mesh.faces) > high_res_faces:
            high_mesh = mesh.simplify_quadric_decimation(high_res_faces)
        else:
            high_mesh = mesh

        high_path = output_dir / f"{structure_name}_high.glb"
        try:
            result.high_res_faces = _apply_color_and_export(high_mesh, high_path)
            result.high_res_path = high_path
        except Exception as exc:
            logger.error(f"High-res export failed for {structure_name}: {exc}")

        # --- Medium resolution ----------------------------------------
        if len(mesh.faces) > medium_res_faces:
            try:
                medium_mesh = mesh.simplify_quadric_decimation(medium_res_faces)
            except Exception as exc:
                logger.warning(f"Medium simplification failed: {exc}; using high-res")
                medium_mesh = high_mesh
        else:
            medium_mesh = high_mesh

        medium_path = output_dir / f"{structure_name}_medium.glb"
        try:
            result.medium_res_faces = _apply_color_and_export(medium_mesh, medium_path)
            result.medium_res_path = medium_path
        except Exception as exc:
            logger.error(f"Medium-res export failed for {structure_name}: {exc}")

        # --- Low resolution (thumbnail) --------------------------------
        actual_low_faces = min(low_res_faces, max(100, len(mesh.faces) // 20))
        if len(mesh.faces) > actual_low_faces:
            try:
                low_mesh = mesh.simplify_quadric_decimation(actual_low_faces)
            except Exception as exc:
                logger.warning(f"Low simplification failed: {exc}; using medium-res")
                low_mesh = medium_mesh
        else:
            low_mesh = medium_mesh

        low_path = output_dir / f"{structure_name}_low.glb"
        try:
            result.low_res_faces = _apply_color_and_export(low_mesh, low_path)
            result.low_res_path = low_path
        except Exception as exc:
            logger.error(f"Low-res export failed for {structure_name}: {exc}")

        return result

    def export_multi_resolution_scene(
        self,
        meshes: Dict[str, "trimesh.Trimesh"],
        output_dir: Path,
        resolution: str = "medium",
    ) -> Path:
        """
        Export multiple structures as a single GLB scene file.

        Useful for the web viewer to load all structures in one network request.

        Args:
            meshes: {structure_name: trimesh.Trimesh}
            output_dir: Output directory
            resolution: "high", "medium", or "low" — target resolution per structure

        Returns:
            Path to the combined .glb scene file
        """
        import trimesh

        face_targets = {"high": 200_000, "medium": 50_000, "low": 3_000}
        target_faces = face_targets.get(resolution, 50_000)

        scene = trimesh.Scene()

        for structure_name, mesh in meshes.items():
            color = ANATOMY_MATERIAL_COLORS.get(
                structure_name.lower(), (200, 200, 200, 220)
            )

            # Decimate if needed
            per_mesh_target = max(100, target_faces // max(1, len(meshes)))
            if len(mesh.faces) > per_mesh_target:
                try:
                    display_mesh = mesh.simplify_quadric_decimation(per_mesh_target)
                except Exception:
                    display_mesh = mesh
            else:
                display_mesh = mesh

            # Apply colour
            display_mesh = display_mesh.copy()
            display_mesh.visual = trimesh.visual.ColorVisuals(
                mesh=display_mesh,
                face_colors=np.tile(list(color), (len(display_mesh.faces), 1)),
            )

            scene.add_geometry(display_mesh, node_name=structure_name)

        output_path = output_dir / f"scene_{resolution}.glb"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        scene_data = scene.export(file_type="glb")
        output_path.write_bytes(scene_data)

        logger.info(
            "scene_exported",
            path=str(output_path),
            structures=len(meshes),
            resolution=resolution,
            size_kb=round(len(scene_data) / 1024, 1),
        )

        return output_path
