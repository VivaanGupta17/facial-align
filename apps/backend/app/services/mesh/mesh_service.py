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
