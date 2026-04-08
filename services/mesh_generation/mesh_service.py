"""
Mesh Generation Service

Extracts 3D surface meshes from segmentation masks and prepares them
for web visualization and surgical planning interaction.

Pipeline: Segmentation Mask → Marching Cubes → Smoothing → Simplification → GLB Export
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MeshExtractionConfig:
    """Configuration for mesh extraction."""

    smoothing_sigma: float = 0.5  # Gaussian smoothing before marching cubes
    marching_cubes_level: float = 0.5
    laplacian_iterations: int = 15
    target_face_count: int = 50000  # Faces per structure
    min_face_count: int = 100  # Discard meshes below this
    export_format: str = "glb"  # glb, stl, ply, obj


@dataclass
class MeshMetrics:
    """Quality metrics for an extracted mesh."""

    num_vertices: int
    num_faces: int
    volume_mm3: float
    surface_area_mm2: float
    centroid_mm: tuple[float, float, float]
    bounding_box: dict  # {min: [x,y,z], max: [x,y,z]}
    is_watertight: bool
    euler_number: int


class MeshService:
    """
    Service for extracting and processing 3D meshes from segmentation masks.

    Used after segmentation to create web-ready 3D assets for the viewer.
    """

    def __init__(self, config: Optional[MeshExtractionConfig] = None):
        self.config = config or MeshExtractionConfig()

    def extract_mesh_from_mask(
        self,
        mask: np.ndarray,
        spacing: tuple[float, float, float],
        label: int = 1,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Optional[object]:
        """
        Extract a 3D mesh from a binary segmentation mask.

        Args:
            mask: 3D numpy array (Z, Y, X) with integer labels
            spacing: Voxel spacing in mm
            label: Label value to extract
            origin: Volume origin in physical coordinates

        Returns:
            trimesh.Trimesh object or None if extraction fails
        """
        import trimesh
        from scipy.ndimage import gaussian_filter
        from skimage.measure import marching_cubes

        # Extract binary mask for this label
        binary = (mask == label).astype(np.float32)

        if binary.sum() < 10:
            logger.warning(f"Label {label}: too few voxels ({binary.sum():.0f})")
            return None

        # Gaussian smoothing before marching cubes (reduces staircase artifacts)
        smoothed = gaussian_filter(binary, sigma=self.config.smoothing_sigma)

        # Marching cubes
        try:
            verts, faces, normals, _ = marching_cubes(
                smoothed,
                level=self.config.marching_cubes_level,
                spacing=spacing,
            )
        except Exception as e:
            logger.error(f"Marching cubes failed for label {label}: {e}")
            return None

        if len(faces) < self.config.min_face_count:
            logger.warning(f"Label {label}: mesh too small ({len(faces)} faces)")
            return None

        # Offset to physical coordinates
        verts += np.array(origin)

        # Create trimesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

        # Simplify if needed
        if len(mesh.faces) > self.config.target_face_count:
            mesh = mesh.simplify_quadric_decimation(self.config.target_face_count)
            logger.debug(f"Simplified to {len(mesh.faces)} faces")

        # Laplacian smoothing
        trimesh.smoothing.filter_laplacian(
            mesh, iterations=self.config.laplacian_iterations
        )

        return mesh

    def compute_mesh_metrics(self, mesh) -> MeshMetrics:
        """Compute quality metrics for a mesh."""
        bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]

        return MeshMetrics(
            num_vertices=len(mesh.vertices),
            num_faces=len(mesh.faces),
            volume_mm3=float(abs(mesh.volume)),
            surface_area_mm2=float(mesh.area),
            centroid_mm=tuple(float(c) for c in mesh.centroid),
            bounding_box={
                "min": [float(b) for b in bounds[0]],
                "max": [float(b) for b in bounds[1]],
            },
            is_watertight=mesh.is_watertight,
            euler_number=mesh.euler_number,
        )

    def export_mesh(
        self,
        mesh,
        output_path: str,
        color: Optional[list[int]] = None,
    ) -> str:
        """
        Export mesh to file.

        Args:
            mesh: trimesh.Trimesh
            output_path: Output file path
            color: RGBA color [r, g, b, a] (0-255)

        Returns:
            Path to exported file
        """
        import trimesh

        if color is not None:
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh,
                face_colors=np.tile(color, (len(mesh.faces), 1)),
            )

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        file_format = path.suffix.lstrip(".")
        if file_format == "glb":
            data = mesh.export(file_type="glb")
            path.write_bytes(data)
        else:
            mesh.export(str(path), file_type=file_format)

        logger.info(f"Exported mesh: {path} ({path.stat().st_size} bytes)")
        return str(path)

    def extract_all_structures(
        self,
        combined_mask: np.ndarray,
        spacing: tuple[float, float, float],
        label_map: dict[str, int],
        output_dir: str,
        color_map: Optional[dict[str, list[int]]] = None,
    ) -> dict[str, dict]:
        """
        Extract meshes for all structures in a multi-label mask.

        Args:
            combined_mask: Multi-label 3D mask
            spacing: Voxel spacing
            label_map: {structure_name: label_id}
            output_dir: Output directory for mesh files
            color_map: Optional {structure_name: [r, g, b, a]}

        Returns:
            dict of {structure_name: {path, metrics, ...}}
        """
        results = {}
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        color_map = color_map or self._default_cmf_colors()

        for structure_name, label_id in label_map.items():
            logger.info(f"Extracting: {structure_name} (label={label_id})")

            mesh = self.extract_mesh_from_mask(combined_mask, spacing, label=label_id)

            if mesh is None:
                logger.warning(f"Skipping {structure_name} — extraction failed")
                continue

            # Export
            color = color_map.get(structure_name, [200, 200, 200, 255])
            glb_path = str(output_path / f"{structure_name}.{self.config.export_format}")
            self.export_mesh(mesh, glb_path, color=color)

            # Metrics
            metrics = self.compute_mesh_metrics(mesh)

            results[structure_name] = {
                "mesh_path": glb_path,
                "metrics": {
                    "vertices": metrics.num_vertices,
                    "faces": metrics.num_faces,
                    "volume_mm3": metrics.volume_mm3,
                    "surface_area_mm2": metrics.surface_area_mm2,
                    "centroid_mm": metrics.centroid_mm,
                    "is_watertight": metrics.is_watertight,
                },
                "color": color,
            }

        logger.info(f"Extracted {len(results)}/{len(label_map)} structures")
        return results

    @staticmethod
    def _default_cmf_colors() -> dict[str, list[int]]:
        """Default visualization colors for CMF structures."""
        return {
            "mandible": [255, 230, 100, 255],
            "maxilla": [100, 150, 255, 255],
            "skull": [240, 230, 220, 180],
            "upper_dental_arch": [255, 255, 255, 255],
            "lower_dental_arch": [255, 255, 255, 255],
            "zygomatic_arch_left": [200, 180, 160, 255],
            "zygomatic_arch_right": [200, 180, 160, 255],
            "maxillary_sinus_left": [100, 200, 255, 100],
            "maxillary_sinus_right": [100, 200, 255, 100],
            "frontal_sinus": [100, 200, 255, 100],
            "masseter_left": [255, 120, 120, 150],
            "masseter_right": [255, 120, 120, 150],
            "temporalis_left": [255, 140, 140, 150],
            "temporalis_right": [255, 140, 140, 150],
            "condyle_left": [255, 200, 50, 255],
            "condyle_right": [255, 200, 50, 255],
        }
