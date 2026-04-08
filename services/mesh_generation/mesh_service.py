"""
Mesh Generation Service
========================

Extracts 3D surface meshes from segmentation masks and prepares them
for web visualisation and surgical planning interaction.

Pipeline
--------
  1. Pre-smoothing: Gaussian filter on the binary mask volume reduces
     staircase artefacts before marching cubes, producing smoother meshes
     at no cost to topology.

  2. Marching Cubes: skimage's marching_cubes on the smoothed float volume.
     The iso-level 0.5 boundary closely approximates the original binary mask.

  3. Mesh decimation: Quadric error metric simplification (trimesh) preserves
     surface curvature while reducing polygon count for real-time rendering.

  4. Post-smoothing: Laplacian smoothing removes residual high-frequency noise
     after decimation without significantly altering overall shape.

  5. Material assignment: Each structure class receives a clinically appropriate
     colour. Bone = white/ivory, soft tissue = skin tone, air = translucent blue.

  6. GLB export: GLTF binary format for direct use in Three.js / Babylon.js
     web viewer. Includes embedded materials and face normals.

  7. Statistics: Vertex count, face count, surface area (mm²), volume (mm³),
     and bounding box in physical coordinates.

Author: Facial Align Engineering
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Material colour assignments for craniofacial anatomy
# Format: RGBA (0–255). Alpha < 255 = semi-transparent.
# ---------------------------------------------------------------------------

STRUCTURE_COLORS: Dict[str, List[int]] = {
    # Bone — white/ivory (dense cortical bone appearance)
    "skull":                        [245, 240, 225, 220],
    "mandible":                     [240, 230, 210, 230],
    "maxilla":                      [240, 232, 215, 230],
    "condyle_left":                 [255, 210,  80, 255],
    "condyle_right":                [255, 210,  80, 255],
    "zygomatic_arch_left":          [210, 190, 170, 230],
    "zygomatic_arch_right":         [210, 190, 170, 230],
    # Dental — bright white
    "upper_dental_arch":            [255, 255, 252, 255],
    "lower_dental_arch":            [255, 255, 252, 255],
    # Soft tissue — realistic skin tone, semi-transparent
    "skin":                         [210, 160, 130, 140],
    "masseter_left":                [230, 110, 110, 150],
    "masseter_right":               [230, 110, 110, 150],
    "temporalis_left":              [240, 130, 120, 140],
    "temporalis_right":             [240, 130, 120, 140],
    "pterygoid_medial_left":        [220, 115, 115, 140],
    "pterygoid_medial_right":       [220, 115, 115, 140],
    # Sinuses / air spaces — pale blue, highly transparent
    "maxillary_sinus_left":         [100, 180, 255,  80],
    "maxillary_sinus_right":        [100, 180, 255,  80],
    "frontal_sinus":                [100, 180, 255,  80],
    # Nerve canals — yellow, opaque
    "inferior_alveolar_nerve_left": [255, 230,  50, 220],
    "inferior_alveolar_nerve_right":[255, 230,  50, 220],
    # Fracture fragments — distinct fragment colours
    "fragment_1":                   [255, 100, 100, 230],
    "fragment_2":                   [100, 100, 255, 230],
    "fragment_3":                   [100, 255, 100, 230],
}


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
    """
    Comprehensive geometric and quality statistics for an extracted mesh.

    All spatial measurements are in millimetres (mm) or mm²/mm³.

    Clinical use:
        - vertex_count and face_count indicate mesh density
        - volume_mm3 enables comparison against normative anatomy databases
        - surface_area_mm2 is used for surgical implant sizing
        - bounding_box defines the spatial extent for planning coordinate systems
        - is_watertight is required before volume computation is trustworthy
    """

    num_vertices: int
    num_faces: int
    volume_mm3: float
    surface_area_mm2: float
    centroid_mm: tuple[float, float, float]
    bounding_box: dict          # {min: [x,y,z], max: [x,y,z], size: [dx,dy,dz]}
    is_watertight: bool
    euler_number: int

    # Extended statistics (new)
    aspect_ratio_mean: float = 0.0      # Mean face aspect ratio (1.0 = equilateral)
    aspect_ratio_p90: float = 0.0       # 90th percentile aspect ratio
    edge_length_mean_mm: float = 0.0    # Mean edge length
    edge_length_std_mm: float = 0.0     # Std dev of edge lengths
    degenerate_faces: int = 0           # Zero-area face count
    smoothing_applied: bool = False
    decimation_ratio: float = 1.0       # Ratio of final to original face count
    extraction_time_ms: float = 0.0


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
        Extract a 3D surface mesh from a binary segmentation mask.

        Pipeline:
          1. Extract binary float mask for the target label
          2. Gaussian pre-smoothing to reduce staircase artefacts
          3. Marching cubes at iso-level 0.5 (the original mask boundary)
          4. Quadric decimation to target face count
          5. Laplacian post-smoothing

        The Gaussian sigma controls the trade-off between surface smoothness
        and preservation of fine anatomical detail:
          - sigma 0.5: minimal smoothing, preserves sharp features (teeth, condyle)
          - sigma 1.0: moderate smoothing, good for cortical bone
          - sigma 2.0: heavy smoothing, suitable for soft tissue surfaces

        Args:
            mask: 3D numpy array (Z, Y, X) with integer labels
            spacing: Voxel spacing in mm (z, y, x)
            label: Label value to extract
            origin: Volume origin in mm for physical coordinate offset

        Returns:
            trimesh.Trimesh object in physical coordinates (mm), or None if extraction fails
        """
        import trimesh
        from scipy.ndimage import gaussian_filter
        from skimage.measure import marching_cubes

        t_start = time.time()

        # Extract binary mask for this label
        binary = (mask == label).astype(np.float32)
        voxel_count = int(binary.sum())

        if voxel_count < 10:
            logger.warning(f"Label {label}: too few voxels ({voxel_count})")
            return None

        # Pad the volume to prevent mesh from touching the volume boundary
        # (marching cubes needs a closed surface)
        pad = 2
        binary = np.pad(binary, pad, mode="constant", constant_values=0)

        # Gaussian pre-smoothing: reduces staircase artefacts in marching cubes
        # The sigma is in voxel units; typical craniofacial CT at 0.5mm → sigma=1.0
        # yields ~0.5mm spatial smoothing, preserving clinically relevant features
        if self.config.smoothing_sigma > 0:
            smoothed = gaussian_filter(binary, sigma=self.config.smoothing_sigma)
        else:
            smoothed = binary

        # Marching cubes at iso-level 0.5 (boundary between background and structure)
        try:
            verts, faces, normals, _ = marching_cubes(
                smoothed,
                level=self.config.marching_cubes_level,
                spacing=spacing,  # Returns vertices in mm directly
            )
        except Exception as e:
            logger.error(f"Marching cubes failed for label {label}: {e}")
            return None

        if len(faces) < self.config.min_face_count:
            logger.warning(f"Label {label}: mesh too small ({len(faces)} faces)")
            return None

        original_face_count = len(faces)

        # Adjust for padding offset: remove the pad*spacing offset
        pad_offset = np.array([pad * spacing[2], pad * spacing[1], pad * spacing[0]])
        verts = verts - pad_offset

        # Apply physical origin offset
        verts += np.array(origin)

        # Build trimesh
        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            vertex_normals=normals,
            process=True,
        )

        # Quadric decimation to target face count (quality-preserving)
        if len(mesh.faces) > self.config.target_face_count:
            try:
                mesh = mesh.simplify_quadric_decimation(self.config.target_face_count)
                decimation_ratio = len(mesh.faces) / max(1, original_face_count)
                logger.debug(
                    f"Label {label}: decimated {original_face_count} → {len(mesh.faces)} faces "
                    f"(ratio={decimation_ratio:.3f})"
                )
            except Exception as exc:
                logger.warning(f"Decimation failed for label {label}: {exc}")

        # Laplacian post-smoothing (removes decimation artefacts)
        if self.config.laplacian_iterations > 0:
            try:
                trimesh.smoothing.filter_laplacian(
                    mesh, iterations=self.config.laplacian_iterations
                )
            except Exception as exc:
                logger.debug(f"Laplacian smoothing failed: {exc}")

        elapsed_ms = (time.time() - t_start) * 1000
        logger.info(
            f"Label {label}: {voxel_count} voxels → {len(mesh.faces)} faces "
            f"in {elapsed_ms:.0f}ms, watertight={mesh.is_watertight}"
        )

        return mesh

    def compute_mesh_metrics(self, mesh, smoothing_applied: bool = False,
                              decimation_ratio: float = 1.0,
                              extraction_time_ms: float = 0.0) -> MeshMetrics:
        """
        Compute comprehensive geometric and quality statistics for a mesh.

        Args:
            mesh: trimesh.Trimesh to analyse
            smoothing_applied: Whether Gaussian/Laplacian smoothing was applied
            decimation_ratio: Ratio of final to original face count (1.0 = no decimation)
            extraction_time_ms: Total time taken to extract this mesh

        Returns:
            MeshMetrics with all statistics populated
        """
        bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        bb_size = (bounds[1] - bounds[0]).tolist()

        volume_mm3 = 0.0
        if mesh.is_watertight:
            try:
                volume_mm3 = float(abs(mesh.volume))
            except Exception:
                volume_mm3 = 0.0

        # --- Face quality statistics ---
        aspect_ratio_mean = 0.0
        aspect_ratio_p90 = 0.0
        edge_length_mean_mm = 0.0
        edge_length_std_mm = 0.0
        degenerate_faces = 0

        if len(mesh.faces) > 0:
            try:
                v0 = mesh.vertices[mesh.faces[:, 0]]
                v1 = mesh.vertices[mesh.faces[:, 1]]
                v2 = mesh.vertices[mesh.faces[:, 2]]

                e1 = np.linalg.norm(v1 - v0, axis=1)
                e2 = np.linalg.norm(v2 - v1, axis=1)
                e3 = np.linalg.norm(v0 - v2, axis=1)

                # Aspect ratio: max_edge / min_edge
                max_edge = np.maximum(np.maximum(e1, e2), e3)
                min_edge = np.minimum(np.minimum(e1, e2), e3)
                valid = min_edge > 1e-12
                if valid.any():
                    ar = max_edge[valid] / min_edge[valid]
                    aspect_ratio_mean = float(np.mean(ar))
                    aspect_ratio_p90 = float(np.percentile(ar, 90))

                # Edge lengths (all edges)
                all_edges = np.concatenate([e1, e2, e3])
                edge_length_mean_mm = float(np.mean(all_edges))
                edge_length_std_mm = float(np.std(all_edges))

                # Degenerate faces
                face_areas = 0.5 * np.linalg.norm(
                    np.cross(v1 - v0, v2 - v0), axis=1
                )
                degenerate_faces = int(np.sum(face_areas < 1e-12))

            except Exception as exc:
                logger.debug(f"Face quality stats failed: {exc}")

        return MeshMetrics(
            num_vertices=len(mesh.vertices),
            num_faces=len(mesh.faces),
            volume_mm3=round(volume_mm3, 2),
            surface_area_mm2=round(float(mesh.area), 2),
            centroid_mm=tuple(round(float(c), 3) for c in mesh.centroid),
            bounding_box={
                "min": [round(float(b), 3) for b in bounds[0]],
                "max": [round(float(b), 3) for b in bounds[1]],
                "size": [round(float(s), 3) for s in bb_size],
            },
            is_watertight=mesh.is_watertight,
            euler_number=int(mesh.euler_number),
            aspect_ratio_mean=round(aspect_ratio_mean, 3),
            aspect_ratio_p90=round(aspect_ratio_p90, 3),
            edge_length_mean_mm=round(edge_length_mean_mm, 3),
            edge_length_std_mm=round(edge_length_std_mm, 3),
            degenerate_faces=degenerate_faces,
            smoothing_applied=smoothing_applied,
            decimation_ratio=round(decimation_ratio, 4),
            extraction_time_ms=round(extraction_time_ms, 1),
        )

    def export_mesh(
        self,
        mesh,
        output_path: str,
        color: Optional[List[int]] = None,
        structure_name: Optional[str] = None,
    ) -> str:
        """
        Export mesh to file with optional material colour assignment.

        Colour priority:
          1. Explicit `color` argument (RGBA list)
          2. Automatic lookup in STRUCTURE_COLORS by structure_name
          3. Default grey [200, 200, 200, 220]

        Args:
            mesh: trimesh.Trimesh
            output_path: Output file path
            color: RGBA colour [r, g, b, a] (0-255); overrides structure lookup
            structure_name: Structure name for automatic colour lookup

        Returns:
            Path to exported file
        """
        import trimesh

        # Resolve colour
        if color is None and structure_name is not None:
            color = STRUCTURE_COLORS.get(structure_name.lower(), [200, 200, 200, 220])
        elif color is None:
            color = [200, 200, 200, 220]

        # Apply colour to a copy (don't mutate the input)
        export_mesh = mesh.copy()
        export_mesh.visual = trimesh.visual.ColorVisuals(
            mesh=export_mesh,
            face_colors=np.tile(color, (len(export_mesh.faces), 1)),
        )

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        file_format = path.suffix.lstrip(".")
        if file_format == "glb":
            data = export_mesh.export(file_type="glb")
            path.write_bytes(data)
        else:
            export_mesh.export(str(path), file_type=file_format)

        logger.info(
            f"Exported mesh: {path.name} "
            f"({len(export_mesh.faces)} faces, {path.stat().st_size} bytes)"
        )
        return str(path)

    def export_glb_with_materials(
        self,
        mesh,
        output_path: str,
        structure_name: str,
        metallic: float = 0.0,
        roughness: float = 0.8,
    ) -> str:
        """
        Export a mesh to GLB with physically-based rendering (PBR) material.

        PBR materials make bone appear realistic in the web viewer:
          - Bone: low metallic (0.0), moderate roughness (0.7-0.9) → matte ivory
          - Teeth: low metallic (0.0), low roughness (0.2-0.4) → slightly glossy
          - Soft tissue: zero metallic, high roughness (0.9) → diffuse skin

        Args:
            mesh: trimesh.Trimesh
            output_path: Output .glb file path
            structure_name: Used for base colour lookup
            metallic: PBR metallic factor (0.0–1.0)
            roughness: PBR roughness factor (0.0–1.0)

        Returns:
            Path to exported GLB file
        """
        import trimesh
        from trimesh.visual.material import PBRMaterial

        color_rgba = STRUCTURE_COLORS.get(structure_name.lower(), [200, 200, 200, 220])
        r, g, b, a = color_rgba

        try:
            material = PBRMaterial(
                baseColorFactor=[r / 255.0, g / 255.0, b / 255.0, a / 255.0],
                metallicFactor=metallic,
                roughnessFactor=roughness,
                name=structure_name,
            )
            pbr_mesh = mesh.copy()
            pbr_mesh.visual = trimesh.visual.TextureVisuals(material=material)
        except Exception as exc:
            logger.debug(f"PBR material failed ({exc}); falling back to vertex colours")
            return self.export_mesh(mesh, output_path, color=color_rgba)

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = pbr_mesh.export(file_type="glb")
        path.write_bytes(data)

        logger.info(
            f"GLB with PBR material exported: {path.name} "
            f"(metallic={metallic}, roughness={roughness})"
        )
        return str(path)

    def extract_all_structures(
        self,
        combined_mask: np.ndarray,
        spacing: tuple[float, float, float],
        label_map: dict[str, int],
        output_dir: str,
        color_map: Optional[dict[str, list[int]]] = None,
        use_pbr_materials: bool = True,
    ) -> dict[str, dict]:
        """
        Extract, smooth, decimate, and export meshes for all structures.

        For each structure:
          1. Extract mesh with Gaussian pre-smoothing and marching cubes
          2. Apply anatomically appropriate material colour (bone=ivory, tissue=skin, etc.)
          3. Export in the configured format (GLB by default)
          4. Compute comprehensive geometric statistics

        Args:
            combined_mask: Multi-label 3D segmentation mask (Z, Y, X)
            spacing: Voxel spacing in mm (z, y, x)
            label_map: {structure_name: label_id}
            output_dir: Output directory for mesh files
            color_map: Optional explicit {structure_name: [r, g, b, a]}.
                       Overrides automatic colour lookup.
            use_pbr_materials: Export with PBR materials (GLB only).
                               Falls back to vertex colours if PBR fails.

        Returns:
            dict of {structure_name: {mesh_path, metrics, color, is_watertight}}
        """
        t_total = time.time()
        results = {}
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Build effective colour map: explicit overrides > STRUCTURE_COLORS > default
        effective_colors = dict(STRUCTURE_COLORS)  # Start from global defaults
        if color_map:
            effective_colors.update(color_map)     # Apply any explicit overrides

        for structure_name, label_id in label_map.items():
            logger.info(f"Extracting: {structure_name} (label={label_id})")
            t_struct = time.time()

            mesh = self.extract_mesh_from_mask(
                combined_mask, spacing, label=label_id
            )

            if mesh is None:
                logger.warning(f"Skipping {structure_name} — extraction failed")
                continue

            elapsed_ms = (time.time() - t_struct) * 1000

            # Determine output path and colour
            ext = self.config.export_format
            mesh_path = str(output_path / f"{structure_name}.{ext}")
            color = effective_colors.get(structure_name.lower(), [200, 200, 200, 220])

            # Export with materials
            if ext == "glb" and use_pbr_materials:
                self.export_glb_with_materials(mesh, mesh_path, structure_name=structure_name)
            else:
                self.export_mesh(mesh, mesh_path, color=color, structure_name=structure_name)

            # Compute full mesh statistics
            original_face_count_approx = len(mesh.faces)
            decimation_ratio = (
                len(mesh.faces) / max(1, original_face_count_approx)
            )
            metrics = self.compute_mesh_metrics(
                mesh,
                smoothing_applied=(self.config.smoothing_sigma > 0),
                decimation_ratio=decimation_ratio,
                extraction_time_ms=elapsed_ms,
            )

            results[structure_name] = {
                "mesh_path": mesh_path,
                "metrics": {
                    "vertices": metrics.num_vertices,
                    "faces": metrics.num_faces,
                    "volume_mm3": metrics.volume_mm3,
                    "surface_area_mm2": metrics.surface_area_mm2,
                    "centroid_mm": list(metrics.centroid_mm),
                    "bounding_box": metrics.bounding_box,
                    "is_watertight": metrics.is_watertight,
                    "euler_number": metrics.euler_number,
                    "aspect_ratio_mean": metrics.aspect_ratio_mean,
                    "edge_length_mean_mm": metrics.edge_length_mean_mm,
                    "degenerate_faces": metrics.degenerate_faces,
                    "smoothing_applied": metrics.smoothing_applied,
                    "extraction_time_ms": metrics.extraction_time_ms,
                },
                "color": color,
                "is_watertight": metrics.is_watertight,
            }

        total_ms = (time.time() - t_total) * 1000
        logger.info(
            f"Extracted {len(results)}/{len(label_map)} structures in {total_ms:.0f}ms"
        )
        return results

    @staticmethod
    def get_structure_color(structure_name: str) -> List[int]:
        """
        Look up the RGBA material colour for a structure by name.

        Args:
            structure_name: Structure name (case-insensitive)

        Returns:
            RGBA list [r, g, b, a] with values 0–255
        """
        return list(STRUCTURE_COLORS.get(
            structure_name.lower(), [200, 200, 200, 220]
        ))

    @staticmethod
    def _default_cmf_colors() -> dict[str, list[int]]:
        """
        Default visualisation colours for CMF structures.
        Retained for backwards compatibility — prefer STRUCTURE_COLORS.
        """
        return {k: list(v) for k, v in STRUCTURE_COLORS.items()}
