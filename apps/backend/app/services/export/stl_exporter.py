"""
Generate 3D-printable STL files from ML model predictions.

Handles six export types — corrected mandible, corrected dentition,
intermediate splint, cutting guide, fixation plate template, and full
assembly.  Every export produces a primary STL file, a metadata JSON
sidecar, and a printability report.

Binary and ASCII STL formats are supported; binary is the default for
file-size efficiency.

Dependencies: trimesh, numpy, json.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ExportType(str, Enum):
    """Types of STL export supported by the pipeline."""

    CORRECTED_MANDIBLE = "corrected_mandible"
    CORRECTED_DENTITION = "corrected_dentition"
    INTERMEDIATE_SPLINT = "intermediate_splint"
    CUTTING_GUIDE = "cutting_guide"
    FIXATION_PLATE_TEMPLATE = "fixation_plate_template"
    FULL_ASSEMBLY = "full_assembly"


class STLFormat(str, Enum):
    """STL encoding format."""

    BINARY = "binary"
    ASCII = "ascii"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExportMetadata:
    """Metadata sidecar for an exported STL file."""

    export_type: str
    case_id: str
    plan_id: str
    model_version: str
    export_timestamp: str
    stl_path: str
    stl_format: str
    stl_hash_sha256: str
    vertex_count: int
    face_count: int
    volume_mm3: float
    surface_area_mm2: float
    bounding_box_mm: Dict[str, float]
    is_watertight: bool
    transforms_applied: Dict[str, Any]
    confidence_scores: Dict[str, float]
    units: str = "millimeters"


@dataclass
class PrintabilityInfo:
    """Lightweight printability summary embedded in the export manifest."""

    is_printable: bool
    min_wall_thickness_mm: float
    max_overhang_angle_deg: float
    volume_mm3: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class ExportResult:
    """Result of a single STL export operation."""

    stl_path: Path
    metadata_path: Path
    metadata: ExportMetadata
    printability: PrintabilityInfo
    elapsed_seconds: float


@dataclass
class ExportManifest:
    """Manifest for a batch of exports (e.g. full assembly)."""

    case_id: str
    plan_id: str
    exports: List[ExportResult]
    total_elapsed_seconds: float
    timestamp: str


# ---------------------------------------------------------------------------
# STLExporter
# ---------------------------------------------------------------------------

class STLExporter:
    """
    Generates 3D-printable STL files from model predictions.

    Each export produces:
    - Primary STL file
    - Metadata JSON sidecar (transforms, model version, confidence)
    - Printability report (wall thickness, overhang angles, volume)

    Thread-safe: no mutable instance state beyond configuration.
    """

    def __init__(
        self,
        output_dir: Path,
        model_version: str = "unknown",
        stl_format: STLFormat = STLFormat.BINARY,
        min_wall_thickness_mm: float = 0.8,
        max_overhang_angle_deg: float = 45.0,
    ) -> None:
        """
        Initialise the STL exporter.

        Args:
            output_dir: Directory to write exported files.
            model_version: Model version string for metadata.
            stl_format: Binary or ASCII STL.
            min_wall_thickness_mm: Minimum wall thickness for printability check.
            max_overhang_angle_deg: Maximum unsupported overhang angle.
        """
        self._output_dir = Path(output_dir)
        self._model_version = model_version
        self._stl_format = stl_format
        self._min_wall = min_wall_thickness_mm
        self._max_overhang = max_overhang_angle_deg

    # ------------------------------------------------------------------
    # Public: Single mesh export
    # ------------------------------------------------------------------

    def export_mesh(
        self,
        mesh: trimesh.Trimesh,
        export_type: ExportType,
        case_id: str,
        plan_id: str,
        transforms_applied: Optional[Dict[str, Any]] = None,
        confidence_scores: Optional[Dict[str, float]] = None,
        filename_prefix: Optional[str] = None,
    ) -> ExportResult:
        """
        Export a single mesh as STL with metadata and printability info.

        Args:
            mesh: Trimesh mesh to export.
            export_type: Type of export (determines subdirectory and naming).
            case_id: Surgical case identifier.
            plan_id: Plan identifier.
            transforms_applied: Dict of transforms for metadata sidecar.
            confidence_scores: Per-fragment/tooth confidence scores.
            filename_prefix: Optional prefix for the output filename.

        Returns:
            ExportResult with paths and metadata.
        """
        t0 = time.monotonic()

        # Create output directory
        type_dir = self._output_dir / case_id / export_type.value
        type_dir.mkdir(parents=True, exist_ok=True)

        prefix = filename_prefix or export_type.value
        timestamp_str = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stl_filename = f"{prefix}_{timestamp_str}.stl"
        stl_path = type_dir / stl_filename

        # Write STL
        self._write_stl(mesh, stl_path)

        # Compute file hash
        stl_hash = self._compute_sha256(stl_path)

        # Compute metrics
        volume = float(mesh.volume) if mesh.is_watertight else 0.0
        surface_area = float(mesh.area)
        bounds = mesh.bounds
        bbox = {
            "min_x": float(bounds[0][0]), "min_y": float(bounds[0][1]), "min_z": float(bounds[0][2]),
            "max_x": float(bounds[1][0]), "max_y": float(bounds[1][1]), "max_z": float(bounds[1][2]),
        }

        # Metadata
        metadata = ExportMetadata(
            export_type=export_type.value,
            case_id=case_id,
            plan_id=plan_id,
            model_version=self._model_version,
            export_timestamp=timestamp_str,
            stl_path=str(stl_path),
            stl_format=self._stl_format.value,
            stl_hash_sha256=stl_hash,
            vertex_count=len(mesh.vertices),
            face_count=len(mesh.faces),
            volume_mm3=volume,
            surface_area_mm2=surface_area,
            bounding_box_mm=bbox,
            is_watertight=bool(mesh.is_watertight),
            transforms_applied=transforms_applied or {},
            confidence_scores=confidence_scores or {},
        )

        # Write metadata JSON sidecar
        meta_path = stl_path.with_suffix(".json")
        self._write_metadata(metadata, meta_path)

        # Printability check
        printability = self._quick_printability_check(mesh)

        elapsed = time.monotonic() - t0
        logger.info(
            "Exported %s: %s (%d verts, %d faces, watertight=%s) in %.2fs",
            export_type.value, stl_path.name,
            len(mesh.vertices), len(mesh.faces),
            mesh.is_watertight, elapsed,
        )

        return ExportResult(
            stl_path=stl_path,
            metadata_path=meta_path,
            metadata=metadata,
            printability=printability,
            elapsed_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # Public: Specialised exporters
    # ------------------------------------------------------------------

    def export_corrected_mandible(
        self,
        mandible_mesh: trimesh.Trimesh,
        case_id: str,
        plan_id: str,
        transforms: Dict[str, Any],
        confidence: Dict[str, float],
    ) -> ExportResult:
        """
        Export corrected mandibular bone mesh.

        Args:
            mandible_mesh: Repositioned mandible trimesh.
            case_id: Case identifier.
            plan_id: Plan identifier.
            transforms: Applied transforms per fragment.
            confidence: Per-fragment confidence scores.

        Returns:
            ExportResult.
        """
        return self.export_mesh(
            mesh=mandible_mesh,
            export_type=ExportType.CORRECTED_MANDIBLE,
            case_id=case_id,
            plan_id=plan_id,
            transforms_applied=transforms,
            confidence_scores=confidence,
        )

    def export_corrected_dentition(
        self,
        upper_arch: Optional[trimesh.Trimesh],
        lower_arch: Optional[trimesh.Trimesh],
        case_id: str,
        plan_id: str,
        transforms: Dict[str, Any],
        confidence: Dict[str, float],
    ) -> List[ExportResult]:
        """
        Export repositioned teeth meshes (upper and lower arches).

        Args:
            upper_arch: Upper dental arch mesh (or None).
            lower_arch: Lower dental arch mesh (or None).
            case_id: Case identifier.
            plan_id: Plan identifier.
            transforms: Transforms applied to teeth.
            confidence: Per-tooth confidence.

        Returns:
            List of ExportResult (one per arch present).
        """
        results: List[ExportResult] = []
        if upper_arch is not None:
            results.append(self.export_mesh(
                mesh=upper_arch,
                export_type=ExportType.CORRECTED_DENTITION,
                case_id=case_id,
                plan_id=plan_id,
                transforms_applied=transforms,
                confidence_scores=confidence,
                filename_prefix="upper_dentition",
            ))
        if lower_arch is not None:
            results.append(self.export_mesh(
                mesh=lower_arch,
                export_type=ExportType.CORRECTED_DENTITION,
                case_id=case_id,
                plan_id=plan_id,
                transforms_applied=transforms,
                confidence_scores=confidence,
                filename_prefix="lower_dentition",
            ))
        return results

    def export_intermediate_splint(
        self,
        upper_arch: trimesh.Trimesh,
        lower_arch: trimesh.Trimesh,
        case_id: str,
        plan_id: str,
        vertical_dimension_mm: float = 2.0,
        confidence: Optional[Dict[str, float]] = None,
    ) -> ExportResult:
        """
        Export an intermediate occlusal splint generated from arch intersection.

        The splint is constructed by computing the inter-occlusal region between
        upper and lower arches and generating a solid body of specified vertical
        dimension.

        Args:
            upper_arch: Upper dental arch mesh.
            lower_arch: Lower dental arch mesh.
            case_id: Case identifier.
            plan_id: Plan identifier.
            vertical_dimension_mm: Splint thickness in mm.
            confidence: Optional confidence scores.

        Returns:
            ExportResult for the splint STL.
        """
        splint_mesh = self._generate_splint_geometry(
            upper_arch, lower_arch, vertical_dimension_mm
        )
        return self.export_mesh(
            mesh=splint_mesh,
            export_type=ExportType.INTERMEDIATE_SPLINT,
            case_id=case_id,
            plan_id=plan_id,
            transforms_applied={"vertical_dimension_mm": vertical_dimension_mm},
            confidence_scores=confidence or {},
            filename_prefix="intermediate_splint",
        )

    def export_cutting_guide(
        self,
        bone_mesh: trimesh.Trimesh,
        cutting_planes: List[Dict[str, Any]],
        case_id: str,
        plan_id: str,
        guide_thickness_mm: float = 3.0,
        guide_offset_mm: float = 1.0,
    ) -> ExportResult:
        """
        Export an osteotomy cutting guide.

        The guide is a shell that sits on the bone surface with slots
        indicating osteotomy planes.

        Args:
            bone_mesh: Bone surface mesh.
            cutting_planes: List of plane definitions (point + normal).
            case_id: Case identifier.
            plan_id: Plan identifier.
            guide_thickness_mm: Guide wall thickness.
            guide_offset_mm: Offset from bone surface.

        Returns:
            ExportResult for the cutting guide STL.
        """
        guide_mesh = self._generate_cutting_guide(
            bone_mesh, cutting_planes, guide_thickness_mm, guide_offset_mm
        )
        transforms_info = {
            "n_cutting_planes": len(cutting_planes),
            "guide_thickness_mm": guide_thickness_mm,
            "guide_offset_mm": guide_offset_mm,
        }
        return self.export_mesh(
            mesh=guide_mesh,
            export_type=ExportType.CUTTING_GUIDE,
            case_id=case_id,
            plan_id=plan_id,
            transforms_applied=transforms_info,
            filename_prefix="cutting_guide",
        )

    def export_fixation_plate_template(
        self,
        bone_mesh: trimesh.Trimesh,
        plate_region_points: np.ndarray,
        case_id: str,
        plan_id: str,
        plate_width_mm: float = 6.0,
        plate_thickness_mm: float = 1.0,
    ) -> ExportResult:
        """
        Export a template for plate bending.

        The template follows the bone surface contour in the fixation region
        so the surgeon can pre-bend the reconstruction plate.

        Args:
            bone_mesh: Bone surface mesh.
            plate_region_points: (N, 3) points defining the plate path.
            case_id: Case identifier.
            plan_id: Plan identifier.
            plate_width_mm: Plate strip width.
            plate_thickness_mm: Template thickness.

        Returns:
            ExportResult for the plate template STL.
        """
        template_mesh = self._generate_plate_template(
            bone_mesh, plate_region_points, plate_width_mm, plate_thickness_mm
        )
        return self.export_mesh(
            mesh=template_mesh,
            export_type=ExportType.FIXATION_PLATE_TEMPLATE,
            case_id=case_id,
            plan_id=plan_id,
            transforms_applied={
                "plate_width_mm": plate_width_mm,
                "plate_thickness_mm": plate_thickness_mm,
                "n_path_points": len(plate_region_points),
            },
            filename_prefix="plate_template",
        )

    def export_full_assembly(
        self,
        meshes: Dict[str, trimesh.Trimesh],
        case_id: str,
        plan_id: str,
        transforms: Dict[str, Any],
        confidence: Dict[str, float],
    ) -> ExportManifest:
        """
        Export all fragments in corrected positions as a unified assembly.

        Produces one STL per fragment plus a combined assembly STL.

        Args:
            meshes: Mapping of fragment_id -> trimesh mesh.
            case_id: Case identifier.
            plan_id: Plan identifier.
            transforms: Per-fragment transforms applied.
            confidence: Per-fragment confidence scores.

        Returns:
            ExportManifest covering all exported files.
        """
        t_total = time.monotonic()
        results: List[ExportResult] = []

        # Export individual fragments
        for frag_id, mesh in meshes.items():
            result = self.export_mesh(
                mesh=mesh,
                export_type=ExportType.FULL_ASSEMBLY,
                case_id=case_id,
                plan_id=plan_id,
                transforms_applied={frag_id: transforms.get(frag_id, {})},
                confidence_scores={frag_id: confidence.get(frag_id, 0.0)},
                filename_prefix=f"fragment_{frag_id}",
            )
            results.append(result)

        # Combined assembly
        if meshes:
            combined = trimesh.util.concatenate(list(meshes.values()))
            combined_result = self.export_mesh(
                mesh=combined,
                export_type=ExportType.FULL_ASSEMBLY,
                case_id=case_id,
                plan_id=plan_id,
                transforms_applied=transforms,
                confidence_scores=confidence,
                filename_prefix="assembly_combined",
            )
            results.append(combined_result)

        total_elapsed = time.monotonic() - t_total
        timestamp_str = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        manifest = ExportManifest(
            case_id=case_id,
            plan_id=plan_id,
            exports=results,
            total_elapsed_seconds=total_elapsed,
            timestamp=timestamp_str,
        )

        # Write manifest JSON
        manifest_path = self._output_dir / case_id / "export_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_manifest(manifest, manifest_path)

        logger.info(
            "Full assembly export: %d files for case %s (%.2fs)",
            len(results), case_id, total_elapsed,
        )
        return manifest

    # ------------------------------------------------------------------
    # Public: Verification
    # ------------------------------------------------------------------

    def verify_watertight(self, mesh: trimesh.Trimesh) -> bool:
        """
        Verify that a mesh is watertight (manifold, no holes).

        Args:
            mesh: Mesh to check.

        Returns:
            True if watertight.
        """
        return bool(mesh.is_watertight)

    # ------------------------------------------------------------------
    # Internal: STL I/O
    # ------------------------------------------------------------------

    def _write_stl(self, mesh: trimesh.Trimesh, path: Path) -> None:
        """
        Write mesh to STL file in configured format (binary or ASCII).

        Args:
            mesh: Mesh to write.
            path: Output file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._stl_format == STLFormat.ASCII:
            stl_data = trimesh.exchange.stl.export_stl_ascii(mesh)
            path.write_text(stl_data)
        else:
            stl_data = trimesh.exchange.stl.export_stl(mesh)
            path.write_bytes(stl_data)

        logger.debug("Wrote STL (%s): %s", self._stl_format.value, path)

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        """
        Compute SHA-256 hash of a file.

        Args:
            path: File to hash.

        Returns:
            Hex digest string.
        """
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _write_metadata(metadata: ExportMetadata, path: Path) -> None:
        """
        Write export metadata to a JSON sidecar file.

        Args:
            metadata: Metadata dataclass.
            path: Output JSON path.
        """
        data = {
            "export_type": metadata.export_type,
            "case_id": metadata.case_id,
            "plan_id": metadata.plan_id,
            "model_version": metadata.model_version,
            "export_timestamp": metadata.export_timestamp,
            "stl_path": metadata.stl_path,
            "stl_format": metadata.stl_format,
            "stl_hash_sha256": metadata.stl_hash_sha256,
            "vertex_count": metadata.vertex_count,
            "face_count": metadata.face_count,
            "volume_mm3": metadata.volume_mm3,
            "surface_area_mm2": metadata.surface_area_mm2,
            "bounding_box_mm": metadata.bounding_box_mm,
            "is_watertight": metadata.is_watertight,
            "transforms_applied": metadata.transforms_applied,
            "confidence_scores": metadata.confidence_scores,
            "units": metadata.units,
        }
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.debug("Wrote metadata: %s", path)

    @staticmethod
    def _write_manifest(manifest: ExportManifest, path: Path) -> None:
        """
        Write export manifest JSON.

        Args:
            manifest: Manifest dataclass.
            path: Output JSON path.
        """
        data = {
            "case_id": manifest.case_id,
            "plan_id": manifest.plan_id,
            "timestamp": manifest.timestamp,
            "total_elapsed_seconds": manifest.total_elapsed_seconds,
            "exports": [
                {
                    "stl_path": str(exp.stl_path),
                    "metadata_path": str(exp.metadata_path),
                    "export_type": exp.metadata.export_type,
                    "is_watertight": exp.metadata.is_watertight,
                    "vertex_count": exp.metadata.vertex_count,
                    "face_count": exp.metadata.face_count,
                    "is_printable": exp.printability.is_printable,
                }
                for exp in manifest.exports
            ],
        }
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.debug("Wrote manifest: %s", path)

    # ------------------------------------------------------------------
    # Internal: Printability
    # ------------------------------------------------------------------

    def _quick_printability_check(self, mesh: trimesh.Trimesh) -> PrintabilityInfo:
        """
        Run a quick printability assessment for the export metadata.

        Full validation is done by PrintabilityValidator; this provides
        a lightweight summary for the sidecar.

        Args:
            mesh: Mesh to check.

        Returns:
            PrintabilityInfo summary.
        """
        warnings: List[str] = []
        is_watertight = bool(mesh.is_watertight)
        if not is_watertight:
            warnings.append("Mesh is not watertight; may require repair before printing")

        # Estimate min wall thickness from edge lengths
        edge_lengths = mesh.edges_unique_length
        min_edge = float(np.min(edge_lengths)) if len(edge_lengths) > 0 else 0.0
        estimated_wall = min_edge * 2.0  # conservative heuristic
        if estimated_wall < self._min_wall:
            warnings.append(
                f"Estimated min wall thickness {estimated_wall:.2f}mm "
                f"below minimum {self._min_wall}mm"
            )

        # Overhang angle estimation
        max_overhang = self._estimate_max_overhang(mesh)
        if max_overhang > self._max_overhang:
            warnings.append(
                f"Max overhang angle {max_overhang:.1f}° exceeds "
                f"{self._max_overhang}° (may need support)"
            )

        volume = float(mesh.volume) if is_watertight else 0.0
        is_printable = is_watertight and len(warnings) == 0

        return PrintabilityInfo(
            is_printable=is_printable,
            min_wall_thickness_mm=estimated_wall,
            max_overhang_angle_deg=max_overhang,
            volume_mm3=volume,
            warnings=warnings,
        )

    @staticmethod
    def _estimate_max_overhang(mesh: trimesh.Trimesh) -> float:
        """
        Estimate the maximum overhang angle from face normals.

        Overhang angle is measured from the vertical (build direction Z+).
        A face pointing straight down has 180° overhang.

        Args:
            mesh: Mesh to analyse.

        Returns:
            Maximum overhang angle in degrees.
        """
        if len(mesh.face_normals) == 0:
            return 0.0

        build_dir = np.array([0.0, 0.0, 1.0])
        cos_angles = mesh.face_normals @ build_dir
        # Overhang angle: angle between the face normal and the downward direction
        # A face pointing straight down has normal [0,0,-1], cos = -1, angle = 180°
        # We care about faces where normal points downward (cos < 0)
        downward_mask = cos_angles < 0
        if not np.any(downward_mask):
            return 0.0

        # Overhang angle from vertical
        overhang_cos = -cos_angles[downward_mask]  # make positive
        overhang_angles = np.degrees(np.arccos(np.clip(overhang_cos, 0.0, 1.0)))
        return float(np.max(90.0 - overhang_angles))

    # ------------------------------------------------------------------
    # Internal: Geometry generation
    # ------------------------------------------------------------------

    def _generate_splint_geometry(
        self,
        upper_arch: trimesh.Trimesh,
        lower_arch: trimesh.Trimesh,
        vertical_dimension_mm: float,
    ) -> trimesh.Trimesh:
        """
        Generate an intermediate occlusal splint from upper and lower arch meshes.

        Strategy: Compute the overlap region between the arches in the Z
        (vertical) direction, then create a slab mesh filling the
        inter-occlusal space.

        Args:
            upper_arch: Upper dental arch mesh.
            lower_arch: Lower dental arch mesh.
            vertical_dimension_mm: Target splint thickness.

        Returns:
            Splint trimesh mesh.
        """
        # Find the overlap bounding box in XY
        upper_bounds = upper_arch.bounds
        lower_bounds = lower_arch.bounds

        overlap_min_xy = np.maximum(upper_bounds[0][:2], lower_bounds[0][:2])
        overlap_max_xy = np.minimum(upper_bounds[1][:2], lower_bounds[1][:2])

        if np.any(overlap_max_xy <= overlap_min_xy):
            logger.warning("No XY overlap between arches; generating bounding-box splint")
            all_min = np.minimum(upper_bounds[0], lower_bounds[0])
            all_max = np.maximum(upper_bounds[1], lower_bounds[1])
            overlap_min_xy = all_min[:2]
            overlap_max_xy = all_max[:2]

        # Sample Z heights from upper and lower arch surfaces in the overlap region
        grid_resolution = 0.5  # mm
        x_range = np.arange(overlap_min_xy[0], overlap_max_xy[0], grid_resolution)
        y_range = np.arange(overlap_min_xy[1], overlap_max_xy[1], grid_resolution)

        if len(x_range) == 0 or len(y_range) == 0:
            # Fallback: create a simple box splint
            center = (upper_bounds[0] + upper_bounds[1]) / 2
            box = trimesh.creation.box(
                extents=[30.0, 60.0, vertical_dimension_mm],
                transform=trimesh.transformations.translation_matrix(center),
            )
            return box

        xx, yy = np.meshgrid(x_range, y_range)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])

        # Ray-cast downward from above to find upper arch surface
        ray_origins_upper = np.column_stack([
            grid_points,
            np.full(len(grid_points), upper_bounds[1][2] + 10.0),
        ])
        ray_dirs_down = np.tile([0.0, 0.0, -1.0], (len(grid_points), 1))

        upper_hits, upper_ray_ids, _ = upper_arch.ray.intersects_location(
            ray_origins_upper, ray_dirs_down
        )

        # Ray-cast upward from below to find lower arch surface
        ray_origins_lower = np.column_stack([
            grid_points,
            np.full(len(grid_points), lower_bounds[0][2] - 10.0),
        ])
        ray_dirs_up = np.tile([0.0, 0.0, 1.0], (len(grid_points), 1))

        lower_hits, lower_ray_ids, _ = lower_arch.ray.intersects_location(
            ray_origins_lower, ray_dirs_up
        )

        if len(upper_hits) == 0 or len(lower_hits) == 0:
            # Fallback box
            mid_z = (upper_bounds[0][2] + lower_bounds[1][2]) / 2.0
            center = np.array([
                (overlap_min_xy[0] + overlap_max_xy[0]) / 2.0,
                (overlap_min_xy[1] + overlap_max_xy[1]) / 2.0,
                mid_z,
            ])
            box = trimesh.creation.box(
                extents=[
                    float(overlap_max_xy[0] - overlap_min_xy[0]),
                    float(overlap_max_xy[1] - overlap_min_xy[1]),
                    vertical_dimension_mm,
                ],
                transform=trimesh.transformations.translation_matrix(center),
            )
            return box

        # Build a splint as a slab between the arches
        # Use the lower surface of the upper arch and offset downward
        upper_z_map = {}
        for hit, ray_id in zip(upper_hits, upper_ray_ids):
            key = ray_id
            if key not in upper_z_map or hit[2] < upper_z_map[key]:
                upper_z_map[key] = hit[2]

        lower_z_map = {}
        for hit, ray_id in zip(lower_hits, lower_ray_ids):
            key = ray_id
            if key not in lower_z_map or hit[2] > lower_z_map[key]:
                lower_z_map[key] = hit[2]

        # Create upper and lower surface vertices for shared ray indices
        common_rays = set(upper_z_map.keys()) & set(lower_z_map.keys())
        if not common_rays:
            mid_z = (upper_bounds[0][2] + lower_bounds[1][2]) / 2.0
            center = np.array([
                (overlap_min_xy[0] + overlap_max_xy[0]) / 2.0,
                (overlap_min_xy[1] + overlap_max_xy[1]) / 2.0,
                mid_z,
            ])
            return trimesh.creation.box(
                extents=[
                    float(overlap_max_xy[0] - overlap_min_xy[0]),
                    float(overlap_max_xy[1] - overlap_min_xy[1]),
                    vertical_dimension_mm,
                ],
                transform=trimesh.transformations.translation_matrix(center),
            )

        # Build the splint as a convex hull of upper and lower contact surfaces
        upper_pts = []
        lower_pts = []
        for ray_id in sorted(common_rays):
            xy = grid_points[ray_id]
            upper_z = upper_z_map[ray_id]
            lower_z = lower_z_map[ray_id]
            # The splint sits between upper and lower surfaces
            # Clamp to desired vertical dimension
            mid_z = (upper_z + lower_z) / 2.0
            half_vd = vertical_dimension_mm / 2.0
            upper_pts.append([xy[0], xy[1], mid_z + half_vd])
            lower_pts.append([xy[0], xy[1], mid_z - half_vd])

        all_pts = np.array(upper_pts + lower_pts)
        splint = trimesh.convex.convex_hull(all_pts)

        return splint

    @staticmethod
    def _generate_cutting_guide(
        bone_mesh: trimesh.Trimesh,
        cutting_planes: List[Dict[str, Any]],
        thickness_mm: float,
        offset_mm: float,
    ) -> trimesh.Trimesh:
        """
        Generate a cutting guide shell from bone surface and osteotomy planes.

        Strategy:
        1. Offset the bone surface outward by offset_mm
        2. Create an outer shell at offset_mm + thickness_mm
        3. Create guide body as the region between inner and outer offsets
        4. Cut slots along each osteotomy plane

        Args:
            bone_mesh: Bone surface mesh.
            cutting_planes: Plane definitions with 'point' and 'normal' keys.
            thickness_mm: Guide wall thickness.
            offset_mm: Offset from bone surface.

        Returns:
            Cutting guide trimesh mesh.
        """
        # Compute vertex normals for offset direction
        bone_mesh.fix_normals()
        vertex_normals = bone_mesh.vertex_normals

        # Inner surface (fits on bone)
        inner_verts = np.array(bone_mesh.vertices) + vertex_normals * offset_mm
        # Outer surface
        outer_verts = np.array(bone_mesh.vertices) + vertex_normals * (offset_mm + thickness_mm)

        inner_mesh = trimesh.Trimesh(vertices=inner_verts, faces=bone_mesh.faces, process=False)
        outer_mesh = trimesh.Trimesh(vertices=outer_verts, faces=bone_mesh.faces, process=False)

        # Flip inner normals so the shell is solid when combined
        inner_mesh.invert()

        # Combine into shell
        guide = trimesh.util.concatenate([outer_mesh, inner_mesh])

        # Cut slots for each cutting plane
        for plane_def in cutting_planes:
            point = np.array(plane_def["point"], dtype=np.float64)
            normal = np.array(plane_def["normal"], dtype=np.float64)
            normal = normal / np.linalg.norm(normal)

            # Create a thin box representing the slot
            slot_size = 100.0  # mm, large enough to cut through guide
            slot_thickness = 1.5  # mm slot width

            slot = trimesh.creation.box(extents=[slot_size, slot_size, slot_thickness])

            # Align slot normal with cutting plane normal
            z_axis = np.array([0.0, 0.0, 1.0])
            rotation_axis = np.cross(z_axis, normal)
            rotation_norm = np.linalg.norm(rotation_axis)
            if rotation_norm > 1e-8:
                rotation_axis = rotation_axis / rotation_norm
                angle = np.arccos(np.clip(np.dot(z_axis, normal), -1.0, 1.0))
                from scipy.spatial.transform import Rotation as _R
                rot_mat = _R.from_rotvec(rotation_axis * angle).as_matrix()
                T = np.eye(4)
                T[:3, :3] = rot_mat
                T[:3, 3] = point
                slot.apply_transform(T)
            else:
                slot.apply_translation(point)

            # Boolean difference to create slot
            try:
                guide = guide.difference(slot)
            except Exception:
                logger.warning("Boolean slot cut failed; guide may lack slots")

        return guide

    @staticmethod
    def _generate_plate_template(
        bone_mesh: trimesh.Trimesh,
        path_points: np.ndarray,
        width_mm: float,
        thickness_mm: float,
    ) -> trimesh.Trimesh:
        """
        Generate a fixation plate bending template following a surface path.

        Creates a strip mesh that follows the bone surface contour at the
        specified path points.

        Args:
            bone_mesh: Bone surface mesh.
            path_points: (N, 3) ordered points defining the plate centreline.
            width_mm: Plate strip width.
            thickness_mm: Template thickness.

        Returns:
            Plate template trimesh mesh.
        """
        path_points = np.asarray(path_points, dtype=np.float64)
        n_pts = len(path_points)
        if n_pts < 2:
            raise ValueError("Need at least 2 path points for plate template")

        # Project path points onto bone surface to get closest surface points and normals
        closest_pts, distances, face_ids = trimesh.proximity.closest_point(
            bone_mesh, path_points
        )
        face_normals = bone_mesh.face_normals[face_ids]

        # Build strip vertices: for each path point, create 4 corners
        # (left/right x inner/outer)
        all_vertices = []
        for i in range(n_pts):
            pt = closest_pts[i]
            normal = face_normals[i]

            # Tangent direction along the path
            if i == 0:
                tangent = closest_pts[1] - closest_pts[0]
            elif i == n_pts - 1:
                tangent = closest_pts[-1] - closest_pts[-2]
            else:
                tangent = closest_pts[i + 1] - closest_pts[i - 1]
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm < 1e-12:
                tangent = np.array([1.0, 0.0, 0.0])
            else:
                tangent = tangent / tangent_norm

            # Lateral direction (perpendicular to tangent and surface normal)
            lateral = np.cross(tangent, normal)
            lat_norm = np.linalg.norm(lateral)
            if lat_norm < 1e-12:
                lateral = np.array([0.0, 1.0, 0.0])
            else:
                lateral = lateral / lat_norm

            half_w = width_mm / 2.0

            # Inner surface (bone side)
            all_vertices.append(pt + lateral * half_w)
            all_vertices.append(pt - lateral * half_w)
            # Outer surface
            all_vertices.append(pt + normal * thickness_mm + lateral * half_w)
            all_vertices.append(pt + normal * thickness_mm - lateral * half_w)

        vertices = np.array(all_vertices)

        # Build faces connecting consecutive cross-sections
        faces = []
        for i in range(n_pts - 1):
            base = i * 4
            nxt = (i + 1) * 4

            # Outer face (vertices 2,3 of each section)
            faces.append([base + 2, nxt + 2, nxt + 3])
            faces.append([base + 2, nxt + 3, base + 3])

            # Inner face (vertices 0,1)
            faces.append([base + 0, nxt + 1, nxt + 0])
            faces.append([base + 0, base + 1, nxt + 1])

            # Left side (vertices 0,2)
            faces.append([base + 0, nxt + 0, nxt + 2])
            faces.append([base + 0, nxt + 2, base + 2])

            # Right side (vertices 1,3)
            faces.append([base + 1, base + 3, nxt + 3])
            faces.append([base + 1, nxt + 3, nxt + 1])

        # Cap start
        faces.append([0, 2, 3])
        faces.append([0, 3, 1])

        # Cap end
        end = (n_pts - 1) * 4
        faces.append([end + 0, end + 1, end + 3])
        faces.append([end + 0, end + 3, end + 2])

        template = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=True)
        template.fix_normals()
        return template
