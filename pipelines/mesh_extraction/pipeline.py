"""
Mesh extraction pipeline: segmentation mask → surface meshes with quality validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from app.core.config import get_settings
from app.core.logging import TimedOperation, get_logger
from app.services.mesh.mesh_service import MeshService

settings = get_settings()
logger = get_logger(__name__)


class MeshExtractionPipeline:
    """
    Pipeline for converting segmentation masks to surface meshes.

    Can be run standalone (re-generate meshes with new parameters)
    or as part of the segmentation pipeline.
    """

    # Quality thresholds
    MIN_VERTEX_COUNT = 100
    MAX_VERTEX_COUNT = 2_000_000
    MIN_WATERTIGHT_STRUCTURES = ["mandible", "maxilla"]

    def __init__(
        self,
        segmentation_id: str,
        structures: Optional[List[str]] = None,
        smooth_iterations: int = 5,
        target_face_ratio: float = 0.25,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        self._seg_id = segmentation_id
        self._structures = structures
        self._smooth_iterations = smooth_iterations
        self._target_face_ratio = target_face_ratio
        self._progress = progress_callback or (lambda pct, step: None)
        self._mesh_service = MeshService()

    async def run(self) -> Dict[str, Any]:
        """Execute the mesh extraction pipeline."""
        from app.db.database import get_db_context
        from app.models.segmentation import SegmentationResult
        from sqlalchemy import select, update

        logger.info("mesh_extraction_pipeline_started", seg_id=self._seg_id)

        with TimedOperation(logger, "mesh_extraction_pipeline", seg_id=self._seg_id):
            # Load segmentation record
            self._progress(10, "Loading segmentation data")
            async with get_db_context() as db:
                seg_row = (
                    await db.execute(
                        select(SegmentationResult).where(SegmentationResult.id == self._seg_id)
                    )
                ).scalar_one_or_none()

            if not seg_row or not seg_row.mask_storage_path:
                raise ValueError(f"Segmentation {self._seg_id} has no mask data")

            if not Path(seg_row.mask_storage_path).exists():
                raise FileNotFoundError(f"Mask file not found: {seg_row.mask_storage_path}")

            # Load mask
            self._progress(20, "Loading segmentation mask")
            masks, spacing = self._load_mask(seg_row.mask_storage_path)

            labels: Dict[str, int] = seg_row.structure_labels or {}
            if self._structures:
                labels = {k: v for k, v in labels.items() if k in self._structures}

            # Extract meshes
            self._progress(30, f"Extracting meshes for {len(labels)} structures")
            output_dir = (
                settings.storage.mesh_path / str(seg_row.case_id) / self._seg_id
            )

            mesh_paths_dict = self._mesh_service.extract_and_process_all_structures(
                masks=masks,
                labels=labels,
                spacing=tuple(spacing),
                output_dir=output_dir,
                smooth_iterations=self._smooth_iterations,
                target_face_ratio=self._target_face_ratio,
            )

            # Quality validation
            self._progress(85, "Validating mesh quality")
            quality_report = await self._validate_mesh_quality(mesh_paths_dict)

            # Update database
            self._progress(95, "Updating segmentation record")
            mesh_storage_paths = {
                structure: {fmt: str(path) for fmt, path in paths.items()}
                for structure, paths in mesh_paths_dict.items()
            }
            async with get_db_context() as db:
                await db.execute(
                    update(SegmentationResult)
                    .where(SegmentationResult.id == self._seg_id)
                    .values(mesh_storage_paths=mesh_storage_paths)
                )

            self._progress(100, "Mesh extraction complete")

            logger.info(
                "mesh_extraction_pipeline_complete",
                seg_id=self._seg_id,
                structures=len(mesh_paths_dict),
                quality_issues=quality_report.get("issues", 0),
            )

            return {
                "segmentation_id": self._seg_id,
                "structures_extracted": len(mesh_paths_dict),
                "mesh_paths": mesh_storage_paths,
                "quality_report": quality_report,
            }

    def _load_mask(self, mask_path: str):
        """Load segmentation mask from NIfTI file."""
        try:
            import SimpleITK as sitk
            image = sitk.ReadImage(mask_path)
            masks = sitk.GetArrayFromImage(image).astype(np.int32)
            spacing = image.GetSpacing()
            return masks, spacing
        except ImportError:
            try:
                import nibabel as nib
                img = nib.load(mask_path)
                masks = np.asarray(img.dataobj).astype(np.int32)
                spacing = img.header.get_zooms()[:3]
                return masks, spacing
            except ImportError:
                raise RuntimeError("Neither SimpleITK nor nibabel is installed")

    async def _validate_mesh_quality(
        self, mesh_paths: Dict[str, Dict[str, Path]]
    ) -> Dict[str, Any]:
        """Validate extracted meshes meet quality thresholds."""
        report = {"structures_checked": 0, "issues": 0, "warnings": [], "errors": []}

        for structure_name, paths in mesh_paths.items():
            try:
                import trimesh
                glb_path = paths.get("glb") or paths.get("stl")
                if not glb_path or not glb_path.exists():
                    continue

                mesh = trimesh.load(str(glb_path))
                if not isinstance(mesh, trimesh.Trimesh):
                    continue

                report["structures_checked"] += 1
                metrics = self._mesh_service.compute_mesh_metrics(mesh)

                if metrics.vertex_count < self.MIN_VERTEX_COUNT:
                    report["warnings"].append(
                        f"{structure_name}: Low vertex count ({metrics.vertex_count})"
                    )
                    report["issues"] += 1

                if structure_name in self.MIN_WATERTIGHT_STRUCTURES:
                    if not metrics.is_watertight:
                        report["warnings"].append(
                            f"{structure_name}: Mesh is not watertight (volume computation unreliable)"
                        )

            except Exception as exc:
                report["errors"].append(f"{structure_name}: Validation failed: {exc}")
                report["issues"] += 1

        return report
