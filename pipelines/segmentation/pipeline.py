"""
Segmentation pipeline: preprocess → segment → postprocess → mesh extraction.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from app.core.config import get_settings
from app.core.logging import TimedOperation, get_logger

settings = get_settings()
logger = get_logger(__name__)


class SegmentationPipeline:
    """
    Full segmentation pipeline for a CT study.

    Stages:
    5%  - Load CT volume from storage
    15% - Preprocess (normalize, validate)
    20% - Initialize ML model
    30% - Run bone segmentation inference
    50% - Post-process segmentation masks
    60% - (Optional) Run dental segmentation
    70% - (Optional) Identify fracture fragments
    75% - Extract surface meshes
    88% - Export GLB/STL files for viewer
    95% - Update database record
    100% - Complete
    """

    def __init__(
        self,
        segmentation_id: str,
        case_id: str,
        study_id: str,
        model_name: str = "totalsegmentator",
        structures: Optional[List[str]] = None,
        run_dental: bool = False,
        identify_fragments: bool = True,
        fast_mode: bool = False,
        gpu_device: Optional[str] = None,
        user_id: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        self._seg_id = segmentation_id
        self._case_id = case_id
        self._study_id = study_id
        self._model_name = model_name
        self._structures = structures
        self._run_dental = run_dental
        self._identify_fragments = identify_fragments
        self._fast_mode = fast_mode
        self._gpu_device = gpu_device or settings.model_registry.default_device
        self._user_id = user_id
        self._progress = progress_callback or (lambda pct, step: None)

    async def run(self) -> Dict[str, Any]:
        """Execute the full segmentation pipeline."""
        from app.db.database import get_db_context
        from app.models.segmentation import SegmentationResult
        from app.models.study import ImagingStudy
        from app.services.segmentation.segmentation_service import ModelRegistry, SegmentationService
        from app.services.mesh.mesh_service import MeshService
        from sqlalchemy import select, update

        logger.info(
            "segmentation_pipeline_started",
            seg_id=self._seg_id,
            model=self._model_name,
        )

        # ── Load CT volume ──
        self._progress(5, "Loading CT volume from storage")
        volume, spacing = await self._load_volume()
        logger.info(
            "volume_loaded",
            shape=list(volume.shape),
            spacing_mm=list(spacing),
        )

        # ── Initialize model registry ──
        self._progress(20, f"Loading segmentation model: {self._model_name}")
        registry = ModelRegistry(settings.model_registry)

        # ── Segmentation inference ──
        self._progress(30, "Running bone segmentation")
        seg_service = SegmentationService(model_registry=registry)
        seg_output = await seg_service.segment_structures(
            volume=volume,
            spacing=spacing,
            model_name=self._model_name,
            structures=self._structures,
            fast_mode=self._fast_mode,
        )

        # ── Dental segmentation ──
        dental_output = None
        if self._run_dental:
            self._progress(60, "Running dental segmentation")
            from app.services.segmentation.dental_segmentation import DentalSegmentationService
            dental_service = DentalSegmentationService(
                model_path=settings.model_registry.dental_segmentation_model_path,
                device=self._gpu_device,
            )
            try:
                dental_output = await dental_service.segment_teeth(volume, spacing)
                logger.info(
                    "dental_segmentation_complete",
                    teeth_found=len(dental_output.present_teeth),
                )
            except Exception as exc:
                logger.warning("dental_segmentation_failed", error=str(exc))

        # ── Fragment identification ──
        fragment_count = None
        fragment_mask_path = None
        if self._identify_fragments and len(seg_output.labels) > 0:
            self._progress(70, "Identifying fracture fragments")
            fragment_count, fragment_mask_path = await self._identify_fragments_from_masks(
                seg_output, volume, spacing
            )

        # ── Save masks ──
        self._progress(75, "Saving segmentation masks")
        mask_path = await self._save_masks(seg_output, volume)

        # ── Extract meshes ──
        self._progress(78, "Extracting surface meshes")
        mesh_service = MeshService()
        output_dir = settings.storage.mesh_path / self._case_id / self._seg_id
        mesh_paths_dict = mesh_service.extract_and_process_all_structures(
            masks=seg_output.masks,
            labels=seg_output.labels,
            spacing=spacing,
            output_dir=output_dir,
        )

        # Convert Path objects to strings for JSON storage
        mesh_storage_paths = {
            structure: {fmt: str(path) for fmt, path in paths.items()}
            for structure, paths in mesh_paths_dict.items()
        }

        # ── Dental meshes ──
        dental_mesh_paths = {}
        if dental_output and dental_output.tooth_masks:
            dental_output_dir = output_dir / "dental"
            dental_output_dir.mkdir(parents=True, exist_ok=True)
            for fdi_num, tooth_mask in dental_output.tooth_masks.items():
                if tooth_mask is not None and np.any(tooth_mask):
                    try:
                        tooth_mesh = mesh_service.extract_mesh_from_mask(
                            tooth_mask.astype(np.int32), spacing, label=1
                        )
                        glb_p = mesh_service.export_glb(
                            tooth_mesh, dental_output_dir / f"tooth_{fdi_num}.glb"
                        )
                        stl_p = mesh_service.export_stl(
                            tooth_mesh, dental_output_dir / f"tooth_{fdi_num}.stl"
                        )
                        dental_mesh_paths[str(fdi_num)] = {
                            "glb": str(glb_p), "stl": str(stl_p)
                        }
                    except Exception as exc:
                        logger.warning(f"Tooth {fdi_num} mesh failed: {exc}")

        # ── Update database ──
        self._progress(95, "Updating segmentation record")
        now = datetime.now(timezone.utc)
        await self._update_segmentation_record(
            seg_output=seg_output,
            mask_path=mask_path,
            mesh_storage_paths=mesh_storage_paths,
            dental_mesh_paths=dental_mesh_paths,
            fragment_count=fragment_count,
            fragment_mask_path=fragment_mask_path,
            completed_at=now,
        )

        self._progress(100, "Segmentation complete")
        logger.info(
            "segmentation_pipeline_complete",
            seg_id=self._seg_id,
            structures=len(seg_output.labels),
            inference_ms=seg_output.inference_time_ms,
        )

        return {
            "segmentation_id": self._seg_id,
            "status": "complete",
            "structures_found": len(seg_output.labels),
            "overall_confidence": seg_output.confidences
                and round(float(np.mean(list(seg_output.confidences.values()))), 3),
            "mesh_count": len(mesh_storage_paths),
            "fragment_count": fragment_count,
            "inference_time_ms": seg_output.inference_time_ms,
        }

    async def _load_volume(self):
        """Load the CT volume for this study."""
        from app.db.database import get_db_context
        from app.models.study import ImagingStudy
        from sqlalchemy import select

        async with get_db_context() as db:
            study = (
                await db.execute(
                    select(ImagingStudy).where(ImagingStudy.id == self._study_id)
                )
            ).scalar_one_or_none()

        if not study:
            raise ValueError(f"Study {self._study_id} not found")

        if study.volume_path and Path(study.volume_path).exists():
            # Load pre-reconstructed NIfTI volume
            import SimpleITK as sitk
            image = sitk.ReadImage(str(study.volume_path))
            volume = sitk.GetArrayFromImage(image)
            spacing = image.GetSpacing()
            return volume, spacing
        elif study.storage_path and Path(study.storage_path).exists():
            # Reconstruct from DICOM
            from app.services.dicom.ingestion import DicomIngestionService
            svc = DicomIngestionService()
            return await svc.reconstruct_volume(Path(study.storage_path))
        else:
            raise FileNotFoundError(
                f"No volume data found for study {self._study_id}. "
                f"storage_path={study.storage_path}, volume_path={study.volume_path}"
            )

    async def _save_masks(self, seg_output, volume) -> Optional[str]:
        """Save segmentation mask volume as NIfTI."""
        try:
            import SimpleITK as sitk
            mask_dir = settings.storage.mask_path / self._case_id / self._seg_id
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask_path = mask_dir / "segmentation.nii.gz"

            mask_image = sitk.GetImageFromArray(seg_output.masks)
            mask_image.SetSpacing(seg_output.spacing_mm)
            sitk.WriteImage(mask_image, str(mask_path))
            return str(mask_path)
        except Exception as exc:
            logger.warning("mask_save_failed", error=str(exc))
            return None

    async def _identify_fragments_from_masks(
        self, seg_output, volume, spacing
    ):
        """Identify individual bone fragments using connected component analysis."""
        try:
            from scipy import ndimage
            fragment_masks = np.zeros_like(seg_output.masks, dtype=np.int32)
            total_fragments = 0

            # Apply connected component analysis to each structure
            for struct_name, label_val in seg_output.labels.items():
                struct_mask = (seg_output.masks == label_val).astype(np.uint8)
                labeled, n_components = ndimage.label(struct_mask)

                if n_components > 1:
                    logger.info(
                        "fragments_identified",
                        structure=struct_name,
                        n_fragments=n_components,
                    )
                    fragment_masks[labeled > 0] = labeled + total_fragments
                    total_fragments += n_components
                else:
                    fragment_masks[struct_mask > 0] = total_fragments + 1
                    total_fragments += 1

            # Save fragment mask
            if total_fragments > 0:
                import SimpleITK as sitk
                mask_dir = settings.storage.mask_path / self._case_id / self._seg_id
                frag_path = mask_dir / "fragments.nii.gz"
                frag_image = sitk.GetImageFromArray(fragment_masks)
                frag_image.SetSpacing(spacing)
                sitk.WriteImage(frag_image, str(frag_path))
                return total_fragments, str(frag_path)

        except Exception as exc:
            logger.warning("fragment_identification_failed", error=str(exc))

        return None, None

    async def _update_segmentation_record(
        self,
        seg_output,
        mask_path: Optional[str],
        mesh_storage_paths: Dict[str, Any],
        dental_mesh_paths: Dict[str, Any],
        fragment_count: Optional[int],
        fragment_mask_path: Optional[str],
        completed_at,
    ) -> None:
        """Update SegmentationResult record with pipeline results."""
        from app.db.database import get_db_context
        from app.models.segmentation import SegmentationResult
        from sqlalchemy import update

        overall_confidence = float(np.mean(list(seg_output.confidences.values()))) \
            if seg_output.confidences else None

        async with get_db_context() as db:
            await db.execute(
                update(SegmentationResult)
                .where(SegmentationResult.id == self._seg_id)
                .values(
                    model_version=seg_output.model_version,
                    structure_labels=seg_output.labels,
                    mask_storage_path=mask_path,
                    mesh_storage_paths=mesh_storage_paths,
                    confidence_scores=seg_output.confidences,
                    overall_confidence=overall_confidence,
                    inference_time_ms=seg_output.inference_time_ms,
                    volume_stats=seg_output.volume_stats,
                    dental_mesh_paths=dental_mesh_paths or None,
                    fragment_count=fragment_count,
                    fragment_masks_path=fragment_mask_path,
                    status="complete",
                    completed_at=completed_at,
                )
            )
