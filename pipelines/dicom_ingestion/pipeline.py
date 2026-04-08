"""
DICOM ingestion pipeline.
Orchestrates the complete flow from raw DICOM upload to stored, quality-checked volume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from app.core.config import get_settings
from app.core.exceptions import DicomValidationError
from app.core.logging import TimedOperation, get_logger
from app.services.dicom.ingestion import DicomIngestionService

settings = get_settings()
logger = get_logger(__name__)


class DicomIngestionPipeline:
    """
    End-to-end DICOM ingestion pipeline.

    Stages:
    0%  - Initialize
    10% - Extract uploaded files
    25% - Parse DICOM metadata
    40% - De-identify PHI
    55% - Reconstruct 3D volume (NIfTI)
    75% - Validate CT quality
    85% - Save to permanent storage
    95% - Update database record
    100% - Complete
    """

    def __init__(
        self,
        study_id: str,
        upload_path: Path,
        patient_id: str,
        user_id: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        self._study_id = study_id
        self._upload_path = upload_path
        self._patient_id = patient_id
        self._user_id = user_id
        self._progress = progress_callback or (lambda pct, step: None)
        self._service = DicomIngestionService()

    async def run(self) -> Dict[str, Any]:
        """
        Execute the full ingestion pipeline.

        Returns:
            Result dictionary with study metadata and quality assessment.
        """
        logger.info(
            "dicom_ingestion_pipeline_started",
            study_id=self._study_id,
            upload_path=str(self._upload_path),
        )

        with TimedOperation(logger, "dicom_ingestion_pipeline", study_id=self._study_id):
            # Stage 1: Extract
            self._progress(10, "Extracting uploaded DICOM files")
            dicom_dir = await self._service._extract_dicom_files(self._upload_path)

            # Stage 2: Parse metadata
            self._progress(25, "Parsing DICOM metadata")
            study_meta = await self._service.parse_dicom_metadata(dicom_dir)

            # Stage 3: De-identify
            self._progress(40, "Removing PHI from DICOM tags")
            await self._service.deidentify_study(dicom_dir)

            # Stage 4: Reconstruct volume
            self._progress(55, "Reconstructing 3D CT volume")
            dest_path = settings.storage.dicom_path / self._patient_id / self._study_id
            dest_path.mkdir(parents=True, exist_ok=True)
            volume_path = dest_path / "volume.nii.gz"

            # Use the primary CT series (largest slice count)
            primary_series = max(study_meta.series, key=lambda s: s.slice_count) \
                if study_meta.series else None

            quality_score = None
            quality_passed = False
            quality_flags = []

            if primary_series and primary_series.file_paths:
                series_dir = Path(primary_series.file_paths[0]).parent
                try:
                    volume, spacing = await self._service.reconstruct_volume(
                        series_dir, output_path=volume_path
                    )

                    # Stage 5: Quality assessment
                    self._progress(75, "Assessing CT quality")
                    quality = await self._service.validate_ct_quality(volume, spacing)
                    quality_score = quality.quality_score
                    quality_passed = quality.passed
                    quality_flags = quality.warnings + quality.errors

                    if not quality_passed and quality.errors:
                        logger.warning(
                            "ct_quality_below_threshold",
                            study_id=self._study_id,
                            errors=quality.errors,
                        )

                except Exception as exc:
                    logger.error(
                        "volume_reconstruction_failed",
                        study_id=self._study_id,
                        error=str(exc),
                    )
                    quality_flags.append(f"Volume reconstruction failed: {exc}")

            # Stage 6: Update database
            self._progress(90, "Updating study record")
            await self._update_study_record(
                study_meta=study_meta,
                volume_path=volume_path if volume_path.exists() else None,
                quality_score=quality_score,
                quality_flags=quality_flags,
                primary_series=primary_series,
                dest_path=dest_path,
            )

            self._progress(100, "Ingestion complete")

            result = {
                "study_id": self._study_id,
                "study_uid": study_meta.study_instance_uid,
                "modality": study_meta.modality,
                "series_count": len(study_meta.series),
                "total_slices": sum(s.slice_count for s in study_meta.series),
                "quality_score": quality_score,
                "quality_passed": quality_passed,
                "quality_flags": quality_flags,
                "volume_path": str(volume_path) if volume_path.exists() else None,
                "storage_path": str(dest_path),
            }

            logger.info(
                "dicom_ingestion_pipeline_complete",
                study_id=self._study_id,
                quality_score=quality_score,
                quality_passed=quality_passed,
            )

            return result

    async def _update_study_record(
        self,
        study_meta,
        volume_path: Optional[Path],
        quality_score: Optional[float],
        quality_flags: list,
        primary_series,
        dest_path: Path,
    ) -> None:
        """Update the ImagingStudy database record with ingestion results."""
        from app.db.database import get_db_context
        from app.models.study import ImagingStudy
        from sqlalchemy import update
        import uuid

        values = {
            "study_uid": study_meta.study_instance_uid,
            "modality": study_meta.modality,
            "series_count": len(study_meta.series),
            "slice_count": sum(s.slice_count for s in study_meta.series),
            "storage_path": str(dest_path / "dicom"),
            "volume_path": str(volume_path) if volume_path and volume_path.exists() else None,
            "ingestion_status": "complete",
            "is_deidentified": True,
            "quality_score": quality_score,
            "quality_flags": quality_flags,
            "body_part_examined": study_meta.body_part_examined,
            "metadata_json": {
                "StudyDescription": study_meta.study_description,
                "InstitutionName": study_meta.institution_name,
                "Manufacturer": study_meta.manufacturer,
                "ManufacturerModelName": study_meta.manufacturer_model,
                "SoftwareVersions": study_meta.software_versions,
            },
        }

        if primary_series:
            values["slice_thickness_mm"] = primary_series.slice_thickness_mm
            if primary_series.pixel_spacing:
                values["pixel_spacing_mm"] = primary_series.pixel_spacing[0]
            values["kv_peak"] = primary_series.kvp

        try:
            async with get_db_context() as db:
                await db.execute(
                    update(ImagingStudy)
                    .where(ImagingStudy.id == self._study_id)
                    .values(**values)
                )
        except Exception as exc:
            logger.error(
                "study_record_update_failed",
                study_id=self._study_id,
                error=str(exc),
            )
