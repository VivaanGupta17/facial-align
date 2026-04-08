"""
DICOM ingestion service.
Handles DICOM parsing, volume reconstruction, PHI de-identification, and quality assessment.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.config import get_settings
from app.core.exceptions import (
    DicomDeidentificationError,
    DicomParseError,
    DicomValidationError,
    InsufficientCoverageError,
    SliceThicknessError,
)
from app.core.logging import TimedOperation, get_logger

settings = get_settings()
logger = get_logger(__name__)


@dataclass
class SeriesMetadata:
    """Metadata extracted from a single DICOM series."""
    series_instance_uid: str
    series_number: Optional[int]
    series_description: Optional[str]
    modality: str
    slice_count: int
    slice_thickness_mm: Optional[float]
    pixel_spacing: Optional[Tuple[float, float]]  # (row, col) spacing in mm
    image_orientation: Optional[List[float]]
    image_position_first: Optional[List[float]]
    image_position_last: Optional[List[float]]
    kvp: Optional[float]
    exposure_mas: Optional[float]
    reconstruction_diameter: Optional[float]
    file_paths: List[str] = field(default_factory=list)


@dataclass
class StudyMetadataInternal:
    """De-identified metadata for a complete DICOM study."""
    study_instance_uid: str
    study_description: Optional[str]
    modality: str
    acquisition_date: Optional[str]  # YYYYMMDD string from DICOM
    body_part_examined: Optional[str]
    institution_name: Optional[str]
    manufacturer: Optional[str]
    manufacturer_model: Optional[str]
    software_versions: Optional[str]
    series: List[SeriesMetadata] = field(default_factory=list)
    raw_tags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VolumeQualityResult:
    """Quality assessment result for a reconstructed CT volume."""
    quality_score: float  # 0.0 - 1.0
    slice_thickness_mm: Optional[float]
    pixel_spacing_mm: Optional[float]
    passed: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    cranial_coverage: bool = False
    mandible_coverage: bool = False


class DicomIngestionService:
    """
    Handles DICOM study ingestion end-to-end.

    Responsibilities:
    - Extract and expand uploaded archives
    - Parse DICOM metadata using pydicom
    - De-identify PHI from DICOM tags
    - Reconstruct 3D volume using SimpleITK
    - Assess CT quality for surgical planning
    - Save results to configured storage
    """

    # DICOM tags to remove for de-identification (subset of DICOM PS 3.15 Annex E)
    PHI_TAGS_TO_REMOVE = {
        "PatientName", "PatientID", "PatientBirthDate", "PatientSex",
        "PatientAge", "PatientWeight", "PatientAddress", "PatientTelephoneNumbers",
        "ReferringPhysicianName", "PerformingPhysicianName", "OperatorsName",
        "InstitutionName", "InstitutionAddress", "StationName",
        "StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate",
        "StudyTime", "SeriesTime", "AcquisitionTime", "ContentTime",
        "AccessionNumber", "StudyID",
        "RequestingPhysician", "ScheduledPerformingPhysicianName",
    }

    # Soft tissue and bone window presets for Hounsfield Units
    BONE_WINDOW_CENTER = 400.0
    BONE_WINDOW_WIDTH = 1500.0

    # Quality thresholds for CMF surgical planning
    MAX_SLICE_THICKNESS_MM = 1.5  # Must be ≤ 1.5mm for CMF planning
    TARGET_SLICE_THICKNESS_MM = 0.625  # Ideal for high-resolution planning
    MAX_PIXEL_SPACING_MM = 0.6  # Max in-plane resolution
    MIN_SLICE_COUNT = 200  # Minimum axial slices for full CMF coverage

    def __init__(self) -> None:
        settings.storage.temp_path.mkdir(parents=True, exist_ok=True)
        settings.storage.dicom_path.mkdir(parents=True, exist_ok=True)

    async def ingest_study(
        self,
        upload_path: Path,
        study_id: str,
        patient_id: str,
    ) -> StudyMetadataInternal:
        """
        Full ingestion pipeline for an uploaded DICOM study.

        Args:
            upload_path: Path to uploaded files (directory or ZIP)
            study_id: Database study ID for storage path organization
            patient_id: Patient ID for storage organization

        Returns:
            De-identified StudyMetadataInternal
        """
        with TimedOperation(logger, "dicom_ingestion", study_id=study_id):
            # Step 1: Extract archive if needed
            dicom_dir = await self._extract_dicom_files(upload_path)

            # Step 2: Parse DICOM metadata
            logger.info("parsing_dicom_metadata", path=str(dicom_dir))
            study_meta = await self.parse_dicom_metadata(dicom_dir)

            # Step 3: De-identify
            logger.info("deidentifying_dicom", study_uid=study_meta.study_instance_uid)
            await self.deidentify_study(dicom_dir)

            # Step 4: Move to permanent storage
            dest_path = settings.storage.dicom_path / patient_id / study_id
            dest_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(dicom_dir), str(dest_path / "dicom"), dirs_exist_ok=True)

            logger.info(
                "dicom_ingestion_complete",
                study_uid=study_meta.study_instance_uid,
                modality=study_meta.modality,
                series_count=len(study_meta.series),
                dest_path=str(dest_path),
            )

            return study_meta

    async def _extract_dicom_files(self, upload_path: Path) -> Path:
        """
        Extract ZIP archive or return directory of DICOM files.

        Args:
            upload_path: Path to uploaded content

        Returns:
            Path to directory containing .dcm files
        """
        if upload_path.is_dir():
            # Find all DICOM files recursively
            dcm_files = list(upload_path.rglob("*.dcm")) + list(upload_path.rglob("*.DCM"))
            if not dcm_files:
                # Some DICOM files have no extension
                # Try reading any file to see if it's DICOM
                all_files = list(upload_path.rglob("*"))
                dcm_files = [f for f in all_files if f.is_file() and self._is_dicom_file(f)]
            return upload_path

        if zipfile.is_zipfile(upload_path):
            extract_path = upload_path.parent / f"extracted_{upload_path.stem}"
            extract_path.mkdir(exist_ok=True)
            with zipfile.ZipFile(upload_path, "r") as zf:
                # Security: validate zip paths don't escape target directory
                for member in zf.infolist():
                    member_path = extract_path / member.filename
                    if not str(member_path.resolve()).startswith(str(extract_path.resolve())):
                        raise DicomParseError(
                            "Zip path traversal detected",
                            context={"member": member.filename},
                        )
                zf.extractall(extract_path)
            logger.info("archive_extracted", extract_path=str(extract_path))
            return extract_path

        raise DicomParseError(
            f"Upload must be a directory or ZIP archive: {upload_path}"
        )

    def _is_dicom_file(self, path: Path) -> bool:
        """Check if a file has a DICOM magic number."""
        try:
            with open(path, "rb") as f:
                f.seek(128)
                return f.read(4) == b"DICM"
        except (OSError, IOError):
            return False

    async def parse_dicom_metadata(self, dicom_dir: Path) -> StudyMetadataInternal:
        """
        Parse DICOM files and extract study/series metadata.

        Args:
            dicom_dir: Directory containing DICOM files

        Returns:
            Structured study metadata (not yet de-identified)
        """
        try:
            import pydicom
            from pydicom.errors import InvalidDicomError
        except ImportError:
            raise DicomParseError("pydicom not installed")

        # Find all DICOM files
        dicom_files: List[Path] = []
        for pattern in ["**/*.dcm", "**/*.DCM", "**/*"]:
            for p in dicom_dir.rglob(pattern.replace("**/", "")):
                if p.is_file() and p not in dicom_files:
                    dicom_files.append(p)

        if not dicom_files:
            raise DicomParseError(
                f"No DICOM files found in {dicom_dir}",
                context={"path": str(dicom_dir)},
            )

        # Parse first file for study-level metadata
        study_meta: Optional[StudyMetadataInternal] = None
        series_map: Dict[str, List[pydicom.Dataset]] = {}

        for fpath in dicom_files:
            try:
                ds = pydicom.dcmread(str(fpath), stop_before_pixels=True)
                if not hasattr(ds, "SOPInstanceUID"):
                    continue

                # Study-level metadata (from first file)
                if study_meta is None:
                    study_uid = str(getattr(ds, "StudyInstanceUID", f"UNKNOWN_{hashlib.md5(str(dicom_dir).encode()).hexdigest()[:8]}"))
                    study_meta = StudyMetadataInternal(
                        study_instance_uid=study_uid,
                        study_description=str(getattr(ds, "StudyDescription", "")),
                        modality=str(getattr(ds, "Modality", "CT")),
                        acquisition_date=str(getattr(ds, "AcquisitionDate", "") or getattr(ds, "StudyDate", "")),
                        body_part_examined=str(getattr(ds, "BodyPartExamined", "")),
                        institution_name=str(getattr(ds, "InstitutionName", "")),
                        manufacturer=str(getattr(ds, "Manufacturer", "")),
                        manufacturer_model=str(getattr(ds, "ManufacturerModelName", "")),
                        software_versions=str(getattr(ds, "SoftwareVersions", "")),
                    )

                # Group by series
                series_uid = str(getattr(ds, "SeriesInstanceUID", "unknown"))
                series_map.setdefault(series_uid, []).append(ds)

            except (InvalidDicomError, Exception) as e:
                logger.debug("skipping_non_dicom_file", path=str(fpath), error=str(e))
                continue

        if not study_meta:
            raise DicomParseError("Could not parse any valid DICOM files")

        # Build series metadata
        for series_uid, datasets in series_map.items():
            ds0 = datasets[0]
            try:
                pixel_spacing_raw = getattr(ds0, "PixelSpacing", None)
                pixel_spacing = (
                    (float(pixel_spacing_raw[0]), float(pixel_spacing_raw[1]))
                    if pixel_spacing_raw else None
                )
                series_meta = SeriesMetadata(
                    series_instance_uid=series_uid,
                    series_number=int(getattr(ds0, "SeriesNumber", 0) or 0),
                    series_description=str(getattr(ds0, "SeriesDescription", "")),
                    modality=str(getattr(ds0, "Modality", "CT")),
                    slice_count=len(datasets),
                    slice_thickness_mm=float(getattr(ds0, "SliceThickness", 0) or 0) or None,
                    pixel_spacing=pixel_spacing,
                    image_orientation=[float(v) for v in (getattr(ds0, "ImageOrientationPatient", []) or [])],
                    image_position_first=[float(v) for v in (getattr(ds0, "ImagePositionPatient", []) or [])],
                    image_position_last=None,
                    kvp=float(getattr(ds0, "KVP", 0) or 0) or None,
                    exposure_mas=float(getattr(ds0, "Exposure", 0) or 0) or None,
                    reconstruction_diameter=float(getattr(ds0, "ReconstructionDiameter", 0) or 0) or None,
                    file_paths=[str(fpath) for fpath in dicom_files],
                )
                study_meta.series.append(series_meta)
            except Exception as e:
                logger.warning("series_parse_error", series_uid=series_uid, error=str(e))

        logger.info(
            "dicom_metadata_parsed",
            study_uid=study_meta.study_instance_uid,
            modality=study_meta.modality,
            series_count=len(study_meta.series),
            total_slices=sum(s.slice_count for s in study_meta.series),
        )
        return study_meta

    async def reconstruct_volume(
        self, series_path: Path, output_path: Optional[Path] = None
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Reconstruct a 3D volumetric array from a DICOM series.

        Uses SimpleITK for robust DICOM reading with proper
        coordinate handling and orientation correction.

        Args:
            series_path: Path to directory containing one series
            output_path: If provided, save as NIfTI (.nii.gz)

        Returns:
            Tuple of (volume_array, (x_spacing, y_spacing, z_spacing))
        """
        try:
            import SimpleITK as sitk
        except ImportError:
            raise DicomParseError("SimpleITK not installed")

        with TimedOperation(logger, "volume_reconstruction", path=str(series_path)):
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(str(series_path))

            if not dicom_names:
                raise DicomParseError(
                    f"No DICOM series found in {series_path}",
                    context={"path": str(series_path)},
                )

            reader.SetFileNames(dicom_names)
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()

            image = reader.Execute()

            # Convert to numpy array (SimpleITK uses RAS convention)
            volume = sitk.GetArrayFromImage(image)  # Shape: (Z, Y, X)
            spacing = image.GetSpacing()  # (x, y, z) spacing in mm

            logger.info(
                "volume_reconstructed",
                shape=list(volume.shape),
                spacing_mm=list(spacing),
                dtype=str(volume.dtype),
            )

            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                sitk.WriteImage(image, str(output_path))
                logger.info("volume_saved", path=str(output_path))

            return volume, spacing

    async def validate_ct_quality(
        self, volume: np.ndarray, spacing: Tuple[float, float, float]
    ) -> VolumeQualityResult:
        """
        Validate CT volume quality for craniofacial surgical planning.

        Checks:
        - Slice thickness ≤ 1.5mm (required for bone detail)
        - In-plane resolution ≤ 0.6mm
        - Sufficient anatomical coverage (cranium through mandible)
        - Minimum slice count

        Args:
            volume: 3D numpy array (Z, Y, X)
            spacing: Voxel spacing (x_mm, y_mm, z_mm)

        Returns:
            VolumeQualityResult with score and flags
        """
        warnings: List[str] = []
        errors: List[str] = []
        score_components: List[float] = []

        x_spacing, y_spacing, z_spacing = spacing
        slice_thickness = z_spacing
        pixel_spacing = (x_spacing + y_spacing) / 2

        # ── Check slice thickness ──
        if slice_thickness > self.MAX_SLICE_THICKNESS_MM:
            errors.append(
                f"Slice thickness {slice_thickness:.2f}mm exceeds maximum {self.MAX_SLICE_THICKNESS_MM}mm "
                f"for CMF surgical planning"
            )
            score_components.append(0.0)
        elif slice_thickness > 0.8:
            warnings.append(f"Slice thickness {slice_thickness:.2f}mm is acceptable but not ideal (target: ≤0.625mm)")
            score_components.append(0.7)
        else:
            score_components.append(1.0)

        # ── Check in-plane resolution ──
        if pixel_spacing > self.MAX_PIXEL_SPACING_MM:
            warnings.append(f"Pixel spacing {pixel_spacing:.3f}mm may reduce surgical planning accuracy")
            score_components.append(0.6)
        else:
            score_components.append(1.0)

        # ── Check slice count ──
        z_slices = volume.shape[0]
        if z_slices < self.MIN_SLICE_COUNT:
            warnings.append(
                f"Slice count {z_slices} may be insufficient for full CMF coverage "
                f"(recommended: ≥{self.MIN_SLICE_COUNT})"
            )
            score_components.append(0.7)
        else:
            score_components.append(1.0)

        # ── Check anatomical coverage using HU statistics ──
        # Approximate coverage by looking for high-density (bone HU > 200) voxels
        bone_threshold = 200
        bone_mask = volume > bone_threshold
        bone_slices = np.any(bone_mask, axis=(1, 2))

        cranial_coverage = bool(np.any(bone_slices[:z_slices // 4]))
        mandible_coverage = bool(np.any(bone_slices[3 * z_slices // 4:]))

        if not cranial_coverage:
            warnings.append("Cranial base coverage may be incomplete")
        if not mandible_coverage:
            warnings.append("Mandible coverage may be incomplete")

        coverage_score = (0.5 * int(cranial_coverage) + 0.5 * int(mandible_coverage))
        score_components.append(coverage_score if coverage_score > 0 else 0.5)

        # ── Compute overall score ──
        quality_score = float(np.mean(score_components)) if score_components else 0.0
        passed = quality_score >= 0.6 and len(errors) == 0

        result = VolumeQualityResult(
            quality_score=round(quality_score, 3),
            slice_thickness_mm=slice_thickness,
            pixel_spacing_mm=pixel_spacing,
            passed=passed,
            warnings=warnings,
            errors=errors,
            cranial_coverage=cranial_coverage,
            mandible_coverage=mandible_coverage,
        )

        logger.info(
            "ct_quality_assessed",
            quality_score=quality_score,
            passed=passed,
            warnings=len(warnings),
            errors=len(errors),
        )
        return result

    async def deidentify_study(self, dicom_dir: Path) -> None:
        """
        Remove PHI from all DICOM files in a directory.

        Implements DICOM PS 3.15 Basic Application Level Confidentiality Profile.
        Modifies files in-place.

        Args:
            dicom_dir: Directory of DICOM files to de-identify
        """
        try:
            import pydicom
            from pydicom.uid import generate_uid
        except ImportError:
            raise DicomDeidentificationError("pydicom not installed")

        dicom_files = list(dicom_dir.rglob("*.dcm")) + list(dicom_dir.rglob("*.DCM"))
        success_count = 0
        error_count = 0

        # Generate consistent UID replacements for this study
        uid_map: Dict[str, str] = {}

        for fpath in dicom_files:
            try:
                ds = pydicom.dcmread(str(fpath))

                # Remove/replace PHI tags
                for tag_name in self.PHI_TAGS_TO_REMOVE:
                    if hasattr(ds, tag_name):
                        try:
                            delattr(ds, tag_name)
                        except AttributeError:
                            pass

                # Replace UID-based identifiers consistently
                for uid_tag in ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"]:
                    if hasattr(ds, uid_tag):
                        original_uid = str(getattr(ds, uid_tag))
                        if original_uid not in uid_map:
                            uid_map[original_uid] = generate_uid()
                        setattr(ds, uid_tag, uid_map[original_uid])

                # Mark as de-identified
                ds.PatientIdentityRemoved = "YES"
                ds.DeidentificationMethod = "DICOM PS 3.15 Basic Application Level Confidentiality Profile"

                pydicom.dcmwrite(str(fpath), ds)
                success_count += 1

            except Exception as e:
                logger.error("deidentification_file_error", path=str(fpath), error=str(e))
                error_count += 1

        if error_count > 0:
            logger.warning(
                "deidentification_partial",
                success=success_count,
                errors=error_count,
            )

        if error_count > success_count:
            raise DicomDeidentificationError(
                f"De-identification failed for majority of files ({error_count} failures)",
                context={"success": success_count, "errors": error_count},
            )

        logger.info(
            "deidentification_complete",
            files_processed=success_count,
            uid_replacements=len(uid_map),
        )
