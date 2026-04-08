"""
DICOM-related Pydantic schemas for upload, metadata, and study management.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, List, Optional

from pydantic import Field, field_validator

from app.schemas.common import BaseSchema


class SeriesInfo(BaseSchema):
    """Information about a single DICOM series."""
    series_instance_uid: str = Field(..., description="DICOM SeriesInstanceUID")
    series_number: Optional[int] = Field(None, description="DICOM SeriesNumber")
    series_description: Optional[str] = Field(None, description="DICOM SeriesDescription")
    modality: str = Field(..., description="DICOM Modality tag (CT, MR, etc.)")
    slice_count: int = Field(..., description="Number of slices/instances in this series")
    slice_thickness_mm: Optional[float] = Field(None, description="Slice thickness in mm")
    pixel_spacing_mm: Optional[list[float]] = Field(
        None, description="In-plane pixel spacing [row, col] in mm"
    )
    image_orientation: Optional[list[float]] = Field(
        None, description="DICOM ImageOrientationPatient (6 values)"
    )
    reconstruction_diameter_mm: Optional[float] = Field(
        None, description="CT reconstruction diameter (FOV) in mm"
    )
    kvp: Optional[float] = Field(None, description="CT tube voltage (kVp)")
    exposure_mas: Optional[float] = Field(None, description="CT exposure in mAs")


class StudyMetadata(BaseSchema):
    """De-identified DICOM study metadata extracted during ingestion."""
    study_uid: str = Field(..., description="De-identified DICOM StudyInstanceUID")
    modality: str = Field(..., description="Primary imaging modality")
    acquisition_date: Optional[date] = Field(None, description="Date imaging was acquired")
    body_part_examined: Optional[str] = Field(None, description="DICOM BodyPartExamined")
    study_description: Optional[str] = Field(None, description="DICOM StudyDescription")
    series: List[SeriesInfo] = Field(default_factory=list, description="Series in this study")
    series_count: int = Field(0, description="Total number of series")
    total_slice_count: int = Field(0, description="Total slice count across all series")
    institution_name: Optional[str] = Field(
        None, description="Institution name (de-identified if PHI)"
    )
    manufacturer: Optional[str] = Field(None, description="Equipment manufacturer")
    manufacturer_model: Optional[str] = Field(None, description="Equipment model name")
    software_versions: Optional[str] = Field(None, description="Reconstruction software version")
    additional_tags: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional de-identified DICOM tags as key-value pairs"
    )


class DicomUploadRequest(BaseSchema):
    """Request schema for DICOM study upload metadata."""
    patient_mrn: str = Field(
        ...,
        description="Patient MRN (will be hashed; never stored in plaintext)",
        min_length=1,
        max_length=64,
    )
    patient_age: Optional[int] = Field(
        None, ge=0, le=120, description="Patient age in years"
    )
    patient_sex: Optional[str] = Field(
        None, description="Biological sex: M, F, O, U"
    )
    institution_code: Optional[str] = Field(
        None, description="Treating institution code"
    )
    case_type: Optional[str] = Field(
        None,
        description="Intended case type hint: TRAUMA, ORTHOGNATHIC, RECONSTRUCTION"
    )
    upload_notes: Optional[str] = Field(
        None, max_length=1000, description="Optional upload notes"
    )

    @field_validator("patient_sex")
    @classmethod
    def validate_sex(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.upper() not in {"M", "F", "O", "U"}:
            raise ValueError("patient_sex must be one of: M, F, O, U")
        return v.upper() if v else v


class DicomUploadResponse(BaseSchema):
    """Response after a successful DICOM study upload and ingestion trigger."""
    upload_id: uuid.UUID = Field(..., description="Upload session UUID")
    study_id: uuid.UUID = Field(..., description="Created ImagingStudy UUID")
    patient_id: uuid.UUID = Field(..., description="Patient UUID (created or matched)")
    ingestion_job_id: str = Field(..., description="Celery job ID for ingestion pipeline")
    study_uid: str = Field(..., description="De-identified DICOM StudyInstanceUID")
    storage_path: str = Field(..., description="Temporary storage path for uploaded files")
    status: str = Field(default="processing", description="Ingestion status")
    message: str = Field(default="DICOM study received and ingestion started")


class DicomQualityReport(BaseSchema):
    """CT quality assessment report produced during ingestion."""
    study_id: uuid.UUID
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
    slice_thickness_mm: Optional[float] = None
    min_acceptable_thickness_mm: float = Field(default=1.5)
    pixel_spacing_mm: Optional[float] = None
    anatomical_coverage: Optional[str] = Field(
        None, description="Coverage assessment: full, partial, insufficient"
    )
    cranial_coverage_complete: bool = False
    mandible_coverage_complete: bool = False
    artifact_level: Optional[str] = Field(
        None, description="Artifact assessment: none, mild, moderate, severe"
    )
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    passed: bool = Field(..., description="Whether the study passed quality thresholds")


class DicomStudyListItem(BaseSchema):
    """Compact study representation for list endpoints."""
    id: uuid.UUID
    study_uid: str
    patient_id: uuid.UUID
    modality: str
    acquisition_date: Optional[date]
    series_count: int
    ingestion_status: str
    quality_score: Optional[float]
    created_at: datetime


class DicomSeriesDownloadResponse(BaseSchema):
    """Pre-signed URL for downloading a DICOM series."""
    series_instance_uid: str
    download_url: str = Field(..., description="Pre-signed URL for download (expires in 1 hour)")
    file_format: str = Field(default="zip", description="Archive format: zip, tar.gz")
    file_size_bytes: Optional[int] = None
    expires_at: datetime
