"""
Data contract for CT imaging study.

This module defines the canonical schemas for CT study and series data as
they flow across service boundaries: DICOM ingestion → segmentation →
fracture planning → occlusion analysis.

A ``CTStudyContract`` encapsulates:
- De-identified DICOM metadata (UIDs, acquisition parameters, scanner info)
- Quality grade and flags from automated quality control
- References to derived artefacts (NIfTI volume, per-slice storage)
- A clinical suitability check (``is_suitable_for_planning()``) used by
  upstream services to gate processing.

Clinical context
----------------
Craniofacial surgical planning requires sub-millimetre isotropic CT data.
Minimum acceptable parameters for reliable segmentation and mesh extraction:
  * Slice thickness ≤ 1.25 mm (ideally ≤ 0.625 mm)
  * In-plane pixel spacing ≤ 0.5 mm (ideally ≤ 0.4 mm)
  * kVp ≥ 100 (bone window quality)
  * Coverage: from skull vertex to C3 (captures mandible and temporomandibular joints)
"""

from __future__ import annotations

import json
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
MillimeterValue = float
HounsfieldUnit = float


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class CTQualityGrade(str, Enum):
    """
    CT scan quality grade for craniofacial surgical planning.

    A — Optimal: ≤0.625 mm isotropic, no artefacts.  Suitable for full planning.
    B — Acceptable: ≤1.25 mm, mild artefacts.  Suitable with caveats.
    C — Marginal: ≤2.5 mm or significant artefacts.  Use with caution.
    D — Unacceptable: >2.5 mm or severe artefacts.  Rescan required.
    """
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class CTModality(str, Enum):
    CT = "CT"
    CBCT = "CBCT"   # Cone-beam CT (lower resolution, dental context)


# ---------------------------------------------------------------------------
# Series-level contract
# ---------------------------------------------------------------------------

class CTSeriesContract(BaseModel):
    """
    A single CT image series within a study.

    In DICOM terminology, a series groups images acquired under the same
    protocol parameters and orientation.  For CMF planning the relevant
    series is the thin-slice axial reconstruction.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [{
                "series_instance_uid": "2.25.12345678901234567890",
                "modality": "CT",
                "series_number": 1,
                "series_description": "Axial Head 0.625mm",
                "slice_count": 320,
                "slice_thickness_mm": 0.625,
                "pixel_spacing_mm": [0.488, 0.488],
                "kvp": 120.0,
                "exposure_mas": 200.0,
                "reconstruction_diameter_mm": 220.0,
                "convolution_kernel": "B30f",
                "storage_path": "/data/series/001/",
            }]
        },
    )

    series_instance_uid: str = Field(..., description="DICOM SeriesInstanceUID")
    modality: str = Field(default="CT", description="DICOM Modality (CT or CBCT)")
    series_number: Optional[int] = Field(None, description="DICOM SeriesNumber")
    series_description: Optional[str] = Field(None, description="DICOM SeriesDescription")

    # Geometry
    slice_count: int = Field(..., ge=1, description="Number of images in series")
    slice_thickness_mm: Optional[MillimeterValue] = Field(
        None, gt=0, description="Slice thickness in mm (DICOM SliceThickness)"
    )
    pixel_spacing_mm: Optional[List[MillimeterValue]] = Field(
        None, min_length=2, max_length=2,
        description="[row_spacing, col_spacing] in mm (DICOM PixelSpacing)"
    )
    image_orientation_patient: Optional[List[float]] = Field(
        None, min_length=6, max_length=6,
        description="Direction cosines of row and column axes (DICOM ImageOrientationPatient)"
    )
    image_position_first_mm: Optional[List[MillimeterValue]] = Field(
        None, description="ImagePositionPatient of first slice [x, y, z] in mm"
    )
    image_position_last_mm: Optional[List[MillimeterValue]] = Field(
        None, description="ImagePositionPatient of last slice [x, y, z] in mm"
    )

    # Acquisition parameters
    kvp: Optional[float] = Field(None, gt=0, description="Peak kilovoltage (kVp)")
    exposure_mas: Optional[float] = Field(None, gt=0, description="Effective mAs (ExposureMAs)")
    ctdi_vol: Optional[float] = Field(None, ge=0, description="CT Dose Index volume (mGy)")
    reconstruction_diameter_mm: Optional[MillimeterValue] = Field(
        None, description="Reconstruction FOV diameter in mm"
    )
    convolution_kernel: Optional[str] = Field(
        None, description="Reconstruction kernel (e.g. 'B30f' for soft tissue)"
    )
    gantry_tilt_degrees: float = Field(
        default=0.0, description="Gantry tilt in degrees (should be 0 for CMF)"
    )

    # Storage
    storage_path: Optional[str] = Field(None, description="Path to DICOM slice directory")

    @field_validator("pixel_spacing_mm")
    @classmethod
    def validate_pixel_spacing(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is not None:
            if any(ps <= 0 for ps in v):
                raise ValueError("pixel_spacing_mm values must be positive")
            if v[0] / v[1] > 2.0 or v[1] / v[0] > 2.0:
                raise ValueError(
                    "pixel_spacing_mm aspect ratio >2:1 is atypical — verify DICOM tag"
                )
        return v

    @property
    def is_isotropic(self) -> bool:
        """Return True if pixel spacing and slice thickness are within 10% of each other."""
        if self.pixel_spacing_mm and self.slice_thickness_mm:
            ps = self.pixel_spacing_mm[0]
            return abs(ps - self.slice_thickness_mm) / max(ps, self.slice_thickness_mm) < 0.10
        return False

    @property
    def coverage_mm(self) -> Optional[MillimeterValue]:
        """Computed z-axis coverage from first to last slice position."""
        if self.image_position_first_mm and self.image_position_last_mm:
            return abs(self.image_position_last_mm[2] - self.image_position_first_mm[2])
        if self.slice_count and self.slice_thickness_mm:
            return self.slice_count * self.slice_thickness_mm
        return None


# ---------------------------------------------------------------------------
# Study-level contract
# ---------------------------------------------------------------------------

class CTStudyContract(BaseModel):
    """
    Canonical CT study data contract.

    Shared between ingestion, quality-control, segmentation, and planning
    services.  All PHI (PatientName, PatientBirthDate, etc.) MUST be removed
    before this contract is populated (``is_deidentified`` must be ``True``).

    Quality gating
    --------------
    Call ``is_suitable_for_planning()`` before dispatching to segmentation.
    The method checks:
      1. Modality is CT (not MR, CBCT below threshold)
      2. Best series has slice_thickness ≤ 1.25 mm
      3. Quality grade is A or B
      4. No critical quality flags (``motion_severe``, ``insufficient_coverage``)
    """
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [{
                "study_uid": "2.25.99887766554433221100",
                "patient_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "modality": "CT",
                "acquisition_date": "2024-03-15",
                "body_part_examined": "HEAD",
                "institution_name": "University Medical Center",
                "manufacturer": "Siemens Healthineers",
                "manufacturer_model": "SOMATOM Definition Flash",
                "software_versions": "syngo.CT 2021A",
                "volume_path": "/data/studies/abc123/volume.nii.gz",
                "spacing_mm": [0.488, 0.488, 0.625],
                "volume_shape": [320, 512, 512],
                "quality_grade": "A",
                "quality_score": 0.95,
                "is_deidentified": True,
            }]
        },
    )

    # Identifiers
    study_uid: str = Field(..., description="De-identified StudyInstanceUID (from DICOM)")
    patient_id: str = Field(..., description="Internal patient UUID (not DICOM PatientID)")

    # Modality and anatomy
    modality: CTModality = Field(default=CTModality.CT, description="Imaging modality")
    acquisition_date: Optional[date] = Field(None, description="DICOM AcquisitionDate")
    body_part_examined: Optional[str] = Field(
        None, description="DICOM BodyPartExamined (expected: HEAD)"
    )

    # Scanner provenance
    institution_name: Optional[str] = None
    manufacturer: Optional[str] = None
    manufacturer_model: Optional[str] = None
    software_versions: Optional[str] = None
    station_name: Optional[str] = None

    # Derived volume
    volume_path: Optional[str] = Field(
        None, description="Absolute path to de-identified NIfTI volume (.nii.gz)"
    )
    spacing_mm: Optional[List[MillimeterValue]] = Field(
        None, min_length=3, max_length=3,
        description="[x, y, z] voxel spacing in mm derived from best series"
    )
    volume_shape: Optional[List[int]] = Field(
        None, min_length=3, max_length=3,
        description="[Z, Y, X] volume dimensions in voxels"
    )

    # Series
    series: List[CTSeriesContract] = Field(default_factory=list)

    # Quality
    quality_grade: Optional[CTQualityGrade] = Field(
        None, description="Automated quality grade (A–D)"
    )
    quality_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Continuous quality score in [0, 1]"
    )
    quality_flags: List[str] = Field(
        default_factory=list,
        description=(
            "Quality issue codes: motion_mild, motion_severe, metal_artefact, "
            "thick_slices, insufficient_coverage, truncation"
        ),
    )

    # De-identification
    is_deidentified: bool = Field(
        True, description="Asserts all PHI has been removed (required before storage)"
    )

    # Arbitrary extra metadata from DICOM header
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Non-PHI supplementary DICOM metadata (e.g. CTDIvol, DLP)"
    )

    # Timestamps
    ingested_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("quality_flags")
    @classmethod
    def validate_quality_flags(cls, v: List[str]) -> List[str]:
        allowed = {
            "motion_mild", "motion_severe", "metal_artefact",
            "thick_slices", "insufficient_coverage", "truncation",
            "cone_beam_quality", "reconstruction_artefact",
        }
        unknown = set(v) - allowed
        if unknown:
            raise ValueError(f"Unknown quality_flags: {unknown}. Allowed: {allowed}")
        return v

    @model_validator(mode="after")
    def check_deidentified(self) -> "CTStudyContract":
        if not self.is_deidentified:
            raise ValueError(
                "CTStudyContract.is_deidentified must be True. "
                "Run the de-identification pipeline before constructing this contract."
            )
        return self

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def best_series(self) -> Optional[CTSeriesContract]:
        """Return the series with the thinnest slice thickness (best for planning)."""
        ct_series = [s for s in self.series if s.modality == "CT"]
        if not ct_series:
            return None
        return min(
            ct_series,
            key=lambda s: s.slice_thickness_mm if s.slice_thickness_mm else 999.0,
        )

    @property
    def voxel_volume_mm3(self) -> Optional[float]:
        """Volume of a single voxel in mm³."""
        if self.spacing_mm and len(self.spacing_mm) == 3:
            return self.spacing_mm[0] * self.spacing_mm[1] * self.spacing_mm[2]
        return None

    # ------------------------------------------------------------------
    # Clinical suitability
    # ------------------------------------------------------------------

    def is_suitable_for_planning(self) -> bool:
        """
        Return True if this CT study meets the minimum requirements for
        automated craniofacial surgical planning.

        Requirements:
          * ``is_deidentified`` must be True (always enforced by validator)
          * Modality is CT
          * Best series slice_thickness_mm ≤ 1.25 mm
          * quality_grade is A or B (or unset)
          * No severe quality flags (``motion_severe``, ``insufficient_coverage``)

        Use the more detailed ``suitability_report()`` for user-facing messages.
        """
        if self.modality not in (CTModality.CT, "CT"):
            return False

        severe_flags = {"motion_severe", "insufficient_coverage"}
        if severe_flags & set(self.quality_flags):
            return False

        if self.quality_grade in (CTQualityGrade.C, CTQualityGrade.D):
            return False

        bs = self.best_series
        if bs and bs.slice_thickness_mm and bs.slice_thickness_mm > 1.25:
            return False

        return True

    def suitability_report(self) -> Dict[str, Any]:
        """Return a structured dict explaining why the study is or is not suitable."""
        issues = []
        if self.modality not in (CTModality.CT, "CT"):
            issues.append(f"Modality {self.modality} is not CT")
        if "motion_severe" in self.quality_flags:
            issues.append("Severe motion artefact detected")
        if "insufficient_coverage" in self.quality_flags:
            issues.append("Insufficient anatomical coverage (does not include condyles/skull base)")
        if self.quality_grade in (CTQualityGrade.C, CTQualityGrade.D):
            issues.append(f"Quality grade {self.quality_grade} is below minimum (A or B required)")
        bs = self.best_series
        if bs and bs.slice_thickness_mm and bs.slice_thickness_mm > 1.25:
            issues.append(
                f"Slice thickness {bs.slice_thickness_mm} mm exceeds 1.25 mm maximum"
            )
        return {
            "suitable": len(issues) == 0,
            "issues": issues,
            "quality_grade": self.quality_grade,
            "best_slice_thickness_mm": bs.slice_thickness_mm if bs else None,
        }

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CTStudyContract":
        return cls.model_validate(data)

    def to_json_schema(self) -> str:
        return json.dumps(self.model_json_schema(), indent=2)
