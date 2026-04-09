"""
Data contract for intraoral scan (IOS) data.

Intraoral scanners (IOS) produce high-accuracy 3D digital impressions of the
dental arches.  In the CMF surgical planning workflow, IOS data serves three
roles:

1. **Pre-injury occlusal reference** — captured before injury (or reconstructed
   from dental records) to define the target occlusion.
2. **Intraoperative verification** — photogrammetric scan taken in the operating
   room to confirm reduction before rigid fixation.
3. **Post-operative outcome** — 6-week follow-up scan to measure occlusal outcome.

Scan quality metrics
--------------------
IOS quality is assessed along four axes:
  * ``point_density_per_mm2`` — surface point density (ideal ≥ 50 pts/mm²)
  * ``noise_level_um`` — RMS surface noise in micrometres (ideal ≤ 20 μm)
  * ``coverage_completeness`` — fraction of expected arch area captured [0, 1]
  * ``occlusal_registration_quality`` — quality of bite registration between arches

A ``is_suitable_for_registration()`` method gates use in registration workflows.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
MillimeterValue = float


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DentalArch(str, Enum):
    UPPER = "upper"
    LOWER = "lower"
    BOTH = "both"


class ScanPurpose(str, Enum):
    PRE_INJURY_REFERENCE = "pre_injury_reference"
    TREATMENT_PLANNING = "treatment_planning"
    INTRAOPERATIVE = "intraoperative"
    POST_OPERATIVE = "post_operative"
    SPLINT_FABRICATION = "splint_fabrication"


class ScanFileFormat(str, Enum):
    STL = "stl"
    PLY = "ply"
    OBJ = "obj"
    THREE_SHAPE = "3shape"          # .dcm (3Shape proprietary)
    ITERO = "itero"                 # .itr
    CEREC = "cerec"                 # .ssi / .stl
    TRIOS = "trios"                 # .3oxz


class ScannerManufacturer(str, Enum):
    THREE_SHAPE = "3Shape"
    ITERO = "iTero"
    CEREC = "Cerec"
    TRIOS = "Trios"
    MEDIT = "Medit"
    CARESTREAM = "Carestream"
    OTHER = "Other"


class OcclusalSurfaceStatus(str, Enum):
    """Status of the automated occlusal surface extraction."""
    PENDING = "pending"
    COMPLETE = "complete"
    FAILED = "failed"
    PARTIAL = "partial"   # Extracted but with gaps in posterior region


# ---------------------------------------------------------------------------
# Scan quality metrics sub-model
# ---------------------------------------------------------------------------

class ScanQualityMetrics(BaseModel):
    """
    Quantitative quality metrics for a single IOS scan.

    These are computed during post-processing and used to gate registration.
    """
    model_config = ConfigDict(populate_by_name=True)

    point_density_per_mm2: Optional[float] = Field(
        None, ge=0,
        description="Point density of the scan mesh in points/mm² (ideal ≥50)"
    )
    noise_level_um: Optional[float] = Field(
        None, ge=0,
        description="RMS surface noise in micrometres (ideal ≤20 μm)"
    )
    coverage_completeness: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Fraction of expected arch surface area captured"
    )
    coverage_gaps: List[str] = Field(
        default_factory=list,
        description="Regions with incomplete coverage (e.g. 'posterior_left', 'palate')"
    )
    occlusal_registration_quality: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Quality score for bite-registration between upper and lower arches"
    )
    mean_edge_length_mm: Optional[float] = Field(
        None, ge=0, description="Mean triangle edge length of the mesh in mm"
    )
    mesh_resolution_um: Optional[float] = Field(
        None, ge=0,
        description="Effective mesh resolution in micrometres (typical IOS: 20–50 μm)"
    )
    scan_duration_seconds: Optional[float] = Field(
        None, ge=0, description="Time to acquire the scan in seconds"
    )
    artefact_score: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Fraction of mesh affected by motion or beam-hardening artefacts"
    )

    @property
    def is_high_resolution(self) -> bool:
        """Return True if mesh_resolution_um ≤ 30 μm (sub-30-micron quality)."""
        return self.mesh_resolution_um is not None and self.mesh_resolution_um <= 30.0


# ---------------------------------------------------------------------------
# Top-level intraoral scan contract
# ---------------------------------------------------------------------------

class IntraoralScanContract(BaseModel):
    """
    Intraoral scan (digital dental impression) data contract.

    Stores metadata, file references, quality metrics, and derived products
    (occlusal surface extraction, tooth segmentation labels) for a single IOS
    capture session.

    File references
    ---------------
    Meshes are typically in STL format.  ``upper_arch_path`` and
    ``lower_arch_path`` are required before the scan can be registered to CT.
    ``bite_registration_path`` is needed for spatial alignment of arches.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [{
                "scan_id": "ios-001",
                "patient_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "arch": "both",
                "purpose": "treatment_planning",
                "scanner_manufacturer": "3Shape",
                "scanner_model": "TRIOS 5",
                "file_format": "stl",
                "upper_arch_path": "/data/ios/001/upper.stl",
                "lower_arch_path": "/data/ios/001/lower.stl",
            }]
        },
    )

    # Identifiers
    scan_id: str = Field(..., description="Unique IOS scan ID")
    patient_id: str = Field(..., description="Owning patient UUID")
    case_id: Optional[str] = Field(None, description="Associated surgical case ID")

    # Session metadata
    scan_date: Optional[datetime] = Field(None, description="Date/time of scan acquisition")
    purpose: ScanPurpose = Field(
        default=ScanPurpose.TREATMENT_PLANNING,
        description="Clinical purpose of this scan"
    )
    arch: DentalArch = Field(
        ..., description="Which dental arch(es) are included"
    )
    operator_id: Optional[str] = Field(None, description="ID of scanning technician")

    # Scanner provenance
    scanner_manufacturer: Optional[ScannerManufacturer] = None
    scanner_model: Optional[str] = None
    scanner_software_version: Optional[str] = None
    file_format: ScanFileFormat = Field(default=ScanFileFormat.STL)

    # File paths
    upper_arch_path: Optional[str] = Field(None, description="Path to upper arch mesh file")
    lower_arch_path: Optional[str] = Field(None, description="Path to lower arch mesh file")
    bite_registration_path: Optional[str] = Field(
        None, description="Path to bite registration mesh (spatial constraint for arch alignment)"
    )
    combined_arch_path: Optional[str] = Field(
        None, description="Pre-aligned combined arch mesh (arches in occlusal position)"
    )

    # Mesh metrics
    upper_vertex_count: Optional[int] = Field(None, ge=0)
    lower_vertex_count: Optional[int] = Field(None, ge=0)
    upper_face_count: Optional[int] = Field(None, ge=0)
    lower_face_count: Optional[int] = Field(None, ge=0)

    # Quality metrics
    quality: Optional[ScanQualityMetrics] = None

    # Tooth labelling
    tooth_labels: Dict[str, int] = Field(
        default_factory=dict,
        description="FDI tooth number string → mesh region label integer"
    )
    tooth_segmentation_complete: bool = Field(
        default=False,
        description="Whether automated tooth segmentation has been completed"
    )

    # Occlusal surface extraction
    occlusal_surface_status: OcclusalSurfaceStatus = Field(
        default=OcclusalSurfaceStatus.PENDING,
        description="Status of automated occlusal surface extraction"
    )
    occlusal_surface_path: Optional[str] = Field(
        None, description="Path to extracted occlusal surface mesh"
    )

    # Clinical flags
    has_restorations: bool = Field(
        default=False,
        description="Patient has crowns/bridges/implants that affect scan interpretation"
    )
    has_missing_teeth: bool = Field(
        default=False,
        description="Edentulous regions present — affects contact point prediction"
    )
    is_pre_injury: bool = Field(
        default=False,
        description="True if this scan was acquired before the injury (occlusal reference)"
    )
    registration_target_id: Optional[str] = Field(
        None, description="ID of the CT study or segmentation this scan is registered to"
    )

    notes: Optional[str] = Field(None, max_length=1000)
    ingested_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("arch")
    @classmethod
    def validate_arch(cls, v: DentalArch) -> DentalArch:
        return v

    @model_validator(mode="after")
    def validate_paths_consistent_with_arch(self) -> "IntraoralScanContract":
        if self.arch in (DentalArch.UPPER, DentalArch.BOTH) and self.upper_arch_path is None:
            # Don't raise — path may be filled in later by ingestion
            pass
        if self.arch in (DentalArch.LOWER, DentalArch.BOTH) and self.lower_arch_path is None:
            pass
        return self

    # ------------------------------------------------------------------
    # Clinical suitability
    # ------------------------------------------------------------------

    def is_suitable_for_registration(self) -> bool:
        """
        Return True if the scan meets minimum requirements for CT-IOS co-registration.

        Requirements:
          * At least one arch mesh path must be populated
          * If arch == "both", bite_registration_path should be present
          * Coverage completeness (if measured) ≥ 0.85
          * Occlusal surface extraction must be COMPLETE or PENDING (not FAILED)
        """
        if self.arch in (DentalArch.UPPER, DentalArch.BOTH) and self.upper_arch_path is None:
            return False
        if self.arch in (DentalArch.LOWER, DentalArch.BOTH) and self.lower_arch_path is None:
            return False
        if self.quality and self.quality.coverage_completeness is not None:
            if self.quality.coverage_completeness < 0.85:
                return False
        if self.occlusal_surface_status == OcclusalSurfaceStatus.FAILED:
            return False
        return True

    @property
    def total_vertex_count(self) -> int:
        """Sum of upper and lower arch vertex counts."""
        return (self.upper_vertex_count or 0) + (self.lower_vertex_count or 0)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntraoralScanContract":
        return cls.model_validate(data)
