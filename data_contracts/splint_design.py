"""
Data contract for occlusal splint design request and specification.

Occlusal splints in jaw fracture surgery serve three distinct purposes:

1. **Intermediate splint** (surgical splint) — used intraoperatively to
   establish and maintain the planned occlusion while the surgeon applies
   plate-and-screw fixation.  Usually acrylic; worn only in the OR.

2. **Final splint** (post-operative) — worn by the patient after surgery
   to guide the healing occlusion.  May be modified by the orthodontist.

3. **Orthopedic / repositioning splint** — used in TMJ management to
   reposition the condyle before definitive fixation.

Manufacturing pipeline
----------------------
Splint designs are exported as STL/3MF/STEP files for:
- Direct CNC milling (CEREC, VHF, Roland)
- 3D printing (SLA resin, for surgical guides)
- Traditional dental lab fabrication (when digital workflow is unavailable)

The ``cad_cam_compatible`` flag indicates whether the geometry is clean
enough for direct digital manufacturing without manual intervention.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
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

class SplintType(str, Enum):
    """
    Classification of the occlusal splint by surgical role.

    INTERMEDIATE — Intraoperative; positions mandible to planned occlusion.
    FINAL — Post-operative; maintains occlusion during healing.
    REPOSITIONING — Pre-operative; repositions condyle (TMJ splint).
    DEPROGRAMMING — Occlusal equilibration; removes muscle memory.
    NIGHTGUARD — Bruxism protection; not surgical.
    """
    INTERMEDIATE = "intermediate"
    FINAL = "final"
    REPOSITIONING = "repositioning"
    DEPROGRAMMING = "deprogramming"
    NIGHTGUARD = "nightguard"


class SplintMaterial(str, Enum):
    """Splint fabrication material."""
    ACRYLIC_RESIN = "acrylic_resin"           # Classic heat-cured PMMA
    ACRYLIC_RESIN_COLD_CURE = "acrylic_resin_cold_cure"  # Chairside acrylic
    SOFT_SILICONE = "soft_silicone"           # Flexible; post-op comfort
    RIGID_RESIN_SLA = "rigid_resin_sla"       # 3D-printed SLA; surgical
    NYLON_SLS = "nylon_sls"                   # SLS printed; high strength
    PEEK = "peek"                             # PEEK; sterilisable for OR use
    POLYCARBONATE = "polycarbonate"


class RetentionMechanism(str, Enum):
    """How the splint is retained on the teeth."""
    CLASPS = "clasps"                         # Wire/metal clasps
    VACUUM_FORMED = "vacuum_formed"           # Thermoplastic vacuum-form
    BONDED = "bonded"                         # Direct bonded with composite
    CIRCUMFERENTIAL_WIRE = "circumferential_wire"   # Tie wire (intraoperative)
    FRICTION_FIT = "friction_fit"             # Tight fit; no active retention


class ManufacturingStatus(str, Enum):
    """Production status of the splint."""
    DESIGN_PENDING = "design_pending"
    DESIGN_COMPLETE = "design_complete"
    SENT_TO_LAB = "sent_to_lab"
    MANUFACTURING = "manufacturing"
    QUALITY_CHECK = "quality_check"
    READY = "ready"
    DELIVERED = "delivered"
    REJECTED = "rejected"


class OutputFormat(str, Enum):
    """CAD/CAM output file format."""
    STL = "stl"
    THREE_MF = "3mf"
    STEP = "step"
    OBJ = "obj"


# ---------------------------------------------------------------------------
# Thickness parameters sub-model
# ---------------------------------------------------------------------------

class SplintThicknessParameters(BaseModel):
    """
    Thickness constraints for splint geometry generation.

    These parameters drive the CAD algorithm that extrudes the splint
    body from the occlusal surface mesh.  All values in mm.
    """
    min_thickness_mm: float = Field(
        default=2.0, ge=0.5, le=10.0,
        description="Minimum body thickness (structural integrity; typical: 2 mm)"
    )
    max_thickness_mm: float = Field(
        default=5.0, ge=1.0, le=15.0,
        description="Maximum body thickness (patient comfort; typical: 4–5 mm)"
    )
    bite_block_height_mm: float = Field(
        default=3.0, ge=0.0, le=10.0,
        description="Height of posterior bite blocks above occlusal surface in mm"
    )
    incisal_ramp_angle_degrees: float = Field(
        default=5.0, ge=0.0, le=30.0,
        description="Angle of anterior guidance ramp in degrees"
    )
    labial_flange_height_mm: float = Field(
        default=3.5, ge=0.0, le=8.0,
        description="Height of the labial flange for retention in mm"
    )

    @model_validator(mode="after")
    def validate_thickness_range(self) -> "SplintThicknessParameters":
        if self.min_thickness_mm >= self.max_thickness_mm:
            raise ValueError("min_thickness_mm must be less than max_thickness_mm")
        return self


# ---------------------------------------------------------------------------
# Retention features sub-model
# ---------------------------------------------------------------------------

class RetentionFeatures(BaseModel):
    """
    Retention features integrated into the splint design.

    Clasp positions are specified as FDI tooth numbers.
    """
    mechanism: RetentionMechanism = Field(
        default=RetentionMechanism.CLASPS,
        description="Primary retention mechanism"
    )
    clasp_positions: List[int] = Field(
        default_factory=list,
        description="FDI tooth numbers where clasps are placed"
    )
    undercut_engagement_depth_mm: float = Field(
        default=0.25, ge=0.0, le=1.0,
        description="Planned undercut engagement depth per clasp in mm"
    )
    uses_lingual_flanges: bool = Field(
        default=True, description="Whether lingual flanges are included"
    )
    uses_palatal_coverage: bool = Field(
        default=False,
        description="Whether palatal coverage is included (full vs. sectional)"
    )

    @field_validator("clasp_positions")
    @classmethod
    def validate_clasp_fdi(cls, v: List[int]) -> List[int]:
        valid_fdi = set(range(11, 29)) | set(range(31, 49))
        invalid = [t for t in v if t not in valid_fdi]
        if invalid:
            raise ValueError(f"Invalid FDI numbers in clasp_positions: {invalid}")
        return v


# ---------------------------------------------------------------------------
# Design request contract
# ---------------------------------------------------------------------------

class SplintDesignRequest(BaseModel):
    """
    Request to generate a new splint design from plan data.

    Submitted to the splint design service once a reduction plan is approved.
    The service uses the IOS scan and planned occlusion to compute the
    inter-occlusal space and generate the splint geometry.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [{
                "case_id": "case-xyz456",
                "plan_id": "plan-abc123",
                "splint_type": "intermediate",
                "arch": "both",
                "material": "rigid_resin_sla",
                "target_vd_mm": 2.5,
                "include_bite_blocks": True,
                "retention_type": "clasps",
                "output_format": "stl",
            }]
        },
    )

    case_id: str
    plan_id: str
    ios_scan_id: Optional[str] = Field(
        None, description="IOS scan to base the splint on (latest if None)"
    )
    splint_type: SplintType = Field(
        default=SplintType.INTERMEDIATE,
        description="Type of splint to generate"
    )
    target_vd_mm: Optional[MillimeterValue] = Field(
        None, ge=0.0, le=20.0,
        description="Target increase in vertical dimension of occlusion in mm"
    )
    arch: str = Field(
        default="both",
        description="Which arch is the splint body on: 'upper', 'lower', 'both'"
    )
    material: SplintMaterial = Field(
        default=SplintMaterial.ACRYLIC_RESIN,
        description="Fabrication material"
    )
    include_bite_blocks: bool = Field(
        default=True, description="Include posterior bite blocks"
    )
    retention_type: RetentionMechanism = Field(
        default=RetentionMechanism.CLASPS,
        description="Primary retention mechanism"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.STL, description="CAD/CAM output file format"
    )
    thickness_params: Optional[SplintThicknessParameters] = None

    @field_validator("arch")
    @classmethod
    def validate_arch(cls, v: str) -> str:
        if v not in {"upper", "lower", "both"}:
            raise ValueError(f"arch must be 'upper', 'lower', or 'both', got {v!r}")
        return v


# ---------------------------------------------------------------------------
# Design specification contract
# ---------------------------------------------------------------------------

class SplintDesignSpec(BaseModel):
    """
    Splint design output specification returned by the splint design service.

    Consumed by: manufacturing workflow, surgeon review UI, case archive.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [{
                "design_id": "splint-001",
                "case_id": "case-xyz456",
                "plan_id": "plan-abc123",
                "splint_type": "intermediate",
                "material": "rigid_resin_sla",
                "target_vertical_dimension_mm": 2.5,
                "cad_cam_compatible": True,
                "manufacturing_status": "design_complete",
            }]
        },
    )

    # Identifiers
    design_id: str = Field(..., description="Unique splint design ID")
    case_id: str
    plan_id: str
    request_id: Optional[str] = Field(None, description="Source SplintDesignRequest ID")

    # Classification
    splint_type: SplintType
    material: SplintMaterial
    output_format: OutputFormat = OutputFormat.STL

    # Geometry artefacts
    upper_component_path: Optional[str] = Field(
        None, description="Path to upper arch component mesh"
    )
    lower_component_path: Optional[str] = Field(
        None, description="Path to lower arch component mesh"
    )
    combined_path: Optional[str] = Field(
        None, description="Path to combined upper+lower assembly mesh"
    )

    # Design parameters
    target_vertical_dimension_mm: float = Field(
        ..., ge=0.0, le=20.0,
        description="Target vertical dimension increase in mm"
    )
    thickness_parameters: Optional[SplintThicknessParameters] = None
    retention_features: Optional[RetentionFeatures] = None
    estimated_thickness_map: Dict[str, float] = Field(
        default_factory=dict,
        description="Region → estimated thickness in mm (for quality review)"
    )
    contact_regions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Predicted occlusal contact regions on the splint surface"
    )

    # Manufacturing
    fabrication_notes: str = Field(
        default="",
        description="Instructions for the dental lab or CAD/CAM operator"
    )
    cad_cam_compatible: bool = Field(
        default=True,
        description="Whether the geometry is clean for direct digital manufacturing"
    )
    requires_manual_finish: bool = Field(
        default=False,
        description="Whether manual polishing or chairside adjustment is required"
    )
    manufacturing_status: ManufacturingStatus = Field(
        default=ManufacturingStatus.DESIGN_COMPLETE
    )
    estimated_fabrication_time_hours: Optional[float] = Field(
        None, ge=0, description="Estimated time to fabricate in hours"
    )

    # Timestamps
    designed_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    delivered_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("target_vertical_dimension_mm")
    @classmethod
    def validate_vd(cls, v: float) -> float:
        if v > 10.0:
            # Unusual but not necessarily invalid; flag for review
            pass
        return v

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def is_ready_for_or(self) -> bool:
        """True if the splint is manufactured and delivered."""
        return self.manufacturing_status == ManufacturingStatus.DELIVERED

    @property
    def has_geometry(self) -> bool:
        """True if at least one mesh file is available."""
        return any([
            self.upper_component_path,
            self.lower_component_path,
            self.combined_path,
        ])

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SplintDesignSpec":
        return cls.model_validate(data)
