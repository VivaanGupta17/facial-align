"""
Data contract for the complete surgical fracture reduction plan.

A ``ReductionPlanContract`` is the primary output of the reduction-planning
pipeline and the primary input to the surgical review workflow.  It aggregates:

- Per-fragment transforms (from ``FractureFragmentContract``)
- Occlusal metrics measured at the planned reduction
- Automated validation results
- Hardware inventory list
- Sequence order for surgical reduction
- Quality metrics (symmetry score, confidence)

Versioning
----------
Each case may have multiple plan versions.  ``plan_version`` increments when
a new plan is generated (either by the ML model or by surgeon edit + re-run).
The ``parent_plan_id`` links derived plans back to their origin for audit.

Quality metrics
---------------
``overall_confidence`` is the harmonic mean of per-fragment confidences,
weighted by fragment volume.  ``symmetry_score`` is the normalised
midsagittal plane deviation of the planned facial skeleton.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from data_contracts.fracture_fragment import FractureFragmentContract

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
MillimeterValue = float


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class PlanStatus(str, Enum):
    """Lifecycle status of a reduction plan."""
    GENERATING = "generating"
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class PlanOrigin(str, Enum):
    """How the plan was created."""
    ML_GENERATED = "ml_generated"
    SURGEON_EDIT = "surgeon_edit"        # Derived from ML plan by surgeon edits
    MANUAL = "manual"                    # Entirely surgeon-specified
    TEMPLATE = "template"               # Instantiated from a plan template


# ---------------------------------------------------------------------------
# Occlusal metrics sub-model
# ---------------------------------------------------------------------------

class OcclusalMetricsContract(BaseModel):
    """
    Occlusal measurements at the planned reduction position.

    These are *predicted* values derived from the 3D CT data and planned
    fragment positions.  They should be verified against intraoral scan
    registration when available.

    Normal clinical ranges (adult):
      overjet_mm:           1–3 mm
      overbite_mm:          2–4 mm
      midline_deviation_mm: ≤1 mm
      cant_degrees:         ≤2°
    """
    overjet_mm: Optional[float] = Field(
        None, ge=-15.0, le=15.0,
        description="Horizontal incisal overjet in mm (negative = anterior cross-bite)"
    )
    overbite_mm: Optional[float] = Field(
        None, ge=-10.0, le=15.0,
        description="Vertical incisal overbite in mm (negative = open bite)"
    )
    molar_relationship: Optional[str] = Field(
        None, description="Angle molar classification: Class_I, Class_II_div1, Class_II_div2, Class_III"
    )
    midline_deviation_mm: Optional[float] = Field(
        None, ge=0.0, le=20.0,
        description="Absolute deviation of dental midlines in mm"
    )
    cant_degrees: Optional[float] = Field(
        None, ge=0.0, le=30.0,
        description="Occlusal plane cant deviation from horizontal in degrees"
    )
    curve_of_spee_mm: Optional[float] = Field(
        None, ge=0.0, le=10.0,
        description="Depth of the curve of Spee in mm (normal: ≤2 mm)"
    )
    posterior_open_bite_mm: Optional[float] = Field(
        None, ge=0.0, description="Posterior open bite in mm (0 if in contact)"
    )
    anterior_open_bite_mm: Optional[float] = Field(
        None, ge=0.0, description="Anterior open bite in mm (0 if in contact)"
    )
    contact_points: Optional[int] = Field(
        None, ge=0, description="Estimated number of occlusal contact points (normal: ≥12)"
    )
    constraints_satisfied: bool = Field(
        False, description="Whether all pre-specified occlusal constraints are met"
    )
    constraint_violations: List[str] = Field(
        default_factory=list,
        description="Human-readable descriptions of any violated constraints"
    )

    @property
    def is_class_i_occlusion(self) -> bool:
        return (
            self.molar_relationship == "Class_I"
            and (self.overjet_mm or 0.0) >= 0
            and (self.overbite_mm or 0.0) >= 0
        )


# ---------------------------------------------------------------------------
# Validation sub-model
# ---------------------------------------------------------------------------

class ValidationContract(BaseModel):
    """
    Automated geometric and clinical validation result for a reduction plan.

    All boolean fields default to False (failed) to require explicit passes.
    ``passed`` is True only when all critical checks pass.
    """
    passed: bool = Field(..., description="Overall pass/fail — True only if ALL critical checks pass")
    symmetry_ok: bool = Field(..., description="Facial skeletal symmetry within tolerance")
    occlusion_ok: bool = Field(..., description="Occlusal constraints satisfied")
    condylar_seating_ok: bool = Field(
        ..., description="Bilateral condylar heads are seated correctly in fossae"
    )
    hardware_placement_ok: bool = Field(
        ..., description="Planned hardware positions do not conflict with anatomy"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Non-critical issues that should be reviewed"
    )
    errors: List[str] = Field(
        default_factory=list, description="Critical issues preventing approval"
    )
    skeletal_symmetry_score: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Normalised facial skeleton symmetry score (1.0 = perfect)"
    )
    condylar_seating_score_L: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Left condylar seating quality score"
    )
    condylar_seating_score_R: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Right condylar seating quality score"
    )

    @model_validator(mode="after")
    def sync_passed(self) -> "ValidationContract":
        """Ensure ``passed`` is False whenever there are errors."""
        if self.errors:
            self.passed = False
        return self


# ---------------------------------------------------------------------------
# Hardware item sub-model
# ---------------------------------------------------------------------------

class HardwareItem(BaseModel):
    """A single piece of fixation hardware in the surgical plan."""
    hardware_id: str = Field(..., description="Unique item identifier")
    hardware_type: str = Field(..., description="Type: miniplate, screw, wire, mesh, etc.")
    size_mm: Optional[str] = Field(
        None, description="Nominal size descriptor (e.g. '2.0mm × 6-hole')"
    )
    material: str = Field(
        default="titanium", description="Material: titanium, resorbable, stainless_steel"
    )
    quantity: int = Field(default=1, ge=1)
    intended_fragment: Optional[str] = Field(
        None, description="Fragment ID this hardware is intended for"
    )
    position_description: Optional[str] = Field(
        None, max_length=200, description="Anatomical placement description"
    )
    manufacturer: Optional[str] = None
    catalogue_number: Optional[str] = None


# ---------------------------------------------------------------------------
# Top-level reduction plan contract
# ---------------------------------------------------------------------------

class ReductionPlanContract(BaseModel):
    """
    Canonical reduction plan contract — the primary surgical planning artefact.

    Lifecycle:
        GENERATING → DRAFT → UNDER_REVIEW → APPROVED | REJECTED

    A plan is immutable once approved.  Any subsequent modifications create a
    new version (``plan_version`` increments, ``parent_plan_id`` is set).
    """
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [{
                "plan_id": "plan-abc123",
                "case_id": "case-xyz456",
                "plan_version": 1,
                "status": "draft",
                "model_name": "baseline_icp",
                "model_version": "1.3.0",
                "overall_confidence": 0.87,
                "surgeon_approved": False,
            }]
        },
    )

    # Identifiers
    plan_id: str = Field(..., description="Unique plan ID (UUID)")
    case_id: str = Field(..., description="Owning surgical case ID")
    parent_plan_id: Optional[str] = Field(
        None, description="ID of the plan this was derived from (for edits/re-runs)"
    )
    plan_version: int = Field(..., ge=1, description="Monotonically incrementing version number")

    # Provenance
    status: PlanStatus = Field(default=PlanStatus.DRAFT)
    origin: PlanOrigin = Field(default=PlanOrigin.ML_GENERATED)
    model_name: str = Field(..., description="Model or algorithm used")
    model_version: str = Field(..., description="Model version string")

    # Fragment data
    fragments: List[FractureFragmentContract] = Field(
        ..., min_length=1, description="All fracture fragments with geometry and planned transforms"
    )

    # Surgical sequence
    surgical_sequence: List[str] = Field(
        default_factory=list,
        description="Ordered list of fragment_ids in recommended surgical reduction sequence"
    )

    # Occlusal metrics
    occlusal_metrics: Optional[OcclusalMetricsContract] = None

    # Validation
    validation: Optional[ValidationContract] = None

    # Quality metrics
    overall_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Volume-weighted harmonic mean of per-fragment confidences"
    )
    symmetry_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Post-reduction facial skeletal symmetry score"
    )

    # Hardware
    hardware_list: List[HardwareItem] = Field(
        default_factory=list,
        description="Complete hardware inventory for this plan"
    )

    # Approval
    surgeon_approved: bool = Field(default=False)
    surgeon_id: Optional[str] = None
    surgeon_notes: Optional[str] = Field(None, max_length=2000)
    approved_at: Optional[datetime] = None

    # Performance
    is_ml_generated: bool = Field(default=True)
    generation_time_ms: int = Field(default=0, ge=0)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("overall_confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if v < 0.5:
            # Plans below 50% confidence should not be auto-approved
            pass  # A validator warning would be raised separately
        return v

    @model_validator(mode="after")
    def validate_approval_consistency(self) -> "ReductionPlanContract":
        if self.surgeon_approved and self.approved_at is None:
            raise ValueError("approved_at must be set when surgeon_approved is True")
        if self.surgeon_approved and self.status not in (PlanStatus.APPROVED, "approved"):
            raise ValueError(
                "status must be PlanStatus.APPROVED when surgeon_approved is True"
            )
        return self

    @model_validator(mode="after")
    def auto_populate_surgical_sequence(self) -> "ReductionPlanContract":
        """Fill surgical_sequence from fragment data if not explicitly set."""
        if not self.surgical_sequence:
            seq = sorted(
                [f for f in self.fragments if f.surgical_sequence is not None],
                key=lambda f: f.surgical_sequence,  # type: ignore[arg-type]
            )
            self.surgical_sequence = [f.geometry.fragment_id for f in seq]
        return self

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def reference_fragment(self) -> Optional[FractureFragmentContract]:
        """Return the reference (anatomical anchor) fragment."""
        for f in self.fragments:
            if f.geometry.is_reference:
                return f
        return None

    @property
    def fragment_count(self) -> int:
        return len(self.fragments)

    @property
    def is_approvable(self) -> bool:
        """
        Return True if this plan meets the minimum criteria for surgeon approval.

        A plan is approvable if:
        - Validation passed (or validation was skipped / not yet run)
        - Confidence ≥ 0.7
        - No error messages in validation
        """
        if self.validation and not self.validation.passed:
            return False
        return self.overall_confidence >= 0.70

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReductionPlanContract":
        return cls.model_validate(data)

    def fragment_by_id(self, fragment_id: str) -> Optional[FractureFragmentContract]:
        """Lookup a fragment by its ID."""
        for f in self.fragments:
            if f.geometry.fragment_id == fragment_id:
                return f
        return None
