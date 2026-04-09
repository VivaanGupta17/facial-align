"""
Data contract for occlusion analysis plan.

The ``OcclusionPlanContract`` captures the complete dental and skeletal
occlusal state targeted by the surgical plan.  It integrates:

- All standard cephalometric measurements (Steiner, Ricketts, and Jarabak analyses)
- Dental constraint set (overjet, overbite, molar class, midline, cant)
- Intraoral contact map
- Splint design requirements
- A computed occlusion grade

Clinical context
----------------
Achieving correct dental occlusion is the primary driver of jaw fracture
reduction outcomes.  This contract serves as both a surgical target
specification and a post-operative outcome record.

Cephalometric measurement conventions
--------------------------------------
All angular measurements use the standard cephalometric landmarks:
  S = Sella, N = Nasion, A = Point A (subspinale), B = Point B (supramentale),
  Gn = Gnathion, Me = Menton, Go = Gonion, MP = Mandibular Plane,
  Po = Porion, Or = Orbitale (defines Frankfort Horizontal).

Ranges from Steiner (1953), Jarabak (1972), Ricketts (1981), and
McNamara (1984) norms for adult Caucasian patients.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
MillimeterValue = float
DegreeValue = float


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AngleMolarClass(str, Enum):
    """Angle molar classification (Angle 1899)."""
    CLASS_I = "Class_I"
    CLASS_II_DIV1 = "Class_II_div1"
    CLASS_II_DIV2 = "Class_II_div2"
    CLASS_III = "Class_III"
    UNCLASSIFIABLE = "unclassifiable"


class OcclusionGrade(str, Enum):
    """
    Clinical grade assigned to the planned occlusion quality.

    A — Ideal: All metrics within normal limits, Class I molar, balanced contacts.
    B — Acceptable: Minor deviations within clinical tolerance.
    C — Marginal: Deviations require splint or further adjustment.
    D — Unacceptable: Significant open bite, severe Class III/II, requires revision.
    """
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class SkeletalPattern(str, Enum):
    """Skeletal pattern classification for treatment planning."""
    CLASS_I = "class_i"
    CLASS_II = "class_ii"
    CLASS_III = "class_iii"
    VERTICAL_EXCESS = "vertical_excess"       # Open bite tendency
    VERTICAL_DEFICIENCY = "vertical_deficiency"  # Deep bite tendency
    ASYMMETRIC = "asymmetric"


# ---------------------------------------------------------------------------
# Tooth contact sub-model
# ---------------------------------------------------------------------------

class ToothContactContract(BaseModel):
    """
    A single occlusal contact between an upper and lower tooth.

    FDI tooth numbering: upper right = 1x, upper left = 2x, lower left = 3x,
    lower right = 4x, where x is tooth number 1–8.
    """
    upper_fdi: int = Field(
        ..., ge=11, le=28,
        description="FDI number of upper tooth (11–28)"
    )
    lower_fdi: int = Field(
        ..., ge=31, le=48,
        description="FDI number of lower tooth (31–48)"
    )
    contact_area_mm2: Optional[float] = Field(
        None, ge=0, description="Contact area in mm²"
    )
    relative_force: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Relative occlusal force (0 = no contact, 1 = maximum)"
    )
    contact_type: Optional[str] = Field(
        None, description="Contact type: cusp_tip, cusp_fossa, marginal_ridge"
    )
    is_premature_contact: bool = Field(
        False, description="Whether this contact occurs before full occlusion is reached"
    )


# ---------------------------------------------------------------------------
# Cephalometric measurement sub-model
# ---------------------------------------------------------------------------

class CephalometricMeasurement(BaseModel):
    """
    A single cephalometric measurement with clinical interpretation.

    Normal ranges are based on adult Caucasian norms from major cephalometric
    analyses.  The ``within_normal_range`` flag is set automatically by the
    service layer.
    """
    name: str = Field(..., description="Standard measurement name (e.g. 'SNA', 'ANB')")
    value: float = Field(..., description="Measured value")
    unit: str = Field(..., description="Unit: 'degrees' or 'mm'")
    within_normal_range: bool = Field(..., description="Whether value is within 2 SD of the norm")
    normal_range: Optional[str] = Field(
        None, description="Normal range as human-readable string (e.g. '80–84°')"
    )
    mean_value: Optional[float] = Field(None, description="Population mean for this measurement")
    sd_value: Optional[float] = Field(None, description="Population standard deviation")
    clinical_significance: Optional[str] = Field(
        None, max_length=200,
        description="Brief clinical interpretation of an abnormal value"
    )


# ---------------------------------------------------------------------------
# Dental constraint sub-model
# ---------------------------------------------------------------------------

class DentalConstraintSet(BaseModel):
    """
    Complete set of dental occlusal constraints that the reduction plan must satisfy.

    These are the *targets* against which planned fragment positions are evaluated
    by the reduction optimiser and the occlusion validator.
    """
    target_overjet_mm: float = Field(
        default=2.0, ge=-5.0, le=10.0,
        description="Target horizontal incisal overjet in mm (normal: 1–3 mm)"
    )
    target_overbite_mm: float = Field(
        default=3.0, ge=-5.0, le=10.0,
        description="Target vertical incisal overbite in mm (normal: 2–4 mm)"
    )
    overjet_tolerance_mm: float = Field(
        default=1.0, ge=0, description="Acceptable deviation from target overjet"
    )
    overbite_tolerance_mm: float = Field(
        default=1.0, ge=0, description="Acceptable deviation from target overbite"
    )
    molar_class_target: AngleMolarClass = Field(
        default=AngleMolarClass.CLASS_I, description="Target Angle molar classification"
    )
    midline_tolerance_mm: float = Field(
        default=1.5, ge=0, description="Maximum acceptable dental midline deviation in mm"
    )
    cant_tolerance_degrees: float = Field(
        default=2.0, ge=0, description="Maximum acceptable occlusal cant in degrees"
    )
    use_pre_injury_occlusion: bool = Field(
        default=True, description="Optimise to restore pre-injury occlusal record if available"
    )
    intermaxillary_fixation: bool = Field(
        default=False, description="Whether IMF wires are planned during fixation"
    )
    condylar_seating: str = Field(
        default="centric_relation",
        description="Target condylar position: centric_relation, centric_occlusion, therapeutic"
    )

    @field_validator("condylar_seating")
    @classmethod
    def validate_condylar_seating(cls, v: str) -> str:
        valid = {"centric_relation", "centric_occlusion", "therapeutic"}
        if v not in valid:
            raise ValueError(f"condylar_seating must be one of {valid}, got {v!r}")
        return v


# ---------------------------------------------------------------------------
# Top-level occlusion plan contract
# ---------------------------------------------------------------------------

class OcclusionPlanContract(BaseModel):
    """
    Canonical occlusion plan contract.

    Aggregates the complete dental and skeletal occlusal specification for a
    surgical case: cephalometric measurements, constraint set, contact map,
    and computed grade.

    Produced by: ``services/occlusion/occlusion_service.py``
    Consumed by: reduction planner, case review, splint design, viewer.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [{
                "plan_id": "occ-plan-001",
                "case_id": "case-xyz456",
                "overjet_mm": 2.0,
                "overbite_mm": 3.0,
                "molar_relationship": "Class_I",
                "midline_deviation_mm": 0.5,
                "constraints_satisfied": True,
                "occlusion_grade": "A",
            }]
        },
    )

    # Identifiers
    plan_id: str = Field(..., description="Unique occlusion plan ID")
    case_id: str = Field(..., description="Owning surgical case ID")
    reduction_plan_id: Optional[str] = Field(
        None, description="Associated reduction plan ID"
    )

    # Primary occlusal measurements
    overjet_mm: Optional[MillimeterValue] = Field(
        None, ge=-15.0, le=15.0,
        description="Horizontal incisal overjet in mm (negative = anterior cross-bite)"
    )
    overbite_mm: Optional[MillimeterValue] = Field(
        None, ge=-10.0, le=15.0,
        description="Vertical incisal overbite in mm (negative = open bite)"
    )
    molar_relationship: Optional[AngleMolarClass] = Field(
        None, description="Measured Angle molar classification"
    )
    midline_deviation_mm: Optional[MillimeterValue] = Field(
        None, ge=0.0, le=20.0,
        description="Absolute deviation of upper-to-lower dental midlines in mm"
    )
    midline_deviation_direction: Optional[str] = Field(
        None, description="Direction of midline deviation: 'left' or 'right'"
    )
    cant_degrees: Optional[DegreeValue] = Field(
        None, ge=0.0, le=30.0,
        description="Occlusal plane cant deviation from horizontal in degrees"
    )
    curve_of_spee_mm: Optional[MillimeterValue] = Field(
        None, ge=0.0, le=10.0,
        description="Depth of curve of Spee in mm (normal ≤2 mm)"
    )
    posterior_open_bite_mm: Optional[MillimeterValue] = Field(
        None, ge=0.0, description="Posterior open bite in mm"
    )
    anterior_open_bite_mm: Optional[MillimeterValue] = Field(
        None, ge=0.0, description="Anterior open bite in mm"
    )

    # Cephalometric analysis
    skeletal_pattern: Optional[SkeletalPattern] = None
    anb_degrees: Optional[DegreeValue] = Field(
        None, ge=-20.0, le=20.0, description="ANB angle in degrees"
    )
    sna_degrees: Optional[DegreeValue] = Field(
        None, ge=60.0, le=100.0, description="SNA angle in degrees"
    )
    snb_degrees: Optional[DegreeValue] = Field(
        None, ge=60.0, le=100.0, description="SNB angle in degrees"
    )
    wits_appraisal_mm: Optional[MillimeterValue] = Field(
        None, ge=-15.0, le=15.0, description="Wits appraisal in mm (normal: −1 to +1 mm)"
    )
    gonial_angle_degrees: Optional[DegreeValue] = Field(
        None, ge=100.0, le=145.0, description="Gonial angle (Ar-Go-Me) in degrees"
    )
    lower_facial_height_degrees: Optional[DegreeValue] = Field(
        None, ge=30.0, le=65.0,
        description="ANS-Xi-Pm (Ricketts lower facial height) in degrees"
    )
    nasolabial_angle_degrees: Optional[DegreeValue] = Field(
        None, ge=80.0, le=130.0, description="Nasolabial angle in degrees"
    )
    upper_lip_to_e_plane_mm: Optional[MillimeterValue] = Field(
        None, ge=-10.0, le=10.0,
        description="Distance from upper lip to Ricketts E-plane in mm"
    )
    lower_lip_to_e_plane_mm: Optional[MillimeterValue] = Field(
        None, ge=-10.0, le=10.0,
        description="Distance from lower lip to Ricketts E-plane in mm"
    )
    cephalometric_measurements: List[CephalometricMeasurement] = Field(
        default_factory=list,
        description="Full list of individual cephalometric measurements with ranges"
    )

    # Dental constraints
    constraint_set: Optional[DentalConstraintSet] = Field(
        None, description="Constraint set used by the reduction optimiser"
    )
    contact_points: List[ToothContactContract] = Field(
        default_factory=list,
        description="Predicted occlusal contact map"
    )
    constraints_satisfied: bool = Field(
        default=False, description="Whether all constraint_set targets are met"
    )
    constraint_violations: List[str] = Field(
        default_factory=list,
        description="Human-readable descriptions of violated constraints"
    )

    # Splint
    splint_required: bool = Field(
        default=False, description="Whether an intermediate occlusal splint is required"
    )
    splint_vd_mm: Optional[MillimeterValue] = Field(
        None, ge=0.0, le=30.0,
        description="Target vertical dimension increase for splint in mm"
    )

    # Grade
    occlusion_grade: Optional[OcclusionGrade] = Field(
        None, description="Computed clinical grade for the planned occlusion"
    )

    notes: Optional[str] = Field(None, max_length=2000)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("anb_degrees")
    @classmethod
    def classify_skeletal_from_anb(cls, v: Optional[float]) -> Optional[float]:
        # No mutation here — skeletal_pattern is set via model_validator
        return v

    @model_validator(mode="after")
    def auto_classify_skeletal(self) -> "OcclusionPlanContract":
        """Derive skeletal_pattern from ANB angle if not explicitly set."""
        if self.skeletal_pattern is None and self.anb_degrees is not None:
            anb = self.anb_degrees
            if anb < 0:
                self.skeletal_pattern = SkeletalPattern.CLASS_III
            elif anb <= 4.0:
                self.skeletal_pattern = SkeletalPattern.CLASS_I
            else:
                self.skeletal_pattern = SkeletalPattern.CLASS_II
        return self

    @model_validator(mode="after")
    def auto_compute_grade(self) -> "OcclusionPlanContract":
        """Compute occlusion_grade if not set and enough data is present."""
        if self.occlusion_grade is not None:
            return self

        issues = 0
        critical = 0

        if self.molar_relationship not in (None, AngleMolarClass.CLASS_I, "Class_I"):
            issues += 1
        if self.overjet_mm is not None and (self.overjet_mm < 0 or self.overjet_mm > 5):
            critical += 1
        if self.midline_deviation_mm is not None and self.midline_deviation_mm > 3.0:
            issues += 1
        if self.cant_degrees is not None and self.cant_degrees > 4.0:
            issues += 1
        if self.anterior_open_bite_mm is not None and self.anterior_open_bite_mm > 1.0:
            critical += 1

        if critical > 0:
            self.occlusion_grade = OcclusionGrade.D
        elif issues >= 2:
            self.occlusion_grade = OcclusionGrade.C
        elif issues == 1:
            self.occlusion_grade = OcclusionGrade.B
        else:
            self.occlusion_grade = OcclusionGrade.A

        return self

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def is_ideal_occlusion(self) -> bool:
        """Return True if all primary metrics are within ideal ranges."""
        return (
            self.occlusion_grade == OcclusionGrade.A
            and self.molar_relationship in (AngleMolarClass.CLASS_I, "Class_I", None)
            and (self.overjet_mm or 0.0) >= 0.0
            and (self.overbite_mm or 0.0) >= 0.0
        )

    @property
    def contact_count(self) -> int:
        return len(self.contact_points)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OcclusionPlanContract":
        return cls.model_validate(data)
