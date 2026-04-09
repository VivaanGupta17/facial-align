"""
Pydantic schemas for fracture reduction planning, occlusal constraints, and metrics.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator, model_validator

from app.schemas.common import BaseSchema, BoundingBox3D, Transform3D


# ─── Occlusion schemas ────────────────────────────────────────────────────────


class OcclusalConstraints(BaseSchema):
    """
    Clinical occlusal targets used to guide the reduction optimization.
    These represent the desired post-reduction dental occlusion.
    """
    # Primary metrics
    target_overjet_mm: float = Field(
        default=2.0,
        description="Target horizontal incisal overjet in mm (normal: 1-3mm)"
    )
    target_overbite_mm: float = Field(
        default=3.0,
        description="Target vertical incisal overbite in mm (normal: 2-4mm)"
    )
    molar_class_target: str = Field(
        default="Class_I",
        description="Target Angle molar classification: Class_I, Class_II_div1, Class_II_div2, Class_III"
    )
    midline_tolerance_mm: float = Field(
        default=1.0,
        ge=0,
        description="Acceptable midline deviation in mm"
    )
    cant_tolerance_degrees: float = Field(
        default=2.0,
        ge=0,
        description="Acceptable occlusal cant in degrees"
    )

    # Arch constraints
    use_pre_injury_occlusion: bool = Field(
        True,
        description="Optimize to match pre-injury occlusal record if available"
    )
    dental_splint_required: bool = Field(
        False,
        description="Whether an intermediate dental splint is planned"
    )
    intermaxillary_fixation: bool = Field(
        False,
        description="Whether IMF wires are planned during fixation"
    )

    # Condylar constraints
    condylar_seating: str = Field(
        default="centric_relation",
        description="Target condylar position: centric_relation, centric_occlusion, therapeutic"
    )
    bilateral_condylar_seating: bool = Field(
        True,
        description="Require bilateral condylar seating"
    )

    @field_validator("molar_class_target")
    @classmethod
    def validate_molar_class(cls, v: str) -> str:
        valid = {"Class_I", "Class_II_div1", "Class_II_div2", "Class_III"}
        if v not in valid:
            raise ValueError(f"molar_class_target must be one of: {valid}")
        return v


class OcclusalMetrics(BaseSchema):
    """
    Measured occlusal metrics from the planned or actual occlusion.
    Used to assess reduction quality.
    """
    overjet_mm: Optional[float] = Field(None, description="Measured incisal overjet in mm")
    overbite_mm: Optional[float] = Field(None, description="Measured incisal overbite in mm")
    molar_relationship: Optional[str] = Field(
        None, description="Measured Angle molar relationship"
    )
    midline_deviation_mm: Optional[float] = Field(
        None, description="Upper-to-lower dental midline deviation in mm"
    )
    cant_degrees: Optional[float] = Field(
        None, description="Occlusal plane cant in degrees"
    )
    curve_of_spee_mm: Optional[float] = Field(
        None, description="Depth of curve of Spee in mm"
    )
    posterior_open_bite_mm: Optional[float] = Field(
        None, description="Posterior open bite in mm (0 if in contact)"
    )
    anterior_open_bite_mm: Optional[float] = Field(
        None, description="Anterior open bite in mm (0 if in contact)"
    )
    contact_points: Optional[int] = Field(
        None, description="Number of estimated occlusal contact points"
    )
    constraints_satisfied: bool = Field(
        False, description="Whether all occlusal constraints are satisfied"
    )
    constraint_violations: List[str] = Field(
        default_factory=list,
        description="List of violated constraint descriptions"
    )


# ─── Fragment schemas ─────────────────────────────────────────────────────────


class FragmentTransform(BaseSchema):
    """Planned rigid body transform for a single fracture fragment."""
    fragment_id: str = Field(..., description="Fragment identifier (e.g., 'frag_01')")
    fragment_label: int = Field(..., description="Voxel label value for this fragment")
    transform: Transform3D = Field(..., description="Planned rigid body transform")
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="ML model confidence in this transform prediction"
    )
    alternative_transforms: Optional[List[Transform3D]] = Field(
        None,
        description="Alternative transform proposals (ranked by confidence)"
    )
    alternative_confidences: Optional[List[float]] = None
    is_reference_fragment: bool = Field(
        False,
        description="True if this fragment is the anatomical reference (identity transform)"
    )
    expected_hardware: Optional[str] = Field(
        None, description="Expected fixation hardware: plate_and_screw, wire, splint, etc."
    )
    surgical_sequence: Optional[int] = Field(
        None, description="Recommended surgical sequence order for this fragment"
    )


class FragmentInfo(BaseSchema):
    """Geometric information about a fracture fragment."""
    fragment_id: str
    fragment_label: int
    mesh_path: Optional[str] = None
    volume_cc: Optional[float] = None
    surface_area_mm2: Optional[float] = None
    centroid_mm: Optional[list[float]] = Field(
        None, description="Fragment centroid [x, y, z] in patient coordinates"
    )
    bounding_box: Optional[BoundingBox3D] = None
    parent_structure: Optional[str] = Field(
        None, description="Parent anatomical structure (e.g., 'mandible')"
    )


# ─── Plan schemas ─────────────────────────────────────────────────────────────


class ReductionPlanRequest(BaseSchema):
    """Request to generate a new fracture reduction plan."""
    case_id: uuid.UUID = Field(..., description="Surgical case ID")
    segmentation_id: uuid.UUID = Field(
        ..., description="Segmentation result to base plan on"
    )
    occlusal_constraints: Optional[OcclusalConstraints] = Field(
        None,
        description="Clinical occlusal targets (uses defaults if not provided)"
    )
    model_name: str = Field(
        default="baseline_icp",
        description="Reduction model: 'baseline_icp', 'learned_v1'"
    )
    use_intact_reference: bool = Field(
        True,
        description="Use contralateral anatomy as reference for symmetric reduction"
    )
    include_alternative_plans: bool = Field(
        False,
        description="Include alternative transform proposals in result"
    )
    max_fragments: Optional[int] = Field(
        None, description="Maximum number of fragments to plan (largest by volume)"
    )


class ValidationResult(BaseSchema):
    """Result of automated reduction plan validation."""
    passed: bool
    symmetry_ok: bool
    occlusion_ok: bool
    condylar_seating_ok: bool
    hardware_placement_ok: bool
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    skeletal_symmetry_score: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Facial skeletal symmetry score after planned reduction"
    )


class SurgeonEditRequest(BaseSchema):
    """Request to apply a surgeon's manual adjustment to a plan."""
    fragment_id: str = Field(..., description="Fragment to modify")
    new_transform: Transform3D = Field(..., description="Surgeon-adjusted transform")
    notes: Optional[str] = Field(None, max_length=500, description="Surgeon's notes")
    re_optimize: bool = Field(
        True,
        description="Re-run constraint optimization after applying this edit"
    )


class MetricOverrideRequest(BaseSchema):
    """Request to override an occlusal metric target and re-optimize the plan."""
    metric_name: str = Field(
        ...,
        description="Metric to override: overjet_mm, overbite_pct, midline_deviation_mm, occlusal_cant_deg"
    )
    target_value: float = Field(..., description="New target value for the metric")
    notes: Optional[str] = Field(None, max_length=500, description="Surgeon's notes on override")

    @field_validator("metric_name")
    @classmethod
    def validate_metric_name(cls, v: str) -> str:
        valid = {"overjet_mm", "overbite_pct", "midline_deviation_mm", "occlusal_cant_deg"}
        if v not in valid:
            raise ValueError(f"metric_name must be one of: {valid}")
        return v


class ReductionPlanResponse(BaseSchema):
    """Complete reduction plan response."""
    id: uuid.UUID
    case_id: uuid.UUID
    plan_version: int
    status: str
    model_name: Optional[str]
    model_version: Optional[str]

    fragments: Optional[Dict[str, FragmentInfo]] = None
    fragment_transforms: Optional[List[FragmentTransform]] = None
    occlusal_constraints: Optional[OcclusalConstraints] = None
    occlusal_metrics: Optional[OcclusalMetrics] = None
    validation: Optional[ValidationResult] = None

    confidence_score: Optional[float] = None
    surgeon_approved: bool = False
    surgeon_notes: Optional[str] = None
    parent_plan_id: Optional[uuid.UUID] = None
    is_ml_generated: bool = True

    generation_time_ms: Optional[int] = None
    created_at: datetime
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
