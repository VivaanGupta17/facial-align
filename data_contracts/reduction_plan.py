"""Data contract for complete surgical reduction plan."""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from data_contracts.fracture_fragment import FractureFragmentContract


class OcclusalMetricsContract(BaseModel):
    overjet_mm: Optional[float] = None
    overbite_mm: Optional[float] = None
    molar_relationship: Optional[str] = None
    midline_deviation_mm: Optional[float] = None
    cant_degrees: Optional[float] = None
    constraints_satisfied: bool = False
    constraint_violations: List[str] = Field(default_factory=list)


class ValidationContract(BaseModel):
    passed: bool
    symmetry_ok: bool
    occlusion_ok: bool
    condylar_seating_ok: bool
    hardware_placement_ok: bool
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    skeletal_symmetry_score: Optional[float] = None


class ReductionPlanContract(BaseModel):
    """Canonical reduction plan used across services and APIs."""
    plan_id: str
    case_id: str
    plan_version: int
    model_name: str
    model_version: str
    fragments: List[FractureFragmentContract]
    occlusal_metrics: Optional[OcclusalMetricsContract] = None
    validation: Optional[ValidationContract] = None
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    symmetry_score: Optional[float] = None
    surgeon_approved: bool = False
    is_ml_generated: bool = True
    generation_time_ms: int = 0
    created_at: datetime
    approved_at: Optional[datetime] = None
