"""Data contract for occlusion analysis plan."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ToothContactContract(BaseModel):
    upper_fdi: int
    lower_fdi: int
    contact_area_mm2: Optional[float] = None
    relative_force: Optional[float] = None


class OcclusionPlanContract(BaseModel):
    """Canonical occlusion plan contract."""
    plan_id: str
    case_id: str
    overjet_mm: Optional[float] = None
    overbite_mm: Optional[float] = None
    molar_relationship: Optional[str] = None
    midline_deviation_mm: Optional[float] = None
    cant_degrees: Optional[float] = None
    curve_of_spee_mm: Optional[float] = None
    contact_points: List[ToothContactContract] = Field(default_factory=list)
    constraints_satisfied: bool = False
    constraint_violations: List[str] = Field(default_factory=list)
    splint_required: bool = False
    splint_vd_mm: Optional[float] = None
    notes: Optional[str] = None
