"""Data contract for case review and clinical sign-off."""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ClinicalMeasurement(BaseModel):
    name: str
    value: float
    unit: str
    within_normal_range: bool
    normal_range: Optional[str] = None


class CaseReview(BaseModel):
    """Clinical case review data contract — surgeon sign-off record."""
    review_id: str
    case_id: str
    plan_id: str
    reviewer_id: str
    reviewer_role: str = Field(..., description="surgeon, fellow, attending")
    review_timestamp: datetime
    approved: bool
    approval_notes: Optional[str] = None
    rejection_reason: Optional[str] = None
    clinical_measurements: List[ClinicalMeasurement] = Field(default_factory=list)
    checklist_items: Dict[str, bool] = Field(
        default_factory=lambda: {
            "occlusion_acceptable": False,
            "symmetry_acceptable": False,
            "condylar_seating_confirmed": False,
            "hardware_plan_reviewed": False,
            "informed_consent_obtained": False,
            "imaging_reviewed": False,
            "plan_matches_clinical_exam": False,
        }
    )
    required_modifications: List[str] = Field(default_factory=list)
    second_reviewer_required: bool = False
    second_reviewer_id: Optional[str] = None
    second_review_timestamp: Optional[datetime] = None
    second_review_approved: Optional[bool] = None
