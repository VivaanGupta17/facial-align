"""
Pydantic schemas for surgical case CRUD, listing, and status management.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import Field, field_validator

from app.models.case import CaseStatus, CaseType, VALID_TRANSITIONS
from app.schemas.common import BaseSchema, PaginatedResponse


class CaseCreate(BaseSchema):
    """Request schema to create a new surgical case."""
    patient_id: uuid.UUID = Field(..., description="Patient UUID")
    study_id: uuid.UUID = Field(..., description="ImagingStudy UUID to associate with the case")
    case_type: CaseType = Field(..., description="Clinical case type")
    surgeon_id: Optional[str] = Field(None, description="Assigning surgeon user ID")
    fracture_classification: Optional[str] = Field(
        None,
        max_length=128,
        description="Fracture classification (e.g., Le Fort I, NOE type III)"
    )
    planned_procedure: Optional[str] = Field(
        None,
        max_length=256,
        description="Planned surgical procedure description"
    )
    diagnosis_codes: Optional[List[str]] = Field(
        None,
        description="ICD-10 diagnosis codes"
    )
    target_surgery_date: Optional[datetime] = Field(
        None, description="Target date for surgical procedure"
    )
    team_ids: Optional[List[str]] = Field(
        None, description="Additional team member user IDs"
    )
    upload_notes: Optional[str] = Field(
        None, max_length=1000, description="Case creation notes"
    )


class CaseUpdate(BaseSchema):
    """Request schema for updating a case (partial updates allowed)."""
    surgeon_id: Optional[str] = None
    reviewer_id: Optional[str] = None
    fracture_classification: Optional[str] = Field(None, max_length=128)
    planned_procedure: Optional[str] = Field(None, max_length=256)
    diagnosis_codes: Optional[List[str]] = None
    target_surgery_date: Optional[datetime] = None
    team_ids: Optional[List[str]] = None


class CaseStatusTransition(BaseSchema):
    """Request schema for case status transitions."""
    new_status: CaseStatus = Field(..., description="Target status")
    notes: Optional[str] = Field(
        None, max_length=500,
        description="Notes explaining the status transition (required for approval/rejection)"
    )

    @field_validator("new_status")
    @classmethod
    def validate_status_is_known(cls, v: CaseStatus) -> CaseStatus:
        # Check it's a valid enum value
        return v


class SegmentationSummary(BaseSchema):
    """Compact segmentation result summary for case response."""
    id: uuid.UUID
    status: str
    model_name: str
    model_version: str
    overall_confidence: Optional[float]
    structure_count: Optional[int]
    created_at: datetime


class PlanSummary(BaseSchema):
    """Compact reduction plan summary for case response."""
    id: uuid.UUID
    plan_version: int
    status: str
    confidence_score: Optional[float]
    surgeon_approved: bool
    created_at: datetime


class CaseResponse(BaseSchema):
    """Full surgical case response."""
    id: uuid.UUID
    case_number: str
    patient_id: uuid.UUID
    study_id: uuid.UUID
    case_type: CaseType
    status: CaseStatus
    surgeon_id: Optional[str]
    reviewer_id: Optional[str]
    fracture_classification: Optional[str]
    planned_procedure: Optional[str]
    diagnosis_codes: Optional[List[str]]
    target_surgery_date: Optional[datetime]
    team_ids: Optional[List[str]]
    current_task_id: Optional[str]
    last_error: Optional[str]
    created_at: datetime
    updated_at: datetime
    approved_at: Optional[datetime]
    created_by: Optional[str]

    # Embedded summaries
    latest_segmentation: Optional[SegmentationSummary] = None
    latest_plan: Optional[PlanSummary] = None
    segmentation_count: int = 0
    plan_count: int = 0

    # Multi-study
    studies: List["CaseStudyInfo"] = Field(default_factory=list)

    # Available transitions from current status
    allowed_transitions: List[str] = Field(
        default_factory=list,
        description="Valid next status values from current state"
    )

    def model_post_init(self, __context) -> None:
        if not self.allowed_transitions:
            self.allowed_transitions = [
                s.value for s in VALID_TRANSITIONS.get(self.status, set())
            ]


class CaseListItem(BaseSchema):
    """Compact case representation for list endpoint."""
    id: uuid.UUID
    case_number: str
    patient_id: uuid.UUID
    case_type: CaseType
    status: CaseStatus
    surgeon_id: Optional[str]
    fracture_classification: Optional[str]
    latest_segmentation_status: Optional[str]
    latest_plan_confidence: Optional[float]
    created_at: datetime
    updated_at: datetime


class CaseListResponse(PaginatedResponse[CaseListItem]):
    """Paginated case list response."""
    pass


class CaseListFilters(BaseSchema):
    """Query filters for case list endpoint."""
    case_type: Optional[CaseType] = None
    status: Optional[CaseStatus] = None
    surgeon_id: Optional[str] = None
    patient_id: Optional[uuid.UUID] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    has_approved_plan: Optional[bool] = None


# ---------------------------------------------------------------------------
# Case-Study junction schemas
# ---------------------------------------------------------------------------


class CaseStudyCreate(BaseSchema):
    """Attach a study to a case."""
    study_id: uuid.UUID = Field(..., description="ImagingStudy UUID to attach")
    study_role: str = Field(default="pre_op", description="Role: pre_op, post_op, follow_up, intra_op")
    study_label: Optional[str] = Field(None, max_length=128, description="Human label")
    is_primary: bool = Field(default=False, description="Set as primary study for this case")


class CaseStudyUpdate(BaseSchema):
    """Partial update for a case-study link."""
    study_role: Optional[str] = None
    study_label: Optional[str] = Field(None, max_length=128)
    is_primary: Optional[bool] = None


class CaseStudyInfo(BaseSchema):
    """Case-study link with study metadata for responses."""
    id: uuid.UUID
    study_id: uuid.UUID
    study_role: str
    study_label: Optional[str]
    is_primary: bool
    display_order: int
    created_at: datetime
    # Denormalised study metadata
    study_uid: Optional[str] = None
    modality: Optional[str] = None
    acquisition_date: Optional[datetime] = None
    ingestion_status: Optional[str] = None
